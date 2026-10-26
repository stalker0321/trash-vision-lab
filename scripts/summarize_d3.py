"""Summarize the complete, fixed D2/D3 evaluation queue without mixing label versions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def collect(root, queue, manifest):
    index = {row['id']: row for row in manifest['images']}
    rows = []
    for job in queue['jobs']:
        hashes = set()
        for split in ['val', 'test', 'hard_test']:
            path = root / (job['name'] + '_' + split) / 'summary.json'
            summary = json.loads(path.read_text())
            if summary['dataset_fingerprint'] != queue['dataset_fingerprint'] or summary['split'] != split:
                raise ValueError(f'Evaluation provenance mismatch: {path}')
            expected_protocol = dict(imgsz=job['imgsz'], batch=8, workers=0, conf=.001,
                                     iou=.7, max_det=300, half=False, augment=False)
            if summary['protocol'] != expected_protocol or summary['class_names'] != manifest['names']:
                raise ValueError(f'Evaluation protocol differs: {path}')
            ids = manifest['splits'][split]
            if summary['images'] != len(ids) or summary['boxes'] != sum(index[i]['bbox_count'] for i in ids):
                raise ValueError(f'Evaluation population differs: {path}')
            sha = summary['weights_sha256']
            if job.get('sha256') and sha != job['sha256']:
                raise ValueError(f'Historical checkpoint differs: {path}')
            hashes.add(sha)
            metrics = summary['metrics']
            ap50, ap = metrics['metrics/mAP50(B)'], metrics['metrics/mAP50-95(B)']
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in [ap50, ap]):
                raise ValueError(f'Invalid AP values: {path}')
            rows.append(dict(name=job['name'], split=split, imgsz=job['imgsz'], images=summary['images'],
                             boxes=summary['boxes'], ap50=ap50, ap50_95=ap, weights_sha256=sha,
                             summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             per_class=summary['per_class']))
        if len(hashes) != 1:
            raise ValueError(f'Checkpoint changed between splits: {job["name"]}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--queue', type=Path, default=Path('configs/d3_evaluation.json'))
    parser.add_argument('--manifest', type=Path, default=Path('manifests/d3.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/d3'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest['fingerprint'] != queue['dataset_fingerprint']:
        raise ValueError('Queue and manifest differ')
    rows = collect(args.root, queue, manifest)
    lookup = {(r['name'], r['split']): r for r in rows}
    deltas = {model: {split: 100 * (lookup[model + '_d3', split]['ap50_95'] -
                                    lookup[model + '_d2', split]['ap50_95'])
                      for split in ['val', 'test', 'hard_test']} for model in ['nano', 'medium']}
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'dataset_fingerprint': manifest['fingerprint'], 'seed': 42, 'rows': rows,
              'd3_minus_d2_ap50_95_percentage_points': deltas,
              'caveats': ['One seed per model; no estimate of run-to-run uncertainty.',
                          'Historical D2 best checkpoints were selected using older validation labels.',
                          'Hard test is a subset of test, not an independent evaluation.',
                          'The real-image pool was involved in historical mining/review; this is a retrospective holdout.',
                          'Nano uses 320 px and Medium 640 px; this does not isolate architecture alone.']}
    training_record = args.output / 'training/runs.json'
    if training_record.exists() and any(r.get('resumed') for r in json.loads(training_record.read_text())):
        result['caveats'].append('Medium resumed from an optimizer checkpoint after a network interruption; resumed input/augmentation order need not match an uninterrupted run.')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = ['name', 'split', 'imgsz', 'images', 'boxes', 'ap50', 'ap50_95', 'weights_sha256', 'summary_sha256']
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    with (args.output / 'per-class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'split', 'class', 'ap50', 'ap50_95'])
        writer.writeheader()
        for row in rows:
            for name, metrics in row['per_class'].items():
                writer.writerow(dict(name=row['name'], split=row['split'], **{'class': name},
                    ap50=metrics['ap50'] if metrics else '', ap50_95=metrics['ap50_95'] if metrics else ''))
    lines = ['# D2 and D3 on the same reviewed holdout', '',
             'Fixed seed42 checkpoints, evaluated against the frozen D3 labels.', '',
             'All AP values below are percentages.', '']
    for split, title in [('val', 'Validation'), ('test', 'Test'), ('hard_test', 'Hard subset of test')]:
        lines += ['## ' + title, '', '| Configuration | AP50 | AP50–95 |', '| --- | ---: | ---: |']
        for row in rows:
            if row['split'] == split:
                lines.append(f'| {row["name"]}, {row["imgsz"]} px | {100*row["ap50"]:.2f} | {100*row["ap50_95"]:.2f} |')
        lines.append('')
    lines += ['## Test AP50–95 by material', '',
              '| Configuration | ' + ' | '.join(manifest['names']) + ' |',
              '| --- | ' + ' | '.join(['---:'] * len(manifest['names'])) + ' |']
    for row in rows:
        if row['split'] == 'test':
            values = [f'{100*row["per_class"][name]["ap50_95"]:.2f}' if row['per_class'][name] else '—'
                      for name in manifest['names']]
            lines.append('| ' + row['name'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'All per-class AP values are retained in [per-class.csv](https://github.com/stalker0321/trash-vision-lab/blob/main/reports/d3/per-class.csv).', '']
    lines += ['## Interpretation limits', ''] + ['- ' + value for value in result['caveats']]
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
        upper = max(.3, min(1., max(r['ap50_95'] for r in rows) + .12))
        for ax, split, title in zip(axes, ['val', 'test', 'hard_test'],
                                   ['Validation · 553 images', 'Test · 552 images', 'Hard subset · 84 images']):
            for version, offset, color in [('d2', -.18, '#8996A3'), ('d3', .18, '#2474B5')]:
                values = [lookup[model + '_' + version, split]['ap50_95'] for model in ['nano', 'medium']]
                bars = ax.bar([offset, 1 + offset], values, width=.34, color=color, label=version.upper())
                ax.bar_label(bars, labels=[f'{value:.1%}' for value in values], padding=3, fontsize=9)
            ax.set_xticks([0, 1], ['Nano 320', 'Medium 640'])
            ax.set_ylim(0, upper)
            ax.set_title(title, fontsize=11)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('mAP50–95')
        axes[0].legend(frameon=False)
        fig.suptitle('D2 and D3 evaluated on the same reviewed labels', fontsize=14)
        fig.text(.08, .045, 'One seed; hard test is part of test. Historical D2 checkpoints were selected on earlier validation labels.', fontsize=9)
        fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.83, wspace=.12)
        fig.savefig(args.output / 'comparison.svg', metadata={'Date': None})
        fig.savefig(args.output / 'comparison.png', dpi=160)
        plt.close(fig)
    print(json.dumps({'evaluations': len(rows), 'deltas_percentage_points': deltas}))


if __name__ == '__main__':
    main()
