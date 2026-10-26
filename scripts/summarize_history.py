"""Recompute historical validation tables from the preserved training CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/historical'))
    parser.add_argument('--plot', action='store_true', help='Also render an SVG with matplotlib')
    args = parser.parse_args()
    root = args.input.resolve()
    experiments = json.loads((root / 'experiments.json').read_text())
    runs = []
    for spec in experiments['runs']:
        path = (root / spec['curve']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != spec['curve_sha256']:
            raise ValueError(f'Historical curve differs: {spec["name"]}')
        with path.open() as stream:
            rows = [{k.strip(): float(v.strip()) for k, v in row.items()} for row in csv.DictReader(stream)]
        if len(rows) != 150 or [int(row['epoch']) for row in rows] != list(range(1, 151)):
            raise ValueError(f'Incomplete historical curve: {spec["name"]}')
        best = max(rows, key=lambda row: row['metrics/mAP50-95(B)'])
        runs.append(dict(spec, best_epoch=int(best['epoch']), ap50=best['metrics/mAP50(B)'],
                         ap50_95=best['metrics/mAP50-95(B)'], last_ap50_95=rows[-1]['metrics/mAP50-95(B)']))
    aggregates = []
    for model in ['yolo26n', 'yolo26m']:
        for dataset in ['d0', 'd1', 'd2']:
            selected = [r for r in runs if r['series'] == 'main' and r['model'] == model and r['dataset'] == dataset]
            if len(selected) != 2 or {r['seed'] for r in selected} != {42, 123}:
                raise ValueError('Expected two recorded seeds for every historical configuration')
            values = [r['ap50_95'] for r in selected]
            aggregates.append(dict(model=model, dataset=dataset, ap50_mean=statistics.mean(r['ap50'] for r in selected),
                                   ap50_95_mean=statistics.mean(values), ap50_95_min=min(values), ap50_95_max=max(values)))
    result = {'metric_policy': experiments['metric_policy'], 'aggregates': aggregates, 'runs': runs,
              'caveats': experiments['caveats']}
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'model', 'dataset', 'seed', 'series', 'imgsz',
                                                   'weight_decay', 'best_epoch', 'ap50', 'ap50_95', 'last_ap50_95'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in runs)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for model, label, color in [('yolo26n', 'Nano · 320 px', '#2474B5'), ('yolo26m', 'Medium · 640 px', '#CE6A24')]:
            rows = [r for r in aggregates if r['model'] == model]
            y = [r['ap50_95_mean'] for r in rows]
            ax.plot(range(3), y, '-o', color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(range(3), [r['ap50_95_min'] for r in rows], [r['ap50_95_max'] for r in rows],
                            color=color, alpha=.18)
            for x, value in enumerate(y):
                ax.annotate(f'{value:.1%}', (x, value), xytext=(0, 9), textcoords='offset points',
                            ha='center', fontsize=10, color=color)
        ax.set_xticks(range(3), ['D0 · official TACO', 'D1 · + community', 'D2 · curated'])
        ax.set_ylim(.10, .53)
        ax.set_xlim(-.18, 2.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Validation mAP50–95')
        ax.set_title('Historical validation on the original annotations', loc='left', fontsize=13, pad=14)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False)
        fig.text(.10, .055, 'Mean of seeds 42 and 123; shading shows their range, not a confidence interval.', fontsize=8)
        fig.text(.10, .025, 'Complete data recipes; Nano/Medium also differ in resolution and weight decay. Not final test results.', fontsize=8)
        fig.subplots_adjust(bottom=.20, top=.88, left=.10, right=.98)
        fig.savefig(root / 'validation.svg', metadata={'Date': None})
        fig.savefig(root / 'validation.png', dpi=160)
        plt.close(fig)
    print(json.dumps({'historical_runs': len(runs), 'main_configurations': len(aggregates), 'output': str(root)}))


if __name__ == '__main__':
    main()
