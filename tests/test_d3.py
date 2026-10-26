import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from datasets.build_d3 import digest, file_hash, freeze, materialize, normalize_labels, read_frozen_manifest, validate_splits, write_json
from runners.train_d3 import verify_dataset


class D3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline, self.review = self.root / 'baseline', self.root / 'review'
        records, source_rows = [], []
        for i in range(3):
            image = f'images/clean/my_photos/{i}.jpg'
            label = f'labels/clean/my_photos/{i}.txt'
            for base in [self.baseline / 'data/prepared', self.review]:
                (base / image).parent.mkdir(parents=True, exist_ok=True)
                (base / image).write_bytes(f'image fixture {i}'.encode())
                (base / label).parent.mkdir(parents=True, exist_ok=True)
                (base / label).write_text('0 0.5 0.5 0.2 0.2\n')
            records.append({'id': f'clean:{i}', 'group_id': f'g{i}', 'kind': 'clean', 'source': 'my_photos',
                'origin': 'real_life', 'image': image, 'label': label, 'class_counts': [1, 0, 0, 0], 'bbox_count': 1,
                'image_sha256': file_hash(self.review / image), 'source_sha256': file_hash(self.review / image),
                'label_sha256': file_hash(self.review / label)})
            source_rows.append({'image_id': str(i), 'file_path': image, 'label_path': label})
        old = {'names': ['plastic', 'paper', 'metal', 'glass'], 'images': records,
               'splits': {'d0_train': [], 'd1_train': [], 'd2_train': ['clean:0'], 'val': ['clean:1'],
                          'test': ['clean:2'], 'hard_test': ['clean:2'], 'calibration': ['clean:0']}}
        old['fingerprint'] = digest(old)
        write_json(self.baseline / 'manifests/dataset.json', old)
        (self.review / 'manifests').mkdir()
        with (self.review / 'manifests/manifest_final.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
            writer.writeheader(); writer.writerows(source_rows)

    def test_review_changes_labels_preserves_images_and_membership(self):
        (self.review / 'labels/clean/my_photos/0.txt').write_text('1 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.1 0.1\n')
        manifest, summary = freeze(self.baseline, self.review)
        self.assertEqual(manifest['splits']['d3_train'], ['clean:0'])
        self.assertEqual(summary['splits']['d3_train']['boxes'], 2)
        self.assertEqual(summary['splits']['d3_train']['changed_label_images'], 1)
        output = self.root / 'd3'
        materialize(manifest, self.baseline, self.review, output)
        verify_dataset(output, manifest)
        source = self.review / manifest['images'][0]['image']
        self.assertNotEqual(source.stat().st_ino, (output / manifest['images'][0]['image']).stat().st_ino)
        with self.assertRaises(FileExistsError):
            materialize(manifest, self.baseline, self.review, output)

    def test_rejects_changed_reviewed_image(self):
        (self.review / 'images/clean/my_photos/0.jpg').write_bytes(b'different scene')
        with self.assertRaisesRegex(ValueError, 'image content changed'):
            freeze(self.baseline, self.review)

    def test_rejects_changed_baseline_label(self):
        (self.baseline / 'data/prepared/labels/clean/my_photos/0.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'Baseline label changed'):
            freeze(self.baseline, self.review)

    def test_rejects_group_leakage_and_catalog_holdout(self):
        manifest, _ = freeze(self.baseline, self.review)
        manifest['images'][1]['group_id'] = manifest['images'][0]['group_id']
        with self.assertRaisesRegex(ValueError, 'Cross-split leakage'):
            validate_splits(manifest)
        manifest['images'][1]['group_id'] = 'g1'
        for source in ['catalog', 'synthetic', 'crops']:
            with self.subTest(source=source):
                manifest['images'][1]['source'] = source
                with self.assertRaisesRegex(ValueError, 'Non-real holdout'):
                    validate_splits(manifest)

    def test_clips_border_without_dropping_box_and_rejects_bad_class(self):
        path = self.root / 'border.txt'
        path.write_text('3 0.95 0.5 0.2 0.2\n')
        text, counts, clipped = normalize_labels(path)
        self.assertEqual(text, '3 0.925000000 0.500000000 0.150000000 0.200000000\n')
        self.assertEqual((counts, clipped), ([0, 0, 0, 1], 1))
        path.write_text('4 0.5 0.5 0.2 0.2\n')
        with self.assertRaises(ValueError):
            normalize_labels(path)

    def test_detects_materialized_label_mutation(self):
        manifest, _ = freeze(self.baseline, self.review)
        output = self.root / 'd3'
        materialize(manifest, self.baseline, self.review, output)
        (output / manifest['images'][0]['label']).write_text('')
        with self.assertRaisesRegex(ValueError, 'Dataset content changed'):
            verify_dataset(output, manifest)

    def test_frozen_manifest_rebuild_needs_only_reviewed_pool(self):
        manifest, _ = freeze(self.baseline, self.review)
        path = self.root / 'frozen.json'
        write_json(path, manifest)
        shutil.rmtree(self.baseline)
        restored = read_frozen_manifest(path)
        output = self.root / 'restored'
        materialize(restored, None, self.review, output)
        verify_dataset(output, manifest)
        self.assertEqual(json.loads((output / 'dataset.json').read_text()), manifest)
        manifest['images'][0]['bbox_count'] += 1
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            read_frozen_manifest(path)

    def test_frozen_rebuild_rejects_source_mutation_even_with_baseline_copy(self):
        manifest, _ = freeze(self.baseline, self.review)
        (self.review / manifest['images'][0]['reviewed_image']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image content changed'):
            materialize(manifest, self.baseline, self.review, self.root / 'changed')


if __name__ == '__main__':
    unittest.main()
