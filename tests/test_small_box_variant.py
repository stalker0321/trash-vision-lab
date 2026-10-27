import copy
import json
from pathlib import Path
import tempfile
import unittest

from datasets.build_d3 import digest, file_hash, write_json
from datasets.build_small_box_variant import filtered_label, freeze, materialize


class SmallBoxVariantTests(unittest.TestCase):
    def test_aspect_ratio_strict_boundary_and_empty_label(self):
        # In 2000x1000 source: 50x50 ->8x8 stays; 20x20 ->3.2x3.2 goes.
        raw = b'0 0.5 0.5 0.025 0.05\n1 0.2 0.2 0.01 0.02\n'
        result, counts, removed = filtered_label(raw, 2000, 1000, 8, 320)
        self.assertEqual(result, raw.splitlines(keepends=True)[0])
        self.assertEqual(counts, [1, 0, 0, 0])
        self.assertEqual(removed[0]['line_number'], 2)
        self.assertEqual(filtered_label(raw.splitlines(keepends=True)[1], 2000, 1000, 8, 320)[0], b'')

    def test_preserves_holdout_and_images_and_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'variant'
            rows = []
            for i in range(3):
                image, label = f'images/{i}.jpg', f'labels/{i}.txt'
                (source / image).parent.mkdir(parents=True, exist_ok=True)
                (source / label).parent.mkdir(parents=True, exist_ok=True)
                (source / image).write_bytes(f'independent image {i}'.encode())
                (source / label).write_bytes(b'0 0.5 0.5 0.01 0.01\r\n')
                rows.append({'id': str(i), 'group_id': str(i), 'source': 'my_photos', 'image': image, 'label': label,
                    'image_sha256': file_hash(source / image), 'label_sha256': file_hash(source / label),
                    'width': 1000, 'height': 1000, 'bbox_count': 1, 'class_counts': [1,0,0,0]})
            parent = {'schema_version': 1, 'version': 'd3', 'names': ['plastic', 'paper', 'metal', 'glass'],
                'images': rows, 'splits': {'d3_train': ['0'], 'val': ['1'], 'test': ['2'], 'hard_test': ['2'],
                                         'calibration': ['0'], 'smoke_train': ['0'], 'smoke_val': ['1']}}
            parent['fingerprint'] = digest(parent)
            write_json(source / 'dataset.json', parent)
            manifest, labels, report = freeze(source)
            self.assertEqual(report['removed_boxes'], 1)
            self.assertEqual(manifest['splits'], parent['splits'])
            self.assertEqual(manifest['images'][1:], parent['images'][1:])
            self.assertEqual(manifest['parent_fingerprint'], parent['fingerprint'])
            self.assertEqual(labels['0'], b'')
            materialize(source, output, manifest, labels)
            self.assertEqual((output / 'labels/1.txt').read_bytes(), b'0 0.5 0.5 0.01 0.01\r\n')
            self.assertNotEqual((source / 'images/0.jpg').stat().st_ino, (output / 'images/0.jpg').stat().st_ino)
            self.assertEqual(json.loads((source / 'dataset.json').read_text()), parent)
            with self.assertRaises(FileExistsError):
                materialize(source, output, manifest, labels)
            (source / 'labels/0.txt').write_text('')
            with self.assertRaisesRegex(ValueError, 'Source label hash mismatch'):
                freeze(source)


if __name__ == '__main__':
    unittest.main()
