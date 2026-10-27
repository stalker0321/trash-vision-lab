import unittest

from scripts.analyze_object_sizes import bucket, dimensions, match, xyxy


class ObjectSizeTests(unittest.TestCase):
    def test_projection_preserves_rectangular_image_aspect_ratio(self):
        # A 100x100px object in a 2000x1000px photo becomes 16x16, not 16x32.
        box = (0, .5, .5, .05, .1)
        self.assertEqual(dimensions(box, 2000, 1000), (16, 16))
        self.assertEqual(xyxy(box, 2000, 1000), (950, 450, 1050, 550))
        self.assertEqual([bucket(v) for v in [3.99, 4, 7.99, 8, 16, 32]], [0, 1, 1, 2, 3, 4])

    def test_duplicates_and_wrong_material_do_not_inflate_recall(self):
        truth = [{'xyxy': (0, 0, 10, 10), 'class': 0},
                 {'xyxy': (20, 0, 30, 10), 'class': 1}]
        predictions = [dict(truth[0], score=.9), dict(truth[0], score=.8),
                       {'xyxy': (20, 0, 30, 10), 'class': 3, 'score': .7}]
        self.assertEqual(match(truth, predictions, class_aware=True), {0})
        self.assertEqual(match(truth, predictions, class_aware=False), {0, 1})
        self.assertEqual(match(truth, [], class_aware=True), set())
        self.assertEqual(match([], predictions, class_aware=True), set())

    def test_loose_localization_is_not_a_match(self):
        truth = [{'xyxy': (0, 0, 10, 10), 'class': 0}]
        predictions = [{'xyxy': (5, 5, 15, 15), 'class': 0, 'score': .9}]
        self.assertEqual(match(truth, predictions, class_aware=False), set())


if __name__ == '__main__':
    unittest.main()
