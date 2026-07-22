import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import split_train_samples_by_leaf_id


class SplitLogicTests(unittest.TestCase):
    def test_split_train_samples_by_leaf_id_handles_multi_label_leaf_ids(self):
        samples = [
            ("/tmp/a.jpg", "Soybean___healthy", "Soybean___healthy:::495.0"),
            ("/tmp/b.jpg", "Soybean___healthy", "Soybean___healthy:::495.0"),
            ("/tmp/c.jpg", "Soybean___healthy", "Soybean___healthy:::495.0"),
            ("/tmp/d.jpg", "Soybean___healthy", "Soybean___healthy:::495.0"),
        ]

        train_split, val_split = split_train_samples_by_leaf_id(samples, 0.25, 42)

        self.assertEqual(len(train_split) + len(val_split), len(samples))
        self.assertGreaterEqual(len(train_split), 1)
        self.assertGreaterEqual(len(val_split), 1)


if __name__ == "__main__":
    unittest.main()
