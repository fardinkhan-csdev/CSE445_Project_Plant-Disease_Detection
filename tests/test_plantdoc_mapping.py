import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from plantdoc_mapping import PlantDocCOCODataset, get_available_plantdoc_splits, load_class_labels
from data.data_loader import get_val_test_transform


def test_plantdoc_mapping_loads_known_splits():
    root_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'plantdoc_roboflow_cocojson')
    assert os.path.isdir(root_dir)

    splits = get_available_plantdoc_splits(root_dir)
    assert 'train' in splits
    assert 'valid' in splits
    assert 'test' in splits

    class_to_idx, _ = load_class_labels()
    dataset = PlantDocCOCODataset(root_dir, 'test', class_to_idx, transform=get_val_test_transform())
    assert len(dataset) > 0
    assert dataset[0][1] in class_to_idx.values()
