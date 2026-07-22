import os
import json
import zipfile
import yaml
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Tuple


class PlantVillageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], transform: transforms.Compose = None):
        self.samples = samples
        self.transform = transform
        self.classes = sorted({class_name for _, class_name in self.samples})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        # Extract crop and disease type from class name (format: "Crop___Disease")
        if '___' in class_name:
            crop, disease = class_name.split('___', 1)
        else:
            crop, disease = class_name, 'healthy'

        return image, label, crop, disease, image_path


class DatasetSubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, crop, disease, image_path = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, crop, disease, image_path


class PlantVillageSplitDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str]], class_to_idx: Dict[str, int], transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name, _ = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.class_to_idx[class_name]
        crop, disease = class_name.split('___', 1)

        if self.transform:
            image = self.transform(image)

        return image, label, crop, disease, image_path


def get_data_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # 224 +32 = 256x256 as per architecture
        transforms.RandomCrop(image_size),  # Random crop to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # No hue, as per architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size + 32),   # 256 — preserves aspect ratio
        transforms.CenterCrop(image_size),    # 224 — matches ImageNet eval protocol
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform
    }


def get_val_test_transform(image_size: int = 224):
    """Compatibility helper used by PlantDoc evaluation scripts."""
    return get_data_transforms(image_size=image_size)['test']


def download_plant_village_dataset(raw_dir: str) -> str:
    """Ensure the PlantVillage color image folders exist locally and return their root directory."""
    os.makedirs(raw_dir, exist_ok=True)
    dataset_root = os.path.join(raw_dir, "plantvillage_hf")
    image_root = os.path.join(dataset_root, "raw", "color")
    archive_root = os.path.join(dataset_root, "archives")
    archive_path = os.path.join(archive_root, "data.zip")

    if os.path.isdir(image_root) and any(os.scandir(image_root)):
        print("Using cached PlantVillage images from local files...")
        return image_root

    os.makedirs(archive_root, exist_ok=True)
    print("PlantVillage images not found locally. Downloading the dataset archive from Hugging Face...")
    downloaded_archive = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename="data.zip",
        repo_type="dataset",
        local_dir=archive_root
    )

    if downloaded_archive != archive_path and not os.path.exists(archive_path):
        archive_path = downloaded_archive

    print("Extracting PlantVillage image archive...")
    with zipfile.ZipFile(archive_path, 'r') as zip_file:
        zip_file.extractall(dataset_root)

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Expected extracted image directory was not found: {image_root}")

    return image_root


def build_image_samples(image_root: str) -> List[Tuple[str, str]]:
    """Collect (image_path, class_name) pairs from the extracted PlantVillage color folders."""
    samples = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for class_entry in sorted(os.scandir(image_root), key=lambda entry: entry.name):
        if not class_entry.is_dir():
            continue

        class_name = class_entry.name
        for root, _, file_names in os.walk(class_entry.path):
            for file_name in sorted(file_names):
                if file_name.lower().endswith(valid_extensions):
                    samples.append((os.path.join(root, file_name), class_name))

    if not samples:
        raise RuntimeError(f"No image files were found under {image_root}")

    return samples


def get_cached_color_image_root(raw_dir: str) -> str:
    """Return the extracted local color-image root required for training."""
    image_root = os.path.join(raw_dir, "plantvillage_hf", "raw", "color")
    if not os.path.isdir(image_root) or not any(os.scandir(image_root)):
        raise FileNotFoundError(
            "PlantVillage color images are not available locally. "
            "Run 'py -3.11 download_assets.py' once before training."
        )
    return image_root


def get_cached_hf_metadata_paths(dataset_name: str) -> Dict[str, str]:
    """Locate the cached official HF split files and leaf map without downloading anything."""
    repo_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{dataset_name.replace('/', '--')}" / "snapshots"
    if not repo_cache.is_dir():
        raise FileNotFoundError(
            "Cached Hugging Face PlantVillage metadata was not found. "
            "Run 'py -3.11 download_assets.py' once before training."
        )

    snapshot_dirs = [path for path in repo_cache.iterdir() if path.is_dir()]
    snapshot_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for snapshot_dir in snapshot_dirs:
        train_split = snapshot_dir / "splits" / "color_train.txt"
        test_split = snapshot_dir / "splits" / "color_test.txt"
        leaf_map = snapshot_dir / "leaf_grouping" / "leaf-map.json"
        if train_split.is_file() and test_split.is_file() and leaf_map.is_file():
            return {
                "train_split": str(train_split),
                "test_split": str(test_split),
                "leaf_map": str(leaf_map),
            }

    raise FileNotFoundError(
        "Cached Hugging Face PlantVillage split files were not found. "
        "Run 'py -3.11 download_assets.py' once before training."
    )


def normalize_leaf_map_key(file_name: str) -> str:
    stem = Path(file_name).stem
    if "___" in stem:
        return stem.split("___", 1)[1].lower()
    return stem.lower()


def load_leaf_map(leaf_map_path: str) -> Dict[str, List[str]]:
    with open(leaf_map_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_official_split_samples(split_file_path: str, dataset_root: str, leaf_map: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    samples = []
    with open(split_file_path, "r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            relative_path = raw_line.strip()
            if not relative_path:
                continue

            image_path = os.path.join(dataset_root, relative_path.replace("/", os.sep))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Expected split image was not found locally: {image_path}. "
                    "Run 'py -3.11 download_assets.py' once before training."
                )

            class_name = Path(relative_path).parent.name
            leaf_map_key = normalize_leaf_map_key(Path(relative_path).name)
            leaf_entries = leaf_map.get(leaf_map_key, [])
            leaf_id = leaf_entries[0] if leaf_entries else f"fallback::{class_name}::{Path(relative_path).stem.lower()}"
            samples.append((image_path, class_name, leaf_id))

    if not samples:
        raise RuntimeError(f"No samples were found in official split file: {split_file_path}")

    return samples


def split_train_samples_by_leaf_id(train_samples: List[Tuple[str, str, str]], val_fraction: float, seed: int):
    """Split official color train samples into train/val by a stable (leaf_id, class_name) grouping."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"Validation fraction must be between 0 and 1, got {val_fraction}")

    group_to_label = {}
    group_to_samples = {}
    for sample in train_samples:
        _, class_name, leaf_id = sample
        group_key = (leaf_id, class_name)
        group_to_label[group_key] = class_name
        group_to_samples.setdefault(group_key, []).append(sample)

    unique_group_keys = list(group_to_samples.keys())
    unique_group_labels = [group_to_label[group_key] for group_key in unique_group_keys]

    if len(unique_group_keys) == 1:
        group_key = unique_group_keys[0]
        group_samples = group_to_samples[group_key]
        num_val_samples = max(1, int(round(len(group_samples) * val_fraction)))
        if num_val_samples >= len(group_samples):
            num_val_samples = max(1, len(group_samples) - 1)
        return group_samples[num_val_samples:], group_samples[:num_val_samples]

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_group_idx, val_group_idx = next(splitter.split(unique_group_keys, unique_group_labels))
    except ValueError:
        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randperm(len(unique_group_keys), generator=generator).tolist()
        num_val_groups = max(1, int(round(len(unique_group_keys) * val_fraction)))
        val_group_idx = shuffled_indices[:num_val_groups]
        train_group_idx = shuffled_indices[num_val_groups:]

    if len(train_group_idx) == 0 or len(val_group_idx) == 0:
        shuffled_indices = torch.randperm(len(unique_group_keys), generator=torch.Generator().manual_seed(seed)).tolist()
        num_val_groups = max(1, int(round(len(unique_group_keys) * val_fraction)))
        val_group_idx = shuffled_indices[:num_val_groups]
        train_group_idx = shuffled_indices[num_val_groups:]

    train_group_keys = {unique_group_keys[idx] for idx in train_group_idx}
    val_group_keys = {unique_group_keys[idx] for idx in val_group_idx}

    train_split = []
    val_split = []
    for group_key, samples in group_to_samples.items():
        if group_key in train_group_keys:
            train_split.extend(samples)
        elif group_key in val_group_keys:
            val_split.extend(samples)
        else:
            raise RuntimeError(f"Group '{group_key}' was not assigned to either train or val.")

    return train_split, val_split


def ensure_no_leaf_id_leakage(*splits: List[Tuple[str, str, str]]) -> None:
    if len(splits) < 2:
        return

    train_split = splits[0]
    val_split = splits[1]
    train_groups = {(sample[2], sample[1]) for sample in train_split}
    val_groups = {(sample[2], sample[1]) for sample in val_split}

    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(
            f"Train/validation leakage detected: {len(overlap)} overlapping (leaf_id, class_name) groups."
        )


def get_processed_manifest_dir(raw_dir: str) -> str:
    data_dir = os.path.dirname(raw_dir)
    return os.path.join(data_dir, "processed")


def write_split_manifest(file_path: str, samples: List[Tuple[str, str, str]]) -> None:
    with open(file_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("image_path\tclass_name\tleaf_id\n")
        for image_path, class_name, leaf_id in samples:
            file_handle.write(f"{image_path}\t{class_name}\t{leaf_id}\n")


def save_split_manifests(processed_dir: str,
                         train_samples: List[Tuple[str, str, str]],
                         val_samples: List[Tuple[str, str, str]],
                         test_samples: List[Tuple[str, str, str]]) -> None:
    """
    Save transparent split manifests only to the processed folder.
    No processed image tensors or cached datasets are written to disk.
    """
    os.makedirs(processed_dir, exist_ok=True)
    write_split_manifest(os.path.join(processed_dir, "train.txt"), train_samples)
    write_split_manifest(os.path.join(processed_dir, "val.txt"), val_samples)
    write_split_manifest(os.path.join(processed_dir, "test.txt"), test_samples)


def export_class_labels(class_to_idx: Dict[str, int], output_path: str = "config/class_labels.json") -> str:
    """Persist the class index mapping for inference and deployment."""
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    payload = {
        "num_classes": len(class_to_idx),
        "idx_to_class": {str(idx): name for idx, name in sorted(idx_to_class.items())},
        "class_to_idx": class_to_idx,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")
    return output_path


def compute_class_weights(train_samples: List[Tuple[str, str, str]], class_to_idx: Dict[str, int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, class_name, _ in train_samples:
        counts[class_to_idx[class_name]] += 1
    counts = counts.clamp(min=1.0)
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights


def build_weighted_sampler(train_samples: List[Tuple[str, str, str]], class_to_idx: Dict[str, int]) -> WeightedRandomSampler:
    """Per-sample weights inversely proportional to class frequency."""
    class_counts = {}
    for _, class_name, _ in train_samples:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    sample_weights = []
    for _, class_name, _ in train_samples:
        sample_weights.append(1.0 / class_counts[class_name])

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_samples),
        replacement=True,
    )


def get_data_loaders(config_path: str = 'config/base_config.yaml') -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    raw_dir = data_config['raw_dir']
    image_size = data_config['image_size']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    dataset_name = data_config.get('dataset_name', 'mohanty/PlantVillage')
    dataset_config = data_config.get('dataset_config', 'color')
    val_split_from_train = data_config.get('val_split_from_train', 0.15)
    split_seed = data_config.get('split_seed', 42)
    use_class_weights = bool(data_config.get('use_class_weights', False))
    use_weighted_sampler = bool(data_config.get('use_weighted_sampler', False))
    if use_class_weights and use_weighted_sampler:
        raise ValueError("Enable only one of use_class_weights or use_weighted_sampler, not both.")

    if dataset_config != 'color':
        raise ValueError(
            f"This training pipeline is locked to the official color split. "
            f"Received dataset_config='{dataset_config}'."
        )

    dataset_root = os.path.join(raw_dir, "plantvillage_hf")
    get_cached_color_image_root(raw_dir)
    metadata_paths = get_cached_hf_metadata_paths(dataset_name)
    leaf_map = load_leaf_map(metadata_paths["leaf_map"])

    official_train_samples = build_official_split_samples(
        metadata_paths["train_split"],
        dataset_root,
        leaf_map
    )
    official_test_samples = build_official_split_samples(
        metadata_paths["test_split"],
        dataset_root,
        leaf_map
    )
    train_samples, val_samples = split_train_samples_by_leaf_id(
        official_train_samples,
        val_split_from_train,
        split_seed
    )
    ensure_no_leaf_id_leakage(train_samples, val_samples, official_test_samples)
    save_split_manifests(
        get_processed_manifest_dir(raw_dir),
        train_samples,
        val_samples,
        official_test_samples
    )
    
    # Get transforms
    transforms_dict = get_data_transforms(image_size)
    class_names = sorted({sample[1] for sample in (train_samples + val_samples + official_test_samples)})
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    train_dataset = PlantVillageSplitDataset(train_samples, class_to_idx, transform=transforms_dict['train'])
    val_dataset = PlantVillageSplitDataset(val_samples, class_to_idx, transform=transforms_dict['val'])
    test_dataset = PlantVillageSplitDataset(official_test_samples, class_to_idx, transform=transforms_dict['test'])
    
    # Create data loaders
    _pin = torch.cuda.is_available()
    train_sampler = None
    if use_weighted_sampler:
        train_sampler = build_weighted_sampler(train_samples, class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=_pin,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin)
    
    # Get class information
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)
    class_weights = compute_class_weights(train_samples, class_to_idx, num_classes) if use_class_weights else None
    export_class_labels(class_to_idx)

    return train_loader, val_loader, test_loader, {
        'num_classes': num_classes,
        'idx_to_class': idx_to_class,
        'class_to_idx': class_to_idx,
        'class_weights': class_weights,
        'use_class_weights': use_class_weights,
        'use_weighted_sampler': use_weighted_sampler,
    }


if __name__ == '__main__':
    # Test data loader
    train_loader, val_loader, test_loader, class_info = get_data_loaders()
    print(f"Number of classes: {class_info['num_classes']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a sample
    images, labels, crops, diseases, paths = next(iter(train_loader))
    print(f"Sample image shape: {images[0].shape}")
    print(f"Sample label: {labels[0]}, Crop: {crops[0]}, Disease: {diseases[0]}")
    print(f"Sample path: {paths[0]}")
