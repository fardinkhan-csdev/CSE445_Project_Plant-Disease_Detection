import csv
import os
import sys
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import (
    get_val_test_transform,
    get_cached_color_image_root,
    get_cached_hf_metadata_paths,
    load_leaf_map,
    build_official_split_samples,
    split_train_samples_by_leaf_id,
    PlantVillageSplitDataset,
)
from plantdoc_mapping import PlantDocCOCODataset, get_available_plantdoc_splits, load_class_labels
from training.lora_trainer import LoRATrainer
from training.qlora_trainer import QLoRATrainer
from training.qklora_trainer import QKLoRATrainer
from evaluation.metrics import calculate_metrics

TRAINER_MAP = {
    'lora': LoRATrainer,
    'qlora': QLoRATrainer,
    'qklora': QKLoRATrainer,
}

_V3_MODE = False

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "eval")

METRIC_FIELDS = [
    'accuracy', 'f1_macro', 'binary_accuracy', 'binary_f1', 'binary_roc_auc',
    'both_correct_pct', 'name_only_correct_pct', 'disease_only_correct_pct', 'none_correct_pct'
]

PROBS_CSV = os.path.join(EVAL_DIR, "plantdoc_segmented_probs.csv")

MULTIRES_DEFAULT_SCALES = [192, 224, 256, 320]


def compute_image_quality_score(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    if gray.size < 4:
        return 0.0
    h, w = gray.shape
    if h < 2 or w < 2:
        return 0.0

    dxx = gray[:, 2:] - 2.0 * gray[:, 1:-1] + gray[:, :-2]
    dyy = gray[2:, :] - 2.0 * gray[1:-1, :] + gray[:-2, :]
    lap_var = float(np.var(np.concatenate([dxx.ravel(), dyy.ravel()])))

    sharpness = min(lap_var / 500.0, 1.0)
    sharpness = max(sharpness - 0.3, 0.0)

    brightness = gray.mean() / 255.0
    bright_score = 1.0 - abs(brightness - 0.5) / 0.5
    bright_score = max(bright_score, 0.0) ** 2

    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()
    hist = np.maximum(hist, 1e-10)
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(64.0)
    entropy_score = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
    entropy_score = max(entropy_score - 0.2, 0.0)

    quality = sharpness * 0.5 + bright_score * 0.25 + entropy_score * 0.25
    return float(np.clip(quality, 0.0, 1.0))


class _QualityFilteredDataset(Dataset):
    def __init__(self, base_dataset: Dataset, keep_indices: List[int]):
        self.base_dataset = base_dataset
        self.keep_indices = keep_indices

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.keep_indices[idx]]


def compute_image_quality_score(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    if gray.size < 4:
        return 0.0
    h, w = gray.shape
    if h < 2 or w < 2:
        return 0.0

    dxx = gray[:, 2:] - 2.0 * gray[:, 1:-1] + gray[:, :-2]
    dyy = gray[2:, :] - 2.0 * gray[1:-1, :] + gray[:-2, :]
    lap_var = float(np.var(np.concatenate([dxx.ravel(), dyy.ravel()])))
    sharpness = min(lap_var / 100.0, 1.0)

    brightness = gray.mean() / 255.0
    bright_score = 1.0 - abs(brightness - 0.5) / 0.5
    bright_score = max(bright_score, 0.0)

    bins = max(8, min(64, gray.size // 256))
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()
    hist = np.maximum(hist, 1e-10)
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(float(bins))
    entropy_score = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0

    quality = sharpness * 0.4 + bright_score * 0.3 + entropy_score * 0.3
    return float(np.clip(quality, 0.0, 1.0))


def build_transform_for_scale(scale: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_metric_row(method: str, split: str, metrics: Dict, sample_count: int, checkpoint: str) -> Dict:
    return {
        'method': method,
        'split': split,
        'accuracy': round(metrics.get('accuracy', 0.0) * 100, 2),
        'f1_macro': round(metrics.get('f1_macro', 0.0) * 100, 2),
        'binary_accuracy': round(metrics.get('binary_accuracy', 0.0) * 100, 2),
        'binary_f1': round(metrics.get('binary_f1', 0.0) * 100, 2),
        'binary_roc_auc': round(metrics.get('binary_roc_auc', 0.0), 4),
        'both_correct_pct': round(metrics.get('both_correct_pct', 0.0), 2),
        'name_only_correct_pct': round(metrics.get('name_only_correct_pct', 0.0), 2),
        'disease_only_correct_pct': round(metrics.get('disease_only_correct_pct', 0.0), 2),
        'none_correct_pct': round(metrics.get('none_correct_pct', 0.0), 2),
        'sample_count': sample_count,
        'checkpoint': checkpoint,
    }


def load_best_checkpoints(keys: Optional[List[str]] = None) -> Dict[str, nn.Module]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_to_idx, idx_to_class = load_class_labels()
    models = {}

    if _V3_MODE:
        from training.lora_trainer_v3 import LoRATrainerV3
        from training.qlora_trainer_v3 import QLoRATrainerV3
        from training.qalora_trainer import QALoRATrainer
        trainer_map = {
            'lora': LoRATrainerV3,
            'qlora': QLoRATrainerV3,
            'qalora': QALoRATrainer,
        }
        checkpoint_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints_v3")
        key_prefix = {
            'lora': 'lora_v3',
            'qlora': 'qlora_v3',
            'qalora': 'qalora',
        }
    else:
        trainer_map = TRAINER_MAP
        checkpoint_dir = CHECKPOINT_DIR
        key_prefix = {k: k for k in TRAINER_MAP}

    keys = keys or list(trainer_map.keys())
    for exp_key in keys:
        trainer_cls = trainer_map[exp_key]
        prefix = key_prefix.get(exp_key, exp_key)
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_best.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {exp_key}: checkpoint not found at {checkpoint_path}")
            continue

        trainer = trainer_cls(None, None, len(class_to_idx))
        trainer.load_checkpoint('best')
        model = trainer.model.to(device)
        model.eval()
        models[exp_key] = model

    return models


def get_plantvillage_train_loader_for_knn(batch_size: int = 32, num_workers: int = 4):
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['data']
    raw_dir = data_config['raw_dir']
    dataset_name = data_config.get('dataset_name', 'mohanty/PlantVillage')
    val_split_from_train = data_config.get('val_split_from_train', 0.15)
    split_seed = data_config.get('split_seed', 42)

    dataset_root = os.path.join(raw_dir, "plantvillage_hf")
    get_cached_color_image_root(raw_dir)
    metadata_paths = get_cached_hf_metadata_paths(dataset_name)
    leaf_map = load_leaf_map(metadata_paths["leaf_map"])

    official_train_samples = build_official_split_samples(
        metadata_paths["train_split"], dataset_root, leaf_map
    )
    train_samples, _ = split_train_samples_by_leaf_id(
        official_train_samples, val_split_from_train, split_seed
    )

    class_names = sorted({sample[1] for sample in official_train_samples})
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    transform = get_val_test_transform()
    dataset = PlantVillageSplitDataset(train_samples, class_to_idx, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return loader, class_to_idx


def extract_backbone_features(model, loader, device, desc="Extracting features"):
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch[0]
            labels = batch[1]
            paths = batch[4]
            images = images.to(device)

            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            all_features.append(x.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    return {
        'features': np.concatenate(all_features, axis=0),
        'labels': np.array(all_labels),
        'paths': list(all_paths),
    }


def extract_backbone_features(model, loader, device, desc="Extracting features"):
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch[0]
            labels = batch[1]
            paths = batch[4]
            images = images.to(device)

            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            all_features.append(x.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    return {
        'features': np.concatenate(all_features, axis=0),
        'labels': np.array(all_labels),
        'paths': list(all_paths),
    }


class StyleNormalize:
    def __init__(self, ref_mean: np.ndarray, ref_std: np.ndarray):
        self.ref_mean = ref_mean
        self.ref_std = ref_std

    def __call__(self, img: 'Image.Image') -> 'Image.Image':
        arr = np.array(img).astype(np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        std = np.maximum(std, 1e-6)
        matched = ((arr - mean) / std) * self.ref_std + self.ref_mean
        matched = np.clip(matched, 0.0, 1.0)
        return Image.fromarray((matched * 255).astype(np.uint8))


def _compute_pv_raw_stats(sample_limit=2000):
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['data']
    raw_dir = data_config['raw_dir']
    dataset_name = data_config.get('dataset_name', 'mohanty/PlantVillage')
    val_split_from_train = data_config.get('val_split_from_train', 0.15)
    split_seed = data_config.get('split_seed', 42)

    dataset_root = os.path.join(raw_dir, "plantvillage_hf")
    get_cached_color_image_root(raw_dir)
    metadata_paths = get_cached_hf_metadata_paths(dataset_name)
    leaf_map = load_leaf_map(metadata_paths["leaf_map"])

    official_train_samples = build_official_split_samples(
        metadata_paths["train_split"], dataset_root, leaf_map
    )
    train_samples, _ = split_train_samples_by_leaf_id(
        official_train_samples, val_split_from_train, split_seed
    )

    means = []
    stds = []
    for img_path, _, _ in tqdm(train_samples[:sample_limit], desc="Computing PV train stats", leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img).astype(np.float32) / 255.0
            means.append(arr.mean(axis=(0, 1)))
            stds.append(arr.std(axis=(0, 1)))
        except Exception:
            continue

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    print(f"PV train RGB stats: mean={mean.round(4)}, std={std.round(4)}")
    return mean, std


def _compute_coral_transform(pv_features: np.ndarray, pd_features: np.ndarray):
    mu_s = pv_features.mean(axis=0)
    mu_t = pd_features.mean(axis=0)

    cov_s = np.cov(pv_features.T) + np.eye(pv_features.shape[1]) * 1e-5
    cov_t = np.cov(pd_features.T) + np.eye(pd_features.shape[1]) * 1e-5

    U_s, S_s, _ = np.linalg.svd(cov_s)
    U_t, S_t, _ = np.linalg.svd(cov_t)

    sqrt_Cs = U_s @ np.diag(np.sqrt(S_s))
    inv_sqrt_Ct = U_t @ np.diag(1.0 / np.sqrt(S_t))

    return {
        'mu_s': mu_s,
        'mu_t': mu_t,
        'sqrt_Cs': sqrt_Cs,
        'inv_sqrt_Ct': inv_sqrt_Ct,
    }


def _apply_coral(x: np.ndarray, transform: dict) -> np.ndarray:
    x_white = (x - transform['mu_t']) @ transform['inv_sqrt_Ct']
    x_aligned = x_white @ transform['sqrt_Cs'] + transform['mu_s']
    return x_aligned


def _run_coral_evaluation(root_dir, segmented, split_keys, style_norm=False):
    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    available_splits = get_available_plantdoc_splits(root_dir)
    if split_keys:
        available_splits = [s for s in available_splits if s in split_keys]
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    label_parts = []
    if style_norm:
        label_parts.append("StyleNorm")
    label_parts.append("CORAL")
    label = "+".join(label_parts)
    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")
    print(f"Segmentation: {segmented}")
    print(f"Domain Adaptation: {label}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_best_checkpoints()
    if not models:
        raise RuntimeError("No checkpoints found")

    pv_mean, pv_std = None, None
    if style_norm:
        pv_mean, pv_std = _compute_pv_raw_stats()

    if style_norm and pv_mean is not None:
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            StyleNormalize(pv_mean, pv_std),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        eval_transform = get_val_test_transform()

    pv_loader_full, _ = get_plantvillage_train_loader_for_knn(batch_size=32, num_workers=0)
    pv_loader = DataLoader(_SubsetLoader(pv_loader_full.dataset, limit=300), batch_size=32, shuffle=False, num_workers=0)
    results: List[Dict] = []

    for split_name in available_splits:
        print(f"\nEvaluating split: {split_name}")

        base_dataset = None
        if segmented:
            base_dataset = PlantDocCOCODataset(
                root_dir, split_name, class_to_idx,
                transform=None, apply_segmentation=True
            )
            if len(base_dataset) == 0:
                continue

            print(f"Segmenting {len(base_dataset)} images...")
            for i in tqdm(range(len(base_dataset)), desc="Segmenting", leave=False):
                base_dataset[i]

        dataset = PlantDocCOCODataset(
            root_dir, split_name, class_to_idx,
            transform=eval_transform, apply_segmentation=segmented
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        for exp_key, model in models.items():
            print(f"Evaluating {exp_key.upper()} on {split_name} ({label})...")

            pv_rec = extract_backbone_features(model, pv_loader, device, desc=f"  PV features ({exp_key})")
            pd_rec = extract_backbone_features(model, loader, device, desc=f"  PD features ({exp_key})")

            coral_t = _compute_coral_transform(pv_rec['features'], pd_rec['features'])
            pd_aligned = _apply_coral(pd_rec['features'], coral_t)

            x = torch.tensor(pd_aligned, device=device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model.classifier(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

            metrics = calculate_metrics(pd_rec['labels'], preds, idx_to_class, probs)
            print(f"  {exp_key.upper()} {label} accuracy: {metrics['accuracy'] * 100:.2f}%")

            results.append(build_metric_row(
                method=exp_key.upper(),
                split=split_name,
                metrics=metrics,
                sample_count=len(dataset),
                checkpoint=f"{exp_key}_best.pth ({label.lower()})",
            ))

    os.makedirs(EVAL_DIR, exist_ok=True)
    if segmented and style_norm:
        out_name = "plantdoc_coral_stylenorm_segmented_results.csv"
    elif segmented:
        out_name = "plantdoc_coral_segmented_results.csv"
    elif style_norm:
        out_name = "plantdoc_stylenorm_coral_results.csv"
    else:
        out_name = "plantdoc_coral_results.csv"

    out_path = os.path.join(EVAL_DIR, out_name)
    fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved CORAL evaluation results to {out_path}")
    return out_path


class _SubsetLoader(Dataset):
    def __init__(self, dataset, limit=300):
        self.dataset = dataset
        self.limit = min(limit, len(dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.dataset[idx]


def _run_knn_evaluation(root_dir, segmented, split_keys, k):
    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    available_splits = get_available_plantdoc_splits(root_dir)
    if split_keys:
        available_splits = [s for s in available_splits if s in split_keys]
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")
    print(f"Segmentation: {segmented}")
    print(f"k-NN k={k}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_best_checkpoints()
    if not models:
        raise RuntimeError("No checkpoints found for k-NN evaluation")

    transform = get_val_test_transform()
    pv_loader, _ = get_plantvillage_train_loader_for_knn()
    results: List[Dict] = []

    for exp_key, model in models.items():
        print(f"\n=== k-NN for {exp_key.upper()} ===")

        pv_rec = extract_backbone_features(model, pv_loader, device, desc=f"PV train features ({exp_key})")
        print(f"  PlantVillage reference features: {pv_rec['features'].shape}")

        dataset = PlantDocCOCODataset(
            root_dir, available_splits[0], class_to_idx,
            transform=transform, apply_segmentation=segmented
        )
        if len(dataset) == 0:
            print(f"No mapped samples found for split '{available_splits[0]}'.")
            continue

        if segmented:
            print(f"Segmenting {len(dataset)} PlantDoc images...")
            for i in tqdm(range(len(dataset)), desc="Segmenting", leave=False):
                dataset[i]

        pd_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        pd_rec = extract_backbone_features(model, pd_loader, device, desc=f"PlantDoc query features ({exp_key})")
        print(f"  PlantDoc query features: {pd_rec['features'].shape}")

        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
        knn.fit(pv_rec['features'], pv_rec['labels'])
        preds = knn.predict(pd_rec['features'])
        probs = knn.predict_proba(pd_rec['features'])

        metrics = calculate_metrics(pd_rec['labels'], preds, idx_to_class, probs)
        print(f"{exp_key.upper()} k-NN accuracy: {metrics['accuracy'] * 100:.2f}%")

        results.append(build_metric_row(
            method=f"KNN-{exp_key.upper()}",
            split=available_splits[0],
            metrics=metrics,
            sample_count=len(dataset),
            checkpoint=f"{exp_key}_best.pth (k={k})",
        ))

    os.makedirs(EVAL_DIR, exist_ok=True)
    if segmented:
        out_name = "plantdoc_knn_segmented_results.csv"
    else:
        out_name = "plantdoc_knn_results.csv"

    out_path = os.path.join(EVAL_DIR, out_name)
    fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved k-NN evaluation results to {out_path}")
    return out_path


class SegmentedResDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label, crop, disease, path = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, crop, disease, path


def _run_multires_evaluation(root_dir, segmented, split_keys, scales):
    if not scales:
        scales = MULTIRES_DEFAULT_SCALES

    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    available_splits = get_available_plantdoc_splits(root_dir)
    if split_keys:
        available_splits = [s for s in available_splits if s in split_keys]
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")
    print(f"Segmentation: {segmented}")
    print(f"Resolution scales: {scales}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_best_checkpoints()
    if not models:
        raise RuntimeError("No checkpoints found for multi-resolution evaluation")

    results: List[Dict] = []

    for split_name in available_splits:
        print(f"\nEvaluating split: {split_name}")

        if segmented:
            base_dataset = PlantDocCOCODataset(
                root_dir, split_name, class_to_idx,
                transform=None, apply_segmentation=True
            )
            print(f"Segmenting {len(base_dataset)} images...")
            for i in tqdm(range(len(base_dataset)), desc="Segmenting", leave=False):
                base_dataset[i]
        else:
            base_dataset = None

        for exp_key, model in models.items():
            print(f"Evaluating {exp_key.upper()} on {split_name} (multi-resolution)...")

            scale_logits = []
            for scale in scales:
                t = build_transform_for_scale(scale)
                if segmented:
                    dataset = SegmentedResDataset(base_dataset, transform=t)
                else:
                    dataset = PlantDocCOCODataset(
                        root_dir, split_name, class_to_idx,
                        transform=t, apply_segmentation=False
                    )

                loader = DataLoader(
                    dataset, batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available(),
                )
                rec = evaluate_single_and_collect(model, loader, device, desc=f"  {exp_key} @ {scale}px")
                scale_logits.append(torch.tensor(rec['logits']))

            avg_logits = torch.stack(scale_logits).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=1).numpy()
            preds = avg_logits.argmax(dim=1).numpy()
            y_true = rec['labels']

            metrics = calculate_metrics(y_true, preds, idx_to_class, probs)
            print(f"{exp_key.upper()} multi-res accuracy on {split_name}: {metrics['accuracy'] * 100:.2f}%")

            scale_str = "x".join(str(s) for s in scales)
            results.append(build_metric_row(
                method=exp_key.upper(),
                split=split_name,
                metrics=metrics,
                sample_count=len(dataset),
                checkpoint=f"{exp_key}_best.pth (multires-{scale_str})",
            ))

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_name = "plantdoc_multires_results.csv"
    if segmented:
        out_name = "plantdoc_multires_segmented_results.csv"

    out_path = os.path.join(EVAL_DIR, out_name)
    fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved multi-resolution evaluation results to {out_path}")
    return out_path


def _run_knn_multires_segmented_evaluation(root_dir, segmented, split_keys, scales, k=11):
    if not scales:
        scales = MULTIRES_DEFAULT_SCALES

    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    available_splits = get_available_plantdoc_splits(root_dir)
    if split_keys:
        available_splits = [s for s in available_splits if s in split_keys]
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    print("=" * 70)
    print(f"QA-LoRA V3 Segmented k-NN Multi-Res Evaluation")
    print(f"Scale pyramid: {'x'.join(str(s) for s in scales)}, k={k}")
    print("=" * 70)
    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")
    print(f"Segmentation: {segmented}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from training.qalora_trainer import QALoRATrainer
    trainer = QALoRATrainer(None, None, len(class_to_idx))
    trainer.load_checkpoint('best')
    model = trainer.model.to(device)
    model.eval()

    transform = get_val_test_transform()
    pv_loader, _ = get_plantvillage_train_loader_for_knn()
    results: List[Dict] = []

    for split_name in available_splits:
        print(f"\nEvaluating split: {split_name}")

        if segmented:
            base_dataset = PlantDocCOCODataset(
                root_dir, split_name, class_to_idx,
                transform=None, apply_segmentation=True
            )
            print(f"Segmenting {len(base_dataset)} images...")
            for i in tqdm(range(len(base_dataset)), desc="Segmenting", leave=False):
                base_dataset[i]
        else:
            base_dataset = None

        pv_rec = extract_backbone_features(model, pv_loader, device, desc=f"PV train features (qalora)")
        print(f"  PlantVillage reference features: {pv_rec['features'].shape}")

        scale_features = []
        for scale in scales:
            t = build_transform_for_scale(scale)
            if segmented:
                dataset = SegmentedResDataset(base_dataset, transform=t)
            else:
                dataset = PlantDocCOCODataset(
                    root_dir, split_name, class_to_idx,
                    transform=t, apply_segmentation=False
                )
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
            rec = extract_backbone_features(model, loader, device, desc=f"  PD features @ {scale}px")
            print(f"  PlantDoc features @ {scale}px: {rec['features'].shape}")
            scale_features.append(torch.tensor(rec['features']))

        avg_features = torch.stack(scale_features).mean(dim=0).numpy()

        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
        knn.fit(pv_rec['features'], pv_rec['labels'])
        preds = knn.predict(avg_features)
        probs = knn.predict_proba(avg_features)

        metrics = calculate_metrics(rec['labels'], preds, idx_to_class, probs)
        print(f"QALORA seg+knn+multires accuracy on {split_name}: {metrics['accuracy'] * 100:.2f}%")

        scale_str = "x".join(str(s) for s in scales)
        results.append(build_metric_row(
            method="QALORA-SEG-KNN-MULTIRES",
            split=split_name,
            metrics=metrics,
            sample_count=len(dataset),
            checkpoint=f"qalora_best.pth (seg-knn-multires-{scale_str})",
        ))

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_name = "plantdoc_qalora_seg_knn_multires_results.csv"
    out_path = os.path.join(EVAL_DIR, out_name)
    fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved Seg+KNN+Multi-Res evaluation results to {out_path}")
    return out_path


def _run_quality_gate_evaluation(root_dir, segmented, split_keys, threshold=0.6):
    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    available_splits = get_available_plantdoc_splits(root_dir)
    if split_keys:
        available_splits = [s for s in available_splits if s in split_keys]
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")
    print(f"Segmentation: {segmented}")
    print(f"Quality gate threshold: {threshold}")

    dual_csv = os.path.join(EVAL_DIR, "plantdoc_dual_split_results.csv")
    baseline_metrics = {}
    if os.path.exists(dual_csv):
        with open(dual_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['method'].upper(), row['split'])
                baseline_metrics[key] = row

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_best_checkpoints()
    if not models:
        raise RuntimeError("No checkpoints found for quality-gate evaluation")

    results: List[Dict] = []

    for split_name in available_splits:
        print(f"\nEvaluating split: {split_name}")
        base_dataset = PlantDocCOCODataset(
            root_dir, split_name, class_to_idx,
            transform=None, apply_segmentation=segmented
        )
        if len(base_dataset) == 0:
            print(f"No mapped samples found for split '{split_name}'.")
            continue

        if segmented:
            print(f"Segmenting {len(base_dataset)} images...")
            for i in tqdm(range(len(base_dataset)), desc="Segmenting", leave=False):
                base_dataset[i]

        paths = [base_dataset.samples[i][0] for i in range(len(base_dataset))]
        quality_scores = np.zeros(len(paths), dtype=np.float32)
        for i, p in enumerate(tqdm(paths, desc="Quality scoring", leave=False)):
            try:
                with Image.open(p) as img:
                    quality_scores[i] = compute_image_quality_score(img)
            except Exception:
                quality_scores[i] = 0.0

        keep_mask = quality_scores >= threshold
        n_total = len(paths)
        n_kept = int(keep_mask.sum())
        n_filtered = n_total - n_kept
        print(f"Quality gate: filtered {n_filtered}/{n_total} images (kept {n_kept})")

        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

        for method in ['LORA', 'QLORA', 'QKLORA']:
            key = (method, split_name)
            if key in baseline_metrics:
                row = baseline_metrics[key]
                results.append({
                    'method': method,
                    'split': split_name,
                    'accuracy': float(row['accuracy']) / 100.0,
                    'f1_macro': float(row['f1_macro']) / 100.0,
                    'binary_accuracy': float(row['binary_accuracy']) / 100.0,
                    'binary_f1': float(row['binary_f1']) / 100.0,
                    'binary_roc_auc': float(row['binary_roc_auc']),
                    'both_correct_pct': float(row['both_correct_pct']),
                    'name_only_correct_pct': float(row['name_only_correct_pct']),
                    'disease_only_correct_pct': float(row['disease_only_correct_pct']),
                    'none_correct_pct': float(row['none_correct_pct']),
                    'sample_count': int(row['sample_count']),
                    'checkpoint': f"{method.lower()}_best.pth (ALL)"
                })

        if n_kept == 0:
            continue

        transform = get_val_test_transform()
        base_tform = base_dataset.transform
        base_dataset.transform = transform
        passed_dataset = _QualityFilteredDataset(base_dataset, keep_indices)

        loader = DataLoader(passed_dataset, batch_size=32, shuffle=False, num_workers=2)

        for exp_key, model in models.items():
            print(f"Evaluating {exp_key.upper()} on quality-passed {split_name}...")
            rec = evaluate_single_and_collect(model, loader, device, desc=f"  {exp_key} quality-gate")
            metrics_passed = calculate_metrics(rec['labels'], rec['preds'], idx_to_class, rec['probs'])
            print(f"  Quality-passed ({threshold}): {metrics_passed['accuracy']*100:.2f}%")

            results.append(build_metric_row(
                method=exp_key.upper(), split=split_name, metrics=metrics_passed,
                sample_count=n_kept, checkpoint=f"{exp_key}_best.pth (quality>={threshold})"
            ))

        base_dataset.transform = base_tform

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_name = "plantdoc_quality_gate_segmented_results.csv" if segmented else "plantdoc_quality_gate_results.csv"
    out_path = os.path.join(EVAL_DIR, out_name)
    fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved quality-gate evaluation results to {out_path}")
    return out_path


def evaluate_single_and_collect(model, loader, device, desc="Evaluating"):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    all_probs = []
    all_logits = []

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch[0]
            labels = batch[1]
            paths = batch[4]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

            pbar.set_postfix({"batch_acc": f"{(preds == labels).float().mean().item():.2%}"})

    return {
        'paths': list(all_paths),
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs),
        'logits': np.array(all_logits),
    }


def save_probs_csv(records: Dict[str, Dict], model_keys: List[str], idx_to_class: Dict[int, str], out_path: str):
    num_classes = len(idx_to_class)
    fieldnames = ['image_path', 'true_idx']
    for key in model_keys:
        for c in range(num_classes):
            fieldnames.append(f"{key}_{c}")

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        n = len(records[model_keys[0]]['labels'])
        for i in range(n):
            row = {
                'image_path': records[model_keys[0]]['paths'][i],
                'true_idx': int(records[model_keys[0]]['labels'][i]),
            }
            for key in model_keys:
                for c in range(num_classes):
                    row[f"{key}_{c}"] = round(float(records[key]['probs'][i, c]), 8)
            writer.writerow(row)

    print(f"Saved per-sample probabilities to {out_path}")


def compute_ensemble_from_probs_csv(csv_path: str, idx_to_class: Dict[int, str], model_keys: Optional[List[str]] = None) -> Dict:
    model_keys = model_keys or ['lora', 'qlora', 'qklora']
    num_classes = len(idx_to_class)

    df = pd.read_csv(csv_path)
    true_idx = df['true_idx'].values.astype(int)

    prob_arrays = []
    for key in model_keys:
        cols = [f"{key}_{c}" for c in range(num_classes)]
        prob_arrays.append(df[cols].values.astype(np.float64))

    avg_probs = np.mean(prob_arrays, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)

    return calculate_metrics(true_idx, y_pred, idx_to_class, avg_probs)


def run_plantdoc_evaluation(
    root_dir: Optional[str] = None,
    segmented: bool = False,
    ensemble: bool = False,
    split_keys: Optional[List[str]] = None,
    knn: bool = False,
    knn_k: int = 11,
    multires: bool = False,
    multires_scales: Optional[List[int]] = None,
    quality_gate: bool = False,
    quality_threshold: float = 0.3,
    style_norm: bool = False,
    feature_align: bool = False,
    v3: bool = False,
):
    global TRAINER_MAP, CHECKPOINT_DIR, EVAL_DIR, PROBS_CSV, _V3_MODE
    _orig_trainer_map = TRAINER_MAP
    _orig_cp_dir = CHECKPOINT_DIR
    _orig_eval_dir = EVAL_DIR
    _orig_probs_csv = PROBS_CSV

    _V3_MODE = v3

    if v3:
        from training.lora_trainer_v3 import LoRATrainerV3
        from training.qlora_trainer_v3 import QLoRATrainerV3
        from training.qalora_trainer import QALoRATrainer
        TRAINER_MAP = {
            'lora': LoRATrainerV3,
            'qlora': QLoRATrainerV3,
            'qalora': QALoRATrainer,
        }
        CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints_v3")
        EVAL_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "eval_v3")
        PROBS_CSV = os.path.join(EVAL_DIR, "plantdoc_segmented_probs_v3.csv")
    else:
        TRAINER_MAP = _orig_trainer_map
        CHECKPOINT_DIR = _orig_cp_dir
        EVAL_DIR = _orig_eval_dir
        PROBS_CSV = _orig_probs_csv

    try:
        print("=" * 70)
        labels = []
        if segmented:
            labels.append("Segmented")
        if knn:
            labels.append("k-NN")
        if ensemble:
            labels.append("Ensemble")
        if multires:
            labels.append(f"Multi-Res-{'-'.join(str(s) for s in (multires_scales or MULTIRES_DEFAULT_SCALES))}")
        if quality_gate:
            labels.append(f"Quality-Gate-{quality_threshold}")
        if style_norm:
            labels.append("StyleNorm")
        if feature_align:
            labels.append("CORAL")
        label = " ".join(labels) if labels else "Standard"
        print(f"PlantDoc {label} Evaluation")
        print("=" * 70)
    
        if feature_align:
            return _run_coral_evaluation(root_dir, segmented, split_keys, style_norm=style_norm)
    
        if quality_gate:
            return _run_quality_gate_evaluation(root_dir, segmented, split_keys, threshold=quality_threshold)
    
        if multires and knn:
            return _run_knn_multires_segmented_evaluation(root_dir, segmented, split_keys, multires_scales, knn_k)

        root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")
    
        class_to_idx, idx_to_class = load_class_labels()
    
        if knn:
            return _run_knn_evaluation(root_dir, segmented, split_keys, knn_k)
    
        if multires:
            return _run_multires_evaluation(root_dir, segmented, split_keys, multires_scales)
    
        if style_norm:
            pv_mean, pv_std = _compute_pv_raw_stats()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                StyleNormalize(pv_mean, pv_std),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = get_val_test_transform()
    
        available_splits = get_available_plantdoc_splits(root_dir)
        if split_keys:
            available_splits = [s for s in available_splits if s in split_keys]
        if not available_splits:
            raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")
    
        print(f"Using PlantDoc dataset root: {root_dir}")
        print(f"Available splits: {', '.join(available_splits)}")
        print(f"Segmentation: {segmented}")
        print(f"Ensemble: {ensemble}")
        print(f"Style Normalization: {style_norm}")
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results: List[Dict] = []
    
        if ensemble and not segmented:
            if not os.path.exists(PROBS_CSV):
                raise RuntimeError(
                    f"Missing {PROBS_CSV}. "
                    "Run option 2 (segmented evaluation) first to generate per-model probabilities."
                )
    
            print(f"Reading per-sample probabilities from {PROBS_CSV}")
            num_rows = sum(1 for _ in open(PROBS_CSV)) - 1
            print(f"Found {num_rows} samples")
    
            for split_name in available_splits:
                print(f"\nComputing ensemble for split: {split_name}")
                metrics = compute_ensemble_from_probs_csv(PROBS_CSV, idx_to_class)
    
                print(f"ENSEMBLE accuracy on {split_name}: {metrics['accuracy'] * 100:.2f}%")
                results.append(build_metric_row(
                    method="ENSEMBLE",
                    split=split_name,
                    metrics=metrics,
                    sample_count=num_rows,
                    checkpoint="lora_best+qlora_best+qklora_best.pth",
                ))
        else:
            models = load_best_checkpoints()
            if not models:
                raise RuntimeError("No checkpoints found for evaluation")
    
            all_records = {}
    
            for split_name in available_splits:
                print(f"\nEvaluating split: {split_name}")
                dataset = PlantDocCOCODataset(
                    root_dir, split_name, class_to_idx,
                    transform=transform, apply_segmentation=segmented
                )
                if len(dataset) == 0:
                    print(f"No mapped samples found for split '{split_name}'.")
                    continue
    
                if segmented:
                    print(f"Segmenting {len(dataset)} images...")
                    for i in tqdm(range(len(dataset)), desc="Segmenting", leave=False):
                        dataset[i]
    
                loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
                split_records = {}
                for exp_key, trainer_cls in TRAINER_MAP.items():
                    if exp_key not in models:
                        continue
    
                    print(f"Evaluating {exp_key.upper()} on {split_name}...")
                    rec = evaluate_single_and_collect(models[exp_key], loader, device)
                    split_records[exp_key] = rec
    
                    metrics = calculate_metrics(rec['labels'], rec['preds'], idx_to_class, rec['probs'])
                    print(f"{exp_key.upper()} accuracy on {split_name}: {metrics['accuracy'] * 100:.2f}%")
                    results.append(build_metric_row(
                        method=exp_key.upper(),
                        split=split_name,
                        metrics=metrics,
                        sample_count=len(dataset),
                        checkpoint=f"{exp_key}_best.pth",
                    ))
    
                all_records[split_name] = split_records
    
            if segmented:
                first_split = available_splits[0]
                save_probs_csv(all_records[first_split], list(TRAINER_MAP.keys()), idx_to_class, PROBS_CSV)
    
            if ensemble and segmented:
                print(f"\nComputing ensemble from saved probabilities...")
                for split_name in available_splits:
                    print(f"Computing ensemble for split: {split_name}")
                    metrics = compute_ensemble_from_probs_csv(PROBS_CSV, idx_to_class)
    
                    print(f"ENSEMBLE accuracy on {split_name}: {metrics['accuracy'] * 100:.2f}%")
                    results.append(build_metric_row(
                        method="ENSEMBLE",
                        split=split_name,
                        metrics=metrics,
                        sample_count=len(all_records[split_name][list(TRAINER_MAP.keys())[0]]['labels']),
                        checkpoint="lora_best+qlora_best+qklora_best.pth",
                    ))
    
        os.makedirs(EVAL_DIR, exist_ok=True)
        if style_norm:
            out_name = "plantdoc_stylenorm_segmented_results.csv" if segmented else "plantdoc_stylenorm_results.csv"
        elif knn:
            out_name = "plantdoc_knn_segmented_results.csv" if segmented else "plantdoc_knn_results.csv"
        elif multires:
            out_name = "plantdoc_multires_segmented_results.csv" if segmented else "plantdoc_multires_results.csv"
        elif ensemble:
            out_name = "plantdoc_ensemble_segmented_results.csv"
        elif segmented:
            out_name = "plantdoc_segmented_results.csv"
        else:
            out_name = "plantdoc_dual_split_results.csv"
    
        out_path = os.path.join(EVAL_DIR, out_name)
        fieldnames = ['method', 'split'] + METRIC_FIELDS + ['sample_count', 'checkpoint']
        with open(out_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
        print(f"\nSaved evaluation results to {out_path}")
        return out_path
    finally:
        TRAINER_MAP = _orig_trainer_map
        CHECKPOINT_DIR = _orig_cp_dir
        EVAL_DIR = _orig_eval_dir
        PROBS_CSV = _orig_probs_csv
        _V3_MODE = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PlantDoc evaluation with optional segmentation and ensemble")
    parser.add_argument('--segmented', action='store_true', help='Apply foreground segmentation before inference')
    parser.add_argument('--ensemble', action='store_true', help='Read saved probs CSV and compute ensemble metrics')
    parser.add_argument('--splits', nargs='*', default=None, help='Limit to specific splits (e.g. test train valid)')
    parser.add_argument('--root-dir', default=None, help='PlantDoc dataset root directory')
    parser.add_argument('--knn', action='store_true', help='Run k-NN on frozen backbone features (no classifier)')
    parser.add_argument('--k', type=int, default=11, help='k for k-NN classifier (default: 11)')
    parser.add_argument('--multires', action='store_true', help='Run multi-resolution inference pyramid (no training)')
    parser.add_argument('--scales', type=int, nargs='*', default=None, help='Resolution scales for multi-res (default: 192 224 256 320)')
    parser.add_argument('--quality-gate', action='store_true', help='Exclude low-quality PlantDoc images from metrics')
    parser.add_argument('--quality-threshold', type=float, default=0.6, help='Minimum quality score to keep (0..1, default: 0.6)')
    parser.add_argument('--style-norm', action='store_true', help='Apply test-time style normalization (match PlantDoc to PlantVillage color stats)')
    parser.add_argument('--feature-align', action='store_true', help='Apply CORAL feature whitening at test time (align PlantDoc features to PlantVillage)')
    parser.add_argument('--v3', action='store_true', help='Use V3 checkpoints (LoRA V3 / QLoRA V3 / QA-LoRA V3)')
    args = parser.parse_args()

    run_plantdoc_evaluation(
        root_dir=args.root_dir,
        segmented=args.segmented,
        ensemble=args.ensemble,
        split_keys=args.splits,
        knn=args.knn,
        knn_k=args.k,
        multires=args.multires,
        multires_scales=args.scales,
        quality_gate=args.quality_gate,
        quality_threshold=args.quality_threshold,
        style_norm=args.style_norm,
        feature_align=args.feature_align,
        v3=args.v3,
    )
