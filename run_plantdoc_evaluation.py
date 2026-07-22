import csv
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import get_val_test_transform
from plantdoc_mapping import PlantDocCOCODataset, get_available_plantdoc_splits, load_class_labels
from training.lora_trainer import LoRATrainer
from training.qlora_trainer import QLoRATrainer
from training.qklora_trainer import QKLoRATrainer

TRAINER_MAP = {
    'lora': LoRATrainer,
    'qlora': QLoRATrainer,
    'qklora': QKLoRATrainer,
}


def evaluate_model_on_split(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def run_plantdoc_evaluation(root_dir: str | None = None):
    print("=" * 70)
    print("PlantDoc Cross-Split Evaluation")
    print("=" * 70)

    root_dir = root_dir or os.path.join(PROJECT_ROOT, "data", "raw", "plantdoc_roboflow_cocojson")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"PlantDoc dataset root not found: {root_dir}")

    class_to_idx, idx_to_class = load_class_labels()
    transform = get_val_test_transform()
    available_splits = get_available_plantdoc_splits(root_dir)
    if not available_splits:
        raise RuntimeError(f"No PlantDoc split folders found under {root_dir}")

    print(f"Using PlantDoc dataset root: {root_dir}")
    print(f"Available splits: {', '.join(available_splits)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results: List[Dict[str, object]] = []

    for split_name in available_splits:
        print(f"\nEvaluating split: {split_name}")
        dataset = PlantDocCOCODataset(root_dir, split_name, class_to_idx, transform=transform)
        if len(dataset) == 0:
            print(f"No mapped samples found for split '{split_name}'.")
            continue

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        for exp_key, trainer_cls in TRAINER_MAP.items():
            checkpoint_path = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints", f"{exp_key}_best.pth")
            if not os.path.exists(checkpoint_path):
                print(f"Skipping {exp_key}: checkpoint not found at {checkpoint_path}")
                continue

            print(f"\nEvaluating {exp_key.upper()} on {split_name}...")
            trainer = trainer_cls(None, None, len(class_to_idx))
            trainer.load_checkpoint('best')
            model = trainer.model.to(device)
            accuracy = evaluate_model_on_split(model, loader, device)
            print(f"{exp_key.upper()} accuracy on {split_name}: {accuracy * 100:.2f}%")
            results.append({
                'method': exp_key.upper(),
                'split': split_name,
                'accuracy': round(accuracy * 100, 2),
                'sample_count': len(dataset),
                'checkpoint': os.path.basename(checkpoint_path),
            })

    eval_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "plantdoc_dual_split_results.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'split', 'accuracy', 'sample_count', 'checkpoint'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved evaluation results to {out_path}")


if __name__ == '__main__':
    run_plantdoc_evaluation()
