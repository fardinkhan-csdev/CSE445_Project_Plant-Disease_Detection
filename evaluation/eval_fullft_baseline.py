"""Evaluate a saved full-FT checkpoint on the PlantVillage test set."""
import os
import sys
import json
import torch

sys.path.append(".")
from data.data_loader import get_data_loaders
from models.backbone.efficientnet_b0 import get_efficientnet_b0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(out.cpu())
            all_labels.append(y)
    accuracy = correct / max(total, 1)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    macro_f1_per_class = []
    for c in range(int(labels.max().item()) + 1):
        tp = ((preds.argmax(1) == c) & (labels == c)).sum().item()
        fp = ((preds.argmax(1) == c) & (labels != c)).sum().item()
        fn = ((preds.argmax(1) != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_f1_per_class.append(f1)
    macro_f1 = sum(macro_f1_per_class) / len(macro_f1_per_class) if macro_f1_per_class else 0.0
    return {"accuracy": accuracy, "f1_macro": macro_f1, "total": total}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = "experiments/results/checkpoints_v3/fullft_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print("Loading data...")
    _, val_loader, test_loader, _ = get_data_loaders(config_path="config/base_config_v3.yaml")
    print(f"  val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    print("Loading model...")
    model = get_efficientnet_b0(num_classes=38, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    print("Evaluating on val set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  val accuracy: {val_metrics['accuracy']*100:.2f}%")
    print(f"  val f1_macro: {val_metrics['f1_macro']:.4f}")

    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"  test accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  test f1_macro: {test_metrics['f1_macro']:.4f}")

    summary_path = "experiments/results/eval_v3/fullft_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {"method": "FullFT"}
    summary["val_accuracy"] = val_metrics["accuracy"]
    summary["val_f1_macro"] = val_metrics["f1_macro"]
    summary["test_accuracy"] = test_metrics["accuracy"]
    summary["test_f1_macro"] = test_metrics["f1_macro"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
