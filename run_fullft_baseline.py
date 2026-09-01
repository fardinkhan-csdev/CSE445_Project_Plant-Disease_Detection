"""Full fine-tuning baseline for the PEFT paper.

Trains the entire EfficientNet-B0 (no frozen backbone) on PlantVillage for a
short number of epochs to get a baseline accuracy/size/time number for
comparison against LoRA / QLoRA / QA-LoRA.
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn

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

    print("Loading data...")
    train_loader, val_loader, test_loader, meta = get_data_loaders(
        config_path="config/base_config_v3.yaml"
    )
    print(f"  train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    print("Building model (full fine-tuning, no freezing)...")
    model = get_efficientnet_b0(num_classes=38, pretrained=True)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable={trainable:,} / {total:,} (100.00%)")

    epochs = 1
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for batch in train_loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = running_loss / n

        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics.get("accuracy", val_metrics.get("acc", 0.0))
        print(f"  epoch {epoch}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    ckpt_path = "experiments/results/checkpoints_v3/fullft_best.pth"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\nCheckpoint saved: {ckpt_path} ({size_mb:.2f} MB)")

    summary = {
        "method": "FullFT",
        "trainable_params": trainable,
        "total_params": total,
        "best_val_acc": best_val_acc,
        "checkpoint_size_mb": size_mb,
        "training_time_s": elapsed,
        "epochs": epochs,
    }
    out_json = "experiments/results/eval_v3/fullft_summary.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to {out_json}")
    print(json.dumps(summary, indent=2))
    print("\nTo evaluate, run: py -3.11 eval_fullft_baseline.py")


if __name__ == "__main__":
    main()
