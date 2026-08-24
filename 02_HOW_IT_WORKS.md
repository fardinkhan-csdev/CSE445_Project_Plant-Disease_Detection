# How This Project Works

## Goal

This project compares four parameter-efficient fine-tuning (PEFT) methods for plant leaf disease classification, all using EfficientNet-B0 as the backbone:

1. **LoRA** — standard low-rank adaptation (Hu et al. 2021)
2. **QLoRA** — 4-bit NF4 quantization + LoRA (Dettmers et al. 2023)
3. **QA-LoRA** — group-wise quantization + grouped LoRA (Xu et al. 2024)
4. **Q/K LoRA** — selective INT8 Q-path + FP32 K-path (custom)

Methods 1–3 are the **V3 track** (current). Method 4 is the **V1 track** (legacy).

## Two Tracks

| Track | Methods | Entry Point | Status |
|-------|---------|-------------|--------|
| **V3** (current) | LoRA V3, QLoRA V3, QA-LoRA V3 | `main_v3.py` / `launcher_v3.py` | Active |
| **V1** (legacy) | LoRA, QLoRA, Q/K LoRA | `main.py` / `launcher.py` | Still runnable |

## Dataset

The training pipeline is locked to the official Hugging Face PlantVillage `color` split only:
- RGB color images only (no grayscale, no segmented)
- `color/train` → split into train/val by `leaf_id`
- `color/test` → held out for final evaluation
- Approximate ratio: 68% train / 12% val / 20% test

Validation is created by `leaf_id` grouping, not by raw image, to prevent leakage from multiple photos of the same physical leaf.

## Offline Training

Training expects all assets to exist locally before starting. If assets are missing, the code stops with a clear message.

Run `download_assets.py` once to prepare:
- PlantVillage `color` images
- HF `color` split metadata
- EfficientNet-B0 pretrained weights

## Image Processing

**Training:**
1. Resize to 256×256
2. Random crop to 224×224
3. Random horizontal flip
4. Random rotation (±15°)
5. Color jitter (brightness, contrast, saturation)
6. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Validation/Test:**
1. Resize to 256×256 (preserve aspect ratio)
2. Center crop to 224×224
3. ImageNet normalization

## PEFT Methods

### LoRA (V3)
- **Paper**: Hu et al. 2021
- **Quantization**: None (full FP32 backbone)
- **Targets**: All pointwise (1×1) convolutions + `classifier.fc`
- **Trainable fraction**: ~8% of total parameters
- **Checkpoint**: ~18 MB
- **V3 extras**: Merge support for zero-overhead inference

### QLoRA (V3)
- **Paper**: Dettmers et al. 2023
- **Quantization**: bitsandbytes 4-bit NF4 on Q-path 1×1 convs
- **Targets**: Q-path pointwise convs + `features.8.0` + `classifier.fc`
- **Compute dtype**: Dequantized to bfloat16 during forward
- **Checkpoint**: ~8 MB

### QA-LoRA (V3)
- **Paper**: Xu et al. 2024 (ICLR)
- **Quantization**: Group-wise INT8 with learned scale/zero-point per group
- **LoRA A**: Grouped — shape `(L, rank)` instead of `(D_in, rank)`
- **Targets**: Q-path pointwise convs + `features.8.0` + `classifier.fc`
- **Does not use PEFT** — `QALoRAConv2d` replaces `nn.Conv2d` directly
- **Checkpoint**: ~9.5 MB

### Q/K LoRA (V1)
- **Custom design** — not from a specific paper
- **Q-path** (Quantized): MBConv 1×1 convs → INT8 + LoRA rank 16
- **K-path** (Kept FP32): SE layers + classifier → FP32 + LoRA rank 4
- **Depthwise convs**: Frozen, no LoRA
- **Checkpoint**: ~12 MB

## Layer Target Summary

| Layer | LoRA V3 | QLoRA V3 | QA-LoRA V3 | Q/K LoRA |
|-------|---------|----------|------------|----------|
| MBConv expand 1×1 | ✅ adapter | ✅ NF4 + adapter | ✅ group-wise + adapter | ✅ INT8 + adapter |
| MBConv project 1×1 | ✅ adapter | ✅ NF4 + adapter | ✅ group-wise + adapter | ✅ INT8 + adapter |
| Head conv (`features.8.0`) | ✅ adapter | ✅ NF4 + adapter | ✅ group-wise + adapter | ❌ frozen |
| SE `fc1`/`fc2` | ❌ frozen | ❌ frozen | ❌ frozen | ✅ FP32 + adapter (r=4) |
| Depthwise 3×3 | ❌ frozen | ❌ frozen | ❌ frozen | ❌ frozen |
| Stem (`features.0.0`) | ❌ frozen | ❌ frozen | ❌ frozen | ❌ frozen |
| `classifier.fc` | ✅ adapter | ✅ adapter | ✅ adapter | ✅ adapter (r=4) |

## Deployment Artifacts

- **Checkpoints**: `experiments/results/checkpoints/<method>_best.pth`
- **Class labels**: `config/class_labels.json` (auto-exported when data loads)
- **Cross-method ranking**: `experiments/results/eval/cross_method_ranking.csv`
- **Dashboard**: `experiments/results/dashboard.html` (self-contained, no server needed)

## Model Roles

- `models/backbone/` — EfficientNet-B0 backbone
- `models/peft/` — All PEFT implementations (LoRA, QLoRA, QA-LoRA, Q/K LoRA)
- `training/` — Trainer classes (base + method-specific)
- `evaluation/` — Metrics, confusion matrix, evaluator
- `experiments/` — Orchestration and saved outputs
- `web_app/` — Web UI for inference and visualization

## What Gets Saved

When training finishes:
- Checkpoints (best, last, optionally per-epoch)
- Training logs
- Learning curves (loss, accuracy vs epoch)
- Confusion matrices
- Class-wise metrics (precision, recall, F1)
- Experiment summary CSV

## Summary

1. Prepare assets once with `download_assets.py`
2. Train with `launcher_v3.py` (or `main_v3.py` directly)
3. Evaluate with `launcher_test_v3.py`
4. Generate dashboard with `generate_dashboard.py`
5. Launch web UI with `run_web_ui.py`
