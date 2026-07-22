# How This Project Works

## Goal
This project compares three parameter-efficient fine-tuning methods for plant leaf disease classification:

1. `LoRA`
2. `QLoRA`
3. `Q/K LoRA`

All three methods use `EfficientNet-B0` as the backbone model.

## Dataset
The project uses the PlantVillage dataset, but the training pipeline is now locked to the official Hugging Face `color` split only.

That means:
- only RGB color images are used
- grayscale images are not used
- segmented images are not used

## Split Strategy
The split logic is now:

1. Use the official Hugging Face `color/train`
2. Use the official Hugging Face `color/test`
3. Split `color/train` into `train` and `val`

The important detail is that validation is created by `leaf_id`, not by raw image file.

Why this matters:
- PlantVillage often has multiple photos of the same physical leaf
- a naive image-level split can leak very similar images across sets
- `leaf_id` grouping reduces that leakage risk

Approximate overall ratio:
- `68%` train
- `12%` val
- `20%` test

## Offline Training Behavior
Training and testing are now designed to be non-downloading.

The launcher and main training path expect these assets to already exist locally:
- PlantVillage `color` images
- official HF `color` split metadata
- cached EfficientNet-B0 pretrained weights

If those assets are missing, the code stops with a clear message instead of downloading during training.

## What `download_assets.py` Does
This separate setup script prepares everything training needs:

1. Downloads the local PlantVillage `color` archive
2. Caches the official HF split metadata used for the offline split logic
3. Downloads the pretrained EfficientNet-B0 weights

After that, `launcher.py` is used only for training or tests.

## Image Processing
Training images:
1. Resize to `256x256`
2. Random crop to `224x224`
3. Random horizontal flip
4. Random rotation up to `15` degrees
5. Brightness, contrast, and saturation jitter
6. ImageNet normalization

Validation and test images:
1. Resize to `256×256` (preserve aspect ratio)
2. Center crop to `224×224`
3. ImageNet normalization

## PEFT Methods (LoRA vs QLoRA vs Q/K LoRA)

All three share the same inference API: build model → load checkpoint → `model(image)`.

| Method | What is quantized | LoRA targets | Typical checkpoint |
|--------|-------------------|--------------|-------------------|
| **LoRA** | Nothing (full FP32 backbone) | All pointwise convs + classifier | ~18 MB |
| **QLoRA** | Q-path MBConv expand/project 1×1 (INT8) | Q-path + classifier only | ~8 MB |
| **Q/K LoRA** | Q-path INT8; K-path SE layers stay FP32 | Q-path (r=16) + K-path SE + classifier (r=4) | ~9 MB |

Implementation lives in `models/peft/` (`lora.py`, `qlora.py`, `qklora.py`, `int8_utils.py`).

**Q/K naming**: Q = **Quantized** path, K = **Kept** high-precision path (not transformer Query/Key).

## Deployment artifacts
- Model checkpoint: `experiments/results/checkpoints/<method>_best.pth`
- Class labels: `config/class_labels.json` (auto-exported when data loads)

## Model Roles
- `models/backbone/`: the EfficientNet-B0 backbone
- `models/peft/`: the LoRA, QLoRA, and Q/K LoRA implementations
- `training/`: the trainer classes
- `evaluation/`: metrics and test-time evaluation
- `experiments/`: orchestration and saved outputs

## What Gets Saved
When training finishes, the project stores:
- checkpoints
- logs
- learning curves
- confusion matrices
- class-wise metrics
- the experiment summary CSV

## Summary
The current workflow is:

1. Prepare assets once with `download_assets.py`
2. Start training with `launcher.py`
3. Train only on the official HF `color` split
4. Validate using a leakage-aware `leaf_id` split
