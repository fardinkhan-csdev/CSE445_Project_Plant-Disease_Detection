# What To Do Now
Use this file as the shortest path from setup to training.

## Step 1
Open a terminal in `d:\Leaf Disease Classification`.

## Step 2
Prepare all required local assets once:

```powershell
py -3.11 download_assets.py
```

This downloads and caches:
- the local PlantVillage `color` image archive
- the official Hugging Face `color` split metadata
- the pretrained EfficientNet-B0 weights

## Step 3
Start the one-click launcher:

```powershell
py -3.11 launcher.py
```

Current launcher menu:
1. Train `LoRA`
2. Train `QLoRA`
3. Train `Q/K LoRA`
4. Train all three
5. Run tests
6. Exit

Important:
- the launcher is now training/test only
- it does **not** download assets
- if required assets are missing, it stops and tells you to run `py -3.11 download_assets.py`

## Direct Commands
If you prefer not to use the menu:

```powershell
py -3.11 main.py lora
py -3.11 main.py qlora
py -3.11 main.py qklora
py -3.11 main.py all
py -3.11 test_code.py
```

## Dataset Split Used
Training now uses the official Hugging Face PlantVillage `color` split only.

- official HF `color/train`
- official HF `color/test`
- then HF `color/train` is split into `train/val`

Validation is created safely by `leaf_id`, not by raw image, to reduce leakage.

Approximate overall ratio:
- `68%` train
- `12%` val
- `20%` test

## Image Processing
For training images:
1. Resize to `256x256`
2. Random crop to `224x224`
3. Random horizontal flip
4. Random rotation up to `15` degrees
5. Color jitter for brightness, contrast, and saturation
6. Normalize with ImageNet statistics

For validation and test images:
1. Resize to `256×256` (preserve aspect ratio)
2. Center crop to `224×224`
3. Normalize with the same ImageNet statistics

## QLoRA retrain note
If your `qlora_*.pth` checkpoints were created before the INT8 duplicate-weight fix, retrain QLoRA (`py -3.11 main.py qlora`) to get the correct ~8 MB checkpoints.

## Outputs
After training, check:
- `experiments/results/checkpoints/`
- `config/class_labels.json`
- `experiments/results/plots/`
- `experiments/results/logs/`
- `experiments/results/experiment_results.csv`

## If Something Fails
- use `py -3.11`, not another Python version
- run `py -3.11 download_assets.py` if assets are missing
- check `SETUP_GUIDE.md` for the exact environment rules
