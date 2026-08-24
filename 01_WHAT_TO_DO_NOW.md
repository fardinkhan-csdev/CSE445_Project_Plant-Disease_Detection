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

Start the V3 launcher (current recommended track):

```powershell
py -3.11 launcher_v3.py
```

V3 launcher menu:
1. Train `LoRA V3`
2. Train `QLoRA V3` (bitsandbytes NF4)
3. Train `QA-LoRA V3` (group-wise quant)
4. Train all three
5. Run tests
6. Exit

## Direct Commands

If you prefer not to use the menu:

```powershell
# V3 (current)
py -3.11 main_v3.py lora
py -3.11 main_v3.py qlora
py -3.11 main_v3.py qalora
py -3.11 main_v3.py all

# V3 evaluation
py -3.11 launcher_test_v3.py

# V1 (legacy, still works)
py -3.11 main.py lora
py -3.11 main.py qlora
py -3.11 main.py qklora
py -3.11 main.py all
```

## Dataset Split

Training uses the official Hugging Face PlantVillage `color` split only.

- official HF `color/train`
- official HF `color/test`
- then HF `color/train` is split into `train/val` by `leaf_id`

Approximate overall ratio:
- `68%` train
- `12%` val
- `20%` test

## Image Processing

**Training images:**
1. Resize to `256x256`
2. Random crop to `224x224`
3. Random horizontal flip
4. Random rotation up to `15` degrees
5. Color jitter (brightness, contrast, saturation)
6. Normalize with ImageNet statistics

**Validation/test images:**
1. Resize to `256x256` (preserve aspect ratio)
2. Center crop to `224x224`
3. Normalize with ImageNet statistics

## Current Results (V3)

| Method | Accuracy | F1 Macro | Trainable Params | Checkpoint |
|--------|----------|----------|------------------|------------|
| QA-LoRA | **99.52%** | 0.993 | 242k | 9.5 MB |
| Q/K LoRA | 99.22% | 0.987 | 445k | 12.1 MB |
| QLoRA | 98.91% | 0.984 | 193k | 7.6 MB |
| LoRA | 98.89% | 0.984 | 193k | 18.1 MB |

## Outputs

After training, check:
- `experiments/results/checkpoints/` — model checkpoints
- `config/class_labels.json` — class index map
- `experiments/results/plots/` — training curves, confusion matrices
- `experiments/results/logs/` — training logs
- `experiments/results/experiment_results.csv` — summary metrics

## Web UI

```powershell
py -3.11 run_web_ui.py
```

Opens at `http://localhost:8000` with model selection, image upload, and inference.

## If Something Fails

- Use `py -3.11`, not another Python version
- Run `py -3.11 download_assets.py` if assets are missing
- Check `SETUP_GUIDE.md` for the exact environment rules
