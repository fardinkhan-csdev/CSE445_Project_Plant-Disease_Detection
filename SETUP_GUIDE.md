# Leaf Disease Classification — Setup Guide

## Non-Negotiable: Python 3.11

- Exact executable: `C:\Users\Fardin Khan\AppData\Local\Programs\Python\Python311\python.exe`
- Short command: `py -3.11`
- NEVER use Python 3.13 or any other version

## Project Root

`d:\Leaf Disease Classification`

## Already Installed in 3.11

- PyTorch 2.x + CUDA (verify: `py -3.11 -c "import torch; print(torch.__version__)\"`)
- TorchVision, TorchAudio
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Hugging Face Datasets, PEFT, Accelerate
- bitsandbytes (for QLoRA V3 NF4 quantization)
- PyYAML, tqdm, Pillow

> CNN INT8 quantization is in `models/peft/int8_utils.py` (custom, not bitsandbytes).
> QLoRA V3 uses bitsandbytes for NF4 quantization specifically.

## Asset Preparation Rule

Training should not download anything on the fly. Before using any launcher, run:

```powershell
py -3.11 download_assets.py
```

That prepares:
- Local PlantVillage `color` images
- Official HF `color` split metadata
- Cached EfficientNet-B0 pretrained weights

## Current Training Split

- Use official HF `color/train`
- Use official HF `color/test`
- Split HF `color/train` into train/val by `leaf_id`
- Approximate ratio: `68% train / 12% val / 20% test`
- Grayscale and segmented variants are not used

## How to Run

Always run from the project root directory.

### V3 Track (Current Recommended)

```powershell
py -3.11 download_assets.py           # one-time setup
py -3.11 launcher_v3.py               # interactive launcher
py -3.11 main_v3.py lora              # train LoRA V3
py -3.11 main_v3.py qlora             # train QLoRA V3 (NF4)
py -3.11 main_v3.py qalora            # train QA-LoRA V3
py -3.11 main_v3.py all               # train all three
py -3.11 launcher_test_v3.py          # evaluate checkpoints
py -3.11 generate_dashboard.py        # generate HTML dashboard
py -3.11 run_web_ui.py                # launch web UI
```

### V1 Track (Legacy)

```powershell
py -3.11 launcher.py                  # V1 interactive launcher
py -3.11 main.py lora                 # train LoRA
py -3.11 main.py qlora                # train QLoRA
py -3.11 main.py qklora               # train Q/K LoRA
py -3.11 launcher_test.py             # V1 evaluation
```

### Utilities

```powershell
py -3.11 test_code.py                 # run tests
py -3.11 simple_test.py               # simple test
py -3.11 test_peft_quant.py           # PEFT quantization test
py -3.11 download_plantdoc.py         # download PlantDoc dataset
py -3.11 rank_experiments.py          # cross-method ranking
```

## Project Goal

Comparative study of LoRA, QLoRA, QA-LoRA, and Q/K LoRA on PlantVillage dataset with EfficientNet-B0 backbone, with PlantDoc transfer evaluation.

## Inference Artifacts

- Checkpoints: `experiments/results/checkpoints/<method>_best.pth`
- Class map: `config/class_labels.json` (created when `get_data_loaders()` runs)
- Cross-method ranking: `experiments/results/eval/cross_method_ranking.csv`

---

*Last updated: 2026-08-24*
