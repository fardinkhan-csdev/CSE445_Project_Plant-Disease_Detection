# Leaf Disease Classification Project Setup Guide
## NON-NEGOTIABLE: Always use Python 3.11
- Exact executable: `C:\Users\Fardin Khan\AppData\Local\Programs\Python\Python311\python.exe`
- Short command: `py -3.11`
- NEVER use Python 3.13 or any other version

## Project Root
- `d:\Leaf Disease Classification`

## Already Installed in 3.11
- PyTorch 2.x + CUDA (see `py -3.11 -c "import torch; print(torch.__version__)"`)
- TorchVision, TorchAudio
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Hugging Face Datasets, PEFT, Accelerate
- PyYAML, tqdm, Pillow

> CNN INT8 quantization is implemented in `models/peft/int8_utils.py` (not `bitsandbytes`).

## Asset Preparation Rule
- Training should not download anything on the fly
- Before using `launcher.py`, run:
  - `py -3.11 download_assets.py`
- That prepares:
  - local PlantVillage `color` images
  - official HF `color` split metadata
  - cached EfficientNet-B0 pretrained weights

## Current Training Split
- Use official HF `color/train`
- Use official HF `color/test`
- Split HF `color/train` into train/val by `leaf_id`
- Approximate overall ratio: `68% train / 12% val / 20% test`
- Grayscale and segmented variants are not used by the training pipeline

## How to Run Anything
- Always run from project root
- `py -3.11 download_assets.py`
- `py -3.11 launcher.py`
- `py -3.11 test_code.py`
- `py -3.11 main.py lora`
- `py -3.11 main.py qlora`
- `py -3.11 main.py qklora`
- `py -3.11 main.py all`

## Project Goal
Comparative study of LoRA, QLoRA, Q/K LoRA on PlantVillage dataset with EfficientNet-B0

## Inference artifacts
- Checkpoints: `experiments/results/checkpoints/<method>_best.pth`
- Class map: `config/class_labels.json` (created when `get_data_loaders()` runs)

---
*Last updated: 2026-07-20*
