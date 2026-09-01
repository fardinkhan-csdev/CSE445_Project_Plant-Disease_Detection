# Leaf Disease Classification — PEFT Comparative Study

A comparative study of parameter-efficient fine-tuning (PEFT) methods for plant leaf disease classification using EfficientNet-B0 as the backbone.

## Methods Compared

| Method | Quantization | Trainable Params | Checkpoint Size | Test Accuracy |
|--------|-------------|------------------|-----------------|---------------|
| **QA-LoRA** | Group-wise INT4 [-8,7] + grouped LoRA | 242k | 9.5 MB | **99.52%** |
| **Q/K LoRA** | INT8 Q-path + FP32 SE (K-path) | 445k | 12.1 MB | 99.22% |
| **QLoRA** | bitsandbytes NF4 on Q-path | 193k | 7.6 MB | 98.91% |
| **LoRA** | None (FP32 backbone) | 193k | 18.1 MB | 98.89% |

## Quick Start

```bash
# 1. Install dependencies (Python 3.11 required)
py -3.11 -m pip install -r requirements.txt

# 2. Prepare assets (one-time download)
py -3.11 download_assets.py

# 3a. V3 training (current — LoRA + QLoRA + QA-LoRA)
py -3.11 launcher_v3.py

# 3b. Or train directly
py -3.11 main_v3.py all        # all three V3 methods
py -3.11 main_v3.py lora       # LoRA V3 only
py -3.11 main_v3.py qlora      # QLoRA V3 (bitsandbytes NF4)
py -3.11 main_v3.py qalora     # QA-LoRA V3 (group-wise quant)

# 4. Evaluate V3 checkpoints
py -3.11 launcher_test_v3.py

# 5. Generate dashboard
py -3.11 generate_dashboard.py

# 6. Launch web UI
py -3.11 run_web_ui.py
```

### Legacy V1 Track (still runnable)

```bash
py -3.11 launcher.py           # V1 launcher (LoRA, QLoRA, Q/K LoRA)
py -3.11 main.py lora          # V1 LoRA
py -3.11 main.py qlora         # V1 QLoRA
py -3.11 main.py qklora        # V1 Q/K LoRA
```

## Project Structure

```
.
├── config/                          # Configuration files
│   ├── base_config.yaml             # V1 base hyperparameters
│   ├── base_config_v3.yaml          # V3 base hyperparameters
│   ├── lora_config.yaml             # LoRA config (shared V1/V3)
│   ├── qlora_config.yaml            # QLoRA V3 config (NF4)
│   ├── qalora_config.yaml           # QA-LoRA V3 config
│   ├── qklora_config.yaml           # Q/K LoRA config (V1)
│   └── class_labels.json            # Auto-exported class index map
├── data/
│   ├── raw/                         # Raw PlantVillage dataset
│   ├── processed/                   # Split manifests (train.txt/val.txt/test.txt)
│   └── data_loader.py               # Data loading and transformations
├── models/
│   ├── backbone/
│   │   └── efficientnet_b0.py       # EfficientNet-B0 backbone
│   ├── peft/
│   │   ├── lora.py                  # V1 LoRA
│   │   ├── lora_v3.py               # V3 LoRA (with merge support)
│   │   ├── qlora.py                 # V1 QLoRA (INT8)
│   │   ├── qlora_v3.py              # V3 QLoRA (bitsandbytes NF4)
│   │   ├── qalora.py                # QA-LoRA (group-wise quant + grouped LoRA)
│   │   ├── qklora.py                # Q/K LoRA (V1)
│   │   ├── int8_utils.py            # INT8 quantization helpers
│   │   ├── mixstyle.py              # MixStyle domain adaptation
│   │   ├── fake_quant.py            # Fake quantization utilities
│   │   └── glcm_branch.py           # GLCM texture feature branch
│   └── classifier.py                # Final classifier head
├── training/
│   ├── trainer.py                   # Base trainer (shared V1/V3)
│   ├── lora_trainer.py              # V1 LoRA trainer
│   ├── lora_trainer_v3.py           # V3 LoRA trainer
│   ├── qlora_trainer.py             # V1 QLoRA trainer
│   ├── qlora_trainer_v3.py          # V3 QLoRA trainer (NF4)
│   ├── qalora_trainer.py            # QA-LoRA trainer
│   └── qklora_trainer.py            # Q/K LoRA trainer (V1)
├── evaluation/
│   ├── metrics.py                   # Metrics calculation
│   ├── confusion_matrix.py          # Confusion matrix visualization
│   └── evaluator.py                 # Evaluator class
├── experiments/
│   └── results/                     # All experiment outputs
│       ├── checkpoints/             # Model checkpoints
│       ├── eval/                    # Per-method ranking CSVs
│       ├── plots/                   # Training curves, confusion matrices
│       ├── logs/                    # Training logs
│       └── dashboard.html           # Self-contained HTML dashboard
├── web_app/                         # Web UI for inference
│   ├── server.py                    # HTTP server with API endpoints
│   ├── inference.py                 # Model loading and prediction engine
│   └── static/                      # HTML/CSS/JS frontend
├── utils/
│   ├── logger.py                    # Logging setup
│   ├── visualization.py             # Training curves, class metrics
│   └── memory_tracker.py            # GPU memory tracking
├── _archive/
│   ├── stale/                       # Outdated docs, research reports
│   └── old_useful/                  # Legacy V1/V2 scripts
├── main_v3.py                       # V3 entry point (recommended)
├── main.py                          # V1 entry point (legacy)
├── launcher_v3.py                   # V3 interactive launcher
├── launcher.py                      # V1 interactive launcher
├── launcher_test_v3.py              # V3 evaluation launcher
├── launcher_test.py                 # V1 evaluation launcher
├── rank_experiments.py              # Cross-method ranking script
├── generate_dashboard.py            # HTML dashboard generator
├── download_assets.py               # One-time asset preparation
├── download_plantdoc.py             # PlantDoc dataset download
├── plantdoc_mapping.py              # PlantVillage ↔ PlantDoc label mapping
├── run_plantdoc_evaluation.py       # PlantDoc evaluation runner
├── eval_all_v3.py                   # V3 batch evaluation
├── eval_plantdoc.py                 # PlantDoc evaluation
├── run_web_ui.py                    # Web UI launcher
├── requirements.txt                 # Python dependencies
└── architecture_design_v3.md        # Current architecture document
```

## Dataset

- **Training**: PlantVillage dataset (54,306 RGB images, 38 classes)
- **Source**: Official Hugging Face `color` split only
- **Split**: 68% train / 12% val / 20% test (leaf-id grouped to prevent leakage)
- **Transfer evaluation**: PlantDoc dataset (real-world field images)

## V3 Methods

| Method | Paper Reference | Quantization | Key Innovation |
|--------|----------------|-------------|----------------|
| **LoRA V3** | Hu et al. 2021 | None | PEFT LoRA on pointwise convs + merge support |
| **QLoRA V3** | Dettmers et al. 2023 | bitsandbytes NF4 | Real 4-bit NF4 quantization via bitsandbytes |
| **QA-LoRA V3** | Xu et al. 2024 | Group-wise INT4 [-8,7] | Algorithm 1: group-wise quant + grouped LoRA A |

## Web UI

Launch with `py -3.11 run_web_ui.py` → opens at `http://localhost:8000`

Features:
- Model selection (LoRA, QLoRA, QA-LoRA, QKLoRA)
- Image upload and real-time classification
- Experiment results and checkpoint rankings
- PlantDoc transfer evaluation results
- Training curves and confusion matrices

## Notes

- Requires **Python 3.11** — do not use other versions
- Designed for RTX 5060 with 8GB VRAM
- Early stopping supported via `training.early_stopping_patience`
- `download_assets.py` must be run once before training
- V1 and V3 tracks coexist; V3 is the current recommended track
