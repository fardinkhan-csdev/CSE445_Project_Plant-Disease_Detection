# Leaf Disease Classification - LoRA vs QLoRA vs Q/K LoRA

A comparative study of parameter-efficient fine-tuning (PEFT) methods for plant leaf disease classification using EfficientNet-B0 as the backbone.

## Project Structure

```
leaf-disease-classification/
├── config/                     # Configuration files
│   ├── base_config.yaml        # Base hyperparameters
│   ├── lora_config.yaml        # LoRA-specific config
│   ├── qlora_config.yaml       # QLoRA-specific config
│   ├── qklora_config.yaml      # Q/K LoRA-specific config
│   └── class_labels.json       # Auto-exported class index map (inference)
├── data/                       # Dataset-related
│   ├── raw/                    # Raw PlantVillage dataset
│   ├── processed/              # Split manifests only (train.txt/val.txt/test.txt)
│   └── data_loader.py          # Data loading and transformations
├── models/                     # Model definitions
│   ├── backbone/               # Backbone model (EfficientNet-B0)
│   │   └── efficientnet_b0.py
│   ├── peft/                   # Parameter-efficient fine-tuning modules
│   │   ├── lora.py
│   │   ├── qlora.py
│   │   ├── qklora.py
│   │   └── int8_utils.py       # INT8 weight-only quantization helpers
│   └── classifier.py           # Final classifier head
├── training/                   # Training pipeline
│   ├── trainer.py              # Base trainer class
│   ├── lora_trainer.py         # LoRA-specific trainer
│   ├── qlora_trainer.py        # QLoRA-specific trainer
│   └── qklora_trainer.py       # Q/K LoRA-specific trainer
├── evaluation/                 # Evaluation pipeline
│   ├── metrics.py              # Metrics calculation
│   ├── confusion_matrix.py     # Confusion matrix visualization
│   └── evaluator.py            # Evaluator class
├── experiments/                # Experiment management
│   ├── experiment_runner.py    # Run experiments
│   └── results/                # Experiment results
├── utils/                      # Utility functions
│   ├── logger.py               # Logging setup
│   ├── visualization.py        # Visualization tools
│   └── memory_tracker.py       # GPU memory tracking
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
└── architecture_design.md      # Architecture design document
```

## Setup Instructions

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   py -3.11 -m pip install -r requirements.txt
   ```
3. **Prepare all local assets once**:
   ```bash
   py -3.11 download_assets.py
   ```
4. **Start the training launcher**:
   ```bash
   py -3.11 launcher.py
   ```
6. **Start the test launcher (evaluate checkpoints)**:
   ```bash
   py -3.11 launcher_test.py
   ```
7. **Run PlantDoc evaluation with the new mapping workflow**:
   ```bash
   py -3.11 launcher_plantdoc.py
   ```
5. **Or run experiments directly**:
   ```bash
   py -3.11 main.py lora
   py -3.11 main.py qlora
   py -3.11 main.py qklora
   py -3.11 main.py all
   ```

## PEFT Methods

| Method | Quantization | Trainable params | Checkpoint size |
|--------|--------------|------------------|-----------------|
| LoRA | None (FP32) | ~343k | ~18 MB |
| QLoRA | INT8 on Q-path MBConv 1×1 | ~193k | ~8 MB |
| Q/K LoRA | INT8 Q-path + FP32 SE (K-path) | ~445k | ~9 MB |

See `architecture_design.md` §8–§9 for full design. Ship `config/class_labels.json` with checkpoints for inference.

## Notes

- The training pipeline is locked to the official Hugging Face PlantVillage `color` split only.
- The official HF `color/test` split is preserved, and validation is created from HF `color/train` using `leaf_id` grouping to reduce leakage.
- Approximate effective ratio is `68% train / 12% val / 20% test`.
- `launcher.py` is training/test only. It does not download assets during execution.
- `download_assets.py` is the one-time preparation step that caches the color images, HF split metadata, and EfficientNet-B0 pretrained weights.
- `data/processed/` is intentionally used only for split manifest files such as `train.txt`, `val.txt`, and `test.txt`; it is not a cache of preprocessed images or tensors.
- The project is designed to run on an RTX 5060 with 8GB VRAM.
- Early stopping is supported via `training.early_stopping_patience` and will stop training when validation accuracy has not improved for the configured number of epochs.

## Results

Experiment results will be saved in `experiments/results/`, including:
- Training logs
- Model checkpoints (best, last, and optionally per-epoch if enabled in `config/base_config.yaml`)
- Training curves
- Confusion matrices
- Class-wise metrics
- Summary CSV file
