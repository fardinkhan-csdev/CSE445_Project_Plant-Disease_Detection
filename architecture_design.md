# Leaf Disease Classification - Architecture Design Document

## Project Overview
**Project Title**: Comparative Study of LoRA, QLoRA, and Q/K LoRA for EfficientNet-B0-Based Plant Leaf Disease Classification

**Goal**: Compare three parameter-efficient fine-tuning approaches for plant leaf disease classification using EfficientNet-B0 as the backbone model.

---

## 1. Folder Structure
```
leaf-disease-classification/
├── config/                     # Configuration files
│   ├── base_config.yaml        # Base hyperparameters
│   ├── class_labels.json       # Auto-exported idx_to_class / class_to_idx (38 classes)
│   ├── lora_config.yaml        # LoRA-specific config
│   ├── qlora_config.yaml       # QLoRA-specific config
│   ├── qklora_config.yaml      # Q/K LoRA-specific config
│   └── class_labels.json       # Auto-exported idx↔class map for inference
├── data/                       # Dataset-related
│   ├── raw/                    # Raw PlantVillage dataset
│   ├── processed/              # Train/val/test splits
│   └── data_loader.py          # Data loading and transformations
├── models/                     # Model definitions
│   ├── backbone/               # Backbone model (EfficientNet-B0)
│   │   └── efficientnet_b0.py
│   ├── peft/                   # Parameter-efficient fine-tuning modules
│   │   ├── lora.py
│   │   ├── qlora.py
│   │   ├── qklora.py
│   │   └── int8_utils.py       # Weight-only INT8 helpers (QLoRA/QKLoRA)
│   └── classifier.py           # Final classifier head
├── training/                   # Training pipeline
│   ├── trainer.py              # Base trainer class
│   ├── lora_trainer.py         # LoRA-specific trainer
│   ├── qlora_trainer.py        # QLoRA-specific trainer
│   └── qklora_trainer.py       # Q/K LoRA-specific trainer
├── evaluation/                 # Evaluation pipeline
│   ├── metrics.py              # Metrics calculation (accuracy, precision, recall, F1)
│   ├── confusion_matrix.py     # Confusion matrix visualization
│   └── evaluator.py            # Evaluator class
├── experiments/                # Experiment management
│   ├── experiment_runner.py    # Run experiments with different configs
│   └── results/                # Experiment results
│       ├── logs/               # Training logs
│       ├── checkpoints/        # Model checkpoints
│       └── plots/              # Training curves, confusion matrices
├── utils/                      # Utility functions
│   ├── logger.py               # Logging setup
│   ├── visualization.py        # Visualization tools
│   └── memory_tracker.py       # GPU memory tracking
├── requirements.txt            # Python dependencies
└── main.py                     # Entry point
```

---

## 2. Data Pipeline

### 2.1 Dataset
- **Source**: PlantVillage Dataset (~54,000 images)
- **Input**: RGB images (224x224 pixels)
- **Output**: Crop type, Disease type, Confidence score

### 2.2 Data Split

The pipeline uses the official Hugging Face **PlantVillage** dataset splits:

* **Training/Test:** Official `train` and `test` splits (approximately **80% / 20%**).
* **Validation:** A portion of the official `train` split is reserved for validation. This fraction is controlled by `val_split_from_train` in `config/base_config.yaml` (default: `0.15`).

With the default configuration:

* **Training:** 68% of the total dataset (85% of the official training split)
* **Validation:** 12% of the total dataset (15% of the official training split)
* **Test:** 20% of the total dataset (the official test split)

Thus, the effective dataset split is **68% training, 12% validation, and 20% testing**. This design preserves the official test set while providing a configurable validation set for model development.

**In other words:**

**A (100% dataset)** → **B (80% train + 20% test)** → **C (68% train + 12% validation + 20% test)**.

**Final effective split (default): 68% Train / 12% Validation / 20% Test.**

### 2.3 Data Transformations
#### Training
- Resize to 256x256
- Random crop to 224x224
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- Normalize (ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### Validation/Testing
- Resize to 256×256 (preserving aspect ratio)
- Center crop to 224×224
- Normalize (same ImageNet stats as training)

> **Why Resize(256) + CenterCrop(224)?** This is the standard ImageNet evaluation protocol. Resizing to a slightly larger size before cropping preserves the subject at the center, avoids distortion from direct resize-to-target, and ensures the val/test distribution closely matches what the model sees during training (which uses random crop from 256).

### 2.4 Data Loading
- Use PyTorch `Dataset` and `DataLoader`
- Batch size: 32 (adjust based on GPU memory)
- Number of workers: 4 (adjust based on CPU cores)

### 2.5 LoRA Insertion Points

**LoRA (baseline)** — broad adaptation:
- Apply LoRA to **all non-depthwise `Conv2d` layers** (`groups == 1`), including stem/head convolutions and MBConv pointwise layers, plus `classifier.fc`
- **Depthwise convolutions** (`groups > 1`) are excluded — the `peft` library cannot attach LoRA adapters to grouped convolutions

**QLoRA / Q/K LoRA (experimental)** — selective adaptation:
- **QLoRA**: INT8 weight-only quantization on MBConv expand/project 1×1 convs (Q-path) + LoRA on Q-path + `classifier.fc`
- **Q/K LoRA**: INT8 on Q-path, FP32 on SE layers (K-path), tiered LoRA ranks (`q_rank=16`, `k_rank=4`)
- Stem (`features.0.0`), depthwise convs, and SE layers (QLoRA only) stay FP32/frozen without LoRA where noted
- Method-specific settings live in `config/qlora_config.yaml` and `config/qklora_config.yaml`

### 2.7 Class Label Map
- `config/class_labels.json` is written automatically when `get_data_loaders()` runs
- Contains `num_classes`, `idx_to_class`, and `class_to_idx` for inference/deployment
- Checkpoints store weights only; always ship this JSON alongside model files for production

### 2.6 Class-Imbalance Notes
- Problem: If some classes have many more examples than others, a model can get high overall accuracy by predicting the dominant classes and still perform poorly on rare classes.
- Two simple mitigation strategies (both **off by default** in `config/base_config.yaml`):
  1. **Class-weighted loss** (`data.use_class_weights: true`): compute per-class weights inversely proportional to class frequency and pass them to `nn.CrossEntropyLoss(weight=class_weights)`.
  2. **Weighted sampling** (`data.use_weighted_sampler: true`): use `torch.utils.data.WeightedRandomSampler` so that each minibatch contains a more balanced mix of classes.
- Enable **only one** of the two toggles at a time.
- **Prior Work**: A study ("Advancing Image Classification through Parameter-Efficient Fine-Tuning: A Study on LoRA with Plant Disease Detection Datasets") achieved **99.89% accuracy on PlantVillage** with LoRA using less than 1% trainable parameters!

### 3.4 CNN-QLoRA Adaptation (see §8)
- **Quantization**: Custom weight-only **INT8** on MBConv expand/project 1×1 convs (Q-path). INT4/NF4 was deferred as too difficult for this CNN pipeline.
- **LoRA**: Adapters on Q-path pointwise convs + `classifier.fc` only (narrower than baseline LoRA).
- **Implementation**: `models/peft/int8_utils.py` + `models/peft/qlora.py` + `peft` LoRA wrappers.

### 3.5 CNN-QKLoRA Adaptation (see §9)
- **Q path (Quantized)**: MBConv expand/project 1×1 convs → INT8 frozen weights + LoRA rank **16**
- **K path (Kept FP32)**: SE `fc1`/`fc2` 1×1 convs + `classifier.fc` → FP32 frozen weights + LoRA rank **4**
- **Depthwise convs**: FP32 and frozen; **no LoRA** (PEFT cannot attach adapters to grouped convolutions)
- **Note**: **Q** = Quantized path, **K** = Kept high-precision path — not transformer Query/Key projections.
- **Implementation**: `models/peft/int8_utils.py` + `models/peft/qklora.py` + `peft` with `rank_pattern`.

---

## 3. Model Architecture

### 3.1 Backbone (EfficientNet-B0)
- Pretrained on ImageNet via `torchvision.models.efficientnet_b0` (cached weights from `download_assets.py`)
- Use EfficientNet-B0 as the feature extractor. Optionally freeze early layers when using PEFT to reduce trainable parameters.

### 3.2 Classifier Head
- Global Average Pooling (built into EfficientNet-B0's `model.avgpool`, applied before this head)
- Dropout (p=0.2)
- Linear layer: `in_features → num_classes`
- **No Softmax in the head.** PyTorch's `CrossEntropyLoss` internally applies `log_softmax`, so adding an explicit `Softmax` here would corrupt the loss computation. For inference probabilities, `torch.softmax()` is applied externally in the evaluator.
- Implemented in: `models/classifier.py` → `PlantDiseaseClassifier`

## 4. Training Pipeline

### 4.1 Base Trainer
- **Components** (all driven by `config/base_config.yaml`):
  - Model initialization
  - Optimizer (`training.optimizer`, default: AdamW)
  - Learning rate scheduler (`training.scheduler`, default: CosineAnnealingLR)
  - Loss function (`training.loss_fn`, default: CrossEntropyLoss; optional class weights via `data.use_class_weights`)
  - Checkpoint saving
  - Logging
  - GPU memory tracking

### 4.2 Training Loop
1. Forward pass
2. Loss calculation
3. Backward pass
4. Optimizer step
5. Evaluate on the validation set every epoch
6. Save the best checkpoint if validation accuracy improves
7. Optionally stop early if validation accuracy fails to improve for `training.early_stopping_patience` consecutive epochs

### 4.3 Hyperparameters (Base)
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 1e-5
- Number of epochs: 20
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.1

---

## 5. Evaluation Pipeline

### 5.1 Metrics
- **Accuracy**: Overall correctness
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-score**: Per-class F1-score
- **Macro-averages**: For all metrics

Checkpoint behavior:
- The trainer saves `best` and `last` by default.
- Per-epoch checkpoints can be enabled via `logging.save_epoch_checkpoints: true` in `config/base_config.yaml`.
- Training also supports early stopping via `training.early_stopping_patience`, which halts training after the configured number of epochs without validation accuracy improvement.
- The `best` checkpoint is chosen by highest validation accuracy.
- The test launcher and experiment runner use the `best` checkpoint by default for final test evaluation.

Additional evaluation outputs implemented:
- Per-sample class probabilities (softmax) are saved as CSV files under `experiments/results/eval/` for evaluated checkpoints, with columns `image_path,true_label,pred_label,prob_0,...,prob_N`.
- Confusion matrix and class-wise metric plots are saved to `experiments/results/plots/`.

Notes:
- By default the experiment runner evaluates the `best` checkpoint after training (multiclass metrics + plots only).
- **Test evaluation is intentionally separate** (`launcher_test.py`) so GPU memory is freed after training before running full checkpoint ranking, confidence CSV export, and binary/crop-disease analysis.
- `launcher_test.py` supports an optional "evaluate all checkpoints" mode that ranks each checkpoint file and writes a summary CSV.

### 5.2 Visualization
- **Training Curves**: Loss vs. epoch, accuracy vs. epoch (train/val)
- **Confusion Matrix**: Heatmap of true vs. predicted labels
- **Class-wise Metrics**: Bar charts for precision/recall/F1

### 5.3 Efficiency Metrics
- **GPU Memory Usage**: Peak memory during training
- **Training Time**: Total time per experiment
- **Number of Trainable Parameters**: Count for each PEFT method

---

## 6. Experiment Pipeline

### 6.1 Experiment Runner
- Load configuration files (YAML)
- Initialize trainer for each method (LoRA, QLoRA, Q/K LoRA)
- Run training
- Evaluate on test set
- Save results (logs, checkpoints, plots)

### 6.2 Experiment Workflow
1. Run LoRA experiment
2. Run QLoRA experiment
3. Run Q/K LoRA experiment
4. Compare results and generate summary report

---

## 7. LoRA Insertion into EfficientNet-B0

EfficientNet-B0 consists of **MBConv blocks** (Mobile Inverted Bottleneck Convolution blocks). Each MBConv block has:
1. **Expand Conv**: 1x1 pointwise convolution (expand channels)
2. **Depthwise Conv**: 3x3 depthwise convolution (spatial features)
3. **Squeeze-and-Excitation (SE)**: Channel attention
4. **Project Conv**: 1x1 pointwise convolution (reduce channels)

**LoRA Insertion Points (baseline)**:
- Apply LoRA to all **non-depthwise `Conv2d` layers** (`groups == 1`) — includes stem, MBConv expand/project (1×1), and head convolutions — plus `classifier.fc`
- **Depthwise convolutions** (`groups > 1`) are always excluded (PEFT limitation)

**QLoRA / Q/K LoRA** use narrower, method-specific `target_modules` defined in their config files.

**Example**: For a pointwise convolution layer:
- Original weight shape: (out_channels, in_channels, 1, 1)
- LoRA A shape: (in_channels, r)
- LoRA B shape: (r, out_channels)
- Forward pass: output = conv2d(input, W0 + (B @ A).unsqueeze(2).unsqueeze(3))

---

## 8. INT8 CNN-QLoRA: QLoRA Adaptation for EfficientNet-B0

Since EfficientNet-B0 is a convolutional neural network (CNN), the original transformer-based QLoRA workflow cannot be applied directly. Therefore, this work adopts a CNN-oriented adaptation of QLoRA that preserves the core principle of parameter-efficient fine-tuning using a frozen weight-quantized backbone and trainable LoRA adapters.

**Steps**:

1. **Weight-only INT8 Quantization**
   - Quantize the weights of the selected EfficientNet-B0 backbone layers to INT8 using a CNN-compatible weight-only quantization method (INT4 OR NF4 WAS TOO HARD TO QUANTIZE).
   - Activation tensors remain in floating-point precision during training.

2. **Freeze Quantized Backbone**
   - Freeze all quantized backbone weights and disable gradient updates.

3. **Insert LoRA Adapters**
   - Insert LoRA adapters into MBConv expand/project 1×1 convs (Q-path) and `classifier.fc` — a selective subset of baseline LoRA targets (excludes stem, SE, and depthwise layers).

4. **Train Only LoRA Parameters**
   - Train only the LoRA A and B matrices while keeping the INT8-quantized backbone fixed.

**Core Principle**
- Weight-only INT8 quantized backbone
- Frozen backbone weights
- Trainable low-rank adapters
- Parameter-efficient fine-tuning

**Key Libraries**
- PyTorch (`F.conv2d` + custom INT8 dequant in `models/peft/int8_utils.py`)
- PEFT (Conv2d LoRA adapters)
- Accelerate (training)

**Difference from Original QLoRA**

Original QLoRA applies 4-bit NF4 weight quantization to transformer Linear layers. In contrast, the proposed CNN-QLoRA applies weight-only INT8 quantization to EfficientNet-B0 convolutional layers while preserving the same parameter-efficient fine-tuning principle of a frozen quantized backbone with trainable LoRA adapters.

---

## 9. INT8 CNN-QKLoRA: Q/K-LoRA Adaptation for EfficientNet-B0

**Key Insight**: In CNNs, different layer types contribute differently to feature extraction and parameter complexity. Therefore, this work selectively applies quantization and LoRA adaptation based on the role of each layer.

### Layer Classification

**Q Layers (Quantized Path)**
- Pointwise convolutions (1×1)
- Responsible for channel projection and channel mixing
- Weight-only INT8 quantization
- Higher-rank LoRA adapters

**K Layers (Kept High-Precision Path)**
- Squeeze-and-Excitation (SE) layers (`fc1`/`fc2` 1×1 convs in torchvision EfficientNet)
- Remain in FP32
- Lower-rank LoRA adapters
- Depthwise convolutions stay FP32 and frozen (no LoRA — PEFT limitation)

### Implementation Steps

1. Identify each EfficientNet-B0 layer as either a Q layer or K layer.
2. Apply weight-only INT8 quantization to all Q layers.
3. Keep all K layers in full precision.
4. Insert LoRA adapters into both Q and K layers using different adaptation ranks.
5. Freeze the backbone weights and train only the LoRA parameters.

### Suggested LoRA Configuration

| Layer Type | Precision | LoRA Rank | LoRA Adapters |
|------------|-----------|-----------|---------------|
| Pointwise Conv (Q) | INT8 | r = 16 | Yes |
| SE Layers (K) | FP32 | r = 4 | Yes |
| Depthwise Conv | FP32 | — | No (PEFT limitation) |
| Classifier `fc` (K) | FP32 | r = 4 | Yes |

### Justification

- Pointwise convolutions contain the majority of EfficientNet-B0 parameters because they perform channel projection between feature maps.
- Depthwise convolutions contain comparatively few parameters and are responsible for extracting spatial features.
- Quantizing only the pointwise convolutions provides most of the memory reduction while preserving important spatial information in the depthwise and SE layers.
- Assigning a higher LoRA rank to quantized layers compensates for the representational capacity lost due to quantization, while lower-rank adapters are sufficient for the high-precision layers.

**Note**: In this work, **Q** denotes the **Quantized Path**, while **K** denotes the **Kept High-Precision Path**, rather than the Query and Key projections used in transformer architectures.

---

## 10. Risk Analysis

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| QLoRA/Q/K LoRA may have lower accuracy than LoRA | High | Medium | Tune LoRA ranks (`q_rank`/`k_rank`), compare INT8 vs FP32 paths |
| GPU memory may still be insufficient | Medium | Medium | Use gradient checkpointing, mixed precision (FP16), smaller batch size |
| PlantVillage dataset may be imbalanced | Medium | Medium | Use class weights in loss function, oversample minority classes |
| Custom Q/K LoRA implementation may have bugs | High | High | Unit test each component, compare with LoRA/QLoRA baselines |
| Training time may exceed 1 month | Medium | Low | Use mixed precision, limit number of epochs, monitor progress closely |

---

## 11. Milestone Plan (1 Month Duration)

| Week | Milestone |
|------|-----------|
| Week 1 | Setup project structure, implement data pipeline, test EfficientNet-B0 backbone |
| Week 2 | Implement LoRA adaptation, run LoRA experiments |
| Week 3 | Implement QLoRA adaptation, run QLoRA experiments; start Q/K LoRA design |
| Week 4 | Implement and test Q/K LoRA, run all experiments, generate final report |

---

## 12. Recommended Implementation Order
1. **Data Pipeline**: Implement `data_loader.py` with transformations and splits
2. **Backbone Model**: Load EfficientNet-B0 and test feature extraction
3. **Base Trainer**: Implement `trainer.py` with basic training loop
4. **LoRA**: Implement LoRA adapters and `lora_trainer.py`
5. **Evaluation**: Implement `metrics.py`, `confusion_matrix.py`, and `evaluator.py`
6. **QLoRA**: Add INT8 weight-only quantization (`int8_utils.py`) and `qlora_trainer.py`
7. **Q/K LoRA**: Add selective Q/K path logic and `qklora_trainer.py`
8. **Experiment Runner**: Implement `experiment_runner.py` to run all experiments
9. **Visualization**: Add training curves and result visualization


---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| `data/data_loader.py` | ✅ Complete | HF PlantVillage, 68/12/20 split, augmentation pipeline, exports `config/class_labels.json` |
| `models/backbone/efficientnet_b0.py` | ✅ Complete | Frozen EfficientNet-B0 backbone via `torchvision` |
| `models/peft/lora.py` | ✅ Complete | `peft` LoRA on all non-depthwise convs (`groups==1`) + classifier head |
| `models/peft/qlora.py` | ✅ Complete | INT8 weight-only CNN-QLoRA on MBConv Q-path + LoRA (~8 MB checkpoint) |
| `models/peft/qklora.py` | ✅ Complete | Selective INT8 Q-path + FP32 K-path with tiered LoRA ranks (~9 MB checkpoint) |
| `models/peft/int8_utils.py` | ✅ Complete | Per-channel INT8 quant, dequant forward, duplicate-weight guard |
| `training/trainer.py` | ✅ Complete | Base trainer, early stopping, checkpoint save/load, GPU memory tracking, timing |
| `training/lora_trainer.py` | ✅ Complete | LoRA training loop (20 epochs, cosine LR schedule) |
| `training/qlora_trainer.py` | ✅ Complete | CNN-QLoRA training loop |
| `training/qklora_trainer.py` | ✅ Complete | CNN-QKLoRA training loop |
| `evaluation/metrics.py` | ✅ Complete | Accuracy, Precision, Recall, F1 (macro), Binary Accuracy/F1, ROC AUC, Crop/Disease correctness |
| `evaluation/evaluator.py` | ✅ Complete | Full inference loop, returns `y_probs` for AUC computation |
| `evaluation/confusion_matrix.py` | ✅ Complete | Seaborn-based confusion matrix plot |
| `experiments/experiment_runner.py` | ✅ Complete | Runs all experiments sequentially, saves results to CSV |
| `launcher_test.py` | ✅ Complete | Lightweight test launcher; calls evaluator for each checkpoint, writes ranking CSVs |
| `generate_dashboard.py` | ✅ Complete | Generates self-contained HTML dashboard with all data embedded |
| `utils/logger.py` | ✅ Complete | Per-experiment file logger |
| `utils/visualization.py` | ✅ Complete | Training curves, class-metrics bar charts |
| `utils/memory_tracker.py` | ✅ Complete | `torch.cuda` peak memory tracking |

---

## 10. Binary Evaluation Pipeline

### 10.1 Purpose
Every saved checkpoint is evaluated in two modes simultaneously:
1. **Multiclass** (primary): 38-class classification as trained
2. **Binary**: multiclass predictions collapsed to `healthy` vs `diseased`

### 10.2 Binary Mapping Rule
```python
predicted_binary = "healthy" if "healthy" in predicted_class.lower() else "diseased"
```
The same mapping is applied to ground-truth labels to form the binary target.

### 10.3 Checkpoint Evaluation Flow
```
launcher_test.py  (lightweight — only orchestrates)
    └─> get_data_loaders()                    # data/data_loader.py
    └─> for each .pth in checkpoints/:
            Trainer.load_checkpoint()          # training/trainer.py
            Evaluator.evaluate()               # evaluation/evaluator.py
                └─> returns y_true, y_pred, y_probs, class_names
            metrics.calculate_metrics(...)     # evaluation/metrics.py
                └─> multiclass: accuracy, precision, recall, F1 (macro)
                └─> binary mapping → binary accuracy, F1
                └─> ROC AUC via sklearn OvR with y_probs
                └─> crop/disease correctness breakdown
    └─> writes eval/<experiment>_checkpoint_ranking.csv
    └─> writes experiments/results/experiment_results.csv
```

> **Design note**: `launcher_test.py` is intentionally separate from training to reduce GPU load. All metric computation logic lives in `evaluation/metrics.py` and `evaluation/evaluator.py`. The launcher only calls these functions and persists the results.

### 10.4 Output: `eval/<experiment>_checkpoint_ranking.csv`
| Column | Description |
|---|---|
| `rank` | Numeric rank by test accuracy (1 = best) |
| `checkpoint` | Full path to the `.pth` file |
| `size_mb` | Checkpoint file size in megabytes |
| `accuracy` | 38-class test accuracy |
| `f1_macro` | Macro-averaged F1 across 38 classes |
| `binary_accuracy` | Accuracy after healthy/diseased collapse |
| `binary_f1` | F1 of the binary classification |
| `binary_roc_auc` | ROC AUC (OvR) using predicted class probabilities |
| `both_correct_pct` | % samples where crop AND disease are both correct |
| `name_only_correct_pct` | % samples where only crop name is correct |
| `disease_only_correct_pct` | % samples where only disease is correct |
| `none_correct_pct` | % samples where neither is correct |
| `confidences_csv` | Path to per-sample confidence scores CSV |

---

## 11. Crop / Disease Correctness Analysis

### 11.1 Purpose
PlantVillage class names follow the pattern `Crop___Disease` (e.g. `Tomato___Late_blight`). This allows the model's 38-class prediction to be decomposed into two independent sub-tasks and evaluated separately.

### 11.2 Parsing
```python
crop_pred,  disease_pred = predicted_class.split("___")
crop_true,  disease_true = ground_truth_class.split("___")
```

### 11.3 Correctness Categories (per sample)
| Category | Condition |
|---|---|
| `both_correct` | `crop_pred == crop_true` AND `disease_pred == disease_true` |
| `name_only_correct` | `crop_pred == crop_true` AND `disease_pred != disease_true` |
| `disease_only_correct` | `crop_pred != crop_true` AND `disease_pred == disease_true` |
| `none_correct` | `crop_pred != crop_true` AND `disease_pred != disease_true` |

### 11.4 Output
Percentages for each category are computed over the full test set and saved in `checkpoint_ranking.csv` alongside the standard metrics.

---

## 12. Results Dashboard

### 12.1 Purpose
A self-contained HTML dashboard that visualises all experiment and checkpoint metrics without requiring a web server.

### 12.2 Generator: `generate_dashboard.py`
Run after any experiment to regenerate the dashboard:
```bash
python generate_dashboard.py
```
The script:
1. Reads `experiments/results/experiment_results.csv`
2. Reads all `eval/*_checkpoint_ranking.csv` files
3. Reads all plot PNGs from `experiments/results/plots/`
4. Embeds CSV data as **inline JSON** and images as **base64** directly in the HTML

**Output**: `experiments/results/dashboard.html` — opens in any browser via `file://`, no server needed.

### 12.3 Dashboard Features
| Tab | Contents |
|---|---|
| **Overview** | Animated stat cards (Test Accuracy, F1, Trainable Params, Peak GPU, Train Time), multiclass metrics bar chart, parameter efficiency bar chart, sortable experiment summary table |
| **Checkpoint Rankings** | Binary Accuracy / F1 / ROC AUC bar chart, crop+disease correctness doughnut chart, sortable checkpoint ranking table with progress bars |
| **Plots & Visuals** | Training curves, confusion matrix, and class-metrics PNGs rendered from embedded base64 |

### 12.4 Manual CSV Upload
A drag-and-drop upload zone allows loading additional CSV files at runtime. The dashboard auto-detects file type by column headers and merges data into the existing charts and tables.

### 12.5 Folder Structure (results)
```
experiments/results/
├── experiment_results.csv          # Aggregate metrics per experiment
├── dashboard.html                  # Generated self-contained dashboard
├── checkpoints/
│   ├── lora_best.pth
│   ├── lora_last.pth
│   └── lora_latest.pth
├── eval/
│   ├── lora_checkpoint_ranking.csv
│   ├── lora_best_confidences.csv
│   ├── lora_last_confidences.csv
│   └── lora_latest_confidences.csv
├── plots/
│   ├── lora_training_curves.png
│   ├── lora_confusion_matrix.png
│   └── lora_class_metrics.png
└── logs/
    └── lora_<timestamp>.log
```

---

## Next Steps
1. **Retrain QLoRA** if checkpoints predate the INT8 duplicate-weight fix (`python main.py qlora`)
2. Run Q/K LoRA experiment: `python main.py qklora`
3. Run full comparative study: LoRA vs QLoRA vs Q/K LoRA
4. Run `launcher_test.py` for each method, then implement `rank_experiments.py` (see FUTURE_FEATURES.md)
5. Regenerate dashboard (`python generate_dashboard.py`) with all three experiments loaded
6. Final comparative analysis and report write-up

### Checkpoint size reference (self-contained `.pth`, post-fix)
| Method | Approx. size | Trainable params |
|--------|--------------|------------------|
| LoRA | ~18 MB | ~343k |
| QLoRA | ~8 MB | ~193k |
| Q/K LoRA | ~9 MB | ~445k |

> **Note**: Bundled checkpoints include backbone + adapters. For deployment, also ship `config/class_labels.json`.

