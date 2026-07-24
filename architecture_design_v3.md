# Leaf Disease Classification — Architecture Design V3

**Version**: V3 (PEFT + Domain Adaptation Track)  
**Project**: Comparative Study of LoRA, QLoRA, and QA-LoRA for EfficientNet-B0-Based Plant Leaf Disease Classification  
**Backbone**: EfficientNet-B0 (ImageNet-pretrained)  
**Classes**: 38 (PlantVillage)  
**Primary reference**: `deep-research-report-ACCURATE.md`

---

## 1. Method Definitions

### 1.1 LoRA (Standard)
**Reference**: Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR).  
**Core idea**: Freeze the pretrained backbone. Insert low-rank adapter matrices A and B into selected layers. Update only the adapters.  
**Why it fits EfficientNet-B0**: MBConv blocks are dominated by 1×1 pointwise convolutions. Original transformer-style LoRA targets linear projections; the CNN analog is 1×1 conv, since `Conv2d(in, out, 1×1)` is mathematically equivalent to `Linear(in, out)` applied per spatial location.  
**Target modules in this project**:
- All non-depthwise `Conv2d` layers (`groups == 1`) in the backbone: stem, MBConv expand/project, head conv
- `classifier.fc`
- **Excluded**: Depthwise convolutions (`groups > 1`) — PEFT cannot attach LoRA to grouped convolutions
**Trainable fraction**: ~7.8% of total parameters (~343k / 4.4M)  
**Implementation**: `models/peft/lora.py` → `get_lora_model()`, `training/lora_trainer.py` → `LoRATrainer`

---

### 1.2 QLoRA (Quantized LoRA)
**Reference**: Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS).  
**Core idea**: Freeze the backbone and quantize it to low bit-width (4-bit NF4 in original work). Backpropagation flows only through LoRA adapters. Backbone weights remain fixed in quantized form.  
**CNN-specific adaptation**: `bitsandbytes` 4-bit kernels are designed for `Linear` layers, not `Conv2d`. This project uses **per-channel INT8** as a CNN-compatible substitute for the Q-path weights, preserving the same principle: frozen quantized backbone + trainable LoRA adapters. Activations remain in FP32/FP16.  
**Why INT8 instead of INT4/NF4**: The deep-research notes "No published CNN-specific QLoRA exists" and that INT4/NF4 quantization of Conv2d weights is non-trivial. INT8 provides a practical middle ground with stable convergence and meaningful memory savings.  
**Target modules**:
- MBConv expand/project 1×1 (the "Q-path") + `classifier.fc`
- SE layers and depthwise convs remain FP32 and frozen
**Memory reduction**: Backbone weight storage drops from FP32 (~4.4 MB for 1.1M params after excluding non-Q-path) to INT8 (~1.1 MB equivalent), plus LoRA adapters (~0.2 MB). Total checkpoint: ~8 MB.  
**Implementation**: `models/peft/qlora.py` → `get_qlora_model()`, `models/peft/int8_utils.py`, `training/qlora_trainer.py` → `QLoRATrainer`

---

### 1.3 QA-LoRA (Quantization-Aware LoRA)
**Reference**: Xu et al., 2024 — *QA-LoRA: Quantization-Aware LoRA for LLMs*.  
**Core idea**: Unlike QLoRA's fixed post-training quantization, QA-LoRA makes quantization **part of the training graph**. Per-channel scale factors and zero-points are learned parameters. Both backbone base weights and LoRA adapters are fake-quantized during training via straight-through estimator (STE). At inference, the merged result can be folded into fully quantized INT8.  
**Why this matters for CNNs**: After INT8 post-training quantization, feature statistics can drift because quantization error is uneven across channels. By learning the scale/zero-point, QA-LoRA finds a quantization frontier that preserves task-relevant information better than fixed clipping.  
**Forward pass**:
1. Frozen backbone `W_fp32` → fake-quant to `W_qa = (clamp(round(W/scale + zp), -128, 127) - zp) * scale`
2. LoRA adapters produce `ΔW = B @ A`
3. Effective weight: `W_eff = W_qa + ΔW`
4. Convolution with `W_eff`
**Trainable parameters**:
- LoRA A/B matrices (same count as vanilla LoRA)
- Per-output-channel `_qa_scale` and `_qa_zp` for each quantized base layer
**Implementation**: `models/peft/qalora.py` → `get_qalora_model()`, `models/peft/fake_quant.py`, `training/qalora_trainer.py` → `QALoRATrainer`

---

### 1.4 Why Q/K LoRA Is Not in V3
The earlier Q/K LoRA design (selective INT8 on pointwise + FP32 on SE with tiered ranks) was a custom invention. It is **not a standard PEFT method** and is **not part of V3**. It remains in the codebase only for backward compatibility with old checkpoints trained under earlier project versions. V3 studies only the three established methods listed above.

---

## 2. Layer Target Summary

| Layer | LoRA | QLoRA | QA-LoRA |
|-------|------|-------|---------|
| MBConv expand 1×1 | ✅ adapter | ✅ INT8 base + adapter | ✅ fake-quant base + adapter |
| MBConv project 1×1 | ✅ adapter | ✅ INT8 base + adapter | ✅ fake-quant base + adapter |
| Head conv (`features.8.0`) | ✅ adapter | ✅ INT8 base + adapter | ✅ fake-quant base + adapter |
| SE `fc1` / `fc2` 1×1 | ❌ frozen | ❌ frozen | ❌ frozen (FP32) |
| Depthwise 3×3 | ❌ frozen | ❌ frozen | ❌ frozen |
| Stem (`features.0.0`) | ✅ adapter | ❌ frozen (FP32) | ❌ frozen (FP32) |
| `classifier.fc` | ✅ adapter | ✅ adapter | ✅ adapter |

**Rationale**:
- Pointwise convolutions contain most MBConv parameters and act as channel-mixing linear projections.
- SE layers are channel-attention, not projection; keeping them FP32 preserves attention precision.
- Depthwise convs are spatial feature extractors; PEFT cannot attach LoRA to `groups > 1`, and they contain few parameters anyway.
- Stem is excluded to reduce adapter count on the first layer, which is already well-pretrained on ImageNet low-level features.

---

## 3. Data Pipeline

### 3.1 Dataset
- **Source**: PlantVillage Dataset (~54,000 RGB images)
- **Task**: 38-class plant disease classification
- **Class format**: `Crop___Disease` (e.g. `Tomato___Late_blight`, `Tomato___healthy`)

### 3.2 Split Strategy
Official Hugging Face PlantVillage splits:
- **Train**: 85% of HF `train` split (68% of total)
- **Val**: 15% of HF `train` split (12% of total)
- **Test**: 100% of HF `test` split (20% of total)
- Splits use `leaf_id` grouping to prevent leaf-image leakage

### 3.3 Transformations
**Training**:
- Resize(256) → RandomCrop(224)
- RandomHorizontalFlip
- RandomRotation(±15°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Val/Test**:
- Resize(256) → CenterCrop(224)
- Normalize(same)

### 3.4 Class Label Map
- `config/class_labels.json` auto-generated by `get_data_loaders()`
- Contains `num_classes`, `idx_to_class`, `class_to_idx`
- Must ship with every checkpoint for deployment

### 3.5 Class-Imbalance Mitigation
Both options (off by default, enable only one):
- `data.use_class_weights: true` → `CrossEntropyLoss(weight=class_weights)`
- `data.use_weighted_sampler: true` → `WeightedRandomSampler`

---

## 4. Training Configuration

### 4.1 Base Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Epochs | 20 (early stopping patience: 7) |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Mixed precision | FP16/BF16 (`torch.amp`) |
| Input resolution | 224×224 |

### 4.2 LoRA Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | none |
| Target modules | all non-depthwise Conv2d + classifier.fc |

### 4.3 QLoRA Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Quantization | bitsandbytes 4-bit NF4 (weight-only, Q-path) |
| Target modules | MBConv expand/project 1×1 + classifier.fc |

### 4.4 QA-LoRA Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Fake-quant | learnable per-channel scale + zero-point (Q-path) |
| Target modules | same as QLoRA |

---

## 5. Model Architecture

### 5.1 Backbone
- `torchvision.models.efficientnet_b0(weights=None)` + manual state_dict load from cached `efficientnet_b0*.pth`
- Feature extractor: `model.features` (MBConv blocks + final conv)
- Pooling: `model.avgpool` (global average pooling, built-in)
- **All backbone parameters frozen** — only PEFT adapters and (for QA-LoRA) quantization parameters are trainable

### 5.2 Classifier Head
Replaces the default EfficientNet classifier:
- Input: 1280 (EfficientNet-B0 pooled features)
- Dropout (p=0.2)
- Linear(1280, 38)
- **No Softmax** — CrossEntropyLoss applies log_softmax internally; softmax is applied only at inference time in the evaluator

---

## 6. Implementation Details

### 6.1 LoRA Adapter Mechanics
For each target Conv2d layer with weight shape `(out, in, 1, 1)`:
- `A` shape: `(in, rank)` — Gaussian init
- `B` shape: `(rank, out)` — zero init
- Forward: `conv2d(x, W0 + (B @ A).view(out, in, 1, 1))`
- Scaled by `alpha / rank` inside PEFT

### 6.2 INT8 Quantization (QLoRA)
File: `models/peft/int8_utils.py`
- Per-output-channel scale: `s = max(|W|) / 127`
- Quantized weight: `W_q = clamp(round(W / s), -128, 127)` stored as `int8`
- Dequant during forward: `W_deq = W_q.float() * s`
- Original FP32 `weight` parameter deleted from `conv._parameters` to save memory
- Checkpoints store `weight_int8` buffer + `weight_scale` buffer

### 6.3 Fake Quantization (QA-LoRA)
File: `models/peft/fake_quant.py`
- Learnable parameters per quantized layer: `_qa_scale` (fp32, shape `[out_channels]`), `_qa_zp` (fp32, shape `[out_channels]`)
- Initialized from per-channel max absolute values
- Forward: `W_qa = (clamp(round(W / scale) + zp, -128, 127) - zp) * scale`
- Backward: straight-through estimator (gradient passes through `clamp` and `round` unchanged)
- At inference: scales and zero-points are baked into INT8 weights

### 6.4 PEFT Integration
- All three methods use `peft.LoraConfig` with `task_type=None` (required for non-transformer models)
- QLoRA applies `int8_utils.quantize_lora_base_layers()` *after* `get_peft_model()` so that base Conv2d layers inside PEFT wrappers are quantized
- QA-LoRA applies fake-quant hooks similarly after PEFT wrapping

---

## 7. Evaluation Protocol

### 7.1 Metrics
- **Primary**: Top-1 accuracy, F1 macro, precision macro, recall macro
- **Binary**: healthy vs diseased (string match on `"healthy"` in class name)
- **AUC**: ROC AUC (One-vs-Rest) from per-class softmax probabilities
- **Crop/Disease decomposition**: `Crop___Disease` split into `crop_pred/disease_pred` vs `crop_true/disease_true`
  - `both_correct` | `name_only_correct` | `disease_only_correct` | `none_correct`

### 7.2 Checkpoint Evaluation
- `training/trainer.py` saves `best`, `last`, and optionally `epoch_N` checkpoints
- `launcher_test.py` / `run_eval_phase2.py` evaluate checkpoints and write `_checkpoint_ranking.csv`
- Per-sample confidences saved as CSV under `experiments/results/eval/`

### 7.3 Visualization
- Training curves: `utils/visualization.py` → `plot_training_curves()`
- Confusion matrix: `evaluation/confusion_matrix.py` → `plot_confusion_matrix()`
- Class metrics: `evaluation/confusion_matrix.py` → `plot_class_metrics()`
- Dashboard: `generate_dashboard.py` → self-contained HTML with embedded CSV + base64 images

---

## 8. File Inventory (V3 Additions)

| File | Role |
|------|------|
| `architecture_design_3method.md` | This document |
| `models/peft/qalora.py` | QA-LoRA model builder |
| `models/peft/fake_quant.py` | Learnable fake-quant utility |
| `training/qalora_trainer.py` | QA-LoRA trainer |
| `config/qalora_config.yaml` | QA-LoRA hyperparameters |

**No original files were modified.** Q/K LoRA code (`models/peft/qklora.py`, `training/qklora_trainer.py`, `config/qklora_config.yaml`) remains untouched.

---

## 9. Usage (V3 Track)

```bash
# Existing methods (unchanged)
py -3.11 main.py lora       # LoRA
py -3.11 main.py qlora      # QLoRA (INT8 backbone)
py -3.11 main.py qklora     # Q/K LoRA (legacy, 4th method)

# New V3 method
py -3.11 main.py qalora     # QA-LoRA (learnable fake-quant)
```

> Note: `main.py` currently does not route `qalora`. QA-LoRA is available via direct trainer import until V3 wiring is complete.

---

## 10. References

1. Hu et al. (2021), ICLR — LoRA
2. Dettmers et al. (2023), NeurIPS — QLoRA
3. Xu et al. (2024) — QA-LoRA
4. Hu et al. (2025), ArXiv — Survey: Bridging Domain Gaps in Agricultural Image Analysis
5. Richter & Kim (2025), Sci. Rep. — Benchmark of Transfer-Learning on Plant Leaf Disease Datasets
6. Li et al. (2016), ICLR — AdaBN
7. Wang et al. (2021), ICLR — TENT
8. Wu et al. (2023), Plant Phenomics — MSUN (PlantVillage→PlantDoc UDA)
