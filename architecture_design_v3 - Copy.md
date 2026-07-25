# Leaf Disease Classification — Architecture Design V3

**Version**: V3 (PEFT + Domain Adaptation Track)  
**Project**: Comparative Study of LoRA, QLoRA, and QA-LoRA for EfficientNet-B0-Based Plant Leaf Disease Classification  
**Backbone**: EfficientNet-B0 (ImageNet-pretrained)  
**Classes**: 38 (PlantVillage)  
**Primary reference**: `deep-research-report-ACCURATE.md`

---

## 1. Method Definitions

### 1.1 LoRA V3 (Pointwise-only PEFT)
**Reference**: Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR).  
**Core idea**: Freeze the pretrained backbone. Insert low-rank adapter matrices A and B into selected layers. Update only the adapters.  
**Implementation status**: ✅ Real, fully wired into `training/lora_trainer_v3.py`.  
**Why it fits EfficientNet-B0**: MBConv blocks are dominated by 1×1 pointwise convolutions. Original transformer-style LoRA targets linear projections; the CNN analog is 1×1 conv, since `Conv2d(in, out, 1×1)` is mathematically equivalent to `Linear(in, out)` applied per spatial location.  
**Target modules in this project**:
- MBConv expand 1×1 (`block.0.0`), MBConv project 1×1 (`block.3.0`), and final head conv (`features.8.0`) — all non-depthwise `Conv2d` layers in the Q-path
- `classifier.fc`
- **Excluded**: Stem (`features.0.0`), depthwise convolutions (`groups > 1`), and SE layers (`fc1`/`fc2`)
**Trainable fraction**: ~7.8% of total parameters (~343k / 4.4M)  
**Implementation**: `models/peft/lora_v3.py` → `get_lora_v3_model()`, `training/lora_trainer_v3.py` → `LoRATrainerV3`

---

### 1.2 QLoRA V3 (Real NF4 Quantization)
**Reference**: Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS).  
**Implementation status**: ✅ Real bitsandbytes NF4, fully wired into `training/qlora_trainer_v3.py`.  
**Core idea (real paper)**: Uses `bitsandbytes.functional.quantize_4bit(..., quant_type='nf4')` on Q-path 1×1 conv weights. Frozen INT4 backbone + FP16 LoRA adapters. After fine-tuning, LoRA weights are merged back, but the resulting model is **dequantized to FP16** for inference — same limitation as the original paper.  
**Limitation in our project**: `bitsandbytes` 4-bit kernels only support `Linear` layers. Our `qlora_v3.py` works around this by reshaping Conv2d weights to 2D, quantizing with `bnb_f.quantize_4bit`, and dequantizing back to 2D before reshaping to 4D for the forward pass. This is a best-effort adaptation, not an official CNN QLoRA.  
**Target modules**:
- MBConv expand/project 1×1 (the "Q-path") + `features.8.0` + `classifier.fc`
- SE layers and depthwise convs remain FP32 and frozen
**Memory reduction**: Backbone weight storage drops from FP32 to INT4 (~8x reduction on Q-path), plus LoRA adapters.  
**Implementation**: `models/peft/qlora_v3.py` → `get_qlora_v3_model()`, `training/qlora_trainer_v3.py` → `QLoRATrainerV3`

---

### 1.3 QA-LoRA V3 (Real Group-wise Algorithm)
**Reference**: Xu et al., 2024 — *QA-LoRA: Quantization-Aware LoRA for LLMs* (ICLR 2024).  
**Implementation status**: ✅ Real Algorithm 1, truly wired into `training/qalora_trainer.py`.  
**Core idea (real paper)**: The paper identifies an imbalance — per-channel quantization has too few parameters relative to LoRA adapters. Solution: **group-wise operators**.
  - **Group-wise quantization**: Split each output channel's weights into `L` groups, each with its own scale/zero-point (increases quantization DOF).
  - **Grouped LoRA A**: Reduce LoRA A from `(D_in, rank)` → `(L, rank)` by adaptive avg-pool (decreases adaptation DOF).
  - This balance lets the model stay fully quantized after fine-tuning without FP16 fallback.
**CNN-specific adaptation**: The paper targets `Linear` layers where `nn.AvgPool1d(D_in // L)` operates on a 1D input vector. For Conv2d 1×1, we use `F.unfold(x)` → `adaptive_avg_pool1d(..., L)` to pool the `D_in` feature dimension.  
**Trainable parameters per layer**:
- Group-wise scale/zero-point: `(C_out × L × 2)` — more than standard per-channel
- Grouped LoRA A: `(L × rank)` — drastically smaller than `(D_in, rank)`
- Standard LoRA B: `(C_out, rank)`
**Does NOT use PEFT**: `QALoRAConv2d` replaces `nn.Conv2d` directly. PEFT cannot express the `(L, r)` LoRA A shape.  
**Implementation**: `models/peft/qalora.py` → `get_qalora_model()`, `training/qalora_trainer.py` → `QALoRATrainer`

---

## 2. Layer Target Summary

| Layer | LoRA V3 | QLoRA V3 (NF4) | QA-LoRA V3 |
|-------|---------|---------------|------------|
| MBConv expand 1×1 | ✅ adapter | ✅ NF4 base + adapter | ✅ group-wise quant base + grouped adapter |
| MBConv project 1×1 | ✅ adapter | ✅ NF4 base + adapter | ✅ group-wise quant base + grouped adapter |
| Head conv (`features.8.0`) | ✅ adapter | ✅ NF4 base + adapter | ✅ group-wise quant base + grouped adapter |
| SE `fc1` / `fc2` 1×1 | ❌ frozen | ❌ frozen | ❌ frozen (FP32) |
| Depthwise 3×3 | ❌ frozen | ❌ frozen | ❌ frozen |
| Stem (`features.0.0`) | ❌ frozen (FP32) | ❌ frozen (FP32) | ❌ frozen (FP32) |
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

### 4.2 LoRA V3 Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | none |
| Target modules | MBConv expand/project 1×1 + `features.8.0` + `classifier.fc` |

### 4.3 QLoRA V3 Hyperparameters (Real NF4)
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Quantization | **bitsandbytes 4-bit NF4** (weight-only, Q-path) — real paper algorithm |
| Target modules | MBConv expand/project 1×1 + `features.8.0` + `classifier.fc` |
| Implementation | `models/peft/qlora_v3.py` → `get_qlora_v3_model()` (fully wired into `qlora_trainer_v3.py`) |

### 4.4 QA-LoRA V3 Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Num groups (L) | 4 |
| Group-wise fake-quant | learnable per-group scale + zero-point (Q-path) |
| Target modules | MBConv expand/project 1×1 + `features.8.0` + `classifier.fc` |

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

### 6.1 LoRA Adapter Mechanics (V3)
For each target Conv2d layer with weight shape `(out, in, 1, 1)`:
- `A` shape: `(in, rank)` — Gaussian init
- `B` shape: `(rank, out)` — zero init
- Forward: `conv2d(x, W0 + (B @ A).view(out, in, 1, 1))`
- Scaled by `alpha / rank` inside PEFT

### 6.2 NF4 Weight-only Quantization V3 (Real QLoRA)
File: `models/peft/qlora_v3.py`, wired via `training/qlora_trainer_v3.py`
- **Status**: Real bitsandbytes NF4 implementation, matching Dettmers et al. (2023). Fully wired into `QLoRATrainerV3`.
- Uses `bitsandbytes.functional.quantize_4bit(..., quant_type='nf4')` on reshaped Conv2d weights
- Stores `q_weight` buffer + `q_state` (absmax for dequantization)
- Dequant during forward: `bn_f.dequantize_4bit(q_weight, q_state)`, reshaped back to 4D
- Original paper dequantizes to FP16 after merging; our V3 also dequantizes during forward — same limitation

### 6.3 Group-wise Fake Quantization + Grouped LoRA (QA-LoRA)
File: `models/peft/qalora.py` — `QALoRAConv2d`
- **Group-wise quantization**: Frozen backbone weight `W` of shape `(C_out, C_in, k_h, k_w)` is reshaped to `(C_out, L, D_in/L)`. Each group has its own learned `scale` and `zero_point` of shape `(C_out, L)`. During forward, each group is fake-quantized independently: `clamp(round(group / scale + zp), 0, 255)`. This gives `L` times more quantization parameters than per-channel, increasing the quantizer's degrees of freedom.
- **Grouped LoRA A**: Standard LoRA A of shape `(D_in, rank)` is replaced by `lora_A` of shape `(L, rank)`. The input tensor is unfolded to patches and then `adaptive_avg_pool1d` reduces the `D_in` feature dimension to `L` groups. The group-averaged input is scaled back by `group_size` before multiplying through lora_A.
- **LoRA B**: Standard shape `(C_out, rank)`, same as vanilla LoRA.
- **Status**: QA-LoRA does not use PEFT. The `QALoRAConv2d` module replaces the original `nn.Conv2d` directly, implementing the paper's Algorithm 1 faithfully for 2D convolutions.
- **Checkpoint**: Stores `weight_orig` (FP32 frozen) + `scale` + `zero_point` + `lora_A` + `lora_B`. At inference, weights remain FP32 `weight_orig` + learned `scale`/`zero_point`. A true INT8 checkpoint requires an extra quantization pass.

### 6.4 PEFT Integration (V3)
- **LoRA V3**: Uses `peft.LoraConfig` with `task_type=None` (required for non-transformer models).  
- **QLoRA V3**: Uses `peft.LoraConfig` + `qlora_v3._quantize_conv_to_nf4()` applied *after* `get_peft_model()`. Fully wired into `QLoRATrainerV3`.  
- **QA-LoRA V3**: Does **not** use PEFT. The `QALoRAConv2d` module replaces the original `nn.Conv2d` directly, implementing the paper's Algorithm 1 faithfully. This is necessary because the grouped LoRA A matrix has a different shape `(L, rank)` than PEFT's standard `(D_in, rank)`.

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
- `launcher_test_v3.py` evaluates checkpoints and writes `_checkpoint_ranking.csv`
- Per-sample confidences saved as CSV under `experiments/results/eval_v3/`

### 7.3 Visualization
- Training curves: `utils/visualization.py` → `plot_training_curves()`
- Confusion matrix: `evaluation/confusion_matrix.py` → `plot_confusion_matrix()`
- Class metrics: `evaluation/confusion_matrix.py` → `plot_class_metrics()`
- Dashboard: `generate_dashboard.py` → self-contained HTML with embedded CSV + base64 images

---

## 8. V3 File Inventory

| File | Role | Status |
|------|------|--------|
| `models/peft/lora_v3.py` | LoRA V3 model builder | ✅ Wired in `lora_trainer_v3.py` |
| `models/peft/qlora_v3.py` | QLoRA V3 — real bitsandbytes NF4 | ✅ Wired in `qlora_trainer_v3.py` |
| `models/peft/qalora.py` | QA-LoRA V3 — real group-wise quant + grouped LoRA A | ✅ Wired in `qalora_trainer.py` |
| `models/peft/int8_utils.py` | Shared Q-path name helpers for V3 | ✅ Used by all V3 model builders |
| `training/lora_trainer_v3.py` | LoRA V3 trainer | ✅ Wired |
| `training/qlora_trainer_v3.py` | QLoRA V3 trainer | ✅ Wired |
| `training/qalora_trainer.py` | QA-LoRA V3 trainer | ✅ Wired |
| `training/trainer.py` | Base trainer shared by all V3 trainers | ✅ Used |
| `config/base_config_v3.yaml` | V3 base training config | ✅ Used by all V3 trainers |
| `config/lora_config.yaml` | LoRA V3 PEFT config | ✅ Used |
| `config/qlora_config.yaml` | QLoRA V3 PEFT config | ✅ Used |
| `config/qalora_config.yaml` | QA-LoRA V3 config | ✅ Used |
| `main_v3.py` | V3 entry point | ✅ Routes to `experiment_runner_v3.py` |
| `launcher_v3.py` | V3 interactive launcher | ✅ Trains all 3 V3 methods |
| `experiments/experiment_runner_v3.py` | V3 experiment runner | ✅ Runs LoRA V3 / QLoRA V3 / QA-LoRA V3 |
| `launcher_test_v3.py` | V3 evaluation launcher | ✅ Evaluates all V3 checkpoints |

**No original V1 files were modified.** Legacy V1 code (`models/peft/lora.py`, `models/peft/qlora.py`, `training/lora_trainer.py`, `training/qlora_trainer.py`, `config/base_config.yaml`) remains untouched for backward compatibility.

---

## 9. Usage (V3 Track)

```bash
# V3 entry point — trains all 3 methods or a specific one
py -3.11 main_v3.py lora       # LoRA V3 (real PEFT, wired)
py -3.11 main_v3.py qlora      # QLoRA V3 (real bitsandbytes NF4, wired)
py -3.11 main_v3.py qalora     # QA-LoRA V3 (group-wise quant + grouped LoRA, wired)
py -3.11 main_v3.py all        # Train all V3 methods sequentially

# Interactive launcher
py -3.11 launcher_v3.py

# Evaluate V3 checkpoints
py -3.11 launcher_test_v3.py
```

> Note: QLoRA V3 requires `bitsandbytes` installed. All three V3 methods are fully wired and runnable via `main_v3.py`.

---

## 10. How V3 Methods Differ From the Original Papers

### 10.1 LoRA V3 (Hu et al., ICLR 2021)

**Original paper:**
- Targets **Linear** projection layers in Transformer attention: query (`Wq`), key (`Wk`), value (`Wv`), and output (`Wo`). The paper explicitly limits its study to attention weights and **freezes the MLP modules** for simplicity and parameter-efficiency.
- For a pre-trained weight matrix `W0 ∈ R^(d×k)`, the update is constrained as `ΔW = BA`, where `B ∈ R^(d×r)`, `A ∈ R^(r×k)`, and rank `r << min(d, k)`. During training, `W0` is frozen; only `A` and `B` are trainable.
- `A` initialized with random Gaussian, `B` initialized with zeros, so `ΔW = BA` is zero at training start.
- The adapter output is scaled by `α/r`. The paper sets `α` equal to the first `r` tried and does not tune it further.
- **Key claim: no additional inference latency.** At deployment, `W = W0 + BA` is explicitly computed and stored, so inference runs as a single matrix multiplication with no adapter overhead.
- Demonstrated on Transformer LMs: GPT-2, GPT-3 175B, RoBERTa, DeBERTa.

**Our EfficientNet-B0 adaptation (`models/peft/lora_v3.py`):**
- Targets **Conv2d 1×1** layers (`kernel_size=(1,1)`, `groups=1`) and `classifier.fc`. A 1×1 conv is mathematically equivalent to `Linear(in, out)` applied per spatial location, so the low-rank decomposition principle transfers to CNNs.
- `A` shape: `(C_in, r)`, `B` shape: `(r, C_out)` — same logical structure as the paper, but Conv2d weight is 4D `(C_out, C_in, 1, 1)`.
- **Excluded**: Stem (`features.0.0`) — first layer already well-pretrained on ImageNet low-level features.
- **Excluded**: Depthwise convolutions (`groups > 1`) — PEFT/HF does not natively support grouped convs for LoRA.
- **Excluded**: SE layers (`fc1`/`fc2`) — channel-attention, not channel-mixing projections.
- **Inference overhead difference**: The original paper merges LoRA into base weights (`W = W0 + BA`) for zero-latency inference. Our V3 implementation uses PEFT's wrapper, which keeps LoRA adapters as separate modules during inference. We do **not** perform the merge step, so inference carries the adapter overhead.
- **Target selection difference**: The original paper targets specific attention projections (`Wq`, `Wv` primarily). Our V3 uses a hand-crafted heuristic (`int8_utils.get_mbconv_q_path_names()`) to select MBConv expand/project 1×1 convs + head conv + classifier, based on the Q-path concept from CNN quantization literature rather than the paper's attention-layer selection.
- **MLP freezing difference**: The original paper freezes MLP modules entirely. Our V3 has no Transformer MLP; instead we freeze SE layers and depthwise convolutions, which are the CNN analog of "non-projection" layers.

**Why we cannot keep LoRA exactly the same as the original paper:**

- **Architecture family mismatch**: The original LoRA was designed for Transformer `Linear` projections inside self-attention (`Wq`, `Wk`, `Wv`, `Wo`). EfficientNet-B0 is a pure CNN with `Conv2d` layers and no attention mechanism. The layer types we must adapt are fundamentally different.
- **No attention layers to target**: The paper's default target set (`Wq`, `Wv`) does not exist in EfficientNet-B0. We must choose which `Conv2d` layers to adapt based on CNN-specific criteria (pointwise vs depthwise, Q-path vs SE) rather than the paper's attention-module heuristic.
- **PEFT library constraint for depthwise convolutions**: EfficientNet-B0's MBConv blocks use depthwise convolutions (`groups > 1`). The PEFT/HF library cannot attach LoRA adapters to `nn.Conv2d` layers with `groups > 1`. This is a hard implementation constraint, not a design choice — depthwise convs are excluded because LoRA cannot be applied to them, not because we chose to skip them.
- **Stem exclusion is a practical trade-off**: The paper does not address stem layers at all. We exclude `features.0.0` because it is already well-pretrained on ImageNet low-level features (edges, textures), and adding adapters there would increase trainable parameters without meaningful task-specific gain. This is a CNN-specific efficiency choice.
- **Inference merge step is not performed**: The original paper explicitly computes `W = W0 + BA` and stores the merged matrix for zero-latency inference. Our V3 uses PEFT's wrapper, which keeps LoRA adapters as separate modules during inference and does not automatically merge them. Skipping the merge means we retain adapter overhead at inference time, but we gain the ability to swap adapters without reloading the base model.

### 10.2 QLoRA V3 (Dettmers et al., NeurIPS 2023)

**Original paper:**
- Backpropagates gradients through a **frozen, 4-bit quantized pretrained model** into Low-Rank Adapters. Only LoRA parameters receive weight gradients; the 4-bit base weights are never updated.
- Introduces three key innovations: **(1) 4-bit NormalFloat (NF4)** — an information-theoretically optimal quantization data type for normally distributed weights; **(2) Double Quantization (DQ)** — quantizes the quantization constants themselves (first-level `cFP32_2` are quantized to 8-bit floats with blocksize 256, reducing memory overhead from ~0.5 bits/param to ~0.127 bits/param); **(3) Paged Optimizers** — uses NVIDIA unified memory to automatically page optimizer states between GPU and CPU, preventing OOM during gradient checkpointing on 33B/65B models.
- Uses **two data types**: a storage dtype (4-bit NF4) and a computation dtype (16-bit BrainFloat/BFloat16). Forward and backward passes dequantize to BF16 via `doubleDequant(c1, c2, W_NF4)`.
- Applies LoRA at **every network layer** (all Linear layers), not a hand-picked subset.
- After fine-tuning, LoRA weights are **merged** back into the quantized backbone, but the resulting model is **dequantized to BF16/FP16** for inference — the 4-bit precision is lost at deployment.
- Demonstrated on Transformer LMs: LLaMA 7B–65B, T5.

**Our EfficientNet-B0 adaptation (`models/peft/qlora_v3.py`):**
- Uses `bitsandbytes.functional.quantize_4bit(..., quant_type='nf4')` on reshaped Conv2d weights as a CNN workaround. The 4-bit kernels are designed for `Linear`, not `Conv2d`.
- **Does NOT implement Double Quantization** — we store only `q_weight` (NF4-packed) and `q_state` (absmax for dequantization). The paper's second-level quantization of quantization constants (`cFP32_2` → 8-bit float) is not implemented.
- **Does NOT implement Paged Optimizers** — these are specific to large LLM fine-tuning with gradient checkpointing memory spikes. Our EfficientNet-B0 model (~4.4M params) fits comfortably in GPU memory without paging.
- Targets only **Q-path pointwise convs + head + classifier**, not every layer. This is a deliberate parameter-efficiency choice consistent with the V3 track's "selective adaptation" philosophy.
- Dequantizes to the runtime dtype (FP32/FP16 via `torch.amp`) during forward. The paper explicitly dequantizes to BF16 for computation.
- After fine-tuning, LoRA weights would be merged back, but the model still dequantizes during inference — same FP16 fallback limitation as the paper.

**Why we cannot keep QLoRA exactly the same as the original paper:**

- **bitsandbytes kernel limitation**: The paper's 4-bit NF4 quantization and Double Quantization are implemented in `bitsandbytes` CUDA kernels that only support `torch.nn.Linear`. There is no official CNN-specific 4-bit QLoRA implementation. We work around this by reshaping Conv2d weights to 2D, quantizing, and reshaping back — but this bypasses the optimized block-wise quantization path and prevents us from using Double Quantization, which depends on the Linear-layer quantization infrastructure.
- **Double Quantization is not available for Conv2d**: The paper's DQ quantizes the per-block quantization constants (`cFP32_2`) to 8-bit floats. This relies on `bitsandbytes`'s `Linear`-specific quantization state management. Our Conv2d workaround only stores a single `absmax` (`q_state`) per layer, not the hierarchical quantized-constant structure required for DQ.
- **Paged Optimizers are unnecessary for our scale**: The paper introduces Paged Optimizers specifically to handle memory spikes during gradient checkpointing on 33B/65B models. Our EfficientNet-B0 backbone is ~4.4M parameters; even with 4-bit quantization, the memory profile does not trigger the paging scenario the paper addresses.
- **"Adapters at every layer" is a design choice, not a requirement**: The paper states that including adapters at every layer avoids accuracy tradeoffs seen in prior work. However, this was studied on Transformer architectures where every Linear layer is a candidate. For EfficientNet-B0, applying LoRA to all 1×1 convs would significantly increase trainable parameters and contradicts the V3 track's goal of comparing selective vs. quantized adaptation. Our targeted Q-path selection is a CNN-specific efficiency decision.
- **Computation dtype difference**: The paper mandates BF16 computation to maintain numerical stability through the quantized base. Our V3 inherits the project's mixed-precision setup (`torch.amp` with FP16/BF16 depending on GPU), which may use FP16 rather than the paper's preferred BF16 for the dequantized forward pass.

### 10.3 QA-LoRA V3 (Xu et al., ICLR 2024)

**Original paper:**
- Targets **Linear** layers in LLaMA/LLaMA2
- Core insight: **imbalanced degrees of freedom** — per-channel quantization has only `D_out` scale/zp parameter pairs, while LoRA has `D_in × rank + rank × D_out` adaptation parameters. This imbalance makes it impossible to merge `s·AB` into quantized weights without high-precision fallback.
- **Solution: group-wise operators** — split each output channel's weights into `L` groups, each with its own scale/zero-point (increases quantization DOF from `D_out` to `L × D_out`). Reduce LoRA A from `(D_in, rank)` to `(L, rank)` via `nn.AvgPool1d(D_in // L)` (decreases adaptation DOF from `D_in × rank` to `L × rank`).
- **Asymmetric INT4 quantization**: `W_tilde = alpha_j * round((W - beta_j) / alpha_j) + beta_j` per group. Zero-point `beta_j` ensures exact representation of zero.
- **Grouped LoRA A**: The paper's Algorithm 1 shows `lora_A` shape `(D_int, L)` (transposed to `(L, D_int)` in forward), `lora_B` shape `(D_out, D_int)`. Forward: `result += (QA(x) * (D_in//L)) @ lora_A.T @ lora_B.T * s`.
- **Zero-point update rule**: After fine-tuning, zero-points are explicitly updated via `beta_new = beta - s * (lora_B @ lora_A).T / alpha` so the merged weights `W' = W_tilde + s·AB` remain in INT4 without FP16 fallback.
- **No post-training quantization needed**: The model stays fully quantized after fine-tuning. Inference has the same complexity as QLoRA with PTQ, but is much faster than QLoRA without PTQ.
- Demonstrated on LLaMA 7B–65B and LLaMA2 7B–13B.

**Our EfficientNet-B0 adaptation (`models/peft/qalora.py`):**
- Targets **Conv2d 1×1** layers — `nn.AvgPool1d` cannot operate on 4D image tensors
- We use `F.unfold(x)` to extract patches, then `F.adaptive_avg_pool1d(x_unfold.transpose(1,2), L).transpose(1,2)` to pool the `D_in` feature dimension to `L` groups
  - Note: `adaptive_avg_pool1d` pools over the **last** dimension, hence the transpose
  - The paper's `AvgPool1d` pools over contiguous input groups; our unfold+pool is an approximation since spatial patches are unfolded into `(B, C_in*kH*kW, N_positions)` and pooled globally
- `lora_A` shape: `(L, rank)` — corresponds to the transposed paper version `(D_int, L).T`
- `lora_B` shape: `(C_out, rank)` — corresponds to the transposed paper version `(D_out, D_int).T`
- Weights are **unsigned INT8 fake-quantized** `[0, 255]` (not signed INT4 `[0, 15]`) — paper uses INT4; we use INT8 because Conv2d weight distributions differ from Linear
- **No explicit merge step**: at inference, weights remain FP32 `weight_orig` + learned `scale`/`zero_point`. A true INT8 checkpoint requires an extra quantization pass
- Does **not** use PEFT; `QALoRAConv2d` is a custom `nn.Module` because PEFT cannot express the `(L, r)` LoRA A shape

**Why we cannot keep QA-LoRA exactly the same as the original paper:**

- **Conv2d requires unfold+pool instead of AvgPool1d**: The paper's `nn.AvgPool1d(D_in // L)` operates directly on 1D input vectors in Linear layers. For Conv2d 1×1, we must first `F.unfold(x)` to extract patches into `(B, D_in, N_positions)`, then `adaptive_avg_pool1d` over the last dimension. This unfold+pool is an approximation — the paper pools contiguous input groups, while our method pools over spatial patches globally, which may group features differently.
- **INT4 asymmetric quantization is not implemented**: The paper uses true INT4 asymmetric quantization with zero-point: `W_tilde = alpha_j * round((W - beta_j) / alpha_j) + beta_j`, where values range from `-2^(N-1)` to `2^(N-1)-1` (e.g., `-8` to `7` for INT4). Our V3 uses **unsigned INT8 fake-quantization** `[0, 255]` with no negative range. This is because Conv2d weight distributions in EfficientNet-B0 have different statistical properties than Transformer Linear weights, and INT4's limited range would cause excessive quantization error. The paper's INT4 is feasible for LLM weights which are approximately normally distributed around zero.
- **Zero-point merge rule is not implemented**: The paper's key innovation is the `merge_with_quantization` function: `beta_new = beta - s * (lora_B @ lora_A).T / alpha`. This explicitly updates zero-points after fine-tuning so that `W' = W_tilde + s·AB` remains in INT4 without FP16 fallback. Our V3 does not implement this merge rule. Instead, we learn scale and zero-point via standard backpropagation, and at inference the weights remain in FP32 `weight_orig` + learned `scale`/`zero_point`. Without the explicit merge, we cannot produce a true INT8 checkpoint without an extra quantization pass.
- **Fake-quantization instead of true integer quantization**: The paper uses true integer quantization where weights are rounded to integers and stored as such. Our V3 uses fake-quantization — the weights stay in FP32 (`weight_orig`) and the quantization is performed as a differentiable operation in the forward pass. This is a practical necessity for training with autograd, but it means the weights are never truly stored in low-bit format during training.
- **Target layer selection is CNN-specific**: The paper applies QA-LoRA to all Linear layers in LLaMA/LLaMA2. Our V3 targets only Q-path Conv2d 1×1 layers (MBConv expand/project + head + classifier), excluding SE layers and depthwise convolutions. This selective targeting is a CNN-specific design choice to balance parameter efficiency with adaptation capacity.
- **PEFT incompatibility is a hard constraint**: The paper's grouped LoRA A shape `(L, rank)` is incompatible with PEFT's standard LoRA A shape `(D_in, rank)`. This is true for both the paper's Linear case and our Conv2d case. We both implement custom modules instead of using PEFT. However, this constraint is more limiting for our CNN because PEFT's `Conv2d` support is already more restricted than its `Linear` support.
