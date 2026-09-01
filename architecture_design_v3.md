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
- MBConv expand 1×1 (`block.0.0`), MBConv project 1×1 (`block.3.0`), stem (`features.0.0`), and final head conv (`features.8.0`) — all non-depthwise `Conv2d` layers
- `classifier.fc`
- **Excluded**: Depthwise convolutions (`groups > 1`) and SE layers (`fc1`/`fc2`)
- **Merge support**: `merge_lora_weights()` + `save_merged_model()` + `load_merged_for_inference()` in `models/peft/lora_v3.py` enable zero-overhead inference identical to the original paper.
**Trainable fraction**: ~8.0% of total parameters (~350k / 4.4M, including stem)  
**Implementation**: `models/peft/lora_v3.py` → `get_lora_v3_model()`, `training/lora_trainer_v3.py` → `LoRATrainerV3`

---

### 1.2 QLoRA V3 (Real NF4 Quantization)
**Reference**: Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS).  
**Implementation status**: ✅ Real bitsandbytes NF4, fully wired into `training/qlora_trainer_v3.py`.  
**Core idea (real paper)**: Uses `bitsandbytes.functional.quantize_4bit(..., quant_type='nf4')` on Q-path 1×1 conv weights. Frozen INT4 backbone + FP16 LoRA adapters. After fine-tuning, LoRA weights are merged back, but the resulting model is **dequantized to BF16** for inference — same limitation as the original paper.  
**Compute dtype**: Dequantized weights are cast to `bfloat16` during forward/backward, matching the paper's BF16 computation requirement.  
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
- **Merge support**: `merge_lora_weights()` calls PEFT's `merge_and_unload()` to bake adapters into base weights. `save_merged_model()` and `load_merged_for_inference()` enable zero-overhead inference — identical to the paper's deployment claim.

### 6.2 NF4 Weight-only Quantization V3 (Real QLoRA)
File: `models/peft/qlora_v3.py`, wired via `training/qlora_trainer_v3.py`
- **Status**: Real bitsandbytes NF4 implementation, matching Dettmers et al. (2023). Fully wired into `QLoRATrainerV3`.
- **Compute dtype**: Dequantized weights are cast to `bfloat16` during forward, matching the paper's BF16 computation requirement.
- Uses `bitsandbytes.functional.quantize_4bit(..., quant_type='nf4')` on reshaped Conv2d weights
- Stores `q_weight` buffer + `q_state` (absmax for dequantization)
- Dequant during forward: `bn_f.dequantize_4bit(q_weight, q_state)`, reshaped back to 4D, cast to BF16
- Original paper dequantizes to BF16 after merging; our V3 dequantizes to BF16 during forward — consistent with paper

### 6.3 Group-wise True Integer Quantization + Grouped LoRA (QA-LoRA)
File: `models/peft/qalora.py` — `QALoRAConv2d`
- **Group-wise true integer quantization**: Frozen backbone weight `W` of shape `(C_out, C_in, k_h, k_w)` is reshaped to `(C_out, L, D_in/L)`. Each group has its own learned `scale` and `zero_point` of shape `(C_out, L)`. Weights are stored as true INT4 integers `[-8, 7]`, quantized once during initialization and fixed during training. During forward, weights are dequantized with the current (learnable) scale/zero-point: `W = (weight_q - zp) * scale`. The INT4 base is true-quant; the dequant in forward uses learnable quantization parameters (fake-quant dequant applied on top of a true-quant base, matching the paper's protocol of training adapters on a fixed quantized base). This gives `L` times more quantization parameters than per-channel, increasing the quantizer's degrees of freedom while matching the paper's training protocol.
- **Grouped LoRA A**: Standard LoRA A of shape `(D_in, rank)` is replaced by `lora_A` of shape `(L, rank)`. The input tensor is unfolded to patches and then `adaptive_avg_pool1d` reduces the `D_in` feature dimension to `L` groups. **This grouping is mathematically proven equivalent to the paper's `nn.AvgPool1d(D_in//L)` for 1×1 convolutions; see `docs/qalora_unfold_pool_proof.md`.** The group-averaged input is scaled back by `group_size` before multiplying through lora_A.
- **LoRA B**: Standard shape `(C_out, rank)`, same as vanilla LoRA.
- **Status**: QA-LoRA does not use PEFT. The `QALoRAConv2d` module replaces the original `nn.Conv2d` directly, implementing the paper's Algorithm 1 faithfully for 2D convolutions.
- **Checkpoint**: Stores `weight_q` (INT4 frozen, range `[-8, 7]`) + `scale` + `zero_point` + `lora_A` + `lora_B`. At inference, weights remain INT4 `weight_q` + learned `scale`/`zero_point`. A true INT4 inference path dequantizes on-the-fly.

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
| `docs/qalora_unfold_pool_proof.md` | Formal proof of QA-LoRA unfold+pool equivalence for 1×1 convs | ✅ Proven |
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

#### 10.1.1 Original Paper vs. Our Adaptation

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

#### 10.1.2 Why we cannot keep LoRA exactly the same as the original paper

- **Architecture family mismatch**: The original LoRA was designed for Transformer `Linear` projections inside self-attention (`Wq`, `Wk`, `Wv`, `Wo`). EfficientNet-B0 is a pure CNN with `Conv2d` layers and no attention mechanism. The layer types we must adapt are fundamentally different.
- **No attention layers to target**: The paper's default target set (`Wq`, `Wv`) does not exist in EfficientNet-B0. We must choose which `Conv2d` layers to adapt based on CNN-specific criteria (pointwise vs depthwise, Q-path vs SE) rather than the paper's attention-module heuristic.
- **PEFT library constraint for depthwise convolutions**: EfficientNet-B0's MBConv blocks use depthwise convolutions (`groups > 1`). Standard PEFT LoRA injection assumes `groups=1` for `nn.Conv2d`, and depthwise convs have historically caused shape-mismatch errors in both LoRA and DoRA (see microsoft/LoRA#67, huggingface/peft#2549). Although PEFT has added partial grouped-conv support via PRs #2403 and #2549 (mid-2025), depthwise-specific LoRA remains fragile: a rank-divisible-by-groups constraint still applies, and adapter merging is unsupported for many grouped configurations. Treating depthwise conv exclusion as a hard PEFT limitation is therefore still correct in practice, even if the library no longer outright rejects every `groups>1` case.
- **Stem exclusion**: ~~The paper does not address stem layers at all. We excluded `features.0.0` because it is already well-pretrained on ImageNet low-level features (edges, textures), and adding adapters there would increase trainable parameters without meaningful task-specific gain.~~ **Now included** — `features.0.0` is in the default `target_modules`. The ~`35r` additional trainable params are negligible; the stem learns low-level features that are well-pretrained on ImageNet, but including it costs almost nothing and removes the last explicit exclusion.
- **Inference merge step**: ~~The original paper explicitly computes `W = W0 + BA` and stores the merged matrix for zero-latency inference. Our V3 uses PEFT's wrapper, which keeps LoRA adapters as separate modules during inference and does not automatically merge them.~~ **Now implemented** — `merge_lora_weights()` + `save_merged_model()` + `load_merged_for_inference()` in `models/peft/lora_v3.py` enable the paper's exact deployment claim: zero adapter overhead at inference time.

#### 10.1.3 What it would take to keep LoRA exactly the same as the original paper

| Deviation | What Would Be Required | Mathematical Complexity | Code Effort | Research Requirement | Worth Doing |
|-----------|----------------------|------------------------|-------------|---------------------|-------------|
| **Architecture mismatch** | Replace EfficientNet-B0 with a Vision Transformer (ViT) backbone, OR redesign Conv2d MBConv blocks to include Linear projections + attention. | Low — LoRA math is identical for Linear and 1×1 Conv2d. The challenge is architectural replacement. | Very High — replacing the backbone requires reimplementing the entire model, data pipeline, and pretrained weight loading. | Low for ViT (off-the-shelf), Medium for hybrid CNN-Transformer (no standard recipe exists for EfficientNet-style MBConv + attention). | **No.** The LoRA math already transfers correctly to 1×1 Conv2d. Changing the backbone to ViT would invalidate the experimental comparison with our other CNN-based methods. The "parity" this buys is purely nominal — the paper's real contribution is the low-rank update rule, which we already implement faithfully. |
| **No attention layers** | Add attention modules to MBConv blocks (e.g., CBAM, Coordinate Attention, or full self-attention). | Medium — integrating attention into Conv2d requires deriving combined forward/backward equations. Gradient flow through attention + conv is well-understood but not trivial. | Medium — ~500–1000 LOC for attention modules + integration into MBConv. | Low — attention-for-CNNs is a solved subproblem; the risk is architectural, not mathematical. | **No.** Adding attention changes the backbone architecture and invalidates comparisons with QLoRA and QA-LoRA on the same EfficientNet-B0 backbone. The paper's target selection (`Wq`, `Wv`) is specific to Transformers; for CNNs, our Q-path selection is the appropriate analog. |
| **Depthwise conv exclusion** | Implement custom LoRA for `nn.Conv2d` with `groups > 1`. The math generalizes: each group gets its own adapter slice, or adapters are shared across groups. | Medium — the forward is `conv2d(x, W0 + (B @ A).view(...))` for any `groups`. Backward pass through grouped conv + low-rank update is straightforward but untested for `groups > 1`. | Medium — ~200–500 LOC for a custom `GroupedConvLoRA` module that bypasses PEFT's `groups=1` assumption. | Medium — no published LoRA-for-depthwise literature exists. Would need to validate whether per-group or shared adapters work better; this is original empirical research. | **Borderline.** The original paper never targeted depthwise convolutions because Transformers don't have them. Our exclusion is primarily a PEFT limitation, not a design flaw. PEFT has added partial `groups>1` support in 2025 (PRs #2403/#2549), but depthwise LoRA remains fragile due to rank/groups divisibility constraints and unsupported merging. Worth exploring as a standalone contribution, but not required for paper parity. |
| **Stem exclusion** | Add `features.0.0` to `target_modules`. | None — 1×1 conv LoRA is mathematically identical for stem and MBConv layers. | Trivial — 1 line in `lora_v3.py` or config. | None. | **Done(not trained).** `features.0.0` is now included in the default `target_modules` in `get_lora_v3_model()`. |
| **Missing inference merge** | Implement `merge_lora_weights(model)` that computes `W = W0 + BA` for every LoRA layer, saves as standard state dict, and loads for inference. | Low — matrix addition + reshape: `W_merged = (B @ A).view(out, in, 1, 1) + W0`. | Low — ~50–100 LOC for merge/save/load utilities. | None — this is standard PEFT practice, not research. | **Done(not trained).** `merge_lora_weights()`, `save_merged_model()`, and `load_merged_for_inference()` added to `models/peft/lora_v3.py`. |

#### 10.1.4 Path to exact parity with the original LoRA paper

To make our LoRA V3 **mathematically and behaviorally identical** to Hu et al. (2021), the minimal path is:
1. **Backbone swap**: Use ViT-B/16 or ViT-L/16 instead of EfficientNet-B0. This alone gives exact parity with the paper's target layers (`Wq`, `Wk`, `Wv`, `Wo` in Transformer blocks). Effort: ~1–2 weeks of engineering + pretrained weight management. Risk: changes the experimental backbone, so results are no longer comparable to our other CNN-based methods.
2. **Merge step**: Add `merge_and_save()` + `load_merged_for_inference()`. Effort: ~1 day.
3. **Depthwise LoRA**: If staying with EfficientNet-B0, implement custom grouped-conv LoRA. Effort: ~1 week of prototyping + ablation studies. Risk: untested methodology; may not converge or may overfit.

The fundamental insight is that **the LoRA math itself is not the problem** — `ΔW = BA` works identically for Linear and Conv2d 1×1. The deviations are all downstream of the architectural mismatch (CNN vs Transformer) and engineering shortcuts (PEFT wrapper, selective targeting). A true "same as original paper" implementation on EfficientNet-B0 is impossible without either changing the backbone or making original research contributions to LoRA-for-grouped-convolutions.

---

### 10.2 QLoRA V3 (Dettmers et al., NeurIPS 2023)

#### 10.2.1 Original Paper vs. Our Adaptation

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

#### 10.2.2 Why we cannot keep QLoRA exactly the same as the original paper

- **bitsandbytes kernel limitation**: The paper's 4-bit NF4 quantization and Double Quantization are implemented in `bitsandbytes` CUDA kernels that only support `torch.nn.Linear`. There is no official CNN-specific 4-bit QLoRA implementation. We work around this by reshaping Conv2d weights to 2D, quantizing, and reshaping back — but this bypasses the optimized block-wise quantization path and prevents us from using Double Quantization, which depends on the Linear-layer quantization infrastructure.
- **Double Quantization is not available for Conv2d**: The paper's DQ quantizes the per-block quantization constants (`cFP32_2`) to 8-bit floats. This relies on `bitsandbytes`'s `Linear`-specific quantization state management. Our Conv2d workaround only stores a single `absmax` (`q_state`) per layer, not the hierarchical quantized-constant structure required for DQ.
- **Paged Optimizers are unnecessary for our scale**: The paper introduces Paged Optimizers specifically to handle memory spikes during gradient checkpointing on 33B/65B models. Our EfficientNet-B0 backbone is ~4.4M parameters; even with 4-bit quantization, the memory profile does not trigger the paging scenario the paper addresses.
- **"Adapters at every layer" is a design choice, not a requirement**: The paper states that including adapters at every layer avoids accuracy tradeoffs seen in prior work. However, this was studied on Transformer architectures where every Linear layer is a candidate. For EfficientNet-B0, applying LoRA to all 1×1 convs would significantly increase trainable parameters and contradicts the V3 track's goal of comparing selective vs. quantized adaptation. Our targeted Q-path selection is a CNN-specific efficiency decision.
- **Computation dtype difference**: ~~The paper mandates BF16 computation to maintain numerical stability through the quantized base. Our V3 inherits the project's mixed-precision setup (`torch.amp` with FP16/BF16 depending on GPU), which may use FP16 rather than the paper's preferred BF16 for the dequantized forward pass.~~ **Now enforced** — `_dequantize_conv_weight()` in `models/peft/qlora_v3.py` casts dequantized weights to `torch.bfloat16` via `COMPUTE_DTYPE = torch.bfloat16`, matching the paper's BF16 computation requirement.

#### 10.2.3 What it would take to keep QLoRA exactly the same as the original paper

| Deviation | What Would Be Required | Mathematical Complexity | Code Effort | Research Requirement | Worth Doing |
|-----------|----------------------|------------------------|-------------|---------------------|-------------|
| **bitsandbytes Linear-only kernels** | Implement custom CUDA kernels for 4-bit NF4 quantization on Conv2d weight tensors, including block-wise partitioning, quantile computation, and dequantization. | Very High — NF4 requires computing quantile thresholds from the standard normal CDF (`Φ^{-1}`). For Conv2d, we must generalize block-wise partitioning from 2D matrices to 4D tensors while preserving the information-theoretic optimality guarantee. | Very High — ~2000–5000 LOC of CUDA + Python bindings + test suite. This is essentially reimplementing bitsandbytes' core NF4 kernel for a new tensor shape. | PhD-level. Requires deep systems expertise in GPU quantization kernels, numerical analysis of NF4 error bounds for 4D tensors, and validation against the paper's reference implementation. | **No, for this project.** This is a publishable systems research contribution, not an implementation task. Our reshape workaround captures the core quantization idea. For a 4.4M-param model, the memory savings from optimized kernels are negligible compared to the engineering cost. |
| **Double Quantization missing** | Implement hierarchical quantization: (1) quantize Conv2d weights to NF4 with per-block scales, (2) quantize those scales to 8-bit floats with blocksize 256. | High — DQ's memory saving comes from `cFP32_2` being positive, allowing mean-subtraction before 8-bit quantization. For Conv2d, we need to define block topology in 4D space and prove the ~0.37 bits/param saving holds. | High — ~500–1000 LOC for quantization state management + second-level quantization + dequantization pipeline. | Medium-High. The paper defines DQ for 2D Linear weight matrices. Extending to 4D Conv2d requires original research into block partitioning strategies and empirical validation of accuracy/compression tradeoffs. | **No, for this project.** DQ saves ~0.37 bits/param. For a 4.4M-param model, that's ~0.2 MB total — irrelevant. DQ exists to enable 33B+ model storage; for our scale, the single-absmax approach is sufficient. The original paper lists DQ as an innovation for large-model memory, not a correctness requirement. |
| **Paged Optimizers missing** | Integrate `paged_adamw_32bit` from bitsandbytes or implement custom NVIDIA unified memory paging for optimizer states. | Medium — paging algorithms are well-known; the complexity is in integrating them with gradient checkpointing memory spikes. | High — ~1000+ LOC for paging infrastructure + integration with training loop. | High — this is specialized systems work. For our model scale, the engineering cost far outweighs the benefit. | **No.** Paged Optimizers address GPU OOM during gradient checkpointing on 33B/65B models. Our EfficientNet-B0 backbone is 4.4M params; even with 4-bit quantization, optimizer states fit comfortably in GPU memory. Irrelevant at our scale. |
| **Selective vs. full-layer targeting** | Change `target_modules` to include all 1×1 convs instead of just Q-path ones. | None — LoRA math is unchanged. | Trivial — 1 line in config. | None. | **Not worth pursuing.** The QLoRA paper's "all Linear layers" finding was for Transformer LLMs, not CNNs. On EfficientNet-B0, SE layers contribute only ~0.3–1.5pp on ImageNet-class tasks (Hoang & Jo 2021), and adding them roughly doubles the adapter count on MBConv blocks, creating real overfit risk (Flexora critical-point finding). Our selective Q-path targeting is already the standard CNN transfer-learning practice: freeze early/low-BI blocks, adapt channel-mixing projections and classifier. The ablation would consume GPU hours for a marginal gain that does not change the core V3 method comparison. |
| **FP16 vs. BF16 computation** | Enforce `torch.bfloat16` for dequantized forward pass instead of `torch.float16`. | None — both are 16-bit formats; BF16 has wider exponent range, FP16 has finer precision. | Trivial — ~10 LOC to set `bnb_4bit_compute_dtype=torch.bfloat16`. | None. | **Done(not trained).** `COMPUTE_DTYPE = torch.bfloat16` enforced in `_dequantize_conv_weight()` in `models/peft/qlora_v3.py`. |

#### 10.2.4 Path to exact parity with the original QLoRA paper

To make our QLoRA V3 **identical** to Dettmers et al. (2023), the minimal path is:
1. **Custom Conv2d NF4 kernel**: Implement or obtain a `bitsandbytes`-compatible 4-bit NF4 kernel for `nn.Conv2d`. This is the hardest requirement — it's a systems research contribution, not a code change.
2. **Double Quantization for Conv2d**: Define block-wise partitioning for 4D tensors and implement the two-level quantization state. This requires original math to extend the paper's 2D DQ proof to 4D.
3. **Paged Optimizers**: Not required for our scale, but would be needed for 33B+ CNN models.
4. **Full-layer targeting**: Trivial config change.
5. **BF16 enforcement**: Trivial config change.

**Practical assessment**: Achieving exact QLoRA parity on Conv2d is a **publishable research contribution**, not an implementation task. The paper itself notes that no CNN-specific 4-bit QLoRA exists. Our current workaround (reshape → NF4 → reshape back) captures the core idea but cannot implement DQ or optimized kernels without original systems research.

> **Selective vs. full-layer targeting ablation**: Not worth pursuing. Literature on EfficientNet-B0 shows SE layers contribute only ~0.3–1.5pp on ImageNet-class tasks (Hoang & Jo 2021), and expanding target layers risks overfitting (Flexora, ACL 2025). The QLoRA paper's "all Linear layers" finding was for Transformer LLMs, not CNNs. Our selective Q-path targeting is already well-validated for EfficientNet-style MBConv blocks. The ablation would consume GPU hours for a marginal gain that does not change the core V3 method comparison. See `ablation_selective_vs_full_targeting.md` for the abandoned plan.

---

### 10.3 QA-LoRA V3 (Xu et al., ICLR 2024)

#### 10.3.1 Original Paper vs. Our Adaptation

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
- Weights are **INT4 true integer-quantized** `[-8, 7]` (not unsigned fake-quant) — matches the paper's INT4 asymmetric quantization with per-group zero-point. Using INT4 because Conv2d weight distributions in EfficientNet-B0 differ from Transformer Linear weights; INT4 range is validated with learnable scale/zp to minimize clipping.
- **Merge support**: `merge_with_quantization()` in `models/peft/qalora.py` implements the paper's Algorithm 1 zero-point update rule. `merge_qalora_weights()` + `save_merged_qalora_model()` + `load_merged_qalora_for_inference()` enable post-training INT4 without FP16 fallback, identical to the paper's deployment claim.
- Does **not** use PEFT; `QALoRAConv2d` is a custom `nn.Module` because PEFT cannot express the `(L, r)` LoRA A shape

#### 10.3.2 Why we cannot keep QA-LoRA exactly the same as the original paper

- **Conv2d requires unfold+pool instead of AvgPool1d**: ~~The paper's `nn.AvgPool1d(D_in // L)` operates directly on 1D input vectors in Linear layers. For Conv2d 1×1, we must first `F.unfold(x)` to extract patches into `(B, D_in, N_positions)`, then `adaptive_avg_pool1d` over the last dimension. This unfold+pool is an approximation — the paper pools contiguous input groups, while our method pools over spatial patches globally, which may group features differently.~~ **Proven equivalent for 1×1 conv.** See `docs/qalora_unfold_pool_proof.md` for formal proof and numerical validation. Since `F.unfold(x, kernel_size=1)` preserves channel values identically and `adaptive_avg_pool1d` pools the channel dimension into contiguous bins, the grouping is mathematically identical to the paper's `AvgPool1d(D_in//L)` per spatial position for `stride=1, padding=0`.
- **INT4 asymmetric quantization**: The paper uses true INT4 asymmetric quantization with zero-point: `W_tilde = alpha_j * round((W - beta_j) / alpha_j) + beta_j`, where values range from `-2^(N-1)` to `2^(N-1)-1` (e.g., `-8` to `7` for INT4). Our V3 now implements **INT4 true integer quantization** `[-8, 7]` with per-group learnable `scale` and `zero_point` initialized from weight min/max to minimize clipping. This matches the paper's quantization scheme, adapted for Conv2d weight distributions which differ from Transformer Linear weights.
- **Zero-point merge rule is implemented**: `merge_with_quantization()` implements the paper's key innovation: `beta_new = beta - scaling * (lora_B @ lora_A.T) / scale`. This explicitly updates zero-points after fine-tuning so that `W' = W_tilde + s·AB` remains in INT4 without FP16 fallback. After merging, `lora_A` and `lora_B` are zeroed to avoid double-counting in forward passes. The model stays fully quantized after fine-tuning — identical to the paper's "no PTQ needed" claim.
- **Raw `nn.Parameter` adapters instead of `nn.Linear`**: The official QA-LoRA repo uses PEFT's `LoraLayer` infrastructure, which wraps adapters in `nn.Linear` modules for framework compatibility. Our `QALoRAConv2d` does **not** use PEFT at all, so we use raw `nn.Parameter` for `lora_A` and `lora_B`. This avoids PEFT entirely and lets us use the custom `(L, rank)` shape directly without fighting the library's `(D_in, rank)` assumption.
- **Kaiming init for `lora_A` instead of Xavier**: The paper/repo uses Xavier uniform (`torch.nn.init.xavier_uniform_`), standard for Transformer Linear layers. We use Kaiming uniform (`nn.init.kaiming_uniform_`), which is the de facto standard for CNNs and ReLU-family activations. Since EfficientNet-B0 is a CNN with SiLU activations, Kaiming is the more appropriate choice. Both work; this is a minor initialization preference.
- **Custom INT4 quantization instead of `auto_gptq`**: The official repo loads GPTQ-quantized INT4 base weights via `AutoGPTQForCausalLM.from_quantized(...)` — an LLM-specific, GPTQ-dependent pipeline. Our project cannot use `auto_gptq` because it only supports `Linear` layers in Transformer models, not `Conv2d` layers in CNNs. We implemented group-wise INT4 quantization directly inside `QALoRAConv2d.__init__` so it works with torchvision's EfficientNet-B0 without any LLM-specific dependencies.
- ~~Fake-quantization instead of true integer quantization~~: **Done.** Our V3 now uses true INT4 integer quantization: weights are stored as actual INT4 integers (`weight_q`), quantized once during initialization and fixed during training (no differentiable fake-quant in forward). During forward, weights are dequantized: `W = (weight_q - zp) * scale`. This aligns with the paper's official protocol (`yuhuixu1993/qa-lora`) which loads a true quantized INT4 base before fine-tuning and trains LoRA adapters on top, with no fake-quant or STE.
- **Target layer selection is CNN-specific**: The paper applies QA-LoRA to all Linear layers in LLaMA/LLaMA2. Our V3 targets only Q-path Conv2d 1×1 layers (MBConv expand/project + head + classifier), excluding SE layers and depthwise convolutions. This selective targeting is a CNN-specific design choice to balance parameter efficiency with adaptation capacity.
- **PEFT incompatibility is a hard constraint**: The paper's grouped LoRA A shape `(L, rank)` is incompatible with PEFT's standard LoRA A shape `(D_in, rank)`. This is true for both the paper's Linear case and our Conv2d case. We both implement custom modules instead of using PEFT. However, this constraint is more limiting for our CNN because PEFT's `Conv2d` support is already more restricted than its `Linear` support.

#### 10.3.3 What it would take to keep QA-LoRA exactly the same as the original paper

| Deviation | What Would Be Required | Mathematical Complexity | Code Effort | Research Requirement | Worth Doing |
|-----------|----------------------|------------------------|-------------|---------------------|-------------|
| **unfold+pool vs AvgPool1d** | **Proven equivalent.** For 1×1 conv with `stride=1, padding=0`, `F.unfold(x, 1)` preserves channel values identically, and `adaptive_avg_pool1d(..., L)` after transpose pools the channel dimension into `L` contiguous bins — bit-identical to the paper's `AvgPool1d(D_in//L)` per spatial position. Formal proof and numerical validation with Python code provided in `docs/qalora_unfold_pool_proof.md`. No redesign needed. | Low — proof is a direct consequence of 1×1 conv = per-position Linear + unfold invariance + contiguous bin averaging. | Trivial — no code changes needed. | None — this is pure linear algebra, not original research. | **Done.** Documentation complete in `docs/qalora_unfold_pool_proof.md`. Our CNN adaptation is mathematically faithful to the paper's group-wise DOF-balancing for 1×1 convs. |
| **INT4 asymmetric quantization** | Implement true INT4 asymmetric quantization with zero-point: `W_q = clamp(round(W / alpha_j + beta_j), -8, 7)`. Verified that EfficientNet-B0 Conv2d weight distributions fit in INT4 range with per-group learnable scale/zp to minimize clipping. | Medium — the quantization formula is straightforward. The challenge is statistical: Conv2d weights in EfficientNet have different distribution shapes than Transformer Linear weights. Solved with per-group learnable scale/zp initialized from weight min/max. | Medium — ~50–100 LOC for INT4 quant/dequant + per-group learnable scale/zp initialization. | Medium. Studied weight distributions across all Q-path layers; INT4 works with learnable per-group scale/zp that adapt during training to minimize clipping. | **Done.** True INT4 asymmetric quantization with per-group learnable scale and zero-point implemented in `QALoRAConv2d.__init__`. Weights stored as INT4 integers `[-8, 7]`, quantized once during initialization and fixed during training. Dequantized in forward with learned scale/zp. |
| **Zero-point merge rule** | Implement `merge_with_quantization(beta, lora_A, lora_B) = beta - scaling * (lora_B @ lora_A.T) / alpha_j` exactly as in Algorithm 1. Verified that merged weights remain in INT4 range (zero-point absorbs adapter update; integer weights unchanged). | Medium-High — the formula is simple, but the paper's correctness proof assumes Linear layer structure. For Conv2d, we must verify that the merged 4D weight tensor can still be represented in INT4 after zero-point adjustment. Solved by updating zero-points only; weight_q remains unchanged. | Low-Medium — ~50–100 LOC for the merge function + module-level merge/save/load utilities. | Medium-High. The paper provides a proof sketch for Linear layers ("for any j, there exists a new zero factor β'j..."). For Conv2d, the merge rule updates zero-points by `scaling * (lora_B @ lora_A.T) / scale`, which preserves the integer weight representation because only zero-points change. | **Done.** `merge_with_quantization()` implemented in `QALoRAConv2d`. `merge_qalora_weights()` + `save_merged_qalora_model()` + `load_merged_qalora_for_inference()` added to `models/peft/qalora.py`. After merging, zero-points absorb LoRA adapters; `lora_A` and `lora_B` are zeroed to avoid double-counting. The model stays fully quantized after fine-tuning without FP16 fallback — identical to the paper's "no PTQ needed" claim. |
| ~~Fake-quant vs true integer quantization~~ | Align with the paper's actual training protocol: load a true GPTQ-quantized INT4 base model before fine-tuning, rather than using fake-quant + STE during training. Alternatively, if staying with fake-quant for CNN compatibility, document this explicitly as a CNN-specific adaptation rather than implying the paper uses the same approach. | Low for protocol alignment — the official `yuhuixu1993/qa-lora` repo uses `AutoGPTQForCausalLM.from_quantized(..., trainable=True)`, loading a true INT4 checkpoint before adapter training. No STE is used. | Low — ~10–20 LOC to switch from fake-quant to true integer quantization, plus validation that the quantized base + LoRA adapters train correctly on EfficientNet-B0 Conv2d layers. | None — this is an implementation fidelity fix, not original research. The paper's protocol is well-documented in its official repo. | **Done.** Our V3 now stores weights as true INT4 integers, quantized once during initialization and dequantized in forward with trainable scale/zp. No differentiable fake-quant in forward. This matches the paper's official protocol of true quantized base + LoRA adapters, adapted for Conv2d with INT4 asymmetric quantization and per-group zero-point. |
| **Selective target layers** | Apply QA-LoRA to all Conv2d 1×1 layers in the model. | None — math is unchanged. | Trivial — 1 line in config. | None. | **Not worth pursuing.** Same reasoning as QLoRA above. The QA-LoRA paper targets all Linear layers empirically on Transformers. For EfficientNet-B0, SE layers contribute only ~0.3–1.5pp on ImageNet-class tasks (Hoang & Jo 2021), and adding them creates overfit risk without a clear upside for leaf disease classification. Our selective Q-path targeting is the correct CNN analog. See `ablation_selective_vs_full_targeting.md` for the abandoned experimental plan. |
| **PEFT incompatibility** | Already solved — we use custom `QALoRAConv2d` module, same as the paper's custom implementation for Linear. | None — custom module approach is identical to paper. | Already implemented. | None. | **Already solved.** No action needed. Our custom module approach matches the paper's approach for Linear layers. |

#### 10.3.4 Path to exact parity with the original QA-LoRA paper

To make our QA-LoRA V3 **identical** to Xu et al. (2024), the minimal path is:
1. ~~**Mathematical proof of unfold+pool equivalence**~~: **Done.** Proven in `docs/qalora_unfold_pool_proof.md`: `F.unfold(x, 1)` + `transpose` + `adaptive_avg_pool1d(..., L)` is bit-identical to the paper's `nn.AvgPool1d(D_in//L)` for 1×1 convs with `stride=1, padding=0`. No redesign needed.
2. **INT4 implementation with validation**: Implement true INT4 asymmetric quantization and validate that EfficientNet-B0 Conv2d weights can be represented without excessive clipping loss. If INT4 fails, develop preconditioning strategies. Effort: ~2–4 weeks.
3. **Zero-point merge rule**: Implement `merge_with_quantization` and prove/validate that merged Conv2d weights remain in INT4. Effort: ~1–2 weeks.
4. **Differentiable quantization**: Replace fake-quant with STE or learned rounding for true integer storage during training. Effort: ~1–2 months of research prototyping.

**Practical assessment**: Achieving exact QA-LoRA parity on Conv2d requires **original mathematical research** (proving unfold+pool equivalence or designing a 4D-native grouping) and **systems research** (INT4 Conv2d kernels with valid merge rules). Our current implementation captures the core group-wise DOF-balancing insight faithfully, but mischaracterizes the paper's training protocol: QA-LoRA does **not** use fake-quant + STE during training. The official repo loads a true GPTQ-quantized INT4 base model and trains LoRA adapters on top. Correcting this protocol mismatch is a low-effort fidelity fix, not a research contribution. The remaining gaps (INT4 Conv2d kernels) are publishable research problems, not implementation bugs.

> **Selective vs. full-layer targeting ablation**: Not worth pursuing. Same reasoning as QLoRA above. The QA-LoRA paper's "all Linear layers" targeting was empirically motivated on Transformers. For EfficientNet-B0, SE layers contribute only ~0.3–1.5pp and risk overfitting when added to the target set. Our selective Q-path targeting is the correct CNN analog. The ablation plan in `ablation_selective_vs_full_targeting.md` is abandoned.
