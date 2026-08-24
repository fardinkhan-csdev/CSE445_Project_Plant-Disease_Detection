# Project Documentation & Viva Defense Guide

This document contains technical explanations, architectural insights, limitations, supported domain coverage, and defense readiness for the **EfficientNet-B0 PEFT Plant Leaf Disease Classification Project**.

---

## 1. Methods Compared

### 1.1 LoRA (Low-Rank Adaptation)
- **Paper**: Hu et al. 2021 (ICLR)
- **Mechanism**: Freeze pretrained backbone. Insert low-rank adapter matrices A and B into selected layers. Update only adapters.
- **Targets**: All non-depthwise Conv2d (1×1 pointwise) + classifier.fc
- **Trainable params**: ~193k (~8% of total)
- **Result**: 98.89% test accuracy, 18 MB checkpoint

### 1.2 QLoRA (Quantized LoRA)
- **Paper**: Dettmers et al. 2023 (NeurIPS)
- **Mechanism**: bitsandbytes NF4 (4-bit NormalFloat) weight-only quantization on Q-path 1×1 convs. Frozen INT4 backbone + FP16 LoRA adapters.
- **Targets**: Q-path pointwise convs + features.8.0 + classifier.fc
- **Trainable params**: ~193k
- **Result**: 98.91% test accuracy, 7.6 MB checkpoint

### 1.3 QA-LoRA (Quantization-Aware LoRA)
- **Paper**: Xu et al. 2024 (ICLR)
- **Mechanism**: Group-wise INT8 quantization with learned scale/zero-point per group. LoRA A is grouped to `(L, rank)` instead of `(D_in, rank)`. Does not use PEFT — `QALoRAConv2d` replaces `nn.Conv2d` directly.
- **Targets**: Q-path pointwise convs + features.8.0 + classifier.fc
- **Trainable params**: ~242k
- **Result**: **99.52% test accuracy**, 9.5 MB checkpoint — **best overall method**

### 1.4 Q/K LoRA (Quantized/Kept LoRA)
- **Custom design** — not from a specific paper
- **Mechanism**: Selective precision allocation. Q-path (1×1 convs) quantized to INT8 with rank-16 LoRA. K-path (SE layers + classifier) kept in FP32 with rank-4 LoRA.
- **Q/K naming**: Q = Quantized path, K = Kept high-precision path (not transformer Query/Key)
- **Trainable params**: ~445k
- **Result**: 99.22% test accuracy, 12.1 MB checkpoint

### Why QA-LoRA Outperforms

1. **Implicit Regularization**: INT8 quantization acts as implicit regularizer, preventing overfitting while maintaining representation quality.
2. **Precision Allocation**: Group-wise quantization provides more quantization degrees of freedom than per-channel. The grouped LoRA A balances the increased quantization DOF.
3. **True Integer Base**: QA-LoRA uses true INT8 integer weights (not fake-quant), matching the paper's deployment protocol exactly.

---

## 2. Architectural Analysis

### EfficientNet-B0 MBConv Blocks
Each MBConv block contains:
1. **Expand Conv**: 1×1 pointwise (channel expansion)
2. **Depthwise Conv**: 3×3 (spatial features)
3. **Squeeze-and-Excitation (SE)**: Channel attention (fc1 → ReLU → fc2 → sigmoid)
4. **Project Conv**: 1×1 pointwise (channel reduction)

### Why Target Pointwise Convs
- 1×1 convolutions are mathematically equivalent to Linear layers applied per spatial location
- They contain the majority of MBConv parameters (channel projection)
- The low-rank decomposition principle from LoRA transfers directly

### Why Exclude Depthwise Convs
- PEFT cannot attach LoRA adapters to grouped convolutions (`groups > 1`)
- They contain comparatively few parameters
- They are spatial feature extractors, not channel-mixing projections

### Why Exclude SE Layers (in most methods)
- SE layers are attention mechanisms, not projection layers
- Q/K LoRA is the exception: it keeps SE in FP32 with a separate LoRA path

---

## 3. Limitations of Trained Models

1. **Background & Environmental Bias (Lab vs. Field)**
   - PlantVillage images: uniform lab conditions, single leaf, neutral background
   - Real-world field conditions: complex soil, shadows, overlapping leaves, variable sunlight
   - Accuracy drops on PlantDoc due to background feature interference

2. **Fixed 38-Class Taxonomy**
   - Cannot diagnose novel pests, nutrient deficiencies, or diseases outside the trained list

3. **Single-Leaf Focus**
   - Expects a centered crop of a single leaf
   - Cannot process full plant canopy or wide-angle field photos without leaf detection pre-processing

4. **Quantization CPU Overhead**
   - INT8/NF4 quantization in standard PyTorch requires on-the-fly dequantization
   - For production deployment, use ONNX Runtime, TensorRT, or OpenVINO

---

## 4. Supported Crop Species (14 Plants)

Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

## 5. Diagnosable Conditions (38 Classes)

- **Healthy Leaves**: Across all 14 crop species
- **Fungal Diseases**: Apple Scab, Cedar Apple Rust, Corn Common Rust, Northern Leaf Blight, Potato Early & Late Blight, Tomato Early & Late Blight, Powdery Mildew, Leaf Mold, Target Spot
- **Bacterial Diseases**: Bacterial Spot (Pepper, Tomato)
- **Viral & Mite Damage**: Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Two-Spotted Spider Mite

## 6. Image Input Specifications

- **Framing**: Single leaf clearly visible and centered in frame
- **Lighting**: Bright, even natural or artificial light
- **Resolution**: Minimum 224×224 pixels (auto-scaled by pre-processing pipeline)

---

## 7. Current Results Summary

### PlantVillage Test Set (10,735 images)

| Method | Accuracy | F1 Macro | Binary F1 | ROC AUC | Params | Size |
|--------|----------|----------|-----------|---------|--------|------|
| QA-LoRA | **99.52%** | 0.993 | 1.000 | 1.000 | 242k | 9.5 MB |
| Q/K LoRA | 99.22% | 0.987 | 0.999 | 1.000 | 445k | 12.1 MB |
| QLoRA | 98.91% | 0.984 | 0.999 | 1.000 | 193k | 7.6 MB |
| LoRA | 98.89% | 0.984 | 0.999 | 1.000 | 193k | 18.1 MB |

### PlantDoc Transfer Evaluation

Real-world field image evaluation using 25 shared label mappings between PlantVillage and PlantDoc. Multiple strategies evaluated:
- Dual-split evaluation (train + test)
- Segmented image evaluation
- Ensemble methods
- Style normalization (CORAL, etc.)

---

## 8. Faculty Defense Readiness Checklist

- [x] **Empirical Rigor**: Evaluated across 10,735 test images with confusion matrices, Macro F1, Binary ROC-AUC, and parameter counts
- [x] **Novel Comparative PEFT Benchmark**: Four methods compared (LoRA, QLoRA, QA-LoRA, Q/K LoRA) — first known CNN-PEFT comparison including QA-LoRA
- [x] **Production Web Application**: Interactive dashboard with real-time inference, model switching, full-screen plot zoom, and checkpoint rankings
- [x] **Transfer Evaluation**: PlantDoc real-world field evaluation with label alignment strategy
- [x] **Reproducibility**: All configs in YAML, all splits deterministic, all results saved to CSV

---

## 9. Key Files for Defense

| File | Purpose |
|------|---------|
| `architecture_design_v3.md` | Full technical architecture (V3 methods) |
| `FINAL_EXPERIMENT_PLAN.md` | Experiment execution plan (all phases complete) |
| `experiments/results/eval/cross_method_ranking.csv` | Final 4-method comparison |
| `experiments/results/experiment_results.csv` | Summary metrics per method |
| `experiments/results/dashboard.html` | Self-contained visual dashboard |
| `web_app/server.py` + `web_app/static/` | Interactive web UI |
| `config/class_labels.json` | 38-class label map for inference |
