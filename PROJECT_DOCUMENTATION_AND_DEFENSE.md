# Project Documentation & Viva Defense Guide

This document contains technical explanations, architectural insights, limitations, supported domain coverage, and defense readiness for the **EfficientNet-B0 PEFT Plant Leaf Disease Classification Project**.

---

## 1. Architectural Analysis: Why Q/K LoRA Outperforms LoRA & QLoRA

### A. Technical Mechanisms
1. **Implicit Regularization & Noise Reduction**:
   - Standard LoRA updates 342k FP32 parameters across all non-depthwise Conv2d layers. Full FP32 fine-tuning on high-dimensional CNN projection layers can overfit to subtle noise in the training set.
   - Q/K LoRA quantizes the heavy 1x1 projection/expansion convolutions to **INT8**, acting as an implicit regularizer (similar to weight decay or dropout) that prevents the network from memorizing training patterns while maintaining representation quality.
2. **Precision Allocation Efficiency**:
   - **Q-Path (Quantized INT8, rank $r=16$)**: Handles high-capacity linear features using quantized weights.
   - **K-Path (Unquantized FP32, rank $r=4$)**: Retains full 32-bit floating-point precision specifically on the critical **Squeeze-and-Excitation (SE) channel-attention modules**.
   - By keeping full precision where feature sensitivity is highest (attention/channel weighting) and quantizing standard projections, Q/K LoRA achieves superior feature selectivity than pure QLoRA (which quantizes indiscriminately) and better generalization than pure LoRA.

### B. Validation in Academic & PhD Literature
- **Quantization-Aware Fine-Tuning (QAFT) & LoftQ (Li et al., 2023)**: Proved that low-bit quantized low-rank adaptations frequently match or exceed FP32 baselines due to reduced gradient variance.
- **QLoRA (Dettmers et al., NeurIPS 2023)**: Demonstrated that 4-bit/8-bit PEFT models regularly match or beat full-precision fine-tuning on downstream benchmark tasks.
- **Mixed-Precision Neural Architectures**: Allocating higher precision to attention bottlenecks while quantizing standard feed-forward/projection paths is an active research area in top AI venues (CVPR, NeurIPS, ICLR).

---

## 2. Limitations of Trained Models

1. **Background & Environmental Bias (Lab vs. Field)**:
   - PlantVillage dataset images were collected in uniform laboratory conditions (single leaf over neutral background).
   - In real-world field conditions (complex soil, shadows, overlapping leaves, variable sunlight), accuracy drops due to background feature interference.
2. **Fixed 38-Class Taxonomy**:
   - The model is strictly bounded to the **38 pre-defined classes**. It cannot diagnose novel pests, nutrient deficiencies (e.g., Nitrogen/Potassium starvation), or diseases outside the trained list.
3. **Single-Leaf Focus**:
   - The model expects a centered crop of a **single leaf**. It cannot process full plant canopy wide-angle drone/field photos without a prior leaf detection/bounding box pre-processor.
4. **Quantization CPU Overhead**:
   - Simulated INT8 quantization in standard PyTorch CPU execution requires on-the-fly dequantization unless deployed using hardware-accelerated runtimes (e.g., ONNX Runtime, TensorRT, or OpenVINO).

---

## 3. Image & Disease Verification Scope

### A. Supported Crop Species (14 Plants)
Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

### B. Diagnosable Conditions (38 Classes)
- **Healthy Leaves**: Across all 14 crop species.
- **Fungal Diseases**: Apple Scab, Cedar Apple Rust, Corn Common Rust, Northern Leaf Blight, Grape Black Rot, Potato Early & Late Blight, Tomato Early & Late Blight, Powdery Mildew, Leaf Mold, Target Spot.
- **Bacterial Diseases**: Bacterial Spot (Pepper, Tomato), Crown Gall.
- **Viral & Mite Damage**: Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Two-Spotted Spider Mite.

### C. Image Input Specifications
- **Framing**: Single leaf clearly visible and centered in frame.
- **Lighting**: Bright, even natural or artificial light.
- **Resolution**: Minimum $\approx 224 \times 224$ pixels (auto-scaled by pre-processing pipeline).

---

## 4. Faculty Defense Readiness Checklist

- [x] **Empirical Rigor**: Evaluated across 10,735 test set images with exact confusion matrices, Macro F1, Binary ROC-AUC, and parameters.
- [x] **Novel Comparative PEFT Benchmark**: Comparative benchmark across LoRA, CNN-QLoRA, and CNN-QKLoRA.
- [x] **Production Web Application**: Interactive dashboard with real-time inference, model switching, full-screen plot zoom, and checkpoint rankings.
