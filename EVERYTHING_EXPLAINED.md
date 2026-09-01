# Everything Explained — From Zero to Understanding

This document explains **every concept** used in this project, written for someone with zero prior knowledge. Read it top to bottom, or jump to any section.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Problem: Plant Diseases](#2-the-problem-plant-diseases)
3. [Datasets](#3-datasets)
4. [Image Classification](#4-image-classification)
5. [Convolutional Neural Networks (CNNs)](#5-convolutional-neural-networks-cnns)
6. [EfficientNet-B0](#6-efficientnet-b0)
7. [Transfer Learning & Pretraining](#7-transfer-learning--pretraining)
8. [Fine-Tuning vs PEFT](#8-fine-tuning-vs-peft)
9. [LoRA — Low-Rank Adaptation](#9-lora--low-rank-adaptation)
10. [Quantization](#10-quantization)
11. [QLoRA — Quantized LoRA](#11-qlora--quantized-lora)
12. [QA-LoRA — Quantization-Aware LoRA](#12-qa-lora--quantization-aware-lora)
13. [Q/K LoRA — Quantized/Kept LoRA](#13-qk-lora--quantizedkept-lora)
14. [MBConv Blocks — The Building Blocks](#14-mbconv-blocks--the-building-blocks)
15. [1×1 Convolutions (Pointwise)](#15-11-convolutions-pointwise)
16. [Depthwise Convolutions](#16-depthwise-convolutions)
17. [Squeeze-and-Excitation (SE)](#17-squeeze-and-excitation-se)
18. [Data Pipeline — Splits, Transforms, Loading](#18-data-pipeline--splits-transforms-loading)
19. [Training — Loss, Optimizer, Scheduler](#19-training--loss-optimizer-scheduler)
20. [Mixed Precision Training](#20-mixed-precision-training)
21. [Early Stopping](#21-early-stopping)
22. [Checkpoints](#22-checkpoints)
23. [Evaluation Metrics](#23-evaluation-metrics)
24. [Confusion Matrix](#24-confusion-matrix)
25. [Crop/Disease Correctness Breakdown](#25-cropdisease-correctness-breakdown)
26. [Transfer Learning to PlantDoc](#26-transfer-learning-to-plantdoc)
27. [Domain Adaptation Techniques](#27-domain-adaptation-techniques)
28. [Cross-Method Ranking](#28-cross-method-ranking)
29. [Web UI & Inference](#29-web-ui--inference)
30. [Dashboard](#30-dashboard)
31. [Key Libraries](#31-key-libraries)
32. [Glossary](#32-glossary)

---

## 1. The Big Picture

This project answers one question:

> **Can we fine-tune a pretrained image classifier to identify plant diseases using only a tiny fraction of the model's parameters?**

Instead of retraining the entire model (millions of parameters), we use **Parameter-Efficient Fine-Tuning (PEFT)** methods that update only 3–8% of parameters while achieving 99%+ accuracy.

We compare **four** such methods on the same backbone, dataset, and evaluation protocol.

---

## 2. The Problem: Plant Diseases

Plant diseases cause **up to 40% of global crop losses** annually. Early detection is critical for food security.

**Current approaches:**
- **Manual inspection**: Experts visually examine leaves. Slow, expensive, requires trained agronomists.
- **Lab testing**: Sending samples to a lab. Too slow for field use.
- **AI-based detection**: Train a neural network on leaf images to automatically classify diseases.

This project uses the AI approach. The model looks at a photo of a single leaf and predicts:
- **What crop** it is (tomato, potato, apple, etc.)
- **What disease** it has (or if it's healthy)

---

## 3. Datasets

### 3.1 PlantVillage (Training Dataset)

**What it is**: A collection of **54,306 RGB photos** of single plant leaves, taken in controlled laboratory conditions.

**Key facts:**
- **38 classes** (e.g., `Tomato___Late_blight`, `Apple___healthy`)
- **14 crop species**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- Each photo shows **one leaf**, centered, against a plain background (gray or black)
- The class name follows the pattern `Crop___Disease` (three underscores)

**Why this dataset?**
- It's the standard benchmark for plant disease classification
- Large enough for training (54K images)
- Well-labeled with clear class boundaries

**Limitation**: All photos are taken in labs. Real-world field photos look very different (different lighting, backgrounds, camera angles).

### 3.2 PlantDoc (Transfer Test Dataset)

**What it is**: A collection of **~2,500 real-world photos** of plant leaves taken in the field.

**Key facts:**
- Photos taken outdoors with natural lighting, soil, shadows, and other plants visible
- Different camera quality and angles
- Different label taxonomy than PlantVillage

**Why this dataset?**
- Tests whether the model generalizes beyond lab conditions
- Represents real-world deployment scenarios
- We use **25 shared label mappings** between PlantVillage and PlantDoc (e.g., `Tomato___Late_blight` → `Tomato leaf late blight`)

**The challenge**: A model trained on clean lab photos will struggle on messy field photos. This is called the **domain gap**.

---

## 4. Image Classification

**What it is**: Given an image, assign it a single label from a fixed set of categories.

**How it works:**
1. Feed an image (224×224 pixels, 3 color channels: RGB) into a neural network
2. The network outputs a **probability distribution** over 38 classes
3. The class with the highest probability is the prediction

**Example:**
```
Input: Photo of a tomato leaf with brown spots
Output: [0.01, 0.02, ..., 0.94, ..., 0.01]
                  ↑
         Tomato___Late_blight (94% confidence)
```

**Not object detection**: Our model classifies the entire image as one class. It does not draw bounding boxes around leaves. (PlantDoc is an object detection dataset, but we crop individual leaves first.)

---

## 5. Convolutional Neural Networks (CNNs)

**What they are**: Neural networks designed for grid-like data (images). They use **convolutions** — sliding small filters over the image to detect patterns.

**Key concepts:**

### Convolution
A small filter (e.g., 3×3) slides across the image, computing dot products at each position. Each filter detects a specific pattern (edge, texture, shape).

### Channels
- Input image: 3 channels (Red, Green, Blue)
- After convolution: many channels (e.g., 32, 64, 128). Each channel detects different features.

### Layers
A CNN stacks many convolution layers:
- **Early layers**: Detect simple patterns (edges, colors)
- **Middle layers**: Detect textures and shapes
- **Deep layers**: Detect complex objects (leaf spots, disease patterns)

### Parameters
Each convolution filter has weights. A 3×3 filter with 3 input channels and 64 output channels has `3 × 3 × 3 × 64 = 1,728` parameters. Modern CNNs have **millions** of parameters.

---

## 6. EfficientNet-B0

**What it is**: A specific CNN architecture designed for image classification. "B0" is the smallest variant.

**Key facts:**
- **Total parameters**: ~5.3 million
- **Input**: 224×224 RGB image
- **Output**: Feature vector of 1,280 dimensions → classification head → 38 classes
- **Pretrained**: Already trained on ImageNet (1.2M images, 1,000 classes). It already knows how to recognize edges, textures, shapes, and common objects.

**Why EfficientNet-B0?**
- **Efficient**: Good accuracy with fewer parameters than ResNet, VGG, etc.
- **Scalable**: Easy to scale up (B1, B2, ...) or down
- **MBConv blocks**: Uses modern mobile-inspired architecture
- **Well-studied**: Widely used in research, good pretrained weights available

**Architecture:**
```
Input (224×224×3)
  → Stem convolution (3×3)
  → MBConv Block 1 (16 channels)
  → MBConv Block 2 (24 channels)
  → MBConv Block 3 (40 channels)
  → MBConv Block 4 (80 channels)
  → MBConv Block 5 (112 channels)
  → MBConv Block 6 (192 channels)
  → MBConv Block 7 (320 channels)
  → Head convolution (1×1)
  → Global Average Pooling
  → Classifier (1280 → 38)
```

---

## 7. Transfer Learning & Pretraining

**The idea**: Instead of training a CNN from scratch (which needs millions of images), start with a model that already knows how to see.

**How it works:**
1. **Pretrain** on ImageNet (huge dataset, general visual features)
2. **Fine-tune** on PlantVillage (smaller dataset, specific task)

**Why this works:**
- Early layers learn universal features (edges, textures) that work for any image task
- Later layers learn task-specific features (disease patterns)
- We only need to adjust the later layers for our specific 38-class problem

**Analogy**: It's like teaching someone who already speaks 5 languages to learn a 6th. Much faster than teaching language from scratch.

---

## 8. Fine-Tuning vs PEFT

### Full Fine-Tuning
- Update **all** 5.3M parameters of EfficientNet-B0
- Requires lots of data and compute
- Risk of overfitting on small datasets

### Parameter-Efficient Fine-Tuning (PEFT)
- **Freeze** most of the backbone (keep pretrained weights unchanged)
- Add a small number of **trainable parameters** (adapters)
- Update only the adapters during training
- Result: 3–8% of parameters are trainable, but accuracy is comparable

**Why PEFT?**
- **Less overfitting**: Fewer trainable parameters = harder to memorize training data
- **Faster training**: Fewer parameters to update = less computation
- **Smaller checkpoints**: Only save the adapter weights (~10 MB vs ~120 MB)
- **Preserves pretrained knowledge**: The backbone stays frozen, retaining ImageNet features

---

## 9. LoRA — Low-Rank Adaptation

**Paper**: Hu et al. 2021 (ICLR)

**The core idea**: Instead of updating a weight matrix `W` directly, decompose the update as a product of two small matrices.

### Math

For a weight matrix `W` of shape `(out, in)`:
- **Full update**: `W_new = W + ΔW` where `ΔW` has `out × in` parameters
- **LoRA update**: `ΔW = B @ A` where:
  - `A` has shape `(rank, in)` — e.g., `(8, 96)`
  - `B` has shape `(out, rank)` — e.g., `(96, 8)`
  - Total parameters: `rank × (in + out)` instead of `in × out`

**Example**: For a 96×96 weight matrix:
- Full update: 96 × 96 = **9,216 parameters**
- LoRA with rank 8: 8 × (96 + 96) = **1,536 parameters** (6× fewer)

### How it's applied to CNNs

EfficientNet-B0 uses `Conv2d` layers. A 1×1 convolution with weight shape `(C_out, C_in, 1, 1)` is mathematically equivalent to a linear layer. LoRA inserts adapters:
- `A` shape: `(C_in, rank)`
- `B` shape: `(rank, C_out)`
- Forward: `output = conv2d(x, W0 + (B @ A).view(C_out, C_in, 1, 1))`

### Initialization
- `A`: Random Gaussian initialization
- `B`: Initialized to **zero**
- So at the start of training, `ΔW = B @ A = 0` — the model behaves exactly like the pretrained model

### Scaling
The adapter output is scaled by `alpha / rank`. With `rank=8` and `alpha=16`, the scale is `16/8 = 2.0`.

### In our project
- **Rank**: 8
- **Alpha**: 16
- **Dropout**: 0.1 (10% of adapter activations randomly zeroed during training)
- **Targets**: All non-depthwise Conv2d + classifier.fc
- **Trainable params**: ~193k (3.6% of backbone)

### Merge Support (V3)
After training, LoRA weights can be **merged** into the base weights:
```
W_merged = W0 + (B @ A)
```
This gives zero-overhead inference — no adapter overhead at deployment.

---

## 10. Quantization

**What it is**: Reducing the precision of model weights from 32-bit floating point (FP32) to lower-bit representations (INT8, INT4, NF4).

### Why quantize?
- **Memory**: FP32 uses 4 bytes per weight. INT8 uses 1 byte. INT4 uses 0.5 bytes.
- **Speed**: Integer operations are faster than floating-point on modern hardware
- **Regularization**: Lower precision acts as noise, preventing overfitting

### Types of Quantization

#### Per-Channel Quantization
- Each output channel gets its own scale and zero-point
- Weights are mapped to integers: `weight_q = round(weight / scale) + zero_point`
- During forward: `weight_dequant = (weight_q - zero_point) * scale`

#### Per-Group Quantization (Group-wise)
- Each output channel is split into `L` groups
- Each group gets its own scale and zero-point
- More parameters → more representational capacity
- This is what QA-LoRA uses

#### Weight-Only Quantization
- Only weights are quantized; activations stay in floating point
- During forward: dequantize weights temporarily, compute, discard
- This is what our project uses (not activation quantization)

### INT8 Quantization
- 8-bit integers: range -128 to 127 (or 0 to 255 unsigned)
- **256 discrete levels** instead of continuous FP32
- Used in our Q/K LoRA and QA-LoRA methods

### NF4 Quantization (NormalFloat 4-bit)
- 4-bit: only **16 discrete levels**
- Designed for normally-distributed weights (which neural network weights approximately are)
- Uses `bitsandbytes` library for implementation
- Used in our QLoRA V3 method

---

## 11. QLoRA — Quantized LoRA

**Paper**: Dettmers et al. 2023 (NeurIPS)

**The idea**: Combine 4-bit quantization with LoRA.

### How it works
1. **Quantize** the backbone weights to 4-bit NF4 using `bitsandbytes`
2. **Freeze** the quantized backbone
3. **Add LoRA adapters** on top
4. **Train** only the adapters (in FP16/BF16)
5. During forward pass: dequantize to BF16 → apply adapters → output

### Key innovation: NF4
- NormalFloat 4-bit is information-theoretically optimal for normally distributed weights
- It allocates more quantization levels where the weight distribution is denser
- Better than uniform INT4 for neural network weights

### In our project
- **Quantization**: bitsandbytes NF4 on Q-path 1×1 convs
- **Compute dtype**: bfloat16 (dequantized weights cast to BF16 during forward)
- **Targets**: Q-path pointwise convs + `features.8.0` + `classifier.fc`
- **Trainable params**: ~193k
- **Checkpoint**: ~7.6 MB (smallest of all methods)

### Why QLoRA works
- The quantized backbone provides a compressed feature extractor
- LoRA adapters compensate for the information lost during quantization
- The result: small model, fast inference, comparable accuracy

---

## 12. QA-LoRA — Quantization-Aware LoRA

**Paper**: Xu et al. 2024 (ICLR)

**The idea**: Balance quantization degrees of freedom with adaptation degrees of freedom.

### The Problem QA-LoRA Solves
Standard per-channel quantization has too few parameters relative to LoRA adapters. The quantizer can't represent the weight space well enough for the adapters to be effective.

### The Solution: Group-wise Operations

**Group-wise quantization:**
- Split each output channel's weights into `L` groups (e.g., L=4)
- Each group gets its own `scale` and `zero_point`
- More quantization parameters → better weight representation

**Grouped LoRA A:**
- Standard LoRA A has shape `(D_in, rank)` — e.g., `(96, 8)`
- QA-LoRA's grouped LoRA A has shape `(L, rank)` — e.g., `(4, 8)`
- Input is unfolded into groups, then pooled to `L` dimensions
- Fewer adapter parameters → balances the increased quantization parameters

### Algorithm 1 (from the paper)
```
1. Freeze backbone weight W (shape: C_out × C_in × 1 × 1)
2. Reshape to (C_out, L, C_in/L) — L groups per output channel
3. Learn scale and zero_point per group: (C_out, L)
4. Quantize: W_q = round(W / scale) + zero_point  (true INT4 integers, range [-8, 7])
5. During forward:
   a. Dequantize: W_dequant = (W_q - zp) * scale
   b. Compute grouped LoRA: unfold input → pool to L dims → multiply by lora_A (L, rank)
   c. Standard LoRA B: (C_out, rank)
   d. Output = conv2d(x, W_dequant) + lora_B @ lora_A_pooled
```

### Key difference from QLoRA
- QLoRA uses NF4 (4-bit, 16 levels) — very aggressive quantization
- QA-LoRA uses INT4 (4-bit, 16 levels, range [-8, 7]) with group-wise scaling — more aggressive but structured by per-group learned scale/zp
- QA-LoRA does **not** use PEFT library — it replaces `nn.Conv2d` directly with `QALoRAConv2d`

### In our project
- **Quantization**: Group-wise INT4 [-8, 7] (true integer base, frozen at init; learnable scale/zp used for forward dequant)
- **Groups (L)**: 4
- **Targets**: Q-path pointwise convs + `features.8.0` + `classifier.fc`
- **Trainable params**: ~242k
- **Checkpoint**: ~9.5 MB
- **Result**: **Best overall method** — 99.52% accuracy

---

## 13. Q/K LoRA — Quantized/Kept LoRA

**Custom design** (not from a specific paper)

**The idea**: Apply different precision and different LoRA ranks to different types of layers.

### Q-Path (Quantized)
- **Layers**: MBConv 1×1 pointwise convolutions
- **Precision**: INT4 quantization (range [-8, 7])
- **LoRA rank**: 16 (higher rank to compensate for quantization)
- **Rationale**: These layers handle channel projection and contain most parameters

### K-Path (Kept High-Precision)
- **Layers**: Squeeze-and-Excitation (SE) layers (`fc1`/`fc2`) + `classifier.fc`
- **Precision**: FP32 (no quantization)
- **LoRA rank**: 4 (lower rank, since precision is preserved)
- **Rationale**: SE layers are attention mechanisms — precision matters more here

### Why this works
- Different layer types contribute differently to feature extraction
- Quantizing projection layers (where capacity is high) is less harmful
- Keeping attention layers in full precision preserves sensitivity
- The naming: **Q** = Quantized, **K** = Kept (not transformer Query/Key)

### In our project
- **Trainable params**: ~445k (most of all methods, but still < 9% of backbone)
- **Checkpoint**: ~12.1 MB
- **Result**: 99.22% accuracy

---

## 14. MBConv Blocks — The Building Blocks

**MBConv** = Mobile Inverted Bottleneck Convolution. It's the core building block of EfficientNet.

### Structure of one MBConv block:
```
Input
  → 1. Expand Conv (1×1 pointwise): increase channels
  → 2. Depthwise Conv (3×3): extract spatial features
  → 3. Squeeze-and-Excitation: channel attention
  → 4. Project Conv (1×1 pointwise): reduce channels
  → Add residual connection (skip connection)
  → Output
```

### Why "Mobile"?
Originally designed for mobile phones (MobileNet). EfficientNet scaled this design up for better accuracy.

### Why "Inverted Bottleneck"?
Standard bottleneck: wide → narrow → wide. Inverted: narrow → wide → narrow. More efficient for mobile.

### Residual Connection
The input is added to the output: `output = block(input) + input`. This helps gradient flow and prevents vanishing gradients.

---

## 15. 1×1 Convolutions (Pointwise)

**What it is**: A convolution filter with kernel size 1×1. It mixes information across channels but doesn't change spatial dimensions.

**Example:**
- Input: feature map of shape `(96, H, W)` — 96 channels
- 1×1 conv with 144 output channels
- Output: feature map of shape `(144, H, W)` — 144 channels
- Spatial size (H, W) unchanged

**Why use them?**
- **Channel mixing**: Combine features from different channels
- **Dimensionality change**: Increase or decrease number of channels
- **Computationally cheap**: Only `C_in × C_out` parameters per spatial location
- **Mathematically equivalent to Linear**: A 1×1 conv is just a matrix multiply applied to each pixel

**Why LoRA targets them:**
- They behave like linear layers → LoRA's low-rank decomposition applies directly
- They contain most of the parameters in MBConv blocks
- Adapting them is the most efficient way to change model behavior

---

## 16. Depthwise Convolutions

**What it is**: A convolution where each input channel is processed independently by its own filter.

**Standard convolution**: Each output channel depends on ALL input channels.
**Depthwise convolution**: Each output channel depends on only ONE input channel.

**Example:**
- Input: 96 channels
- Depthwise conv with 96 filters (one per channel), each 3×3
- Output: 96 channels (same number)
- Parameters: 96 × 3 × 3 = 864 (vs. 96 × 96 × 3 × 3 = 82,944 for standard conv)

**Why use them?**
- **Much fewer parameters** than standard convolution
- **Preserves spatial information** without cross-channel mixing
- Cross-channel mixing is done separately by 1×1 convolutions

**Why LoRA can't target them:**
- Depthwise convolutions have `groups = C_in` (each channel is its own group)
- The PEFT library assumes `groups = 1` for LoRA injection
- Attaching adapters to grouped convolutions causes shape mismatches

---

## 17. Squeeze-and-Excitation (SE)

**What it is**: A channel attention mechanism that learns to weight channels by importance.

**How it works:**
```
Input (C channels)
  → Global Average Pooling → (C, 1, 1)   — "squeeze" spatial info
  → FC1 (C → C/4) → ReLU                 — "excitation" part 1
  → FC2 (C/4 → C) → Sigmoid              — "excitation" part 2
  → Multiply element-wise with input       — re-weight channels
```

**What it learns:**
- Which channels are important for the current input
- Gives higher weight to informative channels, lower weight to noise

**Why keep SE in FP32 (Q/K LoRA):**
- SE layers control information routing (like attention in transformers)
- Quantizing them would hurt the model's ability to focus on relevant features
- They have relatively few parameters (cheap to keep in full precision)

---

## 18. Data Pipeline — Splits, Transforms, Loading

### Split Strategy

```
PlantVillage (54,306 images)
  └→ HuggingFace official splits
       ├→ color/train (80%) → split by leaf_id
       │    ├→ Train (85%) = 68% of total
       │    └→ Val (15%) = 12% of total
       └→ color/test (20%) = 20% of total
```

**Why leaf_id grouping?**
- PlantVillage has multiple photos of the same physical leaf
- A naive random split would put similar images in both train and test
- This is called **data leakage** — the model "cheats" by recognizing the same leaf
- `leaf_id` ensures all photos of one leaf go to the same split

### Training Transforms

Applied to each training image **on the fly** (not pre-computed):

1. **Resize(256)**: Scale the shorter side to 256 pixels
2. **RandomCrop(224)**: Randomly crop a 224×224 patch
3. **RandomHorizontalFlip()**: 50% chance of horizontal mirror
4. **RandomRotation(±15°)**: Random rotation between -15° and +15°
5. **ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)**: Random color adjustments
6. **Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])**: ImageNet normalization

**Why augment?**
- Makes the model robust to variations in position, orientation, and color
- Prevents overfitting by showing different versions of the same image each epoch

### Validation/Test Transforms

Applied to each test image:
1. **Resize(256)**: Scale to 256 pixels
2. **CenterCrop(224)**: Crop the center 224×224 patch (deterministic, not random)
3. **Normalize**: Same as training

**Why different from training?**
- No randomness in test time — we want reproducible results
- Center crop is standard for ImageNet evaluation

### ImageNet Normalization

Raw pixel values are 0–255. After normalization:
```
pixel_normalized = (pixel / 255 - mean) / std
```
With ImageNet statistics: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`

**Why?**
- The pretrained EfficientNet-B0 was trained with these exact statistics
- Normalizing to the same distribution ensures the model receives familiar input

### DataLoader
- **Batch size**: 32 images per forward pass
- **Workers**: 4 parallel data-loading processes
- **Shuffle**: Training data shuffled each epoch; test data not shuffled

---

## 19. Training — Loss, Optimizer, Scheduler

### Loss Function: CrossEntropyLoss

**What it measures**: How far the model's prediction is from the true label.

**How it works:**
1. Model outputs raw scores (logits) for 38 classes
2. CrossEntropyLoss applies softmax to get probabilities
3. Computes `-log(probability of true class)`
4. Averages across the batch

**Lower loss = better predictions.**

**Label smoothing (optional)**: Instead of hard targets `[0, 0, 1, 0, ...]`, use soft targets `[0.0026, 0.0026, 0.974, 0.0026, ...]`. Prevents overconfidence.

### Optimizer: AdamW

**What it does**: Updates model parameters to minimize loss.

**Adam** = Adaptive Moment Estimation:
- Tracks running average of gradients (momentum)
- Tracks running average of squared gradients (adaptive learning rate)
- Each parameter gets its own learning rate

**AdamW** = Adam with decoupled weight decay:
- Adds regularization to prevent large weights
- Weight decay: `1e-5` (very small, gentle regularization)

**Learning rate**: `1e-4` (0.0001) — how big each parameter update step is

### Scheduler: CosineAnnealingLR

**What it does**: Gradually reduces the learning rate during training.

**How it works:**
- Starts at the initial learning rate (1e-4)
- Follows a cosine curve: fast reduction early, slow reduction later
- Ends near zero at the last epoch

**Why?**
- Large LR early: quickly find the right region of parameter space
- Small LR late: fine-tune within that region

**Formula**: `lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / total_epochs))`

---

## 20. Mixed Precision Training

**What it is**: Using FP16 (half precision) or BF16 (brain float) instead of FP32 for some computations.

**How it works:**
- Weights stored in FP32 (master copy)
- Forward/backward pass uses FP16/BF16 for speed
- Loss scaling prevents underflow in FP16 gradients
- Updates applied in FP32

**Why?**
- **Speed**: FP16/BF16 operations are 2× faster on modern GPUs
- **Memory**: Uses half the memory for activations
- **Quality**: Negligible accuracy loss for most tasks

**In our project**: Uses `torch.amp` (automatic mixed precision) with FP16/BF16.

---

## 21. Early Stopping

**What it is**: Stop training when the model stops improving on the validation set.

**How it works:**
1. After each epoch, evaluate on validation set
2. If validation accuracy improves → save checkpoint, reset counter
3. If no improvement → increment counter
4. If counter reaches `patience` (e.g., 7 epochs) → stop training

**Why?**
- Prevents overfitting: training longer doesn't always help
- Saves compute: no wasted epochs
- The best model is already saved as a checkpoint

---

## 22. Checkpoints

**What they are**: Saved snapshots of the model at specific points during training.

**Types:**
- **`<method>_best.pth`**: Saved when validation accuracy is highest. Used for final evaluation.
- **`<method>_last.pth`**: Saved at the end of training (final epoch).
- **`<method>_latest.pth`**: Most recent save (same as last if no crashes).

**What's inside a checkpoint:**
- Model state dict (all weight values)
- Optimizer state dict
- Training epoch number
- Best validation accuracy

**File size**: Depends on method. LoRA ~18 MB, QLoRA ~7.6 MB, QA-LoRA ~9.5 MB, Q/K LoRA ~12 MB.

---

## 23. Evaluation Metrics

### Accuracy
**What**: Percentage of correctly classified images.
**Formula**: `correct / total × 100`
**Example**: 10,617 correct out of 10,735 = 98.89%

### Precision (Macro)
**What**: For each class, what fraction of positive predictions were correct? Then average across all classes.
**Formula**: `macro_precision = mean(precision_per_class)`
**High precision** = few false positives (model doesn't cry wolf)

### Recall (Macro)
**What**: For each class, what fraction of actual positives were correctly identified? Then average.
**Formula**: `macro_recall = mean(recall_per_class)`
**High recall** = few false negatives (model doesn't miss cases)

### F1 Score (Macro)
**What**: Harmonic mean of precision and recall. Balances both.
**Formula**: `F1 = 2 × (precision × recall) / (precision + recall)`
**When to care**: When both false positives and false negatives matter

### Binary Accuracy
**What**: After collapsing 38 classes into "healthy" vs "diseased", what fraction is correct?
**Mapping**: Any class with "healthy" in the name → healthy; all others → diseased

### Binary F1
**What**: F1 score for the binary (healthy/diseased) classification

### ROC AUC (One-vs-Rest)
**What**: Area under the ROC curve. Measures how well the model separates classes.
- AUC = 1.0: Perfect separation
- AUC = 0.5: Random guessing
- AUC > 0.99: Excellent

**One-vs-Rest**: For each class, compute AUC treating it as "this class vs all others", then average.

---

## 24. Confusion Matrix

**What it is**: A table showing what the model predicted vs what the true label was.

**Example (simplified to 3 classes):**
```
              Predicted
              Tomato  Potato  Corn
True Tomato  [  95     3      2  ]
True Potato  [   4    93      3  ]
True Corn    [   1     2     97  ]
```

**How to read:**
- **Diagonal**: Correct predictions (high numbers = good)
- **Off-diagonal**: Errors (low numbers = good)
- **Row**: All images of one true class
- **Column**: All images predicted as one class

**In our project**: Generated as a heatmap using seaborn. Saved as PNG in `experiments/results/plots/`.

---

## 25. Crop/Disease Correctness Breakdown

**What it is**: Since class names follow `Crop___Disease`, we can decompose correctness into crop-level and disease-level.

**Categories per sample:**

| Category | Condition | Meaning |
|----------|-----------|---------|
| **Both correct** | Crop ✓ and Disease ✓ | Fully correct prediction |
| **Crop only** | Crop ✓ and Disease ✗ | Right plant, wrong disease |
| **Disease only** | Crop ✗ and Disease ✓ | Right disease type, wrong plant |
| **None correct** | Crop ✗ and Disease ✗ | Completely wrong |

**Example:**
- True: `Tomato___Late_blight`
- Predicted: `Tomato___Early_blight`
- Result: **Crop only** (Tomato is correct, but Late blight ≠ Early blight)

**Why useful?**
- Shows where the model fails: is it confused about crops or diseases?
- "Crop only" errors are less severe than "None correct"
- Helps diagnose model weaknesses

---

## 26. Transfer Learning to PlantDoc

**What it is**: Testing the model (trained on PlantVillage lab photos) on PlantDoc field photos.

**Why it's hard:**
- PlantVillage: clean, centered, plain background, consistent lighting
- PlantDoc: messy, multiple leaves, soil/grass/shadows, variable quality

**The domain gap:**
The model learned features specific to lab conditions (leaf shape, color patterns against plain backgrounds). These features may not transfer well to field conditions.

**Label alignment:**
PlantVillage and PlantDoc use different naming. We map 25 shared classes:
```
PlantVillage: Tomato___Late_blight  →  PlantDoc: Tomato leaf late blight
PlantVillage: Potato___healthy      →  PlantDoc: Potato leaf
PlantVillage: Apple___Apple_scab    →  PlantDoc: Apple Scab Leaf
```

Classes unique to either dataset are excluded from the benchmark.

---

## 27. Domain Adaptation Techniques

Techniques to improve PlantDoc performance without retraining on PlantDoc data.

### Segmentation
- Remove the background from PlantDoc images
- Keep only the leaf (like PlantVillage's clean images)
- Forces the model to focus on leaf features, not background

### Multi-Resolution Evaluation
- Run the model on the same image at multiple sizes (192, 224, 256, 320)
- Average the predictions
- Captures features at different scales

### KNN (K-Nearest Neighbors)
- Extract feature vectors from the model's penultimate layer
- For each test image, find the K most similar training images
- Use their labels to make a prediction
- Acts as a "sanity check" — if KNN works, the features are good

### Ensemble
- Combine predictions from multiple methods (LoRA + QLoRA + Q/K LoRA)
- Average the softmax probabilities
- Usually better than any single method

### CORAL (CORrelation ALignment)
- Align the feature distribution of source (PlantVillage) and target (PlantDoc)
- Uses linear transformation to match second-order statistics
- Aims to reduce domain shift in feature space

### Style Normalization
- Normalize PlantDoc images to match PlantVillage's color distribution
- Compute per-channel mean/std from PlantVillage training set
- Apply normalization to PlantDoc images before inference

### Quality Gating
- Filter out low-confidence predictions
- If max probability < threshold (e.g., 0.6), reject the prediction
- Improves precision at the cost of coverage

---

## 28. Cross-Method Ranking

**What it is**: A single script (`rank_experiments.py`) that compares all methods side by side.

**How it works:**
1. Read `experiment_results.csv` (one row per method)
2. Read per-method checkpoint rankings
3. For each method, pick the **best checkpoint** (rank 1 or `*_best.pth`)
4. Compare on a scorecard:
   - Primary: test accuracy
   - Tie-breakers: F1 macro, binary F1, GPU memory, parameter count
5. Assign overall rank (1 = best)

**Output**: `cross_method_ranking.csv` — the definitive comparison table.

---

## 29. Web UI & Inference

**What it is**: A local web application for interacting with the trained models.

**Launch**: `py -3.11 run_web_ui.py` → opens at `http://localhost:8000`

### Features
- **Model selector**: Choose LoRA, QLoRA, QA-LoRA, or Q/K LoRA
- **Image upload**: Drag-and-drop a leaf photo
- **Real-time prediction**: Shows predicted class, crop, disease, confidence
- **Experiment results**: Displays cross-method ranking, checkpoint info
- **PlantDoc results**: Shows transfer evaluation data

### How inference works
1. User uploads an image
2. Server loads the selected model checkpoint
3. Image is preprocessed (resize → center crop → normalize)
4. Forward pass through the model
5. Softmax gives probability distribution over 38 classes
6. Top prediction returned with confidence score

---

## 30. Dashboard

**What it is**: A self-contained HTML file with all experiment results embedded.

**Generate**: `py -3.11 generate_dashboard.py` → produces `experiments/results/dashboard.html`

**How it works:**
1. Reads all CSV files from `experiments/results/`
2. Reads all plot PNGs
3. Embeds CSV data as **inline JSON**
4. Embeds images as **base64** (no external files needed)
5. Outputs a single HTML file that opens in any browser

**Tabs:**
- **Overview**: Stat cards, bar charts, summary table
- **Checkpoint Rankings**: Binary metrics, correctness doughnut chart
- **Plots & Visuals**: Training curves, confusion matrices

---

## 31. Key Libraries

| Library | What It Does | Used For |
|---------|-------------|----------|
| **PyTorch** | Deep learning framework | Model definition, training, inference |
| **torchvision** | Computer vision tools | EfficientNet-B0, image transforms |
| **PEFT** | Parameter-efficient fine-tuning | LoRA adapter injection |
| **bitsandbytes** | Quantization | NF4 quantization for QLoRA V3 |
| **accelerate** | Training utilities | Mixed precision, distributed training |
| **datasets** | HuggingFace datasets | Loading PlantVillage from HF Hub |
| **scikit-learn** | Machine learning tools | Metrics, ROC AUC, confusion matrix |
| **pandas** | Data manipulation | CSV reading/writing |
| **matplotlib** | Plotting | Training curves, bar charts |
| **seaborn** | Statistical visualization | Confusion matrix heatmaps |
| **PyYAML** | YAML parsing | Configuration file loading |
| **Pillow** | Image processing | Image loading and transforms |
| **tqdm** | Progress bars | Training progress display |

---

## 32. Glossary

| Term | Definition |
|------|-----------|
| **Backbone** | The main feature extraction network (EfficientNet-B0) |
| **Adapter** | Small trainable module inserted into a frozen network (LoRA) |
| **Checkpoint** | Saved model weights at a specific training point |
| **Conv2d** | 2D convolution layer in PyTorch |
| **Domain gap** | Difference between training data distribution and test data distribution |
| **Epoch** | One complete pass through the entire training dataset |
| **Fine-tuning** | Adjusting a pretrained model for a new task |
| **FP32** | 32-bit floating point (standard precision) |
| **FP16** | 16-bit floating point (half precision) |
| **BF16** | Brain float 16-bit (Google's format, wider exponent than FP16) |
| **GPU** | Graphics Processing Unit — parallel processor for neural network training |
| **INT8** | 8-bit integer quantization |
| **INT4** | 4-bit integer quantization |
| **Inference** | Using a trained model to make predictions (no training) |
| **Logits** | Raw output scores from the model (before softmax) |
| **LR** | Learning rate — step size for parameter updates |
| **MBConv** | Mobile Inverted Bottleneck Convolution block |
| **NF4** | NormalFloat 4-bit — information-theoretically optimal 4-bit format |
| **Overfitting** | Model memorizes training data but fails on new data |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **Pointwise** | 1×1 convolution — mixes channels without changing spatial size |
| **Pretrained** | Already trained on a large dataset (ImageNet) |
| **Quantization** | Reducing weight precision (FP32 → INT8/INT4) |
| **Rank** | LoRA hyperparameter — dimensionality of the low-rank approximation |
| **Alpha** | LoRA scaling factor — multiplied with adapter output |
| **Softmax** | Converts logits to probabilities (sums to 1.0) |
| **State dict** | Dictionary mapping layer names to weight tensors |
| **Transfer learning** | Applying knowledge from one task to a different task |
| **Underfitting** | Model is too simple to capture the patterns |
| **Validation set** | Data held out during training to monitor overfitting |
| **Weight decay** | Regularization that penalizes large weights |

---

*This document covers every concept referenced in the presentation slides, training pipeline, and evaluation system. If something is still unclear, the answer is probably in one of the sections above.*
