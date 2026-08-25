# CNN PEFT — Viva Preparation Notes

> Everything below explains how LoRA / QLoRA / QA-LoRA, originally designed for
> transformer Linear layers, are adapted to the **Conv2d** layers of EfficientNet-B0
> for leaf-disease classification. Numbers are verified against the actual repo
> (`models/peft/*`, `models/backbone/efficientnet_b0.py`).

---

## 1. The core bridge: a Conv2d is a Linear on unfolded patches

A `Conv2d` weight is 4D: `W ∈ ℝ^{C_out × C_in × k × k}`.
Define `D_in = C_in · k · k` and flatten the kernel:

```
W_mat = W.view(C_out, D_in)        # e.g. (40, 24, 1, 1) → (40, 24)
```

Convolution is a **linear** operator, so for the im2col-unfolded input
`X ∈ ℝ^{D_in × N}` (N = number of spatial positions):

```
Y = W_mat · X + b·1ᵀ        ∈ ℝ^{C_out × N}
```

This is **identical** to the Linear equation `output = W × input`.
For `k = 1`: `D_in = C_in = 24`, `C_out = 40` → `W_mat ∈ ℝ^{40×24} = 960` numbers.

**Viva one-liner:** *"Once I flatten the kernel to `out × (in·k·k)`, every LoRA and
quantization formula written for Linear layers applies unchanged. Convolution
becomes ordinary matrix multiplication on unfolded patches."*

---

## 2. The three adaptations (your 1×1 example: 40×24×1×1 = 960 numbers)

### (a) Original (frozen)
```
output = W ⊗ x      (≡ W_mat · X)
frozen params = C_out · C_in · k · k = 960,  trainable = 0
```
`⊗` = `F.conv2d`.

### (b) LoRA — train 512 instead of 960
Add `A ∈ ℝ^{r × D_in}`, `B ∈ ℝ^{C_out × r}` (rank `r = 8`):
```
Ŷ = W_mat · X + (B · A) · X
  = W_mat · X + B(A X)
  ≡ conv(x, W) + (α/r) · conv(conv(x, A_w), B_w)
```
where `A_w ∈ ℝ^{r×C_in×k×k}`, `B_w ∈ ℝ^{C_out×r×1×1}` keep it a real conv.

```
trainable = r·D_in + r·C_out = r(C_in·k·k + C_out)
k=1, r=8:  8·24 + 8·40 = 192 + 320 = 512   ✓
merge:     W_merged = W_mat + (α/r) · B · A     (zero extra inference cost)
```

### (c) NF4 quantization — shrink the frozen base 4×
Per block of 64 elements of `W_mat`:
```
s   = |W_mat|_absmax                  (block scale; QLoRA double-quants s to 8-bit)
q   = quantize_NF4(W_mat / s)  → 4-bit
Ŵ   = dequant_NF4(q) · s
```
```
memory: 960 · 4 bits = 480 B   (vs 1920 B @FP16, 3840 B @FP32)  → 4×
forward: out = conv(x, Ŵ) + lora_term ;  only A, B, scales trained
```

### (d) QA-LoRA (implemented in `models/peft/qalora.py`)
Group `D_in` into `L` groups of size `g = D_in/L`; each group `j` has learned
scale `α_j`, zero-point `β_j` (stored as real INT4 integers):
```
W_q = clamp(round(W_j/α_j + β_j), -8, 7)        (INT4 asymmetric)
W   = (W_q − β_j) · α_j                          (dequant)
A ∈ ℝ^{L×r},  B ∈ ℝ^{C_out×r}                     (group-pooled LoRA)
forward:  out = conv(x, W) + (α/r) · (x_pooled · A) · Bᵀ
```
**Merge rule** (absorbs adapter into the quantized weights so inference stays INT4,
no FP16 fallback):
```
β_new = β − (α/r) · (B · Aᵀ) / α_j
```

---

## 3. How flattening works (and why it loses NO information)

**Reshaping is reindexing, not approximation.** The weight has 960 numbers as
`40×24×1×1`; after `.view(40,24)` it still has exactly those 960 numbers, just
looked up with two indices instead of four. The computation is identical:

```
conv:   Y[c_out] = Σ_{c_in} W[c_out,c_in] · X[c_in]        (per pixel, k=1)
matrix: Y       = W_mat · X        where W_mat = W.view(40,24)
```
Same sum, same terms → bit-for-bit identical output.

For `k×k`: each output pixel uses a `C_in×k×k` patch; flatten that patch to a
vector of length `D_in`, stack patches as columns (`F.unfold`, line 109 of
`qalora.py`). Convolution = matrix multiply on those columns.

**Where information is *actually* reduced (all deliberate, none from flattening):**
1. **LoRA's low-rank constraint** — only a rank-`r` update is learned (modeling choice; base `W` untouched).
2. **Quantization (NF4/INT4)** — 4-bit precision loss on the *frozen base only* (this is the desired memory saving).
3. **QA-LoRA group-wise pooling** — `adaptive_avg_pool1d` (line 117) coarse-grains the adapter's view of the input so it can merge into INT4 zero-points.

**Viva answer:** *"Flattening is just a reshape — reindexing — so it's mathematically
lossless; convolution and the flattened matrix multiply compute the identical sum.
The only information reduction is intentional and separate: LoRA restricts the
adapter to low rank, quantization compresses the frozen base to 4-bit, and QA-LoRA
additionally pools activations into groups so the adapter can merge into INT4
weights. None of it comes from flattening."*

---

## 4. Board-proof toy (2×2 kernel, 2 in / 2 out channels, 3×3 input)

> NOTE: this `2×2` toy exists ONLY to prove `conv = flattened matmul`. The real
> model targets **1×1** convs (see §5). Do not present 2×2 as what you deploy.

Weights (8 numbers) flattened to `W_mat ∈ ℝ^{2×8}` (cols ordered `(in0,p,q)(in1,p,q)`):
```
out0 → [1 2 3 4 | 0 0 0 0]
out1 → [0 0 0 0 | 1 2 3 4]      (out0 depends on in0 kernel [[1,2],[3,4]], out1 on in1)
```
Input: `in0=[[1,2,3],[4,5,6],[7,8,9]]`, `in1=[[10,11,12],[13,14,15],[16,17,18]]`.

**By convolution definition** (top-left output, patch `[[1,2],[4,5]]` etc.):
```
out0(0,0) = 1·1 + 2·2 + 3·4 + 4·5 = 37
out1(0,0) = 1·10 + 2·11 + 3·13 + 4·14 = 127
```
**By flattened matrix multiply** (patch unfolded to `col = [1,2,4,5,10,11,13,14]ᵀ`):
```
out0 = [1 2 3 4 | 0 0 0 0]·col = 37   ✓
out1 = [0 0 0 0 | 1 2 3 4]·col = 127  ✓
```
Same numbers, two notations → flattening is reindexing, not approximation.

---

## 5. Why we adapt 1×1 ONLY (verified parameter breakdown)

Your target selector (`int8_utils.py:27`) gates on
`kernel_size == (1,1) and groups == 1`, so **only 1×1 pointwise convs** get
LoRA/quant. Targeted: MBConv `.block.0.0` (expand), `.block.2.0` (project),
`features.8.0` (head). Excluded: stem, SE (`fc1/fc2`), and **all depthwise**
(`groups>1`) — comment at `int8_utils.py:53`: *"Depthwise convolutions are
FP32/frozen with no LoRA (PEFT limitation)."*

**Measured on EfficientNet-B0:**
```
Total conv params = 3,965,532

1x1 pointwise (groups=1)  → 3,782,652  = 95.4%   ← YOUR LoRA + 4-bit targets
3x3/5x5 depthwise (g>1)   →   182,016  =  4.6%   ← frozen FP32
3x3 stem (groups=1)       →       864  =  0.02%  ← excluded (features.0.0)
```
LoRA trainable at rank 8 on 1×1 only: **331,712** (≈11× fewer than the 3.78M base).

**Viva answer:** *"In EfficientNet's MBConv the expensive, parameter-heavy ops are
the 1×1 pointwise expand/project convs that mix channels — they hold 95.4% of all
conv weights, so they dominate compute and memory. The 3×3/5×5 depthwise conv is
cheap (4.6% of weights, one small kernel per channel) and, because `groups>1`,
doesn't fit PEFT's LoRA Conv2d cleanly. So we target 1×1 only, where the
Linear-equivalence is exact and the savings are largest."*

---

## 6. Would adapting k×k / 2×2 convs give better accuracy?

**Short answer: no, not with standard LoRA — and the 95.4% figure proves 1×1-only is right.**

1. **EfficientNet-B0 has no 2×2 convs** — the spatial convs are 3×3 and 5×5 *depthwise*.
2. **Depthwise has a hard rank-1 ceiling.** Each depthwise group produces exactly
   **1 output channel** from 1 input channel → flattened weight is `1 × k²`,
   rank ≤ 1. A LoRA update `B·A` with `B∈ℝ^{1×r}` is also rank ≤ 1 *regardless of r*.
   So LoRA on depthwise cannot add more than a rank-1 residual per group; raising
   `r` adds parameters but **zero extra capacity**.
3. They hold only **4.6%** of weights → tiny capacity/memory upside.

**Viva answer:** *"Adapting the spatial depthwise convs could, in principle, let the
model relearn texture/shape features — relevant for leaf disease since lesions are
texture-driven. But two limits apply. First, they hold only 4.6% of weights, so the
upside is tiny. Second, each depthwise group is inherently rank-1, so standard
low-rank LoRA is mathematically hamstrung — it learns only a rank-1 residual no
matter the rank. You'd pay complexity for marginal gain. Freezing them as FP32 is
justified; if I wanted spatial adaptation I'd use a purpose-built spatial adapter,
not standard LoRA."*

---

## 7. Anticipated viva Q&A

**Q: "Why not just full fine-tune?"**
A: Full fine-tuning updates all 3.96M conv params in FP16/FP32 (memory + overfitting
risk on a small leaf-disease dataset). PEFT trains ~332K LoRA params, keeps the base
4-bit frozen → far less memory, less overfitting, deployable on edge devices.

**Q: "Does flattening break the spatial structure of the kernel?"**
A: For 1×1 there is no spatial structure to break. For k×k all kernel weights remain
present after flattening; what changes is that the low-rank adapter couples channel
and spatial dims — a capacity choice, not a loss of the original weights. And depthwise
(k×k, groups>1) is excluded anyway.

**Q: "Why QA-LoRA over QLoRA?"**
A: QLoRA dequantizes to 16-bit at inference to add the LoRA. QA-LoRA uses group-wise
INT4 with learned scales/zero-points and a merge rule (`β_new = β − (α/r)·BAᵀ/α_j`)
that absorbs the adapter into the quantized weights, so inference stays fully INT4 —
no 16-bit fallback, which matters for on-device leaf screening.

**Q: "Is the method only for 1×1 convs then?"**
A: It's applied to 1×1 pointwise layers (95.4% of weights). The math generalizes to
any k×k via `W_mat = out × (in·k·k)`; depthwise is skipped by design (rank-1 limit + tiny share).

---

*Generated as a viva reference. All parameter counts verified programmatically on
`torchvision.models.efficientnet_b0(weights=None)`.*
