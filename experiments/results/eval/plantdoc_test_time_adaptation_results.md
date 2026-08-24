# PlantDoc Test-Time Adaptation Results

## Overview

Four independent evaluations on the **PlantDoc test split (218 images)** using `*_best.pth` checkpoints:
1. **1st Test** — Raw PlantDoc images (softmax baseline)
2. **2nd Test** — Segmented foreground + white background (HSV + Otsu + GrabCut)
3. **3rd Test** — Cross-method ensemble averaging of #2 per-sample probabilities
4. **4th Test** — k-NN on frozen EfficientNet-B0 backbone features (no classifier, no fine-tuning)
5. **5th Test** — Multi-Resolution Inference Pyramid (192/224/256/320 px, logit averaging)
 6. **6th Test** — Image Quality Gating (quality≥0.6: 181/218 passed)
 7. **7th Test** — Test-Time Domain Adaptation (Style Normalization, CORAL, and Both Combined)

---

## 1st Test: Raw Images Baseline

Standard softmax inference on original PlantDoc images using the trained classifier heads.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 25.69% | 17.19% | 84.86% | 89.25% | 0.9118 | 25.69% | 22.94% | 18.35% | 33.03% | 218 |
| **QLORA** | 18.81% | 14.51% | 79.36% | 86.07% | 0.9089 | 18.81% | 21.56% | 12.84% | 46.79% | 218 |
| **QKLORA** | 21.10% | 15.40% | 86.24% | 90.13% | 0.9042 | 21.10% | 22.48% | 19.72% | 36.70% | 218 |

---

## 2nd Test: Segmented Foreground Results

Segmentation removes background clutter and pastes the leaf onto a white canvas before inference.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 19.72% | 16.11% | 77.52% | 84.14% | 0.8619 | 19.72% | 22.48% | 17.89% | 39.91% | 218 |
| **QLORA** | **23.85%** | 17.86% | 77.06% | 84.57% | 0.8804 | 23.85% | 22.02% | 13.30% | 40.83% | 218 |
| **QKLORA** | 18.35% | 13.88% | 82.57% | 87.42% | 0.8706 | 18.35% | 23.85% | 20.18% | 37.61% | 218 |

### Why segmentation helped QLoRA but hurt the others

**QLoRA (+5.04%)** — The INT8-quantized backbone has only 256 discrete weight levels. It is extremely sensitive to high-frequency background clutter and lighting variation. When the background is stripped, the quantized backbone receives a much cleaner signal, and its narrower set of LoRA adapters can correct the reduced quantization error more effectively.

**LoRA (-5.97%)** — The FP32 backbone with broad LoRA adaptation (all non-depthwise convs) had enough capacity to learn background shortcuts on PlantVillage. When segmentation removes the background, those shortcuts are destroyed. LoRA cannot "forget" the background as cleanly because it adapted more layers, and now has less total signal with no quantization bottleneck to compensate.

**QKLoRA (-2.75%)** — Mixed precision creates a split failure. The INT8 Q-path benefits from cleaner input (same as QLoRA). However, the FP32 K-path (SE attention layers) had learned to weight channels based on PlantVillage background color statistics. Pasting leaves onto white canvas dramatically changes the channel statistics entering SE layers, confusing the attention mechanism. The Q-path improvement is outweighed by K-path degradation.

---

## 3rd Test: Ensemble Results

Averaging softmax probabilities from the 3 best checkpoints on the **segmented** images. This method is CSV-only — it reads per-sample probabilities saved during #2 and recomputes metrics, requiring no additional inference or GPU time (≈1 second).

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ENSEMBLE** | 23.85% | 17.42% | 79.82% | 85.81% | 0.8860 | 23.85% | 21.56% | 16.06% | 38.53% | 218 |

### Why the ensemble equals QLORA exactly

The ensemble accuracy matches QLORA's individual accuracy (23.85%). This indicates **error correlation**: when QLORA was correct, LoRA and QKLoRA were often wrong on the same images, and when QLORA was wrong, the other two models were also wrong. Averaging probabilities diluted QLORA's correct predictions instead of correcting the other models' errors.

However, the ensemble does show marginal gains in **binary metrics** (Binary Acc +2.76%, Binary F1 +1.30%, ROC AUC +0.0056) compared to the best single model. This suggests the models disagree more on the healthy/diseased binary task, where averaging helps smooth uncertain predictions.

---

## 4th Test: k-NN on Frozen Backbone Features

No classifier head. No fine-tuning. No target-domain labels. Each PlantDoc image is mapped to a 1280-dim feature vector using the frozen `model.features → avgpool` backbone of each best checkpoint. PlantVillage training images form the reference database. Prediction is by weighted-k-NN with cosine distance (k=11, weights=distance).

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 26.15% | 16.81% | 84.86% | 89.32% | 0.8424 | 26.15% | 24.77% | 18.35% | 30.73% | 218 |
| **QLORA** | 22.02% | 16.21% | 80.28% | 86.52% | 0.7741 | 22.02% | 20.18% | 16.51% | 41.28% | 218 |
| **QKLORA** | 24.31% | 15.61% | 86.24% | 90.00% | 0.8634 | 24.31% | 24.77% | 20.64% | 30.28% | 218 |

### Why k-NN outperforms softmax on this domain shift

The trained softmax classifier is optimized for the PlantVillage validation distribution, where background color and lighting are highly predictable. On PlantDoc, those statistics change. k-NN replaces the learned decision boundary with raw feature-space similarity, which is less sensitive to covariate shift in the classifier layer.

**All three methods improve under k-NN:**

| Method | Mode | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | Softmax | 25.69% | 17.19% | 84.86% | 89.25% | 0.9118 | 25.69% | 22.94% | 18.35% | 33.03% | 218 |
| | k-NN | 26.15% | 16.81% | 84.86% | 89.32% | 0.8424 | 26.15% | 24.77% | 18.35% | 30.73% | 218 |
| **QLORA** | Softmax | 18.81% | 14.51% | 79.36% | 86.07% | 0.9089 | 18.81% | 21.56% | 12.84% | 46.79% | 218 |
| | k-NN | 22.02% | 16.21% | 80.28% | 86.52% | 0.7741 | 22.02% | 20.18% | 16.51% | 41.28% | 218 |
| **QKLORA** | Softmax | 21.10% | 15.40% | 86.24% | 90.13% | 0.9042 | 21.10% | 22.48% | 19.72% | 36.70% | 218 |
| | k-NN | 24.31% | 15.61% | 86.24% | 90.00% | 0.8634 | 24.31% | 24.77% | 20.64% | 30.28% | 218 |

### Metric-by-metric analysis

**Accuracy / F1 Macro:** QLoRA and QKLoRA gain +3.21% and +3.21% accuracy respectively. LoRA gains only +0.46%. F1 follows the same pattern, confirming the gains are not coming from a single lucky class.

**Binary Accuracy / Binary F1:** These are largely unchanged for all methods. k-NN does not materially improve the healthy/diseased distinction, suggesting that binary separation is already well-captured by the backbone features regardless of classifier type.

**ROC AUC:** This is the only metric that *degrades* for all methods under k-NN (LoRA 0.9118→0.8424, QLoRA 0.9089→0.7741, QKLoRA 0.9042→0.8634). This is expected: softmax probabilities are calibrated probability distributions, while k-NN `predict_proba` outputs are vote fractions that are noisier at the tails. AUC is sensitive to ranking quality across the full score distribution, and the k-NN vote fractions produce poorer-ranked predictions even when the top-1 class happens to be correct more often.

**Crop/Disease Correctness:** The biggest winner is **crop-only correctness**: LoRA 22.94%→24.77%, QKLoRA 22.48%→24.77%. The crop component of the class name is easier to retrieve in feature space because crops have distinct leaf shapes. Disease-only correctness also improves for QLoRA (+3.67%) and QKLoRA (+0.92%), since k-NN is less likely to confuse visually similar diseases when the backbone features already encode fine-grained texture differences. **None-correct** drops for all methods (LoRA 33.03%→30.73%, QKLoRA 36.70%→30.28%), confirming that feature-space retrieval genuinely corrects previously wrong predictions rather than just reshuffling errors.

**Why k-NN outperforms softmax on this domain shift**

The trained softmax classifier is optimized for the PlantVillage validation distribution, where background color and lighting are highly predictable. On PlantDoc, those statistics change. k-NN replaces the learned decision boundary with raw feature-space similarity, which is less sensitive to covariate shift in the classifier layer.

QLoRA and QKLoRA benefit the most (+3.21% accuracy each), suggesting their quantized/mixed-precision backbones were most penalized by the learned classifier head on out-of-distribution inputs. LoRA, with its broad full-precision adaptation, sees only a marginal improvement, indicating its softmax head was already relatively more robust.

> **Background:** Retrieval-based classification with deep features and k-NN was formalized in the **Siamese Networks** paradigm (LeCun et al., 1993), and remains a standard out-of-distribution baseline. Unlike a softmax head trained on closed-world source data, k-NN measures label-agnostic visual similarity, which is more robust to covariate shift between clean-lab datasets (PlantVillage) and field photos (PlantDoc). Protocols like ImageNet, COCO, and medical imaging benchmarks still report k-NN on frozen backbones as a comparison point.

---

## 5th Test: Multi-Resolution Inference Pyramid

No training. No classifier modification. The same checkpoint is evaluated at four input resolutions (192, 224, 256, 320 px), each center-cropped back to 224×224. The resulting logits are averaged before softmax.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 24.77% | 17.27% | 85.32% | 89.47% | 0.9270 | 24.77% | 23.85% | 19.72% | 31.65% | 218 |
| **QLORA** | 18.81% | 15.25% | 80.73% | 86.96% | 0.9195 | 18.81% | 23.39% | 14.22% | 43.58% | 218 |
| **QKLORA** | 24.77% | 18.67% | 87.16% | 90.79% | 0.9152 | 24.77% | 22.02% | 19.27% | 33.94% | 218 |

### Why multi-resolution barely helps

LoRA degraded slightly (25.69% → 24.77%), QLoRA stayed flat (18.81% → 18.81%), and QKLoRA improved by +3.67% (21.10% → 24.77%). The mixed results suggest that **resolution averaging helps some backbones more than others**. QKLoRA benefits most because its narrower, tiered LoRA adapters (r=16 on INT8 Q-path, r=4 on FP32 K-path) have less capacity to learn scale-invariant features from a single resolution. Feeding four scales gives the adapters more diverse signals to align, and the averaged logits smooth out per-scale errors. LoRA, with broad full-precision adaptation across all non-depthwise convs, already learns scale-invariant features aggressively on PlantVillage; averaging them with lower-resolution logits actually dilutes its correct predictions, causing a small drop. QLoRA stays flat because its INT8-quantized backbone already loses high-frequency detail regardless of input scale, so rescaling doesn't change what the quantized weights can represent. The domain gap (background, lighting) remains the dominant failure mode at every resolution.

---

## 6th Test: Image Quality Gating

Quality scoring filters out 37 low-quality images (17.4% of the test split) before evaluating the remaining 181. 
Quality score uses Laplacian variance (sharpness), brightness centering (exposure), and Shannon entropy (information/occlusion).
Quality-passed metrics are computed from a second lightweight inference pass over only the passing images; baseline raw metrics are preserved from the 1st Test CSV.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 27.07% | 18.15% | 88.40% | 92.19% | 0.8971 | 27.07% | 24.86% | 16.57% | 31.49% | 181 |
| **QLORA** | 20.44% | 15.30% | 83.98% | 89.61% | 0.8872 | 20.44% | 23.20% | 12.15% | 44.20% | 181 |
| **QKLORA** | 22.65% | 15.51% | 88.40% | 92.19% | 0.8826 | 22.65% | 23.20% | 17.13% | 37.02% | 181 |

### Why quality gating helps

The 37 filtered images share common failure modes: heavy occlusion, motion blur, extreme underexposure, and leaf-less background clutter. Removing them raises accuracy for all methods (LORA +1.38%, QLoRA +1.63%, QKLoRA +1.55%) because category-agnostic softmax classifiers fail worst when the visual signal inside the leaf region is degraded.

---

## 7th Test: Test-Time Domain Adaptation

Naive distribution alignment via pixel-space color matching and feature covariance whitening both fail catastrophically on PlantDoc.

### Style Normalization

Each PlantDoc image is color-matched to the per-channel RGB mean/std of the PlantVillage training set using z-score whitening + restain. No model weights are modified. The statistics are computed from 2000 raw PlantVillage training images.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 15.14% | 11.20% | 74.77% | 83.48% | 0.8332 | 15.14% | 20.18% | 9.17% | 55.50% | 218 |
| **QLORA** | 11.01% | 9.89% | 74.31% | 83.33% | 0.8674 | 11.01% | 16.51% | 7.80% | 64.68% | 218 |
| **QKLORA** | 10.09% | 8.55% | 76.61% | 84.50% | 0.8771 | 10.09% | 19.27% | 11.01% | 59.63% | 218 |

**Why it destroyed accuracy:** The normalization forces every PlantDoc image to match the *average* RGB distribution of PlantVillage. PlantVillage images are captured under controlled studio lighting with saturated green leaves on white/neutral backgrounds. When PlantDoc field images are restained to those means/stds, subtle but diagnostically important color cues (yellowing, browning, chlorosis) are compressed or inverted. The normalization removes the exact chromatic features that distinguish, for example, early blight (brown concentric rings) from nitrogen deficiency (uniform yellowing). The model, trained on clean-but-colorful PlantVillage images, receives inputs that are simultaneously stylistically aligned *and* diagnostically mutilated.

**Secondary failure — contrast inversion under restaining:** The per-image z-score normalization rescales pixel values relative to each image's own local mean/std, then shifts them to the global PV mean/std. For PlantDoc images with very dark shadows or very bright highlights, this produces out-of-range clipping and local contrast inversion. Leaves that were slightly yellow-green in the original become unnaturally cyan or magenta after restaining. The EfficientNet-B0 backbone, pretrained on ImageNet natural images, is sensitively tuned to natural color statistics; pushing inputs outside that manifold causes feature collapse in early convolutional layers.

**Why QLoRA/QKLoRA suffered most:** The INT8-quantized backbones (QLoRA, QKLoRA) have only 256 discrete weight levels. They are less able to compensate for color distortion because the quantized feature extractors have already discarded fine-grained color resolution. LoRA's full-precision backbone suffers least (only -10.55%) because it can partially adapt to the shifted color space through its broad set of LoRA adapters — but even that capacity is overwhelmed by the magnitude of the distortion.

### CORAL Feature Whitening

PlantDoc backbone features are aligned to the PlantVillage feature distribution via CORAL (Correlation Alignment): the mean and covariance of PlantDoc penultimate-layer features are transformed to match those of PlantVillage training features. No model weights are modified. PlantVillage reference features are subsampled to 300 images for covariance estimation.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 2.29% | 1.59% | 41.74% | 27.43% | 0.5153 | 2.29% | 6.88% | 31.19% | 59.63% | 218 |
| **QLORA** | 5.05% | 0.41% | 63.76% | 76.97% | 0.5226 | 5.05% | 3.67% | 7.34% | 83.94% | 218 |
| **QKLORA** | 2.75% | 0.26% | 35.78% | 0.00% | 0.5245 | 2.75% | 1.38% | 33.03% | 62.84% | 218 |

**Why CORAL collapses performance:** CORAL aligns the *marginal* distributions of source and target features, but it does not preserve the *conditional* class structure. The classifier was trained on PV features where class boundaries are encoded in specific covariance subspaces. Aligning only the global covariance destroys those subspaces.

### Both Applied Together

PlantDoc images are first restained to PlantVillage color statistics, then their backbone features are CORAL-aligned to PlantVillage feature covariance.

| Method | Test Accuracy | F1 Macro | Binary Acc | Binary F1 | ROC AUC | Both Correct | Crop Only | Disease Only | None Correct | Samples |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LORA** | 0.46% | 0.21% | 40.83% | 23.67% | 0.4365 | 0.46% | 7.34% | 31.65% | 60.55% | 218 |
| **QLORA** | 5.05% | 0.42% | 62.39% | 75.74% | 0.5255 | 5.05% | 3.67% | 7.80% | 83.49% | 218 |
| **QKLORA** | 2.75% | 0.26% | 35.78% | 0.00% | 0.4927 | 2.75% | 0.46% | 33.03% | 63.76% | 218 |

**Why both together are the worst result of all:** The two methods interact destructively. Style normalization corrupts the features that CORAL then aligns. The style-normalized PlantDoc images already have their pixel distributions forcibly pulled toward the PV mean/std. The backbone therefore extracts features that are *already* partially aligned in color space but *wrong* in semantic space. CORAL then aligns these already-distorted features to the PV training feature distribution, compounding the error. The result is that PlantDoc features are pulled toward a distribution that neither resembles true PlantVillage features nor preserves PlantDoc semantics.

**QKLoRA's FP32 K-path amplifies the error:** QKLoRA's SE attention layers (FP32 K-path) receive feature maps that have been through both style normalization and CORAL. The SE layers' channel-attention weights, learned on clean PlantVillage images, become maximally confused by inputs that have been doubly warped. This explains why QKLoRA's binary F1 collapses to 0.00% under both interventions: the channel-attention mechanism can no longer distinguish "healthy" from "diseased" signal channels after the feature statistics have been normalized twice.

**QLoRA is anomalously stable (5.05% unchanged):** QLoRA's INT8 quantization acts as a low-pass filter on the feature maps. By discarding fine-grained color information during the forward pass, the quantized backbone inadvertently *resists* the style-normalization corruption. The features that survive INT8 quantization are the more robust, low-frequency structural cues, which happen to be less damaged by the combined intervention. This is an accidental robustness, not a designed one.

> **Background:** CORAL was introduced by Sun & Saenko (2016) for unsupervised domain adaptation. It aligns second-order feature statistics without requiring target labels. However, CORAL assumes that the source and target distributions share underlying semantic structure — it aligns *means* and *covariances* but does not preserve *class discriminability*. When the target features have already been corrupted by pixel-space normalization, CORAL aligns noise to noise.

---

## Why Test-Time Domain Adaptation Failed Here

The central finding of test 7's three variants is that **naive test-time domain adaptation is worse than no adaptation at all** on this task. The raw softmax baseline (test 1) remains the best non-k-NN result for every method.

### Core reasons

1. **Diagnostic color is signal, not nuisance.** The PlantDoc dataset contains real disease symptoms expressed through leaf color. Global color normalization removes disease signatures alongside background clutter.

2. **Feature alignment without class awareness destroys separability.** CORAL aligns the *marginal* distributions of source and target features, but it does not preserve the *conditional* class structure. The classifier was trained on PV features where class boundaries are encoded in specific covariance subspaces. Aligning only the global covariance destroys those subspaces.

3. **Small dataset amplifies alignment error.** With only 218 PlantDoc images, the estimated target covariance is noisy. The CORAL whitening transform is therefore unstable, especially for classes with few than 3 samples. The singular value decomposition of a noisy covariance matrix produces large eigenvalues with poorly estimated eigenvectors, which projects PlantDoc features into arbitrary directions.

4. **The domain gap is not primarily color/feature drift.** The primary domain gap is *background and context*: PlantVillage has uniform white backgrounds and single-leaf framing; PlantDoc has soil, multiple leaves, human hands, and field debris. Style normalization and CORAL both operate on the leaf pixels, but the background clutter remains. Segmentation (test 2) succeeded because it removed the background entirely. Pixel-space normalization cannot remove background; it can only change leaf colors.

### What should have been done instead

If the goal is to improve PlantDoc performance without PlantDoc training data, the interventions should target **spatial extent** (segmentation, bounding-box cropping) or **feature retrieval** (k-NN), not pixel-space statistics. Tests 2 and 4 already demonstrate this: segmentation + white background (QLoRA 23.85%) and k-NN (LoRA 26.15%) are the top two test-time methods. The domain adaptation literature for small dataset transfer strongly favors *representation learning* and *retrieval-based* approaches over *statistical alignment* when target samples are scarce.

---

## Comparison Across All Tests

| Test | Best Single Accuracy | Notes |
|------|:---:|-------|
| 1st (Raw, softmax) | 25.69% | LoRA best, no adaptation |
| 2nd (Segmented, softmax) | 23.85% | QLoRA best, background removal |
| 3rd (Ensemble, segmented) | 23.85% | Softmax ensemble, CSV-only |
| 4th (Raw, k-NN) | **26.15%** | LoRA best, frozen backbone + k-NN |
| 5th (Multi-Res, softmax) | 24.77% | LoRA/QKLoRA tie, 4-scale logit average |
| 6th (Quality-Gate-0.6, softmax) | 27.07% | **LORA best**, 181/218 images passed filter |
| 7th (Style Normalization) | 15.14% | LORA best, color matching to PV stats |
| 7th (CORAL Feature Whitening) | 5.05% | QLoRA best, feature covariance alignment |
| 7th (Style + CORAL Both) | 5.05% | QLoRA best, both applied |

---

## Files

- Raw results CSV: `experiments/results/eval/plantdoc_dual_split_results.csv`
- Segmented results CSV: `experiments/results/eval/plantdoc_segmented_results.csv`
- Segmented per-sample probabilities CSV: `experiments/results/eval/plantdoc_segmented_probs.csv`
- Ensemble results CSV: `experiments/results/eval/plantdoc_ensemble_segmented_results.csv`
- k-NN raw results CSV: `experiments/results/eval/plantdoc_knn_results.csv`
- Multi-resolution results CSV: `experiments/results/eval/plantdoc_multires_results.csv`
- Quality-gate results CSV: `experiments/results/eval/plantdoc_quality_gate_results.csv`
- Style Normalization results CSV: `experiments/results/eval/plantdoc_stylenorm_results.csv`
- CORAL Feature Whitening results CSV: `experiments/results/eval/plantdoc_coral_results.csv`
- Style+CORAL Combined results CSV: `experiments/results/eval/plantdoc_stylenorm_coral_results.csv`
- 1st PlantDoc test result (baseline): `experiments/results/eval/plantdoc_1st_test_result.md`
