# Domain Adaptation Strategy for Training Method Refinement

**Verification:** The proposed refinements are correct and internally consistent with the deep-research report and the existing codebase.

---

## 1. Compatibility Matrix (Verified)

| Method | Stage | Requires Target Labels | Compatible With |
|--------|-------|------------------------|-----------------|
| **MixStyle** | Training-time | No | LoRA / QLoRA / QKLoRA |
| **Aggressive Augmentation** | Training-time | No | LoRA / QLoRA / QKLoRA |
| **AdaBN** | Test-time | No (unlabeled target only) | All methods |
| **SHOT-style adaptation** | Test-time fine-tuning | No (unlabeled target only) | LoRA (primary), QLoRA/QKLoRA (experimental) |
| **Stylized / AdaIN training** | Training-time | No | LoRA / QLoRA / QKLoRA |
| **GLCM Hybrid** | Architecture change | No | LoRA / QLoRA / QKLoRA |

**Short answer:** Yes, they are compatible. They operate at orthogonal stages (train vs test vs architecture) and use disjoint configuration knobs.

---

## 2. Existing Infrastructure Context (Critical)

Before adding new methods, understand what is **already implemented** in this repo:

- **PlantDoc evaluation harness** (`run_plantdoc_evaluation.py` + `launcher_plantdoc.py`): Already runs 7 distinct test protocols (raw, segmented, k-NN, multi-res, quality-gated, style-normalized, CORAL).
- **`--style-norm` flag**: Applies test-time color-space normalization (computes PV mean/std, remaps PlantDoc images). This is NOT the same as AdaBN. It adjusts the RGB histogram to match PlantVillage lighting. AdaBN adjusts BatchNorm layer statistics.
- **Style + CORAL tests**: Already active in the PlantDoc evaluator. These are feature-distribution alignment methods run entirely at inference time.

Therefore, **MixStyle** and **Aggressive Augmentation** are the primary training-phase additions. **AdaBN** is a new test-time method that sits alongside the existing `--style-norm` / CORAL pipeline. **SHOT**, **Stylized/AdaIN**, and **GLCM Hybrid** are larger additions.

---

## 3. Baseline Accuracy Expectations (Corrected)

The baseline numbers quoted in the original draft (~99% PV / ~30-40% PD) are paper-backed but need calibration against this repo's actual training regime:

| Phase | Setting | PlantVillage Test Acc | PlantDoc Test Acc (est.) |
|-------|---------|------------------------|--------------------------|
| **Baseline** | Current training config, no adaptation | ~99% (research [1], [4]) | 30–40% (research [1]) |
| + MixStyle | Higher domain diversity during training | ~99% (maintained) | +5–10 pp |
| + Aggressive Augmentation | Heavier photo distortion | ~97–98% | +3–8 pp |
| + AdaBN (test-time) | BN stats recomputed on PlantDoc | N/A | +5–10 pp |
| + SHOT (test-time, LoRA) | Unsupervised LoRA adaptation on PlantDoc | N/A | +3–7 pp |

> **Note:** "pp" = percentage points. Exact gains depend on augmentation magnitude and PlantDoc subset size. Cross-method gains are **not additive**; later methods build on the best training config found earlier.

---

## 4. Recommended Phase-by-Phase Pipeline

### Phase 1: Training-Time Isolation (one knob at a time)

> **Rule:** Fix all other config keys; change only one knob per experiment. Run `main.py <method>` and `launcher_plantdoc.py` to compare PV and PD metrics.

| Step | Change | Target Config Key | Expected Signal |
|------|--------|-------------------|-----------------|
| **1a** | Baseline (current) | None | Establish floor: PV ≈ 99%, PD ≈ 30–40% |
| **1b** | Add MixStyle | `training.use_mixstyle: true`, `training.mixstyle_prob: 0.5` | PD boost; if PD drops, PV should stay ≈99% |
| **1c** | Add Aggressive Augmentation | `data.augmentation_level: "heavy"` | PD boost; PV may dip slightly to 97–98% |
| **1d** | Add Stylized / AdaIN | `training.use_adain: true` | Shape-bias increase; best combined with 1b/1c after comparison |

**Implementation notes:**
- **MixStyle hook location**: EfficientNet-B0's `features.0` (stem) is the best insertion point because it affects all subsequent MBConv blocks. Alternatively inject after `features[1]` and `features[2]` (early MBConv stacks). Depthwise convs are excluded from MixStyle because they have `groups > 1` and PEFT cannot attach adapters there anyway; MixStyle applies to `Conv2d` with `groups == 1`.
- **Where to code**: Add `apply_mixstyle(x)` in `training/trainer.py` as a forward-hook wrapper. Toggle with `self.use_mixstyle`. Use `torch.roll` or random permutation within a minibatch to mix channel statistics.
- **Aggressive Augmentation**: Modify `get_data_transforms()` in `data/data_loader.py`. Add `RandomApply` of `GaussianBlur`, `RandomGrayscale`, `RandomChannelPermutation` (custom), `RandomErasing`, and widen `ColorJitter` to `brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1`.

### Phase 2: Test-Time Methods (after Phase 1 winner is stable)

| Step | Method | Implementation location | Notes |
|------|--------|------------------------|-------|
| **2a** | **AdaBN** | `run_plantdoc_evaluation.py` → add `--adabn` flag | Walk BN layers: set `requires_grad=False`, loop over PlantDoc subset (1k–3k imgs) in train-mode without grads, collect `running_mean`/`running_var` per layer, then eval. Write `plantdoc_adabn_results.csv`. |
| **2b** | **SHOT** | New file: `adaptation/shot_adapter.py` | Load LoRA checkpoint, freeze `model.classifier`, keep LoRA `A`/`B` unfrozen, run entropy minimization + pseudo-labeling on PlantDoc images. **Start with LoRA**; QLoRA/QKLoRA complicate SHOT due to quantized weights. |

**Why this order:** Phase 1 improves the model's inherent generalization. Phase 2 leverages the already-decent model to squeeze out more PD accuracy with minimal label cost. Use Phase 2 **only after** Phase 1 produces a stable winning config, or the gains will be confounded.

### Phase 3: Architecture Change (separate track)

| Step | Method | Notes |
|------|--------|-------|
| **3a** | **GLCM Hybrid** | Add `models/peft/glcm_branch.py` with GLCM texture features (contrast, homogeneity, energy, correlation). Concatenate with EfficientNet-B0 `avgpool` output. Freeze CNN; train GLCM branch + LoRA on classifier. Compare GLCM+LoRA against the best Phase 2 winner. Do NOT combine with Phase 1 augmentations initially. |

---

## 5. Relationship to Existing Tests

The current `launcher_plantdoc.py` already quantifies the clean→noisy gap:

- Tests 1–7 cover **segmentation, ensemble, k-NN, multi-resolution, quality gating, style normalization, and CORAL**.
- **Test 7 (Style + CORAL)** is the closest existing analogue to AdaBN + feature alignment, but it is **pixel-space + feature-space alignment**, not BN-stat adaptation.
- **MixStyle is missing** from the current evaluator options. It is a train-time change, so it shows up in `experiment_results.csv` via the normal training pipeline.
- **AdaBN is missing** as a test-time option. It should be added under Test 7 or as Test 8, but it must be labeled distinctly from `--style-norm` to avoid confusion.

---

## 6. Constraints and Cautions

1. **Do not combine Phase 1 changes blindly.** If MixStyle (+3 pp) + Aggressive Augmentation (+5 pp) are both active and the PD gain is +4 pp, it is ambiguous which one contributed what.
2. **SHOT with QLoRA/QKLoRA is tricky.** QLoRA uses INT8-quantized weights with per-channel dequantization. QKLoRA has mixed precision (Q-path INT8, K-path FP32). SHOT requires gradients flowing through the feature extractor. While LoRA adapters are FP32 and trainable, feeding quantized tensors through the backbone during SHOT may introduce instability. **Start SHOT with LoRA only**, then optionally port to QLoRA/QKLoRA with explicit gradient checks in Phase 2b after the LoRA+SHOT result is stable.
3. **GLCM Hybrid is not a quick change.** It requires modifying `models/classifier.py`, adding a new branch, and redefining the training loop. Keep it isolated in Phase 3.
4. **PlantDoc subset size:** For AdaBN and SHOT, use the full PlantDoc split when possible, but for rapid iteration use a stratified subset of 500–1,000 images. The `launcher_plantdoc.py` already supports per-split evaluation.

---

## 7. Concrete Implementation Order

**Priority 1 (This Week)**
- [ ] Ad implement MixStyle in `training/trainer.py` (forward-hook after `features[0]`)
- [ ] Ad `training.use_mixstyle` and `training.mixstyle_prob` to `config/base_config.yaml`
- [ ] Validate with `python main.py lora` to ensure training still converges

**Priority 2 (After MixStyle validation)**
- [ ] Extend `get_data_transforms()` with aggressive augmentation; add `data.augmentation_level` config
- [ ] Run PV→PD comparison for baseline vs MixStyle vs aggressive aug (one at a time)

**Priority 3 (After best training config is fixed)**
- [ ] Add `--adabn` to `run_plantdoc_evaluation.py`; produce `plantdoc_adabn_results.csv`
- [ ] Compare baseline vs AdaBN vs best training config vs AdaBN+training config

**Priority 4 (If time permits)**
- [ ] Implement `adaptation/shot_adapter.py` for LoRA-only SHOT
- [ ] Implement GLCM branch in a separate feature branch for Phase 3

---

## 8. Quick Commands Reference

```bash
# Train one method
python main.py lora        # baseline
python main.py qlora       # QLoRA
python main.py qklora      # Q/K LoRA

# Run PlantDoc evaluations on trained checkpoints
python launcher_plantdoc.py  # interactive menu
# then choose option 7 (Style + CORAL) or add new options for MixStyle/AdaBN later
```

---

## 9. Summary

The original strategy is **correct in principle** but lacked these crucial refinements:
- It did not account for the **existing PlantDoc evaluation infrastructure** already present in the repo.
- It conflated `--style-norm` (color-space) with AdaBN (BN-statistics), which require different implementations.
- It referenced a 4-week milestone plan without mapping methods to specific weeks or files.
- It did not constrain SHOT to LoRA-first, despite SHOT's gradient-flow requirement conflicting with INT8 quantization paths.

This document resolves those gaps.
