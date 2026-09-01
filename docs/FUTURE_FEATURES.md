# Future Features — Pending Work Only

> Items in this file are **not yet implemented**. For completed features, see the architecture doc and this file's git history.

---

## 1. Dashboard Improvements

### Known Issues
- Dashboard has display bugs on certain screen sizes
- Missing cross-method comparison tab
- No single-image inference preview integrated

### Planned Work
- Add a **Method Comparison** tab to `dashboard.html` showing the cross-method ranking side by side
- Wire single-image inference preview into the dashboard (requires checkpoint loading in-browser or API call)
- Regenerate after changes: `python generate_dashboard.py`

### Status
- `rank_experiments.py` produces `cross_method_ranking.csv` ✅
- `generate_dashboard.py` generates self-contained HTML ✅
- Dashboard tab and inference preview: **NOT DONE**

---

## 2. Confidence Calibration (Temperature Scaling)

### Problem
The model outputs raw softmax probabilities that are poorly calibrated on out-of-distribution PlantDoc images. This leads to confidently wrong predictions on field photos.

### Proposed Solution
Apply temperature scaling to calibrate softmax outputs using the PlantVillage validation set. Add a configurable confidence threshold — if max calibrated probability falls below threshold, return "uncertain" or fall back to crop-only prediction.

### Implementation
- Fit a single temperature scalar on the validation set
- `torch.nn.functional.softmax(logits / T, dim=1)`
- Configurable threshold (e.g., 0.6 or 0.7)
- Log calibration metrics (ECE, reliability diagram)

### Expected Gain
Reduces confidently wrong predictions on PlantDoc. Does not necessarily raise top-1 accuracy, but improves practical reliability.

### Files to Modify
- `evaluation/evaluator.py` — temperature scaling + thresholding
- `evaluation/metrics.py` — calibration metrics (ECE, MCE)
- `web_app/server.py` — expose confidence threshold config

### Priority
**Worth doing.** ~30 lines, no retraining, adds a professional reliability layer.

---

## 3. Crop-First Prediction Cascade

### Problem
The 38-class problem space is very large for low-quality PlantDoc images. Predicting crop and disease simultaneously is harder than sequential prediction.

### Proposed Solution
At inference time:
1. Predict crop only by summing class probabilities across all diseases for each crop
2. Select the winning crop
3. Restrict disease prediction to diseases known to affect that crop

Additionally, surface the **both / crop-only / disease-only / none** correctness breakdown as a first-class metric in the web UI and PlantDoc tables.

### Expected Gain
+2–5% accuracy on PlantDoc by reducing the effective class space.

### Files to Modify
- `evaluation/evaluator.py` — crop-first cascade option
- `evaluation/metrics.py` — crop/disease breakdown for PlantDoc
- `web_app/static/app.js` — correctness badges in PlantDoc table
- `web_app/static/index.html` — correctness columns

### Priority
**Worth doing.** Zero retraining, improves both accuracy and analysis depth.

---

## 4. Test-Time Augmentation (TTA) for PlantDoc

### Problem
PlantDoc images have high within-class variance (different framing, zoom, lighting). A single forward pass may miss the correct prediction.

### Proposed Solution
For each test image, generate 5 augmented views:
1. Original (center crop)
2. Horizontal flip
3. Slightly different center crop
4. Brightness shift
5. 90° rotation

Average the softmax probabilities across all 5 views and select the final class.

### Expected Gain
+3–8% PlantDoc test accuracy (zero retraining). This is the single lowest-effort / highest-impact improvement for domain-shift evaluation.

### Files to Modify
- `evaluation/evaluator.py` — TTA flag + augmentation logic (~15 lines)
- `config/base_config_v3.yaml` — add `eval.tta_enabled: true/false`

### Priority
**High for PlantDoc.** Universally accepted technique (Krizhevsky et al. 2012).

---

## 5. Data Augmentation for Domain Robustness

### Problem
Current training augmentations (random crop, flip, rotation ±15°, color jitter) are too mild to simulate real-world field conditions. PlantDoc images appear as a completely alien distribution.

### Proposed Solution
Apply **domain-randomization style** augmentations to PlantVillage images during training:
- Random Gaussian blur (out-of-focus field photos)
- Random coarse dropout / CutOut (occlusion)
- Heavy brightness/contrast/saturation shifts
- Random noise injection (low-light camera noise)
- Random elastic transform / affine warp (camera angle variation)

### Expected Gain
+5–15% PlantDoc accuracy. High risk / high reward — may hurt PlantVillage accuracy if overdone.

### Files to Modify
- `data/data_loader.py` — aggressive augmentation transforms
- `config/base_config_v3.yaml` — augmentation strength toggles

### Priority
**Do only if PlantDoc is a major thesis contribution.** Requires full retraining of all models.

---

## Notes

- This file tracks **only** work that has not been done yet
- For completed features, see `architecture_design_v3.md` and the git log
- Per-method checkpoint ranking CSVs require re-running `launcher_test_v3.py` → evaluate-all
