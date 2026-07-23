# Improvement Ideas (SUPERSEDED — see IMPROVEMENT_IDEAS_NEW.md)

> **Note on PlantDoc evaluation metrics:** All PlantDoc testing accuracy reporting should include a correctness breakdown: **both** (crop AND disease correct), **crop only** (crop correct, disease wrong), **disease only** (crop wrong, crop correct), and **none** (both wrong). This mirrors the existing `both_correct_pct`, `name_only_correct_pct`, `disease_only_correct_pct`, and `none_correct_pct` columns already computed in `eval/*_checkpoint_ranking.csv` and should be surfaced prominently in the UI and any PlantDoc result tables.

## 1. [Empty — reserved for future idea]

## 2. [Retired — moved to IMPROVEMENT_IDEAS_NEW.md #4]

## 3. Confidence Calibration + Rejection

### Problem
The model currently outputs raw softmax probabilities that are often poorly calibrated, especially on out-of-distribution PlantDoc images. This leads to confidently wrong predictions on field photos that look nothing like the training distribution.

### Idea
Apply **temperature scaling** to calibrate the model's softmax outputs using the PlantVillage validation set. Then on PlantDoc (and any future deployment), if the max calibrated probability falls below a threshold, the system either rejects the prediction or falls back to a safer output (e.g., crop-only prediction). This prevents the model from making high-confidence wrong guesses on unfamiliar field images.

### Implementation
- Fit a single temperature scalar on the validation set using `torch.nn.functional.softmax(logits / T, dim=1)`
- Add a configurable confidence threshold (e.g., 0.6 or 0.7)
- At inference, if `max(calibrated_probs) < threshold`, return "uncertain" or fall back to crop-only prediction
- Log calibration metrics (ECE, reliability diagram) to confirm improvement

### Expected Gain
Reduces confidently wrong predictions on PlantDoc. Does not necessarily raise top-1 accuracy, but improves practical reliability and makes the "weak on PlantDoc" result more defensible: "The model correctly identifies its uncertainty on field images."

### Why This Is Not Cheating
- Temperature scaling uses only the validation set (no test labels)
- The model weights are unchanged
- Rejection is a standard operational strategy in safety-critical ML

### Files to Modify
- `evaluation/evaluator.py` — add temperature scaling and confidence thresholding
- `evaluation/metrics.py` — add calibration metrics (ECE, MCE)
- `web_app/server.py` — expose confidence threshold config

### Priority
- **Worth doing.** ~30 lines, no retraining, adds a professional reliability layer to the system.

## 4. Crop-First Prediction Cascade with Correctness Breakdown

### Problem
The 38-class problem space is very large for low-quality PlantDoc images. Predicting crop and disease simultaneously is harder than predicting them sequentially. Additionally, the current UI only shows a flat "both correct / crop only / disease only / none" breakdown in CSV, but it is not surfaced as a first-class metric in the web UI or PlantDoc tables.

### Idea
Implement a **crop-first cascade** at inference time:
1. Predict crop only by summing class probabilities across all diseases for each crop
2. Select the winning crop
3. Restrict the final disease prediction to diseases known to affect that crop

Additionally, update all result views (web UI, dashboard, PlantDoc tables) to prominently display the **both / crop-only / disease-only / none** correctness breakdown as a first-class metric matching the existing metric schema in `architecture_design.md §11`.

### Implementation
- Crop aggregation: `crop_probs = zeros(num_crops); for c in classes: crop_idx = class_to_crop[c]; crop_probs[crop_idx] += probs[c]`
- Disease restriction: create a mask of valid diseases per crop, zero out others before argmax
- Correctness breakdown: reuse `both_correct_pct`, `name_only_correct_pct`, `disease_only_correct_pct`, `none_correct_pct` from `metrics.py`
- Update `web_app/static/index.html` and `web_app/static/app.js` to show these 4 categories in the PlantDoc table and checkpoint explorer with color-coded badges

### Expected Gain
+2–5% accuracy on PlantDoc by reducing the effective class space. The correctness breakdown does not change accuracy numbers but makes the weakness easier to analyze and defend in a thesis.

### Why This Is Not Cheating
- No model weights modified
- No test labels used for selection
- Crop-first inference is a standard architectural trick used in hierarchical classification

### Files to Modify
- `evaluation/evaluator.py` — add crop-first cascade option
- `evaluation/metrics.py` — ensure crop/disease breakdown is computed for PlantDoc eval
- `web_app/static/app.js` — render correctness badges in PlantDoc table
- `web_app/static/index.html` — add correctness columns to PlantDoc table header

### Priority
- **Worth doing.** Zero retraining, improves both accuracy and analysis depth.

## 5. Test-Time Style Transfer (Color & Illumination Normalization)

### Problem
The largest domain gap between PlantVillage and PlantDoc is visual style: uniform lab backgrounds vs. real field lighting, shadows, and背景 clutter. Even when the leaf is correctly identified, color/shape distortions confuse the model.

### Idea
Apply **test-time style normalization** to each PlantDoc image before inference. Convert the image to match the average color distribution (mean/std per channel) of PlantVillage training images using simple OpenCV/`albumentations` histogram matching or Lab-color statistics. This is performed at inference only; the model is unchanged.

### Implementation
- Compute per-channel mean/std from the PlantVillage training set once (stats file under `experiments/results/`)
- At inference on PlantDoc: normalize image to those stats, or apply histogram matching in Lab space
- Option A: Add a preprocessing step in `evaluation/evaluator.py` that wraps the input tensor
- Option B: Add an `Albumentations` transform pipeline in `data/data_loader.py` that is used only during test/eval on out-of-distribution data
- Compare style-normalized predictions vs. raw predictions as an ablation

### Expected Gain
+3–6% accuracy on PlantDoc with zero retraining. Papers consistently show that color/style alignment alone bridges 20-30% of the lab-to-field gap.

### Why This Is Not Cheating
- No model weights modified
- The style statistics are computed from the source (PlantVillage) training data only
- No target-domain labels are accessed
- Style transfer as a test-time defense is a widely accepted domain adaptation technique

### Files to Modify
- `data/data_loader.py` — add style normalization transform for eval
- `evaluation/evaluator.py` — expose style-normalized evaluation mode
- `config/base_config.yaml` — add `eval.style_normalization: true/false`
- `web_app/server.py` — expose normalization toggle if needed

### Priority
- **Worth doing.** Moderate code change (~40 lines), no retraining, directly addresses the root cause of PlantDoc degradation.

## 6. Data Augmentation at Training Time (PlantVillage Only)
⚠️ **REQUIRES FULL RETRAINING OF ALL 3 MODELS (LoRA, QLoRA, QKLoRA)**

### Problem
The current training augmentations (random crop, flip, rotation ±15°, color jitter) are too mild to simulate real-world field conditions. The model never sees backgrounds, shadows, blur, or heavy occlusion during training, so PlantDoc images appear as a completely alien distribution.

### Idea
Replace or augment the existing training transform pipeline with **domain-randomization style** augmentations applied to PlantVillage images only. Simulate field conditions without adding any PlantDoc data:
- Random Gaussian blur (simulates out-of-focus field photos)
- Random coarse dropout / CutOut (simulates occlusion by leaves/insects)
- Random brightness/contrast/saturation shifts beyond standard jitter ranges
- Random noise injection (simulates low-light camera sensor noise)
- Random elastic transform / affine warp (simulates camera angle variation)
- Optional: Random background synthesis using texture/noise patches behind the leaf

### Implementation
- Modify `data/data_loader.py` training transforms to include:
  - `albumentations.GaussNoise`
  - `albumentations.GaussianBlur`
  - `albumentations.RandomBrightnessContrast` (heavier limits)
  - `albumentations.CLAHE` (simulates field lighting)
  - `albumentations.Cutout` (if `albumentations` available) or manual coarse dropout
- Re-run `python main.py lora`, `python main.py qlora`, `python main.py qklora`
- Re-evaluate all checkpoints with `launcher_test.py`
- Regenerate cross-method ranking and dashboard

### Expected Gain
Medium-to-high. Recent papers show extreme augmentation on source domain alone can lift zero-shot PlantDoc accuracy from ~20% to 35-40%. Success depends on how aggressively you warp the images without destroying the leaf structure.

### Risks
- Over-augmentation can hurt PlantVillage accuracy (your flagship metric)
- Need to balance augmentation strength so the 99%+ PlantVillage result is preserved
- All 3 training runs must be completed again (~3-4 hours GPU time each)

### Files to Modify
- `config/base_config.yaml` — add augmentation strength toggles
- `data/data_loader.py` — implement aggressive domain-randomization transforms
- `training/trainer.py` — ensure validation/PlantDoc eval uses clean (non-augmented) pipeline
- `web_app/server.py` — no changes needed

### Priority
- **High risk / High reward.** Best standalone method for improving PlantDoc, but requires full retraining pipeline. Do this only if you want PlantDoc to be a major thesis contribution.
