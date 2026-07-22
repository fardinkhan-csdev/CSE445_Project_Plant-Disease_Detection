# Future Features Reference

Items already implemented in the main pipeline have been removed from this file.

---

## 1. Cross-method experiment ranking script

### Problem
`launcher_test.py` ranks checkpoints **within one experiment** (e.g. `lora_best` vs `lora_last`). After all three methods finish training, there is no single script that answers: **which method won overall?**

### Proposed script: `rank_experiments.py` (name TBD)

Run once after LoRA, QLoRA, and Q/K LoRA have all been trained and individually tested:

```
python rank_experiments.py
```

### Inputs
- `experiments/results/experiment_results.csv` — one row per method (train time, params, test accuracy, etc.)
- `experiments/results/eval/lora_checkpoint_ranking.csv`
- `experiments/results/eval/qlora_checkpoint_ranking.csv`
- `experiments/results/eval/qklora_checkpoint_ranking.csv`

### Logic
1. For each method, pick the **best checkpoint** (rank = 1 from that method's ranking CSV, or `*_best.pth` as fallback).
2. Compare the three winners on a fixed scorecard:
   - Primary: `test_accuracy` (multiclass)
   - Tie-breakers: `f1_macro`, `binary_f1`, `both_correct_pct`, `peak_gpu_memory` (lower is better), `trainable_parameters` (lower is better)
3. Assign an overall `rank` (1 = best method).

### Outputs
- `experiments/results/eval/cross_method_ranking.csv` — one row per method, best checkpoint only
- Optional: append a "Method Comparison" section/tab to `dashboard.html` via `generate_dashboard.py`

### Opinion
- **Worth doing.** This is the natural capstone after the three training runs and keeps evaluation separate from training (GPU-friendly).
- Should be a **small, read-only script** — no model loading, just CSV aggregation. Fast and safe to run anytime.
- Do **not** fold this into `launcher_test.py`; keep per-method testing and cross-method comparison as two distinct steps.
- Re-run `launcher_test.py` (evaluate-all mode) for each method first so per-method ranking CSVs exist; then run the cross-method script.

---

## 2. GUI prototype: image upload and prediction summary

### Purpose
- Allow real-world test images from Google or other sources
- Show the model's predicted crop name and disease label
- Provide a simple interface for reference testing

### Features
- Upload a JPEG/PNG image
- Preprocess the image to the model's input format
- Run the selected checkpoint or model
- Display results:
  - predicted full class (`Crop___Disease`)
  - predicted crop name
  - predicted disease label
  - binary diseased/healthy output
  - optionally confidence scores

### UI components
- File upload button
- Preview of the uploaded image
- Prediction result panel:
  - `Predicted label:`
  - `Crop:`
  - `Disease:`
  - `Healthy/diseased:`
  - `Confidence:`

### Implementation options
- Lightweight web app using Flask or FastAPI (or simple HTML)
- Local desktop app using a simple GUI toolkit (e.g., Tkinter)
- Save test results to a log or CSV for manual review

### Opinion
- Useful for demos and thesis defense.
- `config/class_labels.json` is now auto-exported; wire checkpoints from `experiments/results/checkpoints/<method>_best.pth`.
- Should use the winning checkpoint from `cross_method_ranking.csv` once that exists (see §1 above).

---

## 3. Dashboard fixes

### Known issues (from architecture doc)
- Dashboard has bugs and is missing image-test integration
- Regenerate after all three experiments: `python generate_dashboard.py`

### Future work
- Add cross-method comparison tab (depends on §1 above)
- Wire in single-image inference preview (depends on §2 above)

---

## 4. Notes

- This file is a reference for future implementation and does not change current training or evaluation code.
- Per-method checkpoint ranking columns (`rank`, `size_mb`, `disease_only_correct_pct`) require re-running `launcher_test.py` → evaluate-all — old CSVs are not auto-updated.
