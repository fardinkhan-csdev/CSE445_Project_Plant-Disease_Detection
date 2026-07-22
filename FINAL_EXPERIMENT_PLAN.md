# Final Experiment Execution & Evaluation Plan

This document details the step-by-step roadmap for consolidating experiment results, evaluating trained checkpoints, training/testing Q/K LoRA, and generating the final 3-method comparative scorecard.

---

## Plan Verification: **SOLID & 100% AIRTIGHT** ✅

The plan preserves all previously trained LoRA and QLoRA weights, cleans up legacy/duplicate files, generates per-checkpoint confidence rankings for all methods, trains the missing Q/K LoRA method, and concludes with a unified 3-way scorecard.

---

## Execution Checklist

### Phase 1: CSV Consolidation & Workspace Cleanup — **COMPLETED** ✅
- [x] **Copy Backup CSV**: Overwritten `experiments/results/experiment_results.csv` with `experiment_results_lora_dashboard_plus_latest_qlora.csv`.
- [x] **Delete Outdated CSV Files**:
  - Removed `experiments/results/experiment_results2.csv`
  - Removed `experiments/results/experiment_results_OUTDATED.csv`
  - Removed `experiments/results/checkpoint_rankings.csv`

---

### Phase 2: Test Evaluation for LoRA & QLoRA — **COMPLETED** ✅
- [x] **Re-Evaluate LoRA Checkpoints**: Generated `experiments/results/eval/lora_checkpoint_ranking.csv` & confidence CSVs.
- [x] **Evaluate QLoRA Checkpoints**: Generated `experiments/results/eval/qlora_checkpoint_ranking.csv` & confidence CSVs.

---

### Phase 3: Train & Test Q/K LoRA — **COMPLETED** ✅
- [x] **Train Q/K LoRA**: Completed training (`experiment_results.csv` row appended: 444.5k params, 2980s train time).
- [x] **Evaluate Q/K LoRA Checkpoints**: Generated `experiments/results/eval/qklora_checkpoint_ranking.csv` & confidence CSVs.

---

### Phase 4: Final 3-Method Cross-Method Ranking — **COMPLETED** ✅
- [x] **Generate Final Scorecard**: Executed `rank_experiments.py` to produce `experiments/results/eval/cross_method_ranking.csv` across LoRA, QLoRA, and Q/K LoRA.
- [x] **View Results in Web UI**: Web server active at `http://localhost:8000`.

---

## Additional Evaluation Strategy for PlantVillage → PlantDoc Transfer

To evaluate the real-world transfer effect properly, the final analysis should use two PlantDoc evaluation views:

1. **Primary evaluation:** PlantDoc test split
   - This is the standard and most defensible evaluation set for reporting generalization.
   - It measures how the model performs on held-out real-world examples.

2. **Secondary comparison:** PlantDoc train split
   - This is useful as an additional real-world comparison set.
   - It provides a broader sample of PlantDoc-style images and helps show whether performance is consistent across a larger pool.

### Recommended interpretation
- Report the **PlantDoc test split** as the main result.
- Use the **PlantDoc train split** as a supplementary analysis, not as the main benchmark.
- Keep the two results separate in the report so the comparison is clear and interpretable.

---

## Label Alignment Strategy for PlantVillage and PlantDoc

Because PlantVillage and PlantDoc were built for different tasks and use different label taxonomies, the evaluation set must be filtered to only the labels that are meaningfully shared.

### Principle
- Do **not** force a weak or incorrect label match just because two classes look loosely related.
- Only keep mappings that are semantically clear and biologically consistent.
- If a class exists in only one dataset, it should be **excluded** from the shared evaluation subset.

### What to do with uncommon labels
Labels that are uncommon or appear in only one dataset should be treated as **unsupported for the shared cross-dataset benchmark**. They should not be forced into the main evaluation set because that would introduce label noise, reduce interpretability, and make the accuracy metric less trustworthy. In a professional experiment, the correct approach is to:
- remove those labels from the common evaluation subset,
- keep them in a separate exclusion list for transparency,
- and report that the final comparison is based only on the overlapping, clinically/visually consistent classes.

This is the cleanest and most scientifically defensible approach.

---

## Proposed Shared Label Mapping (PlantVillage → PlantDoc)

The following mappings are logical and keep only the common or near-common classes. Labels that are unique to one dataset are intentionally skipped.

| PlantVillage label | PlantDoc label |
| :--- | :--- |
| `Apple___Apple_scab` | `Apple Scab Leaf` |
| `Apple___Cedar_apple_rust` | `Apple rust leaf` |
| `Apple___healthy` | `Apple leaf` |
| `Blueberry___healthy` | `Blueberry leaf` |
| `Cherry_(including_sour)___healthy` | `Cherry leaf` |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn Gray leaf spot` |
| `Corn_(maize)___Common_rust_` | `Corn rust leaf` |
| `Corn_(maize)___Northern_Leaf_Blight` | `Corn leaf blight` |
| `Peach___healthy` | `Peach leaf` |
| `Pepper,_bell___Bacterial_spot` | `Bell_pepper leaf spot` |
| `Pepper,_bell___healthy` | `Bell_pepper leaf` |
| `Potato___Early_blight` | `Potato leaf early blight` |
| `Potato___Late_blight` | `Potato leaf late blight` |
| `Potato___healthy` | `Potato leaf` |
| `Raspberry___healthy` | `Raspberry leaf` |
| `Soybean___healthy` | `Soybean leaf` |
| `Squash___Powdery_mildew` | `Squash Powdery mildew leaf` |
| `Strawberry___healthy` | `Strawberry leaf` |
| `Tomato___Bacterial_spot` | `Tomato leaf bacterial spot` |
| `Tomato___Early_blight` | `Tomato Early blight leaf` |
| `Tomato___Late_blight` | `Tomato leaf late blight` |
| `Tomato___Septoria_leaf_spot` | `Tomato Septoria leaf spot` |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato leaf yellow virus` |
| `Tomato___Tomato_mosaic_virus` | `Tomato leaf mosaic virus` |
| `Tomato___healthy` | `Tomato leaf` |

### Labels intentionally skipped
These are not included in the shared benchmark because they are either unique to one dataset or too weakly aligned to be confidently mapped:
- `Apple___Black_rot`
- `Grape___Black_rot`
- `Grape___Esca_(Black_Measles)`
- `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
- `Grape___healthy`
- `Orange___Haunglongbing_(Citrus_greening)`
- `Strawberry___Leaf_scorch`
- `Tomato___Leaf_Mold`
- `Tomato___Target_Spot`
- `Tomato___Spider_mites Two-spotted_spider_mite`
- `Tomato___healthy` is included only because the PlantDoc taxonomy has a broad `Tomato leaf` class that is semantically acceptable for a healthy-leaf evaluation.

---

## Final Recommendation
- Train on PlantVillage. **COMPLETED**
- Evaluate on PlantDoc using the shared label subset only.
- Report PlantDoc test performance as the primary transfer result.
- Use PlantDoc train performance as a secondary comparison only, not primary.
- Exclude uncommon or mismatched labels from the main benchmark to keep the evaluation clean and professional.
