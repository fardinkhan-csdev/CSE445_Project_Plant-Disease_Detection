# Ablation Plan — Selective vs Full 1×1 Conv Targeting

**Related:** `architecture_design_v3.md` Section 10.2.3 (QLoRA Worth Doing table, "Selective vs. full-layer targeting" row) and Section 10.3.3 (QA-LoRA Worth Doing table, "Selective target layers" row).

## Background / Literature Findings

Relevant research on layer targeting for LoRA/QLoRA and CNN fine-tuning:

| Source | Finding | Relevance |
|--------|---------|-----------|
| **QLoRA paper** (Dettmers et al., NeurIPS 2023) | "LoRA on all transformer layers is critical to match 16-bit performance." Standard `q_proj`+`v_proj` only underperforms; full Linear-layer coverage is required for parity. | Directly supports our hypothesis that selective targeting may hurt accuracy. |
| **Amazon Science LoRA study** (2026) | Module combinations provide modest gains: `o_proj` + `fc2` typically scores 1–3pp above single-module LoRA. On hard benchmarks, target module choice has larger impact (+15pp over weaker baselines). | Suggests adding SE layers may give 1–3pp gain, not a massive jump. |
| **LoRA-vs-Full-Finetuning benchmark** | On policy generation: Q,V only = 84.8%; Q,K,V,O = 86.4%; +MLP = 87.6%; all-linear = 88.0%. Adding MLP layers added ~1.2pp for 2.7× more params. | Quantifies the marginal gain from expanding target set. MLP-like layers (SE is analogous) contribute but with diminishing returns. |
| **EfficientNet SE study** (Hoang & Jo, 2021) | SE modules improve EfficientNet-B0 ImageNet accuracy by ~0.4–1pp absolute. Removing all SE drops accuracy by ~1pp. | SE layers do carry task-relevant signal; excluding them may cost ~0.5–1pp. |
| **Flexora** (ACL 2025) | There is a critical point beyond which fine-tuning more layers causes overfitting and performance decline. Optimal subset selection matters. | Full targeting may help accuracy but risks overfitting; our decision rule should watch for overfit, not just underfit. |
| **Transfer Learning block selection** (Hasan et al., 2023) | On EfficientNet-B0, output-side blocks and FC layers consistently matter most for transfer learning. Lower block-importance (BI) blocks still yield high accuracy when selectively updated. | Supports selective Q-path + classifier targeting as a principled strategy, but does not rule out SE benefit. |

### Literature-grounded expectations for this ablation

- **Expected direction**: Full 1×1 (+SE) should be ≥ Q-path only. SE layers are channel-attention 1×1 convs; they learn task-specific channel recalibration that is frozen in the current setup.
- **Expected magnitude**: Based on the EfficientNet SE study and LoRA benchmark, expect **0.3–1.5pp accuracy gain** from adding SE. Gains >2pp would be surprising.
- **Overfit risk**: Adding SE roughly doubles the number of LoRA adapters on the MBConv blocks. Flexora warns that too many adapted layers can overfit. Monitor validation loss curves, not just final accuracy.
- **Parameter budget**: SE adds 4 small 1×1 convs per MBConv block. For EfficientNet-B0's 7 MBConv stages, this is ~28 additional LoRA adapter pairs — modest (~50–100k extra params depending on rank).

## Question
Does excluding SE layers and the stem from LoRA/QLoRA/QA-LoRA target sets hurt accuracy on EfficientNet-B0 + PlantVillage?

## Hypothesis
The paper's "target every Linear layer" recommendation was empirically motivated. We have not validated whether selective Q-path targeting causes similar degradation on EfficientNet-B0.

## What Changes

A single boolean `include_se` is threaded through the V3 pipeline:

| File | Change |
|------|--------|
| `models/peft/int8_utils.py:38` | `get_mbconv_q_path_names(model, include_se=False)` — when `True`, unions in SE layer names from `get_mbconv_k_path_se_names()` |
| `models/peft/lora_v3.py:20` | Add `include_se=False` param, forward to `get_mbconv_q_path_names` |
| `models/peft/qlora_v3.py:86` | Same |
| `models/peft/qalora.py:139` | Same |
| `training/lora_trainer_v3.py:11` | Add `include_se=False` to `__init__`, forward to model builder |
| `training/qlora_trainer_v3.py:10` | Same |
| `training/qalora_trainer.py:10` | Same |
| `experiments/experiment_runner_v3.py` | Thread `include_se` through `run_experiment` → trainer constructors |
| `main_v3.py` | Parse `--include-se` CLI flag, pass to `run_experiments` |

Total: ~46 lines across 8 files, all trivial parameter threading.

## How to Run

```bash
# Q-path only (current behavior, include_se=False)
python main_v3.py lora
python main_v3.py qlora
python main_v3.py qalora

# Full 1×1 including SE layers (include_se=True)
python main_v3.py --include-se lora
python main_v3.py --include-se qlora
python main_v3.py --include-se qalora
```

## What to Measure

Each run outputs a CSV at `experiments/results/experiment_results_v3_<timestamp>.csv` containing:

- `best_val_acc`
- `test_accuracy`
- `trainable_parameters`
- `training_time`
- `peak_gpu_memory`

## Comparison Table

After all 6 runs, build this table:

| Method | Strategy | Best Val Acc | Test Acc | Trainable Params | Training Time |
|--------|----------|-------------|-----------|------------------|---------------|
| LoRA V3 | Q-path only (current) | ? | ? | ~343k | ? |
| LoRA V3 | Full 1×1 (+SE) | ? | ? | ? | ? |
| QLoRA V3 | Q-path only (current) | ? | ? | ~343k | ? |
| QLoRA V3 | Full 1×1 (+SE) | ? | ? | ? | ? |
| QA-LoRA V3 | Q-path only (current) | ? | ? | ? | ? |
| QA-LoRA V3 | Full 1×1 (+SE) | ? | ? | ? | ? |

## Decision Rule

| Outcome | Action | Rationale |
|---------|--------|-----------|
| Full targeting **does not improve** accuracy (Δ ≤ 0) over selective | Keep selective targeting. Document that SE exclusion is justified on EfficientNet-B0 + PlantVillage. | Consistent with Hasan et al. finding that lower-BI blocks can be frozen without accuracy loss. |
| Full targeting **improves** accuracy by 0.3–1.5pp | Switch to full targeting. This matches literature expectations (SE study: ~0.4–1pp; LoRA benchmark: ~1.2pp from MLP layers). | The gain is meaningful but not transformative; parameter efficiency trade-off is acceptable. |
| Full targeting **improves** accuracy by >1.5pp | Switch to full targeting and investigate overfit risk. | Higher gains suggest the task benefits strongly from SE adaptation; verify with longer training or regularization. |
| Full targeting **hurts** accuracy or degrades validation loss (overfit) | Keep selective targeting. Document overfit as the justification. | Flexora shows excessive layer coverage causes overfit; selective targeting is the correct regularization. |

### Threshold justification

- **0pp threshold**: If SE layers carry no task-relevant signal for PlantVillage, selective targeting is optimal.
- **0.3–1.5pp band**: Matches literature range from SE removal studies (~0.4–1pp drop when SE is removed) and LoRA MLP-addition studies (~1.2pp gain). Gains in this range are "expected and worth the extra params."
- **>1.5pp**: Anomalously high; likely indicates the task strongly benefits from channel-attention adaptation. Worth further investigation into whether the gain persists with regularization.
- **Overfit criterion**: Validation loss divergence or >2pp train/val accuracy gap, not just test accuracy comparison.

## Effort Estimate

| Phase | Time |
|------|------|
| Code changes (8 files) | ~20–30 min |
| Sanity check (print both target lists) | ~10 min |
| 6 training runs (sequential, overnight) | 2–9 hours wall time, hands-off |
| Collect results from CSV | ~10 min |
| Update `architecture_design_v3.md` Section 10.3.3 Worth Doing verdict | ~5 min |
| **Total human time** | **~1 hour** |
| **Total GPU compute** | **~2–5 hours** |

## Notes

- The "extra" layers added by `include_se=True` are the SE squeeze/excite 1×1 convs (`*.fc1`, `*.fc2`). These are the only 1×1 convs excluded by `get_mbconv_q_path_names` but included in `get_pointwise_conv_names`.
- The stem (`features.0.0`) remains excluded in both strategies — this ablation only toggles SE inclusion, not stem inclusion. Stem inclusion would be a separate ablation.
- For QA-LoRA, `classifier.fc` is NOT automatically included in `target_modules` (unlike LoRA/QLoRA). If you want it included for fair comparison, add it to the QA-LoRA configs.
