# 1st PlantDoc Test Result — Raw Images, No Segmentation

> Baseline evaluated with original PlantDoc images (no foreground removal).
> Checkpoints used: `*_best.pth` for each method.

## Cross-Split Accuracy

| Method | Split   | Accuracy (%) | Samples |
|--------|---------|:------------:|:-------:|
| LORA   | train   | 20.57        | 1818    |
| LORA   | valid   | 17.65        | 289     |
| LORA   | test    | 25.69        | 218     |
| QLORA  | train   | 15.07        | 1818    |
| QLORA  | valid   | 10.38        | 289     |
| QLORA  | test    | 18.81        | 218     |
| QKLORA | train   | 17.88        | 1818    |
| QKLORA | valid   | 15.22        | 289     |
| QKLORA | test    | 21.10        | 218     |

## Test-Set Summary (218 images)

| Rank | Method | Best Checkpoint | Test Accuracy |
|:----:|--------|-----------------|:-------------:|
| 1    | LORA   | lora_best.pth   | 25.69%        |
| 2    | QKLORA | qklora_best.pth | 21.10%        |
| 3    | QLORA  | qlora_best.pth  | 18.81%        |

## Notes

- Evaluation uses the best checkpoint per method (`*_best.pth`).
- PlantDoc images contain real-world field conditions (lighting, background clutter, occlusion).
- PlantVillage training images have clean uniform backgrounds.
- This baseline shows the raw domain gap without any test-time adaptation.
