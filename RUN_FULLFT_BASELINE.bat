@echo off
echo ============================================================
echo   Full Fine-Tuning Baseline (1 epoch) - PlantVillage
echo ============================================================
echo.

cd /d "D:\Leaf Disease Classification"

echo [1/2] Training (1 epoch, ~9 min)...
py -3.11 run_fullft_baseline.py
if errorlevel 1 (
    echo TRAINING FAILED
    pause
    exit /b 1
)

echo.
echo [2/2] Evaluating checkpoint...
py -3.11 eval_fullft_baseline.py

echo.
echo ============================================================
echo   Done. Check experiments\results\eval_v3\fullft_summary.json
echo ============================================================
pause
