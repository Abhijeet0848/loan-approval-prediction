@echo off
title Retrain Loan Models
echo ===================================================
echo   Retraining Loan Approval Prediction Models...
echo ===================================================
echo.

if exist ".\venv\Scripts\python.exe" (
    ".\venv\Scripts\python.exe" train.py
) else (
    python train.py
)

echo.
pause
