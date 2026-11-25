@echo off
title Video Anomaly Detection - Full Stack System
cd /d D:\VideoAnomalyDetection
echo ============================================================
echo   Starting Video Anomaly Detection System (Full Stack)
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_all.py %*
) else (
    python run_all.py %*
)

pause
