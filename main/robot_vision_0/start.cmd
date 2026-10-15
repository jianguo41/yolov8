@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if defined ROBOT_PYTHON (
    "%ROBOT_PYTHON%" -u app.py %*
) else if exist "D:\anaconda3\envs\yolov8\python.exe" (
    "D:\anaconda3\envs\yolov8\python.exe" -u app.py %*
) else (
    python -u app.py %*
)
if errorlevel 1 pause
endlocal
