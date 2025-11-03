@echo off
REM Training Wrapper Script
REM
REM Convenience wrapper for training that automatically uses venv_training.
REM This ensures the CUDA-enabled PyTorch environment is used.
REM
REM Usage:
REM     scripts\train.bat
REM     scripts\train.bat --mode dpo
REM     scripts\train.bat --epochs 10

REM Change to project root
cd /d "%~dp0\.."

REM Check if venv_training exists
if not exist "venv_training\Scripts\python.exe" (
    echo.
    echo ============================================================
    echo ERROR: Training environment not found!
    echo ============================================================
    echo.
    echo The venv_training virtual environment has not been set up yet.
    echo This environment is required for GPU training with CUDA support.
    echo.
    echo To set it up, run:
    echo     python scripts/setup_training_environment.py
    echo.
    echo This will install PyTorch with CUDA, Unsloth, and all training dependencies.
    echo ============================================================
    echo.
    exit /b 1
)

REM Show which environment we're using
echo.
echo ============================================================
echo Using Training Environment: venv_training
echo ============================================================
echo.

REM Run training script with venv_training Python
venv_training\Scripts\python.exe scripts\3_train_model.py %*

REM Capture exit code
set EXITCODE=%ERRORLEVEL%

REM Show completion message
echo.
if %EXITCODE%==0 (
    echo ============================================================
    echo Training completed successfully!
    echo ============================================================
) else (
    echo ============================================================
    echo Training failed with exit code: %EXITCODE%
    echo ============================================================
)
echo.

exit /b %EXITCODE%
