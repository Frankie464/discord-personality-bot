#!/bin/bash
# Training Wrapper Script
#
# Convenience wrapper for training that automatically uses venv_training.
# This ensures the CUDA-enabled PyTorch environment is used.
#
# Usage:
#     bash scripts/train.sh
#     bash scripts/train.sh --mode dpo
#     bash scripts/train.sh --epochs 10

# Change to project root
cd "$(dirname "$0")/.."

# Check if venv_training exists
if [ ! -f "venv_training/bin/python" ] && [ ! -f "venv_training/Scripts/python.exe" ]; then
    echo
    echo "============================================================"
    echo "ERROR: Training environment not found!"
    echo "============================================================"
    echo
    echo "The venv_training virtual environment has not been set up yet."
    echo "This environment is required for GPU training with CUDA support."
    echo
    echo "To set it up, run:"
    echo "    python scripts/setup_training_environment.py"
    echo
    echo "This will install PyTorch with CUDA, Unsloth, and all training dependencies."
    echo "============================================================"
    echo
    exit 1
fi

# Show which environment we're using
echo
echo "============================================================"
echo "Using Training Environment: venv_training"
echo "============================================================"
echo

# Determine Python path (Linux/Mac vs Windows Git Bash)
if [ -f "venv_training/bin/python" ]; then
    PYTHON="venv_training/bin/python"
else
    PYTHON="venv_training/Scripts/python.exe"
fi

# Run training script with venv_training Python
$PYTHON scripts/3_train_model.py "$@"

# Capture exit code
EXITCODE=$?

# Show completion message
echo
if [ $EXITCODE -eq 0 ]; then
    echo "============================================================"
    echo "Training completed successfully!"
    echo "============================================================"
else
    echo "============================================================"
    echo "Training failed with exit code: $EXITCODE"
    echo "============================================================"
fi
echo

exit $EXITCODE
