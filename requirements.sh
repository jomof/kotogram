#!/bin/bash
# Setup script for kotogram training environment
# Run with: source requirements.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Kotogram Environment Setup"
echo "========================================"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Detect CUDA
echo "Detecting GPU/CUDA environment..."
TORCH_INDEX=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
    if [ -n "$CUDA_VERSION" ]; then
        echo "  Found CUDA version: $CUDA_VERSION"
        # Map CUDA version to PyTorch index URL
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            if [ "$CUDA_MINOR" -ge 4 ]; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu124"
                echo "  Using PyTorch with CUDA 12.4 support"
            else
                TORCH_INDEX="https://download.pytorch.org/whl/cu121"
                echo "  Using PyTorch with CUDA 12.1 support"
            fi
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            echo "  Using PyTorch with CUDA 11.8 support"
        fi
    fi
fi

if [ -z "$TORCH_INDEX" ]; then
    echo "  No CUDA detected, installing CPU-only PyTorch"
fi

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch..."
if [ -n "$TORCH_INDEX" ]; then
    pip install torch --index-url "$TORCH_INDEX"
else
    pip install torch
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install the project in editable mode
echo "Installing kotogram in editable mode..."
pip install -e .

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Virtual environment is now active."
echo "You can now run: ./train_style"
echo ""
echo "To reactivate later: source .venv/bin/activate"
