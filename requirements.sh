#!/bin/bash
# Setup script for kotogram training environment
# Run with: ./requirements.sh (or source requirements.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Kotogram Environment Setup"
echo "========================================"

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

# Show which python we're using
echo ""
echo "Installing to: $(which python3)"
echo ""

# Install/upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch..."
if [ -n "$TORCH_INDEX" ]; then
    python3 -m pip install torch==2.9.1 --index-url "$TORCH_INDEX"
else
    python3 -m pip install torch==2.9.1
fi

# Install other requirements
echo "Installing other dependencies..."
python3 -m pip install -r requirements.txt

# Install TypeScript dependencies
if [ -f "package.json" ]; then
    echo "Installing TypeScript dependencies..."
    npm install
fi

# Install the project in editable mode
echo "Installing kotogram in editable mode..."
python3 -m pip install -e .

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "You can now run: ./train_style"
