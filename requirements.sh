#!/bin/bash
# Setup script for kotogram training environment
# Run with: ./requirements.sh (or source requirements.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Kotogram Environment Setup"
echo "========================================"

# Install PyTorch
echo "Installing PyTorch..."
python3 -m pip install torch==2.6.0

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
