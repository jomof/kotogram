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
python3 -m pip install torch==2.7.1

# Install other requirements
echo "Installing other dependencies..."
python3 -m pip install -r requirements.txt

# Add .local/bin to PATH for the duration of the script (fixes warnings)
export PATH="$HOME/.local/bin:$PATH"

# Install TypeScript dependencies
if [ -f "package.json" ]; then
    if ! command -v npm &> /dev/null; then
        echo "npm not found. Attempting to install..."
        if [ -n "$CONDA_PREFIX" ] || command -v conda &> /dev/null; then
             echo "Detected Conda environment. Installing nodejs..."
             conda install -y nodejs
        elif command -v brew &> /dev/null; then
             echo "Installing nodejs via Homebrew..."
             brew install node
        elif command -v apt-get &> /dev/null; then
             echo "Installing npm via apt-get (requires sudo)..."
             sudo apt-get update && sudo apt-get install -y npm
        else
             echo "ERROR: Could not install npm automatically. Please install Node.js manually."
             exit 1
        fi
    fi

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
