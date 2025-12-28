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
    NEED_NODE_INSTALL=true
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v | cut -d. -f1 | tr -d 'v')
        if [ "$NODE_VERSION" -ge 18 ]; then
            NEED_NODE_INSTALL=false
        else
            echo "Node.js version v$NODE_VERSION is too old (requires >= 18). Upgrading..."
        fi
    else
        echo "Node.js not found..."
    fi

    if [ "$NEED_NODE_INSTALL" = true ]; then
        if [ -n "$CONDA_PREFIX" ] || command -v conda &> /dev/null; then
             echo "Detected Conda environment. Installing nodejs >= 18..."
             # Use conda-forge to ensure newer versions are available if default channel is stale
             conda install -y -c conda-forge nodejs>=18
        elif command -v brew &> /dev/null; then
             echo "Installing nodejs via Homebrew..."
             brew install node
             brew upgrade node
        elif command -v apt-get &> /dev/null; then
             echo "Installing latest nodejs via npm/n..."
             # Try to update via npm n if possible, or warn
             if command -v npm &> /dev/null; then
                 sudo npm install -g n
                 sudo n stable
             else
                 echo "Installing npm via apt-get..."
                 sudo apt-get update && sudo apt-get install -y npm
             fi
        else
             echo "ERROR: Could not install/upgrade Node.js automatically."
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
