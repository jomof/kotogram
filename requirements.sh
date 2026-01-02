#!/bin/bash
# Setup script for kotogram training environment
# Run with: ./requirements.sh (or source requirements.sh)


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Save original prompt to prevent clobbering by activate script
ORIG_PS1="${PS1:-}"


if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Using existing virtual environment: $VIRTUAL_ENV"
else
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Creating virtual environment in .venv..."
        python3 -m venv .venv
    fi
    
    # Disable default venv prompt change, we'll restore user's prompt if needed
    VIRTUAL_ENV_DISABLE_PROMPT=1 source .venv/bin/activate
fi

# Restore prompt if it was cleared or modified
if [ -n "$ORIG_PS1" ]; then
    PS1="$ORIG_PS1"
    # Prepend (venv) if not already present
    if [[ "$PS1" != *"(venv) "* ]]; then
        PS1="(venv) $PS1"
    fi
fi

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

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "You can now run: ./train_style"
