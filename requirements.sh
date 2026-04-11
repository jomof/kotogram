#!/bin/bash
# Setup script for kotogram training environment
# Run with: source requirements.sh

# Fail if not sourced (environment changes would be lost in a subshell)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: This script must be sourced, not executed directly."
  echo "Please run: source requirements.sh"
  exit 1
fi

# Fast path: already in a venv, skip setup
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -f "$VIRTUAL_ENV/bin/python" ]]; then
  echo "Already in virtual environment: $VIRTUAL_ENV"
  return 0
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Save original prompt to prevent clobbering by activate script
ORIG_PS1="${PS1:-}"

# Reject user site-packages to ensure strictly isolated environments
export PYTHONNOUSERSITE=1


if [[ -n "$VIRTUAL_ENV" ]] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    VENV_PY_VER=$("$VIRTUAL_ENV/bin/python" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ "$VENV_PY_VER" == "3.10" ]]; then
        echo "Using existing virtual environment: $VIRTUAL_ENV"
        PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
    else
        echo "Existing venv is Python $VENV_PY_VER, but Python 3.10 is required."
        echo "Please deactivate the current virtual environment (run 'deactivate') and re-run this script."
        return 1
    fi
else
        # If .venv exists, verify its version
        if [ -d ".venv" ]; then
            VENV_PY_VER=$(.venv/bin/python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [[ "$VENV_PY_VER" != "3.10" ]]; then
                echo "Removing existing .venv because it uses Python $VENV_PY_VER instead of 3.10..."
                rm -rf .venv
            fi
        fi

        if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
            echo "Creating virtual environment in .venv..."
        PYTHON_BASE=""
        for candidate in python3.10 /opt/homebrew/bin/python3.10 /usr/local/bin/python3.10 python3; do
            if command -v "$candidate" &>/dev/null; then
                VER=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
                if [ "$VER" == "3.10" ]; then
                    PYTHON_BASE="$candidate"
                    break
                fi
            fi
        done
        
        if [ -z "$PYTHON_BASE" ]; then
            echo "ERROR: Python 3.10 not found."
            echo "Please install it (e.g. brew install python@3.10 on macOS, or apt install python3.10)."
            exit 1
        fi
        
        # If .venv exists but it's not exactly 3.10, Python's venv module will safely replace it 
        # or we could explicitly remove it here. Let's explicitly remove.
        if [ -d ".venv" ]; then
            rm -rf .venv
        fi
        
        "$PYTHON_BASE" -m venv .venv
    fi
    
    # Disable default venv prompt change, we'll restore user's prompt if needed
    VIRTUAL_ENV_DISABLE_PROMPT=1 source .venv/bin/activate
    PYTHON_EXEC=".venv/bin/python"
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
"$PYTHON_EXEC" -m pip install torch==2.7.1

# Install Ruff
echo "Installing Ruff..."
"$PYTHON_EXEC" -m pip install ruff==0.14.10

# Install Rich
echo "Installing Rich..."
"$PYTHON_EXEC" -m pip install rich==14.2.0

# Install other requirements
echo "Installing other dependencies..."
"$PYTHON_EXEC" -m pip install -r requirements.txt

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
             echo "WARNING: Could not install/upgrade Node.js automatically."
             echo "         TypeScript compilation will be skipped."
             NEED_NODE_INSTALL=false
        fi
    fi

    if [ "$NEED_NODE_INSTALL" != false ]; then
        echo "Installing TypeScript dependencies..."
        npm install
    fi
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "You can now run: ./train_style"
