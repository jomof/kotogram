#!/bin/bash
set -e

# --- Configuration ---
VENV_DIR=".venv"
PYTHON_CMD="python3"

# --- Functions ---
log() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
    exit 1
}

# Run a command quietly, only showing output on failure
run_quiet() {
    local tmpfile
    tmpfile=$(mktemp /tmp/kotogram_setup_XXXXXX)
    if ! "$@" > "$tmpfile" 2>&1; then
        cat "$tmpfile"
        rm "$tmpfile"
        return 1
    fi
    rm "$tmpfile"
    return 0
}

# --- Setup Python Environment ---
if [ ! -d "$VENV_DIR" ]; then
    run_quiet $PYTHON_CMD -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

run_quiet pip install --upgrade pip
run_quiet pip install -e .
run_quiet pip install ruff mypy pytest

# --- Setup TypeScript Environment ---
if [ -f "package.json" ]; then
    run_quiet npm install
fi

# --- Run Checks ---
FAILED=0

log "Running ruff check..."
ruff check . --config pyproject.toml || FAILED=1

log "Running mypy..."
# We run mypy on the main package and scripts. 
# We use --explicit-package-bases to handle the 'scripts' directory without __init__.py.
mypy kotogram scripts --explicit-package-bases || FAILED=1

log "Running Python unittests..."
python -m pytest tests-py/ || FAILED=1

if [ -f "package.json" ]; then
    log "Running TypeScript unittests..."
    run_quiet npm run build
    npm test || FAILED=1
fi

if [ $FAILED -eq 0 ]; then
    log "All checks passed successfully!"
    exit 0
else
    error "Some checks failed. Please see the output above."
    exit 1
fi
