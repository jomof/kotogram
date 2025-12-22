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
run_quiet pip install ruff mypy pytest vulture

# --- Setup TypeScript Environment ---
if [ -f "package.json" ]; then
    run_quiet npm install
fi

# --- Run Checks ---

# Record initial git status
INITIAL_GIT_STATUS=$(git status --short)

log "Running ruff check and format..."
ruff check --fix . --config pyproject.toml
ruff format .

log "Running mypy..."
# We run mypy on the main package and scripts. 
# We use --explicit-package-bases to handle the 'scripts' directory without __init__.py.
mypy kotogram scripts --explicit-package-bases

log "Running vulture..."
vulture kotogram scripts tests-py scripts/vulture_whitelist.py

log "Running Python unittests..."
python -m pytest tests-py/

if [ -f "package.json" ]; then
    log "Running TypeScript unittests..."
    run_quiet npm run build
    npm test
fi

# Verify git status hasn't changed
FINAL_GIT_STATUS=$(git status --short)
if [ "$INITIAL_GIT_STATUS" != "$FINAL_GIT_STATUS" ]; then
    log "Comparing git status..."
    diff <(echo "$INITIAL_GIT_STATUS") <(echo "$FINAL_GIT_STATUS") || true
    error "git status changed during tests. New/changed files detected in repository."
fi

log "All checks passed successfully!"
