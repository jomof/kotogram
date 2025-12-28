#!/bin/bash
set -euo pipefail

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

success() {
    echo -e "\033[1;32m✅\033[0m $1"
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
# --- Setup Python Environment ---
if [ -z "${CI:-}" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        run_quiet $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
fi

run_quiet pip install --upgrade pip
run_quiet pip install -e .
run_quiet pip install ruff mypy pytest vulture build pylint

# --- Setup TypeScript Environment ---
if [ -f "package.json" ]; then
    run_quiet npm install
fi

# --- Run Checks ---

# --- Argument Parsing ---
SPECIFIC_TEST=""

# Parse args to find --specific-python-test
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --specific-python-test)
            SPECIFIC_TEST="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            ARGS+=("$1")
            shift # past argument
            ;;
    esac
done

# If running a specific test, bypass all maintenance/hygiene checks
if [ -n "$SPECIFIC_TEST" ]; then
    log "Running specific python test: $SPECIFIC_TEST"
    # We still need the environment
    if [ -z "${CI:-}" ]; then
        if [ ! -d "$VENV_DIR" ]; then
             log "Creating venv for specific test..."
             run_quiet $PYTHON_CMD -m venv "$VENV_DIR"
             source "$VENV_DIR/bin/activate"
             run_quiet pip install --upgrade pip
             run_quiet pip install -e .
             run_quiet pip install ruff mypy pytest vulture build pylint
        else
             source "$VENV_DIR/bin/activate"
        fi
    fi
    
    # Just run the requested test module using unittest
    export PYTHONPATH="tests-py:${PYTHONPATH:-}"
    exec python3 -m unittest "$SPECIFIC_TEST"
    exit 0
fi

# --- Full Test Suite ---

# Record initial git status (captured in python now, but let's just delegate)
exec $PYTHON_CMD scripts/test_runner.py --confinement-config confine/python-test.json "${ARGS[@]+"${ARGS[@]}"}"
