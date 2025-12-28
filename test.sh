#!/bin/bash
set -euo pipefail

echo "Starting test.sh..."

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
# In CI, we run verbose to debug issues
run_quiet() {
    if [ -n "${CI:-}" ]; then
        echo "Running: $*"
        "$@"
        return $?
    fi

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
        # If venv doesn't exist, we can't really "rely on requirements.sh" if we create a fresh one here.
        # But per instructions, we just assume requirements.sh was run or we are in the right env.
        # However, to be helpful, if we activate a venv, we expect it to feature the deps.
        # I will leave the VENV activation logic but remove the installation.
        # If the user hasn't run requirements.sh inside this venv, it will fail, which matches "rely on requirements.sh".
        :
    fi

    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi
fi

# Dependencies are assumed to be installed via requirements.sh


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

             # Dependencies must be installed via requirements.sh
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
