#!/bin/bash
set -euo pipefail

# --- Configuration ---
VENV_DIR=".venv"

# --- Setup Python Environment ---
if [ -z "${CI:-}" ] && [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Determine python command
PYTHON_CMD="python3"

# Delegate to the Python test runner
# We pass --confinement-config here as a default for the full suite context
# The runner handles argument parsing including --specific-python-test
exec $PYTHON_CMD scripts/test_runner.py "$@"
