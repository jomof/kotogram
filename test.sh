#!/bin/bash
set -euo pipefail

if [[ ! -t 1 ]] || [[ ! -t 2 ]]; then
  echo "Don't redirect stdout/stderr, the user can't see the output if you do. You can use --hygiene, --no-hygiene, or --pytests * to limit what gets run."
  exit 1
fi

# --- Configuration ---
VENV_DIR=".venv"

# --- Setup Python Environment ---
# --- Setup Python Environment ---
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Determine python command
PYTHON_CMD="python3"

# Delegate to the Python test runner
# We pass --confinement-config here as a default for the full suite context
# The runner handles argument parsing including --specific-python-test
exec $PYTHON_CMD scripts/test_runner.py "$@"
