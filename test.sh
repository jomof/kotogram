#!/bin/bash
set -euo pipefail

# Cursor often runs commands like: your_script | tail -40
# That truncates stdout. But stderr is typically shown in full.
# If stdout is not a TTY (piped/redirected), clone stdout to stderr too.
if [[ ! -t 1 ]]; then
  exec > >(tee /dev/fd/2)
fi

# --- Configuration ---
VENV_DIR=".venv"

# --- Setup Python Environment ---
if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

# --- Verify venv is active ---
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Error: Not running in a Python virtual environment."
  echo "Please run: source requirements.sh"
  exit 1
fi

PYTHON_CMD="python3"
exec "$PYTHON_CMD" scripts/test_runner.py "$@"
