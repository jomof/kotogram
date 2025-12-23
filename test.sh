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
if [ ! -d "$VENV_DIR" ]; then
    run_quiet $PYTHON_CMD -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

run_quiet pip install --upgrade pip
run_quiet pip install -e .
run_quiet pip install ruff mypy pytest vulture build

# --- Setup TypeScript Environment ---
if [ -f "package.json" ]; then
    run_quiet npm install
fi

# --- Run Checks ---

# Record initial git status
INITIAL_GIT_STATUS=$(git status --short)

run_quiet ruff check --fix . --config pyproject.toml && run_quiet ruff format .
success "Ruff check and format passed"

run_quiet mypy kotogram scripts --explicit-package-bases
success "Mypy passed"

run_quiet vulture kotogram scripts tests-py scripts/vulture_whitelist.py
success "Vulture passed"

if [ -f "package.json" ]; then
    run_quiet npm run fix && run_quiet npm test
    success "TypeScript checks passed"
fi



# Clean dist to ensure fresh builds
rm -rf dist
run_quiet $PYTHON_CMD -m build

# Extract filenames, remove header/footer, sort, and normalize version
unzip -l dist/*.whl 2>/dev/null | \
awk '{print $4}' | \
grep -v "Name" | \
grep -v "\-\-\-\-" | \
grep -v "^\s*$" | \
sed 's/kotogram-.*\.dist-info/kotogram-*.dist-info/g' | \
LC_ALL=C sort > /tmp/package_files.txt

if ! diff -u tests/python_package_baseline.txt /tmp/package_files.txt > /dev/null; then
    diff -u tests/python_package_baseline.txt /tmp/package_files.txt
    error "Python package contents do not match baseline!"
fi
success "Python package verification passed"

if [ -f "package.json" ]; then
    # Ensure fresh dist for TS build
    rm -rf dist
    run_quiet npm run build

    # Create package and list contents (quietly to get just filename)
    PACK_FILE=$(npm pack --quiet | tail -n 1)

    # List files, strip 'package/' prefix, sort
    tar -tf "$PACK_FILE" | sed 's/^package\///' | LC_ALL=C sort > /tmp/ts_package_files.txt

    if ! diff -u tests/typescript_package_baseline.txt /tmp/ts_package_files.txt > /dev/null; then
        diff -u tests/typescript_package_baseline.txt /tmp/ts_package_files.txt
        error "TypeScript package contents do not match baseline!"
    fi
    rm "$PACK_FILE"
    success "TypeScript package verification passed"
fi

python -m pytest --no-header tests-py/

# Verify git status hasn't changed
FINAL_GIT_STATUS=$(git status --short)
if [ "$INITIAL_GIT_STATUS" != "$FINAL_GIT_STATUS" ]; then
    diff <(echo "$INITIAL_GIT_STATUS") <(echo "$FINAL_GIT_STATUS") || true
    error "git status changed during tests. New/changed files detected in repository."
fi
success "Git status clean"

success "All checks passed successfully!"
