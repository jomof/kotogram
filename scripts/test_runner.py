"""
Test Runner Script for Kotogram.

This script acts as the primary CI/CD entry point and local development verification tool.
It orchestrates a suite of checks including:
1. Static Analysis (Ruff, Mypy, Vulture, Pylint).
2. Code Hygiene (Whitespace stripping, forbidden pattern detection).
3. Package Integrity (Verifying built Python and TypeScript artifacts against baselines).
4. Unit Testing (Pytest, npm test).
5. Sandboxed Execution (Running tests under strict confinement on macOS).

Usage:
    python scripts/test_runner.py [--hygiene] [--confinement-config path/to/config.json]
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, NamedTuple, Optional

if os.environ.get("VULTURE_WHITELIST"):
    # This block is used solely for static analysis by Vulture.
    # It explicitly references symbols that might otherwise appear unused (e.g., used dynamically
    # or only in specific OS environments), preventing false positives in dead code detection.
    from scripts import confine

    _v1 = confine.confine
    from kotogram.model import StyleClassifier, PositionalEncoding, MultiFieldEmbedding, KCHead

    _v2 = StyleClassifier.forward
    _v3 = PositionalEncoding.forward
    _v4 = MultiFieldEmbedding.forward
    _v5 = KCHead.forward
    _v6 = KCHead.forward_with_raw


PYTHON_BASELINE = "tests/python_package_baseline.txt"
TS_BASELINE = "tests/typescript_package_baseline.txt"

GREEN = "\033[1;32m"
RED = "\033[1;31m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


class CheckResult(NamedTuple):
    """
    Structured result of a single check/command execution.

    Attributes:
        name: unique identifier for the check (e.g., 'mypy', 'ruff').
        success: True if the check passed (exit code 0), False otherwise.
        output: Captured stdout and stderr from the command.
    """

    name: str
    success: bool
    output: str


async def run_command(
    command: str, env: Optional[Dict[str, str]] = None
) -> CheckResult:
    """
    Asynchronously run a shell command and capture its output.

    Args:
        command: The shell command string to execute.
        env: Optional dictionary of environment variables to override.

    Returns:
        CheckResult containing the command name, success status, and combined output.
    """
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode() + stderr.decode()
    return CheckResult(name=command, success=proc.returncode == 0, output=output)


def print_success(message: str) -> None:
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message: str) -> None:
    print(f"{RED}[ERROR] {message}{RESET}")


async def check_undone() -> CheckResult:
    """
    Enforce a strict "no UN-DONE" policy in comments.

    "UN-DONE" is often used as a marker for incomplete code or technical debt.
    We strictly forbid this to ensure all code committed is considered complete
    and production-ready, or that debt is tracked in a more formal system than
    code comments.
    """
    # Split the string to avoid this script itself triggering the check grep
    undone_str = "UN" + "DONE"
    cmd = f'grep -rnI "{undone_str}" kotogram scripts tests-py train train_style bin/kotogram'

    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        return CheckResult(
            "undone check",
            False,
            f"Found forbidden '{undone_str}' comments! Fix them.\n{stdout.decode()}",
        )

    print_success(f"No '{undone_str}' comments found")
    return CheckResult("undone check", True, "")


async def check_noqa_e402() -> CheckResult:
    """
    Enforce strict import ordering by forbidding 'noqa: E402'.

    E402 (module level import not at top of file) is often suppressed to allow
    monkey-patching or conditional imports. We strictly forbid this suppression
    to maintain a clean and predictable import structure across the codebase.
    """
    # Split the string to avoid this script itself triggering the check grep
    noqa_str = "# noqa" + ": E402"
    cmd = f'grep -rnI "{noqa_str}" kotogram scripts tests-py train train_style bin/kotogram'

    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        return CheckResult(
            "noqa check",
            False,
            f"Found forbidden '{noqa_str}' comments!\n{stdout.decode()}",
        )

    print_success(f"No '{noqa_str}' comments found")
    return CheckResult("noqa check", True, "")


async def check_vulture_circumvention() -> CheckResult:
    """
    Detect potential circumvention of Vulture dead code analysis.

    We strictly forbid files named '*vulture*' (e.g., custom whitelists) and
    mentions of 'vulture' in code (e.g., in commit hooks or other scripts),
    except for this runner script itself.
    """
    # 1. Check for files named *vulture*
    # Exclude common noise directories
    # Note: excluding .git, .venv, etc. happens naturally if we use 'find' with specific exclusions or logic,
    # but a simple 'find . -name "*vulture*"' catches everything.
    # We'll filter out .git/ and .venv/ in Python to be robust/portable-ish.
    find_cmd = 'find . -name "*vulture*.py"'
    proc_find = await asyncio.create_subprocess_shell(
        find_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout_find, _ = await proc_find.communicate()

    found_files = []
    for line in stdout_find.decode().splitlines():
        line = line.strip()
        # Filter out hidden/system internal directories
        if "/.git/" in line or "/.venv/" in line or "/__pycache__" in line or "/.mypy_cache" in line:
            continue
        # Allow usage in this specific file
        if line.endswith("scripts/test_runner.py"):
            continue
        # Allow usage in this specific file (though file name itself doesn't match *vulture*)
        found_files.append(line)

    # 2. Check for content mentions of "vulture"
    # grep -r "vulture" .
    # Exclude binary files (-I), line numbers (-n)
    grep_cmd = 'grep -rnIi --include="*.py" "vulture" .'
    proc_grep = await asyncio.create_subprocess_shell(
        grep_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout_grep, _ = await proc_grep.communicate()

    found_mentions = []
    for line in stdout_grep.decode().splitlines():
        line = line.strip()
        # Exclude this file (scripts/test_runner.py)
        if "scripts/test_runner.py" in line:
            continue
        # Filter out hidden/system internal directories
        if "/.git/" in line or "/.venv/" in line or "/.mypy_cache" in line:
            continue

        # Format is filename:lineno:content
        parts = line.split(":", 1)
        if parts:
            found_mentions.append(parts[0])  # Just the filename as requested

    all_violations = sorted(list(set(found_files + found_mentions)))

    if all_violations:
        return CheckResult(
            "vulture circumvention check",
            False,
            "Vulture circumvention detected. DO NOT CIRCUMVENT VULTURE IN ANY WAY, YOU WILL BE FLAGGED AT CODEREVIEW TIME. Remove dead code or move the code to the right location. Test only code goes in test-py, training only code goes in train/:\n" + "\n".join(all_violations),
        )

    print_success("No Vulture circumvention detected")
    return CheckResult("vulture circumvention check", True, "")


async def check_kotogram_dependencies() -> CheckResult:
    """
    Ensure kotogram/ only references libraries, not local scripts/tests/train code.
    """
    # Look for imports of scripts, tests-py, train, train_style
    # Also look for relative imports that might reach up, e.g. "from ..scripts" or "from ..train"
    # But since kotogram is a package, ".." from inside it goes to root.
    cmd = 'grep -rE "^(from|import) (scripts|tests-py|train|train_style)" kotogram/'
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    # We want NO output (exit code 1 is good, 0 is bad if matches found)
    if stdout:
        return CheckResult(
            "kotogram dependency check",
            False,
            f"Forbidden dependencies found in kotogram/:\n{stdout.decode()}",
        )

    print_success("Kotogram dependencies OK")
    return CheckResult("kotogram dependency check", True, "")





async def check_vulture_inference() -> CheckResult:
    """
    Phase 1: Inference-Only Check
    Ensure code in kotogram/ is used by bin/kotogram.
    If unused here, it might belong in train/ or scripts/.
    """
    # Min confidence 60 to catch everything
    cmd = "vulture kotogram/ bin/kotogram scripts/test_runner.py --min-confidence 60"
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode()

    # Filter for violations in kotogram/ directory only
    violations = [
        line for line in output.splitlines()
        if line.strip().startswith("kotogram/")
    ]

    if violations:
        return CheckResult(
            "vulture (inference)",
            False,
            f"Code in kotogram/ not reachable from bin/kotogram (Move to train/ or scripts/?):\n{chr(10).join(violations)}",
        )

    print_success("Vulture (Inference) OK")
    return CheckResult("vulture (inference)", True, "")


async def check_vulture_production() -> CheckResult:
    """
    Phase 2: Production Check
    Ensure code in kotogram/ and train/ and scripts/ is used by production entry points.
    If unused here, it might belong in tests-py/.
    """
    # exclude tests-py
    cmd = "vulture kotogram/ train/ scripts/ bin/kotogram scripts/test_runner.py --exclude tests-py --min-confidence 60"
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode()

    # We care about violations in kotogram/, train/, scripts/
    # (bin/kotogram is entry point)
    violations = [
        line for line in output.splitlines()
        if "tests-py/" not in line # Should be excluded by --exclude but double check
    ]

    if violations:
        return CheckResult(
            "vulture (production)",
            False,
            f"Code unused in production (Move to tests-py/ or delete?):\n{chr(10).join(violations)}",
        )

    print_success("Vulture (Production) OK")
    return CheckResult("vulture (production)", True, "")


async def check_vulture_full() -> CheckResult:
    """
    Phase 3: Full Check
    Ensure everything is used somewhere (including tests).
    If unused here, it is dead code.
    """
    cmd = "vulture kotogram/ train/ scripts/ tests-py/ bin/kotogram scripts/test_runner.py --min-confidence 60"
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    if stdout:
        return CheckResult(
            "vulture (full)",
            False,
            f"Dead code detected (Delete it!):\n{stdout.decode()}",
        )

    print_success("Vulture (Full) OK")
    return CheckResult("vulture (full)", True, "")


async def check_file_structure() -> CheckResult:
    """
    Ensure no .py files exist outside of approved source roots:
    kotogram/, train/, scripts/, tests-py/.
    """
    # Allowed directories
    allowed_dirs = {"kotogram", "train", "scripts", "tests-py"}

    # Use git ls-files to respect .gitignore (e.g. node_modules)
    # --cached: tracked files
    # --others: untracked files
    # --exclude-standard: respect .gitignore
    cmd = 'git ls-files --cached --others --exclude-standard'
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    violations = []
    for line in stdout.decode().splitlines():
        line = line.strip()
        if not line.endswith(".py"):
            continue

        # line is path/to/file.py (git ls-files usually relative to root without ./)
        path = line

        # Ignore hidden/system dirs and build artifacts (extra safety)
        if ".venv" in path or ".git" in path or ".mypy_cache" in path or "__pycache__" in path or "dist_py" in path:
            continue
        if "build/" in path or ".tmp" in path or "egg-info" in path or "models/" in path or "node_modules/" in path:
            continue
        if path.startswith("debug_") or path.startswith("tests/"):
            continue

        # Get top-level dir
        parts = path.split(os.sep)
        if len(parts) > 1:
            top_dir = parts[0]
            if top_dir not in allowed_dirs:
                violations.append(path)
        else:
            # File in root (e.g. setup.py)
            if path != "setup.py": # Whitelist setup.py if it exists
                violations.append(path)

    if violations:
        return CheckResult(
            "file structure check",
            False,
            "Found .py files in unapproved locations (Moved to scripts/ or delete?):\n" + "\n".join(violations),
        )

    print_success("File structure OK")
    return CheckResult("file structure check", True, "")


async def verify_exception_usage() -> CheckResult:
    """
    Enforce a strict whitelist of allowed exceptions in `except` blocks.

    This static analysis ensures that code does not catch generic exceptions or
    exceptions that mask critical system failures. Only tailored, specific exceptions
    (e.g., `KeyboardInterrupt`, `subprocess.CalledProcessError`) are allowed.

    This policy prevents "pokemon exception handling" (gotta catch 'em all) which
    swallows bugs and makes debugging impossible.
    """

    exception_whitelist = {
        "KeyboardInterrupt",
        "subprocess.CalledProcessError",
        "MissingMappingError",
        "queue.Empty",
        "subprocess.TimeoutExpired",
        "BrokenPipeError",
    }

    cmd = (
        # Find lines starting with 'except' (ignoring indentation)
        r'grep -rnH "^\s*except\b.*:" '
        "kotogram scripts train tests-py train_style bin/kotogram "
        # Exclude this specific line which is necessary for the grep itself to work safely
        '| grep -v "worker-init=special-carveout"'
    )

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        if proc.returncode == 1:
            print_success("No exception handlers found (clean but unlikely)")
            return CheckResult("exception usage check", True, "")
        return CheckResult(
            "exception usage check", False, f"Grep failed: {stdout.decode()}"
        )

    output = stdout.decode()
    violations = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        if "scripts/test_runner.py" in line:
            continue

        parts = line.split(":", 2)
        if len(parts) < 3:
            continue

        fpath, lineno, content = parts[0], parts[1], parts[2]

        code_part = content.split("#", 1)[0].strip()

        if code_part.endswith(":"):
            code_part = code_part[:-1].strip()

        if code_part.startswith("except"):
            code_part = code_part[6:].strip()

        if " as " in code_part:
            code_part = code_part.split(" as ", 1)[0].strip()

        if code_part.startswith("(") and code_part.endswith(")"):
            code_part = code_part[1:-1]

        if not code_part:
            violations.append(f"{fpath}:{lineno}: Bare 'except:' is FORBIDDEN")
            continue

        caught_types = [t.strip() for t in code_part.split(",")]

        for t in caught_types:
            if t not in exception_whitelist:
                violations.append(
                    f"{fpath}:{lineno}: Exception '{t}' is NOT WHITELISTED"
                )

    if violations:
        shaming_msg = (
            "Strict Exception Policy Violation!\n"
            f"Allowed Exceptions: {', '.join(sorted(exception_whitelist))}\n"
            "Overbroad exceptions are FORBIDDEN. You will not be able to circumvent this policy so don't try. Rethink your code to stop relying on this exception (let it propagate, most likely). If you _really_ think this exception is needed, you can prepare a rationale and present it to request a _specific_, _narrow_ exception be added to the whitelist (hint, it won't be Exception or RuntimeError). But it probably will be turned down, so just write the code a different way."
        )
        return CheckResult(
            "exception usage check",
            False,
            f"{shaming_msg}\n\nViolations:\n" + "\n".join(violations),
        )

    print_success("Exception usage compliant")
    return CheckResult("exception usage check", True, "")


async def run_ruff() -> CheckResult:
    """Run Ruff for fast linting and auto-formatting."""
    # --fix applies safe fixes; format standardizes code style
    cmd = "ruff check --fix . --config pyproject.toml && ruff format ."
    res = await run_command(cmd)
    if not res.success:
        return CheckResult("ruff", False, f"Ruff failed:\n{res.output}")
    print_success("Ruff check and format passed")
    return CheckResult("ruff", True, "")


async def run_mypy() -> CheckResult:
    """Run MyPy for static type checking across all modules."""
    cmds = [
        "mypy kotogram scripts train --explicit-package-bases",
        "mypy train_style",
        "mypy bin/kotogram",
    ]

    for cmd in cmds:
        res = await run_command(cmd)
        if not res.success:
            return CheckResult("mypy", False, f"Mypy failed on '{cmd}':\n{res.output}")

    print_success("Mypy passed")
    return CheckResult("mypy", True, "")





async def run_pylint() -> CheckResult:
    """Run Pylint, specifically enabling code duplication detection."""
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{cwd}:{cwd}/tests-py"

    cmd = "pylint --enable=duplicate-code --ignore=vulture_whitelist.py kotogram scripts train tests-py train_style bin/kotogram"
    res = await run_command(cmd, env=env)

    if not res.success:
        return CheckResult("pylint", False, f"Pylint failed:\n{res.output}")
    print_success("Pylint duplication check passed")
    return CheckResult("pylint", True, "")


async def run_typescript() -> CheckResult:
    """Run standard npm hygiene (lint/fix) and tests."""
    if not os.path.exists("package.json"):
        return CheckResult("typescript", True, "Skipped (no package.json)")

    cmd = "npm run fix && npm test"
    res = await run_command(cmd)
    if not res.success:
        return CheckResult(
            "typescript", False, f"TypeScript checks failed:\n{res.output}"
        )
    print_success("TypeScript checks passed")
    return CheckResult("typescript", True, "")


async def check_python_package() -> CheckResult:
    """
    Verify the Python package build artifact integrity.

    This function:
    1. Builds the wheel package.
    2. Extracts the file list from the generated wheel.
    3. Normalizes paths (to ignore version numbers and dynamic metadata).
    4. Compares the file list against a known 'baseline' to prevent accidental leaks or omissions.
    """
    shutil.rmtree("dist_py", ignore_errors=True)

    res = await run_command("python3 -m build --no-isolation --outdir dist_py")
    if not res.success:
        return CheckResult("py-build", False, f"Build failed:\n{res.output}")

    if not os.path.exists("dist_py"):
        return CheckResult("py-pkg", False, "dist_py not found")

    whls = [f for f in os.listdir("dist_py") if f.endswith(".whl")]
    if not whls:
        return CheckResult("py-pkg", False, "No wheel file generated")
    whl_path = os.path.join("dist_py", whls[0])

    import zipfile

    with zipfile.ZipFile(whl_path, "r") as z:
        files = z.namelist()

    # Normalize file names to avoid version-dependent diffs
    # e.g., kotogram-0.1.0.dist-info -> kotogram-*.dist-info
    norm_files = []
    for f in files:
        f = re.sub(r"kotogram-.*\.dist-info", "kotogram-*.dist-info", f)
        f = f.replace(
            "kotogram-*.dist-info/licenses/LICENSE", "kotogram-*.dist-info/LICENSE"
        )
        norm_files.append(f)

    norm_files.sort()

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(norm_files) + "\n")
        tmp_path = tmp.name

    cmd = f"diff -u {PYTHON_BASELINE} {tmp_path}"
    diff_res = await run_command(cmd)
    os.remove(tmp_path)

    if not diff_res.success:
        return CheckResult(
            "py-pkg-verify",
            False,
            f"Python package contents do not match baseline!\n{diff_res.output}",
        )

    print_success("Python package verification passed")
    return CheckResult("py-pkg-verify", True, "")


async def check_typescript_package() -> CheckResult:
    """
    Verify the TypeScript package build artifact integrity.

    Similar to the Python check, this ensures `npm pack` produces exactly the expected set of files
    defined in `tests/typescript_package_baseline.txt`.
    """
    if not os.path.exists("package.json"):
        return CheckResult("ts-pkg", True, "Skipped")

    shutil.rmtree("dist", ignore_errors=True)

    res = await run_command("npm run build")
    if not res.success:
        return CheckResult("ts-build", False, f"npm build failed:\n{res.output}")

    res = await run_command("npm pack --quiet")
    if not res.success:
        return CheckResult("npm-pack", False, f"npm pack failed:\n{res.output}")

    pack_file = res.output.strip().splitlines()[-1]

    res = await run_command(f"tar -tf {pack_file}")
    if not res.success:
        os.remove(pack_file)
        return CheckResult("tar-tf", False, f"tar failed:\n{res.output}")

    files = res.output.strip().splitlines()
    # Normalize paths: 'package/lib/index.js' -> 'lib/index.js'
    norm_files = sorted([f.replace("package/", "", 1) for f in files])

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(norm_files) + "\n")
        tmp_path = tmp.name

    cmd = f"diff -u {TS_BASELINE} {tmp_path}"
    diff_res = await run_command(cmd)

    os.remove(tmp_path)
    if os.path.exists(pack_file):
        os.remove(pack_file)

    if not diff_res.success:
        return CheckResult(
            "ts-pkg-verify",
            False,
            f"TypeScript package contents do not match baseline!\n{diff_res.output}",
        )

    print_success("TypeScript package verification passed")
    return CheckResult("ts-pkg-verify", True, "")


def auto_fix_whitespaces() -> None:
    """
    Recursively remove trailing whitespace from all tracked source files.

    This enforces a strict "no trailing whitespace" policy on every test run,
    minimizing diff noise and maintaining code hygiene.
    """
    print("Auto-fixing trailing whitespace...")

    for root, _, files in os.walk("."):
        if (
            ".git" in root
            or ".venv" in root
            or ".mypy_cache" in root
            or "__pycache__" in root
        ):
            continue

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                _strip_file(path)

    if os.path.exists("train_style"):
        _strip_file("train_style")

    print_success("Whitespace cleanup complete.")


def _strip_file(path: str) -> None:
    """Strip trailing whitespace from a single file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [line.rstrip() + "\n" for line in lines]

    if lines != new_lines:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)


async def main() -> None:
    # pylint: disable=too-many-locals
    """
    Main entry point for the test runner.

    Orchestration flow:
    1. Static Exception Analysis (Blocking) - Fails fast if policy is violated.
    2. Whitespace Auto-fix - Modifies files in place.
    3. Git Status Snapshot - Records state to detect uncommitted changes during tests.
    4. Parallel Static Checks (Ruff, Mypy, Vulture, Pylint, Pkg Verification).
    5. Conditional Test Execution:
       - If hygiene passes, runs TypeScript tests.
       - Then runs Python tests (Pytest).
       - Supports Confinement (Sandboxing) for Pytest on macOS.
    6. Git Status Verification - Fails if tests generated untracked artifacts.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Kotogram Test Runner")
    parser.add_argument(
        "--hygiene",
        action="store_true",
        help="Run only hygiene checks (linting, static analysis, build verification), skipping Python tests.",
    )
    parser.add_argument(
        "--confinement-config",
        help="JSON configuration file for confinement (applies only to Pytest).",
    )
    parser.add_argument(
        "--specific-python-test",
        help="Run a specific Python test module using unittest, skipping all other checks.",
    )
    args = parser.parse_args()

    # --- Handling Specific Test Mode ---
    if args.specific_python_test:
        print(f"{BLUE}Running specific python test: {args.specific_python_test}{RESET}")

        # Ensure PYTHONPATH includes project root and tests-py
        cwd = os.getcwd()
        sys.path.insert(0, os.path.join(cwd, "tests-py"))
        sys.path.insert(0, cwd)

        # Use subprocess to run unittest to ensure clean environment semantics match previous shell script
        # explicitly setting PYTHONPATH in environment just to be safe
        env = os.environ.copy()
        env["PYTHONPATH"] = f".:tests-py:{env.get('PYTHONPATH', '')}"

        cmd = [sys.executable, "-m", "unittest", args.specific_python_test]

        # We don't use confinement for specific tests currently (per legacy test.sh behavior)
        test_res = subprocess.run(cmd, env=env, check=False)
        sys.exit(test_res.returncode)

    exception_res = await verify_exception_usage()
    if not exception_res.success:
        print_error(exception_res.output)
        sys.exit(1)
    auto_fix_whitespaces()

    initial_git_status = subprocess.check_output(["git", "status", "--short"]).decode()

    # Run check_undone synchronously/blocking so it fails fast and prints first
    undone_res = await check_undone()
    if not undone_res.success:
        print_error(undone_res.output)
        sys.exit(1)

    # Run check_vulture_circumvention synchronously/blocking
    vulture_res = await check_vulture_circumvention()
    if not vulture_res.success:
        print_error(vulture_res.output)
        sys.exit(1)

    process_noqa = check_noqa_e402()

    # Run Ruff first and serially to avoid race conditions (since it writes/formats files)
    ruff_res = await run_ruff()
    if not ruff_res.success:
        print_error(ruff_res.output)
        sys.exit(1)

    tasks = [
        run_mypy(),
        check_vulture_inference(),
        check_vulture_production(),
        check_vulture_full(),
        run_pylint(),
        check_python_package(),
        check_kotogram_dependencies(),
        check_file_structure(),
    ]
    tasks.insert(0, process_noqa)

    pending = [asyncio.create_task(t) for t in tasks]
    failed = False

    results = await asyncio.gather(*pending)

    for result in results:
        if not result.success:
            print_error(result.output)
            failed = True

    if not failed:
        ts_tasks = [run_typescript(), check_typescript_package()]
        for ts_t in ts_tasks:
            res = await ts_t
            if not res.success:
                print_error(res.output)
                failed = True

    if failed:
        for t in pending:
            if not t.done():
                t.cancel()
        sys.exit(1)

    if not args.hygiene:
        print(f"\n{BLUE}Running Pytest...{RESET}")

        env = os.environ.copy()
        env["CI"] = "true"  # Force CI mode in tests
        pytest_cmd = [sys.executable, "-m", "pytest", "-x", "--no-header", "tests-py/"]

        if args.confinement_config:
            import importlib.util
            import json

            if importlib.util.find_spec("confine"):
                import confine as confine_lib  # type: ignore
            else:
                from scripts import confine as confine_lib  # type: ignore

            with open(args.confinement_config, "r", encoding="utf-8") as f:
                config = json.load(f)

            config["mode"] = "run"

            print(
                f"{BLUE}Running Pytest confined with {args.confinement_config}{RESET}"
            )

            # --- Confinement Verification (Probe) ---
            # To ensure the sandbox isn't just a placebo, we first try to break out of it.
            # We run a small script that attempts to write a file to the project root.
            # If the write SUCCEEDS, the confinement is broken -> we fail hard.
            # If the write FAILS (OS error), the confinement is working -> we proceed.

            if sys.platform == "darwin":
                print(f"{BLUE}Verifying confinement (Probe)...{RESET}")
                probe_file = "confinement_probe_fail.txt"
                probe_cmd = [
                    sys.executable,
                    "-c",
                    f"import sys\ntry:\n    open('{probe_file}', 'w').close()\nexcept OSError:\n    sys.exit(1)",
                ]
                probe_res = confine_lib.confine(probe_cmd, config, env=env, check=False)  # type: ignore

                if probe_res.returncode == 0:
                    print_error(
                        "Confinement Verification FAILED: Able to write to project root."
                    )
                    if os.path.exists(probe_file):
                        os.remove(probe_file)
                    sys.exit(1)

                print_success("Confinement verified (Write denied).")
            else:
                print(
                    f"{BLUE}Skipping confinement verification (Non-Mac detected){RESET}"
                )
            # pylint: disable=no-member
            pytest_res = confine_lib.confine(pytest_cmd, config, env=env, check=False)  # type: ignore
        else:
            pytest_res = subprocess.run(
                pytest_cmd,
                env=env,
                check=False,
            )

        if pytest_res.returncode != 0:
            sys.exit(pytest_res.returncode)

    # Verify that the test run left the workspace clean.
    # We do NOT want tests that succeed but leave behind trash or modify files.
    final_git_status = subprocess.check_output(["git", "status", "--short"]).decode()
    if initial_git_status != final_git_status:
        print_error(
            "git status changed during tests. New/changed files detected in repository."
        )
        print("Initial:\n", initial_git_status)
        print("Final:\n", final_git_status)
        sys.exit(1)

    print_success("Git status clean")
    print_success("All checks passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
