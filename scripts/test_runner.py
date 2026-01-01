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
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from typing import Awaitable, Dict, NamedTuple, Optional

if os.environ.get("VULTURE_WHITELIST"):
    # This block is used solely for static analysis by Vulture.
    # It explicitly references symbols that might otherwise appear unused (e.g., used dynamically
    # or only in specific OS environments), preventing false positives in dead code detection.
    sys.path.append(os.path.abspath("tests-py"))
    import lib_confine  # type: ignore

    _v1 = lib_confine.confine  # type: ignore
    from kotogram.model import StyleClassifier, PositionalEncoding, MultiFieldEmbedding, KCHead

    _v2 = StyleClassifier.forward
    _v3 = PositionalEncoding.forward
    _v4 = MultiFieldEmbedding.forward
    _v5 = KCHead.forward
    _v6 = KCHead.forward_with_raw



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
    duration: float = 0.0


async def check_confinement_probe(config_path: Optional[str]) -> CheckResult:
    """
    Verify that the confinement system (sandbox) is correctly blocking writes.
    """
    if sys.platform != "darwin":
        return CheckResult("Confinement probe", True, "Skipped (Non-Mac)")

    if not config_path:
        return CheckResult("Confinement probe", True, "Skipped (No config)")

    try:
        # Dynamically import lib_confine from tests-py
        if os.path.abspath("tests-py") not in sys.path:
            sys.path.append(os.path.abspath("tests-py"))

        import importlib.util
        if importlib.util.find_spec("lib_confine"):
            import lib_confine as confine_lib # type: ignore
        else:
            return CheckResult("Confinement probe", False, "Could not import lib_confine")

        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Allow basic reads for python to start
        config["mode"] = "run"
        if "allow_read" not in config:
            config["allow_read"] = []
        config["allow_read"].append(f"{sys.prefix}/")
        config["allow_read"].append(f"{sys.base_prefix}/")

        # Probe: try to write to a file in CWD (project root)
        probe_file = "confinement_probe_fail.txt"
        probe_code = (
            "import sys\n"
            "try:\n"
            "    open('confinement_probe_fail.txt', 'w').close()\n"
            "except OSError:\n"
            "    sys.exit(1)"
        )
        probe_cmd = [
            sys.executable,
            "-c",
            probe_code,
        ]

        env = os.environ.copy()

        # Run probe
        # We expect check=False behavior (don't raise).
        probe_res = confine_lib.confine(probe_cmd, config, env=env, check=False)

        if probe_res.returncode == 0:
            # It succeeded in writing -> FAILURE of confinement
            msg = "Confinement Verification FAILED: Able to write to project root."
            if os.path.exists(probe_file):
                os.remove(probe_file)
            return CheckResult("Confinement probe", False, msg)

        # It failed to write -> SUCCESS of confinement
        return CheckResult("Confinement probe", True, "")

    except Exception as e: # pylint: disable=broad-exception-caught
        return CheckResult("Confinement probe", False, f"Probe failed with error: {e}")



async def measure_check(coro: Awaitable[CheckResult]) -> CheckResult:
    """Wrapper to measure execution time of a check coroutine."""
    start_time = asyncio.get_event_loop().time()
    res = await coro
    duration = asyncio.get_event_loop().time() - start_time
    # We override the duration to capture the full wrapper time, which includes python logic overhead
    return CheckResult(res.name, res.success, res.output, duration)


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
    # Ensure we propagate specific environment variables if they are set in the parent process
    # but not explicitly passed in 'env' (though usually 'env' is None or a copy).
    # If 'env' is provided, we assume the caller handled it, but we can enforce propagation here
    # to be safe for all run_command usages.

    start_time = asyncio.get_event_loop().time()
    final_env = env if env is not None else os.environ.copy()

    # Explicitly ensure these are passed if present in os.environ (redundant if using os.environ.copy above,
    # but critical if env was passed as a restricted dict).
    for key in ["TRAIN_RECORD_ROOTS", "TRAIN_RECORD_FAIL_ON_CONST"]:
        if key in os.environ and key not in final_env:
            final_env[key] = os.environ[key]

    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=final_env
    )
    stdout, stderr = await proc.communicate()
    output = stdout.decode() + stderr.decode()
    duration = asyncio.get_event_loop().time() - start_time
    return CheckResult(name=command, success=proc.returncode == 0, output=output, duration=duration)


def print_success(message: str, duration: float = 0.0) -> None:
    time_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"{GREEN}✅ {message}{time_str}{RESET}", flush=True)


def print_error(message: str, duration: float = 0.0) -> None:
    time_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"{RED}[ERROR] {message}{time_str}{RESET}", flush=True)

def report_slowest_tests(xml_path: str, count: int = 5) -> None:
    if not os.path.exists(xml_path):
        return
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        testcases = []
        for tc in root.iter("testcase"):
            time_str = tc.get("time")
            if time_str is None:
                continue
            try:
                t = float(time_str)
                name = tc.get("name", "unknown")
                classname = tc.get("classname", "unknown")
                testcases.append((t, classname, name))
            except ValueError:
                continue

        testcases.sort(key=lambda x: x[0], reverse=True)
        top_n = testcases[:count]

        if top_n:
            print("\nTop 5 Slowest Tests:")
            for t, c, n in top_n:
                print(f"  {t:.2f}s  {c}::{n}")
    except Exception as e: # pylint: disable=broad-exception-caught
        print_error(f"Failed to parse test report: {e}")
    finally:
        if os.path.exists(xml_path):
            os.remove(xml_path)



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
            "Undone check",
            False,
            f"Found forbidden '{undone_str}' comments! Fix them.\n{stdout.decode()}",
        )

    # print_success(f"No '{undone_str}' comments found") -> moved to main/result
    return CheckResult("Undone check", True, "")


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
            "Noqa check",
            False,
            f"Found forbidden '{noqa_str}' comments!\n{stdout.decode()}",
        )

    # print_success(f"No '{noqa_str}' comments found")
    return CheckResult("Noqa check", True, "")


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
        if "tests-py/test_code_duplication.py" in line:
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
            "Vulture circumvention check",
            False,
            "Vulture circumvention detected. DO NOT CIRCUMVENT VULTURE IN ANY WAY, YOU WILL BE FLAGGED AT CODEREVIEW TIME. Remove dead code or move the code to the right location. Test only code goes in test-py, training only code goes in train/:\n" + "\n".join(all_violations),
        )

    # print_success("No Vulture circumvention detected")
    return CheckResult("Vulture circumvention check", True, "")


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
            "Kotogram dependency check",
            False,
            f"Forbidden dependencies found in kotogram/:\n{stdout.decode()}",
        )

    # print_success("Kotogram dependencies OK")
    return CheckResult("Kotogram dependency check", True, "")





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
            "Vulture (inference)",
            False,
            f"Code in kotogram/ not reachable from bin/kotogram (Move to train/ or scripts/?):\n{chr(10).join(violations)}",
        )

    # print_success("Vulture (Inference) OK")
    return CheckResult("Vulture (inference)", True, "")


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
            "Vulture (production)",
            False,
            f"Code unused in production (Move to tests-py/ or delete?):\n{chr(10).join(violations)}",
        )

    # print_success("Vulture (Production) OK")
    return CheckResult("Vulture (production)", True, "")


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
            "Vulture (full)",
            False,
            f"Dead code detected (Delete it!):\n{stdout.decode()}",
        )

    # print_success("Vulture (Full) OK")
    return CheckResult("Vulture (full)", True, "")


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
            "File structure check",
            False,
            "Found .py files in unapproved locations (Moved to scripts/ or delete?):\n" + "\n".join(violations),
        )

    # print_success("File structure OK")
    return CheckResult("File structure check", True, "")


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
            # print_success("No exception handlers found (clean but unlikely)")
            return CheckResult("Exception usage check", True, "")
        return CheckResult(
            "Exception usage check", False, f"Grep failed: {stdout.decode()}"
        )

    output = stdout.decode()
    violations = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        if "scripts/test_runner.py" in line:
            continue

        if "tests-py/instrumentation.py" in line:
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
            "Exception usage check",
            False,
            f"{shaming_msg}\n\nViolations:\n" + "\n".join(violations),
        )

    # print_success("Exception usage compliant")
    return CheckResult("Exception usage check", True, "")


async def run_ruff() -> CheckResult:
    """Run Ruff for fast linting and auto-formatting."""
    # --fix applies safe fixes; format standardizes code style
    cmd = "ruff check --fix . --config pyproject.toml && ruff format ."
    res = await run_command(cmd)
    if not res.success:
        return CheckResult("Ruff", False, f"Ruff failed:\n{res.output}")
    # print_success("Ruff check and format passed")
    return CheckResult("Ruff", True, "")


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
            return CheckResult("Mypy", False, f"Mypy failed on '{cmd}':\n{res.output}")

    # print_success("Mypy passed")
    return CheckResult("Mypy", True, "")





async def run_pylint() -> CheckResult:
    """Run Pylint, specifically enabling code duplication detection."""
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{cwd}:{cwd}/tests-py"

    cmd = "pylint -j 0 --disable=duplicate-code --ignore=vulture_whitelist.py kotogram scripts train tests-py train_style bin/kotogram"
    res = await run_command(cmd, env=env)

    if not res.success:
        return CheckResult("Pylint", False, f"Pylint failed:\n{res.output}")
    # print_success("Pylint duplication check passed")
    return CheckResult("Pylint", True, "")


async def run_typescript() -> CheckResult:
    """Run standard npm hygiene (lint/fix) and tests."""
    if not os.path.exists("package.json"):
        return CheckResult("typescript", True, "Skipped (no package.json)")

    cmd = "npm run fix && npm test"
    res = await run_command(cmd)
    if not res.success:
        return CheckResult(
            "Typescript", False, f"TypeScript checks failed:\n{res.output}"
        )
    # print_success("TypeScript checks passed")
    return CheckResult("Typescript", True, "")





def auto_fix_whitespaces() -> None:
    """
    Recursively remove trailing whitespace from all tracked source files.

    This enforces a strict "no trailing whitespace" policy on every test run,
    minimizing diff noise and maintaining code hygiene.
    """
    print("Auto-fixing trailing whitespace...")
    start = time.time()

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

    duration = time.time() - start
    print_success("Whitespace cleanup complete.", duration)


def _strip_file(path: str) -> None:
    """Strip trailing whitespace from a single file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [line.rstrip() + "\n" for line in lines]

    if lines != new_lines:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)


async def main() -> None:
    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-lines
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
        "--pytests",
        nargs="+",
        help="Run specific Python tests using pytest infrastructure (e.g. tests-py/test_foo.py).",
    )
    parser.add_argument(
        "--no-hygiene",
        action="store_true",
        help="Disable hygiene checks (linting, static analysis, etc).",
    )
    parser.add_argument(
        "--no-instrument",
        action="store_true",
        help="Disable instrumentation (parameter recorder) to speed up tests.",
    )
    args = parser.parse_args()

    # --- Setup Default Environment for Parameter Recorder ---
    if not args.no_instrument:
        if "TRAIN_RECORD_ROOTS" not in os.environ:
            # Default to tracking entire project (kotogram, scripts, bin)
            os.environ["TRAIN_RECORD_ROOTS"] = os.getcwd()
        # Ensure we don't accidentally fail on const unless explicitly asked,
        # or maybe we should? User didn't specify default failure, just reporting.
        # "Also, make this mode the default for test_runner" -> implies reporting.

    # Remove specific-python-test block (superseded by --pytests)

    if not args.no_hygiene:
        exception_res = await verify_exception_usage()
        if not exception_res.success:
            print_error(exception_res.output)
            sys.exit(1)
        auto_fix_whitespaces()

    initial_git_status = subprocess.check_output(["git", "status", "--short"]).decode()

    if not args.no_hygiene:
        process_noqa = measure_check(check_noqa_e402())

        # Run Ruff first and serially to avoid race conditions (since it writes/formats files)
        ruff_res = await measure_check(run_ruff())
        if not ruff_res.success:
            print_error(ruff_res.output, ruff_res.duration)
            sys.exit(1)
        print_success("Ruff check OK", ruff_res.duration)

        tasks = [
            run_pylint(),
            check_undone(),
            check_vulture_circumvention(),
            run_mypy(),
            check_vulture_inference(),
            check_vulture_production(),
            check_vulture_full(),
            check_kotogram_dependencies(),
            check_file_structure(),
            check_confinement_probe("confine/python-test.json"),
        ]
        tasks.insert(0, process_noqa)

        pending = [asyncio.create_task(measure_check(t)) for t in tasks]
        failed = False

        for coro in asyncio.as_completed(pending):
            result = await coro
            if not result.success:
                print_error(result.output, result.duration)
                failed = True
            else:
                print_success(f"{result.name} OK", result.duration)

        if not failed:
            ts_pending = [
                asyncio.create_task(measure_check(run_typescript())),
            ]
            for coro in asyncio.as_completed(ts_pending):
                res = await coro
                if not res.success:
                    print_error(res.output, res.duration)
                    failed = True
                else:
                    print_success(f"{res.name} OK", res.duration)

        if failed:
            for t in pending:
                if not t.done():
                    t.cancel()
            sys.exit(1)

    if not args.hygiene:
        print(f"\n{BLUE}Running Pytest...{RESET}")

        env = os.environ.copy()
        env["CI"] = "true"  # Force CI mode in tests
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "auto",
            "-x",
            "--no-header",
            "--junitxml=test-results.xml",
        ]

        # Append targets
        # Append targets
        if args.pytests:
            pytest_cmd.extend(args.pytests)
        else:
            pytest_cmd.append("tests-py/")

        # Setup Recorder Output Dir
        recorder_dir = None
        tests_py_dir = os.path.abspath("tests-py")

        if not args.no_instrument:
            import tempfile

            recorder_dir = tempfile.mkdtemp(prefix="kotogram_recorder_")
            env["TRAIN_RECORD_OUTPUT_DIR"] = recorder_dir
            # print(f"{BLUE}Writing parameter reports to {recorder_dir}{RESET}")

            # Inject instrumentation into subprocesses
            # We create a sitecustomize.py in the recorder dir and add it to PYTHONPATH
            site_cust_path = os.path.join(recorder_dir, "sitecustomize.py")
            with open(site_cust_path, "w", encoding="utf-8") as f:
                f.write(
                    "try:\n"
                    "    import instrumentation\n"
                    "    instrumentation.auto_enable()\n"
                    "except ImportError:\n"
                    "    pass\n"
                )

            # Update PYTHONPATH to include recorder_dir (for sitecustomize) and tests-py (for instrumentation)
            env["PYTHONPATH"] = f"{recorder_dir}:{tests_py_dir}:{env.get('PYTHONPATH', '')}"
        else:
            # Update PYTHONPATH to include tests-py (needed for tests)
            env["PYTHONPATH"] = f"{tests_py_dir}:{env.get('PYTHONPATH', '')}"

        try:
            if args.confinement_config:
                import importlib.util
                import json

                if importlib.util.find_spec("lib_confine"):
                    import lib_confine as confine_lib  # type: ignore
                else:
                    sys.path.append(os.path.abspath("tests-py"))
                    import lib_confine as confine_lib # type: ignore

                with open(args.confinement_config, "r", encoding="utf-8") as f:
                    config = json.load(f)

                config["mode"] = "run"

                # Inject Python environment access
                # Needed for Homebrew/Conda python which isn't covered by system.sb
                if "allow_read" not in config:
                    config["allow_read"] = []

                # Allow reading paths under sys.prefix (e.g. /opt/homebrew...)
                config["allow_read"].append(f"{sys.prefix}/")
                config["allow_read"].append(f"{sys.base_prefix}/")
                # Also allow the executable location itself if different
                exe_dir = os.path.dirname(sys.executable)
                config["allow_read"].append(f"{exe_dir}/")

                # Allow writing to recorder dir
                if "allow_write" not in config:
                    config["allow_write"] = []
                if recorder_dir:
                    config["allow_write"].append(f"{recorder_dir}/")

                print(
                    f"{BLUE}Running Pytest confined with {args.confinement_config}{RESET}"
                )

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

            report_slowest_tests("test-results.xml")

            if pytest_res.returncode != 0:
                sys.exit(pytest_res.returncode)

        finally:
            # Aggregate reports
            if not args.no_instrument and recorder_dir:
                try:
                    print(f"{BLUE}Aggregating reports...{RESET}")
                    # Add tests-py to sys.path to import instrumentation
                    project_root_aggr = os.getcwd()
                    sys.path.append(os.path.join(project_root_aggr, "tests-py"))
                    import instrumentation  # type: ignore
                    instrumentation.aggregate_reports(recorder_dir, project_root=project_root_aggr)

                    # Cleanup
                    shutil.rmtree(recorder_dir, ignore_errors=True)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"{RED}Failed to aggregate reports: {e}{RESET}")

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
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _global_start_time = time.time()
    try:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print(f"\n{RED}Interrupted by user{RESET}")
            sys.exit(1)
    finally:
        _global_duration = time.time() - _global_start_time
        print(f"\nTotal execution time: {_global_duration:.2f}s", flush=True)
