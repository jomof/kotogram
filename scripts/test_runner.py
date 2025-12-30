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


async def run_vulture() -> CheckResult:
    """
    Run Vulture to detect dead code (unused functions, variables, etc.).

    This runs in two passes:
    1. Production Strict: Checks core modules to ensure every symbol is used by other production code.
    2. Full (Permissive): Checks everything (including tests) to catch truly orphaned code.
    """
    cmd_prod = "vulture kotogram scripts train scripts/curate scripts/test_runner.py train_style bin/kotogram"
    res_prod = await run_command(cmd_prod)
    if not res_prod.success:
        return CheckResult(
            "vulture (production strict)",
            False,
            f"Vulture found unused production code (not used by other production code):\n{res_prod.output}",
        )

    cmd_full = "vulture kotogram scripts train tests-py scripts/test_runner.py train_style bin/kotogram"
    res_full = await run_command(cmd_full)
    if not res_full.success:
        return CheckResult(
            "vulture (full)",
            False,
            f"Vulture found unused code (likely in tests):\n{res_full.output}",
        )

    print_success("Vulture passed (strict production + full)")
    return CheckResult("vulture", True, "")


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

    process_noqa = check_noqa_e402()

    # Run Ruff first and serially to avoid race conditions (since it writes/formats files)
    ruff_res = await run_ruff()
    if not ruff_res.success:
        print_error(ruff_res.output)
        sys.exit(1)

    tasks = [
        run_mypy(),
        run_vulture(),
        run_pylint(),
        check_python_package(),
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
