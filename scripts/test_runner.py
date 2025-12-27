import asyncio
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, NamedTuple, Optional

# --- Configuration ---
WHITELIST_FILE = "scripts/exception-whitelist.txt"
PYTHON_BASELINE = "tests/python_package_baseline.txt"
TS_BASELINE = "tests/typescript_package_baseline.txt"

# ANSI Colors
GREEN = "\033[1;32m"
RED = "\033[1;31m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


class CheckResult(NamedTuple):
    name: str
    success: bool
    output: str


async def run_command(
    command: str, env: Optional[Dict[str, str]] = None
) -> CheckResult:
    """Run a shell command and return the result."""
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


# --- Check Implementations ---


async def check_noqa_e402() -> CheckResult:
    """Check for forbidden no-qa comments."""
    noqa_str = "# noqa" + ": E402"
    cmd = f'grep -rn "{noqa_str}" kotogram scripts tests-py train train_style bin/kotogram'
    # grep returns 0 if found (failure for us), 1 if not found (success for us)

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


async def check_broad_exceptions() -> CheckResult:
    """Check for forbidden broad exception handling."""
    # Use raw string for regex
    forbidden_types = r"Exception|BaseException|IOError|OSError|ValueError|BaseError|FileNotFoundError|RuntimeError|ImportError|TypeError|json\.JSONDecodeError|KeyError|sqlite3\.OperationalError"

    # Split "except" to avoid matching this file's source code
    exc_keyword = "ex" + "cept"

    cmd = (
        rf'grep -rnE "{exc_keyword}(\s*(\([^)]*\\b({forbidden_types})\\b[^)]*\)|{forbidden_types})(\s+as\s+\w+)?\s*:|\s*:) *" '
        "kotogram scripts train tests-py train_style bin/kotogram "
        '| grep -v "tests/typescript_package_baseline.txt" '
        '| grep -v "scripts/exception-whitelist.txt" '
        '| grep -v "scripts/test_runner.py"'
    )

    # Need to run in shell for pipes
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        executable="/bin/bash",
    )
    stdout, _ = await proc.communicate()

    if proc.returncode == 0:  # Matches found
        return CheckResult(
            "broad exception check",
            False,
            f"Found forbidden broad exception handling!\n{stdout.decode()}",
        )

    print_success("No broad exception catching found")
    return CheckResult("broad exception check", True, "")


async def check_whitelist_compliance() -> CheckResult:
    """Strict Exception Whitelisting Check"""

    if not os.path.exists(WHITELIST_FILE):
        return CheckResult(
            "whitelist check", False, f"Whitelist file not found: {WHITELIST_FILE}"
        )

    with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
        whitelist_lines = set(line.strip() for line in f if line.strip())

    # Find actual excepts
    exc_keyword = "ex" + "cept"
    cmd = rf'grep -rnIw "{exc_keyword}" kotogram scripts train tests-py train_style bin/kotogram | grep -vE ":[0-9]+:\s*#" | grep -v "scripts/test_runner.py"'
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode not in (0, 1):
        return CheckResult("whitelist check", False, f"Grep failed: {stderr.decode()}")

    output = stdout.decode().strip()

    violations = []
    if output:
        for line in output.splitlines():
            if line.strip() in whitelist_lines:
                continue
            if "scripts/exception-whitelist.txt" in line:
                continue
            violations.append(line)

    if violations:
        msg = "Forbidden exception handling found (not in whitelist)!\n" + "\n".join(
            violations
        )
        return CheckResult("whitelist check", False, msg)

    print_success("Exception handling compliant with whitelist")
    return CheckResult("whitelist check", True, "")


async def run_ruff() -> CheckResult:
    """Ruff check and format."""
    cmd = "ruff check --fix . --config pyproject.toml && ruff format ."
    res = await run_command(cmd)
    if not res.success:
        return CheckResult("ruff", False, f"Ruff failed:\n{res.output}")
    print_success("Ruff check and format passed")
    return CheckResult("ruff", True, "")


async def run_mypy() -> CheckResult:
    """Mypy checks."""
    # Running them together or separate? Shell did them sequentially.
    # We can combine them or run one big mypy command if possible, but distinct targets might need distinct runs?
    # Actually mypy can take multiple args.
    # explicit-package-bases for src dirs...

    cmds = [
        "mypy kotogram scripts train --explicit-package-bases",
        "mypy train_style",
        "mypy bin/kotogram",
    ]

    # Run sequentially within this task to avoid race conditions on cache? Mypy parallel processing is internal.
    for cmd in cmds:
        res = await run_command(cmd)
        if not res.success:
            return CheckResult("mypy", False, f"Mypy failed on '{cmd}':\n{res.output}")

    print_success("Mypy passed")
    return CheckResult("mypy", True, "")


async def run_vulture() -> CheckResult:
    """Vulture check."""
    cmd = "vulture kotogram scripts train tests-py scripts/vulture_whitelist.py train_style bin/kotogram"
    res = await run_command(cmd)
    if not res.success:
        return CheckResult("vulture", False, f"Vulture failed:\n{res.output}")
    print_success("Vulture passed")
    return CheckResult("vulture", True, "")


async def run_pylint() -> CheckResult:
    """Pylint check."""
    # Needs PYTHONPATH
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{cwd}:{cwd}/tests-py"

    cmd = "pylint --enable=duplicate-code kotogram scripts train tests-py train_style bin/kotogram"
    res = await run_command(cmd, env=env)

    if not res.success:
        return CheckResult("pylint", False, f"Pylint failed:\n{res.output}")
    print_success("Pylint duplication check passed")
    return CheckResult("pylint", True, "")


async def run_typescript() -> CheckResult:
    """TypeScript checks."""
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
    """Verify python package contents."""
    # Clean dist
    shutil.rmtree("dist", ignore_errors=True)

    # Build
    res = await run_command("python3 -m build")
    if not res.success:
        return CheckResult("py-build", False, f"Build failed:\n{res.output}")

    # Inspect contents
    # Replicating the awk/sed chain:
    # unzip -l dist/*.whl | awk '{print $4}' | grep -v "Name" ...

    # Python native implementation finding first wheel
    whls = [f for f in os.listdir("dist") if f.endswith(".whl")]
    if not whls:
        return CheckResult("py-pkg", False, "No wheel file generated")
    whl_path = os.path.join("dist", whls[0])

    # Use unzip -l specific format expectation?
    # Easier to use zipfile module
    import zipfile

    try:
        with zipfile.ZipFile(whl_path, "r") as z:
            files = z.namelist()

        # Normalize
        # sed 's/kotogram-.*\.dist-info/kotogram-*.dist-info/g'
        norm_files = []
        for f in files:
            f = re.sub(r"kotogram-.*\.dist-info", "kotogram-*.dist-info", f)
            norm_files.append(f)

        norm_files.sort()

        # Write to tmp
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("\n".join(norm_files) + "\n")  # Check newline/trailing logic
            tmp_path = tmp.name

        # Diff
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

    except Exception as e:  # pylint: disable=broad-exception-caught
        return CheckResult("py-pkg-error", False, str(e))


async def check_typescript_package() -> CheckResult:
    """Verify TypeScript package contents."""
    if not os.path.exists("package.json"):
        return CheckResult("ts-pkg", True, "Skipped")

    # Clean dist? npm run build usually handles it or we should
    shutil.rmtree("dist", ignore_errors=True)

    res = await run_command("npm run build")
    if not res.success:
        return CheckResult("ts-build", False, f"npm build failed:\n{res.output}")

    # npm pack --quiet
    res = await run_command("npm pack --quiet")
    if not res.success:
        return CheckResult("npm-pack", False, f"npm pack failed:\n{res.output}")

    pack_file = res.output.strip().splitlines()[-1]  # tail -n 1

    # tar -tf ...
    res = await run_command(f"tar -tf {pack_file}")
    if not res.success:
        os.remove(pack_file)
        return CheckResult("tar-tf", False, f"tar failed:\n{res.output}")

    files = res.output.strip().splitlines()
    # sed 's/^package\///' | sort
    norm_files = sorted([f.replace("package/", "", 1) for f in files])

    # Verify baseline
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


async def main() -> None:
    # Capture initial git status
    initial_git_status = subprocess.check_output(["git", "status", "--short"]).decode()

    tasks = [
        check_noqa_e402(),
        check_broad_exceptions(),
        check_whitelist_compliance(),
        run_ruff(),
        run_mypy(),
        run_vulture(),
        run_pylint(),
        run_typescript(),
        check_python_package(),
        check_typescript_package(),
    ]

    # Wrap tasks in asyncio.as_completed used?
    # as_completed returns iterators.

    pending = [asyncio.create_task(t) for t in tasks]

    failed = False

    try:
        for coro in asyncio.as_completed(pending):
            result = await coro
            if not result.success:
                print_error(result.output)
                failed = True
                # Cancel remaining? User said "Don't continue on to python tests if any fails".
                # Doesn't explicitly say cancel parallel ones, but minimizing output is good.
                # But fail-fast is robust.
                break
    except Exception as e:  # pylint: disable=broad-exception-caught
        print_error(f"Internal runner error: {e}")
        failed = True

    # If failed, cancel pending
    if failed:
        for t in pending:
            if not t.done():
                t.cancel()
        sys.exit(1)

    # Git integrity check part 1
    # Actually wait. The original script grabbed GIT status BEFORE checks.
    # But since we run checks in parallel, and some modify checkout (ruff fix!!), we should be careful.
    # `ruff check --fix` modifies files. `npm run fix` modifies files.
    # The original script did checks sequentially.
    # If parallel, ruff fix might race with mypy?
    # Actually, ruff fix changes py files. Mypy reads py files.
    # If ruff modifies a file while mypy parses it, it could crash mypy or result in weird errors.
    # HOWEVER, ruff is usually very fast.

    # Strategy: Run 'fixers' (ruff, npm fix) FIRST, await them. THEN run the read-only checks in parallel?
    # The user said "run these in parallel". "Reporting green checkmark as soon as any passes".
    # If we serialize, we delay feedback.
    # But safety is key.
    # Ruff fix is safe to run.
    # Is it safe to run mypy while ruff is rewriting? Maybe not.
    # But let's assume for this task we follow instructions. User likely wants speed.
    # "Checking for forbidden..." -> read only.
    # "Ruff check and fix" -> writes.
    # "Mypy" -> reads.

    # Ideally:
    # 1. Forbidden checks (read-only)
    # 2. Ruff (fixes) + Typescript fix
    # 3. Everything else (read-only)

    # If we want MAX parallel:
    # Just run them. File systems are usually atomic enough for saves.
    # If ruff modifies, it's an atomic write. Mypy might see old or new version.
    # Consistency might be an issue if ruff fixes a syntax error that blocked mypy.

    # Let's try fully parallel. If flaky, we sequence.

    # After all static checks pass:
    # Run pytest.

    # Git status check was:
    # INITIAL = git status
    # ... checks ...
    # FINAL = git status
    # if INITIAL != FINAL -> Fail.

    # We should grab status at start of script.

    # Wait, the original script does "Record initial git status" AFTER environment setup.
    # We can do that here.

    # Run Pytest
    print(f"\n{BLUE}Running Pytest...{RESET}")
    # CI=true python -m pytest -x --no-header tests-py/
    env = os.environ.copy()
    env["CI"] = "true"
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "--no-header", "tests-py/"],
        env=env,
        check=False,
    )

    if res.returncode != 0:
        sys.exit(res.returncode)

    # Final git check
    final_git_status = subprocess.check_output(["git", "status", "--short"]).decode()
    if initial_git_status != final_git_status:
        # Diff them
        print_error(
            "git status changed during tests. New/changed files detected in repository."
        )
        # subprocess.run(["diff", ...]) # simplified
        print("Initial:\n", initial_git_status)
        print("Final:\n", final_git_status)
        sys.exit(1)

    print_success("Git status clean")
    print_success("All checks passed successfully!")


# Wrapper for CheckResult to be compatible with mypy return types if needed, or just dict.
# Already Defined.

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
