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

# Forbidden types to check for
# Bare 'except:' is also forbidden (matches empty capture)
FORBIDDEN_EXCEPTIONS = {
    "Exception",
    "BaseException",
    "IOError",
    "OSError",
    "ValueError",
    "BaseError",
    "FileNotFoundError",
    "RuntimeError",
    "ImportError",
    "TypeError",
    "json.JSONDecodeError",
    "KeyError",
    "sqlite3.OperationalError",
}


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
    # pylint: disable=too-many-locals

    # Load whitelist to ignore approved instances
    whitelist_entries = set()
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Whitelist format is roughly "filename:lineno: content"
                    whitelist_entries.add(line)

    # Grep for all exception handlers
    cmd = (
        r'grep -rnH "^\s*except\b.*:" '
        "kotogram scripts train tests-py train_style bin/kotogram "
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
            return CheckResult("broad exception check", True, "")

    output = stdout.decode()
    val_errors = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        if line in whitelist_entries:
            continue

        parts = line.split(":", 2)
        if len(parts) < 3:
            continue

        fpath, lineno, content = parts[0], parts[1], parts[2]

        # Analyze content for forbidden types
        code_part = content.split("#", 1)[0].strip()

        if code_part.endswith(":"):
            code_part = code_part[:-1].strip()

        if code_part.startswith("except"):
            code_part = code_part[6:].strip()

        if " as " in code_part:
            code_part = code_part.split(" as ", 1)[0].strip()

        if code_part.startswith("(") and code_part.endswith(")"):
            code_part = code_part[1:-1]

        caught_types = [t.strip() for t in code_part.split(",")]

        for t in caught_types:
            if not t:
                val_errors.append(f"{fpath}:{lineno}: Bare 'except:' is forbidden")
                break

            if t in FORBIDDEN_EXCEPTIONS:
                val_errors.append(f"{fpath}:{lineno}: Forbidden exception '{t}'")
                break

    if val_errors:
        shaming_msg = (
            "To whoever is working on this code right now: catching broad exceptions "
            "hides errors and makes the system difficult to debug. You know this, "
            "and you should be ashamed. Stop it!"
        )
        return CheckResult(
            "broad exception check",
            False,
            "Found forbidden broad exception handling:\n"
            + "\n".join(val_errors)
            + f"\n\n{RED}{shaming_msg}{RESET}",
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

    cmd = (
        r'grep -rnH "^\s*except\b" kotogram scripts train train_style bin/kotogram '
        '| grep -v "worker-init=special-carveout"'
    )
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
    """Vulture check in two passes."""
    cmd_prod = "vulture kotogram scripts train scripts/vulture_whitelist.py train_style bin/kotogram"
    res_prod = await run_command(cmd_prod)
    if not res_prod.success:
        return CheckResult(
            "vulture (production strict)",
            False,
            f"Vulture found unused production code (not used by other production code):\n{res_prod.output}",
        )

    cmd_full = "vulture kotogram scripts train tests-py scripts/vulture_whitelist.py train_style bin/kotogram"
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
    """Pylint check."""
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
    shutil.rmtree("dist_py", ignore_errors=True)

    # Build to isolated directory
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

    norm_files = []
    for f in files:
        f = re.sub(r"kotogram-.*\.dist-info", "kotogram-*.dist-info", f)
        # Normalize license location (some builds put it in licenses/ subdir)
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
    """Verify TypeScript package contents."""
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


async def main() -> None:
    # pylint: disable=too-many-locals
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
    args = parser.parse_args()

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
        check_python_package(),
    ]

    pending = [asyncio.create_task(t) for t in tasks]
    failed = False

    # Run general python static checks in parallel
    results = await asyncio.gather(*pending)

    for result in results:
        if not result.success:
            print_error(result.output)
            failed = True

    # Run TypeScript tasks serially to avoid build conflicts
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

    # Run Pytest (if not hygiene mode)
    if not args.hygiene:
        print(f"\n{BLUE}Running Pytest...{RESET}")

        env = os.environ.copy()
        env["CI"] = "true"
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

            # Ensure mode is run
            config["mode"] = "run"

            print(
                f"{BLUE}Running Pytest confined with {args.confinement_config}{RESET}"
            )

            # --- Confinement Verification Probe (Mac Only) ---
            if sys.platform == "darwin":
                print(f"{BLUE}Verifying confinement (Probe)...{RESET}")
                probe_file = "confinement_probe_fail.txt"
                # Python one-liner to attempt write
                probe_cmd = [
                    sys.executable,
                    "-c",
                    f"import sys\ntry:\n    open('{probe_file}', 'w').close()\nexcept OSError:\n    sys.exit(1)",
                ]
                # Should fail
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
            # --------------------------------------
            # pylint: disable=no-member
            pytest_res = confine_lib.confine(pytest_cmd, config, env=env, check=False)  # type: ignore[attr-defined]
        else:
            pytest_res = subprocess.run(
                pytest_cmd,
                env=env,
                check=False,
            )

        if pytest_res.returncode != 0:
            sys.exit(pytest_res.returncode)

    # Final git check
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
