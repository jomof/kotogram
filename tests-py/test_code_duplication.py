import os
import subprocess


def test_no_code_duplication():
    """
    Run pylint's duplicate code detection as a test.
    We separate this from the main hygiene check because it can be slow and
    is more of a code quality metric than a strict correctness check.
    """
    # Target directories
    targets = "kotogram scripts train tests-py train_style bin/kotogram"

    # Command:
    # -j 0: Parallel execution
    # --disable=all: Turn off all other checks
    # --enable=duplicate-code: Turn on ONLY duplication check
    # --ignore=vulture_whitelist.py: Ignore this file
    cmd = [
        "pylint",
        "-j",
        "0",
        "--disable=all",
        "--enable=duplicate-code",
        "--ignore=vulture_whitelist.py",
    ] + targets.split()

    env = os.environ.copy()
    cwd = os.getcwd()
    # Ensure PYTHONPATH includes project root and tests-py for proper resolution if needed
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{cwd}:{cwd}/tests-py"

    # Run the command
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

    # Check if successful
    if result.returncode != 0:
        # Pylint returns non-zero if issues are found
        # We fail the test and print the output
        print("\n" + result.stdout)
        print(result.stderr)
        assert False, (
            f"Code duplication detected by Pylint (exit code {result.returncode})"
        )
