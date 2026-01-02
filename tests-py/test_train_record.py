import os
import re
import shutil
import subprocess
import tempfile
import unittest
from typing import Tuple


def _setup_test_env(temp_dir: str, test_content: str) -> str:
    """Writes the test script and copies necessary plugin files."""
    # 1. Write the test file
    target_file = os.path.join(temp_dir, "test_script.py")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    # 2a. Copy the instrumentation library
    instr_src = os.path.join(os.getcwd(), "tests-py", "instrumentation.py")
    instr_dst = os.path.join(temp_dir, "instrumentation.py")
    if os.path.exists(instr_src):
        shutil.copy(instr_src, instr_dst)

    # 2b. Copy the plugin file
    plugin_src = os.path.join(os.getcwd(), "tests-py", "train_record_conftest.py")
    plugin_dst = os.path.join(temp_dir, "train_record_conftest.py")
    if os.path.exists(plugin_src):
        shutil.copy(plugin_src, plugin_dst)

    # 3. Create conftest.py to load the plugin
    conftest_content = """
import sys
import os
sys.path.append(os.path.dirname(__file__))
import train_record_conftest

def pytest_configure(config): train_record_conftest.pytest_configure(config)
def pytest_sessionstart(session): train_record_conftest.pytest_sessionstart(session)
def pytest_sessionfinish(session, exitstatus): train_record_conftest.pytest_sessionfinish(session, exitstatus)
"""
    conftest_path = os.path.join(temp_dir, "conftest.py")
    with open(conftest_path, "w", encoding="utf-8") as f:
        f.write(conftest_content)

    return target_file


def run_test_with_env(
    enable_roots: bool = True, fail_on_const: str = ""
) -> Tuple[int, str]:
    # content of the test script to run
    test_content = """
import pytest

def constant_func(a, b):
    pass

def varying_func(x):
    pass

def complex_func(c):
    pass

from typing import Optional
def optional_func(opt: Optional[int] = None):
    pass


def test_trigger_execution():
    # Constant calls
    constant_func(1, "static")
    constant_func(1, "static")

    # Varying calls
    varying_func(10)
    varying_func(20)

    # Complex calls (list is not basic)
    complex_func([1, 2])
    complex_func([1, 2])

    # Optional func called with value ONLY
    optional_func(1)
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        target_file = _setup_test_env(temp_dir, test_content)

        # 4. Prepare Environment
        env = os.environ.copy()
        if enable_roots:
            env["TRAIN_RECORD_ROOTS"] = temp_dir
        else:
            # Explicitly remove if it exists (inherited from parent process)
            env.pop("TRAIN_RECORD_ROOTS", None)

        if fail_on_const:
            env["TRAIN_RECORD_FAIL_ON_CONST"] = fail_on_const

        # Add project root to PYTHONPATH so 'train' and 'kotogram' can be imported
        # This is needed now that instrumentation depends on train.paths -> kotogram.locations
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        # 5. Run Pytest
        cmd = ["pytest", target_file, "-s"]

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

        # Strip ANSI codes for easier assertion
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_output = ansi_escape.sub("", proc.stdout + proc.stderr)

        return proc.returncode, clean_output


class TestTrainRecorder(unittest.TestCase):
    def test_recorder_active(self):
        rc, output = run_test_with_env(enable_roots=True)
        self.assertEqual(rc, 0)

        # Check output for report
        self.assertIn("Found 5 universally constant parameters", output)

        # Verify constant_func args are caught
        # a=1
        self.assertIn("constant_func", output)
        self.assertIn("a=1", output)
        self.assertIn("b='static'", output)

        # Check for Optional Never None Report
        self.assertIn("Found 1 optional parameters that were never None", output)
        # optional_func(opt) should be listed because it was only called with 1, never None
        # and it has Optional in signature
        self.assertIn("optional_func(opt)", output)

        # Verify varying_func args are NOT caught
        if "varying_func" in output:
            self.assertNotIn("x=", output)

        # Verify complex_func ignored (lists are not basic)
        if "complex_func" in output:
            self.assertNotIn("c=", output)

    def test_recorder_inactive_when_no_env(self):
        _, output = run_test_with_env(enable_roots=False)
        self.assertNotIn("[train-record]", output)

    def test_recorder_fail_on_const(self):
        rc, output = run_test_with_env(enable_roots=True, fail_on_const="1")

        self.assertNotEqual(rc, 0)
        self.assertIn("FAILING session due to TRAIN_RECORD_FAIL_ON_CONST=1", output)
