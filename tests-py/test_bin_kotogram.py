import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestBinKotogram(unittest.TestCase):
    """Integration tests for bin/kotogram CLI."""

    def setUp(self):
        """Find the bin/kotogram script."""
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent
        self.script_path = project_root / "bin" / "kotogram"

        if not self.script_path.exists():
            self.skipTest(f"bin/kotogram not found at {self.script_path}")

    def run_script(self, args, input_text=None):
        """Run the script as a subprocess."""
        cmd = [sys.executable, str(self.script_path)] + args

        # Ensure we use the project root for finding models/style in dev mode
        env = os.environ.copy()
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent
        env["TRAIN_ROOT"] = str(project_root)

        result = subprocess.run(
            cmd, input=input_text, text=True, capture_output=True, env=env
        )
        return result

    def test_help(self):
        """Test --help argument."""
        result = self.run_script(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("Command-line tool for working with kotograms", result.stdout)

    def test_parse_argument(self, input_text="猫"):
        """Test 'parse' command with text as argument."""
        # Note: input_text defaults to "猫" but can be overridden if needed
        result = self.run_script(["parse", input_text])
        if result.returncode != 0:
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")

        self.assertEqual(result.returncode, 0)
        # Check for kotogram output (should contain delimiters)
        self.assertIn("⌈", result.stdout)
        self.assertIn("⌉", result.stdout)

    def test_parse_stdin(self):
        """Test 'parse' command with stdin input."""
        input_text = "猫"
        result = self.run_script(["parse", "-"], input_text=input_text)

        self.assertEqual(result.returncode, 0)
        self.assertIn("⌈", result.stdout)

    def test_raw_argument(self):
        """Test 'raw' command."""
        result = self.run_script(["raw", "猫"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("Sudachi raw output:", result.stdout)
        self.assertIn("Surface: 猫", result.stdout)

    def test_grammar_command(self):
        """Test 'grammar' command with JSON output."""
        input_text = "私は猫です"
        result = self.run_script(["grammar", input_text])

        self.assertEqual(result.returncode, 0)

        # Verify result is valid JSON
        import json

        # Output might contain warnings/logs, so find the start of JSON
        # Pretty-printed JSON spans multiple lines
        stdout = result.stdout
        json_start = stdout.find("{")

        if json_start == -1:
            self.fail(f"No JSON output found. Full output:\n{stdout}")

        json_str = stdout[json_start:]
        data = json.loads(json_str)

        # Verify required fields
        self.assertIn("is_grammatic", data)
        self.assertIn("formality_score", data)
        self.assertIn("gender_score", data)
