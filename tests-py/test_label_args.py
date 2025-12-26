import subprocess
import sys
import unittest


class TestLabelArgs(unittest.TestCase):
    def test_help_contains_verbose(self):
        """Regression test: Ensure --verbose is a valid argument in scripts/label.py.

        This guards against the regression where missing --verbose definition caused
        AttributeError: 'Namespace' object has no attribute 'verbose'.
        """
        cmd = [sys.executable, "-m", "scripts.label", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, "Failed to run scripts.label --help")
        self.assertIn(
            "--verbose", result.stdout, "Missing --verbose argument in label.py usage"
        )


if __name__ == "__main__":
    unittest.main()
