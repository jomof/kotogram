"""Tests for mypy type checking."""

import subprocess
import sys
import unittest


class TestMypy(unittest.TestCase):
    """Test cases for mypy type checking."""

    def test_mypy_kotogram_package(self) -> None:
        """Run mypy on the kotogram package."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "kotogram"],
            capture_output=True,
            text=True,
        )

        # If mypy fails, print the output for debugging
        if result.returncode != 0:
            print("\n--- mypy output for kotogram ---")
            print(result.stdout)
            print(result.stderr)

        self.assertEqual(
            result.returncode,
            0,
            f"mypy found type errors in kotogram package:\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
