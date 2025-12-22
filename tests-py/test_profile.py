"""Tests for profiling module."""

import os
import tempfile
import unittest
from unittest import mock

from kotogram import SudachiJapaneseParser
from kotogram.profile import (
    ProfileReport,
    cleanup_shared_memory,
    get_profile_report,
    increment_profile_counter,
    reset_profile_counters,
)


class TestProfileCounter(unittest.TestCase):
    """Test cases for profile counter functionality."""

    def setUp(self) -> None:
        """Set up test with profiling enabled and temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.env_patcher = mock.patch.dict(
            os.environ, {"PROFILE_KOTOGRAM": "1", "TRAIN_ROOT": self.temp_dir}
        )
        self.env_patcher.start()
        reset_profile_counters()

    def tearDown(self) -> None:
        """Clean up after tests."""
        cleanup_shared_memory()
        self.env_patcher.stop()

    def test_increment_counter_creates_entry(self) -> None:
        """Incrementing counter creates an entry with caller name."""
        increment_profile_counter()

        report = get_profile_report()
        # Should have exactly one counter
        self.assertEqual(len(report.counters), 1)
        # Counter name should include this test method
        keys = list(report.counters.keys())
        self.assertIn("test_increment_counter_creates_entry", keys[0])
        # Count should be 1
        self.assertEqual(report.counters[keys[0]], 1)

    def test_increment_counter_accumulates(self) -> None:
        """Multiple increments accumulate correctly."""
        for _ in range(5):
            increment_profile_counter()

        report = get_profile_report()
        keys = list(report.counters.keys())
        self.assertEqual(report.counters[keys[0]], 5)

    def test_different_functions_have_different_counters(self) -> None:
        """Different calling functions get different counters."""

        def func_a() -> None:
            increment_profile_counter()

        def func_b() -> None:
            increment_profile_counter()

        func_a()
        func_b()
        func_a()

        report = get_profile_report()
        self.assertEqual(len(report.counters), 2)
        # Find counters by function name
        func_a_count = 0
        func_b_count = 0
        for key, count in report.counters.items():
            if "func_a" in key:
                func_a_count = count
            elif "func_b" in key:
                func_b_count = count
        self.assertEqual(func_a_count, 2)
        self.assertEqual(func_b_count, 1)

    def test_report_has_timestamp(self) -> None:
        """Profile report includes timestamp."""
        report = get_profile_report()
        self.assertIsInstance(report.timestamp, str)
        self.assertGreater(len(report.timestamp), 0)


class TestProfileDisabled(unittest.TestCase):
    """Test cases when profiling is disabled."""

    def setUp(self) -> None:
        """Set up test with profiling disabled."""
        self.env_patcher = mock.patch.dict(os.environ, {"PROFILE_KOTOGRAM": "0"})
        self.env_patcher.start()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.env_patcher.stop()

    def test_increment_does_nothing_when_disabled(self) -> None:
        """Increment silently does nothing when profiling disabled."""
        # This should not raise any errors
        increment_profile_counter()

    def test_report_empty_when_disabled(self) -> None:
        """Report returns empty counters when profiling disabled."""
        report = get_profile_report()
        self.assertEqual(report.counters, {})


class TestProfileReport(unittest.TestCase):
    """Test cases for profile report dataclass."""

    def test_profile_report_structure(self) -> None:
        """ProfileReport has expected structure."""
        report = ProfileReport(counters={"test": 5}, timestamp="2024-01-01T00:00:00")
        self.assertEqual(report.counters["test"], 5)
        self.assertEqual(report.timestamp, "2024-01-01T00:00:00")


class TestSudachiParserProfileIntegration(unittest.TestCase):
    """Integration tests for profiling with SudachiJapaneseParser."""

    def setUp(self) -> None:
        """Set up test with profiling enabled and temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.env_patcher = mock.patch.dict(
            os.environ, {"PROFILE_KOTOGRAM": "1", "TRAIN_ROOT": self.temp_dir}
        )
        self.env_patcher.start()
        reset_profile_counters()

    def tearDown(self) -> None:
        """Clean up after tests."""
        cleanup_shared_memory()
        self.env_patcher.stop()

    def test_japanese_to_kotogram_increments_counter(self) -> None:
        """SudachiJapaneseParser.japanese_to_kotogram increments profile counter."""
        parser = SudachiJapaneseParser(dict_type="full")
        parser.japanese_to_kotogram("テスト")
        parser.japanese_to_kotogram("日本語")

        report = get_profile_report()

        # Find the counter for japanese_to_kotogram
        found = False
        for key, count in report.counters.items():
            if "japanese_to_kotogram" in key:
                self.assertEqual(count, 2)
                found = True
                break

        self.assertTrue(
            found, f"Counter not found. Keys: {list(report.counters.keys())}"
        )

    def test_report_writes_json_file(self) -> None:
        """get_profile_report writes report to .profile/report.json."""
        increment_profile_counter()

        get_profile_report()

        report_path = os.path.join(self.temp_dir, ".profile", "report.json")
        self.assertTrue(os.path.exists(report_path))

        # Verify JSON content
        import json

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("counters", data)
        self.assertIn("timestamp", data)


if __name__ == "__main__":
    unittest.main()
