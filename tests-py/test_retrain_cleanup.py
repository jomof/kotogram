import unittest
from unittest.mock import patch

# pylint: disable=too-many-positional-arguments
from scripts.train_style import cleanup_profile_if_retrain


class TestRetrainCleanup(unittest.TestCase):
    def setUp(self):
        """Setup patches for all tests."""
        self.patches = {
            "get_profile_dir": patch("scripts.train_style.get_profile_dir"),
            "exists": patch("scripts.train_style.os.path.exists"),
            "rmtree": patch("scripts.train_style.shutil.rmtree"),
            "makedirs": patch("scripts.train_style.os.makedirs"),
        }
        self.mocks = {k: p.start() for k, p in self.patches.items()}
        self.addCleanup(patch.stopall)

    def test_cleanup_with_retrain_flag(self):
        """Verify profile directory is deleted when --retrain is passed."""
        # Setup
        self.mocks["get_profile_dir"].return_value = "/tmp/profile-test"
        self.mocks["exists"].return_value = True
        args = ["train_style.py", "--config", "foo.json", "--retrain"]

        # Execute
        cleanup_profile_if_retrain(args)

        # Verify
        self.mocks["get_profile_dir"].assert_called_once()
        self.mocks["exists"].assert_called_once_with("/tmp/profile-test")
        self.mocks["rmtree"].assert_called_once()
        self.assertIn("/tmp/profile-test", self.mocks["rmtree"].call_args[0][0])

    def test_cleanup_without_retrain_flag(self):
        """Verify profile directory is NOT deleted when --retrain is missing."""
        # Setup
        args = ["train_style.py", "--config", "foo.json"]

        # Execute
        cleanup_profile_if_retrain(args)

        # Verify
        self.mocks["rmtree"].assert_not_called()

    def test_cleanup_directory_exists(self):
        """Verify rmtree and makedirs are called if directory exists."""
        # Setup
        self.mocks["get_profile_dir"].return_value = "/tmp/profile-test"
        self.mocks["exists"].return_value = True

        # Run the cleanup function with --retrain
        cleanup_profile_if_retrain(["script.py", "--retrain"])

        # Should have called rmtree
        self.mocks["rmtree"].assert_called_once()
        # Should recreate the directory
        self.mocks["makedirs"].assert_called_once()

    def test_cleanup_directory_missing(self):
        """Verify logic handles missing directory gracefully."""
        # Setup
        self.mocks["get_profile_dir"].return_value = "/tmp/profile-test"
        # The logic: if profile_dir and os.path.exists(profile_dir): ...
        self.mocks["exists"].return_value = False

        # Execute
        cleanup_profile_if_retrain(["train_style.py", "--retrain"])

        # Verify
        self.mocks["rmtree"].assert_not_called()
        self.mocks["makedirs"].assert_not_called()
