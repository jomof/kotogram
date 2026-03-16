import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from train.artifact_uploader import ArtifactUploader, create_uploader


class TestArtifactUploaderSnapshot(unittest.TestCase):
    """Verify snapshot-then-upload: originals can be modified while upload runs."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.src_file = os.path.join(self.test_dir, "checkpoint.pt")
        with open(self.src_file, "w", encoding="utf-8") as f:
            f.write("v1")

        self.src_dir = os.path.join(self.test_dir, "model")
        os.makedirs(self.src_dir)
        with open(os.path.join(self.src_dir, "model.pt"), "w", encoding="utf-8") as f:
            f.write("weights-v1")
        with open(os.path.join(self.src_dir, "model.json"), "w", encoding="utf-8") as f:
            f.write("{}")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_file_snapshot_is_independent_of_original(self, mock_log):
        captured = {}

        def capture_path(local_path, _artifact_path):
            with open(local_path, encoding="utf-8") as f:
                captured["content"] = f.read()

        mock_log.side_effect = capture_path

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_file(self.src_file, "checkpoint")

        # Mutate original immediately
        with open(self.src_file, "w", encoding="utf-8") as f:
            f.write("v2-overwritten")

        uploader.drain(timeout=10)

        self.assertEqual(captured["content"], "v1")

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifacts")
    def test_dir_snapshot_is_independent_of_original(self, mock_log):
        captured = {}

        def capture_dir(local_dir, _artifact_path):
            with open(os.path.join(local_dir, "model.pt"), encoding="utf-8") as f:
                captured["content"] = f.read()

        mock_log.side_effect = capture_dir

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_dir(self.src_dir, "model")

        # Mutate original immediately
        with open(os.path.join(self.src_dir, "model.pt"), "w", encoding="utf-8") as f:
            f.write("weights-v2-overwritten")

        uploader.drain(timeout=10)

        self.assertEqual(captured["content"], "weights-v1")

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifacts")
    def test_dir_snapshot_excludes_pycache(self, mock_log):
        os.makedirs(os.path.join(self.src_dir, "__pycache__"))
        with open(
            os.path.join(self.src_dir, "__pycache__", "mod.pyc"), "w", encoding="utf-8"
        ) as f:
            f.write("bytecode")
        with open(
            os.path.join(self.src_dir, "__init__.py"), "w", encoding="utf-8"
        ) as f:
            f.write("")

        snapshot_dirs = []

        def capture_dir(local_dir, _artifact_path):
            snapshot_dirs.append(local_dir)

        mock_log.side_effect = capture_dir

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_dir(self.src_dir, "model")
        uploader.drain(timeout=10)

        self.assertEqual(len(snapshot_dirs), 1)
        snap = snapshot_dirs[0]
        self.assertFalse(os.path.exists(os.path.join(snap, "__pycache__")))
        self.assertFalse(os.path.exists(os.path.join(snap, "__init__.py")))


class TestArtifactUploaderGeneration(unittest.TestCase):
    """Verify latest-wins deduplication via generation counters."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_stale_upload_is_skipped(self, mock_log):
        """When multiple versions are queued, only the latest is uploaded."""
        gate = threading.Event()
        upload_contents = []

        def slow_upload(local_path, _artifact_path):
            gate.wait(timeout=10)
            with open(local_path, encoding="utf-8") as f:
                upload_contents.append(f.read())

        mock_log.side_effect = slow_upload

        uploader = ArtifactUploader("test-run-id")

        # Queue v1 (will block on gate)
        f1 = os.path.join(self.test_dir, "v1.pt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("version-1")
        uploader.queue_file(f1, "checkpoint")

        # Queue v2 and v3 while v1 is being "uploaded"
        time.sleep(0.05)
        f2 = os.path.join(self.test_dir, "v2.pt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("version-2")
        uploader.queue_file(f2, "checkpoint")

        f3 = os.path.join(self.test_dir, "v3.pt")
        with open(f3, "w", encoding="utf-8") as f:
            f.write("version-3")
        uploader.queue_file(f3, "checkpoint")

        # Release the gate -- v1 uploads, v2 should be skipped, v3 uploads
        gate.set()
        uploader.drain(timeout=10)

        self.assertIn("version-1", upload_contents)
        self.assertNotIn("version-2", upload_contents)
        self.assertIn("version-3", upload_contents)

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_different_artifact_paths_are_independent(self, mock_log):
        """Generation counters are per-artifact_path."""
        uploaded = []

        def record(_local_path, artifact_path):
            uploaded.append(artifact_path)

        mock_log.side_effect = record

        uploader = ArtifactUploader("test-run-id")

        f1 = os.path.join(self.test_dir, "a.pt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("data")
        f2 = os.path.join(self.test_dir, "b.pt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("data")

        uploader.queue_file(f1, "checkpoint")
        uploader.queue_file(f2, "model")
        uploader.drain(timeout=10)

        self.assertIn("checkpoint", uploaded)
        self.assertIn("model", uploaded)


class TestArtifactUploaderErrorHandling(unittest.TestCase):
    """Verify upload failures are raised, not silenced."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self._clean_stale_snapshots()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self._clean_stale_snapshots()

    @staticmethod
    def _clean_stale_snapshots():
        tmp = tempfile.gettempdir()
        for name in os.listdir(tmp):
            if name.startswith("kotogram_artifact_"):
                shutil.rmtree(os.path.join(tmp, name), ignore_errors=True)

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_upload_failure_raises_from_drain(self, mock_log):
        mock_log.side_effect = RuntimeError("GCS auth failed")

        f = os.path.join(self.test_dir, "checkpoint.pt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("data")

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_file(f, "checkpoint")
        with self.assertRaises(RuntimeError):
            uploader.drain(timeout=10)

    def test_snapshot_of_missing_file_does_not_crash(self):
        uploader = ArtifactUploader("test-run-id")
        uploader.queue_file("/nonexistent/checkpoint.pt", "checkpoint")
        uploader.drain(timeout=10)

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_temp_files_cleaned_after_upload(self, mock_log):
        mock_log.return_value = None

        f = os.path.join(self.test_dir, "checkpoint.pt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("data")

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_file(f, "checkpoint")
        uploader.drain(timeout=10)

        remaining = [
            d
            for d in os.listdir(tempfile.gettempdir())
            if d.startswith("kotogram_artifact_")
        ]
        self.assertEqual(remaining, [])

    @patch("train.artifact_uploader.ArtifactUploader._mlflow_log_artifact")
    def test_temp_files_cleaned_after_failure(self, mock_log):
        mock_log.side_effect = RuntimeError("upload failed")

        f = os.path.join(self.test_dir, "checkpoint.pt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("data")

        uploader = ArtifactUploader("test-run-id")
        uploader.queue_file(f, "checkpoint")
        with self.assertRaises(RuntimeError):
            uploader.drain(timeout=10)

        remaining = [
            d
            for d in os.listdir(tempfile.gettempdir())
            if d.startswith("kotogram_artifact_")
        ]
        self.assertEqual(remaining, [])


class TestCreateUploader(unittest.TestCase):
    def test_returns_none_for_none_run_id(self):
        self.assertIsNone(create_uploader(None))

    @patch("train.artifact_uploader.ArtifactUploader.preflight")
    def test_returns_uploader_for_valid_run_id(self, mock_preflight):
        mock_preflight.return_value = None
        uploader = create_uploader("some-run-id")
        self.assertIsInstance(uploader, ArtifactUploader)
        uploader.drain(timeout=1)

    @patch("train.artifact_uploader.ArtifactUploader.preflight")
    def test_preflight_failure_propagates(self, mock_preflight):
        mock_preflight.side_effect = ModuleNotFoundError(
            "No module named 'google.cloud'"
        )
        with self.assertRaises(ModuleNotFoundError):
            create_uploader("some-run-id")


if __name__ == "__main__":
    unittest.main()
