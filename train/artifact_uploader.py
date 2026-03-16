"""Non-blocking MLflow artifact uploader with snapshot-then-upload semantics.

Copies artifacts to a temp directory (fast local I/O), then uploads the copy
to MLflow/GCS in a background thread so training is never blocked.  Uses
MlflowClient (thread-safe, no active-run dependency) and a generation counter
to skip stale uploads when training outpaces the network.

Fail-fast policy: a preflight check at creation time verifies the artifact
backend is reachable (imports, auth, connectivity).  Upload failures in the
background thread are logged at ERROR level immediately and re-raised from
drain() so they are never silent.
"""

import logging
import os
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SNAPSHOT_PREFIX = "kotogram_artifact_"


def _cleanup(path: str, is_dir: bool) -> None:
    if is_dir:
        shutil.rmtree(path, ignore_errors=True)
    else:
        Path(path).unlink(missing_ok=True)


class ArtifactUploader:
    """Background uploader that snapshots local files and uploads to MLflow.

    Thread-safety: queue_file / queue_dir are called from the main training
    thread.  The actual upload runs on a single background worker via
    ThreadPoolExecutor so uploads are serialised and won't saturate bandwidth.

    Failure policy: upload failures propagate into the Future and are
    re-raised from drain().  A done-callback logs each failure at ERROR
    level the moment it happens so problems are visible immediately.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="artifact"
        )
        self._generations: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._futures: List[Future[None]] = []

    def preflight(self) -> None:
        """Verify the artifact backend is functional (read AND write).

        Uploads a small probe file to the artifact store, then deletes it.
        This catches missing packages, broken auth, and insufficient
        permissions before training starts.
        """
        from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

        probe_dir = tempfile.mkdtemp(prefix=_SNAPSHOT_PREFIX)
        try:
            probe_path = os.path.join(probe_dir, ".preflight")
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("ok")
            client = MlflowClient()
            client.log_artifact(self._run_id, probe_path, ".preflight")
        finally:
            _cleanup(probe_dir, is_dir=True)

    # -- public API (called from training thread) ----------------------------

    def queue_file(self, src_path: str, artifact_path: str) -> None:
        """Snapshot a single file and queue it for background upload.

        *artifact_path* is the directory inside the MLflow run's artifact
        store (e.g. ``"checkpoint"``).
        """
        if not os.path.isfile(src_path):
            logger.warning("Artifact source not found, skipping upload: %s", src_path)
            return
        tmp_dir = tempfile.mkdtemp(prefix=_SNAPSHOT_PREFIX)
        ok = False
        try:
            snapshot = shutil.copy2(src_path, tmp_dir)
            ok = True
        finally:
            if not ok:
                _cleanup(tmp_dir, is_dir=True)
        gen = self._bump_generation(artifact_path)
        self._submit(self._upload_file, snapshot, tmp_dir, artifact_path, gen)

    def queue_dir(self, src_dir: str, artifact_path: str) -> None:
        """Snapshot a directory tree and queue it for background upload.

        *artifact_path* is the directory inside the MLflow run's artifact
        store (e.g. ``"model"``).
        """
        if not os.path.isdir(src_dir):
            logger.warning(
                "Artifact source dir not found, skipping upload: %s", src_dir
            )
            return
        tmp_parent = tempfile.mkdtemp(prefix=_SNAPSHOT_PREFIX)
        ok = False
        try:
            snapshot = shutil.copytree(
                src_dir,
                os.path.join(tmp_parent, "snapshot"),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__init__.py"),
            )
            ok = True
        finally:
            if not ok:
                _cleanup(tmp_parent, is_dir=True)
        gen = self._bump_generation(artifact_path)
        self._submit(self._upload_dir, snapshot, tmp_parent, artifact_path, gen)

    def drain(self, timeout: float = 600) -> None:
        """Block until every pending upload finishes (or *timeout* expires).

        Must be called before ``mlflow_logging.end_run()`` so the MLflow run
        is still active while uploads complete.  Re-raises the first upload
        failure so broken artifact pipelines are never silent.
        """
        done, _ = futures_wait(self._futures, timeout=timeout)
        first_failure = None
        for fut in done:
            exc = fut.exception()
            if exc is not None and first_failure is None:
                first_failure = exc
        self._futures.clear()
        self._executor.shutdown(wait=False)
        if first_failure is not None:
            raise first_failure

    # -- internals -----------------------------------------------------------

    def _bump_generation(self, artifact_path: str) -> int:
        with self._lock:
            gen = self._generations.get(artifact_path, 0) + 1
            self._generations[artifact_path] = gen
            return gen

    def _is_stale(self, artifact_path: str, generation: int) -> bool:
        with self._lock:
            return generation < self._generations.get(artifact_path, 0)

    def _submit(self, fn, *args) -> None:  # type: ignore[no-untyped-def]
        fut: Future[None] = self._executor.submit(fn, *args)
        fut.add_done_callback(self._on_done)
        self._futures.append(fut)

    @staticmethod
    def _on_done(fut: Future[None]) -> None:
        exc = fut.exception()
        if exc is not None:
            logger.error("Artifact upload failed: %s", exc)

    def _upload_file(
        self,
        snapshot_path: str,
        tmp_dir: str,
        artifact_path: str,
        generation: int,
    ) -> None:
        if self._is_stale(artifact_path, generation):
            logger.debug(
                "Skipping stale artifact upload (%s gen %d)", artifact_path, generation
            )
            _cleanup(tmp_dir, is_dir=True)
            return
        try:
            self._mlflow_log_artifact(snapshot_path, artifact_path)
        finally:
            _cleanup(tmp_dir, is_dir=True)

    def _upload_dir(
        self,
        snapshot_dir: str,
        tmp_parent: str,
        artifact_path: str,
        generation: int,
    ) -> None:
        if self._is_stale(artifact_path, generation):
            logger.debug(
                "Skipping stale artifact upload (%s gen %d)", artifact_path, generation
            )
            _cleanup(tmp_parent, is_dir=True)
            return
        try:
            self._mlflow_log_artifacts(snapshot_dir, artifact_path)
        finally:
            _cleanup(tmp_parent, is_dir=True)

    # -- MLflow wrappers (run on background thread) --------------------------

    def _mlflow_log_artifact(self, local_path: str, artifact_path: str) -> None:
        from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

        client = MlflowClient()
        client.log_artifact(self._run_id, local_path, artifact_path)

    def _mlflow_log_artifacts(self, local_dir: str, artifact_path: str) -> None:
        from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

        client = MlflowClient()
        client.log_artifacts(self._run_id, local_dir, artifact_path)


def create_uploader(run_id: Optional[str]) -> Optional[ArtifactUploader]:
    """Factory: returns an ArtifactUploader if *run_id* is not None.

    Runs a preflight check that verifies the artifact backend (GCS module,
    auth, connectivity).  Raises immediately on failure -- training should
    not start with a broken artifact pipeline.
    """
    if run_id is None:
        return None
    uploader = ArtifactUploader(run_id)
    uploader.preflight()
    return uploader
