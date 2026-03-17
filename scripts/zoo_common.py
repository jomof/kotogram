"""Shared MLflow helpers for the zoo CLI (run resolution, tracking URI)."""

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from train.mlflow_logging import can_connect, pg_available

logger = logging.getLogger(__name__)

_CLOUD_SQL_PRIVATE_IP = "10.41.0.3"
_DEFAULT_PG_URI = f"postgresql+psycopg2://postgres:mlflow-kotogram-2026@{_CLOUD_SQL_PRIVATE_IP}:5432/mlflow"
_LOCAL_PG_URI = (
    "postgresql+psycopg2://postgres:mlflow-kotogram-2026@localhost:5432/mlflow"
)
_DEFAULT_EXPERIMENT_NAME = "kotogram-kc"


def get_tracking_uri() -> str:
    """Resolve MLflow tracking URI (env, Cloud SQL, local PG, or local mlruns)."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    if pg_available():
        if can_connect(_CLOUD_SQL_PRIVATE_IP, 5432):
            return _DEFAULT_PG_URI
        return _LOCAL_PG_URI
    root = Path.cwd()
    mlruns = root / "mlruns"
    mlruns.mkdir(exist_ok=True)
    return str(mlruns)


def resolve_run_by_name_fragment(
    name_frag: str,
    experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
) -> str:
    """Find the single MLflow run whose name contains *name_frag*.

    Returns the run_id of the matching run.
    Raises ValueError if no runs match or more than one run matches.
    """
    from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

    uri = tracking_uri if tracking_uri is not None else get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    # Resolve experiment by name
    experiments = client.search_experiments(
        filter_string=f"name = '{experiment_name}'",
    )
    exp = next((e for e in experiments if e.name == experiment_name), None)
    if exp is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            "Ensure MLflow tracking is reachable and the experiment exists."
        )

    # Search runs by run name (substring match). Escape LIKE metacharacters in fragment.
    escaped = name_frag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    filter_string = f'tags."mlflow.runName" LIKE "%{escaped}%"'
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
    )

    run_ids: Sequence[str] = [r.info.run_id for r in runs]
    if not run_ids:
        raise ValueError(
            f"No MLflow run found with name containing '{name_frag}' "
            f"in experiment '{experiment_name}'."
        )
    if len(run_ids) > 1:
        raise ValueError(
            f"Name fragment '{name_frag}' matches {len(run_ids)} runs; "
            "use a more specific fragment so exactly one run matches."
        )
    return run_ids[0]
