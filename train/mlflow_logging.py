"""MLflow experiment tracking for KC training."""

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from kotogram.model import ModelConfig
from train.config import TrainerConfig

logger = logging.getLogger(__name__)

_CLOUD_SQL_PRIVATE_IP = "10.41.0.3"
_DEFAULT_PG_URI = f"postgresql+psycopg2://postgres:mlflow-kotogram-2026@{_CLOUD_SQL_PRIVATE_IP}:5432/mlflow"

_GCS_ARTIFACT_LOCATION = "gs://jomof-public-files/mlflow-artifacts"


def _default_run_name(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
) -> str:
    """Generate a short descriptive run name from key config for easy comparison."""
    base = (
        f"L{model_config.num_layers}_bs{trainer_config.batch_size}_kc{trainer_config.kc_epochs}_"
        f"{int(time.time())}"
    )
    prefix = getattr(trainer_config, "mlflow_run_prefix", "")
    if prefix:
        return f"{prefix} | {base}"
    return base


def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check TCP connectivity without raising exceptions."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def _pg_available() -> bool:
    """Check if the Cloud SQL instance is reachable (direct or via proxy)."""
    if _can_connect(_CLOUD_SQL_PRIVATE_IP, 5432):
        return True
    return _can_connect("localhost", 5432)


def _get_machine_id() -> str:
    """Return a short identifier for the machine (hostname, optional platform)."""
    host = platform.node().split(".")[0]
    return host or "unknown"


def _params_from_config(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build MLflow params from configs."""
    params: Dict[str, Any] = {}

    # Model
    params["num_layers"] = model_config.num_layers
    params["d_model"] = model_config.d_model
    params["hidden_dim"] = model_config.hidden_dim
    params["num_heads"] = model_config.num_heads
    params["dropout"] = model_config.dropout
    params["max_seq_len"] = model_config.max_seq_len
    params["kc_vocab_size"] = model_config.kc_vocab_size
    params["kc_temperature"] = model_config.kc_temperature
    params["kc_threshold"] = model_config.kc_threshold

    # Trainer
    params["batch_size"] = trainer_config.batch_size
    params["learning_rate"] = trainer_config.learning_rate
    params["epochs"] = trainer_config.epochs
    params["kc_epochs"] = trainer_config.kc_epochs
    params["grad_accum_steps"] = trainer_config.grad_accum_steps
    params["ramp_step"] = trainer_config.ramp_step
    params["ramp_posp_threshold"] = trainer_config.ramp_posp_threshold
    params["sample_ratio"] = trainer_config.sample_ratio
    params["gradient_clip"] = trainer_config.gradient_clip
    params["patience"] = trainer_config.patience
    params["eval_every_n_epochs"] = trainer_config.eval_every_n_epochs
    params["retrain"] = trainer_config.retrain

    # KC config subset
    kc = trainer_config.kc_config
    params["kl_sparse_weight"] = kc.kl_sparse_weight
    params["kl_target_rho"] = kc.kl_target_rho
    params["temperature_thawed"] = kc.temperature_thawed
    params["input_mask_ratio"] = kc.input_mask_ratio
    params["gp_unlabeled_weight"] = kc.gp_unlabeled_weight

    if config_path:
        params["config_path"] = config_path

    return params


def start_run(  # pylint: disable=too-many-positional-arguments
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "kotogram-kc",
    artifact_location: Optional[str] = _GCS_ARTIFACT_LOCATION,
) -> Optional[str]:
    """Start an MLflow run and log params + machine.

    Returns the run_id of the started run (used by ArtifactUploader),
    or None if the run could not be started.
    """
    import mlflow  # type: ignore[import-untyped]

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI"):
        pass  # MLflow reads it automatically
    elif _pg_available():
        if _can_connect(_CLOUD_SQL_PRIVATE_IP, 5432):
            mlflow.set_tracking_uri(_DEFAULT_PG_URI)
        else:
            mlflow.set_tracking_uri(
                "postgresql+psycopg2://postgres:mlflow-kotogram-2026@localhost:5432/mlflow"
            )
    else:
        root = Path.cwd()
        mlruns = root / "mlruns"
        mlruns.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(str(mlruns))

    _ensure_experiment(mlflow, experiment_name, artifact_location)
    mlflow.start_run(
        run_name=run_name
        if run_name is not None
        else _default_run_name(model_config, trainer_config)
    )

    params = _params_from_config(model_config, trainer_config, config_path)
    for k, v in params.items():
        if v is not None:
            mlflow.log_param(k, json.dumps(v) if isinstance(v, (dict, list)) else v)

    mlflow.log_param("machine", _get_machine_id())

    active = mlflow.active_run()
    return active.info.run_id if active else None


def _ensure_experiment(
    mlflow: Any,
    experiment_name: str,
    artifact_location: Optional[str],
) -> None:
    """Create experiment with GCS artifact location, or reuse existing one.

    Handles every lifecycle state MLflow can leave an experiment in:
    active with correct location (reuse), active with wrong location
    (rename + recreate), or soft-deleted (restore + rename + recreate).
    """
    from mlflow.entities import ViewType  # type: ignore[import-untyped]
    from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

    client = MlflowClient()
    matches = client.search_experiments(
        view_type=ViewType.ALL,
        filter_string=f"name = '{experiment_name}'",
    )
    existing = next((e for e in matches if e.name == experiment_name), None)

    if existing is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location or ""
        )
        mlflow.set_experiment(experiment_name)
        return

    already_correct = existing.lifecycle_stage == "active" and (
        not artifact_location or existing.artifact_location == artifact_location
    )
    if already_correct:
        mlflow.set_experiment(experiment_name)
        return

    archived_name = f"{experiment_name}-archived-{existing.experiment_id}"
    was_deleted = existing.lifecycle_stage != "active"
    logger.info(
        "Migrating experiment '%s' (id %s, %s): "
        "artifact_location '%s' -> '%s'. Renaming old experiment to '%s'.",
        experiment_name,
        existing.experiment_id,
        existing.lifecycle_stage,
        existing.artifact_location,
        artifact_location,
        archived_name,
    )
    if was_deleted:
        client.restore_experiment(existing.experiment_id)
    client.rename_experiment(existing.experiment_id, archived_name)
    if was_deleted:
        client.delete_experiment(existing.experiment_id)
    mlflow.create_experiment(experiment_name, artifact_location=artifact_location or "")
    mlflow.set_experiment(experiment_name)


def _collect_diagnostic_metrics(diags: dict, to_log: Dict[str, float]) -> None:
    """Extract family-level metrics from kc_diagnostics into to_log."""
    families = diags.get("families", {})
    gp = families.get("grammar_point")
    if isinstance(gp, dict) and "prob_pos_mean" in gp:
        to_log["kc/grammar_point_posp"] = float(gp["prob_pos_mean"])

    for name, mse in diags.get("mse_families", {}).items():
        if isinstance(mse, dict) and "discrete_accuracy" in mse:
            to_log[f"kc/{name}_acc"] = float(mse["discrete_accuracy"])

    for name, fam in families.items():
        if isinstance(fam, dict):
            if fam.get("avg_pos") is not None:
                to_log[f"kc/{name}_avg_pos"] = float(fam["avg_pos"])
            if fam.get("med_pos") is not None:
                to_log[f"kc/{name}_med_pos"] = float(fam["med_pos"])

    for name, bert in diags.get("bert_families", {}).items():
        if isinstance(bert, dict):
            if "loss_mean" in bert:
                to_log[f"kc/{name}_loss"] = float(bert["loss_mean"])
            if "top1_accuracy" in bert:
                to_log[f"kc/{name}_top1"] = float(bert["top1_accuracy"])
            if "top5_accuracy" in bert:
                to_log[f"kc/{name}_top5"] = float(bert["top5_accuracy"])
            if bert.get("top1_pos_only_accuracy"):
                pos_only = float(bert["top1_pos_only_accuracy"])
                to_log[f"kc/{name}_top1_pos_only"] = pos_only
                to_log[f"kc/{name}_kc_gain"] = float(bert["top1_accuracy"]) - pos_only


def log_kc_epoch(epoch: int, metrics: Dict[str, Any]) -> None:
    """Log KC epoch metrics to the active MLflow run."""
    import mlflow  # type: ignore[import-untyped]

    if not mlflow.active_run():
        return

    to_log: Dict[str, float] = {}

    if "total_loss" in metrics:
        val = metrics["total_loss"]
        if isinstance(val, list):
            val = val[-1] if val else 0.0
        to_log["kc/total_loss"] = float(val)

    diags = metrics.get("kc_diagnostics")
    if isinstance(diags, dict):
        _collect_diagnostic_metrics(diags, to_log)

    for key in (
        "alive_kcs",
        "total_k_mean",
        "total_k_p10",
        "total_k_p50",
        "total_k_p90",
        "kc_threshold",
    ):
        if metrics.get(key) is not None:
            to_log[f"kc/{key}"] = float(metrics[key])

    if metrics.get("sentence_count") is not None:
        val = metrics["sentence_count"]
        if isinstance(val, list):
            val = val[-1] if val else 0
        to_log["kc/sentence_count"] = int(val)

    for k, v in to_log.items():
        mlflow.log_metric(k, v, step=epoch)


def end_run() -> None:
    """End the active MLflow run."""
    import mlflow  # type: ignore[import-untyped]

    if mlflow.active_run():
        mlflow.end_run()
