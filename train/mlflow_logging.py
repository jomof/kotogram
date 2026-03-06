"""MLflow experiment tracking for KC training."""

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from kotogram.model import ModelConfig
from train.config import KCConfig, TrainerConfig

_CLOUD_SQL_PRIVATE_IP = "10.41.0.3"
_DEFAULT_PG_URI = (
    f"postgresql+psycopg2://postgres:mlflow-kotogram-2026@{_CLOUD_SQL_PRIVATE_IP}:5432/mlflow"
)


def _default_run_name(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
) -> str:
    """Generate a short descriptive run name from key config for easy comparison."""
    return (
        f"L{model_config.num_layers}_bs{trainer_config.batch_size}_kc{trainer_config.kc_epochs}_"
        f"{int(time.time())}"
    )


def _pg_available() -> bool:
    """Check if the Cloud SQL instance is reachable (direct or via proxy)."""
    import socket

    try:
        with socket.create_connection((_CLOUD_SQL_PRIVATE_IP, 5432), timeout=1):
            return True
    except OSError:
        pass
    # Fall back to localhost (proxy running on Mac or other non-VPC machine)
    try:
        with socket.create_connection(("localhost", 5432), timeout=1):
            return True
    except OSError:
        return False


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


def start_run(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "kotogram-kc",
) -> None:
    """Start an MLflow run and log params + machine."""
    import mlflow  # type: ignore[import-untyped]

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI"):
        pass  # MLflow reads it automatically
    elif _pg_available():
        import socket

        # Use private IP directly if reachable, otherwise localhost (proxy)
        try:
            with socket.create_connection((_CLOUD_SQL_PRIVATE_IP, 5432), timeout=1):
                mlflow.set_tracking_uri(_DEFAULT_PG_URI)
        except OSError:
            mlflow.set_tracking_uri(
                "postgresql+psycopg2://postgres:mlflow-kotogram-2026@localhost:5432/mlflow"
            )
    else:
        root = Path.cwd()
        mlruns = root / "mlruns"
        mlruns.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(str(mlruns))

    mlflow.set_experiment(experiment_name)
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


def log_kc_epoch(epoch: int, metrics: Dict[str, Any]) -> None:
    """Log KC epoch metrics to the active MLflow run."""
    import mlflow  # type: ignore[import-untyped]

    if not mlflow.active_run():
        return

    # Core metrics
    to_log: Dict[str, float] = {}

    if "total_loss" in metrics:
        val = metrics["total_loss"]
        if isinstance(val, list):
            val = val[-1] if val else 0.0
        to_log["kc/total_loss"] = float(val)

    # Grammar point PosP from kc_diagnostics
    diags = metrics.get("kc_diagnostics")
    if isinstance(diags, dict):
        families = diags.get("families", {})
        gp = families.get("grammar_point")
        if isinstance(gp, dict) and "prob_pos_mean" in gp:
            to_log["kc/grammar_point_posp"] = float(gp["prob_pos_mean"])

    # Sizing metrics
    if metrics.get("alive_kcs") is not None:
        to_log["kc/alive_kcs"] = int(metrics["alive_kcs"])
    if metrics.get("total_k_mean") is not None:
        to_log["kc/total_k_mean"] = float(metrics["total_k_mean"])
    if metrics.get("total_k_p10") is not None:
        to_log["kc/total_k_p10"] = float(metrics["total_k_p10"])
    if metrics.get("total_k_p50") is not None:
        to_log["kc/total_k_p50"] = float(metrics["total_k_p50"])
    if metrics.get("total_k_p90") is not None:
        to_log["kc/total_k_p90"] = float(metrics["total_k_p90"])
    if metrics.get("kc_threshold") is not None:
        to_log["kc/kc_threshold"] = float(metrics["kc_threshold"])
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
