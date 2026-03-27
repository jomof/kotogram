#!/usr/bin/env python3
"""Optuna hyperparameter search for recon_bpd training.

Minimizes BPD (bits per attended token-position) over the full
TrainConfig search space.  Each trial is logged as an independent
MLflow run with per-epoch step metrics (matching the train_style
pattern).  Supports MedianPruner for early stopping of unpromising
trials and optional persistent storage for resumable studies.

Usage:
    python -m scripts.recon_bpd_optuna
    python -m scripts.recon_bpd_optuna --n-trials 100 --epochs-per-trial 50
    python -m scripts.recon_bpd_optuna --storage sqlite:///optuna.db
    python -m scripts.recon_bpd_optuna --no-mlflow
    python3 -m scripts.recon_bpd_optuna --n-trials 1000 --sample-ratio 0.15 --epochs-per-trial 15 --storage sqlite:///.cache/recon_bpd/optuna.db --consistency-weight-only
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
from typing import Optional

import optuna

from scripts.recon_bpd import TrainConfig, load_checkpoint, train


def suggest_config(
    trial: optuna.Trial,
    epochs: int,
    sample_ratio: Optional[float] = None,
    patience: Optional[int] = None,
    consistency_weight_only: bool = False,
) -> TrainConfig:
    """Build a TrainConfig from Optuna trial suggestions."""
    if consistency_weight_only:
        return TrainConfig(
            epochs=epochs,
            sample_ratio=sample_ratio,
            patience=patience,
            verbose=True,
            seed=42,
            consistency_weight=trial.suggest_categorical(
                "consistency_weight", [0.0, 0.00001, 0.0003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
            ),
            input_mask_ratio=trial.suggest_categorical(
                "input_mask_ratio", [0.125, 0.15, 0.20, 0.25],
            ),
        )

    d_model = trial.suggest_categorical("d_model", [256, 512])
    return TrainConfig(
        epochs=epochs,
        sample_ratio=sample_ratio,
        patience=patience,
        verbose=True,
        seed=42,
        # Learning dynamics
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        temperature=trial.suggest_float("temperature", 0.5, 5.0),
        grad_cap=trial.suggest_float("grad_cap", 1.0, 10.0),
        input_mask_ratio=trial.suggest_float("input_mask_ratio", 0.15, 0.3),
        # Regularization
        kl_sparse_weight=trial.suggest_float(
            "kl_sparse_weight", 0.0001, 1e-1, log=True,
        ),
        kl_target_rho=trial.suggest_float("kl_target_rho", 0.01, 0.2),
        cov_penalty_weight=trial.suggest_float(
            "cov_penalty_weight", 0.1, 20.0,
        ),
        consistency_weight=trial.suggest_float(
            "consistency_weight", 0.0, 1.0,
        ),
        # Model architecture
        d_model=d_model,
        ffn_dim=trial.suggest_categorical("ffn_dim", [1024, 2048, 4096]),
        num_layers=trial.suggest_int("num_layers", 3, 9),
        num_heads=trial.suggest_categorical("num_heads", [4, 8, 16]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        kc_vocab_size=trial.suggest_categorical(
            "kc_vocab_size", [256, 512, 1024, 2048],
        ),
        recon_pos_embed_dim=trial.suggest_categorical(
            "recon_pos_embed_dim", [32, 64, 128],
        ),
        recon_hidden_dim=trial.suggest_categorical(
            "recon_hidden_dim", [128, 256, 512],
        ),
    )



_SCRIPT_HASH = hashlib.sha256(
    open(os.path.join(os.path.dirname(__file__), "recon_bpd.py"), "rb").read(),
).hexdigest()[:12]


def _config_hash(config: TrainConfig) -> str:
    """Deterministic hash of training config for checkpoint keying.

    Includes a hash of ``recon_bpd.py`` so code changes automatically
    invalidate stale checkpoints.  Excludes ``epochs``, ``verbose``,
    and ``patience`` so that extending the epoch budget or toggling
    verbosity still reuses checkpoints.
    """
    d = dataclasses.asdict(config)
    del d["epochs"]
    del d["verbose"]
    del d["patience"]
    canonical = _SCRIPT_HASH + json.dumps(sorted(d.items()), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def objective(
    trial: optuna.Trial,
    epochs: int,
    sample_ratio: Optional[float],
    patience: Optional[int],
    use_mlflow: bool,
    study_name: str,
    consistency_weight_only: bool = False,
    checkpoint_dir: str = "",
    adhoc_prefix: str = "",
) -> float:
    """Optuna objective: minimize BPD."""
    config = suggest_config(
        trial, epochs, sample_ratio, patience, consistency_weight_only,
    )

    # Checkpoint keyed by config hash — resume if same params seen before.
    checkpoint_path = ""
    existing = None
    if checkpoint_dir:
        config_hash = _config_hash(config)
        checkpoint_path = os.path.join(checkpoint_dir, f"{config_hash}.pt")
        existing = load_checkpoint(checkpoint_path)
        if existing is not None and existing.epoch >= epochs - 1:
            # Already fully trained with these params — skip.
            trial.report(existing.latest_metrics["bpd"], existing.epoch)
            return existing.latest_metrics["bpd"]
        if existing is not None:
            print(
                f"  Resuming trial {trial.number} from "
                f"epoch {existing.epoch + 1} (checkpoint)",
            )

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow  # type: ignore[import-untyped]

        mlflow = _mlflow
        run_prefix = f"{adhoc_prefix} " if adhoc_prefix else ""
        mlflow.start_run(run_name=f"{run_prefix}trial-{trial.number}: {study_name}")
        for field in dataclasses.fields(config):
            mlflow.log_param(field.name, getattr(config, field.name))
        mlflow.log_param("machine", platform.node().split(".")[0] or "unknown")
        mlflow.set_tag("optuna_trial", str(trial.number))
        if existing is not None:
            mlflow.set_tag("resumed_from_epoch", str(existing.epoch))

    try:

        def on_epoch_end(epoch: int, metrics: dict) -> None:
            if mlflow is not None:
                for k, v in metrics.items():
                    mlflow.log_metric(f"bpd/{k}", v, step=epoch)
            trial.report(metrics["bpd"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result, _checkpoint = train(
            config,
            on_epoch_end=on_epoch_end,
            checkpoint=existing,
            checkpoint_path=checkpoint_path or None,
        )
        if mlflow is not None:
            mlflow.log_metric("final_bpd", result.final_bpd)
        return result.final_bpd
    except optuna.TrialPruned:
        if mlflow is not None:
            mlflow.set_tag("optuna_pruned", "true")
        raise
    finally:
        if mlflow is not None:
            mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for recon_bpd",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--epochs-per-trial", type=int, default=30,
        help="Training epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=None,
        help="Dataset sample ratio override (default: device-specific)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop a trial after N epochs without BPD improvement",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL (default: sqlite:///.cache/recon_bpd/optuna.db)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="TPE sampler seed (default: 42)",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--tracking-uri", type=str, default=None,
        help="MLflow tracking URI override",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="kotogram-bpd",
        help="MLflow experiment name (default: kotogram-bpd)",
    )
    parser.add_argument(
        "--consistency-weight-only", action="store_true",
        help="Optimize ONLY consistency_weight",
    )
    parser.add_argument(
        "--adhoc", nargs="?", const="", default=None, metavar="PREFIX",
        help="Run a single trial with default parameters, logging to adhoc experiment. "
             "Optional PREFIX is prepended to the MLflow run name.",
    )
    parser.add_argument(
        "--pruner", type=str, default="hyperband",
        choices=["hyperband", "percentile"],
        help="Pruner algorithm (default: hyperband)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=os.path.join(".cache", "optuna", "checkpoints"),
        help="Directory for per-trial checkpoints (default: .cache/optuna/checkpoints)",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint persistence and reuse",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    if args.adhoc is not None:
        exp_name = "adhoc-kotogram-bpd"
        args.n_trials = 1
    adhoc_prefix = args.adhoc or ""

    suffixes = []
    if args.consistency_weight_only:
        suffixes.append("cw-imr-only")
    if args.sample_ratio is not None:
        suffixes.append(f"{args.sample_ratio * 100:g}%")
    suffixes.append(f"{args.epochs_per_trial}ep")
    study_name = f"{exp_name} ({', '.join(suffixes)})"

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        from train.mlflow_logging import configure_tracking

        configure_tracking(
            tracking_uri=args.tracking_uri,
            experiment_name=exp_name,
        )

    storage = args.storage
    if storage is None and args.adhoc is None:
        db_dir = os.path.join(".cache", "recon_bpd")
        os.makedirs(db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(db_dir, 'optuna.db')}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=args.epochs_per_trial,
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.PercentilePruner(
            percentile=25.0, n_startup_trials=5, n_warmup_steps=5,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.adhoc is None,
    )

    defaults = TrainConfig()

    if args.consistency_weight_only:
        initial_params = [
            {
                 "consistency_weight": defaults.consistency_weight,
                 "input_mask_ratio": defaults.input_mask_ratio,
            },
        ]
    else:
        initial_params = [
            {
                "lr": defaults.lr,
                "temperature": defaults.temperature,
                "grad_cap": defaults.grad_cap,
                "input_mask_ratio": defaults.input_mask_ratio,
                "kl_sparse_weight": defaults.kl_sparse_weight,
                "kl_target_rho": defaults.kl_target_rho,
                "cov_penalty_weight": defaults.cov_penalty_weight,
                "consistency_weight": defaults.consistency_weight,
                "d_model": defaults.d_model,
                "ffn_dim": defaults.ffn_dim,
                "num_heads": defaults.num_heads,
                "dropout": defaults.dropout,
                "kc_vocab_size": defaults.kc_vocab_size,
                "recon_pos_embed_dim": defaults.recon_pos_embed_dim,
                "recon_hidden_dim": defaults.recon_hidden_dim,
            }
        ]

    for params in initial_params:
        study.enqueue_trial(params, skip_if_exists=args.adhoc is None)

    checkpoint_dir = "" if args.no_checkpoint else args.checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")

    study.optimize(
        lambda trial: objective(
            trial, args.epochs_per_trial, args.sample_ratio,
            args.patience, use_mlflow, study_name, args.consistency_weight_only,
            checkpoint_dir, adhoc_prefix,
        ),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print("Best trial:")
    best = study.best_trial
    print(f"  BPD:   {best.value:.4f}")
    print(f"  Trial: {best.number}")
    print("  Params:")
    for key, value in sorted(best.params.items()):
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
