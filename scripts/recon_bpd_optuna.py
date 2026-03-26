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
import os
import platform
from typing import Optional

import optuna

from scripts.recon_bpd import OriginalTrainConfig, TrainConfig, train


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
            consistency_weight=trial.suggest_float(
                "consistency_weight", 0.0, 0.2,
            ),
            input_mask_ratio=trial.suggest_float(
                "input_mask_ratio", 0.15, 0.3,
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


BASELINE_MARGIN = 0.05


def _get_baseline_curve(study: optuna.Study) -> dict:
    """BPD-per-epoch from the first completed trial (the enqueued defaults)."""
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            return dict(t.intermediate_values)
    return {}


def objective(
    trial: optuna.Trial,
    epochs: int,
    sample_ratio: Optional[float],
    patience: Optional[int],
    use_mlflow: bool,
    consistency_weight_only: bool = False,
) -> float:
    """Optuna objective: minimize BPD."""
    config = suggest_config(
        trial, epochs, sample_ratio, patience, consistency_weight_only,
    )
    baseline = _get_baseline_curve(trial.study)

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow  # type: ignore[import-untyped]

        mlflow = _mlflow
        mlflow.start_run(run_name=f"trial-{trial.number}")
        for field in dataclasses.fields(config):
            mlflow.log_param(field.name, getattr(config, field.name))
        mlflow.log_param("machine", platform.node().split(".")[0] or "unknown")
        mlflow.set_tag("optuna_trial", str(trial.number))

    try:

        def on_epoch_end(epoch: int, metrics: dict) -> None:
            if mlflow is not None:
                for k, v in metrics.items():
                    mlflow.log_metric(f"bpd/{k}", v, step=epoch)
            trial.report(metrics["bpd"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if baseline and epoch in baseline and epoch > 3:
                if metrics["bpd"] > baseline[epoch] * (1 + BASELINE_MARGIN):
                    raise optuna.TrialPruned()

        result = train(config, on_epoch_end=on_epoch_end)
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
        "--experiment-name", type=str, default="kotogram-bpd#2",
        help="MLflow experiment name (default: kotogram-bpd#2)",
    )
    parser.add_argument(
        "--consistency-weight-only", action="store_true",
        help="Optimize ONLY consistency_weight and input_mask_ratio",
    )
    args = parser.parse_args()

    name = args.experiment_name
    suffixes = []
    if args.consistency_weight_only:
        suffixes.append("cw-imr-only")
    if args.sample_ratio is not None:
        suffixes.append(f"{args.sample_ratio * 100:g}%")
    suffixes.append(f"{args.epochs_per_trial}ep")
    name += f" ({', '.join(suffixes)})"

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        from train.mlflow_logging import configure_tracking

        configure_tracking(
            tracking_uri=args.tracking_uri,
            experiment_name=name,
        )

    storage = args.storage
    if storage is None:
        db_dir = os.path.join(".cache", "recon_bpd")
        os.makedirs(db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(db_dir, 'optuna.db')}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.PercentilePruner(
        percentile=25.0, n_startup_trials=5, n_warmup_steps=5,
    )

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    defaults = TrainConfig()
    original = OriginalTrainConfig()

    if args.consistency_weight_only:
        initial_params = [
            {"consistency_weight": original.consistency_weight, "input_mask_ratio": original.input_mask_ratio},
            {"consistency_weight": defaults.consistency_weight, "input_mask_ratio": defaults.input_mask_ratio},
        ]
    else:
        initial_params = [
            {
                "lr": original.lr,
                "temperature": original.temperature,
                "grad_cap": original.grad_cap,
                "input_mask_ratio": original.input_mask_ratio,
                "kl_sparse_weight": original.kl_sparse_weight,
                "kl_target_rho": original.kl_target_rho,
                "cov_penalty_weight": original.cov_penalty_weight,
                "consistency_weight": original.consistency_weight,
                "d_model": original.d_model,
                "ffn_dim": original.ffn_dim,
                "num_layers": original.num_layers,
                "num_heads": original.num_heads,
                "dropout": original.dropout,
                "kc_vocab_size": original.kc_vocab_size,
                "recon_pos_embed_dim": original.recon_pos_embed_dim,
                "recon_hidden_dim": original.recon_hidden_dim,
            },
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
                "num_layers": defaults.num_layers,
                "num_heads": defaults.num_heads,
                "dropout": defaults.dropout,
                "kc_vocab_size": defaults.kc_vocab_size,
                "recon_pos_embed_dim": defaults.recon_pos_embed_dim,
                "recon_hidden_dim": defaults.recon_hidden_dim,
            },
        ]

    for params in initial_params:
        study.enqueue_trial(params, skip_if_exists=True)

    study.optimize(
        lambda trial: objective(
            trial, args.epochs_per_trial, args.sample_ratio,
            args.patience, use_mlflow, args.consistency_weight_only,
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
