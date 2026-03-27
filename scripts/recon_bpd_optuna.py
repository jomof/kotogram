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
                "consistency_weight", _CW_SEARCH_SPACE["consistency_weight"],
            ),
            input_mask_ratio=trial.suggest_categorical(
                "input_mask_ratio", _CW_SEARCH_SPACE["input_mask_ratio"],
            ),
            num_layers=trial.suggest_categorical(
                "num_layers", _CW_SEARCH_SPACE["num_layers"],
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
        input_mask_ratio=trial.suggest_float("input_mask_ratio", 0.1, 0.3),
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


# Consistency-weight-only search space — changing this auto-creates a new study.
_CW_SEARCH_SPACE: dict = {
    "consistency_weight": [0.0, 0.0000001, 0.0000003, 0.000001, 0.000003],
    "input_mask_ratio": [0.15, 0.175],
    "num_layers": [2],
}

_CW_SPACE_HASH = hashlib.sha256(
    json.dumps(_CW_SEARCH_SPACE, sort_keys=True).encode(),
).hexdigest()[:6]

# Default overrides for --adhoc runs.
ADHOC_OVERRIDES: dict = {
    "input_mask_ratio": 0.15,
    "consistency_weight": 1e-06,
}

_SCRIPT_HASH = hashlib.sha256(
    open(os.path.join(os.path.dirname(__file__), "recon_bpd.py"), "rb").read(),
).hexdigest()[:12]


def _dataset_fingerprint() -> str:
    """Content hash of cached dataset files that affect training data."""
    from train import paths as train_paths

    cache_dir = train_paths.get_style_dataset_cache_dir()
    h = hashlib.sha256()
    for name in ("labels.bin_gram", "sentences.txt"):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            h.update(open(path, "rb").read())
    return h.hexdigest()[:12]


_DATASET_HASH = _dataset_fingerprint()


def _config_hash(config: TrainConfig) -> str:
    """Deterministic hash of training config for checkpoint keying.

    Includes hashes of ``recon_bpd.py`` source and dataset cache files
    so code or data changes automatically invalidate stale checkpoints.
    Excludes ``epochs``, ``verbose``, and ``patience`` so that extending
    the epoch budget or toggling verbosity still reuses checkpoints.
    """
    d = dataclasses.asdict(config)
    del d["epochs"]
    del d["verbose"]
    del d["patience"]
    canonical = (
        _SCRIPT_HASH
        + _DATASET_HASH
        + json.dumps(sorted(d.items()), sort_keys=True)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _find_mlflow_run(run_name: str) -> Optional[str]:
    """Find an existing MLflow run by name in the active experiment.

    Returns the run_id if found, or None to create a new run.
    """
    import mlflow as _mlflow  # type: ignore[import-untyped]

    client = _mlflow.tracking.MlflowClient()
    experiment_id = _mlflow.tracking.fluent._get_experiment_id()  # type: ignore[attr-defined]
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"',
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    return None


def objective(
    trial: optuna.Trial,
    epochs: int,
    sample_ratio: Optional[float],
    patience: Optional[int],
    use_mlflow: bool,
    study_name: str,
    consistency_weight_only: bool = False,
    checkpoint_dir: str = "",
    adhoc_overrides: Optional[dict] = None,
) -> float:
    """Optuna objective: minimize BPD."""
    config = suggest_config(
        trial, epochs, sample_ratio, patience, consistency_weight_only,
    )

    # Checkpoint keyed by config hash — resume if same params seen before.
    config_hash = _config_hash(config)
    run_name = config_hash[:8]
    params_to_show = adhoc_overrides or trial.params
    if params_to_show:
        parts = " ".join(
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(params_to_show.items())
        )
        run_name = f"{parts} {run_name}"
    checkpoint_path = ""
    existing = None
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config_hash}.pt")
        existing = load_checkpoint(checkpoint_path)
        if existing is not None and existing.epoch >= epochs - 1:
            # Already fully trained with these params — log and skip.
            # Use BPD from the target epoch, not the latest (which may
            # be from a deeper progressive round).
            target_epoch = epochs - 1
            era_bpd = existing.latest_metrics["bpd"]
            for ep, metrics in existing.epoch_history:
                if ep == target_epoch:
                    era_bpd = metrics["bpd"]
                    break
            if use_mlflow:
                import mlflow as _mlflow  # type: ignore[import-untyped]

                run_id = _find_mlflow_run(run_name)
                _mlflow.start_run(run_id=run_id, run_name=run_name)
                if run_id is None:
                    for field in dataclasses.fields(config):
                        if field.name not in ("epochs", "verbose", "patience"):
                            _mlflow.log_param(field.name, getattr(config, field.name))
                _mlflow.set_tag("cached", "true")
                for ep, metrics in existing.epoch_history:
                    for k, v in metrics.items():
                        _mlflow.log_metric(f"bpd/{k}", v, step=ep)
                _mlflow.log_metric("final_bpd", era_bpd)
                _mlflow.log_metric(f"final_bpd_{epochs}ep", era_bpd)
                _mlflow.end_run()
            trial.report(era_bpd, target_epoch)
            return era_bpd
        if existing is not None:
            print(
                f"  Resuming {run_name} from "
                f"epoch {existing.epoch + 1} (checkpoint)",
            )

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow  # type: ignore[import-untyped]

        mlflow = _mlflow
        run_id = _find_mlflow_run(run_name)
        mlflow.start_run(run_id=run_id, run_name=run_name)
        if run_id is None:
            for field in dataclasses.fields(config):
                if field.name not in ("epochs", "verbose", "patience"):
                    mlflow.log_param(field.name, getattr(config, field.name))
            mlflow.log_param("machine", platform.node().split(".")[0] or "unknown")
        mlflow.set_tag("optuna_trial", str(trial.number))
        if existing is not None:
            mlflow.set_tag("resumed_from_epoch", str(existing.epoch))
            for ep, metrics in existing.epoch_history:
                for k, v in metrics.items():
                    mlflow.log_metric(f"bpd/{k}", v, step=ep)

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
            mlflow.log_metric(f"final_bpd_{epochs}ep", result.final_bpd)
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
    parser.add_argument(
        "--override", nargs=2, action="append", metavar=("KEY", "VALUE"),
        help="Override a TrainConfig field for adhoc runs, e.g. --override lr 1e-3",
    )
    parser.add_argument(
        "--convergence-patience", type=int, default=20,
        help="Stop if no improvement after this many completed trials (default: 20)",
    )
    parser.add_argument(
        "--no-progressive", action="store_true",
        help="Disable progressive epoch extension (default: progressive is on)",
    )
    parser.add_argument(
        "--epoch-step", type=int, default=5,
        help="Epochs to add each progressive round (default: 5)",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    if args.adhoc is not None:
        exp_name = "adhoc-kotogram-bpd"
        args.n_trials = 1

    # Build adhoc overrides: start from defaults, layer on CLI --override.
    adhoc_overrides: dict = {}
    if args.adhoc is not None:
        adhoc_overrides = dict(ADHOC_OVERRIDES)
        for key, value in (args.override or []):
            adhoc_overrides[key] = type(ADHOC_OVERRIDES.get(key, 0.0))(value)
        print("Adhoc overrides:")
        for k, v in sorted(adhoc_overrides.items()):
            print(f"  {k}: {v}")

    suffixes = []
    if args.consistency_weight_only:
        suffixes.append(f"cw-imr-only")
        suffixes.append(_CW_SPACE_HASH)
    if args.sample_ratio is not None:
        suffixes.append(f"{args.sample_ratio * 100:g}%")
    study_name = f"{exp_name} ({', '.join(suffixes)})" if suffixes else exp_name

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
    if args.pruner != "hyperband":
        pruner = optuna.pruners.PercentilePruner(
            percentile=25.0, n_startup_trials=5, n_warmup_steps=5,
        )

    defaults = TrainConfig()

    if args.consistency_weight_only:
        initial_params: list = []
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
        if adhoc_overrides:
            params.update(adhoc_overrides)

    checkpoint_dir = "" if args.no_checkpoint else args.checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")

    epochs = args.epochs_per_trial
    progressive_round = 0
    while True:
        # Each era gets its own study with epoch count in the name.
        suffixes_round = list(suffixes)
        suffixes_round.append(f"{epochs}ep")
        study_name_round = f"{exp_name} ({', '.join(suffixes_round)})"

        pruner_round = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=epochs,
            reduction_factor=4,
        ) if args.pruner == "hyperband" else pruner

        study = optuna.create_study(
            study_name=study_name_round,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            pruner=pruner_round,
            load_if_exists=True,
        )

        for params in initial_params:
            study.enqueue_trial(params, skip_if_exists=True)

        if progressive_round > 0:
            print(f"\n{'=' * 60}")
            print(
                f"Progressive round {progressive_round}: "
                f"extending to {epochs} epochs",
            )
            print("=" * 60)

        def _convergence_callback(
            study: optuna.Study, trial: optuna.trial.FrozenTrial,
        ) -> None:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            best_number = study.best_trial.number
            if trial.number - best_number >= args.convergence_patience:
                print(
                    f"\nConverged: no improvement for "
                    f"{args.convergence_patience} completed trials "
                    f"(best was trial {best_number})",
                )
                study.stop()

        study.optimize(
            lambda trial: objective(
                trial, epochs, args.sample_ratio,
                args.patience, use_mlflow, study_name_round,
                args.consistency_weight_only,
                checkpoint_dir, adhoc_overrides or None,
            ),
            n_trials=args.n_trials,
            callbacks=[_convergence_callback],
        )

        print(f"\n{'=' * 60}")
        print(f"Best trial ({epochs} epochs):")
        best = study.best_trial
        print(f"  BPD:   {best.value:.4f}")
        print(f"  Trial: {best.number}")
        print("  Params:")
        for key, value in sorted(best.params.items()):
            print(f"    {key}: {value}")

        if args.no_progressive:
            break

        epochs += args.epoch_step
        progressive_round += 1


if __name__ == "__main__":
    main()
