#!/usr/bin/env python3
"""Optuna hyperparameter search for recon_bpd training.

Minimizes BPD (bits per attended token-position) over the full
TrainConfig search space.  Each trial is logged as an independent
MLflow run with per-epoch step metrics (matching the train_style
pattern).  Supports MedianPruner for early stopping of unpromising
trials and optional persistent storage for resumable studies.

Usage:
    python -m scratch.recon_bpd_optuna
    python -m scratch.recon_bpd_optuna --n-trials 100 --epochs-per-trial 50
    python -m scratch.recon_bpd_optuna --storage sqlite:///optuna.db
    python -m scratch.recon_bpd_optuna --no-mlflow
    python3 -m scratch.recon_bpd_optuna --n-trials 1000 --sample-ratio 0.15 --epochs-per-trial 15 --storage sqlite:///.cache/recon_bpd/optuna.db --consistency-weight-only
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
from typing import Optional

import optuna
import torch

from scratch.recon_bpd import TrainConfig, train
from scratch.recon_bpd_checkpoint import EpochContext, load_checkpoint


class GpuMemoryMonitor(threading.Thread):
    """Background thread to sample GPU memory and hardware stats during training."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.reserved_samples = []
        self.utilization_samples = []
        self.temperature_samples = []

        self.phase = "train"  # "train" or "test"
        self.train_peak = 0.0
        self.test_peak = 0.0

        self.stop_event = threading.Event()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def run(self):
        if self.device.type == "cpu":
            return
        while not self.stop_event.is_set():
            allocated = 0.0
            reserved = 0.0

            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                # Sample utilization and temperature
                try:
                    res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=1,
                    )
                    if res.returncode == 0:
                        u, t = map(float, res.stdout.strip().split(","))
                        self.utilization_samples.append(u)
                        self.temperature_samples.append(t)
                except Exception:
                    pass
            elif self.device.type == "mps":
                try:
                    allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                    reserved = allocated  # MPS doesn't have a direct 'reserved' API
                except AttributeError:
                    pass

            if allocated > 0:
                self.samples.append(allocated)
                self.reserved_samples.append(reserved)
                if self.phase == "train":
                    self.train_peak = max(self.train_peak, allocated)
                else:
                    self.test_peak = max(self.test_peak, allocated)

            self.stop_event.wait(self.interval)

    def get_stats_and_reset(self):
        """Return dict of metrics and clear samples."""
        if not self.samples:
            return {}

        med = statistics.median(self.samples)
        peak = max(self.samples)
        res_med = statistics.median(self.reserved_samples)
        frag = (res_med - med) / max(1.0, res_med)

        stats = {
            "median_mb": med,
            "peak_mb": peak,
            "reserved_mb": res_med,
            "fragmentation": frag,
            "train_peak_mb": self.train_peak,
            "test_peak_mb": self.test_peak,
        }

        if self.utilization_samples:
            stats["utilization_pct"] = statistics.median(self.utilization_samples)
        if self.temperature_samples:
            stats["temperature_c"] = statistics.median(self.temperature_samples)

        # Reset per-epoch buffers
        self.samples = []
        self.reserved_samples = []
        self.utilization_samples = []
        self.temperature_samples = []
        self.train_peak = 0.0
        self.test_peak = 0.0

        return stats

    def stop(self):
        self.stop_event.set()


def suggest_config(
    trial: optuna.Trial,
    epochs: int,
    sample_ratio: float,
    sweep: bool = False,
) -> TrainConfig:
    """Build a TrainConfig from Optuna trial suggestions."""
    if sweep:
        config = TrainConfig(
            epochs=epochs,
            sample_ratio=sample_ratio,
            seed=42,
        )
        for key, values in _SWEEP_SEARCH_SPACE.items():
            setattr(config, key, trial.suggest_categorical(key, values))
        return config

    d_model = trial.suggest_categorical("d_model", [256, 512])
    return TrainConfig(
        epochs=epochs,
        sample_ratio=sample_ratio,
        seed=42,
        # Learning dynamics
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        temperature=trial.suggest_float("temperature", 0.5, 5.0),
        grad_cap=trial.suggest_float("grad_cap", 1.0, 10.0),
        input_mask_ratio=trial.suggest_float("input_mask_ratio", 0.1, 0.3),
        # Regularization
        kl_sparse_weight=trial.suggest_float(
            "kl_sparse_weight",
            0.0001,
            1e-1,
            log=True,
        ),
        kl_target_rho=trial.suggest_float("kl_target_rho", 0.01, 0.2),
        cov_penalty_weight=trial.suggest_float(
            "cov_penalty_weight",
            0.1,
            20.0,
        ),
        consistency_weight=trial.suggest_float(
            "consistency_weight",
            0.0,
            1.0,
        ),
        # Model architecture
        d_model=d_model,
        ffn_dim=trial.suggest_categorical("ffn_dim", [1024, 2048, 4096]),
        num_layers=trial.suggest_int("num_layers", 3, 9),
        num_heads=trial.suggest_categorical("num_heads", [4, 8, 16]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        kc_vocab_size=trial.suggest_categorical(
            "kc_vocab_size",
            [256, 512, 1024, 2048],
        ),
        recon_pos_embed_dim=trial.suggest_categorical(
            "recon_pos_embed_dim",
            [32, 64, 128],
        ),
        recon_hidden_dim=trial.suggest_categorical(
            "recon_hidden_dim",
            [128, 256, 512],
        ),
    )


# Pareto frontier parameter space configuration
_SWEEP_SEARCH_SPACE: dict = {
    "kl_target_rho": [0.01, 0.03, 0.06, 0.065],
    "kc_vocab_size": [512, 768, 1024, 1536, 2048],
}

# Default overrides for --adhoc runs.
ADHOC_OVERRIDES: dict = {
    "num_layers": 6,
}

_SWEEP_SPACE_HASH = hashlib.sha256(
    json.dumps(_SWEEP_SEARCH_SPACE, sort_keys=True).encode(),
).hexdigest()[:6]
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
    Excludes ``epochs`` so that extending
    the epoch budget still reuses checkpoints.
    """
    d = dataclasses.asdict(config)
    del d["epochs"]
    canonical = (
        _SCRIPT_HASH + _DATASET_HASH + json.dumps(sorted(d.items()), sort_keys=True)
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
    sample_ratio: float,
    use_mlflow: bool,
    sweep: bool,
    checkpoint_dir: str,
    adhoc_overrides: Optional[dict] = None,
    adhoc_name: str = "",
) -> float:
    """Optuna objective: minimize BPD."""
    config = suggest_config(
        trial,
        epochs,
        sample_ratio,
        sweep,
    )
    if adhoc_overrides:
        for k, v in adhoc_overrides.items():
            setattr(config, k, v)

    # Checkpoint keyed by config hash — resume if same params seen before.
    config_hash = _config_hash(config)
    run_name = config_hash[:8]
    PARAM_ABBREV = {
        "consistency_weight": "cw",
        "input_mask_ratio": "mask",
        "kl_sparse_weight": "kl",
        "kl_target_rho": "rho",
        "cov_penalty_weight": "cov",
        "lr": "lr",
        "num_layers": "L",
        "temperature": "temp",
        "weight_decay": "wd",
    }

    params_to_show = adhoc_overrides or trial.params
    if params_to_show:
        parts = " ".join(
            f"{PARAM_ABBREV.get(k, k)}={v:g}"
            if isinstance(v, float)
            else f"{PARAM_ABBREV.get(k, k)}={v}"
            for k, v in sorted(params_to_show.items())
        )
        run_name = f"{parts} {run_name}"
    if adhoc_name:
        run_name = f"[{adhoc_name}] {run_name}"
    checkpoint_path = ""
    existing = None
    log_path = os.path.join(checkpoint_dir, "debug.log") if checkpoint_dir else ""
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config_hash}.pt")
        existing = load_checkpoint(checkpoint_path)
        if existing is not None:
            history_epochs = [ep for ep, _ in existing.epoch_history]
            with open(log_path, "a") as f:
                f.write(
                    f"LOAD  {config_hash[:8]}  "
                    f"epoch={existing.epoch}  "
                    f"history={history_epochs}  "
                    f"target={epochs}  "
                    f"run_name={run_name}\n",
                )
        if existing is not None and existing.epoch >= epochs - 1:
            target_epoch = epochs - 1
            era_bpd = existing.latest_metrics["bpd"]
            for ep, metrics in existing.epoch_history:
                if ep == target_epoch:
                    era_bpd = metrics["bpd"]
                    break
            with open(log_path, "a") as f:
                f.write(
                    f"CACHE {config_hash[:8]}  "
                    f"era_bpd={era_bpd:.4f}  "
                    f"target_epoch={target_epoch}\n",
                )
            if use_mlflow:
                import mlflow as _mlflow  # type: ignore[import-untyped]

                run_id = _find_mlflow_run(run_name)
                _mlflow.start_run(run_id=run_id, run_name=run_name)
                if run_id is None:
                    for field in dataclasses.fields(config):
                        if field.name not in ("epochs",):
                            _mlflow.log_param(field.name, getattr(config, field.name))
                _mlflow.set_tag("cached", "true")
                if run_id is None:
                    for ep, metrics in existing.epoch_history:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                _mlflow.log_metric(f"bpd/{k}", v, step=ep)
                        k_toks = (
                            int(metrics.get("cumulative_tokens_trained", ep * 1000))
                            // 1000
                        )
                        _mlflow.log_metric(
                            "inv/bpd", metrics.get("bpd", 0.0), step=k_toks
                        )
                        if "pooled_std" in metrics:
                            _mlflow.log_metric(
                                "inv/pooled_std", metrics["pooled_std"], step=k_toks
                            )
                _mlflow.log_metric("final_bpd", era_bpd)
                _mlflow.log_metric(f"final_bpd_{epochs}ep", era_bpd)
                _mlflow.end_run()
            trial.report(era_bpd, target_epoch)
            return era_bpd
        if existing is not None:
            with open(log_path, "a") as f:
                f.write(
                    f"RESUME {config_hash[:8]}  "
                    f"from_epoch={existing.epoch + 1}  "
                    f"target={epochs}\n",
                )
            print(
                f"  Resuming {run_name} from epoch {existing.epoch + 1} (checkpoint)",
            )

    # Start background GPU monitor
    memory_monitor = GpuMemoryMonitor()
    memory_monitor.start()

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow  # type: ignore[import-untyped]

        mlflow = _mlflow
        run_id = _find_mlflow_run(run_name)
        mlflow.start_run(run_id=run_id, run_name=run_name)

        # Capture git commit for reproducibility (safe against git missing/errors)
        git_log = subprocess.run(
            ["git", "log", "-1", "--format=%h %s"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if git_log.returncode == 0 and git_log.stdout.strip():
            mlflow.set_tag("git_commit", git_log.stdout.strip())

        if run_id is None:
            for field in dataclasses.fields(config):
                if field.name not in ("epochs",):
                    mlflow.log_param(field.name, getattr(config, field.name))
            mlflow.log_param("machine", platform.node().split(".")[0] or "unknown")
        mlflow.set_tag("optuna_trial", str(trial.number))
        if existing is not None:
            mlflow.set_tag("resumed_from_epoch", str(existing.epoch))
            # Only backfill epoch history for NEW runs; existing runs already
            # have these metrics from the previous session.
            if run_id is None:
                for ep, metrics in existing.epoch_history:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"bpd/{k}", v, step=ep)
                    k_toks = (
                        int(metrics.get("cumulative_tokens_trained", ep * 1000)) // 1000
                    )
                    mlflow.log_metric("inv/bpd", metrics.get("bpd", 0.0), step=k_toks)
                    if "pooled_std" in metrics:
                        mlflow.log_metric(
                            "inv/pooled_std", metrics["pooled_std"], step=k_toks
                        )

    try:

        def on_epoch_start(epoch: int, metrics: dict, ctx: EpochContext) -> None:
            print(f"\n{run_name}  epoch {epoch + 1}/{epochs}")
            # Run reconstruction test at the start of the epoch for baseline/pre-epoch eval
            from scratch.recon_bpd_test import run_reconstruction_test

            memory_monitor.phase = "test"
            run_reconstruction_test(ctx, epoch, metrics)
            memory_monitor.phase = "train"

        def on_epoch_end(epoch: int, metrics: dict, ctx: EpochContext) -> None:
            consist_str = (
                f"consistency={metrics['consistency']:.4f}  "
                f"mask-agree={metrics['mask-agree']:.3f}  "
                if metrics.get("consistency", 0) > 0
                else ""
            )
            recon_str = ""
            if metrics.get("recon_test_total", 0) > 0:
                strict = metrics.get("recon_test_pass_strict", 0)
                passed = metrics["recon_test_pass"]
                total = metrics["recon_test_total"]
                recon_str = f"recon={strict:.0f}/{passed:.0f}/{total:.0f}  "
            # Sample GPU memory for the epoch
            gpu_stats = memory_monitor.get_stats_and_reset()
            gpu_str = ""
            if gpu_stats:
                med_mb = gpu_stats["median_mb"]
                peak_mb = gpu_stats["peak_mb"]
                frag = gpu_stats["fragmentation"]
                gpu_str = f"gpu={med_mb:.0f}/{peak_mb:.0f}MB (frag={frag:.0%})  "

            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"bpd={metrics['bpd']:.4f}  "
                f"To-1={metrics['To-1']:.1f}%  "
                f"cos={metrics['cos']:.3f}  "
                f"sharp={metrics['sharp']:.3f}  "
                f"s1={metrics['s1']:.0%} s0={metrics['s0']:.0%} "
                f"fuzzy={metrics['fuzzy']:.0%}  "
                f"loss={metrics['loss']:.4f}  "
                f"sparsity={metrics['sparsity']:.4f}  "
                f"mdl={metrics.get('mdl', 0):.4f}  "
                f"rank={metrics.get('rank', 0):.4f}  "
                f"orthogonality={metrics['orthogonality']:.4f}  "
                f"{consist_str}"
                f"{recon_str}"
                f"{gpu_str}"
                f"lr={metrics['lr']:.2e}  "
                f"{metrics['el_per_sec']:.1f} el/s  "
                f"{metrics['samples']} samples  "
                f"{metrics['epoch_secs']:.1f}s"
            )
            if mlflow is not None:
                for k, v in gpu_stats.items():
                    mlflow.log_metric(f"gpu/{k}", v, step=epoch)

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"bpd/{k}", v, step=epoch)

                # Log invariance diagnostic metrics against tokens trained (in thousands)
                k_toks = (
                    int(metrics.get("cumulative_tokens_trained", epoch * 1000)) // 1000
                )
                mlflow.log_metric("inv/bpd", metrics.get("bpd", 0.0), step=k_toks)
                if "pooled_std" in metrics:
                    mlflow.log_metric(
                        "inv/pooled_std", metrics["pooled_std"], step=k_toks
                    )

                # Upload all registered artifacts
                for artifact_path in ctx.artifact_paths:
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, "recon_test")

                # Sample GPU memory for the epoch
                med_mb, peak_mb = memory_monitor.get_stats_and_reset()
                if peak_mb > 0:
                    mlflow.log_metric("gpu/median_mb", med_mb, step=epoch)
                    mlflow.log_metric("gpu/peak_mb", peak_mb, step=epoch)

            trial.report(metrics["bpd"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result, _checkpoint = train(
            config,
            on_epoch_start=on_epoch_start,
            on_epoch_end=on_epoch_end,
            checkpoint_path=checkpoint_path,
            checkpoint=existing,
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
        memory_monitor.stop()
        if mlflow is not None:
            mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for recon_bpd",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--epochs-per-trial",
        "--epochs",
        dest="epochs_per_trial",
        type=int,
        default=30,
        help="Training epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=100.0,
        help="Dataset sample percentage (default: 100.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="TPE sampler seed (default: 42)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI override",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="kotogram-bpd",
        help="MLflow experiment name (default: kotogram-bpd)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a discrete sweep over core invariant hyperparameters",
    )
    parser.add_argument(
        "--adhoc",
        nargs="?",
        const="adhoc",
        default=None,
        metavar="PREFIX",
        help="Run a single trial with default parameters, logging to adhoc experiment. "
        "Optional PREFIX is prepended to the MLflow run name.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override the number of layers",
    )
    parser.add_argument(
        "--layer-dropout",
        type=float,
        default=None,
        metavar="PERCENT",
        help="Override layer drop probability as a percentage (e.g. 50 = drop 50%% of layers)",
    )
    parser.add_argument(
        "--mdl",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Enable MDL bits-back sparsity with given weight (e.g. 0.1)",
    )
    parser.add_argument(
        "--rank",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Enable pairwise ranking margin with given weight (e.g. 0.5)",
    )
    parser.add_argument(
        "--no-kl",
        action="store_true",
        help="Disable KL rho-based sparsity (set kl_sparse_weight=0)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="hyperband",
        choices=["hyperband", "percentile", "none"],
        help="Pruner algorithm (default: hyperband)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(".cache", "optuna", "checkpoints"),
        help="Directory for per-trial checkpoints (default: .cache/optuna/checkpoints)",
    )

    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=20,
        help="Stop if no improvement after this many completed trials (default: 20)",
    )
    parser.add_argument(
        "--no-progressive",
        action="store_true",
        help="Disable progressive epoch extension (default: progressive is on)",
    )
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=5,
        help="Epochs to add each progressive round (default: 5)",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    if args.adhoc is not None:
        exp_name = "adhoc-kotogram-bpd"
        args.n_trials = 1
        # Auto-derive trial name from last git commit touching recon_bpd.py
        if args.adhoc == "adhoc":
            git_subject = subprocess.run(
                ["git", "log", "-1", "--format=%s", "--", "scratch/recon_bpd.py"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            args.adhoc = (
                git_subject.stdout.strip()
                if git_subject.returncode == 0 and git_subject.stdout.strip()
                else "adhoc"
            )

    adhoc_overrides: dict = {}
    if args.adhoc is not None:
        adhoc_overrides = dict(ADHOC_OVERRIDES)
    if args.layers is not None:
        adhoc_overrides["num_layers"] = args.layers
    if args.layer_dropout is not None:
        adhoc_overrides["layer_drop_prob"] = args.layer_dropout / 100.0
    if args.mdl is not None:
        adhoc_overrides["mdl_weight"] = args.mdl
    if args.rank is not None:
        adhoc_overrides["rank_margin_weight"] = args.rank
    if args.no_kl:
        adhoc_overrides["kl_sparse_weight"] = 0.0

    if adhoc_overrides:
        print("Overrides:")
        for k, v in sorted(adhoc_overrides.items()):
            print(f"  {k}: {v}")

    suffixes = []
    if args.sweep:
        suffixes.append("sweep")
        suffixes.append(_SWEEP_SPACE_HASH)
    if args.percent != 100.0:
        suffixes.append(f"{args.percent:g}%")

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        from train.mlflow_logging import configure_tracking

        configure_tracking(
            tracking_uri=args.tracking_uri,
            experiment_name=exp_name,
        )

    if args.pruner != "hyperband":
        if args.pruner == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=5,
                n_warmup_steps=5,
            )

    defaults = TrainConfig()

    if args.sweep:
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
            for k, v in adhoc_overrides.items():
                if k == "kl_sparse_weight" and v == 0.0:
                    continue
                params[k] = v

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")

    epochs = args.epochs_per_trial

    study_name = f"{exp_name} ({', '.join(suffixes)})"
    study_pruner = (
        optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=1000,
            reduction_factor=4,
        )
        if args.pruner == "hyperband"
        else pruner
    )

    progressive_round = 0
    while True:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=study_pruner,
        )

        for params in initial_params:
            study.enqueue_trial(params, skip_if_exists=True)

        if progressive_round > 0:
            print(f"\n{'=' * 60}")
            print(
                f"Progressive round {progressive_round}: extending to {epochs} epochs",
            )
            print("=" * 60)

        def _convergence_callback(
            study: optuna.Study,
            trial: optuna.trial.FrozenTrial,
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
                trial,
                epochs,
                args.percent / 100.0,
                use_mlflow,
                args.sweep,
                checkpoint_dir,
                adhoc_overrides or None,
                args.adhoc or "",
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
