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
import subprocess
from typing import Optional

import optuna

from scratch.recon_bpd import TrainConfig, train
from scratch.recon_bpd_checkpoint import EpochContext, load_checkpoint


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
            if key == "vicreg_enabled":
                # Expand the single vicreg on/off toggle into the two weight fields.
                on = trial.suggest_categorical(key, values)
                if on:
                    config.vicreg_var_weight = _VICREG_ON_WEIGHTS["var"]
                    config.vicreg_cov_weight = _VICREG_ON_WEIGHTS["cov"]
                else:
                    config.vicreg_var_weight = 0.0
                    config.vicreg_cov_weight = 0.0
            else:
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
        mdl_weight=trial.suggest_float(
            "mdl_weight",
            0.001,
            1.0,
            log=True,
        ),
        rank_margin_weight=trial.suggest_float("rank_margin_weight", 0.0, 2.0),
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


# VICReg default weights (used when vicreg_enabled = True).
_VICREG_ON_WEIGHTS = {"var": 11.0, "cov": 5.0}

# Feature-ablation sweep space.
#
# Goal: measure the independent contribution of each new training feature
# introduced on the collapse-firewall branch, and produce a "pre-branch
# baseline" point by disabling all of them simultaneously.
#
# Design rules:
#   - Each feature has exactly one primary on/off knob listed here.
#   - Parameters that only matter when a feature is enabled are NOT listed;
#     they keep their TrainConfig default and can be tuned in a separate
#     focused sweep if the feature proves useful.
#   - The full-off combination reproduces pre-branch training behaviour.
#
# Feature inventory (from collapse-firewall.md):
#   F1a. Stop-gradient consistency    → consistency_stop_gradient
#   F1b. VICReg                       → vicreg_var_weight + vicreg_cov_weight
#   F1c. LayerDrop (stochastic depth) → layer_drop_prob
#   F2.  Stochastic rescue gate       → semantic_rescue_gate
#   F3.  (test suite — no training impact, no flag needed)
#   F4.  (checkpoint infra — no training impact, no flag needed)
#   F5.  Temperature annealing        → temperature_start_multiplier (1.0 = off)
#   F5b. KL warmup                    → temperature_anneal_epochs (0.0 = off)
#   F6.  Bidirectional decoder        → recon_bidirectional_pos
#   F7.  Non-content masking          → non_content_mask_ratio (0.0 = off)
#
# Pre-branch baseline: all features disabled simultaneously. This is
# automatically included as one of the sweep combinations.
_SWEEP_SEARCH_SPACE: dict = {
    # ── F1a: Stop-gradient consistency ───────────────────────────────
    # False = plain cosine similarity (pre-branch)
    # True  = BYOL/SimSiam symmetrized stop-gradient (branch default)
    # Only has effect when consistency_weight > 0.
    "consistency_stop_gradient": [False, True],

    # ── F1b: VICReg ───────────────────────────────────────────────────
    # False = no VICReg (pre-branch)
    # True  = variance + covariance regularization on pooled encoder output
    #         (var_weight=11.0, cov_weight=5.0 — branch default)
    # Expanded to vicreg_var_weight / vicreg_cov_weight in suggest_config.
    "vicreg_enabled": [False, True],

    # ── F1c: LayerDrop ────────────────────────────────────────────────
    # 0.0  = no layer dropping (pre-branch)
    # 0.5  = aggressive stochastic depth (branch default)
    "layer_drop_prob": [0.0, 0.5],

    # ── F2: Stochastic rescue gate ────────────────────────────────────
    # False = fully deterministic cos_sim < threshold gate (pre-branch)
    # True  = deterministic-hard + stochastic-rescue (branch default)
    # Only has effect when semantic_gating_threshold > 0.
    "semantic_rescue_gate": [False, True],

    # ── F5: Temperature annealing ─────────────────────────────────────
    # 1.0  = no warmup, start at target temperature immediately (pre-branch)
    # 3.0  = start at 3x target temperature and anneal (branch default)
    "temperature_start_multiplier": [1.0, 3.0],

    # ── F5b: KL warmup ───────────────────────────────────────────────
    # 0.0  = full KL weight from epoch 0 (pre-branch)
    # 30.0 = quadratic KL weight ramp over 30 effective epochs (branch default)
    "temperature_anneal_epochs": [0.0, 30.0],

    # ── F6: Bidirectional decoder ─────────────────────────────────────
    # False = end-relative only (pre-branch)
    # True  = end-relative + start-relative pos embeddings (branch default)
    "recon_bidirectional_pos": [False, True],

    # ── F7: Non-content masking ───────────────────────────────────────
    # 0.0  = no dropout, raw surface tokens (pre-branch)
    # 0.5  = 50% random dropout of non-content tokens per batch (branch default)
    "non_content_mask_ratio": [0.0, 0.5],
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
        "mdl_weight": "mdl",
        "rank_margin_weight": "rank",
        "cov_penalty_weight": "cov",
        "lr": "lr",
        "num_layers": "L",
        "temperature": "temp",
        "weight_decay": "wd",
    }

    params_to_show = adhoc_overrides or trial.params
    if params_to_show:
        parts = " ".join(
            f"{PARAM_ABBREV.get(k, k)}={v:g}" if isinstance(v, float) else f"{PARAM_ABBREV.get(k, k)}={v}"
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
                        k_toks = int(metrics.get("cumulative_tokens_trained", ep * 1000)) // 1000
                        _mlflow.log_metric("inv/bpd", metrics.get("bpd", 0.0), step=k_toks)
                        if "pooled_std" in metrics:
                            _mlflow.log_metric("inv/pooled_std", metrics["pooled_std"], step=k_toks)
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
                    k_toks = int(metrics.get("cumulative_tokens_trained", ep * 1000)) // 1000
                    mlflow.log_metric("inv/bpd", metrics.get("bpd", 0.0), step=k_toks)
                    if "pooled_std" in metrics:
                        mlflow.log_metric("inv/pooled_std", metrics["pooled_std"], step=k_toks)

    try:

        def on_epoch_start(epoch: int) -> None:
            print(f"\n{run_name}  epoch {epoch + 1}/{epochs}")

        def on_epoch_end(epoch: int, metrics: dict, ctx: EpochContext) -> None:
            # ── Reconstruction spot-check (observability, not training) ──
            from scratch.recon_bpd_test import run_reconstruction_test

            run_reconstruction_test(ctx, epoch, metrics)

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
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"bpd={metrics['bpd']:.4f}  "
                f"To-1={metrics['To-1']:.1f}%  "
                f"cos={metrics['cos']:.3f}  "
                f"sharp={metrics['sharp']:.3f}  "
                f"s1={metrics['s1']:.0%} s0={metrics['s0']:.0%} "
                f"fuzzy={metrics['fuzzy']:.0%}  "
                f"loss={metrics['loss']:.4f}  "
                f"mdl={metrics['mdl']:.4f}  "
                f"rank={metrics['rank']:.4f}  "
                f"orthogonality={metrics['orthogonality']:.4f}  "
                f"{consist_str}"
                f"{recon_str}"
                f"lr={metrics['lr']:.2e}  "
                f"{metrics['el_per_sec']:.1f} el/s  "
                f"{metrics['samples']} samples  "
                f"{metrics['epoch_secs']:.1f}s"
            )
            if mlflow is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"bpd/{k}", v, step=epoch)
                
                # Log invariance diagnostic metrics against tokens trained (in thousands)
                k_toks = int(metrics.get("cumulative_tokens_trained", epoch * 1000)) // 1000
                mlflow.log_metric("inv/bpd", metrics.get("bpd", 0.0), step=k_toks)
                if "pooled_std" in metrics:
                    mlflow.log_metric("inv/pooled_std", metrics["pooled_std"], step=k_toks)

                # Upload all registered artifacts
                for artifact_path in ctx.artifact_paths:
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, "recon_test")
                
            trial.report(metrics["bpd"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result, _checkpoint = train(
            config,
            on_epoch_start=on_epoch_start,
            on_epoch_end=on_epoch_end,
            checkpoint_path=checkpoint_path,
            checkpoint=existing,
            run_name=run_name,
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
                capture_output=True, text=True, check=False, timeout=2,
            )
            args.adhoc = git_subject.stdout.strip() if git_subject.returncode == 0 and git_subject.stdout.strip() else "adhoc"

    adhoc_overrides: dict = {}
    if args.adhoc is not None:
        adhoc_overrides = dict(ADHOC_OVERRIDES)
    if args.layers is not None:
        adhoc_overrides["num_layers"] = args.layers

    if adhoc_overrides:
        print("Overrides:")
        for k, v in sorted(adhoc_overrides.items()):
            print(f"  {k}: {v}")

    suffixes = []
    if args.sweep:
        suffixes.append("sweep")
        suffixes.append(_SWEEP_SPACE_HASH)
        # In sweep mode every config must run to completion — convergence
        # stopping mid-round would skip configs and invalidate the ablation.
        args.convergence_patience = args.n_trials
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
                "mdl_weight": defaults.mdl_weight,
                "rank_margin_weight": defaults.rank_margin_weight,
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

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")

    epochs = args.epochs_per_trial
    
    study_name = f"{exp_name} ({', '.join(suffixes)})"
    if args.sweep and args.pruner == "hyperband":
        # Sweep/ablation mode: never prune.
        #
        # Every config in the ablation grid must run to completion so results
        # are directly comparable. Pruning based on early BPD is unreliable
        # (epoch-1 BPD is noisy) and would kill the pre-branch baseline configs
        # that intentionally look worse early. Use --pruner hyperband to override.
        study_pruner = optuna.pruners.NopPruner()
    elif args.pruner == "hyperband":
        # Non-sweep TPE mode: use Hyperband, but with max_resource matched to
        # the actual epoch budget so brackets are correctly sized.
        # max_resource=1000 caused Hyperband to treat every run as <0.1%
        # complete, assigning everything to the 1-epoch explore bracket.
        study_pruner = optuna.pruners.HyperbandPruner(
            min_resource=max(1, epochs // 10),  # first rung at ~10% of budget
            max_resource=epochs,
            reduction_factor=3,  # keep top 1/3 at each rung
        )
    else:
        study_pruner = pruner

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
