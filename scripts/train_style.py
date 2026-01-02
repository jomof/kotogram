"""Supervised style classifier for Japanese sentences using Kotogram representations.

This script orchestrates the training pipeline, including argument parsing,
data loading, and calling the trainers from the kotogram.train package.
"""

import dataclasses
import glob
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Union, cast

import torch

from kotogram import locations
from kotogram.model import StyleClassifier

# pylint: disable=ungrouped-imports
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train import history, paths
from train import io as train_io
from train.config import (
    KCConfig,
    TrainerConfig,
)
from train.dataset import DatasetConfig, StyleDataset
from train.io import save_model
from train.models import StyleClassifierWithKC
from train.profile import get_profile_dir, profiling_enabled
from train.trainer import KCTrainer, Trainer
from train.types import KCTrainingHistory, TrainingHistory


def generate_profile_report() -> None:
    # pylint: disable=too-many-locals
    """Generate human-readable performance report from JSONL logs."""

    prof_dir = get_profile_dir()
    if not prof_dir or not os.path.exists(prof_dir):
        # If no profile dir, nothing to do (or profiling disabled)
        return

    # Use a dynamic summary filename based on hostname/pid or just "summary.txt"?
    # User said "aggregate .txt summary reports".
    # Let's use training-profile.txt in the profile dir.
    output_path = os.path.join(prof_dir, "training-profile.txt")

    print(f"Generating profile report from {prof_dir}...")

    files = glob.glob(os.path.join(prof_dir, "*.jsonl"))
    if not files:
        print("No .jsonl profile files found.")
        return

    all_entries = []
    for p in files:
        with open(p, "r", encoding="utf-8") as profile_file:
            for line in profile_file:
                if line.strip():
                    all_entries.append(json.loads(line))

    if not all_entries:
        print("No valid entries found.")
        return

    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get("timestamp", ""))

    # Analytics
    epochs = sorted(list(set(e.get("epoch", 0) for e in all_entries)))
    thrashing_events = [e for e in all_entries if e.get("majflt", 0) > 0]

    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write("KOTOGRAM TRAINING PERFORMANCE PROFILE\n")
        report_file.write("======================================\n")
        report_file.write(f"Generated at: {time.ctime()}\n")
        report_file.write(
            f"Source logs: {len(files)} files, {len(all_entries)} samples\n\n"
        )

        report_file.write("SYSTEM HEALTH SUMMARY\n")
        report_file.write("---------------------\n")
        max_rss = max(e.get("maxrss", 0) for e in all_entries)
        report_file.write(f"Peak RSS: {max_rss:,}\n")
        report_file.write(
            f"Total Major Page Faults (Thrashing): {sum(e.get('majflt', 0) for e in all_entries)}\n"
        )
        report_file.write(f"Thrashing Events (>0 faults): {len(thrashing_events)}\n\n")

        if thrashing_events:
            report_file.write("THRASHING TIMELINE (Top 10)\n")
            report_file.write("---------------------------\n")
            for e in sorted(thrashing_events, key=lambda x: x["majflt"], reverse=True)[
                :10
            ]:
                report_file.write(
                    f"Epoch {e.get('epoch')} Batch {e.get('batch')}: {e['majflt']} faults, Duration: {e['duration_s']:.2f}s, RSS: {e['maxrss']}\n"
                )
            report_file.write("\n")

        report_file.write("PER-EPOCH TIMING\n")
        report_file.write("----------------\n")
        for ep in epochs:
            ep_entries = [e for e in all_entries if e.get("epoch") == ep]
            if not ep_entries:
                continue

            data_entries = [e for e in ep_entries if "data" in e.get("name", "")]
            comp_entries = [e for e in ep_entries if "compute" in e.get("name", "")]

            avg_data = (
                sum(e["duration_s"] for e in data_entries) / len(data_entries)
                if data_entries
                else 0
            )
            avg_comp = (
                sum(e["duration_s"] for e in comp_entries) / len(comp_entries)
                if comp_entries
                else 0
            )

            report_file.write(f"Epoch {ep}:\n")
            report_file.write(f"  Avg Data Loading: {avg_data * 1000:.1f}ms\n")
            report_file.write(f"  Avg Compute:      {avg_comp * 1000:.1f}ms\n")
            if avg_data + avg_comp > 0:
                report_file.write(
                    f"  Data Overhead:    {avg_data / (avg_data + avg_comp):.1%}\n"
                )
            report_file.write("\n")

    print(f"Report written to {output_path}")

    # Cleanup JSONL files
    for p in files:
        os.remove(p)
    print(f"Cleaned up {len(files)} .jsonl profile files.")


def cleanup_profile_if_retrain(argv_list: List[str]) -> None:
    """Delete .profile directory if --retrain is present in arguments."""
    if "--retrain" in argv_list:
        # Use get_profile_dir to ensure we clean the correct machine-specific directory
        profile_dir = get_profile_dir()
        if profile_dir and os.path.exists(profile_dir):
            print(f"Cleaning up profile directory: {profile_dir}")
            shutil.rmtree(profile_dir, ignore_errors=True)
            os.makedirs(profile_dir, exist_ok=True)


if __name__ == "__main__":
    cleanup_profile_if_retrain(sys.argv)

    # Internal profiling when enabled
    from train.profile import setup_profiling

    setup_profiling()

    print("Starting training script...", flush=True)
    import argparse

    parser = argparse.ArgumentParser(
        description="Train style classifier (formality + gender)"
    )
    # Model architecture args (these define ModelConfig, not TrainerConfig)
    parser.add_argument(
        "--embed-dim", type=int, default=192, help="Model dimension (d_model)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=384, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=6, help="Number of attention heads"
    )
    # Training phase flags
    # Config file (required - contains TrainerConfig)
    parser.add_argument(
        "--config", type=str, required=False, help="Path to unified config.json file"
    )
    parser.add_argument(
        "--agrammatic-pattern",
        type=str,
        default=None,
        help="Pattern for agrammatic data",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain from scratch using parameters from existing checkpoint",
    )

    parser.add_argument(
        "--percent",
        type=float,
        default=None,
        help="Percentage of data to use for training (1-100)",
    )
    parser.add_argument(
        "--pretrain-kc",
        action="store_true",
        help="Run Knowledge Component (KC) sparse concept pretraining",
    )
    parser.add_argument("--kc-k", type=int, default=1024, help="KC vocabulary size")
    parser.add_argument(
        "--kc-topk", type=int, default=8, help="Number of active KCs per sample"
    )
    parser.add_argument(
        "--kc-freeze-encoder-epochs",
        type=int,
        default=1,
        help="Number of epochs to freeze encoder during KC training",
    )
    parser.add_argument(
        "--kc-sparsity-weight",
        type=float,
        default=1e-3,
        help="Sparsity regularization weight for KC activations",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps",
    )

    parser.add_argument(
        "--label",
        action="store_true",
        help="Run labeling/preprocessing only",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report from .profile logs and exit",
    )

    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Exit after loading and caching data",
    )

    args = parser.parse_args()

    if args.report:
        generate_profile_report()
        sys.exit(0)

    if not args.config:
        parser.error("--config is required for training/labeling")

    # Resolve and inject paths
    cache_dir = paths.get_cache_dir()
    args.data = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.agrammatic_data = os.path.join(cache_dir, "agrammatic_combined.tsv")
    args.output = locations.get_style_output_dir()
    args.support_dir = paths.get_style_support_dir()

    # Epoch history logging
    history_path = os.path.join(args.support_dir, "training-history.tsv")

    # Clear history if starting fresh (and not just labeling)
    if not args.resume and not args.label:
        history.clear_history(history_path)

    def _log_epoch_event(
        raw_history: Union[TrainingHistory, KCTrainingHistory],
        phase_type: str,
    ) -> None:
        """Log the latest epoch from raw_history to the TSV file."""
        if not raw_history:
            return

        # Handle dataclass history objects
        history_map = (
            raw_history.to_dict()
            if hasattr(raw_history, "to_dict")
            else (
                vars(raw_history) if not isinstance(raw_history, dict) else raw_history
            )
        )

        # Find a list column to determine current epoch index
        valid_keys = [k for k, v in history_map.items() if isinstance(v, list) and v]
        if not valid_keys:
            return

        # Get latest index from the first valid key
        idx = len(history_map[valid_keys[0]]) - 1
        if idx < 0:
            return

        current_epoch = idx + 1

        # Extract metrics for this epoch
        metrics = {}
        for k, v in history_map.items():
            if isinstance(v, list) and len(v) > idx:
                val = v[idx]
                if dataclasses.is_dataclass(val) and not isinstance(val, type):
                    val = dataclasses.asdict(val)
                metrics[k] = val

        event: history.HistoryEvent
        if phase_type == "pretrain-kc":
            # Check for diagnostics
            if (
                "kc_diagnostics" in history_map
                and isinstance(history_map["kc_diagnostics"], list)
                and len(history_map["kc_diagnostics"]) > idx
            ):
                diags = history_map["kc_diagnostics"]
                if diags[idx]:
                    d_val = diags[idx]
                    if dataclasses.is_dataclass(d_val) and not isinstance(d_val, type):
                        d_val = dataclasses.asdict(d_val)
                    diag_event = history.KcDiagEvent(epoch=current_epoch, stats=d_val)
                    history.append_event(history_path, diag_event)

            event = history.KcEpochEvent(epoch=current_epoch, metrics=metrics)
        else:
            event = history.StyleEpochEvent(epoch=current_epoch, metrics=metrics)

        history.append_event(history_path, event)

    model: Optional[StyleClassifier] = None
    tokenizer: Optional[Tokenizer] = None
    # pylint: disable=invalid-name
    checkpoint: Optional[Dict[str, Any]] = None
    vocab_grew = False

    data_files = [args.data]
    grammaticality_labels = [1]

    # Always prefer cached pre-processed files if available
    cache_dir_data = paths.get_style_dataset_cache_dir()
    gram_cache = os.path.join(cache_dir_data, "grammatic_combined.tsv")
    agram_cache = os.path.join(cache_dir_data, "agrammatic_combined.tsv")

    if os.path.exists(gram_cache):
        data_files = [gram_cache]
        grammaticality_labels = [1]

        if os.path.exists(agram_cache):
            data_files.append(agram_cache)
            grammaticality_labels.append(0)
    elif os.path.exists(args.agrammatic_data):
        data_files.append(args.agrammatic_data)
        grammaticality_labels.append(0)

    # --- Model and Data Initialization ---
    # Load tokenizer to get vocab sizes
    # STRICT: Load only from final output location. Wrapper guarantees its existence.
    tokenizer_path = os.path.join(locations.get_style_output_dir(), "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Critical: Tokenizer not found at {tokenizer_path}. "
            "Please run with --label first or via the wrapper script."
        )

    tokenizer = Tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path}")

    # Create a single TrainerConfig and ModelConfig
    if args.config and os.path.exists(args.config):
        model_config, trainer_config = TrainerConfig.load_config(args.config)

        # Override resume_from if --resume flag is present but not in config
        # This handles auto-resume from the wrapper while keeping config.json stable
        if args.resume and not trainer_config.checkpoint.resume_from:
            # type: ignore[misc]
            object.__setattr__(
                trainer_config.checkpoint, "resume_from", args.support_dir
            )

        # Override kc_enabled if specified in arguments (CLI takes precedence over config)
        if args.pretrain_kc:
            model_config.kc_enabled = True
    else:
        print(
            "ERROR: --config is required. Configuration must be passed from the wrapper script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tokenizer and Device after config load
    device = torch.device(trainer_config.device)

    # Initialize model if not already loaded from checkpoint
    if model is None:
        if model_config.kc_enabled:
            # Populate KC target specs based on vocab availability (Implicit "All Targets" policy)
            current_vocab_sizes = tokenizer.get_vocab_sizes()
            kc_specs = {}

            # 1. Base Fields (Bag + Tail + Ngram)
            # checking for "reading" and mapping to "bag_reading" is correct because
            # dataset will alias "reading" -> "bag_reading_gram" if needed,
            # but model spec "bag_reading" is what creates the decoder.
            # Wait, dataset output keys are "kc_targets_bag_reading_gram".
            # Trainer expects keys in batch to match keys in model.kc_target_specs?
            # No, Trainer calculates losses for keys in `kc_losses` which comes from model forward.
            # Model forward needs correct head names.
            # Let's rely on the established naming convention from train/kc.py: "bag_{field}"

            for field in [
                "reading",
                "pos",
                "pos_detail1",
                "conjugated_form",
                "conjugated_type",
            ]:
                if field in current_vocab_sizes:
                    v_size = current_vocab_sizes[field]
                    kc_specs[f"bag_{field}"] = v_size
                    kc_specs[f"tail_{field}"] = v_size
                    kc_specs[f"ngram_{field}"] = (
                        16384  # KC_HASH_BUCKETS constant, hardcoded for now or import
                    )
                    kc_specs[f"tail_ngram_{field}"] = 16384

            # 2. Pairs
            def add_pair(name: str, f1: str, f2: str) -> None:
                if f1 in current_vocab_sizes and f2 in current_vocab_sizes:
                    kc_specs[name] = 16384

            add_pair("pair_pos_conj", "pos", "conjugated_form")
            add_pair("pair_pos1_conjform", "pos_detail1", "conjugated_form")
            add_pair("pair_pos1_conjtype", "pos_detail1", "conjugated_type")

            model_config.kc_target_specs = kc_specs
            model = StyleClassifierWithKC(model_config)
        else:
            model = StyleClassifier(model_config)

    # Load labeled data for remaining phases
    old_vocab_sizes = model_config.vocab_sizes.copy()
    # pylint: disable=protected-access
    tokenizer._frozen = False
    labeled_dataset = StyleDataset.from_multiple_tsv(
        data_files,
        tokenizer,
        config=DatasetConfig(
            verbose=True,
            grammaticality_labels=grammaticality_labels,
            sample_ratio=args.percent / 100.0 if args.percent else 1.0,
        ),
    )
    train_data, val_data = labeled_dataset.split()
    new_vocab_sizes = tokenizer.get_vocab_sizes()
    vocab_grew = any(new_vocab_sizes[f] > old_vocab_sizes[f] for f in FEATURE_FIELDS)

    # Force update configuration (handles frozen dataclass if applicable)
    object.__setattr__(model_config, "vocab_sizes", new_vocab_sizes)

    if vocab_grew:
        model.resize_embeddings(new_vocab_sizes)
        model_config.vocab_sizes = new_vocab_sizes
        # Save updated tokenizer to output_dir so resumption finds it
        train_io.save_tokenizer(
            tokenizer,
            os.path.join(locations.get_style_output_dir(), "tokenizer.json"),
        )

    if args.preprocess_only:
        sys.exit(0)

    if args.pretrain_kc and not args.preprocess_only:
        # Load history to check KC progress
        events = history.read_events(history_path)
        kc_epochs_done = sum(1 for e in events if isinstance(e, history.KcEpochEvent))
        if kc_epochs_done < trainer_config.kc_epochs or args.retrain:
            kc_trainer = KCTrainer(
                cast(StyleClassifierWithKC, model),
                train_data,
                trainer_config,
                dl_config=trainer_config.resolve_dataloader_config(device),
                kc_config=KCConfig(
                    sparsity_weight=args.kc_sparsity_weight,
                    freeze_encoder_epochs=args.kc_freeze_encoder_epochs,
                ),
            )
            kc_hist: KCTrainingHistory = kc_trainer.train(
                epochs=trainer_config.kc_epochs,
                on_epoch_end=lambda h: _log_epoch_event(h, "pretrain-kc"),
            )
            if kc_hist.total_loss:
                print(
                    f"KC Pretraining finished. Final loss: {kc_hist.total_loss[-1]:.4f}"
                )
            # Ensure final state is logged if not already (redundant if using callback)
            # But callback might be skipped if 0 epochs? No, train loop handles it.

            # Update model reference (Trainer may have wrapped/moved it)
            model = kc_trainer.model
            if hasattr(model, "module"):
                model = cast(StyleClassifierWithKC, model.module)
            model.reset_classifier()

    # Final supervised training
    style_trainer = Trainer(
        model,
        train_data,
        val_data,
        trainer_config,
        dl_config_train=trainer_config.resolve_dataloader_config(device, mode="train"),
        dl_config_val=trainer_config.resolve_dataloader_config(device, mode="val"),
        output_path=args.output,
    )

    style_start = time.perf_counter()
    style_end = style_start
    if args.resume:
        # Auto-resume handled inside trainer.train() if checkpoint_dir set
        pass

    style_hist: TrainingHistory = style_trainer.train(
        epochs=trainer_config.epochs,
        on_epoch_end=lambda h: _log_epoch_event(h, "style"),
    )
    if style_hist.train_loss:
        print(f"Style Training finished. Final loss: {style_hist.train_loss[-1]:.4f}")

    # _log_epoch_event already logs during callbacks
    style_end = time.perf_counter()

    # Test evaluation and model saving
    res = style_trainer.evaluate()
    print("----------------------------------")
    print(
        f"Final Test Results:\n"
        f"Accuracy: form={res.formality_accuracy:.4f}, gender={res.gender_accuracy:.4f}, gram={res.grammaticality_accuracy:.4f}, register={res.register_accuracy:.4f}"
    )
    print("----------------------------------")

    # Save model
    output_dir = locations.get_style_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    # from train.io import save_model  # Already imported
    # pylint: disable=reimported

    # Ensure we use the trained model
    trained_model = style_trainer.model
    if hasattr(trained_model, "module"):
        trained_model = cast(StyleClassifier, trained_model.module)

    # Create __init__.py to make the model directory a valid Python package
    # This is required for 'kotogram.model_data' redirection in pyproject.toml
    init_path = os.path.join(output_dir, "__init__.py")
    with open(init_path, "w", encoding="utf-8"):
        pass

    save_model(
        cast(StyleClassifier, trained_model),
        output_dir,
        model_config,
    )
    print(f"Model saved to: {output_dir}")

    # Final timing report
    print("-" * 34)
    print("Performance Summary:")
    print("-" * 34)
    if args.pretrain_kc and trainer_config.kc_epochs > 0:
        # Approximate pretraining time using what logic we have left or just remove details
        pass
    print(f"  Style Training: {style_end - style_start:.1f}s")
    print("-" * 34)

    # Auto-generate report and cleanup if profiling was enabled
    # We check environment because arguments might not settle it alone (defaults)
    # But usually if we ran code, we generated logs.
    if profiling_enabled() and not args.report:
        # args.report exits early, so we only need to do this for a normal run
        generate_profile_report()
