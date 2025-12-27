"""Supervised style classifier for Japanese sentences using Kotogram representations.

This script orchestrates the training pipeline, including argument parsing,
data loading, and calling the trainers from the kotogram.train package.
"""

import importlib.util

if importlib.util.find_spec("_setup_path"):
    import _setup_path  # type: ignore # noqa: F401 # pylint: disable=unused-import
else:
    from scripts import (
        _setup_path,  # type: ignore # noqa: F401 # pylint: disable=unused-import
    )

# pylint: disable=wrong-import-position
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, cast

import torch
import torch.distributed as dist

from kotogram import locations
from kotogram.model import (
    StyleClassifier,
)
from kotogram.tokenizer import (
    FEATURE_FIELDS,
    Tokenizer,
)
from train.config import (
    TrainerConfig,
)
from train.dataset import DatasetConfig, StyleDataset
from train.distributed import is_main_process, setup_distributed
from train.io import save_model
from train.models import StyleClassifierWithKC
from train.profile import get_profile_dir
from train.trainer import KCTrainer, Trainer


def generate_profile_report() -> None:
    # pylint: disable=too-many-locals
    """Generate human-readable performance report from JSONL logs."""
    prof_dir = get_profile_dir()
    if not prof_dir or not os.path.exists(prof_dir):
        # If no profile dir, nothing to do (or TRAIN_PROFILE=0)
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


if __name__ == "__main__":
    # Internal profiling when TRAIN_PROFILE is set
    _profiler = None
    if os.environ.get("TRAIN_PROFILE", "1") != "0":
        import atexit
        import cProfile

        _profiler = cProfile.Profile()
        _profiler.enable()

        def _save_profile() -> None:
            if _profiler:
                _profiler.disable()
                import pstats

                prof_dir = get_profile_dir()
                if prof_dir:
                    os.makedirs(prof_dir, exist_ok=True)

                    # Write .pstats file
                    pstats_file = os.path.join(
                        prof_dir, f"train_style_{os.getpid()}.pstats"
                    )
                    _profiler.dump_stats(pstats_file)

                    # Write human-readable summary
                    summary_file = os.path.join(
                        prof_dir, f"train_style_{os.getpid()}.txt"
                    )
                    with open(
                        summary_file, "w", encoding="utf-8"
                    ) as summary_file_handle:
                        stats = pstats.Stats(_profiler, stream=summary_file_handle)

                        stats.sort_stats("cumulative")
                        summary_file_handle.write("TOP 50 BY CUMULATIVE TIME\n")
                        summary_file_handle.write("=" * 80 + "\n")
                        stats.print_stats(50)

                        summary_file_handle.write("\n")

                        stats.sort_stats("calls")
                        summary_file_handle.write("TOP 50 BY INVOCATION COUNT\n")
                        summary_file_handle.write("=" * 80 + "\n")
                        stats.print_stats(50)

        atexit.register(_save_profile)

    if os.environ.get("RANK", "0") == "0":
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
        "--fp16",
        action="store_true",
        default=None,
        help="Save model in float16 precision",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=None,
        help="Save model in float8 precision",
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

    # Check if we should clean up profile directory
    if "--retrain" in sys.argv:
        profile_dir = os.path.join(os.environ.get("TRAIN_ROOT", "."), ".profile")
        if os.path.exists(profile_dir):
            import shutil

            shutil.rmtree(profile_dir, ignore_errors=True)
            if is_main_process():
                print(f"Cleaned up profile directory: {profile_dir}")
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
        "--kc-target-heads",
        type=str,
        default="lemma,pos,conjugated_form",
        help="Target heads for KC supervision",
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
    cache_dir = locations.get_cache_dir()
    args.data = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.agrammatic_data = os.path.join(cache_dir, "agrammatic_combined.tsv")
    args.output = locations.get_style_output_dir()
    args.support_dir = locations.get_style_support_dir()

    rank, world_size, local_rank = setup_distributed()

    # Epoch history logging
    epochs_json_path = os.path.join(args.support_dir, "epochs.json")
    training_history: List[Dict[str, Any]] = []
    if is_main_process() and os.path.exists(epochs_json_path):
        with open(epochs_json_path, "r", encoding="utf-8") as history_file:
            training_history = json.load(history_file)

    def _append_history(
        raw_history: Dict[str, Any], phase_type: str, start_epoch: int = 0
    ) -> None:
        """Convert columnar history to rows and append to training_history."""
        if not raw_history:
            return

        # Determine num_epochs from the first list value found
        # (e.g., 'loss', 'total_loss')
        num_epochs = 0
        for v in raw_history.values():
            if isinstance(v, list):
                num_epochs = len(v)
                break

        if num_epochs == 0:
            return

        # Replace existing entries for this phase type to support incremental updates
        training_history[:] = [
            e for e in training_history if e.get("type") != phase_type
        ]

        for i in range(num_epochs):
            epoch_data = {
                "type": phase_type,
                "epoch": start_epoch + i + 1,
            }
            # Flatten metrics
            for k, v in raw_history.items():
                if isinstance(v, list):
                    if i < len(v):
                        epoch_data[k] = v[i]
                elif isinstance(v, dict):
                    # Flatten nested dicts (keys -> list of vals)
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, list) and i < len(sub_v):
                            epoch_data[f"{k}_{sub_k}"] = sub_v[i]
            training_history.append(epoch_data)

        if is_main_process():
            # Save immediately
            with open(epochs_json_path, "w", encoding="utf-8") as h_file:
                json.dump(training_history, h_file, indent=2)

    model: Optional[StyleClassifier] = None
    tokenizer: Optional[Tokenizer] = None
    # pylint: disable=invalid-name
    checkpoint: Optional[Dict[str, Any]] = None
    vocab_grew = False

    if args.fp16 is None:
        args.fp16 = False
    if args.fp8 is None:
        args.fp8 = not args.fp16

    data_files = [args.data]
    grammaticality_labels = [1]

    # Always prefer cached pre-processed files if available
    cache_dir_data = locations.get_style_dataset_cache_dir()
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
    if is_main_process():
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
    else:
        print(
            "ERROR: --config is required. Configuration must be passed from the wrapper script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tokenizer and Device after config load
    device = torch.device(trainer_config.device)

    # Initialize model if not already loaded from checkpoint
    # Initialize model if not already loaded from checkpoint
    if model is None:
        if model_config.kc_enabled:
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
            verbose=is_main_process(),
            grammaticality_labels=grammaticality_labels,
            sample_ratio=args.percent / 100.0 if args.percent else 1.0,
            use_cache=True,
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
        if is_main_process():
            tokenizer.save(
                os.path.join(locations.get_style_output_dir(), "tokenizer.json")
            )

    if args.preprocess_only:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    if args.pretrain_kc and not args.preprocess_only:
        kc_epochs_done = len(
            [e for e in training_history if e.get("type") == "pretrain-kc"]
        )
        if kc_epochs_done < trainer_config.kc_epochs or args.retrain:
            kc_trainer = KCTrainer(
                cast(StyleClassifierWithKC, model),
                train_data,
                trainer_config,
                dl_config=trainer_config.resolve_dataloader_config(
                    device, is_main_process()
                ),
                kc_config={
                    "sparsity_weight": args.kc_sparsity_weight,
                    "freeze_encoder_epochs": args.kc_freeze_encoder_epochs,
                },
                args=args,
            )
            kc_history = kc_trainer.train(
                epochs=trainer_config.kc_epochs,
                on_epoch_end=lambda h: _append_history(h, "pretrain-kc"),
            )
            if is_main_process():
                _append_history(kc_history, "pretrain-kc")
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
        dl_config_train=trainer_config.resolve_dataloader_config(
            device, is_main_process(), mode="train"
        ),
        dl_config_val=trainer_config.resolve_dataloader_config(
            device, is_main_process(), mode="val"
        ),
    )

    style_start = time.perf_counter()
    style_end = style_start
    if args.resume:
        # Auto-resume handled inside trainer.train() if checkpoint_dir set
        pass

    history = style_trainer.train(
        epochs=trainer_config.epochs,
        on_epoch_end=lambda h: _append_history(h, "style"),
    )
    if is_main_process():
        _append_history(history, "style")
    style_end = time.perf_counter()

    # Test evaluation and model saving
    res = style_trainer.evaluate()
    if is_main_process():
        print("-" * 34)
        print("Final Test Results:")
        print("-" * 34)
        print(
            f"Accuracy: form={res['formality_accuracy']:.4f}, gender={res['gender_accuracy']:.4f}, gram={res['grammaticality_accuracy']:.4f}, register={res['register_accuracy']:.4f}"
        )
        print("-" * 34)

        # Save model
        output_dir = locations.get_style_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        # from train.io import save_model  # Already imported
        # pylint: disable=reimported

        # Ensure we use the trained model
        trained_model = style_trainer.model
        if hasattr(trained_model, "module"):
            trained_model = cast(StyleClassifier, trained_model.module)

        save_model(
            cast(StyleClassifier, trained_model),
            output_dir,
            None,  # Tokenizer already saved by wrapper/growth logic
            model_config,
            fp16=trainer_config.use_amp,
            fp8=args.fp8,
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

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Auto-generate report and cleanup if profiling was enabled
    # We check environment because arguments might not settle it alone (defaults)
    # But usually if we ran code, we generated logs.
    if os.environ.get("TRAIN_PROFILE", "1") != "0" and not args.report:
        # args.report exits early, so we only need to do this for a normal run
        generate_profile_report()
