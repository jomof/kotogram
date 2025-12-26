"""Supervised style classifier for Japanese sentences using Kotogram representations.

This script orchestrates the training pipeline, including argument parsing,
data loading, and calling the trainers from the kotogram.train package.
"""

try:
    import _setup_path  # type: ignore # noqa: F401
except ImportError:
    from scripts import _setup_path  # type: ignore # noqa: F401

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
from train.dataset import StyleDataset
from train.io import save_model
from train.trainer import (
    KCTrainer,
    MLMTrainer,
    StyleClassifierWithMLM,
    Trainer,
    is_main_process,
    setup_distributed,
)


def generate_profile_report() -> None:
    """Generate human-readable performance report from JSONL logs."""
    profile_dir = os.path.join(os.environ.get("TRAIN_ROOT", "."), ".profile")
    support_dir = locations.get_style_support_dir()
    os.makedirs(support_dir, exist_ok=True)
    output_path = os.path.join(support_dir, "training-profile.txt")

    print(f"Generating profile report from {profile_dir}...")

    files = glob.glob(os.path.join(profile_dir, "*.jsonl"))
    if not files:
        print("No .jsonl profile files found.")
        return

    all_entries = []
    for p in files:
        with open(p, "r") as f:
            for line in f:
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not all_entries:
        print("No valid entries found.")
        return

    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get("timestamp", ""))

    # Analytics
    epochs = sorted(list(set(e.get("epoch", 0) for e in all_entries)))
    thrashing_events = [e for e in all_entries if e.get("majflt", 0) > 0]

    with open(output_path, "w") as f:
        f.write("KOTOGRAM TRAINING PERFORMANCE PROFILE\n")
        f.write("======================================\n")
        f.write(f"Generated at: {time.ctime()}\n")
        f.write(f"Source logs: {len(files)} files, {len(all_entries)} samples\n\n")

        f.write("SYSTEM HEALTH SUMMARY\n")
        f.write("---------------------\n")
        max_rss = max(e.get("maxrss", 0) for e in all_entries)
        f.write(f"Peak RSS: {max_rss:,}\n")
        f.write(
            f"Total Major Page Faults (Thrashing): {sum(e.get('majflt', 0) for e in all_entries)}\n"
        )
        f.write(f"Thrashing Events (>0 faults): {len(thrashing_events)}\n\n")

        if thrashing_events:
            f.write("THRASHING TIMELINE (Top 10)\n")
            f.write("---------------------------\n")
            for e in sorted(thrashing_events, key=lambda x: x["majflt"], reverse=True)[
                :10
            ]:
                f.write(
                    f"Epoch {e.get('epoch')} Batch {e.get('batch')}: {e['majflt']} faults, Duration: {e['duration_s']:.2f}s, RSS: {e['maxrss']}\n"
                )
            f.write("\n")

        f.write("PER-EPOCH TIMING\n")
        f.write("----------------\n")
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

            f.write(f"Epoch {ep}:\n")
            f.write(f"  Avg Data Loading: {avg_data * 1000:.1f}ms\n")
            f.write(f"  Avg Compute:      {avg_comp * 1000:.1f}ms\n")
            if avg_data + avg_comp > 0:
                f.write(f"  Data Overhead:    {avg_data / (avg_data + avg_comp):.1%}\n")
            f.write("\n")

    print(f"Report written to {output_path}")


if __name__ == "__main__":
    # Internal profiling when TRAIN_PROFILE is set
    _profiler = None
    if os.environ.get("TRAIN_PROFILE"):
        import atexit
        import cProfile

        _profiler = cProfile.Profile()
        _profiler.enable()

        def _save_profile() -> None:
            if _profiler:
                _profiler.disable()
                import pstats

                profile_dir = os.path.join(
                    os.environ.get("TRAIN_ROOT", "."), ".profile"
                )
                os.makedirs(profile_dir, exist_ok=True)

                # Write .pstats file
                pstats_file = os.path.join(
                    profile_dir, f"train_style_{os.getpid()}.pstats"
                )
                _profiler.dump_stats(pstats_file)

                # Write human-readable summary
                summary_file = os.path.join(
                    profile_dir, f"train_style_{os.getpid()}.txt"
                )
                with open(summary_file, "w") as f:
                    stats = pstats.Stats(_profiler, stream=f)
                    stats.sort_stats("cumulative")
                    f.write("TOP 50 BY CUMULATIVE TIME\n")
                    f.write("=" * 80 + "\n")
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
    parser.add_argument(
        "--pretrain-mlm",
        action="store_true",
        help="Pre-train with masked language modeling",
    )
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
        try:
            with open(epochs_json_path, "r") as f:
                training_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Safe to ignore, might be first run or race condition
            pass

    def _append_history(
        raw_history: Dict[str, Any], phase_type: str, start_epoch: int = 0
    ) -> None:
        """Convert columnar history to rows and append to training_history."""
        if not raw_history:
            return

        # Determine num_epochs from the first list value found
        # (e.g., 'loss', 'mlm_loss', 'total_loss')
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
                    # Flatten nested dicts (like field_losses in MLM: keys -> list of vals)
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, list) and i < len(sub_v):
                            epoch_data[f"{k}_{sub_k}"] = sub_v[i]
            training_history.append(epoch_data)

        if is_main_process():
            # Save immediately
            with open(epochs_json_path, "w") as f:
                json.dump(training_history, f, indent=2)

    model: Optional[StyleClassifier] = None
    tokenizer: Optional[Tokenizer] = None
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
    # Priority: support_dir (training progress) > cache (labeling result)
    tokenizer_path_support = os.path.join(args.support_dir, "tokenizer.json")
    tokenizer_path_cache = os.path.join(cache_dir, "style_dataset", "vocab.json")

    tokenizer = None
    if os.path.exists(tokenizer_path_support):
        try:
            tokenizer = Tokenizer.load(tokenizer_path_support)
            if is_main_process():
                print(f"Loaded tokenizer from {tokenizer_path_support}")
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer from {tokenizer_path_support}: {e}")

    if tokenizer is None:
        if os.path.exists(tokenizer_path_cache):
            try:
                tokenizer = Tokenizer.load(tokenizer_path_cache)
            except Exception as e:
                print(
                    f"ERROR: Failed to load tokenizer from {tokenizer_path_cache}: {e}"
                )
                tokenizer = Tokenizer()
        else:
            # Fallback to loading via StyleDataset logic or empty
            tokenizer = Tokenizer()
            vocab_legacy = os.path.join(
                locations.get_style_dataset_cache_dir(), "vocab.json"
            )
            if os.path.exists(vocab_legacy):
                StyleDataset._load_vocab(vocab_legacy, tokenizer)
            else:
                raise ValueError(f"Vocabulary not found at {tokenizer_path_cache}")

    # Create a single TrainerConfig and ModelConfig
    if args.config and os.path.exists(args.config):
        try:
            model_config, trainer_config = TrainerConfig.load_config(args.config)

            # Override resume_from if --resume flag is present but not in config
            # This handles auto-resume from the wrapper while keeping config.json stable
            if args.resume and not trainer_config.checkpoint.resume_from:
                # type: ignore[misc]
                object.__setattr__(
                    trainer_config.checkpoint, "resume_from", args.support_dir
                )
        except Exception as e:
            if is_main_process():
                print(f"Failed to load existing config: {e}")
            sys.exit(1)
    else:
        print(
            "ERROR: --config is required. Configuration must be passed from the wrapper script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tokenizer and Device after config load
    device = torch.device(trainer_config.device)

    # Tokenizer may need to be saved to support_dir for checkpoint loading
    if is_main_process():
        os.makedirs(args.support_dir, exist_ok=True)
        tokenizer.save(os.path.join(args.support_dir, "tokenizer.json"))

    # Initialize model if not already loaded from checkpoint
    # Initialize model if not already loaded from checkpoint
    if model is None:
        if model_config.mlm_enabled or model_config.kc_enabled:
            model = StyleClassifierWithMLM(model_config)
        else:
            model = StyleClassifier(model_config)

    # MLM/KC Pretraining
    mlm_start = time.perf_counter()
    mlm_end = mlm_start
    # Phase 1: MLM Pretraining
    if args.pretrain_mlm and not args.preprocess_only:
        # Check if already done in history?
        mlm_epochs_done = len(
            [e for e in training_history if e.get("type") == "pretrain-mlm"]
        )
        if mlm_epochs_done < trainer_config.mlm_epochs or args.retrain:
            unlabeled_dataset = StyleDataset.from_tsv(
                args.data,
                tokenizer,
                verbose=is_main_process(),
                labeled=False,
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                use_cache=True,
            )
            mlm_trainer = MLMTrainer(
                cast(StyleClassifierWithMLM, model),
                unlabeled_dataset,
                trainer_config,
                dl_config=trainer_config.resolve_dataloader_config(
                    device, is_main_process()
                ),
                args=args,
            )
            mlm_history = mlm_trainer.train(
                epochs=trainer_config.mlm_epochs,
                verbose=is_main_process(),
                on_epoch_end=lambda h: _append_history(h, "pretrain-mlm"),
            )
            if is_main_process():
                _append_history(mlm_history, "pretrain-mlm")
            # Update model reference (Trainer may have wrapped/moved it)
            model = mlm_trainer.model
            if hasattr(model, "module"):
                model = cast(StyleClassifierWithMLM, model.module)
            model.reset_classifier()

    # Load labeled data for remaining phases
    old_vocab_sizes = tokenizer.get_vocab_sizes()
    tokenizer._frozen = False
    labeled_dataset = StyleDataset.from_multiple_tsv(
        data_files,
        tokenizer,
        verbose=is_main_process(),
        grammaticality_labels=grammaticality_labels,
        sample_ratio=args.percent / 100.0 if args.percent else 1.0,
        use_cache=True,
    )
    train_data, val_data = labeled_dataset.split()
    new_vocab_sizes = tokenizer.get_vocab_sizes()
    vocab_grew = any(new_vocab_sizes[f] > old_vocab_sizes[f] for f in FEATURE_FIELDS)

    # Force update configuration (handles frozen dataclass if applicable)
    object.__setattr__(model_config, "vocab_sizes", new_vocab_sizes)

    if vocab_grew:
        model.resize_embeddings(new_vocab_sizes)
        model_config.vocab_sizes = new_vocab_sizes
        # Save updated tokenizer to support_dir so resumption finds it
        if is_main_process():
            tokenizer.save(os.path.join(args.support_dir, "tokenizer.json"))

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
                cast(StyleClassifierWithMLM, model),
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
                model = cast(StyleClassifierWithMLM, model.module)
            model.reset_classifier()
    mlm_end = time.perf_counter()

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
            f"Accuracy: form={res['formality_accuracy']:.4f}, gender={res['gender_pragmatic_accuracy']:.4f}, gram={res['grammaticality_accuracy']:.4f}, register={res['register_accuracy']:.4f}"
        )
        print("-" * 34)

        # Save model
        output_dir = locations.get_style_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        from train.io import save_model

        # Ensure we use the trained model
        trained_model = style_trainer.model
        if hasattr(trained_model, "module"):
            trained_model = cast(StyleClassifier, trained_model.module)

        save_model(
            cast(StyleClassifier, trained_model),
            tokenizer,
            output_dir,
            model_config,
            fp16=trainer_config.use_amp,
            fp8=args.fp8,
        )
        print(f"Model saved to: {output_dir}")

        # Final timing report
        print("-" * 34)
        print("Performance Summary:")
        print("-" * 34)
        if (args.pretrain_mlm and trainer_config.mlm_epochs > 0) or (
            args.pretrain_kc and trainer_config.kc_epochs > 0
        ):
            print(f"  Pretraining: {mlm_end - mlm_start:.1f}s")
        print(f"  Style Training: {style_end - style_start:.1f}s")
        print("-" * 34)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
