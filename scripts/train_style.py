"""Supervised style classifier for Japanese sentences using Kotogram representations.

This script orchestrates the training pipeline, including argument parsing,
data loading, and calling the trainers from the kotogram.train package.
"""

try:
    import _setup_path  # type: ignore # noqa: F401
except ImportError:
    from scripts import _setup_path  # type: ignore # noqa: F401

import json
import os
import sys
from typing import Any, Dict, List, Optional, cast

import torch
import torch.distributed as dist

from kotogram import locations
from kotogram.model import (
    NUM_FORMALITY_PRAGMATIC_CLASSES,
    NUM_GENDER_PRAGMATIC_CLASSES,
    NUM_GRAMMATICALITY_CLASSES,
    ModelConfig,
    StyleClassifier,
)
from kotogram.tokenizer import (
    FEATURE_FIELDS,
    Tokenizer,
)
from train.config import TrainerConfig
from train.dataset import StyleDataset
from train.io import load_checkpoint, save_model
from train.trainer import (
    KCTrainer,
    MLMTrainer,
    StyleClassifierWithMLM,
    Trainer,
    is_main_process,
    setup_distributed,
)

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
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
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
    parser.add_argument(
        "--pretrain-mlm",
        action="store_true",
        help="Pre-train with masked language modeling",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=5, help="MLM pretraining epochs"
    )
    parser.add_argument(
        "--encoder-lr-factor",
        type=float,
        default=0.1,
        help="Learning rate multiplier for encoder during fine-tuning",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Base learning rate"
    )
    parser.add_argument(
        "--formality-weight",
        type=float,
        default=1.0,
        help="Loss weight for formality task",
    )
    parser.add_argument(
        "--gender-weight", type=float, default=1.0, help="Loss weight for gender task"
    )
    parser.add_argument(
        "--grammaticality-weight",
        type=float,
        default=1.0,
        help="Loss weight for grammaticality task",
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
    parser.add_argument(
        "--kc-epochs", type=int, default=3, help="Number of KC pretraining epochs"
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
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Exit after loading and caching data",
    )

    args = parser.parse_args()

    # Resolve and inject paths
    cache_dir = locations.get_cache_dir()
    args.data = os.path.join(cache_dir, "grammatic_combined.tsv")
    args.agrammatic_data = os.path.join(cache_dir, "agrammatic_combined.tsv")
    args.output = locations.get_style_output_dir()
    args.support_dir = locations.get_style_support_dir()

    if is_main_process():
        print("Setting up distributed training...", flush=True)
    rank, world_size, local_rank = setup_distributed()

    # Epoch history logging
    epochs_json_path = os.path.join(args.support_dir, "epochs.json")
    training_history: List[Dict[str, Any]] = []
    if os.path.exists(epochs_json_path):
        try:
            with open(epochs_json_path, "r") as f:
                training_history = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {epochs_json_path}", file=sys.stderr)

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

        # Save immediately
        with open(epochs_json_path, "w") as f:
            json.dump(training_history, f, indent=2)

    model: Optional[StyleClassifier] = None
    tokenizer: Optional[Tokenizer] = None
    checkpoint: Optional[Dict[str, Any]] = None
    strict_load_failed = False
    vocab_grew = False

    if args.resume or args.retrain:
        checkpoint_path = os.path.join(args.support_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            saved_args = checkpoint_data["args"]

            if args.resume:
                # Restore flags needed for model reconstruction FIRST
                args.pretrain_mlm = saved_args.get("pretrain_mlm", False)
                args.pretrain_kc = saved_args.get("pretrain_kc", False)

                # Use StyleClassifierWithMLM if we are doing pretraining, as it has the 'mode' argument in forward
                model_class = (
                    StyleClassifierWithMLM
                    if (args.pretrain_mlm or args.pretrain_kc)
                    else StyleClassifier
                )
                model, tokenizer, checkpoint, strict_load_failed = load_checkpoint(
                    args.support_dir, model_class=model_class
                )

            args.embed_dim = saved_args["embed_dim"]
            args.hidden_dim = saved_args["hidden_dim"]
            args.num_layers = saved_args["num_layers"]
            args.num_heads = saved_args["num_heads"]
            args.kc_k = saved_args.get("kc_k", 1024)
            args.kc_topk = saved_args.get("kc_topk", 8)
            args.kc_target_heads = saved_args.get(
                "kc_target_heads", "lemma,pos,conjugated_form"
            )
            args.learning_rate = saved_args["learning_rate"]
            args.encoder_lr_factor = saved_args.get("encoder_lr_factor", 0.1)
            args.formality_weight = saved_args.get("formality_weight", 1.0)
            args.gender_weight = saved_args.get("gender_weight", 1.0)
            args.grammaticality_weight = saved_args.get("grammaticality_weight", 1.0)

            if args.percent is None:
                args.percent = saved_args.get("percent", None)
            if args.epochs is None:
                args.epochs = saved_args.get("epochs", 20)
            if args.fp16 is None:
                args.fp16 = saved_args.get("fp16", False)
            if args.fp8 is None:
                args.fp8 = saved_args.get("fp8", False)

    if args.fp16 is None:
        args.fp16 = False
    if args.fp8 is None:
        args.fp8 = not args.fp16
    if args.epochs is None:
        args.epochs = 20

    data_files = [args.data]
    grammaticality_labels = [1]
    if os.path.exists(args.agrammatic_data):
        data_files.append(args.agrammatic_data)
        grammaticality_labels.append(0)

    # --- Model and Data Initialization ---
    if tokenizer is None:
        tokenizer = Tokenizer()
        vocab_path = os.path.join(locations.get_style_dataset_cache_dir(), "vocab.json")
        if os.path.exists(vocab_path):
            StyleDataset._load_vocab(vocab_path, tokenizer)
            if is_main_process():
                print(f"  Loaded vocabulary from cache: {vocab_path}")
        else:
            raise ValueError(f"Vocabulary not found at {vocab_path}")

    # Prepare model configuration
    model_config = ModelConfig(
        vocab_sizes=tokenizer.get_vocab_sizes(),
        num_formality_pragmatic_classes=NUM_FORMALITY_PRAGMATIC_CLASSES,
        num_gender_pragmatic_classes=NUM_GENDER_PRAGMATIC_CLASSES,
        num_grammaticality_classes=NUM_GRAMMATICALITY_CLASSES,
        d_model=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kc_enabled=args.pretrain_kc,
        kc_vocab_size=args.kc_k,
        kc_topk=args.kc_topk,
        kc_target_specs={
            h.strip(): tokenizer.get_vocab_sizes().get(h.strip(), 100)
            for h in args.kc_target_heads.split(",")
            if h.strip()
        }
        if args.pretrain_kc
        else {},
    )

    # Initialize model if not already loaded from checkpoint
    if model is None:
        if args.pretrain_mlm or args.pretrain_kc:
            model = StyleClassifierWithMLM(model_config)
        else:
            model = StyleClassifier(model_config)

    # Load checkpoint if resuming
    if args.resume and checkpoint is not None:
        old_vocab_sizes = tokenizer.get_vocab_sizes()
        # load_checkpoint already handles model_class if passed, but we already have the model.
        # Actually, let's use the existing load_checkpoint logic to be safe, but we've already inited.
        # Restoring state dict later.
        pass

    # Phase 1: MLM Pretraining
    if args.pretrain_mlm and not args.preprocess_only:
        # Check if already done in history?
        has_mlm = any(e.get("type") == "pretrain-mlm" for e in training_history)
        if not has_mlm or args.retrain:
            unlabeled_dataset = StyleDataset.from_tsv(
                args.data,
                tokenizer,
                verbose=is_main_process(),
                labeled=False,
                sample_ratio=args.percent / 100.0 if args.percent else 1.0,
                use_cache=False,
            )
            pretrain_config = TrainerConfig(
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_amp=args.fp16 or args.fp8,
                local_rank=local_rank,
                world_size=world_size,
                grad_accum_steps=args.grad_accum_steps,
            )
            mlm_loader = MLMTrainer(
                cast(StyleClassifierWithMLM, model), unlabeled_dataset, pretrain_config
            )
            mlm_history = mlm_loader.train(
                epochs=args.pretrain_epochs,
                verbose=is_main_process(),
                on_epoch_end=lambda h: _append_history(h, "pretrain-mlm"),
            )
            if is_main_process():
                _append_history(mlm_history, "pretrain-mlm")
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
    )
    train_data, val_data = labeled_dataset.split()
    new_vocab_sizes = tokenizer.get_vocab_sizes()
    vocab_grew = any(new_vocab_sizes[f] > old_vocab_sizes[f] for f in FEATURE_FIELDS)
    if vocab_grew:
        model.resize_embeddings(new_vocab_sizes)

    if args.preprocess_only:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    if args.pretrain_kc and not args.preprocess_only:
        has_kc = any(e.get("type") == "pretrain-kc" for e in training_history)
        if not has_kc or args.retrain:
            kc_trainer_config = TrainerConfig(
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.kc_epochs,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=args.fp16 or args.fp8,
                world_size=world_size,
                local_rank=local_rank,
            )
            kc_history = KCTrainer(
                cast(StyleClassifierWithMLM, model),
                train_data,
                kc_trainer_config,
                {
                    "sparsity_weight": args.kc_sparsity_weight,
                    "freeze_encoder_epochs": args.kc_freeze_encoder_epochs,
                },
            ).train(
                epochs=args.kc_epochs,
                on_epoch_end=lambda h: _append_history(h, "pretrain-kc"),
            )
            if is_main_process():
                _append_history(kc_history, "pretrain-kc")
            model.reset_classifier()

    # Final supervised training
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        formality_loss_weight=args.formality_weight,
        gender_loss_weight=args.gender_weight,
        grammaticality_loss_weight=args.grammaticality_weight,
        use_amp=args.fp16 or args.fp8,
        local_rank=local_rank,
        world_size=world_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    trainer = Trainer(
        model,
        train_data,
        val_data,
        trainer_config,
        encoder_lr_factor=args.encoder_lr_factor if args.pretrain_mlm else 1.0,
        support_dir=args.support_dir,
    )

    if args.resume and checkpoint is not None:
        trainer.restore_from_checkpoint(
            checkpoint, reset_optimizer=vocab_grew or strict_load_failed
        )

    history = trainer.train(
        checkpoint_dir=args.support_dir,
        checkpoint_args=args,
        model_config=model_config,
        verbose=is_main_process(),
        on_epoch_end=lambda h: _append_history(h, "style"),
    )
    if is_main_process():
        _append_history(history, "style")

    # Test evaluation and model saving
    if is_main_process():
        # Simple test evaluation summary
        res = trainer.evaluate()
        print(
            f"\nFinal Test Accuracy: form={res['formality_accuracy']:.4f}, gender={res['gender_pragmatic_accuracy']:.4f}, gram={res['grammaticality_accuracy']:.4f}, register={res['register_accuracy']:.4f}"
        )

        save_model(
            cast(
                StyleClassifier, model if not hasattr(model, "module") else model.module
            ),
            tokenizer,
            args.output,
            model_config,
            fp16=args.fp16,
            fp8=args.fp8,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
