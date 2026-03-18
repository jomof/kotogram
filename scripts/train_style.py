"""Supervised style classifier for Japanese sentences using Kotogram representations.

This script orchestrates the training pipeline, including argument parsing,
data loading, and calling the trainers from the kotogram.train package.
"""

import dataclasses
import glob

# MLflow logging (optional -- only available when mlflow is installed)
import importlib.util
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Union, cast

import torch
from torch import nn

from kotogram import locations
from kotogram.model import InferenceClassifier, load_model

# pylint: disable=ungrouped-imports
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train import history, paths
from train import io as train_io
from train.config import (
    TrainerConfig,
)
from train.dataset import DatasetConfig, StyleDataset
from train.io import (
    get_checkpoint_path,
    save_model,
)
from train.kc import (
    ALL_KC_FAMILIES,
    FAMILY_FEATURES,
    KcFamilyId,
    get_family_bucket_size,
    is_family_db_sourced,
    is_family_sparse,
)
from train.models import TrainingClassifier
from train.profile import (
    get_profile_dir,
    profiling_enabled,
)
from train.train_style_view import (
    FinalResults,
    TrainStyleDiagnosticsView,
    TrainStyleView,
)
from train.trainer import KCTrainer, Trainer
from train.types import KCTrainingHistory, TrainingHistory

if importlib.util.find_spec("mlflow") is not None:
    from train import mlflow_logging
    from train.artifact_uploader import create_uploader
else:
    mlflow_logging = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    create_uploader = None  # type: ignore[assignment]  # pylint: disable=invalid-name

# Global view instance for display output
_view: TrainStyleView = TrainStyleDiagnosticsView()


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
    profile_report_path = os.path.join(prof_dir, "training-profile.txt")

    _view.on_profile_report_start(prof_dir)

    files = glob.glob(os.path.join(prof_dir, "*.jsonl"))
    if not files:
        _view.on_profile_no_data()
        return

    all_entries = []
    for p in files:
        with open(p, "r", encoding="utf-8") as profile_file:
            for line in profile_file:
                if line.strip():
                    all_entries.append(json.loads(line))

    if not all_entries:
        _view.on_profile_no_data()
        return

    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get("timestamp", ""))

    # Analytics
    epochs = sorted(list(set(e.get("epoch", 0) for e in all_entries)))
    thrashing_events = [e for e in all_entries if e.get("majflt", 0) > 0]

    with open(profile_report_path, "w", encoding="utf-8") as report_file:
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

    _view.on_profile_report_complete(profile_report_path)

    # Cleanup JSONL files
    for p in files:
        os.remove(p)
    _view.on_profile_cleanup(len(files))


def cleanup_profile_if_retrain(argv_list: List[str]) -> None:
    """Delete .profile directory if --retrain is present in arguments."""
    if "--retrain" in argv_list:
        # Use get_profile_dir to ensure we clean the correct machine-specific directory
        profile_dir = get_profile_dir()
        if profile_dir and os.path.exists(profile_dir):
            _view.on_profile_dir_cleanup(profile_dir)
            shutil.rmtree(profile_dir, ignore_errors=True)
            os.makedirs(profile_dir, exist_ok=True)


if __name__ == "__main__":
    cleanup_profile_if_retrain(sys.argv)

    # Internal profiling when enabled
    from train.profile import setup_profiling

    setup_profiling()

    _view.on_script_start()
    import argparse

    parser = argparse.ArgumentParser(
        description="Train style classifier (formality + gender)"
    )
    # Config file (required - contains all configuration)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to unified config.json file"
    )
    args = parser.parse_args()

    # Load configuration
    model_config, trainer_config = TrainerConfig.load_config(args.config)

    use_mlflow = trainer_config.mlflow and mlflow_logging is not None

    # Handle report_only mode
    if trainer_config.report_only:
        generate_profile_report()
        sys.exit(0)

    # Resolve paths
    cache_dir = paths.get_cache_dir()
    data_path = os.path.join(cache_dir, "grammatic_combined.tsv")
    agrammatic_data_path = os.path.join(cache_dir, "agrammatic_combined.tsv")
    output_path = locations.get_style_output_dir()
    history_dir = paths.get_style_history_dir()

    # Epoch history logging
    history_path = os.path.join(history_dir, "training-history.tsv")

    # Determine how many epochs have already been completed
    initial_kc_epochs = 0
    initial_style_epochs = 0
    continuation_path = os.path.join(output_path, "continuation.json")
    if os.path.exists(continuation_path):
        with open(continuation_path, "r", encoding="utf-8") as cont_file:
            cont_data = json.load(cont_file)
            initial_kc_epochs = cont_data.get("kc_epochs_trained", 0)
            initial_style_epochs = cont_data.get("style_epochs_trained", 0)

    # Clear history if starting fresh (retrain mode, and not just labeling)
    if trainer_config.retrain and not trainer_config.label_only:
        history.clear_history(history_path)
        if os.path.exists(continuation_path):
            os.remove(continuation_path)
        initial_kc_epochs = 0
        initial_style_epochs = 0

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

        base_epoch = (
            initial_kc_epochs if phase_type == "pretrain-kc" else initial_style_epochs
        )
        current_epoch = base_epoch + idx + 1

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
            if use_mlflow and mlflow_logging:
                mlflow_logging.log_kc_epoch(current_epoch, metrics)
        else:
            event = history.StyleEpochEvent(epoch=current_epoch, metrics=metrics)

        history.append_event(history_path, event)

    def _export_model(model_to_save: Union[nn.Module, InferenceClassifier]) -> None:
        """Save the model weights and config to the output directory."""
        # Ensure we use the base model (unwrap DDP if present)
        inner_model = model_to_save
        if hasattr(inner_model, "module"):
            inner_model = cast(nn.Module, inner_model.module)

        os.makedirs(output_path, exist_ok=True)

        # Create __init__.py to make the model directory a valid Python package
        # This is required for 'kotogram.model_data' redirection in pyproject.toml
        init_path = os.path.join(output_path, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8"):
                pass

        save_model(
            cast(InferenceClassifier, inner_model),
            output_path,
            model_config,
        )

        # Save continuation info
        cont_json_path = os.path.join(output_path, "continuation.json")
        with open(cont_json_path, "w", encoding="utf-8") as cont_f_handle:
            json.dump(
                {
                    "kc_epochs_trained": kc_epochs_done,
                    "style_epochs_trained": style_epochs_done,
                },
                cont_f_handle,
                indent=2,
            )

        _view.on_model_saved(output_path)

    model: Optional[InferenceClassifier] = None
    tokenizer: Optional[Tokenizer] = None
    # pylint: disable=invalid-name
    checkpoint: Optional[Dict[str, Any]] = None
    vocab_grew = False
    loaded_from_checkpoint = False
    pending_checkpoint_path: Optional[str] = None

    # Load existing model if resuming
    # Priority: checkpoint.pt (full state) > model.pt (stripped/fp8)
    checkpoint_path = get_checkpoint_path()
    model_pt_path = os.path.join(output_path, "model.pt")
    if not trainer_config.retrain and os.path.exists(checkpoint_path):
        # Checkpoint exists - don't load model.pt, we'll load checkpoint after model init
        # The checkpoint contains full TrainingClassifier state, so we skip model.pt loading
        loaded_from_checkpoint = True
        pending_checkpoint_path = checkpoint_path
        _view.on_model_loaded(checkpoint_path)
    elif not trainer_config.retrain and os.path.exists(model_pt_path):
        model, _loaded_tokenizer = load_model(output_path)
        _view.on_model_loaded(model_pt_path)

    data_files = [data_path]
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
    elif os.path.exists(agrammatic_data_path):
        data_files.append(agrammatic_data_path)
        grammaticality_labels.append(0)

    # --- Model and Data Initialization ---
    # Load full tokenizer from cache (has all field vocabs for KC specs).
    # models/style/tokenizer.json is inference-only (surface vocab only).
    tokenizer_path = os.path.join(cache_dir_data, "vocab.json")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Critical: Tokenizer not found at {tokenizer_path}. "
            "Please run with --label first or via the wrapper script."
        )

    tokenizer = Tokenizer.load(tokenizer_path)
    _view.on_tokenizer_loaded(tokenizer_path)

    # Tokenizer and Device after config load
    device = torch.device(trainer_config.device)

    # Build KC specification (always enabled for ubiquity)
    kc_specs: Dict[KcFamilyId, int] = {}
    targets = ALL_KC_FAMILIES
    current_vocabs = tokenizer.get_vocab_sizes()

    # Compute GP vocab size from dataset (max ID + 1)
    gp_vocab_size = 0
    gp_pos_path = os.path.join(cache_dir_data, "gp_pos_ids.bin")
    gp_neg_path = os.path.join(cache_dir_data, "gp_neg_ids.bin")
    if os.path.exists(gp_pos_path):
        # Read the int32 IDs and find max
        with open(gp_pos_path, "rb") as f:
            import numpy as np

            ids = np.frombuffer(f.read(), dtype=np.int32)
            if len(ids) > 0:
                gp_vocab_size = max(gp_vocab_size, int(ids.max()) + 1)
    if os.path.exists(gp_neg_path):
        with open(gp_neg_path, "rb") as f:
            import numpy as np

            ids = np.frombuffer(f.read(), dtype=np.int32)
            if len(ids) > 0:
                gp_vocab_size = max(gp_vocab_size, int(ids.max()) + 1)

    for fid in targets:
        # DB-sourced families need special handling based on type
        if is_family_db_sourced(fid):
            from train.kc import (
                KcBertFamily,
                KcDbMultilabelFamily,
                KcPnuFamily,
                KcReconFamily,
                get_family,
            )

            family_def = get_family(fid)
            if isinstance(family_def, KcBertFamily):
                # BERT cloze: predict surface token, needs surface vocab size
                kc_specs[fid] = current_vocabs["surface"]
            elif isinstance(family_def, KcReconFamily):
                kc_specs[fid] = current_vocabs["surface"]
            elif isinstance(family_def, KcPnuFamily):
                # PNU families (GRAMMAR_POINT) use dynamically computed GP vocab size
                if gp_vocab_size > 0:
                    kc_specs[fid] = gp_vocab_size
            elif isinstance(family_def, KcDbMultilabelFamily):
                # Multi-label DB families (REGISTER)
                kc_specs[fid] = family_def.num_classes
            else:
                # MSE families (GENDER/FORMALITY) output a single scalar
                kc_specs[fid] = 1
            continue
        if is_family_sparse(fid):
            kc_specs[fid] = get_family_bucket_size(fid)
        else:
            fname = FAMILY_FEATURES[fid]
            kc_specs[fid] = current_vocabs[fname]

    # Always update with computed KC specs to ensure trainers have full configuration
    trainer_config = dataclasses.replace(trainer_config, kc_target_specs=kc_specs)

    # Initialize model if not already loaded, OR upgrade if loaded model is not WithKC
    if model is None:
        model = TrainingClassifier(model_config, kc_target_specs=kc_specs)
        # If we have a pending checkpoint, load it now that model is properly initialized
        if pending_checkpoint_path is not None:
            checkpoint_state = torch.load(pending_checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint_state, strict=True)
    elif not isinstance(model, TrainingClassifier):
        # Ubiquity: Upgrade base InferenceClassifier to TrainingClassifier
        # This handles cases where we resume from a checkpoint that matches the base
        # InferenceClassifier structure (e.g. stripped checkpoints).
        _view.on_model_upgrade()
        new_model = TrainingClassifier(model_config, kc_target_specs=kc_specs)
        # Load weights from base model. strict=False is required because
        # the base model lacks kc_decoders, which the new model has.

        keys = new_model.load_state_dict(model.state_dict(), strict=False)
        # Verify that only kc_decoders keys are missing in the source
        non_kc_missing = [
            k for k in keys.missing_keys if not k.startswith("kc_decoders.")
        ]
        if non_kc_missing:
            raise RuntimeError(
                f"Validation failed during model upgrade. Missing keys: {non_kc_missing}"
            )
        if keys.unexpected_keys:
            raise RuntimeError(
                f"Validation failed during model upgrade. Unexpected keys: {keys.unexpected_keys}"
            )
        model = new_model

    # Load pretrained chiVe surface vectors for fresh models (not resuming)
    if not loaded_from_checkpoint:
        from train.chive import (
            get_chive_cache_path,
            load_chive_for_vocab,
            load_pretrained_surface,
        )

        chive_cache = get_chive_cache_path()
        if os.path.exists(chive_cache):
            surface_vocab = tokenizer.field_vocabs.get("surface", {})
            chive_vectors = load_chive_for_vocab(surface_vocab)
            freeze = not trainer_config.unfreeze_surface
            n_loaded = load_pretrained_surface(
                model.embedding, chive_vectors, freeze=freeze
            )
            _view.on_chive_loaded(n_loaded, freeze)
        else:
            _view.on_chive_cache_missing()

    # Generate and display architecture report (uses slim model to show export size)
    from train.architecture_report import generate_architecture_report

    slim_model = InferenceClassifier(model_config)
    arch_report = generate_architecture_report(
        slim_model, model_name="InferenceClassifier"
    )
    _view.on_architecture_report(arch_report)

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
            sample_ratio=trainer_config.sample_ratio,
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
        # Save full tokenizer back to cache for resumption
        train_io.save_tokenizer(
            tokenizer,
            os.path.join(cache_dir_data, "vocab.json"),
        )
        # Save inference-only tokenizer to output dir for deployment
        train_io.save_tokenizer(
            tokenizer,
            os.path.join(locations.get_style_output_dir(), "tokenizer.json"),
            inference_only=True,
        )

    if trainer_config.label_only:
        sys.exit(0)

    # Scale learning rate inversely with sample_ratio
    # 100% training set = base LR, smaller percentages = proportionally higher LR
    # This compensates for fewer gradient samples per epoch when training on subsets
    sample_ratio = trainer_config.sample_ratio
    if sample_ratio < 1.0:
        lr_scale = 1.0 / sample_ratio
        scaled_lr = trainer_config.learning_rate * lr_scale
        _view.on_lr_scaled(
            trainer_config.learning_rate,
            lr_scale,
            scaled_lr,
            sample_ratio,
        )
        trainer_config = dataclasses.replace(trainer_config, learning_rate=scaled_lr)

    if trainer_config.retrain or loaded_from_checkpoint:
        trainer_config = dataclasses.replace(trainer_config, freeze_encoder_epochs=0)

    # Interleaved KC + Style Training
    # KC epochs run first in each round to prevent forgetting
    kc_epochs_done = initial_kc_epochs
    style_epochs_done = initial_style_epochs

    kc_epochs_target = trainer_config.kc_epochs
    style_epochs_target = trainer_config.epochs

    # Use full dataset for KC training (grammatic + ungrammatic)
    kc_dataset = labeled_dataset

    # KCTrainer uses configuration from TrainerConfig (which was built by wrapper)
    # But we force thaw the encoder if retraining or loading from checkpoint to ensure correct initialization
    kc_config = trainer_config.kc_config

    # Propagate training temperature to model config so inference uses the same value
    object.__setattr__(model_config, "kc_temperature", kc_config.temperature_thawed)
    if trainer_config.retrain or loaded_from_checkpoint:
        kc_config = dataclasses.replace(kc_config, freeze_encoder_epochs=0)

    kc_trainer = KCTrainer(
        cast(TrainingClassifier, model),
        kc_dataset,
        trainer_config,
        dl_config=trainer_config.resolve_dataloader_config(device),
        kc_config=kc_config,
    )

    style_config = dataclasses.replace(trainer_config, grammaticality_loss_weight=0.0)

    style_trainer = Trainer(
        model,
        train_data,
        val_data,
        style_config,
        dl_config_train=trainer_config.resolve_dataloader_config(device, mode="train"),
        dl_config_val=trainer_config.resolve_dataloader_config(device, mode="val"),
        output_path=output_path,
    )

    _view.on_kc_training_info(len(kc_dataset))

    style_start = time.perf_counter()

    uploader = None
    if use_mlflow and mlflow_logging:
        run_id = mlflow_logging.start_run(
            model_config,
            trainer_config,
            config_path=args.config,
            run_name=None,
            sentence_count=int(
                (kc_dataset.labels["gram"][kc_dataset.indices] == 1).sum()
            ),
        )
        if create_uploader is not None:
            uploader = create_uploader(run_id)

    # Interleaving loop: KC first, then style, until both are done
    try:
        max_rounds = max(kc_epochs_target, style_epochs_target)
        for _ in range(max_rounds):
            # KC epoch (if remaining)
            if kc_epochs_done < kc_epochs_target:
                kc_trainer.train(
                    epochs=kc_epochs_done + 1,  # Train up to next epoch
                    on_epoch_end=lambda h: _log_epoch_event(h, "pretrain-kc"),
                    start_epoch=kc_epochs_done,
                )
                kc_epochs_done += 1
                # Save model.pt after every KC epoch as requested
                _export_model(model)
                train_io.save_checkpoint(model)
                if uploader:
                    uploader.queue_dir(output_path, "model")
                    uploader.queue_file(get_checkpoint_path(), "checkpoint")
                    uploader.queue_file(tokenizer_path, "vocab")

            # Style epoch (if remaining)
            if style_epochs_done < style_epochs_target:
                style_trainer.train(
                    epochs=style_epochs_done + 1,  # Train up to next epoch
                    on_epoch_end=lambda h: _log_epoch_event(h, "style"),
                    start_epoch=style_epochs_done,
                )
                style_epochs_done += 1
                # Save model and continuation.json after every style epoch
                _export_model(model)
                if uploader:
                    uploader.queue_dir(output_path, "model")
                    uploader.queue_file(tokenizer_path, "vocab")

            # Both done?
            if (
                kc_epochs_done >= kc_epochs_target
                and style_epochs_done >= style_epochs_target
            ):
                break

        style_end = time.perf_counter()

        if kc_trainer.history.total_loss:
            _view.on_kc_training_complete(kc_trainer.history.total_loss[-1])
        if style_trainer.history.train_loss:
            _view.on_style_training_complete(style_trainer.history.train_loss[-1])

        # Test evaluation and model saving
        res, _worst_samples = style_trainer.evaluate()
        _view.on_final_results(
            FinalResults(
                formality_accuracy=res.formality_accuracy,
                gender_accuracy=res.gender_accuracy,
                grammaticality_accuracy=res.grammaticality_accuracy,
            )
        )

        # Final model export
        if model:
            _export_model(model)
            if uploader:
                uploader.queue_dir(output_path, "model")
                uploader.queue_file(tokenizer_path, "vocab")

        # Final timing report
        _view.on_timing_summary(style_end - style_start)

        # Auto-generate report and cleanup if profiling was enabled
        if profiling_enabled() and not trainer_config.report_only:
            generate_profile_report()
    finally:
        if uploader:
            uploader.drain()
        if use_mlflow and mlflow_logging:
            mlflow_logging.end_run()
