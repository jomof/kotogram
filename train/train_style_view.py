"""View interfaces for train_style orchestration scripts.

This module provides the Model-View separation for training output. All display
logic for ./train_style (wrapper) and scripts/train_style.py (inner script)
should go through these interfaces.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from train.architecture_report import ArchitectureReport, format_count, format_size
from train.display import console


@dataclass
class PreprocessingResult:
    """Result of preprocessing phase for display."""

    dataset_size: int
    cached: bool
    duration_s: float


@dataclass
class FinalResults:
    """Final evaluation results for display.

    Note: Register accuracy is handled by KC trainer, not style trainer.
    """

    formality_accuracy: float
    gender_accuracy: float
    grammaticality_accuracy: float


# pylint: disable=too-many-public-methods,too-many-positional-arguments,too-many-locals
class TrainStyleView(Protocol):
    """Interface for train_style visualization and logging.

    This Protocol defines all display callbacks for the training orchestration.
    The wrapper (./train_style) and inner script (scripts/train_style.py) call
    these methods instead of using print() directly.
    """

    # --- Configuration Display ---
    def on_config_banner(
        self,
        corpus_db: str,
        output_dir: str,
        epochs: Optional[str],
        learning_rate: float,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        formality_weight: Optional[float],
        gender_weight: Optional[float],
        grammaticality_weight: Optional[float],
        kc_epochs: Optional[str],
        kc_k: int,
        kc_topk: int,
        kc_freeze_str: str,
        kc_sparsity_str: str,
        kc_target_spill_str: str,
        retrain: bool,
        label_only: bool,
        percent: Optional[float],
    ) -> None:
        """Display the training configuration banner at startup."""

    def on_effective_config(
        self,
        num_devices: int,
        device_type: str,
        micro_batch: int,
        grad_accum: int,
        effective_batch: int,
        launcher: str,
        kc_epochs: Optional[str],
        kc_freeze_str: str,
        kc_k: int,
        kc_topk: int,
    ) -> None:
        """Display the effective training configuration."""

    # --- Preprocessing ---
    def on_preprocessing_start(self) -> None:
        """Display preprocessing phase header."""

    def on_preprocessing_cached(self) -> None:
        """Notify that cached labels are being used."""

    def on_preprocessing_db_source(self) -> None:
        """Notify that corpus.db is being used as source."""

    def on_preprocessing_complete(self, result: PreprocessingResult) -> None:
        """Display preprocessing completion with stats."""

    def on_preprocessing_error(self, message: str) -> None:
        """Display preprocessing error message."""

    # --- Tokenizer ---
    def on_tokenizer_staged(self, path: str) -> None:
        """Notify tokenizer was staged to output location."""

    def on_tokenizer_created(self, path: str) -> None:
        """Notify fresh tokenizer was created."""

    def on_tokenizer_loaded(self, path: str) -> None:
        """Notify tokenizer was loaded."""

    # --- Configuration I/O ---
    def on_config_resuming(self, path: str) -> None:
        """Notify config is being loaded for resume."""

    def on_config_saved(self, path: str) -> None:
        """Notify config was saved."""

    def on_config_unchanged(self, path: str) -> None:
        """Notify config was unchanged (no write needed)."""

    def on_config_json(self, config_dict: Dict[str, Any]) -> None:
        """Display config as JSON (for --show-config)."""

    # --- Training Phases ---
    def on_training_start(self) -> None:
        """Notify training is starting."""

    def on_training_complete(self, output_dir: str, log_path: str) -> None:
        """Display training completion with paths."""

    def on_training_error(self, message: str) -> None:
        """Display training error message."""

    def on_training_duration(self, duration_s: float) -> None:
        """Display training duration."""

    # --- Confusion Phase ---

    # --- Profiling ---
    def on_profiling_enabled(self, profile_dir: str) -> None:
        """Notify profiling is enabled."""

    def on_profile_report_start(self, profile_dir: str) -> None:
        """Notify profile report generation is starting."""

    def on_profile_report_complete(self, report_path: str) -> None:
        """Notify profile report was written."""

    def on_profile_cleanup(self, count: int) -> None:
        """Notify profile files were cleaned up."""

    def on_profile_no_data(self) -> None:
        """Notify no profile data was found."""

    def on_profile_dir_cleanup(self, profile_dir: str) -> None:
        """Notify profile directory is being cleaned for retrain."""

    # --- Inner Script Events ---
    def on_script_start(self) -> None:
        """Notify inner training script is starting."""

    def on_model_upgrade(self) -> None:
        """Notify model is being upgraded to TrainingClassifier."""

    def on_model_loaded(self, path: str) -> None:
        """Notify model was loaded from exported model.pt."""

    def on_lr_scaled(
        self, base_lr: float, scale: float, scaled_lr: float, sample_ratio: float
    ) -> None:
        """Display learning rate scaling information."""

    def on_kc_training_info(self, sentence_count: int) -> None:
        """Display KC training dataset info."""

    def on_kc_training_complete(self, final_loss: float) -> None:
        """Display KC training completion."""

    def on_style_training_complete(self, final_loss: float) -> None:
        """Display style training completion."""

    def on_final_results(self, results: FinalResults) -> None:
        """Display final evaluation results."""

    def on_model_saved(self, output_dir: str) -> None:
        """Notify model was saved."""

    def on_timing_summary(self, style_duration_s: float) -> None:
        """Display timing summary."""

    # --- Label Only Mode ---
    def on_label_only_exit(self) -> None:
        """Notify exiting after label-only mode."""

    # --- Architecture Report ---
    def on_architecture_report(self, report: ArchitectureReport) -> None:
        """Display model architecture report."""


# pylint: disable=too-many-public-methods,too-many-positional-arguments,too-many-locals
class TrainStyleDiagnosticsView(TrainStyleView):
    """Default implementation with rich console output for ML engineers."""

    def _header(self, char: str = "=", width: int = 56) -> None:
        """Print a separator line."""
        console.print(char * width)

    def _phase_header(self, title: str) -> None:
        """Print a phase header."""
        self._header("=", 46)
        console.print(f"[dim]{title}[/dim]")
        self._header("=", 46)

    # --- Configuration Display ---
    def on_config_banner(
        self,
        corpus_db: str,
        output_dir: str,
        epochs: Optional[str],
        learning_rate: float,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        formality_weight: Optional[float],
        gender_weight: Optional[float],
        grammaticality_weight: Optional[float],
        kc_epochs: Optional[str],
        kc_k: int,
        kc_topk: int,
        kc_freeze_str: str,
        kc_sparsity_str: str,
        kc_target_spill_str: str,
        retrain: bool,
        label_only: bool,
        percent: Optional[float],
    ) -> None:
        self._header("=", 56)
        console.print(
            "[bold]Style Classifier Training (Formality + Gender + Grammaticality)[/bold]"
        )
        self._header("=", 56)

        console.print(f"[dim]Source Data:    {corpus_db}[/dim]")
        console.print(f"Output:         {output_dir}")

        if epochs is not None:
            console.print(f"Epochs:         {epochs}")
        else:
            # We no longer use 'checkpoint' terminology
            console.print("Epochs:         (default or restored)")

        console.print(f"Learning rate:  {learning_rate}")
        console.print(f"Model dim:      {embed_dim}")
        console.print(f"Hidden dim:     {hidden_dim}")
        console.print(f"Num layers:     {num_layers}")
        console.print(f"Num heads:      {num_heads}")
        console.print(f"Formality wt:   {formality_weight}")
        console.print(f"Gender wt:      {gender_weight}")
        console.print(f"Grammatic wt:   {grammaticality_weight}")

        if kc_epochs is not None:
            console.print(f"KC pretrain:    {kc_epochs} epochs")
            console.print(f"KC K:           {kc_k}")
            console.print(f"KC Top-k:       {kc_topk}")
            console.print(f"KC Freeze:      {kc_freeze_str}")
            console.print(f"KC Sparsity:    {kc_sparsity_str}")
            console.print(f"KC Target Spill: {kc_target_spill_str}")
        else:
            console.print("KC pretrain:    (default/saved)")

        if retrain:
            console.print("Retrain:        from scratch")
        if label_only:
            console.print("Action:         Preprocessing/Labeling only")
        if percent:
            console.print(f"Data usage:     {percent}%")

        self._header("=", 46)
        console.print()

    def on_effective_config(
        self,
        num_devices: int,
        device_type: str,
        micro_batch: int,
        grad_accum: int,
        effective_batch: int,
        launcher: str,
        kc_epochs: Optional[str],
        kc_freeze_str: str,
        kc_k: int,
        kc_topk: int,
    ) -> None:
        self._header("-", 46)
        console.print("[dim]Effective Training Configuration:[/dim]")
        self._header("-", 46)
        console.print(f"[dim]  Devices:          {num_devices} ({device_type})[/dim]")
        console.print(f"[dim]  Micro batch:      {micro_batch}[/dim]")
        console.print(f"[dim]  Grad accum:       {grad_accum}[/dim]")
        console.print(f"[dim]  Effective batch:  {effective_batch}[/dim]")
        console.print(f"[dim]  Launcher:         {launcher}[/dim]")
        if kc_epochs is not None:
            console.print(
                f"[dim]  KC Epochs:        {kc_epochs} (freeze: {kc_freeze_str})[/dim]"
            )
            console.print(f"[dim]  KC Vocab (K):     {kc_k}[/dim]")
            console.print(f"[dim]  KC Top-k:         {kc_topk}[/dim]")
        else:
            console.print(
                f"[dim]  KC Epochs:        (default/saved) (freeze: {kc_freeze_str})[/dim]"
            )
        self._header("-", 46)
        console.print()

    # --- Preprocessing ---
    def on_preprocessing_start(self) -> None:
        self._phase_header("Running Preprocessing Phase...")

    def on_preprocessing_cached(self) -> None:
        console.print("[dim]Using cached labels (V2 Binary)[/dim]")

    def on_preprocessing_db_source(self) -> None:
        console.print(
            "[dim]Executing preprocessing script (Source: corpus.db)...[/dim]"
        )

    def on_preprocessing_complete(self, result: PreprocessingResult) -> None:
        if result.cached:
            console.print(
                f"[dim]Preprocessing skipped (using cached labels). "
                f"Took {result.duration_s:.1f}s[/dim]"
            )
        else:
            console.print(
                f"[dim]Preprocessing script took {result.duration_s:.1f}s[/dim]"
            )
        console.print(f"[dim]Dataset size: {result.dataset_size} samples[/dim]")
        console.print("[dim]Preprocessing complete.[/dim]")

    def on_preprocessing_error(self, message: str) -> None:
        console.print(f"[bold red]ERROR:[/bold red] {message}")

    # --- Tokenizer ---
    def on_tokenizer_staged(self, path: str) -> None:
        console.print(f"[dim]Staged tokenizer to {path}[/dim]")

    def on_tokenizer_created(self, path: str) -> None:
        console.print(f"Creating fresh tokenizer at {path}")

    def on_tokenizer_loaded(self, path: str) -> None:
        console.print(f"Loaded tokenizer from {path}")

    # --- Configuration I/O ---
    def on_config_resuming(self, path: str) -> None:
        console.print(f"[cyan]Resuming: Loading ModelConfig from {path}[/cyan]")

    def on_config_saved(self, path: str) -> None:
        console.print(f"[green]Orchestrated configuration saved to: {path}[/green]")

    def on_config_unchanged(self, path: str) -> None:
        console.print(f"[dim]Configuration unchanged: {path}[/dim]")

    def on_config_json(self, config_dict: Dict[str, Any]) -> None:
        import json

        # Use plain print for machine-parseable output (--show-config)
        # Console.print adds ANSI codes that break JSON parsing
        print(json.dumps(config_dict, indent=2))

    # --- Training Phases ---
    def on_training_start(self) -> None:
        console.print("[dim]Starting training run...[/dim]")

    def on_training_complete(self, output_dir: str, log_path: str) -> None:
        console.print()
        self._header("=", 46)
        console.print("[bold]Training complete![/bold]")
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")
        console.print(f"[dim]Training log:   {log_path}[/dim]")
        self._header("=", 46)
        console.print()

    def on_training_error(self, message: str) -> None:
        console.print(f"[bold red]Training failed![/bold red] {message}")

    def on_training_duration(self, duration_s: float) -> None:
        console.print(f"[dim]Training run took {duration_s:.1f}s[/dim]")

    # --- Profiling ---
    def on_profiling_enabled(self, profile_dir: str) -> None:
        console.print(
            f"[dim]Profiling enabled. Results will be written to: {profile_dir}[/dim]"
        )

    def on_profile_report_start(self, profile_dir: str) -> None:
        console.print(f"[dim]Generating profile report from {profile_dir}...[/dim]")

    def on_profile_report_complete(self, report_path: str) -> None:
        console.print(f"[dim]Report written to {report_path}[/dim]")

    def on_profile_cleanup(self, count: int) -> None:
        console.print(f"[dim]Cleaned up {count} .jsonl profile files.[/dim]")

    def on_profile_no_data(self) -> None:
        console.print("[dim]No .jsonl profile files found.[/dim]")

    def on_profile_dir_cleanup(self, profile_dir: str) -> None:
        console.print(f"[dim]Cleaning up profile directory: {profile_dir}[/dim]")

    # --- Inner Script Events ---
    def on_script_start(self) -> None:
        console.print("[dim]Starting training script...[/dim]")

    def on_model_upgrade(self) -> None:
        console.print("[dim]Upgrading loaded model to TrainingClassifier...[/dim]")

    def on_model_loaded(self, path: str) -> None:
        console.print(f"[cyan]Resuming: Loaded model weights from {path}[/cyan]")

    def on_lr_scaled(
        self, base_lr: float, scale: float, scaled_lr: float, sample_ratio: float
    ) -> None:
        console.print(
            f"[dim]Scaling learning rate: {base_lr:.2e} × {scale:.2f} = {scaled_lr:.2e} "
            f"(for {sample_ratio:.1%} sample)[/dim]"
        )

    def on_kc_training_info(self, sentence_count: int) -> None:
        console.print(
            f"[dim]KC training using {sentence_count} grammatical sentences (full dataset)[/dim]"
        )

    def on_kc_training_complete(self, final_loss: float) -> None:
        console.print(
            f"[bold]KC Pretraining finished.[/bold] Final loss: {final_loss:.4f}"
        )

    def on_style_training_complete(self, final_loss: float) -> None:
        console.print(
            f"[bold]Style Training finished.[/bold] Final loss: {final_loss:.4f}"
        )

    def on_final_results(self, results: FinalResults) -> None:
        self._header("-", 34)
        console.print("[bold]Final Test Results:[/bold]")
        console.print(
            f"  Accuracy: form={results.formality_accuracy:.4f}, "
            f"gender={results.gender_accuracy:.4f}, "
            f"gram={results.grammaticality_accuracy:.4f}"
        )
        self._header("-", 34)

    def on_model_saved(self, output_dir: str) -> None:
        console.print(f"[dim]Model saved to: {output_dir}[/dim]")

    def on_timing_summary(self, style_duration_s: float) -> None:
        self._header("-", 34)
        console.print("[dim]Performance Summary:[/dim]")
        self._header("-", 34)
        console.print(f"[dim]  Style Training: {style_duration_s:.1f}s[/dim]")
        self._header("-", 34)

    # --- Label Only Mode ---
    def on_label_only_exit(self) -> None:
        console.print("[dim]Labeling only requested. Exiting.[/dim]")

    # --- Architecture Report ---
    def on_architecture_report(self, report: ArchitectureReport) -> None:
        from rich import box
        from rich.table import Table

        # Build tree structure to determine ├── vs └── and │ continuation lines
        # Group layers by parent path to find siblings
        all_names = [layer.name for layer in report.layers]

        def get_parent(name: str) -> str:
            """Get parent path from a dotted module name."""
            parts = name.rsplit(".", 1)
            return parts[0] if len(parts) > 1 else ""

        def is_last_sibling(name: str, all_names: list[str]) -> bool:
            """Check if this module is the last sibling under its parent."""
            parent = get_parent(name)
            # Find all siblings (same parent, same depth)
            siblings = [n for n in all_names if get_parent(n) == parent]
            return name == siblings[-1] if siblings else True

        def get_tree_prefix(name: str, all_names: list[str]) -> str:
            """Generate tree-style prefix like ├── or └── with continuation lines."""
            if "." not in name:
                # Top-level module, no prefix
                return ""

            parts = name.split(".")
            prefix_parts: list[str] = []

            # Build prefix for each level except the last (which gets ├── or └──)
            for depth_idx in range(len(parts) - 1):
                # Reconstruct the ancestor path at this depth
                ancestor_path = ".".join(parts[: depth_idx + 1])
                # Check if this ancestor is the last sibling at its level
                if is_last_sibling(ancestor_path, all_names):
                    prefix_parts.append("    ")  # No continuation line
                else:
                    prefix_parts.append("│   ")  # Continuation line

            # Add the final connector for this node
            if is_last_sibling(name, all_names):
                prefix_parts.append("└── ")
            else:
                prefix_parts.append("├── ")

            return "".join(prefix_parts)

        # Calculate display names with tree prefixes
        display_names: list[str] = []
        type_names: list[str] = []
        for layer in report.layers:
            # Get just the last part of the name (the module's own name)
            name_parts = layer.name.split(".")
            short_name = name_parts[-1]
            tree_prefix = get_tree_prefix(layer.name, all_names)
            display_names.append(f"{tree_prefix}{short_name}")
            type_names.append(layer.module_type)

        # Dynamic column widths - use max length, with minimums
        col_layer = max(20, max(len(n) for n in display_names) if display_names else 20)
        col_type = max(10, max(len(t) for t in type_names) if type_names else 10)
        col_dims = 12
        col_params = 8
        col_size = 10

        console.print()

        # Create table with horizontal lines only (no vertical separators)
        table = Table(
            title=f"Model Architecture: {report.model_name}",
            title_style="dim",
            show_header=True,
            header_style="dim purple",
            box=box.HORIZONTALS,
            border_style="dim",
            padding=(0, 1),
        )

        # Define columns
        table.add_column("Layer", style="dim", width=col_layer, no_wrap=True)
        table.add_column("Type", style="dim", width=col_type, no_wrap=True)
        table.add_column("Dims", style="dim", justify="right", width=col_dims)
        table.add_column("Params", style="dim", justify="right", width=col_params)
        table.add_column("Size", style="dim", justify="right", width=col_size)

        # Display each layer
        for idx, layer in enumerate(report.layers):
            name_display = display_names[idx]

            # Format dimensions
            if layer.input_dim > 0 and layer.output_dim > 0:
                dims_str = f"{layer.input_dim}→{layer.output_dim}"
            elif layer.output_dim > 0:
                dims_str = f"→{layer.output_dim}"
            else:
                dims_str = "-"

            params_str = format_count(layer.param_count)
            size_str = format_size(layer.size_bytes)

            # Color only top-level modules (depth 0) based on trainer role
            # Children are dim (reduces visual noise, focuses on architecture)
            if layer.depth == 0:
                if layer.trainer_role == "shared":
                    style = "dim cyan"
                elif layer.trainer_role == "kc":
                    style = "dim magenta"
                elif layer.trainer_role == "style":
                    style = "dim green"
                else:
                    style = "dim"
            else:
                style = "dim"

            table.add_row(
                name_display,
                layer.module_type,
                dims_str,
                params_str,
                size_str,
                style=style,
            )

        # Add totals row to table (below separator)
        table.add_section()
        table.add_row(
            "",
            "",
            "",
            f"[bold]{format_count(report.total_params)}[/bold]",
            f"[bold]{format_size(report.total_size_bytes)}[/bold]",
            style="",
        )

        console.print(table)
        # Color legend (compact single line)
        console.print(
            "[dim cyan]■[/dim cyan] Shared input  "
            "[dim magenta]■[/dim magenta] KC input  "
            "[dim green]■[/dim green] Style input"
        )
        console.print()


# Explicitly reference methods for static analysis tools.
# These are called dynamically via _view.on_xxx() in ./train_style and scripts/train_style.py
# pylint: disable=pointless-statement
TrainStyleView.on_config_banner
TrainStyleView.on_effective_config
TrainStyleView.on_preprocessing_start
TrainStyleView.on_preprocessing_cached
TrainStyleView.on_preprocessing_db_source
TrainStyleView.on_preprocessing_complete
TrainStyleView.on_preprocessing_error
TrainStyleView.on_tokenizer_staged
TrainStyleView.on_tokenizer_created
TrainStyleView.on_config_resuming
TrainStyleView.on_config_saved
TrainStyleView.on_config_unchanged
TrainStyleView.on_config_json
TrainStyleView.on_training_start
TrainStyleView.on_training_complete
TrainStyleView.on_training_error
TrainStyleView.on_training_duration
TrainStyleView.on_profiling_enabled
TrainStyleView.on_label_only_exit
TrainStyleView.on_architecture_report
