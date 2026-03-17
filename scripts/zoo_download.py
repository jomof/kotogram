"""Download a single MLflow run's artifacts (checkpoint, model) into project paths."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_cache_dir() -> str:
    from train.paths import get_cache_dir

    return get_cache_dir()


def _get_style_output_dir() -> str:
    from kotogram import locations

    return locations.get_style_output_dir()


def _get_style_dataset_cache_dir() -> str:
    from train.paths import get_style_dataset_cache_dir

    return get_style_dataset_cache_dir()


def download(  # pylint: disable=too-many-locals
    name_frag: str, tracking_uri: Optional[str] = None
) -> None:
    """Resolve run by name fragment and download checkpoint + model + vocab artifacts.

    - Artifact "checkpoint" (file checkpoint.pt) -> .cache/checkpoint.pt
    - Artifact "model" (directory model.pt, model.json, etc.) -> models/style/
    - Artifact "vocab" (file vocab.json) -> .cache/style_dataset/vocab.json

    Raises ValueError if name_frag does not match exactly one run.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from rich.console import Console

    from scripts import zoo_common

    console = Console()
    with console.status("[bold blue]Resolving run..."):
        run_id = zoo_common.resolve_run_by_name_fragment(
            name_frag, tracking_uri=tracking_uri
        )

    from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

    uri = tracking_uri if tracking_uri is not None else zoo_common.get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    run_name = run.info.run_name or run_id
    console.print(f"[green]✓[/green] Run [dim]{run_name}[/dim]")

    # List root artifacts to avoid downloading missing paths (and hitting exception policy)
    root_infos = client.list_artifacts(run_id, path="")
    available = {info.path.split("/")[0] for info in root_infos}

    cache_dir = _get_cache_dir()
    style_dir = _get_style_output_dir()
    style_dataset_cache = _get_style_dataset_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    os.makedirs(style_dataset_cache, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zoo_download_") as tmp:
        # Download "checkpoint" artifact -> single file checkpoint.pt
        if "checkpoint" not in available:
            console.print(
                "[yellow]Warning:[/yellow] checkpoint not available for this run; skipping."
            )
        else:
            with console.status("[bold blue]Downloading checkpoint..."):
                checkpoint_dest = client.download_artifacts(run_id, "checkpoint", tmp)
            checkpoint_src = None
            if os.path.isfile(checkpoint_dest) and checkpoint_dest.endswith(".pt"):
                checkpoint_src = checkpoint_dest
            elif os.path.isdir(checkpoint_dest):
                candidate = os.path.join(checkpoint_dest, "checkpoint.pt")
                if os.path.isfile(candidate):
                    checkpoint_src = candidate
            if checkpoint_src is None:
                raise FileNotFoundError(
                    f"Expected checkpoint.pt under artifact 'checkpoint'; got {checkpoint_dest!r}"
                )
            out_checkpoint = os.path.join(cache_dir, "checkpoint.pt")
            shutil.copy2(checkpoint_src, out_checkpoint)
            console.print(f"[green]✓[/green] checkpoint → [dim]{out_checkpoint}[/dim]")

        # Download "model" artifact -> directory contents into models/style
        if "model" not in available:
            console.print(
                "[yellow]Warning:[/yellow] model artifact not available for this run; skipping."
            )
        else:
            with console.status("[bold blue]Downloading model..."):
                model_dest = client.download_artifacts(run_id, "model", tmp)
            for name in os.listdir(model_dest):
                src = os.path.join(model_dest, name)
                dst = os.path.join(style_dir, name)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                console.print(
                    f"  [green]✓[/green] {name} → [dim]{os.path.join(style_dir, name)}[/dim]"
                )

        # Download "vocab" artifact -> .cache/style_dataset/vocab.json
        if "vocab" not in available:
            console.print(
                "[yellow]Warning:[/yellow] vocab.json not available for this run; skipping."
            )
        else:
            with console.status("[bold blue]Downloading vocab..."):
                vocab_dest = client.download_artifacts(run_id, "vocab", tmp)
            vocab_src = None
            if os.path.isfile(vocab_dest) and vocab_dest.endswith(".json"):
                vocab_src = vocab_dest
            elif os.path.isdir(vocab_dest):
                candidate = os.path.join(vocab_dest, "vocab.json")
                if os.path.isfile(candidate):
                    vocab_src = candidate
            if vocab_src is None:
                raise FileNotFoundError(
                    f"Expected vocab.json under artifact 'vocab'; got {vocab_dest!r}"
                )
            out_vocab = os.path.join(style_dataset_cache, "vocab.json")
            shutil.copy2(vocab_src, out_vocab)
            console.print(f"[green]✓[/green] vocab → [dim]{out_vocab}[/dim]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download model and checkpoint from MLflow run by name fragment."
    )
    parser.add_argument(
        "name_frag",
        help='Fragment of MLflow run name (e.g. "chive | L6"); must match exactly one run.',
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: from MLFLOW_TRACKING_URI or project default).",
    )
    args = parser.parse_args()
    download(args.name_frag, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
