from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from episodevault.diff.engine import diff as compute_diff
from episodevault.parsers.lerobot import parse as parse_lerobot
from episodevault.store.lineage_store import LineageStore
from episodevault.store.version_store import VersionStore

_console = Console(force_terminal=True)

def _resolve_store(dataset_path: Path) -> Path:
    return dataset_path / ".episodevault"


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("dataset_path")
def track(dataset_path: str) -> None:
    if "/" in dataset_path and not Path(dataset_path).exists():
        local_name = dataset_path.replace("/", "__")
        _console.print(
            f"[red]error:[/red] '{dataset_path}' looks like a HuggingFace repo ID, "
            "not a local path."
        )
        _console.print(
            f"Download first with:\n"
            f"  huggingface-cli download --repo-type dataset {dataset_path} "
            f"--local-dir ./{local_name}"
        )
        sys.exit(1)

    root = Path(dataset_path)
    if not root.exists():
        _console.print(f"[red]error:[/red] path '{dataset_path}' does not exist.")
        sys.exit(1)

    store_path = _resolve_store(root)
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / ".gitignore").write_text("*.parquet\n")
    _console.print(f"[green]Tracking[/green] {root.resolve()}")
    _console.print(f"Store initialised at {store_path}")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("-m", "--message", required=True, help="Commit message")
def commit(dataset_path: str, message: str) -> None:
    root = Path(dataset_path)
    store_path = _resolve_store(root)

    if not store_path.exists():
        _console.print(
            "[red]error:[/red] dataset not tracked. Run `episodevault track` first."
        )
        sys.exit(1)

    _console.print("Parsing dataset…")
    try:
        manifest = parse_lerobot(root)
    except (FileNotFoundError, ValueError) as exc:
        _console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    store = VersionStore(store_path)
    version_id = store.commit(manifest, message)

    _console.print(f"[green]Committed[/green] {version_id}  {message}")
    _console.print(
        f"  {manifest.total_episodes} episodes · "
        f"{len(manifest.tasks)} tasks · "
        f"{manifest.robot_type}"
    )


@cli.command(name="diff")
@click.argument("version_before")
@click.argument("version_after")
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
def diff_cmd(version_before: str, version_after: str, dataset_path: str) -> None:
    root = Path(dataset_path)
    store_path = _resolve_store(root)
    store = VersionStore(store_path)

    try:
        manifest_before = store.read_version(version_before)
        manifest_after = store.read_version(version_after)
    except KeyError as exc:
        _console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    object.__setattr__(manifest_before, "_version_id", version_before)
    object.__setattr__(manifest_after, "_version_id", version_after)

    result = compute_diff(manifest_before, manifest_after)
    _console.print(result.format())


@cli.command()
@click.argument("model_version")
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
def blame(model_version: str, dataset_path: str) -> None:
    root = Path(dataset_path)
    store_path = _resolve_store(root)
    lineage = LineageStore(store_path)
    store = VersionStore(store_path)

    dataset_version = lineage.dataset_version_for_model(model_version)
    if dataset_version is None:
        _console.print(
            f"[red]error:[/red] No training run found for model '{model_version}'. "
            "Log runs with ev.log_training_run() in your training script."
        )
        sys.exit(1)

    versions = store.list_versions()
    version_ids = [v["version_id"] for v in versions]

    if dataset_version not in version_ids:
        _console.print(
            f"[yellow]warning:[/yellow] Model '{model_version}' trained on "
            f"dataset version '{dataset_version}' which is not in the store."
        )
        sys.exit(1)

    idx = version_ids.index(dataset_version)
    _console.print(
        f"[bold]{model_version}[/bold] was trained on dataset version "
        f"[bold]{dataset_version}[/bold]"
    )

    if idx == 0:
        _console.print("No prior version to diff against.")
        return

    prior_version = version_ids[idx - 1]
    _console.print(f"Diffing {prior_version} → {dataset_version}:\n")

    manifest_before = store.read_version(prior_version)
    manifest_after = store.read_version(dataset_version)
    result = compute_diff(manifest_before, manifest_after)
    _console.print(result.format())


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
def log(dataset_path: str) -> None:
    root = Path(dataset_path)
    store_path = _resolve_store(root)
    store = VersionStore(store_path)
    versions = store.list_versions()

    if not versions:
        _console.print("No commits yet.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Version")
    table.add_column("Episodes", justify="right")
    table.add_column("Message")

    for v in reversed(versions):
        table.add_row(
            str(v["version_id"]),
            str(v["total_episodes"]),
            str(v["commit_message"]),
        )

    _console.print(table)
