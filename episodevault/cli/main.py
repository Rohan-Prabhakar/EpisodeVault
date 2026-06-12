from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from episodevault.diff.engine import detect_anomalies
from episodevault.diff.engine import diff as compute_diff
from episodevault.parsers.lerobot import parse as parse_lerobot
from episodevault.parsers.lerobot import parse_hub as parse_lerobot_hub
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
    """Initialize version tracking for a local LeRobot dataset."""
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
    """Snapshot the current episode manifest as a new version."""
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
@click.option("--html", "html_out", type=click.Path(), default=None,
              help="Write a self-contained HTML report to this path.")
def diff_cmd(version_before: str, version_after: str, dataset_path: str,
             html_out: str | None) -> None:
    """Compare two committed versions — task shifts, quality deltas, regression hints."""
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

    if html_out:
        report = result.to_html(
            detect_anomalies(manifest_after),
            versions=store.list_versions(),
        )
        Path(html_out).write_text(report, encoding="utf-8")
        _console.print(f"\n[green]HTML report written to[/green] {html_out}")


@cli.command(name="diff-hub")
@click.argument("version_local")
@click.argument("repo_id")
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
@click.option("--revision", default=None, help="Hub branch, tag, or commit SHA.")
@click.option("--html", "html_out", type=click.Path(), default=None,
              help="Write a self-contained HTML report to this path.")
def diff_hub_cmd(version_local: str, repo_id: str, dataset_path: str,
                 revision: str | None, html_out: str | None) -> None:
    """Diff a local version against a dataset on the HuggingFace Hub."""
    root = Path(dataset_path)
    store = VersionStore(_resolve_store(root))

    try:
        manifest_local = store.read_version(version_local)
    except KeyError as exc:
        _console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    _console.print(f"Downloading [bold]{repo_id}[/bold] from the HuggingFace Hub…")
    try:
        manifest_hub = parse_lerobot_hub(repo_id, revision=revision)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        _console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    object.__setattr__(manifest_local, "_version_id", version_local)
    object.__setattr__(manifest_hub, "_version_id", f"hub:{repo_id}")

    result = compute_diff(manifest_local, manifest_hub)
    _console.print(result.format())

    if html_out:
        report = result.to_html(detect_anomalies(manifest_hub))
        Path(html_out).write_text(report, encoding="utf-8")
        _console.print(f"\n[green]HTML report written to[/green] {html_out}")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
@click.option("--version", "version_id", default=None,
              help="Inspect a committed version instead of re-parsing the dataset.")
def anomalies(dataset_path: str, version_id: str | None) -> None:
    """Flag outlier episodes to prune before training."""
    root = Path(dataset_path)

    if version_id:
        store = VersionStore(_resolve_store(root))
        try:
            manifest = store.read_version(version_id)
        except KeyError as exc:
            _console.print(f"[red]error:[/red] {exc}")
            sys.exit(1)
    else:
        try:
            manifest = parse_lerobot(root)
        except (FileNotFoundError, ValueError) as exc:
            _console.print(f"[red]error:[/red] {exc}")
            sys.exit(1)

    found = detect_anomalies(manifest)
    if not found:
        _console.print("[green]No anomalous episodes detected.[/green]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Episode")
    table.add_column("Task")
    table.add_column("Severity", justify="right")
    table.add_column("Reasons")
    for a in found:
        table.add_row(a.episode_id, a.task[:24], f"{a.severity:.2f}", "; ".join(a.reasons))

    _console.print(f"[yellow]{len(found)}[/yellow] anomalous episode(s):")
    _console.print(table)


@cli.command()
@click.argument("model_version")
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
def blame(model_version: str, dataset_path: str) -> None:
    """Show which dataset version a model was trained on, and diff it against the prior."""
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
@click.option("--html", "html_out", type=click.Path(), default=None,
              help="Write a self-contained HTML report for the selected diff.")
def tree(dataset_path: str, html_out: str | None) -> None:
    """Render version history as a tree, then optionally jump to a diff."""
    import datetime
    root = Path(dataset_path)
    store = VersionStore(_resolve_store(root))
    versions = store.list_versions()

    if not versions:
        _console.print("No commits yet.")
        return

    version_ids = [v["version_id"] for v in versions]

    t = Tree(f"[bold]{root.name}[/bold]")
    for v in versions:
        ts = datetime.datetime.fromtimestamp(
            v["committed_at"]
        ).strftime("%Y-%m-%d %H:%M")
        t.add(
            f"[green]{v['version_id']}[/green]  "
            f"{v['commit_message']}  "
            f"[dim]({v['total_episodes']} eps · {ts})[/dim]"
        )
    _console.print(t)

    if len(versions) < 2:
        return

    _console.print(
        "\n[dim]Diff two versions? Enter e.g. [bold]v1.0 v2.0[/bold]"
        " — or press Enter to skip.[/dim]"
    )
    raw = click.prompt("", default="", show_default=False).strip()
    if not raw:
        return

    parts = raw.split()
    if len(parts) != 2:
        _console.print("[red]error:[/red] enter exactly two version IDs.")
        return

    va, vb = parts
    for vid in (va, vb):
        if vid not in version_ids:
            _console.print(
                f"[red]error:[/red] '{vid}' not found. "
                f"Available: {', '.join(version_ids)}"
            )
            return

    try:
        manifest_a = store.read_version(va)
        manifest_b = store.read_version(vb)
    except KeyError as exc:
        _console.print(f"[red]error:[/red] {exc}")
        return

    object.__setattr__(manifest_a, "_version_id", va)
    object.__setattr__(manifest_b, "_version_id", vb)

    result = compute_diff(manifest_a, manifest_b)
    _console.print()
    _console.print(result.format())

    if html_out:
        report = result.to_html(
            detect_anomalies(manifest_b),
            versions=versions,
        )
        Path(html_out).write_text(report, encoding="utf-8")
        _console.print(f"\n[green]HTML report written to[/green] {html_out}")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True), default=".")
def log(dataset_path: str) -> None:
    """List all committed versions in chronological order."""
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
