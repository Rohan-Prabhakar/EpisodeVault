from __future__ import annotations

from pathlib import Path
from typing import Any

from episodevault.store.lineage_store import LineageStore


def log_training_run(
    model_version: str,
    dataset_version: str,
    dataset_path: str | Path = ".",
    framework: str = "lerobot",
    **extras: Any,
) -> None:
    store_path = Path(dataset_path) / ".episodevault"
    lineage = LineageStore(store_path)
    lineage.log_training_run(
        model_version=model_version,
        dataset_version=dataset_version,
        framework=framework,
        **extras,
    )
