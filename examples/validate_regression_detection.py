"""
End-to-end validation of EpisodeVault regression detection.

Creates two synthetic LeRobot v3 datasets (v1 and v2), commits both to a
version store, runs a semantic diff, logs a training run, and exercises the
blame command. No network access required — all data is generated in-memory.

Run:
    pip install episodevault
    python examples/validate_regression_detection.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import episodevault as ev
from episodevault.diff.engine import diff
from episodevault.parsers.lerobot import parse
from episodevault.store.lineage_store import LineageStore
from episodevault.store.version_store import VersionStore


def make_dataset(
    root: Path,
    episodes: list[dict],
    fps: int = 30,
    robot_type: str = "so101",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_dir.mkdir()
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": sum(ep.get("frame_count", fps * 5) for ep in episodes),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
            "timestamp": {"dtype": "float32", "shape": []},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))

    unique_tasks = list({ep.get("task", "default") for ep in episodes})
    task_index_map = {t: i for i, t in enumerate(unique_tasks)}

    episodes_meta = [
        {
            "episode_index": i,
            "length": ep.get("frame_count", fps * 5),
            "task_index": task_index_map.get(ep.get("task", "default"), 0),
            "success": ep.get("success", None),
        }
        for i, ep in enumerate(episodes)
    ]
    pd.DataFrame(episodes_meta).to_json(
        meta_dir / "episodes.jsonl", orient="records", lines=True
    )

    tasks_records = [{"task_index": i, "task": t} for t, i in task_index_map.items()]
    pd.DataFrame(tasks_records).to_json(
        meta_dir / "tasks.jsonl", orient="records", lines=True
    )

    all_rows = []
    for i, ep in enumerate(episodes):
        frame_count = ep.get("frame_count", fps * 5)
        task_idx = task_index_map.get(ep.get("task", "default"), 0)
        for f in range(frame_count):
            all_rows.append({
                "episode_index": i,
                "frame_index": f,
                "task_index": task_idx,
                "timestamp": round(f / fps, 6),
                "index": i * frame_count + f,
            })

    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(all_rows)),
        data_dir / "data.parquet",
    )
    return root


def section(title: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}")
    if not condition:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        section("1 — Build synthetic datasets")

        v1_episodes = [
            {"task": "kitchen_grasp", "frame_count": 150, "success": True},
            {"task": "kitchen_grasp", "frame_count": 120, "success": True},
            {"task": "kitchen_grasp", "frame_count": 140, "success": True},
            {"task": "kitchen_grasp", "frame_count": 130, "success": False},
            {"task": "bottle_open",   "frame_count": 90,  "success": True},
            {"task": "bottle_open",   "frame_count": 95,  "success": True},
            {"task": "factory_pick",  "frame_count": 80,  "success": True},
            {"task": "factory_pick",  "frame_count": 85,  "success": True},
        ]

        v2_episodes = [
            {"task": "kitchen_grasp", "frame_count": 150, "success": True},
            {"task": "bottle_open",   "frame_count": 90,  "success": True},
            {"task": "factory_pick",  "frame_count": 80,  "success": True},
            {"task": "factory_pick",  "frame_count": 85,  "success": False},
            {"task": "factory_pick",  "frame_count": 82,  "success": False},
            {"task": "factory_pick",  "frame_count": 78,  "success": False},
            {"task": "factory_pick",  "frame_count": 84,  "success": False},
            {"task": "factory_pick",  "frame_count": 81,  "success": False},
        ]

        ds_v1 = make_dataset(root / "dataset_v1", v1_episodes)
        ds_v2 = make_dataset(root / "dataset_v2", v2_episodes)
        print("  Datasets created at:")
        print(f"    {ds_v1}")
        print(f"    {ds_v2}")

        section("2 — Parse datasets")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            manifest_v1 = parse(ds_v1)
            manifest_v2 = parse(ds_v2)

        check("v1 parsed — 8 episodes", manifest_v1.total_episodes == 8)
        check("v2 parsed — 8 episodes", manifest_v2.total_episodes == 8)
        check("v1 tasks detected", "kitchen_grasp" in manifest_v1.tasks)
        check("v2 factory_pick dominant", sum(
            1 for e in manifest_v2.episodes if e.task == "factory_pick"
        ) >= 5)
        check("episode IDs formatted correctly",
              manifest_v1.episodes[0].episode_id == "episode_000000")
        check("source hashes populated",
              all(len(e.source_hash) == 16 for e in manifest_v1.episodes))
        check("sync scores in [0,1]",
              all(0.0 <= e.camera_sync_score <= 1.0 for e in manifest_v1.episodes))

        print(f"\n  v1 tasks: {manifest_v1.tasks}")
        print(f"  v2 tasks: {manifest_v2.tasks}")

        section("3 — Commit to version store")

        store_path = root / ".episodevault"
        store = VersionStore(store_path)
        lineage = LineageStore(store_path)

        vid1 = store.commit(manifest_v1, "baseline — kitchen heavy")
        vid2 = store.commit(manifest_v2, "added factory episodes")

        check("first commit is v1.0", vid1 == "v1.0")
        check("second commit is v2.0", vid2 == "v2.0")
        check("latest version is v2.0", store.latest_version_id() == "v2.0")

        recovered = store.read_version(vid1)
        check("v1 roundtrips — episode count",
              len(recovered.episodes) == len(manifest_v1.episodes))
        check("v1 roundtrips — robot type",
              recovered.robot_type == manifest_v1.robot_type)

        print(f"\n  Committed: {vid1}  '{manifest_v1.total_episodes} episodes'")
        print(f"  Committed: {vid2}  '{manifest_v2.total_episodes} episodes'")

        section("4 — Semantic diff")

        result = diff(manifest_v1, manifest_v2)

        kitchen_delta = next(
            td for td in result.task_deltas if td.task == "kitchen_grasp"
        )
        factory_delta = next(
            td for td in result.task_deltas if td.task == "factory_pick"
        )

        check("kitchen_grasp count dropped", kitchen_delta.pct_change < 0)
        check("kitchen_grasp flagged as significant drop", kitchen_delta.flagged)
        check("factory_pick count increased", factory_delta.pct_change > 0)
        check("success rate degraded",
              result.quality_delta.success_rate_after is not None
              and result.quality_delta.success_rate_before is not None
              and result.quality_delta.success_rate_after
              < result.quality_delta.success_rate_before)
        check("regression hint generated", result.regression_hint is not None)
        check("regression hint mentions kitchen_grasp",
              result.regression_hint is not None
              and "kitchen_grasp" in result.regression_hint)

        print("\n  Diff output:")
        for line in result.format().splitlines():
            print(f"    {line}")

        section("5 — Lineage and blame")

        ev.log_training_run(
            model_version="model_v2",
            dataset_version=vid2,
            dataset_path=str(root),
            framework="lerobot",
        )

        resolved = lineage.dataset_version_for_model("model_v2")
        check("blame resolves model_v2 → v2.0", resolved == "v2.0")
        check("unknown model returns None",
              lineage.dataset_version_for_model("nonexistent") is None)

        runs = lineage.list_runs()
        check("one training run logged", len(runs) == 1)
        check("run framework is lerobot", runs[0]["framework"] == "lerobot")

        print(f"\n  model_v2 trained on dataset: {resolved}")

        section("6 — Python API surface")

        check("ev.log_training_run callable", callable(ev.log_training_run))
        check("ev.parse_lerobot callable", callable(ev.parse_lerobot))
        check("ev.diff callable", callable(ev.diff))
        check("ev.VersionStore importable", ev.VersionStore is not None)
        check("ev.LineageStore importable", ev.LineageStore is not None)

        section("All checks passed")
        print("\n  EpisodeVault is working correctly.")
        print("  Next step: run against a real LeRobot dataset from HuggingFace.\n")
        print("  huggingface-cli download --repo-type dataset \\")
        print("    lerobot/aloha_static_pro_pencil --local-dir ./aloha_pencil")
        print("  episodevault track ./aloha_pencil")
        print("  episodevault commit -m 'baseline' ./aloha_pencil\n")


if __name__ == "__main__":
    main()
