from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def make_lerobot_v3_dataset(
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

    episodes_meta = [
        {
            "episode_index": i,
            "length": ep.get("frame_count", fps * 5),
            "task_index": ep.get("task_index", 0),
            "success": ep.get("success", None),
        }
        for i, ep in enumerate(episodes)
    ]
    episodes_df = pd.DataFrame(episodes_meta)
    episodes_df.to_json(meta_dir / "episodes.jsonl", orient="records", lines=True)

    # Map each task_index to its task name from the episode rows, so the tasks
    # table stays consistent with the per-frame task_index values. Building this
    # from a set() instead made task_index assignment depend on hash-seed
    # ordering, which flipped task labels and made diff tests flaky.
    task_by_index: dict[int, str] = {}
    for ep in episodes:
        task_by_index[ep.get("task_index", 0)] = ep.get("task", "default_task")
    tasks_records = [
        {"task_index": i, "task": task_by_index[i]} for i in sorted(task_by_index)
    ]
    pd.DataFrame(tasks_records).to_json(meta_dir / "tasks.jsonl", orient="records", lines=True)

    all_rows = []
    for i, ep in enumerate(episodes):
        frame_count = ep.get("frame_count", fps * 5)
        task_index = ep.get("task_index", 0)
        for f in range(frame_count):
            all_rows.append({
                "episode_index": i,
                "frame_index": f,
                "task_index": task_index,
                "timestamp": f / fps,
                "index": i * frame_count + f,
            })

    df = pd.DataFrame(all_rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, data_dir / "data.parquet")

    return root
