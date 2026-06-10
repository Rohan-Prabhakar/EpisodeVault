from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from episodevault.models import DatasetManifest, EpisodeManifest, EpisodeQuality

_SCHEMA = pa.schema([
    pa.field("version_id", pa.string()),
    pa.field("commit_message", pa.string()),
    pa.field("committed_at", pa.float64()),
    pa.field("dataset_id", pa.string()),
    pa.field("total_episodes", pa.int64()),
    pa.field("total_frames", pa.int64()),
    pa.field("fps", pa.int64()),
    pa.field("robot_type", pa.string()),
    pa.field("modalities", pa.list_(pa.string())),
    pa.field("tasks", pa.list_(pa.string())),
    pa.field("format_version", pa.string()),
    pa.field("episode_id", pa.string()),
    pa.field("task", pa.string()),
    pa.field("duration_s", pa.float64()),
    pa.field("frame_count", pa.int64()),
    pa.field("camera_sync_score", pa.float64()),
    pa.field("success", pa.bool_()),
    pa.field("quality", pa.string()),
    pa.field("source_hash", pa.string()),
    pa.field("raw_extras", pa.string()),
])


class VersionStore:
    def __init__(self, store_path: str | Path) -> None:
        self._root = Path(store_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._versions_path = self._root / "versions.parquet"
        self._meta_path = self._root / "meta.json"
        self._db = duckdb.connect()

        if not self._meta_path.exists():
            self._write_meta({"dataset_id": None, "version_count": 0})

    def commit(self, manifest: DatasetManifest, message: str) -> str:
        version_id = self._next_version_id()
        committed_at = time.time()
        rows = self._manifest_to_rows(manifest, version_id, message, committed_at)
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)

        if self._versions_path.exists():
            existing = pq.read_table(self._versions_path)
            combined = pa.concat_tables([existing, table])
        else:
            combined = table

        pq.write_table(combined, self._versions_path, compression="snappy")

        meta = self._read_meta()
        meta["dataset_id"] = manifest.dataset_id
        meta["version_count"] = meta.get("version_count", 0) + 1
        self._write_meta(meta)

        return version_id

    def list_versions(self) -> list[dict[str, Any]]:
        if not self._versions_path.exists():
            return []
        df = self._db.execute(
            f"""
            SELECT DISTINCT
                version_id,
                commit_message,
                committed_at,
                total_episodes,
                total_frames,
                dataset_id
            FROM read_parquet('{self._versions_path}')
            ORDER BY committed_at ASC
            """
        ).df()
        return df.to_dict(orient="records")

    def read_version(self, version_id: str) -> DatasetManifest:
        if not self._versions_path.exists():
            raise KeyError(f"Version '{version_id}' not found — store is empty.")
        df = self._db.execute(
            f"""
            SELECT * FROM read_parquet('{self._versions_path}')
            WHERE version_id = '{version_id}'
            """
        ).df()
        if df.empty:
            raise KeyError(f"Version '{version_id}' not found.")
        return self._rows_to_manifest(df)

    def latest_version_id(self) -> str | None:
        versions = self.list_versions()
        if not versions:
            return None
        return versions[-1]["version_id"]

    def _next_version_id(self) -> str:
        meta = self._read_meta()
        n = meta.get("version_count", 0) + 1
        return f"v{n}.0"

    def _manifest_to_rows(
        self,
        manifest: DatasetManifest,
        version_id: str,
        message: str,
        committed_at: float,
    ) -> list[dict[str, Any]]:
        rows = []
        for ep in manifest.episodes:
            rows.append({
                "version_id": version_id,
                "commit_message": message,
                "committed_at": committed_at,
                "dataset_id": manifest.dataset_id,
                "total_episodes": manifest.total_episodes,
                "total_frames": manifest.total_frames,
                "fps": manifest.fps,
                "robot_type": manifest.robot_type,
                "modalities": list(manifest.modalities),
                "tasks": list(manifest.tasks),
                "format_version": manifest.format_version,
                "episode_id": ep.episode_id,
                "task": ep.task,
                "duration_s": ep.duration_s,
                "frame_count": ep.frame_count,
                "camera_sync_score": ep.camera_sync_score,
                "success": ep.success,
                "quality": ep.quality.value,
                "source_hash": ep.source_hash,
                "raw_extras": json.dumps(ep.raw_extras),
            })
        return rows

    def _rows_to_manifest(self, df: pd.DataFrame) -> DatasetManifest:
        first = df.iloc[0]
        episodes = tuple(
            EpisodeManifest(
                episode_id=row["episode_id"],
                task=row["task"],
                duration_s=row["duration_s"],
                frame_count=int(row["frame_count"]),
                fps=int(first["fps"]),
                robot_type=row["robot_type"] if "robot_type" in row else first["robot_type"],
                modalities=tuple(first["modalities"]),
                camera_sync_score=row["camera_sync_score"],
                success=row["success"] if pd.notna(row["success"]) else None,
                quality=EpisodeQuality(row["quality"]),
                source_hash=row["source_hash"],
                raw_extras=json.loads(row["raw_extras"]) if row["raw_extras"] else {},
            )
            for _, row in df.iterrows()
        )
        return DatasetManifest(
            dataset_id=first["dataset_id"],
            total_episodes=int(first["total_episodes"]),
            total_frames=int(first["total_frames"]),
            fps=int(first["fps"]),
            robot_type=first["robot_type"],
            modalities=tuple(first["modalities"]),
            tasks=tuple(first["tasks"]),
            episodes=episodes,
            format_version=first["format_version"],
        )

    def _read_meta(self) -> dict[str, Any]:
        if not self._meta_path.exists():
            return {}
        return json.loads(self._meta_path.read_text())

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self._meta_path.write_text(json.dumps(meta, indent=2))
