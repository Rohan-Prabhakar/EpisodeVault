from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import fsspec
from episodevault.models import DatasetManifest, EpisodeManifest, EpisodeQuality

_MIN_EPISODE_DURATION_S = 0.5
_SYNC_DRIFT_THRESHOLD_S = 0.005
_SYNC_SCORE_DEGRADED = 0.95
_HASH_SAMPLE_ROWS = 10
_HF_REPO_ID_PATTERN = "/"

# --- Custom quality metrics (feature: standardized "quality" metrics) -------
QualityMetricFn = Callable[[pd.DataFrame], "float | None"]
QUALITY_METRICS: dict[str, QualityMetricFn] = {}

def register_quality_metric(name: str, fn: QualityMetricFn) -> None:
    """Register a custom per-episode quality metric, computed on parse."""
    QUALITY_METRICS[name] = fn

def _episode_action_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """Stack a list/array-valued 'action' column into an (n_frames, dim) array."""
    if "action" not in df.columns:
        return None
    try:
        arr = np.stack([np.asarray(a, dtype=np.float64) for a in df["action"]])
    except (ValueError, TypeError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    return arr

def _metric_action_smoothness(df: pd.DataFrame) -> float | None:
    """1 / (1 + mean jerk). 1.0 = perfectly smooth, →0 as actions get jerky."""
    arr = _episode_action_matrix(df)
    if arr is None:
        return None
    jerk = float(np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1).mean())
    return 1.0 / (1.0 + jerk)

def _metric_gripper_closure_rate(df: pd.DataFrame) -> float | None:
    """Fraction of frames where the gripper (last action dim) is closed (>0.5)."""
    arr = _episode_action_matrix(df)
    if arr is None:
        return None
    gripper = arr[:, -1]
    return float((gripper > 0.5).mean())

register_quality_metric("action_smoothness", _metric_action_smoothness)
register_quality_metric("gripper_closure_rate", _metric_gripper_closure_rate)

# --- Cloud-aware parsing ----------------------------------------------------

def _get_fs_and_path(dataset_uri: str) -> tuple[fsspec.AbstractFileSystem, str]:
    """Convert a URI (s3://, gs://, file://, memory://, or local path) to fsspec filesystem and path."""
    fs, path = fsspec.core.url_to_fs(dataset_uri)
    path = path.rstrip("/")
    return fs, path

def parse(dataset_uri: str | Path) -> DatasetManifest:
    """Parse a LeRobot dataset from any supported URI (local, s3://, gs://, hf://, memory://)."""
    dataset_uri = str(dataset_uri)
    _reject_hub_path(dataset_uri)
    
    fs, root_path = _get_fs_and_path(dataset_uri)
    _assert_lerobot_v3(fs, root_path)

    info = _read_info(fs, root_path)
    episodes_meta = _read_episodes_meta(fs, root_path, info)
    tasks_meta = _read_tasks_meta(fs, root_path)

    if not tasks_meta:
        tasks_meta = _infer_tasks_from_episodes(episodes_meta)

    episode_manifests = tuple(
        _build_episode_manifest(fs, root_path, ep_row, info, tasks_meta)
        for _, ep_row in episodes_meta.iterrows()
    )

    _warn_missing_success_flags(episode_manifests)

    modalities = tuple(sorted(info.get("features", {}).keys()))
    tasks = tuple(sorted({e.task for e in episode_manifests}))

    dataset_id = root_path.split("/")[-1] if "/" in root_path else root_path

    return DatasetManifest(
        dataset_id=dataset_id,
        total_episodes=info["total_episodes"],
        total_frames=info["total_frames"],
        fps=info["fps"],
        robot_type=info.get("robot_type", "unknown"),
        modalities=modalities,
        tasks=tasks,
        episodes=episode_manifests,
    )

def _download_hub_dataset(repo_id: str, revision: str | None, cache_dir: str | Path | None) -> Path:
    """Download a LeRobot dataset from the HuggingFace Hub and return its local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Diffing against the HuggingFace Hub requires the 'huggingface_hub' "
            "package. Install it with: pip install huggingface_hub"
        ) from exc

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local)

def parse_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> DatasetManifest:
    """Download a Hub-hosted LeRobot dataset and parse it into a DatasetManifest."""
    if _HF_REPO_ID_PATTERN not in repo_id:
        raise ValueError(
            f"'{repo_id}' does not look like a HuggingFace repo ID "
            "(expected the form 'owner/name')."
        )
    local_root = _download_hub_dataset(repo_id, revision, cache_dir)
    manifest = parse(str(local_root))
    object.__setattr__(manifest, "dataset_id", repo_id)
    return manifest

def _reject_hub_path(dataset_uri: str) -> None:
    """Reject HuggingFace repo IDs passed as dataset paths."""
    allowed_prefixes = ("s3://", "gs://", "az://", "file://", "memory://", "/")
    if _HF_REPO_ID_PATTERN in dataset_uri and not dataset_uri.startswith(allowed_prefixes):
        raise ValueError(
            f"'{dataset_uri}' looks like a HuggingFace repo ID, not a URI. "
            "Use 'episodevault diff-hub' for Hub datasets, or provide a full URI."
        )

def _assert_lerobot_v3(fs: fsspec.AbstractFileSystem, root_path: str) -> None:
    """Verify the dataset is LeRobot v3 format."""
    info_path = f"{root_path}/meta/info.json"
    if not fs.exists(info_path):
        raise FileNotFoundError(f"Not a LeRobot v3 dataset — missing {info_path}")
    
    with fs.open(info_path, "r") as f:
        info = json.load(f)
    
    version = info.get("codebase_version", "")
    if not version.startswith("v3"):
        raise ValueError(
            f"Expected LeRobot codebase_version v3.x, got '{version}'. "
            "Migrate with: lerobot-convert-dataset --raw-dir . --out-dir ./v3"
        )

def _read_info(fs: fsspec.AbstractFileSystem, root_path: str) -> dict[str, Any]:
    """Read meta/info.json from the dataset."""
    info_path = f"{root_path}/meta/info.json"
    with fs.open(info_path, "r") as f:
        return json.load(f)

def _read_episodes_meta(fs: fsspec.AbstractFileSystem, root_path: str, info: dict[str, Any]) -> pd.DataFrame:
    """Read episode metadata from various possible locations."""
    episodes_path = f"{root_path}/meta/episodes.jsonl"
    if fs.exists(episodes_path):
        with fs.open(episodes_path, "r") as f:
            df = pd.read_json(f, lines=True)
        if not df.empty:
            return df

    episodes_dir = f"{root_path}/meta/episodes"
    if fs.exists(episodes_dir) and fs.isdir(episodes_dir):
        parquet_files = sorted([p for p in fs.ls(episodes_dir, detail=False) if p.endswith(".parquet")])
        if parquet_files:
            frames = []
            for pf in parquet_files:
                with fs.open(pf, "rb") as f:
                    frames.append(pd.read_parquet(f))
            if frames:
                return pd.concat(frames, ignore_index=True)

    data_dir = f"{root_path}/data"
    if fs.exists(data_dir):
        parquet_files = sorted([p for p in fs.find(data_dir) if p.endswith(".parquet")])
        for pf in parquet_files:
            try:
                with fs.open(pf, "rb") as f:
                    schema = pq.read_schema(f)
                if "episode_index" not in schema.names:
                    continue
                
                cols = ["episode_index"]
                if "task_index" in schema.names:
                    cols.append("task_index")
                
                with fs.open(pf, "rb") as f:
                    df = pq.read_table(f, columns=cols).to_pandas()
                
                grouped = df.groupby("episode_index").size().reset_index(name="length")
                if "task_index" in df.columns:
                    task_mode = (
                        df.groupby("episode_index")["task_index"]
                        .agg(lambda x: x.mode().iloc[0])
                        .reset_index()
                    )
                    grouped = grouped.merge(task_mode, on="episode_index")
                
                total_eps = info.get("total_episodes", len(grouped))
                if len(grouped) >= total_eps * 0.8:
                    warnings.warn(
                        "episodes.jsonl not found — inferred episode boundaries from Parquet data. "
                        "Episode metadata (success flags, task labels) may be incomplete.",
                        UserWarning,
                        stacklevel=4,
                    )
                    return grouped
            except Exception:
                continue

    raise FileNotFoundError(
        f"Cannot locate episode metadata under {root_path}/meta. "
        "Expected meta/episodes.jsonl, meta/episodes/*.parquet, or a data/*.parquet "
        "file containing an episode_index column."
    )

def _read_tasks_meta(fs: fsspec.AbstractFileSystem, root_path: str) -> dict[int, str]:
    """Read task metadata from various possible locations."""
    tasks_jsonl = f"{root_path}/meta/tasks.jsonl"
    if fs.exists(tasks_jsonl):
        try:
            with fs.open(tasks_jsonl, "r") as f:
                tasks_df = pd.read_json(f, lines=True)
            if not tasks_df.empty and "task_index" in tasks_df.columns and "task" in tasks_df.columns:
                return dict(zip(tasks_df["task_index"], tasks_df["task"]))
        except ValueError:
            pass

    tasks_parquet = f"{root_path}/meta/tasks.parquet"
    if fs.exists(tasks_parquet):
        try:
            with fs.open(tasks_parquet, "rb") as f:
                tasks_df = pq.read_table(f).to_pandas().reset_index()
            if "task_index" in tasks_df.columns:
                task_col = next(
                    (c for c in tasks_df.columns if c not in ("task_index", "__index_level_0__")),
                    None
                )
                if task_col is None and "__index_level_0__" in tasks_df.columns:
                    task_col = "__index_level_0__"
                if task_col:
                    return dict(zip(tasks_df["task_index"], tasks_df[task_col]))
        except Exception:
            pass

    tasks_dir = f"{root_path}/meta/tasks"
    if fs.exists(tasks_dir) and fs.isdir(tasks_dir):
        try:
            parquet_files = sorted([p for p in fs.ls(tasks_dir, detail=False) if p.endswith(".parquet")])
            frames = []
            for p in parquet_files:
                with fs.open(p, "rb") as f:
                    frames.append(pd.read_parquet(f))
            if frames:
                tasks_df = pd.concat(frames, ignore_index=True)
                if "task_index" in tasks_df.columns and "task" in tasks_df.columns:
                    return dict(zip(tasks_df["task_index"], tasks_df["task"]))
        except Exception:
            pass

    return {}

def _infer_tasks_from_episodes(episodes_meta: pd.DataFrame) -> dict[int, str]:
    """Infer task labels from episode metadata if tasks.jsonl is missing."""
    if "task" in episodes_meta.columns:
        unique_tasks = episodes_meta["task"].dropna().unique()
        if len(unique_tasks) > 0:
            return {i: str(t) for i, t in enumerate(unique_tasks)}

    if "language_instruction" in episodes_meta.columns:
        unique_tasks = episodes_meta["language_instruction"].dropna().unique()
        if len(unique_tasks) > 0:
            warnings.warn(
                "tasks.jsonl not found — using language_instruction column as task labels.",
                UserWarning,
                stacklevel=3,
            )
            return {i: str(t) for i, t in enumerate(unique_tasks)}

    if "task_index" in episodes_meta.columns:
        unique_indices = episodes_meta["task_index"].dropna().unique()
        all_zero = len(unique_indices) == 1 and int(unique_indices[0]) == 0
        if all_zero:
            warnings.warn(
                "tasks.jsonl not found and all episodes share task_index=0. "
                "Task distribution analysis will show a single 'unspecified' task. "
                "Add a tasks.jsonl file to enable per-task regression analysis.",
                UserWarning,
                stacklevel=3,
            )
            return {0: "unspecified"}
        return {int(i): f"task_{int(i)}" for i in unique_indices}

    return {0: "unspecified"}

def _warn_missing_success_flags(episodes: tuple[EpisodeManifest, ...]) -> None:
    """Warn if no success flags are present in any episode."""
    if all(e.success is None for e in episodes):
        warnings.warn(
            "No success flags found in any episode. "
            "Success-rate regression analysis will be unavailable. "
            "Set success=True/False in your episode metadata for richer diffs.",
            UserWarning,
            stacklevel=3,
        )

def _build_episode_manifest(
    fs: fsspec.AbstractFileSystem,
    root_path: str,
    ep_row: pd.Series,
    info: dict[str, Any],
    tasks_meta: dict[int, str],
) -> EpisodeManifest:
    """Build an EpisodeManifest for a single episode."""
    episode_index: int = int(ep_row["episode_index"])
    frame_count: int = int(ep_row.get("length", ep_row.get("num_frames", 0)))
    fps: int = int(info["fps"])
    duration_s: float = frame_count / fps if fps > 0 else 0.0
    
    task_index = ep_row.get("task_index", None)
    if task_index is not None and int(task_index) in tasks_meta:
        task = tasks_meta[int(task_index)]
    elif "task" in ep_row and ep_row["task"] and not _is_na(ep_row["task"]):
        task = str(ep_row["task"])
    elif task_index is not None:
        task = tasks_meta.get(int(task_index), f"task_{int(task_index)}")
    else:
        task = "unspecified"

    success: bool | None = None
    if "success" in ep_row and not _is_na(ep_row["success"]):
        success = bool(ep_row["success"])

    quality = _classify_quality(duration_s, frame_count)
    camera_sync_score = _compute_sync_score(fs, root_path, episode_index, fps)
    metrics = _compute_custom_metrics(fs, root_path, episode_index)
    robot_type = info.get("robot_type", ep_row.get("robot_type", "unknown"))
    
    source_hash = _compute_episode_hash(
        fs,
        root_path,
        episode_index,
        manifest_fields={
            "task": task,
            "task_index": None if task_index is None else int(task_index),
            "frame_count": frame_count,
            "fps": fps,
            "success": success,
            "robot_type": str(robot_type),
        },
    )
    
    modalities = tuple(sorted(info.get("features", {}).keys()))

    _excluded = {
        "episode_index", "length", "num_frames", "task_index",
        "task", "success", "robot_type",
    }

    raw_extras: dict[str, Any] = {
        k: v for k, v in ep_row.items()
        if k not in _excluded and not _is_na(v)
    }

    return EpisodeManifest(
        episode_id=f"episode_{episode_index:06d}",
        task=task,
        duration_s=round(duration_s, 4),
        frame_count=frame_count,
        fps=fps,
        robot_type=str(robot_type),
        modalities=modalities,
        camera_sync_score=round(camera_sync_score, 4),
        success=success,
        quality=quality,
        source_hash=source_hash,
        raw_extras=raw_extras,
        metrics=metrics,
    )

def _load_episode_frames(fs: fsspec.AbstractFileSystem, root_path: str, episode_index: int) -> pd.DataFrame | None:
    """Load all per-frame rows for one episode from the data Parquet files."""
    data_dir = f"{root_path}/data"
    if not fs.exists(data_dir):
        return None
    
    parquet_files = sorted([p for p in fs.find(data_dir) if p.endswith(".parquet")])
    for pf in parquet_files:
        try:
            with fs.open(pf, "rb") as f:
                schema = pq.read_schema(f)
            if "episode_index" not in schema.names:
                continue
            
            with fs.open(pf, "rb") as f:
                table = pq.read_table(
                    f, filters=[("episode_index", "=", episode_index)]
                )
            if table.num_rows == 0:
                continue
            return table.to_pandas()
        except Exception:
            continue
    return None

def _compute_custom_metrics(fs: fsspec.AbstractFileSystem, root_path: str, episode_index: int) -> dict[str, float]:
    """Compute all registered custom quality metrics for an episode."""
    if not QUALITY_METRICS:
        return {}
    
    df = _load_episode_frames(fs, root_path, episode_index)
    if df is None or df.empty:
        return {}
    
    out: dict[str, float] = {}
    for name, fn in QUALITY_METRICS.items():
        try:
            value = fn(df)
        except Exception:
            continue
        if value is not None:
            out[name] = round(float(value), 4)
    return out

def _is_na(v: object) -> bool:
    """Check if a value is NaN/None."""
    if isinstance(v, (list, dict)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False

def _classify_quality(duration_s: float, frame_count: int) -> EpisodeQuality:
    """Classify episode quality based on duration and frame count."""
    if frame_count == 0 or duration_s < _MIN_EPISODE_DURATION_S:
        return EpisodeQuality.corrupted
    if duration_s < _MIN_EPISODE_DURATION_S * 2:
        return EpisodeQuality.partial
    return EpisodeQuality.complete

def _compute_sync_score(fs: fsspec.AbstractFileSystem, root_path: str, episode_index: int, fps: int) -> float:
    """Compute camera synchronization score for an episode."""
    data_dir = f"{root_path}/data"
    if not fs.exists(data_dir):
        return 1.0
    
    parquet_files = sorted([p for p in fs.find(data_dir) if p.endswith(".parquet")])
    if not parquet_files:
        return 1.0

    target_file: str | None = None
    for pf in parquet_files:
        try:
            with fs.open(pf, "rb") as f:
                schema = pq.read_schema(f)
            if "episode_index" not in schema.names:
                continue
            
            with fs.open(pf, "rb") as f:
                ep_col = pq.read_table(f, columns=["episode_index"]).to_pandas()
            if episode_index in ep_col["episode_index"].values:
                target_file = pf
                break
        except Exception:
            continue

    if target_file is None:
        return 1.0

    try:
        with fs.open(target_file, "rb") as f:
            df = pq.read_table(
                f,
                filters=[("episode_index", "=", episode_index)],
                columns=["episode_index", "timestamp"],
            ).to_pandas()
    except Exception:
        return 1.0

    if df.empty or "timestamp" not in df.columns or len(df) < 2:
        return 1.0

    timestamps = df["timestamp"].sort_values().values.astype(np.float64)
    diffs = timestamps[1:] - timestamps[:-1]

    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 1.0

    expected_interval = float(positive_diffs.mean())
    if expected_interval <= 0:
        return 1.0

    max_drift = float(np.abs(positive_diffs - expected_interval).max())
    relative_drift = max_drift / expected_interval

    if relative_drift <= 0.05:
        return 1.0
    if relative_drift >= 0.5:
        return float(max(0.0, _SYNC_SCORE_DEGRADED - 0.05))

    drift_ratio = relative_drift / 0.5
    return round(float(1.0 - drift_ratio * (1.0 - _SYNC_SCORE_DEGRADED)), 4)

def _compute_episode_hash(
    fs: fsspec.AbstractFileSystem,
    root_path: str,
    episode_index: int,
    manifest_fields: dict[str, Any] | None = None,
) -> str:
    """Compute a content-addressed hash for an episode."""
    h = hashlib.sha256()
    h.update(str(episode_index).encode())
    
    if manifest_fields:
        h.update(
            json.dumps(manifest_fields, sort_keys=True, default=str).encode()
        )

    data_dir = f"{root_path}/data"
    parquet_files = sorted([p for p in fs.find(data_dir) if p.endswith(".parquet")]) if fs.exists(data_dir) else []
    
    for pf in parquet_files:
        try:
            with fs.open(pf, "rb") as f:
                schema = pq.read_schema(f)
            if "episode_index" not in schema.names:
                continue

            sample_cols = ["episode_index"]
            for col in ("observation.state", "action", "timestamp"):
                if col in schema.names:
                    sample_cols.append(col)
                    break

            with fs.open(pf, "rb") as f:
                table = pq.read_table(
                    f,
                    filters=[("episode_index", "=", episode_index)],
                    columns=sample_cols,
                )
            if table.num_rows == 0:
                continue

            h.update(pf.split("/")[-1].encode())
            h.update(str(table.num_rows).encode())

            df = table.to_pandas()
            value_cols = [c for c in df.columns if c != "episode_index"]
            if value_cols:
                col = value_cols[0]
                head = df[col].head(_HASH_SAMPLE_ROWS).astype(str).str.cat()
                tail = df[col].tail(_HASH_SAMPLE_ROWS).astype(str).str.cat()
                h.update(head.encode())
                h.update(tail.encode())

        except Exception:
            continue

    return h.hexdigest()[:16]