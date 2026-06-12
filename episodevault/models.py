from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EpisodeQuality(str, Enum):
    complete = "complete"
    partial = "partial"
    corrupted = "corrupted"


@dataclass(frozen=True)
class EpisodeManifest:
    episode_id: str
    task: str
    duration_s: float
    frame_count: int
    fps: int
    robot_type: str
    modalities: tuple[str, ...]
    camera_sync_score: float
    success: bool | None
    quality: EpisodeQuality
    source_hash: str
    raw_extras: dict[str, Any] = field(default_factory=dict)
    # User-defined quality metrics (e.g. action_smoothness, gripper_closure_rate)
    # computed at parse time and tracked across versions.
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "modalities": list(self.modalities),
            "camera_sync_score": self.camera_sync_score,
            "success": self.success,
            "quality": self.quality.value,
            "source_hash": self.source_hash,
            "raw_extras": self.raw_extras,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    episodes: tuple[EpisodeManifest, ...]
    format_version: str = "lerobot_v3"

    @property
    def avg_episode_duration_s(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.duration_s for e in self.episodes) / len(self.episodes)

    @property
    def success_rate(self) -> float | None:
        with_success = [e for e in self.episodes if e.success is not None]
        if not with_success:
            return None
        return sum(1 for e in with_success if e.success) / len(with_success)

    @property
    def avg_camera_sync_score(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.camera_sync_score for e in self.episodes) / len(self.episodes)

    @property
    def avg_metrics(self) -> dict[str, float]:
        """Mean of each custom quality metric across episodes that report it."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for e in self.episodes:
            for name, value in e.metrics.items():
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
        return {name: sums[name] / counts[name] for name in sums}
