from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from episodevault.models import DatasetManifest, EpisodeQuality


@dataclass(frozen=True)
class TaskDelta:
    task: str
    count_before: int
    count_after: int
    pct_change: float
    flagged: bool


@dataclass(frozen=True)
class QualityDelta:
    avg_duration_s_before: float
    avg_duration_s_after: float
    success_rate_before: float | None
    success_rate_after: float | None
    avg_sync_score_before: float
    avg_sync_score_after: float
    corrupted_before: int
    corrupted_after: int


@dataclass(frozen=True)
class EpisodeDiff:
    version_before: str
    version_after: str
    episodes_added: int
    episodes_removed: int
    task_deltas: tuple[TaskDelta, ...]
    quality_delta: QualityDelta
    regression_hint: str | None

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"Dataset diff: {self.version_before} → {self.version_after}")
        lines.append("─" * 52)
        lines.append(f"Episodes added:    +{self.episodes_added}")
        lines.append(f"Episodes removed:  -{self.episodes_removed}")
        lines.append("")
        lines.append("Distribution shift:")

        for td in sorted(self.task_deltas, key=lambda x: abs(x.pct_change), reverse=True):
            direction = "↓" if td.pct_change < 0 else "↑"
            flag = "  ⚠️" if td.flagged else ""
            lines.append(
                f"  {td.task[:32]:<32} "
                f"{td.count_before} → {td.count_after}  "
                f"{direction} {abs(td.pct_change):.0f}%{flag}"
            )

        lines.append("")
        lines.append("Quality metrics:")
        qd = self.quality_delta

        dur_dir = "↓" if qd.avg_duration_s_after < qd.avg_duration_s_before else "↑"
        lines.append(
            f"  avg episode length:    "
            f"{qd.avg_duration_s_before:.1f}s → {qd.avg_duration_s_after:.1f}s  {dur_dir}"
        )

        if qd.success_rate_before is not None and qd.success_rate_after is not None:
            sr_dir = "↓" if qd.success_rate_after < qd.success_rate_before else "↑"
            lines.append(
                f"  success_rate:          "
                f"{qd.success_rate_before:.2f} → {qd.success_rate_after:.2f}  {sr_dir}"
            )
        elif qd.success_rate_before is None and qd.success_rate_after is None:
            lines.append(
                "  success_rate:          n/a  "
                "⚠️  add success flags to episodes for regression analysis"
            )

        sync_dir = "↓" if qd.avg_sync_score_after < qd.avg_sync_score_before else "↑"
        lines.append(
            f"  camera_sync_score:     "
            f"{qd.avg_sync_score_before:.2f} → {qd.avg_sync_score_after:.2f}  {sync_dir}"
        )

        if qd.corrupted_after > qd.corrupted_before:
            lines.append(
                f"  corrupted episodes:    "
                f"{qd.corrupted_before} → {qd.corrupted_after}  ↑  ⚠️"
            )

        if self.regression_hint:
            lines.append("")
            lines.append("Regression candidates (ranked by magnitude; correlate with your eval):")
            for candidate in self.regression_hint.split("\n"):
                lines.append(f"  - {candidate}")

        return "\n".join(lines)


_DISTRIBUTION_FLAG_THRESHOLD = 0.25
_SYNC_DEGRADATION_THRESHOLD = 0.95
# Soft floor: ignore tasks with fewer than this many episodes before the change.
# A hard floor of ~5 would suppress almost everything on 18–56 episode LeRobot
# datasets, so we only filter out singleton (1 → 0) noise.
_MIN_TASK_EPISODES = 2
# Minimum success-rate drop (absolute, 0–1) before it is treated as a candidate.
_SUCCESS_RATE_DROP_THRESHOLD = 0.05
# Number of ranked candidates to surface.
_MAX_HINTS = 3


def diff(before: DatasetManifest, after: DatasetManifest) -> EpisodeDiff:
    ids_before = {e.episode_id for e in before.episodes}
    ids_after = {e.episode_id for e in after.episodes}
    added = len(ids_after - ids_before)
    removed = len(ids_before - ids_after)

    task_deltas = _compute_task_deltas(before, after)
    quality_delta = _compute_quality_delta(before, after)
    hint = _derive_regression_hint(task_deltas, quality_delta)

    version_before = getattr(before, "_version_id", "before")
    version_after = getattr(after, "_version_id", "after")

    return EpisodeDiff(
        version_before=version_before,
        version_after=version_after,
        episodes_added=added,
        episodes_removed=removed,
        task_deltas=tuple(task_deltas),
        quality_delta=quality_delta,
        regression_hint=hint,
    )


def _compute_task_deltas(
    before: DatasetManifest, after: DatasetManifest
) -> list[TaskDelta]:
    def task_counts(manifest: DatasetManifest) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ep in manifest.episodes:
            counts[ep.task] = counts.get(ep.task, 0) + 1
        return counts

    counts_before = task_counts(before)
    counts_after = task_counts(after)
    all_tasks = set(counts_before) | set(counts_after)

    deltas = []
    for task in sorted(all_tasks):
        cb = counts_before.get(task, 0)
        ca = counts_after.get(task, 0)
        if cb == 0:
            pct = 100.0
        elif ca == 0:
            pct = -100.0
        else:
            pct = (ca - cb) / cb * 100.0

        flagged = abs(pct) >= _DISTRIBUTION_FLAG_THRESHOLD * 100

        deltas.append(TaskDelta(
            task=task,
            count_before=cb,
            count_after=ca,
            pct_change=round(pct, 1),
            flagged=flagged,
        ))
    return deltas


def _compute_quality_delta(
    before: DatasetManifest, after: DatasetManifest
) -> QualityDelta:
    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def success_rate(manifest: DatasetManifest) -> float | None:
        with_flag = [e for e in manifest.episodes if e.success is not None]
        if not with_flag:
            return None
        return sum(1 for e in with_flag if e.success) / len(with_flag)

    def corrupted_count(manifest: DatasetManifest) -> int:
        return sum(1 for e in manifest.episodes if e.quality == EpisodeQuality.corrupted)

    return QualityDelta(
        avg_duration_s_before=round(avg([e.duration_s for e in before.episodes]), 2),
        avg_duration_s_after=round(avg([e.duration_s for e in after.episodes]), 2),
        success_rate_before=success_rate(before),
        success_rate_after=success_rate(after),
        avg_sync_score_before=round(avg([e.camera_sync_score for e in before.episodes]), 3),
        avg_sync_score_after=round(avg([e.camera_sync_score for e in after.episodes]), 3),
        corrupted_before=corrupted_count(before),
        corrupted_after=corrupted_count(after),
    )


def _derive_regression_hint(
    task_deltas: list[TaskDelta],
    quality_delta: QualityDelta,
) -> str | None:
    # Each candidate carries a severity normalized to 0–1 so signals on different
    # scales (relative task change vs. absolute success-rate drop) can be ranked
    # against each other. These are heuristic magnitudes, not proven causes — the
    # output header makes that explicit.
    candidates: list[tuple[float, str]] = []

    flagged_drops = [
        td for td in task_deltas
        if td.flagged
        and td.pct_change < 0
        and td.count_before >= _MIN_TASK_EPISODES
    ]
    for td in flagged_drops:
        # Fraction of the task's episodes lost (0–1); a full drop is severity 1.0.
        severity = min(1.0, abs(td.pct_change) / 100.0)
        label = td.task if len(td.task) <= 40 else td.task[:37] + "..."
        candidates.append((
            severity,
            f"'{label}' episodes dropped {abs(td.pct_change):.0f}% "
            f"({td.count_before} → {td.count_after}). "
            f"Restore from prior version if this task is in your eval benchmark.",
        ))

    if (
        quality_delta.success_rate_before is not None
        and quality_delta.success_rate_after is not None
        and quality_delta.success_rate_after
        < quality_delta.success_rate_before - _SUCCESS_RATE_DROP_THRESHOLD
    ):
        # Absolute drop in success rate is already on a 0–1 scale.
        drop = quality_delta.success_rate_before - quality_delta.success_rate_after
        candidates.append((
            min(1.0, drop),
            f"Success rate fell {drop:.0%} "
            f"({quality_delta.success_rate_before:.2f} → "
            f"{quality_delta.success_rate_after:.2f}). "
            f"New episodes may contain failed demonstrations. "
            f"Run score_lerobot_episodes to identify low-quality additions.",
        ))

    if quality_delta.avg_sync_score_after < _SYNC_DEGRADATION_THRESHOLD:
        # How far below a perfect sync score (0–1).
        severity = min(1.0, 1.0 - quality_delta.avg_sync_score_after)
        candidates.append((
            severity,
            f"Camera sync score degraded to {quality_delta.avg_sync_score_after:.2f}. "
            f"Multimodal alignment issues may cause perception regressions.",
        ))

    if quality_delta.corrupted_after > quality_delta.corrupted_before:
        delta = quality_delta.corrupted_after - quality_delta.corrupted_before
        # Modest severity that grows with the number of newly corrupted episodes.
        severity = min(0.5, 0.15 * delta)
        candidates.append((
            severity,
            f"{delta} corrupted episode(s) added. "
            f"Exclude episodes with quality=corrupted before retraining.",
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(msg for _, msg in candidates[:_MAX_HINTS])
