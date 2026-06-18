from __future__ import annotations

import html
import statistics
from dataclasses import dataclass, field
from typing import Any

from episodevault.models import DatasetManifest, EpisodeManifest, EpisodeQuality


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
class MetricDelta:
    """Shift in a custom quality metric's dataset-wide average across versions."""
    name: str
    avg_before: float | None
    avg_after: float | None


@dataclass(frozen=True)
class EpisodeAnomaly:
    """An episode flagged as a statistical or rule-based outlier."""
    episode_id: str
    task: str
    severity: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeDiff:
    version_before: str
    version_after: str
    episodes_added: int
    episodes_removed: int
    task_deltas: tuple[TaskDelta, ...]
    quality_delta: QualityDelta
    regression_hint: str | None
    metric_deltas: tuple[MetricDelta, ...] = ()

    def format(self) -> str:
        lines: list[str] = []
        lines.append(f"Dataset diff: {self.version_before} → {self.version_after}")
        lines.append("─" * 52)
        lines.append(f"Episodes added:    +{self.episodes_added}")
        lines.append(f"Episodes removed:  -{self.episodes_removed}")
        lines.append("")
        lines.append("Distribution shift:")

        for td in sorted(self.task_deltas, key=lambda x: abs(x.pct_change), reverse=True):
            direction = "↓" if td.pct_change < 0 else ("-" if td.pct_change == 0 else "↑")
            flag = "  ⚠️" if td.flagged else ""
            lines.append(
                f"  {td.task[:32]:<32} "
                f"{td.count_before} → {td.count_after}  "
                f"{direction} {abs(td.pct_change):.0f}%{flag}"
            )

        lines.append("")
        lines.append("Quality metrics:")
        qd = self.quality_delta

        dur_dir = "↓" if qd.avg_duration_s_after < qd.avg_duration_s_before else ("-" if qd.avg_duration_s_after == qd.avg_duration_s_before else "↑")
        lines.append(
            f"  avg episode length:    "
            f"{qd.avg_duration_s_before:.1f}s → {qd.avg_duration_s_after:.1f}s  {dur_dir}"
        )

        if qd.success_rate_before is not None and qd.success_rate_after is not None:
            sr_dir = "↓" if qd.success_rate_after < qd.success_rate_before else ("-" if qd.success_rate_after == qd.success_rate_before else "↑")
            lines.append(
                f"  success_rate:          "
                f"{qd.success_rate_before:.2f} → {qd.success_rate_after:.2f}  {sr_dir}"
            )
        elif qd.success_rate_before is None and qd.success_rate_after is None:
            lines.append(
                "  success_rate:          n/a  "
                "⚠️  add success flags to episodes for regression analysis"
            )

        sync_dir = "↓" if qd.avg_sync_score_after < qd.avg_sync_score_before else ("-" if qd.avg_sync_score_after == qd.avg_sync_score_before else "↑")
        lines.append(
            f"  camera_sync_score:     "
            f"{qd.avg_sync_score_before:.2f} → {qd.avg_sync_score_after:.2f}  {sync_dir}"
        )

        if qd.corrupted_after > qd.corrupted_before:
            lines.append(
                f"  corrupted episodes:    "
                f"{qd.corrupted_before} → {qd.corrupted_after}  ↑  ⚠️"
            )

        if self.metric_deltas:
            lines.append("")
            lines.append("Custom quality metrics:")
            for md in self.metric_deltas:
                before = "n/a" if md.avg_before is None else f"{md.avg_before:.3f}"
                after = "n/a" if md.avg_after is None else f"{md.avg_after:.3f}"
                if md.avg_before is not None and md.avg_after is not None:
                    arrow = "↓" if md.avg_after < md.avg_before else ("-" if md.avg_after == md.avg_before else "↑")
                else:
                    arrow = " "
                lines.append(f"  {md.name[:28]:<28} {before} → {after}  {arrow}")

        if self.regression_hint:
            lines.append("")
            lines.append("Regression candidates (ranked by magnitude; correlate with your eval):")
            for candidate in self.regression_hint.split("\n"):
                lines.append(f"  - {candidate}")

        return "\n".join(lines)

    def to_html(
        self,
        anomalies: "tuple[EpisodeAnomaly, ...]" = (),
        versions: "list[dict] | None" = None,
    ) -> str:
        """Render a self-contained, shareable HTML audit report (no external deps).

        anomalies: pass detect_anomalies() on the "after" manifest to include
                   a flagged-episodes table.
        versions:  pass store.list_versions() to include a visual version
                   history graph showing where this diff sits in the timeline.
        """
        return _render_html_report(self, anomalies, versions)


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
    metric_deltas = _compute_metric_deltas(before, after)
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
        metric_deltas=metric_deltas,
    )


def _compute_metric_deltas(
    before: DatasetManifest, after: DatasetManifest
) -> tuple[MetricDelta, ...]:
    avg_before = before.avg_metrics
    avg_after = after.avg_metrics
    names = sorted(set(avg_before) | set(avg_after))
    return tuple(
        MetricDelta(
            name=name,
            avg_before=(round(avg_before[name], 4) if name in avg_before else None),
            avg_after=(round(avg_after[name], 4) if name in avg_after else None),
        )
        for name in names
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


# --- Anomaly detection (feature: automated outlier flagging) ----------------

# Robust z-score threshold (median/MAD based) above which a numeric field is an
# outlier. 3.5 is the standard cutoff for the modified z-score.
_ANOMALY_Z_THRESHOLD = 3.5
# Episodes with a sync score below this are flagged regardless of distribution.
_ANOMALY_SYNC_FLOOR = 0.9


def _modified_zscores(values: list[float]) -> list[float]:
    """Median/MAD-based robust z-scores; resistant to the outliers we hunt for."""
    n = len(values)
    if n < 3:
        return [0.0] * n
    median = statistics.median(values)
    abs_dev = [abs(v - median) for v in values]
    mad = statistics.median(abs_dev)
    if mad == 0:
        # Fall back to mean absolute deviation when MAD collapses (many ties).
        mean_ad = sum(abs_dev) / n
        if mean_ad == 0:
            return [0.0] * n
        return [(v - median) / (1.253314 * mean_ad) for v in values]
    return [0.6744898 * (v - median) / mad for v in values]


def detect_anomalies(
    manifest: DatasetManifest,
    *,
    z_threshold: float = _ANOMALY_Z_THRESHOLD,
) -> tuple[EpisodeAnomaly, ...]:
    """Flag outlier episodes so bad data can be pruned before training.

    Combines distribution-based outlier detection (robust z-score on duration,
    frame count, sync score, and every custom quality metric) with rule-based
    checks (corrupted quality, severely desynced cameras).
    """
    episodes = manifest.episodes
    if not episodes:
        return ()

    reasons: dict[str, list[str]] = {e.episode_id: [] for e in episodes}
    severity: dict[str, float] = {e.episode_id: 0.0 for e in episodes}

    def bump(ep_id: str, sev: float, msg: str) -> None:
        reasons[ep_id].append(msg)
        severity[ep_id] = max(severity[ep_id], min(1.0, sev))

    # Distribution-based: low-side duration/frame_count and both-side metrics.
    numeric_fields: dict[str, list[float]] = {
        "duration_s": [e.duration_s for e in episodes],
        "frame_count": [float(e.frame_count) for e in episodes],
        "camera_sync_score": [e.camera_sync_score for e in episodes],
    }
    metric_names = sorted({name for e in episodes for name in e.metrics})
    for name in metric_names:
        # Only score episodes that actually report the metric.
        present = [(e.episode_id, e.metrics[name]) for e in episodes if name in e.metrics]
        if len(present) >= 3:
            zs = _modified_zscores([v for _, v in present])
            for (ep_id, value), z in zip(present, zs):
                if abs(z) >= z_threshold:
                    bump(ep_id, abs(z) / 10.0, f"{name}={value:.3f} is a statistical outlier (z={z:.1f})")

    for field_name, values in numeric_fields.items():
        zs = _modified_zscores(values)
        for ep, z in zip(episodes, zs):
            if abs(z) < z_threshold:
                continue
            if field_name in ("duration_s", "frame_count") and z > 0:
                # Unusually long episodes are rarely "bad data"; focus on short.
                continue
            label = {
                "duration_s": "unusually short" if z < 0 else "unusually long",
                "frame_count": "unusually few frames" if z < 0 else "unusually many frames",
                "camera_sync_score": "camera sync outlier",
            }[field_name]
            bump(ep.episode_id, abs(z) / 10.0, f"{label} (z={z:.1f})")

    # Rule-based checks independent of the distribution.
    for ep in episodes:
        if ep.quality == EpisodeQuality.corrupted:
            bump(ep.episode_id, 0.9, "quality=corrupted")
        if ep.camera_sync_score < _ANOMALY_SYNC_FLOOR:
            bump(ep.episode_id, 1.0 - ep.camera_sync_score, f"camera_sync_score={ep.camera_sync_score:.2f} below floor")

    anomalies = [
        EpisodeAnomaly(
            episode_id=ep.episode_id,
            task=ep.task,
            severity=round(severity[ep.episode_id], 3),
            reasons=tuple(reasons[ep.episode_id]),
        )
        for ep in episodes
        if reasons[ep.episode_id]
    ]
    anomalies.sort(key=lambda a: a.severity, reverse=True)
    return tuple(anomalies)


# --- HTML report (feature: shareable visual audits) -------------------------

def _svg_bar_chart(pairs: list[tuple[str, float, float]], title: str) -> str:
    """A small grouped bar chart (before vs after) as inline SVG."""
    if not pairs:
        return ""
    row_h = 34
    label_w = 180
    bar_max = 360
    height = len(pairs) * row_h + 30
    peak = max((max(b, a) for _, b, a in pairs), default=1.0) or 1.0

    rows = [f'<text x="0" y="16" class="cap">{html.escape(title)}</text>']
    for i, (label, before, after) in enumerate(pairs):
        y = 30 + i * row_h
        bw = int((before / peak) * bar_max)
        aw = int((after / peak) * bar_max)
        rows.append(
            f'<text x="0" y="{y + 14}" class="lbl">{html.escape(label[:26])}</text>'
            f'<rect x="{label_w}" y="{y + 2}" width="{bw}" height="11" class="b"/>'
            f'<rect x="{label_w}" y="{y + 15}" width="{aw}" height="11" class="a"/>'
            f'<text x="{label_w + max(bw, aw) + 6}" y="{y + 16}" class="val">{before:g} → {after:g}</text>'
        )
    return (
        f'<svg width="{label_w + bar_max + 90}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">' + "".join(rows) + "</svg>"
    )


def _svg_version_graph(versions: list[dict]) -> str:
    """Vertical commit-graph SVG — circles on a line, newest at top."""
    if not versions:
        return ""
    ordered = list(reversed(versions))  # newest first
    row_h = 48
    cx, r = 16, 8
    width, height = 560, len(ordered) * row_h + 16

    nodes: list[str] = []
    # Connector line through all nodes
    if len(ordered) > 1:
        y_top = row_h // 2
        y_bot = (len(ordered) - 1) * row_h + row_h // 2
        nodes.append(
            f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{y_bot}" '
            f'stroke="#c0c7d6" stroke-width="2"/>'
        )
    for i, v in enumerate(ordered):
        y = i * row_h + row_h // 2
        is_active = i == 0  # newest = highlighted
        fill = "#4361ee" if is_active else "#9aa5c4"
        nodes.append(
            f'<circle cx="{cx}" cy="{y}" r="{r}" fill="{fill}"/>'
            f'<text x="{cx + r + 10}" y="{y - 5}" class="vid">'
            f'{html.escape(v["version_id"])}</text>'
            f'<text x="{cx + r + 10}" y="{y + 11}" class="vmsg">'
            f'{html.escape(str(v["commit_message"])[:55])}'
            f'  <tspan class="vmeta">· {int(v["total_episodes"])} eps</tspan></text>'
        )
    return (
        f'<svg width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="vgraph">'
        + "".join(nodes) + "</svg>"
    )


def _render_html_report(d: EpisodeDiff, anomalies: tuple[EpisodeAnomaly, ...] = (),
                        versions: list[dict] | None = None) -> str:
    task_pairs = [
        (td.task, float(td.count_before), float(td.count_after))
        for td in sorted(d.task_deltas, key=lambda x: abs(x.pct_change), reverse=True)
    ]
    qd = d.quality_delta
    quality_pairs = [
        ("avg duration (s)", qd.avg_duration_s_before, qd.avg_duration_s_after),
        ("avg camera sync", qd.avg_sync_score_before, qd.avg_sync_score_after),
    ]
    if qd.success_rate_before is not None and qd.success_rate_after is not None:
        quality_pairs.append(("success rate", qd.success_rate_before, qd.success_rate_after))
    metric_pairs = [
        (md.name, md.avg_before or 0.0, md.avg_after or 0.0)
        for md in d.metric_deltas
        if md.avg_before is not None or md.avg_after is not None
    ]

    hint_html = ""
    if d.regression_hint:
        items = "".join(f"<li>{html.escape(c)}</li>" for c in d.regression_hint.split("\n"))
        hint_html = f'<h2>Regression candidates</h2><ul class="hints">{items}</ul>'

    metric_chart = _svg_bar_chart(metric_pairs, "Custom quality metrics") if metric_pairs else ""

    version_graph_html = ""
    if versions:
        svg = _svg_version_graph(versions)
        # Highlight the two versions involved in this diff in the label
        version_graph_html = f'<h2>Version history</h2>{svg}'

    anomaly_html = ""
    if anomalies:
        rows = "".join(
            f"<tr><td>{html.escape(a.episode_id)}</td>"
            f"<td>{html.escape(a.task[:30])}</td>"
            f"<td>{a.severity:.2f}</td>"
            f"<td>{html.escape('; '.join(a.reasons))}</td></tr>"
            for a in anomalies
        )
        anomaly_html = (
            f'<h2>Flagged episodes ({len(anomalies)})</h2>'
            '<table class="anom"><tr><th>Episode</th><th>Task</th>'
            '<th>Severity</th><th>Reasons</th></tr>' + rows + "</table>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>EpisodeVault report: {html.escape(d.version_before)} → {html.escape(d.version_after)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:820px;color:#1a1a2e;}}
 h1{{font-size:1.4rem;}} h2{{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #eee;padding-bottom:.3rem;}}
 .summary{{display:flex;gap:2rem;margin:1rem 0;}}
 .summary div{{background:#f5f6fa;border-radius:8px;padding:.8rem 1.2rem;}}
 .summary b{{display:block;font-size:1.6rem;}}
 .chart .cap{{font-weight:600;font-size:13px;fill:#1a1a2e;}}
 .chart .lbl{{font-size:12px;fill:#444;}} .chart .val{{font-size:11px;fill:#888;}}
 .chart .b{{fill:#9aa5c4;}} .chart .a{{fill:#4361ee;}}
 .legend span{{font-size:12px;margin-right:1rem;}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;}}
 ul.hints li{{margin:.3rem 0;}}
 table.anom{{border-collapse:collapse;width:100%;font-size:13px;}}
 table.anom th,table.anom td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #eee;}}
 table.anom th{{color:#888;font-weight:600;}}
 .vgraph .vid{{font-size:13px;font-weight:600;fill:#1a1a2e;}}
 .vgraph .vmsg{{font-size:12px;fill:#555;}}
 .vgraph .vmeta{{fill:#aaa;}}
 footer{{margin-top:2rem;color:#aaa;font-size:11px;}}
</style></head><body>
<h1>Dataset diff — {html.escape(d.version_before)} → {html.escape(d.version_after)}</h1>
<div class="summary">
 <div><b>+{d.episodes_added}</b> episodes added</div>
 <div><b>-{d.episodes_removed}</b> episodes removed</div>
</div>
<p class="legend"><span><i style="background:#9aa5c4"></i>before</span><span><i style="background:#4361ee"></i>after</span></p>
{version_graph_html}
<h2>Task distribution</h2>
{_svg_bar_chart(task_pairs, "Episodes per task")}
<h2>Quality metrics</h2>
{_svg_bar_chart(quality_pairs, "Dataset-wide quality")}
{('<h2>Custom quality metrics</h2>' + metric_chart) if metric_chart else ''}
{anomaly_html}
{hint_html}
<footer>Generated by EpisodeVault. Self-contained — safe to email or archive.</footer>
</body></html>"""
