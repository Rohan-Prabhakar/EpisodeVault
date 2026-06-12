from __future__ import annotations

import pytest

from episodevault.diff.engine import EpisodeAnomaly, detect_anomalies, diff
from episodevault.parsers.lerobot import parse
from tests.fixtures import make_lerobot_v3_dataset


def _parse(tmp_path, name, episodes):
    root = make_lerobot_v3_dataset(tmp_path / name, episodes)
    return parse(root)


def test_diff_detects_episode_additions(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    result = diff(before, after)
    assert result.episodes_added > 0


def test_diff_detects_task_distribution_shift(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
    ])
    result = diff(before, after)

    kitchen_delta = next(
        td for td in result.task_deltas if td.task == "kitchen_grasp"
    )
    assert kitchen_delta.pct_change < 0
    assert kitchen_delta.flagged


def test_diff_quality_delta_duration(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "grasp", "task_index": 0, "frame_count": 150},
        {"task": "grasp", "task_index": 0, "frame_count": 150},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "grasp", "task_index": 0, "frame_count": 60},
        {"task": "grasp", "task_index": 0, "frame_count": 60},
    ])
    result = diff(before, after)
    assert result.quality_delta.avg_duration_s_after < result.quality_delta.avg_duration_s_before


def test_diff_regression_hint_on_large_task_drop(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 90},
    ])
    result = diff(before, after)
    assert result.regression_hint is not None
    assert "kitchen_grasp" in result.regression_hint


def test_diff_no_hint_when_stable(tmp_path):
    episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ]
    before = _parse(tmp_path, "v1", episodes)
    after = _parse(tmp_path, "v2", episodes)
    result = diff(before, after)
    assert result.regression_hint is None


def test_diff_format_output_contains_expected_sections(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 120, "success": True},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "factory_pick", "task_index": 1, "frame_count": 60, "success": False},
    ])
    result = diff(before, after)
    formatted = result.format()

    assert "Distribution shift" in formatted
    assert "Quality metrics" in formatted
    assert "Episodes added" in formatted


def test_diff_success_rate_delta(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": True},
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": True},
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": True},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": True},
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": False},
        {"task": "grasp", "task_index": 0, "frame_count": 90, "success": False},
    ])
    result = diff(before, after)
    sr_before = result.quality_delta.success_rate_before
    sr_after = result.quality_delta.success_rate_after
    assert sr_before is not None
    assert sr_after is not None
    assert sr_after < sr_before


# --- Anomaly detection ------------------------------------------------------

def test_detect_anomalies_flags_corrupted_episode(tmp_path):
    episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 1},  # corrupted
    ]
    manifest = _parse(tmp_path, "v1", episodes)
    anomalies = detect_anomalies(manifest)

    assert anomalies
    top = anomalies[0]
    assert any("corrupted" in r for r in top.reasons)


def test_detect_anomalies_flags_statistically_short_episode(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 150} for _ in range(8)]
    episodes.append({"task": "grasp", "task_index": 0, "frame_count": 20})
    manifest = _parse(tmp_path, "v1", episodes)
    anomalies = detect_anomalies(manifest)

    flagged_ids = {a.episode_id for a in anomalies}
    assert "episode_000008" in flagged_ids


def test_detect_anomalies_empty_when_uniform(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 90} for _ in range(5)]
    manifest = _parse(tmp_path, "v1", episodes)
    assert detect_anomalies(manifest) == ()


# --- Custom quality metric diffing ------------------------------------------

def test_diff_reports_metric_deltas(tmp_path):
    root_b = make_lerobot_v3_dataset(
        tmp_path / "v1",
        [{"task": "grasp", "task_index": 0, "frame_count": 40, "jerky": False}],
        include_actions=True,
    )
    root_a = make_lerobot_v3_dataset(
        tmp_path / "v2",
        [{"task": "grasp", "task_index": 0, "frame_count": 40, "jerky": True}],
        include_actions=True,
    )
    result = diff(parse(root_b), parse(root_a))

    names = {md.name for md in result.metric_deltas}
    assert "action_smoothness" in names
    smooth = next(md for md in result.metric_deltas if md.name == "action_smoothness")
    # Jerky actions in the "after" version => lower smoothness score.
    assert smooth.avg_after < smooth.avg_before


# --- HTML report ------------------------------------------------------------

def test_to_html_is_self_contained(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 120, "success": True},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "factory_pick", "task_index": 1, "frame_count": 60, "success": False},
    ])
    report = diff(before, after).to_html()

    assert report.startswith("<!DOCTYPE html>")
    assert "<svg" in report
    # No external resources => safe to email/archive.
    assert "http://" not in report.replace("http://www.w3.org/2000/svg", "")
    assert "src=" not in report


def test_to_html_includes_anomalies_section(tmp_path):
    before = _parse(tmp_path, "v1", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    after = _parse(tmp_path, "v2", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    anomalies = (
        EpisodeAnomaly(
            episode_id="episode_000003",
            task="grasp",
            severity=0.9,
            reasons=("quality=corrupted",),
        ),
    )
    report = diff(before, after).to_html(anomalies)
    assert "Flagged episodes (1)" in report
    assert "episode_000003" in report
    assert "quality=corrupted" in report
