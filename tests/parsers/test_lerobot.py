from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from episodevault.models import EpisodeQuality
from episodevault.parsers.lerobot import parse
from tests.fixtures import make_lerobot_v3_dataset


def test_parse_returns_dataset_manifest(tmp_path):
    episodes = [
        {"task": "pick_apple", "task_index": 0, "frame_count": 150, "success": True},
        {"task": "pick_apple", "task_index": 0, "frame_count": 120, "success": True},
        {"task": "place_cup", "task_index": 1, "frame_count": 90, "success": False},
    ]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    manifest = parse(root)

    assert manifest.total_episodes == 3
    assert manifest.fps == 30
    assert manifest.robot_type == "so101"
    assert len(manifest.episodes) == 3


def test_parse_episode_durations(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    manifest = parse(root)

    ep = manifest.episodes[0]
    assert ep.duration_s == pytest.approx(2.0, rel=1e-3)
    assert ep.frame_count == 60


def test_parse_task_distribution(tmp_path):
    episodes = [
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "kitchen_grasp", "task_index": 0, "frame_count": 90},
        {"task": "factory_pick", "task_index": 1, "frame_count": 120},
    ]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)
    manifest = parse(root)

    tasks = {e.task for e in manifest.episodes}
    assert "kitchen_grasp" in tasks
    assert "factory_pick" in tasks


def test_parse_success_flag(tmp_path):
    episodes = [
        {"task": "task_a", "task_index": 0, "frame_count": 60, "success": True},
        {"task": "task_a", "task_index": 0, "frame_count": 60, "success": False},
        {"task": "task_a", "task_index": 0, "frame_count": 60},
    ]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)
    manifest = parse(root)

    successes = [e.success for e in manifest.episodes]
    assert True in successes
    assert False in successes
    assert None in successes


def test_parse_corrupted_episode(tmp_path):
    episodes = [{"task": "task_a", "task_index": 0, "frame_count": 1}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    manifest = parse(root)

    assert manifest.episodes[0].quality == EpisodeQuality.corrupted


def test_parse_complete_episode(tmp_path):
    episodes = [{"task": "task_a", "task_index": 0, "frame_count": 150}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    manifest = parse(root)

    assert manifest.episodes[0].quality == EpisodeQuality.complete


def test_parse_source_hash_stable(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)
    m1 = parse(root)
    m2 = parse(root)

    assert m1.episodes[0].source_hash == m2.episodes[0].source_hash


def test_parse_rejects_non_v3(tmp_path):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": 30})
    )
    with pytest.raises(ValueError, match="v3"):
        parse(root)


def test_parse_rejects_missing_info(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        parse(root)


def test_parse_modalities_extracted(tmp_path):
    episodes = [{"task": "task_a", "task_index": 0, "frame_count": 60}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)
    manifest = parse(root)

    assert "action" in manifest.modalities
    assert "observation.state" in manifest.modalities


# --- Edge case fixes ---

def test_parse_rejects_hub_repo_id(tmp_path):
    with pytest.raises(ValueError, match="HuggingFace repo ID"):
        parse("lerobot/aloha_static_pro_pencil")


def test_parse_migration_error_mentions_correct_command(tmp_path):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.0", "fps": 30})
    )
    with pytest.raises(ValueError, match="lerobot-convert-dataset"):
        parse(root)


def test_parse_missing_tasks_jsonl_warns_and_uses_unspecified(tmp_path):
    episodes = [
        {"task_index": 0, "frame_count": 90},
        {"task_index": 0, "frame_count": 90},
    ]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks_path.unlink()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest = parse(root)

    task_labels = {e.task for e in manifest.episodes}
    assert "unspecified" in task_labels
    assert any("tasks.jsonl" in str(w.message) for w in caught)


def test_parse_missing_success_flags_warns(tmp_path):
    episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse(root)

    assert any("success" in str(w.message).lower() for w in caught)


def test_parse_episode_hash_differs_for_different_content(tmp_path):
    eps_a = [{"task": "grasp", "task_index": 0, "frame_count": 90}]
    eps_b = [{"task": "place", "task_index": 0, "frame_count": 90}]
    root_a = make_lerobot_v3_dataset(tmp_path / "ds_a", eps_a)
    root_b = make_lerobot_v3_dataset(tmp_path / "ds_b", eps_b)

    m_a = parse(root_a)
    m_b = parse(root_b)

    assert m_a.episodes[0].source_hash != m_b.episodes[0].source_hash


def test_parse_episodes_jsonl_empty_falls_back(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)

    (root / "meta" / "episodes.jsonl").write_text("")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        manifest = parse(root)

    assert len(manifest.episodes) >= 1


def test_parse_computes_custom_quality_metrics(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 40, "gripper": 0.5}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, include_actions=True)
    manifest = parse(root)

    ep = manifest.episodes[0]
    assert "action_smoothness" in ep.metrics
    assert "gripper_closure_rate" in ep.metrics
    assert 0.0 <= ep.metrics["action_smoothness"] <= 1.0
    # Gripper closed for ~the last half of frames.
    assert ep.metrics["gripper_closure_rate"] == pytest.approx(0.5, abs=0.05)


def test_parse_custom_metric_registration(tmp_path):
    from episodevault.parsers import lerobot

    lerobot.register_quality_metric("frame_total", lambda df: float(len(df)))
    try:
        episodes = [{"task": "grasp", "task_index": 0, "frame_count": 40}]
        root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, include_actions=True)
        manifest = parse(root)
        assert manifest.episodes[0].metrics["frame_total"] == 40.0
    finally:
        lerobot.QUALITY_METRICS.pop("frame_total", None)


def test_parse_metrics_empty_without_action_data(tmp_path):
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 40}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes)  # no actions
    manifest = parse(root)
    # Built-in metrics abstain (return None) when the action column is absent.
    assert "action_smoothness" not in manifest.episodes[0].metrics


def test_parse_hub_rejects_non_repo_id():
    from episodevault.parsers.lerobot import parse_hub

    with pytest.raises(ValueError, match="repo ID"):
        parse_hub("not_a_repo_id")


def test_parse_hub_downloads_and_parses(tmp_path, monkeypatch):
    from episodevault.parsers import lerobot

    local = make_lerobot_v3_dataset(
        tmp_path / "hub_cache",
        [{"task": "grasp", "task_index": 0, "frame_count": 90}],
    )

    def fake_download(repo_id, revision, cache_dir):
        return local

    monkeypatch.setattr(lerobot, "_download_hub_dataset", fake_download)

    manifest = lerobot.parse_hub("lerobot/aloha_static_pro_pencil")
    assert manifest.dataset_id == "lerobot/aloha_static_pro_pencil"
    assert manifest.total_episodes == 1


def test_parse_sync_score_tolerates_floating_point_drift(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 100}]
    root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)

    data_file = next((root / "data").glob("**/*.parquet"))
    df = pq.read_table(data_file).to_pandas()

    df["timestamp"] = [i / 30.0 + (i * 1e-10) for i in range(len(df))]
    pq.write_table(pa.Table.from_pandas(df), data_file)

    manifest = parse(root)
    assert manifest.episodes[0].camera_sync_score >= 0.95
