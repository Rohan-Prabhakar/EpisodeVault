from __future__ import annotations

import pytest

from episodevault.parsers.lerobot import parse
from episodevault.store.lineage_store import LineageStore
from episodevault.store.version_store import VersionStore
from tests.fixtures import make_lerobot_v3_dataset


def _make_manifest(tmp_path, name, episodes):
    root = make_lerobot_v3_dataset(tmp_path / name, episodes)
    return parse(root)


def test_commit_returns_version_id(tmp_path):
    manifest = _make_manifest(tmp_path, "ds", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    store = VersionStore(tmp_path / "store")
    vid = store.commit(manifest, "initial commit")
    assert vid == "v1.0"


def test_commit_increments_version(tmp_path):
    manifest = _make_manifest(tmp_path, "ds", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    store = VersionStore(tmp_path / "store")
    v1 = store.commit(manifest, "first")
    v2 = store.commit(manifest, "second")
    assert v1 == "v1.0"
    assert v2 == "v2.0"


def test_read_version_roundtrips_manifest(tmp_path):
    manifest = _make_manifest(tmp_path, "ds", [
        {"task": "pick_apple", "task_index": 0, "frame_count": 120, "success": True},
        {"task": "place_cup", "task_index": 1, "frame_count": 90, "success": False},
    ])
    store = VersionStore(tmp_path / "store")
    vid = store.commit(manifest, "test commit")
    recovered = store.read_version(vid)

    assert recovered.total_episodes == manifest.total_episodes
    assert recovered.fps == manifest.fps
    assert recovered.robot_type == manifest.robot_type
    assert {e.task for e in recovered.episodes} == {e.task for e in manifest.episodes}


def test_read_version_raises_on_missing(tmp_path):
    store = VersionStore(tmp_path / "store")
    with pytest.raises(KeyError, match="v99.0"):
        store.read_version("v99.0")


def test_list_versions_ordered(tmp_path):
    manifest = _make_manifest(tmp_path, "ds", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    store = VersionStore(tmp_path / "store")
    store.commit(manifest, "first")
    store.commit(manifest, "second")
    store.commit(manifest, "third")

    versions = store.list_versions()
    assert [v["version_id"] for v in versions] == ["v1.0", "v2.0", "v3.0"]


def test_latest_version_id(tmp_path):
    manifest = _make_manifest(tmp_path, "ds", [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ])
    store = VersionStore(tmp_path / "store")
    store.commit(manifest, "first")
    store.commit(manifest, "second")
    assert store.latest_version_id() == "v2.0"


def test_latest_version_id_empty(tmp_path):
    store = VersionStore(tmp_path / "store")
    assert store.latest_version_id() is None


def test_lineage_log_and_retrieve(tmp_path):
    lineage = LineageStore(tmp_path / "store")
    lineage.log_training_run("model_v1", "v2.0", framework="lerobot")
    result = lineage.dataset_version_for_model("model_v1")
    assert result == "v2.0"


def test_lineage_returns_none_for_unknown_model(tmp_path):
    lineage = LineageStore(tmp_path / "store")
    assert lineage.dataset_version_for_model("nonexistent_model") is None


def test_lineage_returns_latest_run_for_model(tmp_path):
    lineage = LineageStore(tmp_path / "store")
    lineage.log_training_run("model_v1", "v1.0")
    lineage.log_training_run("model_v1", "v3.0")
    assert lineage.dataset_version_for_model("model_v1") == "v3.0"


def test_lineage_list_runs(tmp_path):
    lineage = LineageStore(tmp_path / "store")
    lineage.log_training_run("model_v1", "v1.0")
    lineage.log_training_run("model_v2", "v2.0")
    runs = lineage.list_runs()
    assert len(runs) == 2
    assert {r["model_version"] for r in runs} == {"model_v1", "model_v2"}
