from __future__ import annotations
import fsspec
import pytest
from pathlib import Path
from episodevault.parsers.lerobot import parse
from episodevault.diff.engine import diff
from tests.fixtures import make_lerobot_v3_dataset


def _copy_local_to_fsspec(local_path: Path, fs: fsspec.AbstractFileSystem, fs_path: str):
    """Copy a local directory tree into an fsspec filesystem."""
    if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)
    fs.mkdir(fs_path)
    for item in local_path.rglob("*"):
        if item.is_file():
            relative = item.relative_to(local_path)
            target = f"{fs_path}/{relative}".replace("\\", "/")
            # Ensure parent directory exists
            parent = "/".join(target.split("/")[:-1])
            if parent and not fs.exists(parent):
                fs.makedirs(parent, exist_ok=True)
            fs.put_file(str(item), target)


@pytest.fixture
def mem_fs():
    """Set up a clean fsspec memory filesystem for each test."""
    fs = fsspec.filesystem("memory")
    # Clear any previous state
    for path in fs.ls("/", detail=False):
        try:
            fs.rm(path, recursive=True)
        except Exception:
            pass
    return fs


def test_parse_dataset_from_cloud_uri(tmp_path, mem_fs):
    """Test parsing a LeRobot dataset from a cloud-like URI (memory:// simulates s3://)."""
    # 1. Create a local dataset
    episodes = [
        {"task": "pick_apple", "task_index": 0, "frame_count": 90, "success": True},
        {"task": "place_cup", "task_index": 1, "frame_count": 60, "success": False},
    ]
    local_root = make_lerobot_v3_dataset(tmp_path / "local_ds", episodes, fps=30)
    
    # 2. Copy to memory filesystem (simulates uploading to S3)
    _copy_local_to_fsspec(local_root, mem_fs, "/datasets/lerobot-v1")
    
    # 3. Parse directly from the cloud-like URI
    # This tests the exact same fsspec code path as s3://, gs://, az://
    manifest = parse("memory:///datasets/lerobot-v1")
    
    # 4. Verify the manifest
    assert manifest.total_episodes == 2
    assert manifest.fps == 30
    assert manifest.robot_type == "so101"
    assert len(manifest.episodes) == 2
    assert manifest.episodes[0].task == "pick_apple"
    assert manifest.episodes[1].task == "place_cup"


def test_diff_two_cloud_versions(tmp_path, mem_fs):
    """Test diffing two versions of a dataset stored in cloud-like storage."""
    # Create v1
    v1_episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
    ]
    v1_root = make_lerobot_v3_dataset(tmp_path / "v1", v1_episodes, fps=30)
    _copy_local_to_fsspec(v1_root, mem_fs, "/datasets/v1")
    
    # Create v2 (with distribution shift)
    v2_episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "place", "task_index": 1, "frame_count": 60},
    ]
    v2_root = make_lerobot_v3_dataset(tmp_path / "v2", v2_episodes, fps=30)
    _copy_local_to_fsspec(v2_root, mem_fs, "/datasets/v2")
    
    # Parse both from cloud URIs
    manifest_v1 = parse("memory:///datasets/v1")
    manifest_v2 = parse("memory:///datasets/v2")
    
    # Diff them
    result = diff(manifest_v1, manifest_v2)
    
    assert result.episodes_added == 2
    assert any(td.task == "place" for td in result.task_deltas)


def test_cloud_with_custom_quality_metrics(tmp_path, mem_fs):
    """Test that custom quality metrics work with cloud-hosted datasets."""
    episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 40, "gripper": 0.5}
    ]
    local_root = make_lerobot_v3_dataset(
        tmp_path / "metrics_ds", episodes, fps=30, include_actions=True
    )
    _copy_local_to_fsspec(local_root, mem_fs, "/datasets/metrics")
    
    manifest = parse("memory:///datasets/metrics")
    ep = manifest.episodes[0]
    
    assert "action_smoothness" in ep.metrics
    assert "gripper_closure_rate" in ep.metrics
    assert 0.0 <= ep.metrics["action_smoothness"] <= 1.0
    assert ep.metrics["gripper_closure_rate"] == pytest.approx(0.5, abs=0.05)


def test_cloud_uri_with_nested_paths(tmp_path, mem_fs):
    """Test parsing from deeply nested cloud paths (common in real S3 buckets)."""
    episodes = [{"task": "nested_task", "task_index": 0, "frame_count": 60}]
    local_root = make_lerobot_v3_dataset(tmp_path / "nested_ds", episodes, fps=30)
    
    # Simulate a real S3 path structure: bucket/year/month/dataset
    _copy_local_to_fsspec(local_root, mem_fs, "/my-bucket/2026/06/robotics-data")
    
    manifest = parse("memory:///my-bucket/2026/06/robotics-data")
    
    assert manifest.total_episodes == 1
    assert manifest.episodes[0].task == "nested_task"