from __future__ import annotations
import re
from pathlib import Path
import pytest
from click.testing import CliRunner
import fsspec

from episodevault.cli.main import cli
from tests.fixtures import make_lerobot_v3_dataset


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colors/styles) from rich output."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


@pytest.fixture
def runner():
    return CliRunner()


def _copy_local_to_memory(local_path: Path, mem_path: str):
    """Helper to copy a local directory to fsspec memory filesystem."""
    mem_fs = fsspec.filesystem("memory")
    if mem_fs.exists(mem_path):
        mem_fs.rm(mem_path, recursive=True)
    mem_fs.mkdir(mem_path)
    for item in local_path.rglob("*"):
        if item.is_file():
            relative = item.relative_to(local_path)
            target = f"{mem_path}/{relative}".replace("\\", "/")
            parent = "/".join(target.split("/")[:-1])
            if parent and not mem_fs.exists(parent):
                mem_fs.makedirs(parent, exist_ok=True)
            mem_fs.put_file(str(item), target)


# =============================================================================
# Local Path Tests (Backward Compatibility)
# =============================================================================

def test_track_and_commit_local(tmp_path, runner):
    """Test tracking and committing a local dataset."""
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
    local_root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    
    # Track
    result = runner.invoke(cli, ["track", str(local_root)])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "Tracking" in out
    assert "Store initialised" in out
    
    # Commit
    result = runner.invoke(cli, ["commit", str(local_root), "-m", "initial commit"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "Committed v1.0" in out
    assert "1 episodes" in out


def test_diff_local(tmp_path, runner):
    """Test diffing two local versions."""
    episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
    local_root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    
    runner.invoke(cli, ["track", str(local_root)])
    runner.invoke(cli, ["commit", str(local_root), "-m", "v1"])
    
    # Modify dataset (simulate v2 by adding an episode)
    episodes_v2 = [
        {"task": "grasp", "task_index": 0, "frame_count": 60},
        {"task": "place", "task_index": 1, "frame_count": 60},
    ]
    make_lerobot_v3_dataset(tmp_path / "ds", episodes_v2, fps=30)
    
    runner.invoke(cli, ["commit", str(local_root), "-m", "v2"])
    
    # Diff
    result = runner.invoke(cli, ["diff", "v1.0", "v2.0", str(local_root)])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "Dataset diff: v1.0" in out
    assert "Episodes added" in out


def test_anomalies_local(tmp_path, runner):
    """Test anomaly detection on a local dataset."""
    episodes = [
        {"task": "grasp", "task_index": 0, "frame_count": 90},
        {"task": "grasp", "task_index": 0, "frame_count": 1},  # corrupted
    ]
    local_root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
    
    result = runner.invoke(cli, ["anomalies", str(local_root)])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "anomalous episode" in out
    assert "corrupted" in out


# =============================================================================
# Cloud URI Tests (using memory:// to simulate S3/GCS)
# =============================================================================

def test_track_and_commit_cloud_uri(tmp_path, runner):
    """Test tracking and committing a dataset via a cloud-like URI."""
    # Use isolated_filesystem so the .episodevault store doesn't pollute the project dir
    with runner.isolated_filesystem():
        episodes = [{"task": "cloud_task", "task_index": 0, "frame_count": 60}]
        local_root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
        
        mem_path = "/cloud_ds"
        _copy_local_to_memory(local_root, mem_path)
        uri = f"memory://{mem_path}"
        
        # Track
        result = runner.invoke(cli, ["track", uri])
        assert result.exit_code == 0, result.output
        out = strip_ansi(result.output)
        assert "Tracking" in out
        assert "Cloud datasets are tracked by URI" in out
        assert Path(".episodevault/cloud_uri.txt").exists()
        
        # Commit
        result = runner.invoke(cli, ["commit", uri, "-m", "cloud v1"])
        assert result.exit_code == 0, result.output
        out = strip_ansi(result.output)
        assert "Committed v1.0" in out
        assert "1 episodes" in out


def test_diff_cloud_uri(tmp_path, runner):
    """Test diffing versions of a cloud-hosted dataset."""
    with runner.isolated_filesystem():
        # Create v1
        v1_episodes = [{"task": "grasp", "task_index": 0, "frame_count": 60}]
        v1_root = make_lerobot_v3_dataset(tmp_path / "v1", v1_episodes, fps=30)
        _copy_local_to_memory(v1_root, "/cloud_diff_ds")
        uri = "memory:///cloud_diff_ds"
        
        runner.invoke(cli, ["track", uri])
        runner.invoke(cli, ["commit", uri, "-m", "v1"])
        
        # Create v2 and overwrite the same memory path
        v2_episodes = [
            {"task": "grasp", "task_index": 0, "frame_count": 60},
            {"task": "place", "task_index": 1, "frame_count": 60},
        ]
        v2_root = make_lerobot_v3_dataset(tmp_path / "v2", v2_episodes, fps=30)
        _copy_local_to_memory(v2_root, "/cloud_diff_ds")
        
        runner.invoke(cli, ["commit", uri, "-m", "v2"])
        
        # Diff
        result = runner.invoke(cli, ["diff", "v1.0", "v2.0", uri])
        assert result.exit_code == 0, result.output
        out = strip_ansi(result.output)
        assert "Dataset diff: v1.0" in out
        assert "Episodes added" in out


def test_anomalies_cloud_uri(tmp_path, runner):
    """Test anomaly detection on a cloud-hosted dataset."""
    with runner.isolated_filesystem():
        episodes = [
            {"task": "grasp", "task_index": 0, "frame_count": 90},
            {"task": "grasp", "task_index": 0, "frame_count": 1},  # corrupted
        ]
        local_root = make_lerobot_v3_dataset(tmp_path / "ds", episodes, fps=30)
        _copy_local_to_memory(local_root, "/cloud_anom")
        uri = "memory:///cloud_anom"
        
        result = runner.invoke(cli, ["anomalies", uri])
        assert result.exit_code == 0, result.output
        out = strip_ansi(result.output)
        assert "anomalous episode" in out
        assert "corrupted" in out