# EpisodeVault: find out exactly why your robot model regressed.

[![PyPI](https://img.shields.io/pypi/v/episodevault)](https://pypi.org/project/episodevault/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LeRobot v3](https://img.shields.io/badge/LeRobot-v3-green.svg)](https://github.com/huggingface/lerobot)

---

## The problem

Every robotics ML engineer has retrained a model and watched performance drop with no clear cause. DVC tracks which files changed. MLflow tracks which hyperparameters ran. Nobody tracks what changed at the episode level, which tasks dropped out, which quality metrics shifted, which task distribution moved between v1 and v2 of your dataset.

EpisodeVault fills that gap.

---

## What EpisodeVault does

Run `episodevault diff v1.0 v2.0` and get this:

```
Dataset diff: v1.0 → v2.0
────────────────────────────────────────────────────
Episodes added:    +0
Episodes removed:  -7

Distribution shift:
  factory_pick                     2 → 6  ↑ 200%  ⚠️
  kitchen_grasp                    4 → 1  ↓ 75%  ⚠️

Quality metrics:
  avg episode length:    3.7s → 3.0s  ↓
  success_rate:          0.88 → 0.38  ↓
  camera_sync_score:     1.00 → 1.00  →

Regression candidates (ranked by magnitude; correlate with your eval):
  - 'kitchen_grasp' episodes dropped 75% (4 → 1). Restore from prior
    version if this task is in your eval benchmark.
  - Success rate fell 50% (0.88 → 0.38). New episodes may contain failed
    demonstrations. Run score_lerobot_episodes to identify low-quality additions.
```

---

## Install

```bash
pip install episodevault
```

Requires Python 3.10+. Key dependencies: `pyarrow`, `pandas`, `duckdb`, `click`, `rich`, `pydantic`.

---

## Quickstart

```bash
# Start tracking a local LeRobot dataset
episodevault track ./my_dataset

# Snapshot the current state with a message
episodevault commit -m "added 500 kitchen episodes"

# Compare two snapshots
episodevault diff v1.0 v2.0

# Find what dataset a model was trained on and diff against the prior version
episodevault blame model_v3
```

`track` initializes a `.episodevault/` store inside your dataset directory. `commit` snapshots the episode manifest (not raw sensor data -- fast). `diff` computes task distribution shift and quality deltas between any two versions. `blame` looks up which dataset version trained a given model and diffs it against the version before.

---

## Python API

Log a training run from your training script so `blame` can trace it back:

```python
import episodevault as ev

ev.log_training_run(
    model_version="model_v3",
    dataset_version="v2.0",
    framework="lerobot"
)
```

One call. That's all `blame` needs.

---

## Compatibility

Tested against real HuggingFace LeRobot v3 datasets:

| Dataset          | Robot  | Format     | Episodes | Parse time | Status |
|------------------|--------|------------|----------|------------|--------|
| aloha_pencil     | aloha  | LeRobot v3 | 25       | 0.33s      | OK     |
| aloha_shrimp     | aloha  | LeRobot v3 | 18       | 0.38s      | OK     |
| so100_stacking   | so100  | LeRobot v3 | 56       | 0.65s      | OK     |
| aloha_cabinet    | aloha  | LeRobot v3 | 85       | 2.65s      | OK     |

Parse time is for the episode manifest only. Raw sensor data (video, joint trajectories) is never loaded.

---

## How it works

- Parses episode manifests (`meta/episodes/`, `meta/tasks.parquet`, `meta/info.json`) without loading raw sensor data -- sub-second parse regardless of frame count or video size.
- Snapshots manifests into a version store on every `commit` -- diff and time travel are built in from the start.
- Diff engine computes task distribution shift and quality deltas between any two snapshots -- regression candidates are ranked by a normalized severity score and the top few are surfaced, not asserted as proven causes.


---

## Credits

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) team for the v3 dataset format that EpisodeVault parses.
- Berkeley AutoLab (Kaiyuan Chen et al.) for [Robo-DM / fog_x](https://github.com/BerkeleyAutomation/fog_x), prior work on robot dataset management.
- [score_lerobot_episodes](https://github.com/RobotData/score-lerobot-episodes) by RobotData for quality signal methodology.
- [Evidently AI](https://github.com/evidentlyai/evidently) for drift detection methodology that informed the distribution shift logic.

---

## License

MIT. See [LICENSE](LICENSE).
