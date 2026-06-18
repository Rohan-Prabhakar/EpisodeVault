# EpisodeVault: find out exactly why your robot model regressed.

[![PyPI](https://img.shields.io/pypi/v/episodevault)](https://pypi.org/project/episodevault/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LeRobot v3](https://img.shields.io/badge/LeRobot-v3-green.svg)](https://github.com/huggingface/lerobot)


## The problem

Every robotics ML engineer has retrained a model and watched performance drop with no clear cause. DVC tracks which files changed. MLflow tracks which hyperparameters ran. Nobody tracks what changed at the episode level, which tasks dropped out, which quality metrics shifted, which task distribution moved between v1 and v2 of your dataset.

EpisodeVault fills that gap.


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


## Install

```bash
pip install episodevault
```

Requires Python 3.10+. Key dependencies: `pyarrow`, `pandas`, `duckdb`, `click`, `rich`, `pydantic`.


## Quickstart

```bash
# Start tracking a local LeRobot dataset
episodevault track ./my_dataset

# Snapshot the current state with a message
episodevault commit -m "added 500 kitchen episodes"

# Compare two snapshots
episodevault diff v1.0 v2.0

# Write a shareable, self-contained HTML report alongside the diff
episodevault diff v1.0 v2.0 --html audit.html

# Compare a local version against a dataset hosted on the HuggingFace Hub
episodevault diff-hub v2.0 lerobot/aloha_static_pro_pencil

# Flag outlier episodes (too short, jerky, desynced) before training
episodevault anomalies

# Show version history as a tree, then jump straight to a diff
episodevault tree

# Find what dataset a model was trained on and diff against the prior version
episodevault blame model_v3
```

`track` initializes a `.episodevault/` store inside your dataset directory. `commit` snapshots the episode manifest (not raw sensor data -- fast). `diff` computes task distribution shift and quality deltas between any two versions. `blame` looks up which dataset version trained a given model and diffs it against the version before.


## Cloud-Native Storage (S3, GCS, Azure)

EpisodeVault doesn't need to download terabytes of video to diff your dataset. It reads metadata directly from cloud object storage using `fsspec`. You can track, commit, and diff datasets stored in AWS S3, Google Cloud Storage, or Azure Blob Storage without pulling the raw sensor data to your local machine.

```bash
# Track and commit a dataset directly in AWS S3
episodevault track s3://my-robotics-bucket/datasets/v1
episodevault commit s3://my-robotics-bucket/datasets/v1 -m "initial cloud snapshot"

# Diff two versions stored in the cloud
episodevault diff v1.0 v2.0 s3://my-robotics-bucket/datasets/v1

# Run anomaly detection on a cloud dataset
episodevault anomalies gs://my-gcs-bucket/datasets/v2

```

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


## Custom quality metrics

EpisodeVault ships two built-in per-episode metrics: `action_smoothness` (1 / (1 + mean jerk); 1.0 is perfectly smooth) and `gripper_closure_rate`. The point is that **you define your own**. A quality metric is any function that takes one episode's per-frame DataFrame and returns a float (or `None` to abstain when the columns it needs are not present). Register it once, and EpisodeVault computes it for every episode at parse time, stores it in the version snapshot, and diffs it across versions automatically.

```python
from episodevault.parsers.lerobot import register_quality_metric
import numpy as np

def wrist_travel(frames):
    """Total distance the wrist joint travels over the episode."""
    if "observation.state" not in frames.columns:
        return None
    state = np.stack(frames["observation.state"].to_numpy())
    wrist = state[:, 3]                       # joint index 3 = wrist
    return float(np.abs(np.diff(wrist)).sum())

register_quality_metric("wrist_travel", wrist_travel)
```

After registering, `episodevault commit` records `wrist_travel` for every episode, `episodevault diff` reports how its dataset-wide average shifted between versions, and `episodevault anomalies` will flag episodes whose `wrist_travel` is a statistical outlier. No extra wiring. Each metric value is also available programmatically on `episode.metrics`.

Register your metrics in a small Python module (or your `conftest`/startup script) that runs before you invoke the parser. The frame DataFrame contains whatever columns your LeRobot data Parquet has — typically `action`, `observation.state`, `timestamp` — with list-valued columns (e.g. `action`) stacked per row.


## Anomaly detection

`episodevault anomalies` flags episodes that are likely bad data, so you can prune them before training:

```
3 anomalous episode(s):
┏━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Episode        ┃ Task  ┃ Severity ┃ Reasons                          ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ episode_000017 │ grasp │     0.90 │ quality=corrupted                │
│ episode_000004 │ grasp │     0.62 │ unusually short (z=-4.1)         │
│ episode_000031 │ place │     0.51 │ action_smoothness=0.12 outlier   │
└────────────────┴───────┴──────────┴──────────────────────────────────┘
```

It combines a robust (median/MAD) z-score over duration, frame count, camera sync, and every custom metric with rule-based checks (corrupted quality, severely desynced cameras). Pass `--version v2.0` to inspect a committed snapshot instead of re-parsing the working tree.


## Version history tree

`episodevault tree` renders your commit history and then prompts you to jump directly to a diff:

```
my_dataset
├── v1.0  initial import       (1 eps · 2026-06-11 20:23)
├── v2.0  add place task       (3 eps · 2026-06-11 20:31)
└── v3.0  kitchen heavy run    (6 eps · 2026-06-11 20:45)

Diff two versions? Enter e.g. v1.0 v2.0, or press Enter to skip.
> v1.0 v3.0

Dataset diff: v1.0 → v3.0
...
```

Press Enter to skip the diff and just view the tree. Add `--html report.html` to also export a full HTML report for the chosen diff.


## Shareable HTML reports

`episodevault diff v1.0 v2.0 --html audit.html` writes a self-contained HTML report containing:

- **Version history graph**: a visual timeline of all commits so recipients can see where this diff sits
- **Distribution and quality bar charts**: before vs. after, inline SVG
- **Custom metric shifts**: every metric you've registered, diffed across versions
- **Flagged episodes table**: anomalies detected in the after version

No external scripts, fonts, or network requests — safe to email or archive for non-technical stakeholders. `diff-hub` and `tree` both support `--html` too.


## Diff against the HuggingFace Hub

`episodevault diff-hub <local-version> <repo-id>` downloads a Hub-hosted LeRobot dataset and diffs your committed local version against it — useful for catching drift between what you're training on and what's published upstream.

```bash
episodevault diff-hub v2.0 lerobot/aloha_static_pro_pencil --revision main --html audit.html
```

Requires the optional `huggingface_hub` package (`pip install huggingface_hub`).


## Compatibility

Tested against real HuggingFace LeRobot v3 datasets:

| Dataset          | Robot  | Format     | Episodes | Parse time | Status |
|------------------|--------|------------|----------|------------|--------|
| aloha_pencil     | aloha  | LeRobot v3 | 25       | 0.33s      | OK     |
| aloha_shrimp     | aloha  | LeRobot v3 | 18       | 0.38s      | OK     |
| so100_stacking   | so100  | LeRobot v3 | 56       | 0.65s      | OK     |
| aloha_cabinet    | aloha  | LeRobot v3 | 85       | 2.65s      | OK     |

Parse time is for the episode manifest only. Raw sensor data (video, joint trajectories) is never loaded.


## How it works

- Parses episode manifests (`meta/episodes/`, `meta/tasks.parquet`, `meta/info.json`) without loading raw sensor data -- sub-second parse regardless of frame count or video size.
- Snapshots manifests into a version store on every `commit` -- diff and time travel are built in from the start.
- Diff engine computes task distribution shift and quality deltas between any two snapshots -- regression candidates are ranked by a normalized severity score and the top few are surfaced, not asserted as proven causes.
- **Cloud-native:** Uses `fsspec` to read metadata directly from S3, GCS, and Azure. It only pulls the tiny Parquet manifests over the network, keeping cloud egress costs and latency near zero.


## Credits

- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) team for the v3 dataset format that EpisodeVault parses.
- Berkeley AutoLab (Kaiyuan Chen et al.) for [Robo-DM / fog_x](https://github.com/BerkeleyAutomation/fog_x), prior work on robot dataset management.
- [score_lerobot_episodes](https://github.com/RobotData/score-lerobot-episodes) by RobotData for quality signal methodology.
- [Evidently AI](https://github.com/evidentlyai/evidently) for drift detection methodology that informed the distribution shift logic.


## License

MIT. See [LICENSE](LICENSE).
