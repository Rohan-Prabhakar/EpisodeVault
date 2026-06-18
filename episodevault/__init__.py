from episodevault.api import log_training_run
from episodevault.diff.engine import diff
from episodevault.models import DatasetManifest, EpisodeManifest, EpisodeQuality
from episodevault.parsers.lerobot import parse as parse_lerobot
from episodevault.store.lineage_store import LineageStore
from episodevault.store.version_store import VersionStore

__all__ = [
    "parse_lerobot",
    "VersionStore",
    "LineageStore",
    "diff",
    "log_training_run",
    "DatasetManifest",
    "EpisodeManifest",
    "EpisodeQuality",
]

<<<<<<< HEAD
try:
    from importlib.metadata import version
    __version__ = version("episodevault")
except Exception:
    __version__ = "0.2.2"
