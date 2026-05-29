"""General utilities."""

from pathlib import Path
import json


def ensure_dir(path):
    """Create directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    """Save dictionary-like object as JSON."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
