"""Dataset loading utilities."""

from pathlib import Path
import json
import pandas as pd


def load_raw_samples(data_dir: str | Path) -> pd.DataFrame:
    """Load sample-level IMU data."""
    data_dir = Path(data_dir)
    path = data_dir / "raw_imu_samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path)


def load_session_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load session-level metadata."""
    return pd.read_csv(Path(data_dir) / "session_metadata.csv")


def load_activity_segments(data_dir: str | Path) -> pd.DataFrame:
    """Load segment-level annotations."""
    return pd.read_csv(Path(data_dir) / "activity_segments.csv")


def load_session_splits(data_dir: str | Path) -> pd.DataFrame:
    """Load predefined train/validation/test splits."""
    return pd.read_csv(Path(data_dir) / "session_splits.csv")


def load_label_dictionary(data_dir: str | Path) -> dict:
    """Load label definitions."""
    with open(Path(data_dir) / "label_dictionary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def attach_split(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach split column to sample-level dataframe."""
    return df.merge(splits, on="session_id", how="left")
