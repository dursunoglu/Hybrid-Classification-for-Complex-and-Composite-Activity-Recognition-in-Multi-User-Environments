"""Preprocessing and sliding-window segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from config import SENSOR_LOCATIONS, SENSOR_CHANNELS


def sensor_columns() -> list[str]:
    """Return all sensor channel column names."""
    return [f"{loc}_{ch}" for loc in SENSOR_LOCATIONS for ch in SENSOR_CHANNELS]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values from simulated dropout.

    The method uses forward fill and backward fill within each session/user,
    followed by zero fill as a fallback.
    """
    df = df.copy()
    cols = sensor_columns()
    df[cols] = (
        df.groupby(["session_id", "user_id"], group_keys=False)[cols]
        .apply(lambda x: x.ffill().bfill())
    )
    df[cols] = df[cols].fillna(0.0)
    return df


def majority_label(values: pd.Series):
    """Return majority label in a window."""
    return values.value_counts().idxmax()


def create_windows(
    df: pd.DataFrame,
    label_col: str,
    window_size: int = 100,
    step: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create fixed-length windows and majority labels.

    Each window is generated independently for each session/user stream.
    """
    cols = sensor_columns()
    rows = []
    labels = []

    for (session_id, user_id), group in df.groupby(["session_id", "user_id"]):
        group = group.sort_values("sample_index").reset_index(drop=True)
        for start in range(0, max(0, len(group) - window_size + 1), step):
            end = start + window_size
            window = group.iloc[start:end]
            if len(window) < window_size:
                continue
            feat = {
                "session_id": session_id,
                "user_id": user_id,
                "start_sample": int(window["sample_index"].iloc[0]),
                "end_sample": int(window["sample_index"].iloc[-1]),
            }
            for col in cols:
                x = window[col].to_numpy(dtype=float)
                feat[f"{col}_mean"] = np.mean(x)
                feat[f"{col}_std"] = np.std(x)
                feat[f"{col}_min"] = np.min(x)
                feat[f"{col}_max"] = np.max(x)
                feat[f"{col}_energy"] = np.mean(x ** 2)
            rows.append(feat)
            labels.append(majority_label(window[label_col]))

    return pd.DataFrame(rows), pd.Series(labels, name=label_col)
