"""Sequence-window utilities for deep learning baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SENSOR_LOCATIONS, SENSOR_CHANNELS


def sensor_columns() -> list[str]:
    """Return all raw IMU sensor columns."""
    return [f"{loc}_{ch}" for loc in SENSOR_LOCATIONS for ch in SENSOR_CHANNELS]


def majority_label(values: pd.Series):
    """Return majority label in a window."""
    return values.value_counts().idxmax()


def create_sequence_windows(
    df: pd.DataFrame,
    label_col: str,
    window_size: int = 100,
    step: int = 50,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sequence windows for neural baselines.

    Returns
    -------
    X : np.ndarray
        Shape: (num_windows, window_size, num_channels)
    y : np.ndarray
        String/object labels.
    meta : pd.DataFrame
        Window metadata with session_id, user_id, start_sample, and end_sample.
    """
    cols = sensor_columns()
    X_windows = []
    y_labels = []
    meta_rows = []

    for (session_id, user_id), group in df.groupby(["session_id", "user_id"]):
        group = group.sort_values("sample_index").reset_index(drop=True)

        for start in range(0, max(0, len(group) - window_size + 1), step):
            end = start + window_size
            window = group.iloc[start:end]
            if len(window) < window_size:
                continue

            X = window[cols].to_numpy(dtype=np.float32)
            y = majority_label(window[label_col])

            X_windows.append(X)
            y_labels.append(y)
            meta_rows.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "start_sample": int(window["sample_index"].iloc[0]),
                    "end_sample": int(window["sample_index"].iloc[-1]),
                }
            )

    if not X_windows:
        return (
            np.empty((0, window_size, len(cols)), dtype=np.float32),
            np.array([], dtype=object),
            pd.DataFrame(columns=["session_id", "user_id", "start_sample", "end_sample"]),
        )

    return np.stack(X_windows), np.array(y_labels, dtype=object), pd.DataFrame(meta_rows)
