"""Configuration constants for the multi-user HAR framework."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RESULTS_DIR = Path("results")

SENSOR_LOCATIONS = ["wrist", "waist", "chest"]
SENSOR_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

LABEL_COLUMNS = [
    "atomic_activity",
    "composite_activity",
    "interaction_type",
    "user_id",
]

DEFAULT_WINDOW_SIZE = 100  # 2 seconds at 50 Hz
DEFAULT_WINDOW_STEP = 50   # 50% overlap

RANDOM_STATE = 42
