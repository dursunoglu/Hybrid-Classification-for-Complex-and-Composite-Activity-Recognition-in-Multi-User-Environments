"""Feature extraction helpers.

The main feature extraction is implemented in preprocessing.create_windows.
This file is reserved for additional feature functions.
"""

import numpy as np


def signal_energy(x):
    """Mean squared energy of a signal."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x ** 2))


def zero_crossing_rate(x):
    """Compute zero-crossing rate."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))
