from __future__ import annotations

import numpy as np


DEFAULT_NUMPY_DTYPE = np.float32
DEFAULT_MAX_MCS_ROUNDS = 3
DEFAULT_MCS_TIMEOUT_SECONDS = 10_000_000
DEFAULT_SCAFFOLD_FAILURE_DISTANCE = 1.0
DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
SPLIT_NAMES = ("train", "val", "test")
FLOAT_TOLERANCE = 1e-8
MAX_SUBSTRUCT_MATCHES = 100000
