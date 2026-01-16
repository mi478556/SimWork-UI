from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, List


def rolling_window(arr: np.ndarray, window: int) -> np.ndarray:


    if window <= 0 or window > len(arr):
        raise ValueError("Invalid window length")

    shape = (arr.size - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def ema_smooth(arr: np.ndarray, alpha: float = 0.2) -> np.ndarray:


    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def downsample(arr: np.ndarray, stride: int) -> np.ndarray:


    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride == 1:
        return arr
    return arr[::stride]


def event_intervals(mask: Iterable[bool]) -> List[Tuple[int, int]]:


    mask = list(mask)
    intervals = []
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            intervals.append((start, i))
            start = None

    if start is not None:
        intervals.append((start, len(mask)))

    return intervals
