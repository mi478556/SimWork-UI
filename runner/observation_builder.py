from typing import Any, Dict
import numpy as np


def build_observation(snapshot: Dict[str, Any], frame: np.ndarray) -> Dict[str, Any]:
    def _sget(key, default=None):
        try:
            if hasattr(snapshot, "get"):
                return snapshot.get(key, default)
        except Exception:
            pass
        try:
            return getattr(snapshot, key)
        except Exception:
            return default

    stomach = float(_sget("stomach", 0.0))

    return {
        "frame": frame,
        "stomach": stomach,
        "oracle_distance": None,
    }
