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
    phase = int(_sget("phase", 1))
    agent_pos = np.array(_sget("agent_pos", [0.0, 0.0]), dtype=np.float32)
    agent_vel = np.array(_sget("agent_vel", [0.0, 0.0]), dtype=np.float32)
    pods = _sget("pods", []) or []
    wall = _sget("wall", None)
    rooms = _sget("rooms", []) or []
    bucket_side = _sget("bucket_side", "left")

    pod_obs = []
    for p in pods:
        try:
            active = bool(p.active) if hasattr(p, "active") else bool(p.get("active", False))
            pos_raw = p.pos if hasattr(p, "pos") else p.get("pos", [0.0, 0.0])
            food = float(p.food) if hasattr(p, "food") else float(p.get("food", 0.0))
            pod_obs.append(
                {
                    "active": active,
                    "pos": np.array(pos_raw, dtype=np.float32),
                    "food": food,
                }
            )
        except Exception:
            continue

    wall_enabled = False
    wall_blocking = False
    wall_open_until = 0.0
    try:
        if wall is not None:
            if hasattr(wall, "enabled"):
                wall_enabled = bool(getattr(wall, "enabled"))
                wall_blocking = bool(getattr(wall, "blocking", False))
                wall_open_until = float(getattr(wall, "open_until", 0.0))
            elif hasattr(wall, "get"):
                wall_enabled = bool(wall.get("enabled", False))
                wall_blocking = bool(wall.get("blocking", False))
                wall_open_until = float(wall.get("open_until", 0.0))
    except Exception:
        wall_enabled = False
        wall_blocking = False
        wall_open_until = 0.0

    room_obs = []
    for r in rooms:
        try:
            rr = np.array(r, dtype=np.float32).reshape(-1)
            if rr.shape[0] >= 4:
                room_obs.append(rr[:4].astype(np.float32))
        except Exception:
            continue

    return {
        "frame": frame,
        "stomach": stomach,
        "phase": phase,
        "agent_pos": agent_pos,
        "agent_vel": agent_vel,
        "pods": pod_obs,
        "wall": {
            "enabled": wall_enabled,
            "blocking": wall_blocking,
            "open_until": wall_open_until,
        },
        "rooms": room_obs,
        "bucket_side": str(bucket_side),
        "oracle_distance": None,
    }
