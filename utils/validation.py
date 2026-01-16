from __future__ import annotations
import numpy as np
from typing import Dict, Any


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_agent_pos(pos: np.ndarray) -> np.ndarray:


    if pos.shape != (2,):
        raise ValueError("agent_pos must be shape (2,)")

    if not np.isfinite(pos).all():
        raise ValueError("agent_pos contains NaN or inf")

    x = clamp(float(pos[0]), -0.98, 0.98)
    y = clamp(float(pos[1]), -0.98, 0.98)

    return np.array([x, y], dtype=np.float32)


def validate_pods(arr: np.ndarray) -> np.ndarray:


    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("pods must be shape (N,2)")

    if not np.isfinite(arr).all():
        raise ValueError("pods contain NaN or inf")

    arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32)


def validate_wall_state(state: Dict[str, Any]) -> Dict[str, Any]:


    if "enabled" not in state or "blocking" not in state:
        raise ValueError("wall_state missing required fields")

    return {
        "enabled": bool(state["enabled"]),
        "blocking": bool(state["blocking"]),
    }


FORBIDDEN_FIELDS = {
                                                                         
    "collision_cache",
    "render_cache",
    "spawn_rng_state",
    "trigger_contact_state",
    "wall_rooms",
    "next_food_schedule",
    "physics_debug_surfaces",
    "internal_handles",
}


def validate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:


    out = dict(snapshot)

                               
    for k in list(out.keys()):
        if k in FORBIDDEN_FIELDS:
            del out[k]

                               
    if "agent_pos" in out:
        out["agent_pos"] = validate_agent_pos(
            np.asarray(out["agent_pos"], dtype=float)
        )

                               
    if "pods" in out:
        out["pods"] = validate_pods(
            np.asarray(out["pods"], dtype=float)
        )

                               
    if "wall_state" in out:
        out["wall_state"] = validate_wall_state(out["wall_state"])

                               
    return out
