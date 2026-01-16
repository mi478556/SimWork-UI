

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


SNAPSHOT_SCHEMA_VERSION = 1


REQUIRED_FIELDS = {
    "schema_version",
    "step_index",
    "agent_pos",
    "agent_vel",
    "stomach",
    "phase",
    "total_food_eaten_life",
    "pods",
    "wall",
    "sequence_index",
    "bucket_side",
}

 
DERIVED_FIELDS = {
    "wall_rooms",
    "collision_cache",
    "trigger_contact_state",
    "spawn_rng_state",
    "next_food_schedule",
    "sim_tick_substep",
}

FORBIDDEN_FIELDS = (
    DERIVED_FIELDS
    | {
        "physics_debug_surfaces",
        "render_cache",
        "internal_handles",
    }
)


@dataclass(frozen=True)
class ProvenanceMeta:


    source: str                                                                
    session_id: Optional[str]
    clip_id: Optional[str]
    source_step: Optional[int]

    branch_parent_id: Optional[str]
    branch_depth: int

    modifications: Dict[str, str]


@dataclass
class PodState:
    active: bool
    pos: np.ndarray
    spawn: np.ndarray
    food: float
    sequence_stage: int


@dataclass
class WallState:
    enabled: bool
    blocking: bool
    open_until: float


@dataclass
class EnvStateSnapshot:


    schema_version: int

    step_index: int

    agent_pos: np.ndarray
    agent_vel: np.ndarray

    stomach: float
    phase: int
    total_food_eaten_life: int

    pods: List[PodState]

    wall: WallState

    sequence_index: int
    bucket_side: str
    rooms: List[np.ndarray]


@dataclass
class SnapshotAdvisory:


    level: str
    message: str


def snapshot_from_runtime_dict(d: Dict[str, Any]) -> EnvStateSnapshot:


    pods = [
        PodState(
            active=bool(p["active"]),
            pos=np.array(p["pos"], dtype=np.float32),
            spawn=np.array(p["spawn"], dtype=np.float32),
            food=float(p["food"]),
            sequence_stage=int(p["sequence_stage"]),
        )
        for p in d["pods"]
    ]

    wall = WallState(
        enabled=bool(d["wall"]["enabled"]),
        blocking=bool(d["wall"]["blocking"]),
        open_until=float(d["wall"]["open_until"]),
    )

    rooms = []
    for r in d.get("rooms", []) or []:
        # accept either list-like or ndarray-like
        rooms.append(np.array(r, dtype=np.float32))

    return EnvStateSnapshot(
        schema_version=int(d.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),

        step_index=int(d["step_index"]),

        agent_pos=np.array(d["agent_pos"], dtype=np.float32),
        agent_vel=np.array(d["agent_vel"], dtype=np.float32),

        stomach=float(d["stomach"]),
        phase=int(d["phase"]),
        total_food_eaten_life=int(d["total_food_eaten_life"]),

        pods=pods,
        wall=wall,

        sequence_index=int(d["sequence_index"]),
        bucket_side=str(d["bucket_side"]),
        rooms=rooms,
    )


def snapshot_to_runtime_dict(s: EnvStateSnapshot) -> Dict[str, Any]:


    return {
        "schema_version": s.schema_version,

        "step_index": s.step_index,

        "agent_pos": s.agent_pos.copy(),
        "agent_vel": s.agent_vel.copy(),

        "stomach": float(s.stomach),
        "phase": int(s.phase),
        "total_food_eaten_life": int(s.total_food_eaten_life),

        "pods": [
            {
                "active": p.active,
                "pos": p.pos.copy(),
                "spawn": p.spawn.copy(),
                "food": float(p.food),
                "sequence_stage": int(p.sequence_stage),
            }
            for p in s.pods
        ],

        "wall": {
            "enabled": s.wall.enabled,
            "blocking": s.wall.blocking,
            "open_until": float(s.wall.open_until),
        },

        "sequence_index": s.sequence_index,
        "bucket_side": s.bucket_side,
        "rooms": [r.copy().tolist() for r in (s.rooms or [])],
    }


SnapshotState = Dict[str, Any]


class SnapshotValidationError(Exception):
    pass


def validate_snapshot(snap: SnapshotState):


    required = [
        "agent_pos",
        "stomach",
        "phase",
        "pods",
                                                        
        "wall_state",
        "sequence_index",
        "snapshot_schema_version",
    ]

    if not isinstance(snap, dict):
        raise SnapshotValidationError("Snapshot must be a dict-like object")

    for k in required:
        if k not in snap:
            raise SnapshotValidationError(f"Snapshot missing required key: {k}")

    if int(snap.get("snapshot_schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotValidationError(
            f"Snapshot schema version mismatch (got={snap.get('snapshot_schema_version')}, expected={SNAPSHOT_SCHEMA_VERSION})"
        )

    return True
