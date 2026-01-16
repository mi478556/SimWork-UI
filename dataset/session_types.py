# session_types.py


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ProvenanceMeta:
    origin: str

    branch_parent_id: Optional[str]
    branch_depth: int

    source_session_id: Optional[str]
    source_clip_id: Optional[str]
    source_step_index: Optional[int]

    agent_id: Optional[str]
    notes: Optional[str]


@dataclass
class SessionMeta:
    session_id: str
    created_at: str
    snapshot_schema_version: str
    fps: int
    dt: float
    env_version: str
    pod_count: int
    wall_mode: str
    total_deaths: int
    total_food_eaten: int
    provenance: ProvenanceMeta


@dataclass
class SessionData:
    session_id: str
    meta: SessionMeta

    frames: Any
    stomach: Any
    agent_pos: Any
    actions: Any

    phases: Any
    food_positions: Any

    wall_enabled: Any
    wall_blocking: Any

    sequence_index: Any

    oracle_queries: Any
    oracle_distances: Any

    def __post_init__(self):
        # Ensure arrays are numpy-compatible
        pass

    @property
    def T(self) -> int:
        try:
            return int(self.frames.shape[0])
        except Exception:
            return 0

    def assert_column_alignment(self):
        # Basic sanity checks for column lengths
        t = self.T
        if t == 0:
            return
        cols = [
            self.stomach,
            self.agent_pos,
            self.actions,
            self.phases,
            self.food_positions,
            self.wall_enabled,
            self.wall_blocking,
            self.sequence_index,
        ]
        for c in cols:
            if len(c) != t:
                raise RuntimeError("Column length mismatch in SessionData")

