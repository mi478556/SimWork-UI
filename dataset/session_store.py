# session_store.py

from __future__ import annotations

import os
import json
import numpy as np
from typing import List
from copy import deepcopy

from dataset.session_types import SessionData, SessionMeta
from dataset.clip_index import ClipIndex

from env.snapshot_state import (
    SnapshotState,
    SNAPSHOT_SCHEMA_VERSION,
    validate_snapshot,
    SnapshotValidationError,
)


class SessionStore:
    """
    Canonical interface to persisted sessions.

    Responsibilities:
    - Owns on-disk layout (dataset/saves/<session_id>/)
    - Saves and loads SessionData
    - Enforces column alignment invariants
    - Enforces snapshot schema compatibility
    - Produces:
        * raw frames
        * validated SnapshotState objects
    - Stores and retrieves thumbnail metadata
    """

    # ----------------------------------------------------
    # Initialization
    # ----------------------------------------------------
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

        # All sessions live here
        self.saves_dir = os.path.join(root_dir, "saves")
        os.makedirs(self.saves_dir, exist_ok=True)

        # Clip index scoped to saves/
        self.clips = ClipIndex(self.saves_dir)

    # ----------------------------------------------------
    # Internal path helpers (PRIVATE)
    # ----------------------------------------------------
    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.saves_dir, session_id)

    def _ensure_session_dir(self, session_id: str):
        os.makedirs(self._session_dir(session_id), exist_ok=True)

    def _session_npz_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "session.npz")

    # ----------------------------------------------------
    # Save / Load
    # ----------------------------------------------------
    def save_session(self, session: SessionData) -> str:
        """
        Persist a fully-formed session to disk.
        This is the ONLY valid write path.
        """

        if session.T == 0:
            raise RuntimeError("Refusing to save empty session")

        session.assert_column_alignment()

        # Ensure session directory exists only when writing
        self._ensure_session_dir(session.session_id)

        path = self._session_npz_path(session.session_id)

        np.savez_compressed(
            path,
            meta=session.meta,

            frames=session.frames,
            stomach=session.stomach,
            agent_pos=session.agent_pos,
            actions=session.actions,

            phases=session.phases,
            food_positions=session.food_positions,

            wall_enabled=session.wall_enabled,
            wall_blocking=session.wall_blocking,

            sequence_index=session.sequence_index,

            oracle_queries=session.oracle_queries,
            oracle_distances=session.oracle_distances,
        )

        return path

    def load_session(self, session_id: str, mmap: bool = True) -> SessionData:
        """
        Load a session and enforce all invariants.
        """

        path = self._session_npz_path(session_id)

        data = np.load(
            path,
            allow_pickle=True,
            mmap_mode="r" if mmap else None,
        )

        meta: SessionMeta = data["meta"].item()

        session = SessionData(
            session_id=session_id,
            meta=meta,

            frames=data["frames"],
            stomach=data["stomach"],
            agent_pos=data["agent_pos"],
            actions=data["actions"],

            phases=data["phases"],
            food_positions=data["food_positions"],

            wall_enabled=data["wall_enabled"],
            wall_blocking=data["wall_blocking"],

            sequence_index=data["sequence_index"],

            oracle_queries=data["oracle_queries"],
            oracle_distances=data["oracle_distances"],
        )

        session.assert_column_alignment()
        return session

    # ----------------------------------------------------
    # Enumeration
    # ----------------------------------------------------
    def list_sessions(self) -> List[str]:
        return sorted(
            d for d in os.listdir(self.saves_dir)
            if os.path.isdir(os.path.join(self.saves_dir, d))
        )

    # ----------------------------------------------------
    # Frame access
    # ----------------------------------------------------
    def get_step_frame(self, session_id: str, step: int):
        """
        Return raw RGB frame (H, W, 3) for a session step.
        """
        session = self.load_session(session_id, mmap=True)
        i = int(step)

        if i < 0 or i >= session.T:
            raise IndexError(f"Step {i} out of bounds for session {session_id}")

        return session.frames[i]

    # ----------------------------------------------------
    # Snapshot access
    # ----------------------------------------------------
    def get_step_snapshot(self, session_id: str, step: int) -> SnapshotState:
        """
        Return a validated SnapshotState for a given step.
        """

        session = self.load_session(session_id, mmap=True)
        meta = session.meta

        if meta.snapshot_schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotValidationError(
                f"Dataset snapshot schema mismatch "
                f"(session={meta.snapshot_schema_version}, "
                f"runtime={SNAPSHOT_SCHEMA_VERSION})"
            )

        i = int(step)
        if i < 0 or i >= session.T:
            raise IndexError(f"Step {i} out of bounds for session {session_id}")

        snapshot: SnapshotState = dict(
            agent_pos=session.agent_pos[i].tolist(),
            agent_vel=None,

            stomach=float(session.stomach[i]),
            phase=int(session.phases[i]),

            pods=session.food_positions[i].tolist(),

            wall_state=dict(
                enabled=bool(session.wall_enabled[i]),
                blocking=bool(session.wall_blocking[i]),
            ),

            sequence_index=int(session.sequence_index[i]),

            provenance_meta=deepcopy(meta.provenance),
            snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        )

        validate_snapshot(snapshot)
        return snapshot