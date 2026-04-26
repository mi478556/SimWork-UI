# playback_runner.py

from __future__ import annotations
from typing import Optional

from viewer.event_bus import EventBus
from dataset.session_store import SessionStore
from dataset.clip_index import ClipRef
from engine.environment_runtime import DT


class PlaybackRunner:
    """
    Deterministic playback of recorded sessions.

    Responsibilities:
    - Map (session_id, clip_id) -> absolute step range
    - Maintain a cursor over recorded steps
    - Emit:
        * TraceCursorMoved (symbolic state)
        * EnvRenderPacket (raw recorded frame + snapshot)
    - Never mutates environment state
    """

    def __init__(self, store: SessionStore, bus: EventBus):
        self.store = store
        self.bus = bus

        self._session_id: Optional[str] = None
        self._clip_id: Optional[str] = None
        self._clip_start: int = 0
        self._clip_end: int = 0
        self._cursor: int = 0
        self._session = None

        self.playing: bool = False

    # ------------------------------------------------------------
    # Clip selection
    # ------------------------------------------------------------
    def load_clip(self, session_id: str, clip_id: str):
        clips = self.store.clips.load(session_id)
        if clip_id not in clips:
            raise KeyError(f"Clip {clip_id} not found in session {session_id}")

        ref: ClipRef = clips[clip_id]

        self._session_id = session_id
        self._clip_id = clip_id
        self._clip_start = int(ref.start)
        self._clip_end = int(ref.end)
        self._cursor = self._clip_start
        self._session = self.store.load_session(session_id, mmap=True)

        self.playing = False
        self._emit_cursor_moved()

    # ------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------
    def seek_step(self, step: int):
        if self._session_id is None:
            return

        step_abs = int(step)
        step_abs = max(self._clip_start, min(step_abs, self._clip_end - 1))
        self._cursor = step_abs
        if self._cursor >= self._clip_end - 1:
            self.playing = False
        self._emit_cursor_moved()

    def play(self):
        if self._session_id is None:
            return
        if self._clip_end > self._clip_start and self._cursor >= self._clip_end - 1:
            self._cursor = self._clip_start
            self._emit_cursor_moved()
        self.playing = True

    def pause(self):
        self.playing = False

    def step_once(self):
        if self._session_id is None:
            return
        if self._cursor >= self._clip_end - 1:
            self.playing = False
            return
        self._cursor += 1
        self._emit_cursor_moved()
        if self._cursor >= self._clip_end - 1:
            self.playing = False

    # ------------------------------------------------------------
    # Playback ticking
    # ------------------------------------------------------------
    def tick(self):
        if not self.playing:
            return
        if self._session_id is None:
            return
        if self._cursor >= self._clip_end - 1:
            self.playing = False
            return

        self._cursor += 1
        self._emit_cursor_moved()
        if self._cursor >= self._clip_end - 1:
            self.playing = False

    # ------------------------------------------------------------
    # Core emission
    # ------------------------------------------------------------
    def _emit_cursor_moved(self):
        if self._session_id is None:
            return

        # 1) Snapshot for tables, overlays, and inspection
        if self._session is None:
            self._session = self.store.load_session(self._session_id, mmap=True)

        snapshot = self.store.build_step_snapshot(self._session, self._cursor)

        self.bus.publish(
            "TraceCursorMoved",
            {
                "session_id": self._session_id,
                "clip_id": self._clip_id,
                "step_index": self._cursor,
                "snapshot": snapshot,
            },
        )

        # 2) Raw recorded frame for video-like rendering
        frame = self._session.frames[int(self._cursor)]

        # Absolute step index ensures time coherence
        step_idx = int(self._cursor)
        sim_time = step_idx * DT

        packet = {
            "frame": frame,      # always present during playback
            "snapshot": snapshot,
            "telemetry": {},
            "sim_time": sim_time,
            "step_index": step_idx,
        }

        self.bus.publish("EnvRenderPacket", packet)
