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

        self.playing = False
        print(f"[PlaybackRunner] load_clip: session={session_id} clip={clip_id} start={self._clip_start} end={self._clip_end}")
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
        print(f"[PlaybackRunner] seek_step -> cursor={self._cursor}")
        self._emit_cursor_moved()

    def play(self):
        self.playing = True
        print("[PlaybackRunner] play() called; playing=True")

    def pause(self):
        self.playing = False
        print("[PlaybackRunner] pause() called; playing=False")

    # ------------------------------------------------------------
    # Playback ticking
    # ------------------------------------------------------------
    def tick(self):
        if not self.playing:
            # Helpful trace when tick is invoked but runner is paused
            print("[PlaybackRunner] tick() called but playing==False; skipping")
            return
        if self._session_id is None:
            return
        if self._cursor >= self._clip_end:
            print("[PlaybackRunner] reached clip end; stopping playback")
            self.playing = False
            return

        print(f"[PlaybackRunner] tick() advancing cursor {self._cursor} -> {self._cursor + 1}")
        self._emit_cursor_moved()
        self._cursor += 1

    # ------------------------------------------------------------
    # Core emission
    # ------------------------------------------------------------
    def _emit_cursor_moved(self):
        if self._session_id is None:
            return

        print(f"[PlaybackRunner] _emit_cursor_moved: session={self._session_id} cursor={self._cursor}")

        # 1) Snapshot for tables, overlays, and inspection
        snapshot = self.store.get_step_snapshot(
            self._session_id,
            self._cursor,
        )

        self.bus.publish(
            "TraceCursorMoved",
            {
                "session_id": self._session_id,
                "clip_id": self._clip_id,
                "step_index": self._cursor,
                "snapshot": snapshot,
            },
        )
        print(f"[PlaybackRunner] published TraceCursorMoved for step={self._cursor}")

        # 2) Raw recorded frame for video-like rendering
        frame = self.store.get_step_frame(
            self._session_id,
            self._cursor,
        )

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
        print(f"[PlaybackRunner] published EnvRenderPacket for step={step_idx}")
