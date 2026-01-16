# playback_runner.py

from __future__ import annotations

from typing import Optional

from viewer.event_bus import EventBus
from dataset.session_store import SessionStore
from dataset.clip_index import ClipRef

from engine.environment_runtime import DT


class PlaybackRunner:


    def __init__(self, store: SessionStore, bus: EventBus):
        self.store = store
        self.bus = bus

                              
        self._session_id: Optional[str] = None
        self._clip_id: Optional[str] = None
        self._clip_start: int = 0
        self._clip_end: int = 0             
        self._cursor: int = 0                                    

        self.playing: bool = False

                                                          
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

                                      
        self._emit_cursor_moved()

                                                          
    def seek_step(self, step: int):


        if self._session_id is None:
            return

        step_abs = int(step)
        step_abs = max(self._clip_start, min(step_abs, self._clip_end - 1))
        self._cursor = step_abs

        self._emit_cursor_moved()

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

                                                          
    def tick(self):


        if not self.playing:
            return
        if self._session_id is None:
            return

        if self._cursor >= self._clip_end:
                                                    
            self.playing = False
            return

        self._emit_cursor_moved()
        self._cursor += 1

                                                          
    def _emit_cursor_moved(self):


        if self._session_id is None:
            return

        snapshot = self.trace_store.get_step_snapshot(
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

        # Also publish an EnvRenderPacket so UI panels can render playback frames
        step_idx = int(snapshot.get("step_index", snapshot.get("sequence_index", self._cursor)))
        sim_time = step_idx * DT

        packet = {
            "frame": None,
            "snapshot": snapshot,
            "telemetry": {},
            "sim_time": sim_time,
            "step_index": step_idx,
        }

        self.bus.publish("EnvRenderPacket", packet)
