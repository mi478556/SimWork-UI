                       

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from dataset.session_store import SessionStore
from dataset.clip_index import ClipRef
from viewer.event_bus import EventBus


class TracePlayerPanel(QWidget):

    def __init__(self, store: SessionStore, bus: EventBus):
        super().__init__()

        # basic layout so this widget can be added to Qt layouts
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.store = store
        self.bus = bus

                              
        self.session_id: Optional[str] = None
        self.clip: Optional[ClipRef] = None

        self.clip_start: int = 0
        self.clip_end: int = 0                

                                                  
        self.t: int = 0

        self.playing: bool = False

                                                              
        bus.subscribe("PlaybackSelectClip", self._on_clip_selected)

                                                          
    def _on_clip_selected(self, payload):


        session_id = payload["session_id"]
        clip_id = payload["clip_id"]

        clips = self.store.clips.load(session_id)
        if clip_id not in clips:
            raise KeyError(f"Clip {clip_id} not found for session {session_id}")

        ref: ClipRef = clips[clip_id]

        self.session_id = session_id
        self.clip = ref

        self.clip_start = int(ref.start)
        self.clip_end = int(ref.end)

                                               
        self.t = self.clip_start
        self.playing = False

        self._emit_cursor_state()

                                                          
    def _build_provenance_payload(self):


        session = self.store.load_session(self.session_id, mmap=True)
        prov = session.meta.provenance

                                                     
        clip_meta = {}
        if self.clip is not None and hasattr(self.clip, "meta") and self.clip.meta:
            clip_meta = dict(self.clip.meta)

        return {
                                          
            "origin": prov.origin,                                                           

            "session_id": self.session_id,
            "clip_id": self.clip.clip_id if self.clip else None,

            "source_session_id": prov.source_session_id,
            "source_clip_id": prov.source_clip_id,
            "source_step_index": prov.source_step_index,

                            
            "branch_parent_id": prov.branch_parent_id,
            "branch_depth": int(prov.branch_depth),

                                     
            "lineage_token": getattr(prov, "lineage_token", None),

                                     
            "agent_id": prov.agent_id,
            "notes": prov.notes,

                                 
            "clip_meta": clip_meta,
        }

    def _emit_cursor_state(self):


        if self.session_id is None or self.clip is None:
            return

        snapshot = self.store.get_step_snapshot(
            self.session_id,
            self.t,
        )

        provenance = self._build_provenance_payload()

        self.bus.publish(
            "TraceCursorMoved",
            {
                "session_id": self.session_id,
                "clip_id": self.clip.clip_id,
                "step_index": self.t,                   
                "snapshot": snapshot,                            
                "provenance": provenance,
            },
        )

                                                          
    def play(self):
        if self.session_id is None or self.clip is None:
            return

        self.playing = True
        self.bus.publish("PlaybackStarted", {})

    def pause(self):
        self.playing = False
        self.bus.publish("PlaybackPaused", {})

    def step_forward(self):
        if self.session_id is None or self.clip is None:
            return

        self.t = min(self.t + 1, self.clip_end - 1)
        self._emit_cursor_state()

    def step_backward(self):
        if self.session_id is None or self.clip is None:
            return

        self.t = max(self.t - 1, self.clip_start)
        self._emit_cursor_state()

    def scrub_to(self, t: int):


        if self.session_id is None or self.clip is None:
            return

        t_abs = int(t)
        t_abs = max(self.clip_start, min(t_abs, self.clip_end - 1))

        self.t = t_abs
        self._emit_cursor_state()

                                                          
    def tick(self):


        if not self.playing:
            return
        if self.session_id is None or self.clip is None:
            return

        if self.t >= self.clip_end:
            self.playing = False
            return

        self._emit_cursor_state()
        self.t += 1

