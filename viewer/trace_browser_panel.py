                               

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from viewer.event_bus import EventBus
from dataset.session_store import SessionStore
from dataset.clip_index import ClipRef



class TraceBrowserPanel(QWidget):

    def __init__(self, store: SessionStore, bus: EventBus):
        super().__init__()

        # basic layout so this widget can be added to Qt layouts
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.store = store
        self.bus = bus

                                         
        self.session_meta: Dict[str, Dict] = {}

                                                
        self.clips: Dict[str, Dict[str, ClipRef]] = {}

                              
        self.sessions: List[str] = []

                          
        self.children: Dict[str, List[str]] = {}
        self.roots: List[str] = []

                                                          
    def refresh_sessions(self):


        self.sessions = self.store.list_sessions()
        self.session_meta.clear()
        self.clips.clear()

        for sid in self.sessions:
            session = self.store.load_session(sid, mmap=True)
            prov = session.meta.provenance

            self.session_meta[sid] = {
                "origin": prov.origin,

                "branch_parent_id": prov.branch_parent_id,
                "branch_depth": int(prov.branch_depth),

                "source_session_id": prov.source_session_id,
                "source_clip_id": prov.source_clip_id,
                "source_step_index": prov.source_step_index,

                "agent_id": prov.agent_id,
                "notes": prov.notes,

                                                            
                "lineage_token": getattr(prov, "lineage_token", None),
            }

                                                        
        self._build_lineage_tree()

                                                          
    def _build_lineage_tree(self):


        self.children = {sid: [] for sid in self.sessions}
        self.roots = []

        for sid, meta in self.session_meta.items():
            parent = meta.get("branch_parent_id")

            if parent is None or parent not in self.children:
                                                              
                self.roots.append(sid)
            else:
                self.children[parent].append(sid)

                                                          
    def iter_lineage_order(self):


        def walk(sid: str, depth: int):
            yield sid, depth, self.session_meta.get(sid, {})
            for child in sorted(self.children.get(sid, [])):
                yield from walk(child, depth + 1)

        for root in sorted(self.roots):
            yield from walk(root, 0)

                                                          
    def get_session_display_label(self, session_id: str) -> str:


        meta = self.session_meta.get(session_id, {})

        origin = meta.get("origin", "?")
        depth = meta.get("branch_depth", 0)

        parent = meta.get("branch_parent_id")
        tag = "(root)" if parent is None else f"(from {parent[:6]})"

        return f"[{depth}] {origin} {session_id[:8]} {tag}"

                                                          
    def load_clips_for_session(self, session_id: str):


        clips = self.store.clips.load(session_id)
        self.clips[session_id] = clips

                                                          
    def select_clip(self, session_id: str, clip_id: str):


        if session_id not in self.clips:
            self.load_clips_for_session(session_id)

        clips = self.clips.get(session_id, {})
        if clip_id not in clips:
            return

        clip: ClipRef = clips[clip_id]

        provenance = {
            **self.session_meta.get(
                session_id,
                {
                    "origin": "unknown",
                    "branch_parent_id": None,
                    "branch_depth": 0,
                    "source_session_id": None,
                    "source_clip_id": None,
                    "source_step_index": None,
                    "agent_id": None,
                    "notes": None,
                    "lineage_token": None,
                },
            ),
            "session_id": session_id,
            "clip_id": clip_id,
        }

        self.bus.publish(
            "PlaybackSelectClip",
            {
                "session_id": session_id,
                "clip_id": clip_id,
                "provenance": provenance,
            },
        )
