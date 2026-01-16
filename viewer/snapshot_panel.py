from __future__ import annotations

from typing import Optional, Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from viewer.event_bus import EventBus
from engine.snapshot_state import EnvStateSnapshot


class SnapshotPanel(QWidget):

    def __init__(self, bus: EventBus):
        super().__init__()

        # minimal layout so widget can be embedded
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # god-mode edit controls
        btn_row = QHBoxLayout()
        self.toggle_wall_btn = QPushButton("Toggle Wall")
        self.set_stomach_btn = QPushButton("Set Stomach 0.2")
        self.advance_seq_btn = QPushButton("Advance Sequence")
        self.phase_down_btn = QPushButton("Phase -")
        self.phase_up_btn = QPushButton("Phase +")
        self.reset_env_btn = QPushButton("Reset Env")

        btn_row.addWidget(self.toggle_wall_btn)
        btn_row.addWidget(self.set_stomach_btn)
        btn_row.addWidget(self.advance_seq_btn)
        btn_row.addWidget(self.phase_down_btn)
        btn_row.addWidget(self.phase_up_btn)
        btn_row.addWidget(self.reset_env_btn)

        layout.addLayout(btn_row)

        self.toggle_wall_btn.clicked.connect(self._on_toggle_wall)
        self.set_stomach_btn.clicked.connect(self._on_set_stomach)
        self.advance_seq_btn.clicked.connect(self._on_advance_sequence)
        self.phase_down_btn.clicked.connect(self._on_phase_down)
        self.phase_up_btn.clicked.connect(self._on_phase_up)
        self.reset_env_btn.clicked.connect(self._on_reset_env)

        self.bus = bus

        self.current_preview: Optional[Dict[str, Any]] = None
        bus.subscribe("TraceCursorMoved", self._on_cursor_state)
        bus.subscribe("EnvStateUpdated", self._on_live_env_state)
        bus.subscribe("EditModeToggled", self._on_edit_mode_toggled)

        self.edit_mode_enabled: bool = False

                                                          
    def _on_cursor_state(self, payload: Dict[str, Any]):


        self.current_preview = payload

    def _on_live_env_state(self, snapshot: Dict[str, Any]):
        # store canonical live snapshot (may be EnvStateSnapshot dataclass)
        self.current_live_snapshot = snapshot

                                                          
    def build_snapshot(self) -> Optional[EnvStateSnapshot]:
        # God mode edits always apply to LIVE env snapshot
        if getattr(self, "edit_mode_enabled", False) and getattr(self, "current_live_snapshot", None) is not None:
            return self.current_live_snapshot

        # Otherwise fall back to trace preview (for eval, injection)
        if self.current_preview:
            return self.current_preview.get("snapshot")

        return None

                                                          
    def inject_into_env(self):
        if not self.current_preview:
            return

        snapshot = self.current_preview.get("snapshot")
        provenance = self.current_preview.get("provenance", {})

        payload = {
            "snapshot": snapshot,

                                                                           
            "provenance": {
                "source": "trace",

                                         
                "session_id": self.current_preview.get("session_id"),
                "clip_id": self.current_preview.get("clip_id"),
                "step_index": self.current_preview.get("step_index"),

                                                          
                "branch_parent_id": provenance.get("branch_parent_id"),
                "branch_depth": int(provenance.get("branch_depth", 0)),

                                                          
                "modifications": provenance.get("modifications", {}),
            },
        }

        self.bus.publish("InjectRequested", payload)

    # ------------------------------------------------------------
    # God-mode edit handlers
    # ------------------------------------------------------------
    def _emit_env_edit(self, mutation: Dict[str, Any]):
        # Only emit edits when edit mode is enabled
        if not getattr(self, "edit_mode_enabled", False):
            return

        self.bus.publish("EnvEditRequested", {"mutation": mutation})

    def _on_edit_mode_toggled(self, payload: Dict[str, Any]):
        self.edit_mode_enabled = bool(payload.get("enabled", False))

    def _on_toggle_wall(self):
        snap = self.build_snapshot()
        if not snap:
            return

        enabled = False

        # Case 1: dict-like snapshot
        try:
            if hasattr(snap, "get"):
                wall = snap.get("wall", None)
                if isinstance(wall, dict):
                    enabled = bool(wall.get("enabled", False))
                else:
                    # wall might be a dataclass-like object even in dict snapshot
                    enabled = bool(getattr(wall, "enabled", False))
            else:
                raise TypeError("not dict-like")
        except Exception:
            # Case 2: EnvStateSnapshot dataclass
            try:
                wall = getattr(snap, "wall", None)
                enabled = bool(getattr(wall, "enabled", False))
            except Exception:
                enabled = False

        self._emit_env_edit({"wall": {"enabled": (not enabled)}})

    def _on_set_stomach(self):
        # set stomach to example value
        self._emit_env_edit({"stomach": 0.2})

    def _get_phase_from_snapshot(self, snap) -> int:
        try:
            if hasattr(snap, "get"):
                return int(snap.get("phase", 1))
            return int(getattr(snap, "phase", 1))
        except Exception:
            return 1

    def _on_phase_up(self):
        snap = self.build_snapshot()
        if not snap:
            return

        cur = self._get_phase_from_snapshot(snap)
        new = min(cur + 1, 3)

        self._emit_env_edit({"phase": new})

    def _on_phase_down(self):
        snap = self.build_snapshot()
        if not snap:
            return

        cur = self._get_phase_from_snapshot(snap)
        new = max(cur - 1, 1)

        self._emit_env_edit({"phase": new})

    def _on_advance_sequence(self):
        snap = self.build_snapshot()
        if not snap:
            return
        try:
            cur = int(snap.get("sequence_index", 0))
        except Exception:
            try:
                cur = int(getattr(snap, "sequence_index", 0))
            except Exception:
                cur = 0
        self._emit_env_edit({"sequence_index": cur + 1})

                                                          
    def run_eval_from_here(
        self,
        *,
        num_steps: int = 50,
        agent_id: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
    ):


        if not self.current_preview:
            return

        snapshot = self.current_preview.get("snapshot")
        provenance = self.current_preview.get("provenance", {})

        parent_session = self.current_preview.get("session_id")
        parent_clip = self.current_preview.get("clip_id")
        parent_step = self.current_preview.get("step_index")

        parent_depth = int(provenance.get("branch_depth", 0))

        payload = {
            "snapshot": snapshot,

            "num_steps": num_steps,
            "agent_id": agent_id,

                                       
            "session_id": parent_session,
            "clip_id": parent_clip,
            "step_index": parent_step,

                     
            "branch_parent_id": parent_session,
            "branch_depth": parent_depth + 1,

                                             
            "modifications": modifications or {},
        }

        self.bus.publish("EvalRunRequested", payload)

    def _on_reset_env(self):
        # Publish a reset request; controller will handle authoritative reset
        self.bus.publish("EnvResetRequested", {})
