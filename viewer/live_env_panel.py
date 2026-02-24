# live_env_panel.py

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent

from viewer.event_bus import EventBus
from engine.env_renderer import WORLD_MIN, WORLD_MAX


class LiveEnvPanel(QWidget):
    """Render panel that consumes EnvRenderPacket (frame+snapshot+telemetry).

    This panel is purely a consumer of an already-prepared frame and
    must not implement any derived-geometry or simulation logic.
    """

    def __init__(self, bus: EventBus):
        super().__init__()

        self.bus = bus

        # latest frame (H,W,C) normalized float32 in [0,1]
        self.frame: Optional[np.ndarray] = None
        # authoritative snapshot object
        self.snapshot: Optional[Dict[str, Any]] = None
        # flattened telemetry dict for overlay
        self.telemetry: Optional[Dict[str, Any]] = None

        self.setMinimumHeight(300)
        self.setMinimumWidth(400)

        # make focusable to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # keys currently pressed (Qt.Key values)
        self._keys_down: set[int] = set()

        # subscribe to packet-based rendering events
        bus.subscribe("EnvRenderPacket", self._on_render_packet)
        bus.subscribe("EditModeToggled", self._on_edit_mode_toggled)

        self.edit_mode_enabled: bool = False
        self._drag_target: Optional[tuple[str, Optional[int]]] = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_render_packet(self, packet: Dict[str, Any]):
        # packet expected to contain: frame, snapshot, telemetry, sim_time, step_index
        self.frame = packet.get("frame")
        self.snapshot = packet.get("snapshot")
        self.telemetry = packet.get("telemetry")
        self.update()

    # ------------------------------------------------------------------
    # Human input handling
    # ------------------------------------------------------------------

    def _publish_human_input(self):
        left = (Qt.Key.Key_Left in self._keys_down) or (Qt.Key.Key_A in self._keys_down)
        right = (Qt.Key.Key_Right in self._keys_down) or (Qt.Key.Key_D in self._keys_down)
        up = (Qt.Key.Key_Up in self._keys_down) or (Qt.Key.Key_W in self._keys_down)
        down = (Qt.Key.Key_Down in self._keys_down) or (Qt.Key.Key_S in self._keys_down)

        dx = (1.0 if right else 0.0) + (-1.0 if left else 0.0)
        # invert vertical axis so Up arrow produces negative world-y (move up)
        dy = (-1.0 if up else 0.0) + (1.0 if down else 0.0)

        # Publish legacy dx/dy payload and a canonical action payload so
        # human input is available independent of simulation stepping.
        self.bus.publish("HumanInputUpdated", {"dx": dx, "dy": dy})
        try:
            self.bus.publish("HumanInputEvent", {"action": np.array([dx, dy], dtype=np.float32)})
        except Exception:
            # best-effort: event may be consumed by older listeners
            pass

    def _on_edit_mode_toggled(self, payload: Dict[str, Any]):
        self.edit_mode_enabled = bool(payload.get("enabled", False))

    def keyPressEvent(self, e: QKeyEvent):
        self._keys_down.add(e.key())
        self._publish_human_input()
        e.accept()

    def keyReleaseEvent(self, e: QKeyEvent):
        if e.key() in self._keys_down:
            self._keys_down.remove(e.key())
        self._publish_human_input()
        e.accept()

    def mousePressEvent(self, e):
        # Only allow editing by mouse when edit mode is enabled
        if not getattr(self, "edit_mode_enabled", False):
            return

        try:
            pos = e.position()
            x = float(pos.x())
            y = float(pos.y())
        except Exception:
            x = float(e.x())
            y = float(e.y())

        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))

        # map widget coords -> world coords in [-1,1]
        wx = WORLD_MIN + (x / w) * (WORLD_MAX - WORLD_MIN)
        wy = WORLD_MIN + (y / h) * (WORLD_MAX - WORLD_MIN)

        target = self._pick_drag_target(x, y)
        if target is None:
            self._drag_target = None
            return

        self._drag_target = target
        self._publish_drag_edit(wx, wy)
        e.accept()

    def mouseMoveEvent(self, e):
        if not getattr(self, "edit_mode_enabled", False):
            return
        if self._drag_target is None:
            return

        try:
            pos = e.position()
            x = float(pos.x())
            y = float(pos.y())
        except Exception:
            x = float(e.x())
            y = float(e.y())

        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        wx = WORLD_MIN + (x / w) * (WORLD_MAX - WORLD_MIN)
        wy = WORLD_MIN + (y / h) * (WORLD_MAX - WORLD_MIN)
        self._publish_drag_edit(wx, wy)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_target = None
        e.accept()

    def _world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        sx = ((float(wx) - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)) * w
        sy = ((float(wy) - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)) * h
        return sx, sy

    def _pick_drag_target(self, click_x: float, click_y: float) -> Optional[tuple[str, Optional[int]]]:
        snap = self.snapshot
        if snap is None:
            return ("agent", None)

        candidates: list[tuple[str, Optional[int], float]] = []

        try:
            if hasattr(snap, "get"):
                agent = snap.get("agent_pos", [0.0, 0.0])
                pods = snap.get("pods", []) or []
            else:
                agent = getattr(snap, "agent_pos", [0.0, 0.0])
                pods = getattr(snap, "pods", []) or []
        except Exception:
            return ("agent", None)

        try:
            ax, ay = float(agent[0]), float(agent[1])
            asx, asy = self._world_to_screen(ax, ay)
            d2 = (asx - click_x) ** 2 + (asy - click_y) ** 2
            candidates.append(("agent", None, d2))
        except Exception:
            pass

        for idx, p in enumerate(pods):
            try:
                pos = p.get("pos") if hasattr(p, "get") else getattr(p, "pos")
                px, py = float(pos[0]), float(pos[1])
                psx, psy = self._world_to_screen(px, py)
                d2 = (psx - click_x) ** 2 + (psy - click_y) ** 2
                candidates.append(("pod", idx, d2))
            except Exception:
                continue

        if not candidates:
            return ("agent", None)

        target, idx, best_d2 = min(candidates, key=lambda t: t[2])
        max_pick_px = 22.0
        if best_d2 > (max_pick_px * max_pick_px):
            return None
        return (target, idx)

    def _publish_drag_edit(self, wx: float, wy: float):
        if self._drag_target is None:
            return
        target, idx = self._drag_target
        if target == "agent":
            self.bus.publish("EnvEditRequested", {"mutation": {"agent_pos": [wx, wy]}})
            return
        if target == "pod" and idx is not None:
            self.bus.publish(
                "EnvEditRequested",
                {
                    "mutation": {
                        "pod_pos": {
                            "index": int(idx),
                            "pos": [wx, wy],
                        }
                    }
                },
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.frame is None:
            # simple placeholder text
            rect = self.rect()
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No live frame")
            return

        try:
            img = (self.frame * 255.0).astype(np.uint8)
            h, w = img.shape[0], img.shape[1]
            # handle grayscale single-channel
            if img.ndim == 3 and img.shape[2] == 1:
                qformat = QImage.Format.Format_Grayscale8
                bytes_per_line = img.strides[0]
                qimg = QImage(img.data, w, h, bytes_per_line, qformat)
            elif img.ndim == 3 and img.shape[2] == 3:
                qformat = QImage.Format.Format_RGB888
                bytes_per_line = img.strides[0]
                qimg = QImage(img.data, w, h, bytes_per_line, qformat)
            else:
                # fallback: try to convert to 2D grayscale
                flat = np.squeeze(img)
                if flat.ndim == 2:
                    flat = np.expand_dims(flat, -1)
                    qformat = QImage.Format.Format_Grayscale8
                    bytes_per_line = flat.strides[0]
                    qimg = QImage(flat.data, flat.shape[1], flat.shape[0], bytes_per_line, qformat)
                else:
                    # can't render
                    rect = self.rect()
                    painter.setPen(Qt.GlobalColor.white)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Frame format unsupported")
                    return

            painter.drawImage(self.rect(), qimg)
        except Exception:
            rect = self.rect()
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Error rendering frame")


        # draw telemetry overlay (top-left)
        try:
            info_lines = []
            if self.telemetry:
                t = self.telemetry
                # prefer explicit keys
                if "phase" in t:
                    info_lines.append(f"phase: {t.get('phase')}")
                if "stomach" in t:
                    info_lines.append(f"stomach: {t.get('stomach'):.3f}")
                if "wall_enabled" in t:
                    info_lines.append(f"wall: {t.get('wall_enabled')}")
                if "sequence_index" in t:
                    info_lines.append(f"seq: {t.get('sequence_index')}")
            elif self.snapshot:
                s = self.snapshot
                try:
                    info_lines.append(f"phase: {s.get('phase')}")
                except Exception:
                    try:
                        info_lines.append(f"phase: {getattr(s, 'phase', '')}")
                    except Exception:
                        pass

            if info_lines:
                painter.setPen(Qt.GlobalColor.white)
                x0, y0 = 8, 12
                for i, line in enumerate(info_lines):
                    painter.drawText(x0, y0 + i * 14, line)
        except Exception:
            pass
    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def refresh(self):
        self.update()

    def update_overlay(self, telemetry: dict):
        self.telemetry = telemetry
        self.update()
