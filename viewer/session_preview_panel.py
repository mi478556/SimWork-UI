# session_preview_panel.py

from __future__ import annotations

from typing import List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QGraphicsOpacityEffect,
)
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtCore import Qt, QPropertyAnimation

from dataset.session_store import SessionStore
from viewer.event_bus import EventBus
from viewer.thumbnail_cache import ThumbnailCache


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

class _ImageSlot(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        # Ensure image slots have a reasonable minimum so thumbnails render
        self.setMinimumSize(64, 64)
        self._image: Optional[QImage] = None

        # Opacity/fade animation for subtle image transitions
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(1.0)

        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setDuration(80)

    def set_image(self, image: Optional[QImage]):
        self._image = image
        if image is None:
            self.clear()
            return
        self._update_pixmap()

        # subtle fade-in
        try:
            self._fade.stop()
            self._opacity.setOpacity(0.85)
            self._fade.setStartValue(0.85)
            self._fade.setEndValue(1.0)
            self._fade.start()
        except Exception:
            pass

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._image is None:
            return
        # Defer scaling until we have a visible non-zero size to avoid
        # scaling to zero on initial show and never recovering.
        if self.width() <= 1 or self.height() <= 1:
            return
        pix = QPixmap.fromImage(self._image)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


# ------------------------------------------------------------
# SessionPreviewPanel
# ------------------------------------------------------------

class SessionPreviewPanel(QWidget):
    """
    Visual browser for saved sessions.

    Publishes:
      - SessionPreviewFocused(session_id)
      - SessionPreviewActivated(session_id)
      - SessionPreviewThumbnailSet(session_id, step_index)
    """

    def __init__(self, store: SessionStore, bus: EventBus):
        super().__init__()

        self.store = store
        self.bus = bus

        self.sessions: List[str] = []
        self.index: int = 0

        self.thumb_cache = ThumbnailCache(self.store, max_items=48)
        self.thumb_cache.thumbnail_ready.connect(self._on_thumbnail_ready)

        # Listen for render packets so playback frames can be shown
        self.bus.subscribe("EnvRenderPacket", self._on_env_render_packet)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        title = QLabel("Session Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        image_row = QHBoxLayout()
        image_row.setSpacing(8)

        self.prev_img = _ImageSlot()
        self.curr_img = _ImageSlot()
        # Center image should be larger so thumbnails are visible
        self.curr_img.setMinimumSize(256, 256)
        self.next_img = _ImageSlot()

        image_row.addWidget(self.prev_img, stretch=1)
        image_row.addWidget(self.curr_img, stretch=3)
        image_row.addWidget(self.next_img, stretch=1)

        root.addLayout(image_row)

        # --------------------------------------------------
        # Controls
        # --------------------------------------------------
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.left_btn = QPushButton("◀")
        self.right_btn = QPushButton("▶")
        self.activate_btn = QPushButton("Load")
        self.set_thumb_btn = QPushButton("Set Thumbnail")

        self.left_btn.clicked.connect(self._step_left)
        self.right_btn.clicked.connect(self._step_right)
        self.activate_btn.clicked.connect(self._activate_current)
        self.set_thumb_btn.clicked.connect(self._set_thumbnail)

        controls.addWidget(self.left_btn)
        controls.addStretch(1)
        controls.addWidget(self.activate_btn)
        controls.addWidget(self.set_thumb_btn)
        controls.addStretch(1)
        controls.addWidget(self.right_btn)

        root.addLayout(controls)

        # Initial population
        self.refresh()

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    def refresh(self):
        try:
            self.sessions = sorted(self.store.list_sessions())
            print(f"SessionPreviewPanel: found {len(self.sessions)} sessions")
        except Exception:
            self.sessions = []
            raise

        if not self.sessions:
            self.index = 0
            self._update_view()
            return

        self.index = min(self.index, len(self.sessions) - 1)
        self._update_view()

        try:
            self.bus.publish(
                "SessionPreviewFocused",
                {"session_id": self.sessions[self.index]},
            )
        except Exception:
            raise
            pass

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------
    def _step_left(self):
        if not self.sessions:
            return
        self.index = max(0, self.index - 1)
        self._on_index_changed()

    def _step_right(self):
        if not self.sessions:
            return
        self.index = min(len(self.sessions) - 1, self.index + 1)
        self._on_index_changed()

    def wheelEvent(self, event: QWheelEvent):
        if not self.hasFocus():
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._step_left()
        elif delta < 0:
            self._step_right()

    # --------------------------------------------------
    # Actions
    # --------------------------------------------------
    def _activate_current(self):
        if not self.sessions:
            return
        sid = self.sessions[self.index]
        self.bus.publish(
            "SessionPreviewActivated",
            {"session_id": sid},
        )

    def _set_thumbnail(self):
        if not self.sessions:
            return
        sid = self.sessions[self.index]
        self.bus.publish(
            "SessionPreviewThumbnailSet",
            {
                "session_id": sid,
                "step_index": 0,
            },
        )

    # --------------------------------------------------
    # View updates
    # --------------------------------------------------
    def _on_index_changed(self):
        self._update_view()
        sid = self.sessions[self.index]
        self.bus.publish(
            "SessionPreviewFocused",
            {"session_id": sid},
        )

    def _update_view(self):
        if not self.sessions:
            self.prev_img.set_image(None)
            self.curr_img.set_image(None)
            self.next_img.set_image(None)
            return

        def img_at(i: int) -> Optional[QImage]:
            if 0 <= i < len(self.sessions):
                return self.thumb_cache.get(self.sessions[i])
            return None

        prev_img = img_at(self.index - 1)
        curr_img = img_at(self.index)
        next_img = img_at(self.index + 1)

        self.prev_img.set_image(prev_img)
        self.curr_img.set_image(curr_img)
        self.next_img.set_image(next_img)

        # With async loading, None is expected temporarily for cache misses.
        # Do not assert — UI will update when thumbnail_ready fires.

        # Force repaint so changes are visible immediately
        self.update()

    def _on_thumbnail_ready(self, session_id: str):
        # Only repaint if the thumbnail affects the current view
        if session_id not in self.sessions:
            return

        idx = self.sessions.index(session_id)

        if abs(idx - self.index) <= 1:
            self._update_view()

    def _frame_to_qimage(self, frame: np.ndarray) -> Optional[QImage]:
        """Convert float32 [0,1] RGB numpy frame to a QImage (RGB888).

        Returns a copied QImage to avoid lifetime issues with the numpy buffer.
        """
        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            return None

        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        # frame is float32 [0,1]
        img = np.clip(frame, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        img = np.ascontiguousarray(img)

        h, w, _ = img.shape
        bytes_per_line = img.strides[0]

        # copy() is critical to avoid lifetime issues
        return QImage(
            img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

    def _on_env_render_packet(self, packet):
        # Expect an EnvRenderPacket dict with a `frame` key
        if not isinstance(packet, dict):
            return

        # Debug: trace incoming packets to ensure playback frames flow
        # print(f"[SessionPreviewPanel] EnvRenderPacket received: sim_time={packet.get('sim_time')} step_index={packet.get('step_index')} frame_present={packet.get('frame') is not None}")

        frame = packet.get("frame")
        if frame is None:
            return

        qimg = self._frame_to_qimage(frame)
        if qimg is None:
            return

        # Playback replaces the center slot
        self.curr_img.set_image(qimg)

    # --------------------------------------------------
    # Focus handling
    # --------------------------------------------------
    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.sessions:
            try:
                self.bus.publish(
                    "SessionPreviewFocused",
                    {"session_id": self.sessions[self.index]},
                )
            except Exception:
                raise
                pass
