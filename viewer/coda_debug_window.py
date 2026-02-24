from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CodaDebugWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CODA Debug Viewer")
        self.resize(1100, 760)

        self.policy: Optional[Any] = None
        self.last_step: int = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top controls: logging settings.
        controls_group = QGroupBox("Logging Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setSpacing(10)

        self.logging_enabled_cb = QCheckBox("Enable Logging")
        self.logging_enabled_cb.setChecked(False)

        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 100000)
        self.every_spin.setValue(200)
        self.every_spin.setPrefix("Every ")
        self.every_spin.setSuffix(" steps")
        self.every_spin.setFixedWidth(140)

        self.out_dir_edit = QLineEdit("data/coda_debug")
        self.out_dir_edit.setMinimumWidth(320)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_logging_to_policy)

        controls_layout.addWidget(self.logging_enabled_cb)
        controls_layout.addWidget(self.every_spin)
        controls_layout.addWidget(QLabel("Output Dir"))
        controls_layout.addWidget(self.out_dir_edit, stretch=1)
        controls_layout.addWidget(self.apply_btn)

        root.addWidget(controls_group)

        # Frame panel.
        frame_group = QGroupBox("Frames")
        frame_layout = QGridLayout(frame_group)
        frame_layout.setSpacing(8)

        self.recon_title = QLabel("Reconstruction (Current Tokens)")
        self.pred_title = QLabel("Predicted (Previous Step -> Current)")

        self.recon_label = QLabel("No reconstruction yet")
        self.pred_label = QLabel("No prediction yet")

        self._setup_image_label(self.recon_label)
        self._setup_image_label(self.pred_label)

        frame_layout.addWidget(self.recon_title, 0, 0)
        frame_layout.addWidget(self.pred_title, 0, 1)
        frame_layout.addWidget(self.recon_label, 1, 0)
        frame_layout.addWidget(self.pred_label, 1, 1)

        root.addWidget(frame_group, stretch=4)

        # Data panel under frames.
        data_group = QGroupBox("Metrics")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(6)

        quick_row = QHBoxLayout()
        self.step_label = QLabel("Step: -")
        self.loss_label = QLabel("Train Loss: -")
        self.pred_err_label = QLabel("Pred Err: -")
        self.active_label = QLabel("Active Slots: -")
        self.align_label = QLabel("Prediction Aligned: -")
        for w in [self.step_label, self.loss_label, self.pred_err_label, self.active_label, self.align_label]:
            quick_row.addWidget(w)
        quick_row.addStretch(1)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("CODA metrics will appear here.")
        self.details.setMinimumHeight(170)

        data_layout.addLayout(quick_row)
        data_layout.addWidget(self.details, stretch=1)
        root.addWidget(data_group, stretch=3)

    def _setup_image_label(self, label: QLabel):
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(320, 320)
        label.setFrameShape(QFrame.Shape.StyledPanel)
        label.setStyleSheet("QLabel { background: #0f1116; color: #a6b0be; }")

    def set_policy(self, policy: Optional[Any]):
        self.policy = policy

        enabled = bool(
            policy is not None
            and hasattr(policy, "get_latest_debug_packet")
            and hasattr(policy, "set_debug_logging")
        )
        self.apply_btn.setEnabled(enabled)
        self.logging_enabled_cb.setEnabled(enabled)
        self.every_spin.setEnabled(enabled)
        self.out_dir_edit.setEnabled(enabled)

        if enabled:
            try:
                self.logging_enabled_cb.setChecked(bool(getattr(policy, "debug_logging_enabled", False)))
            except Exception:
                pass
            try:
                self.every_spin.setValue(int(getattr(policy, "debug_dump_every", 200)))
            except Exception:
                pass
            try:
                self.out_dir_edit.setText(str(getattr(policy, "debug_dir", "data/coda_debug")))
            except Exception:
                pass

    def _fmt(self, v: Any) -> str:
        if isinstance(v, float):
            if np.isfinite(v):
                return f"{v:.6f}"
            return "nan"
        return str(v)

    def _summary_to_text(self, summary: Dict[str, Any]) -> str:
        if not summary:
            return ""
        lines = []
        for k in sorted(summary.keys()):
            val = summary[k]
            if isinstance(val, dict):
                lines.append(f"{k}:")
                for sk in sorted(val.keys()):
                    lines.append(f"  {sk}: {self._fmt(val[sk])}")
            else:
                lines.append(f"{k}: {self._fmt(val)}")
        return "\n".join(lines)

    def _to_qpixmap(self, image: np.ndarray, target_w: int, target_h: int) -> Optional[QPixmap]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, repeats=3, axis=2)
        elif arr.ndim != 3 or arr.shape[2] < 3:
            return None

        arr = arr[:, :, :3].astype(np.float32, copy=False)
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        h, w = arr.shape[0], arr.shape[1]
        bytes_per_line = int(arr.strides[0])
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(
            max(1, target_w),
            max(1, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _set_image(self, label: QLabel, image: Optional[np.ndarray], empty_text: str):
        if image is None:
            label.clear()
            label.setText(empty_text)
            return

        pix = self._to_qpixmap(image, label.width(), label.height())
        if pix is None:
            label.clear()
            label.setText(empty_text)
            return
        label.setPixmap(pix)

    def update_packet(self, packet: Optional[Dict[str, Any]]):
        if not packet:
            return
        step = int(packet.get("step", -1))
        if step <= self.last_step:
            return

        self.last_step = step
        summary = packet.get("summary", {}) or {}
        recon = packet.get("reconstruction")
        pred = packet.get("predicted_reconstruction")

        self._set_image(self.recon_label, recon, "No reconstruction")
        self._set_image(self.pred_label, pred, "No aligned prediction yet")

        self.step_label.setText(f"Step: {step}")
        self.loss_label.setText(f"Train Loss: {self._fmt(summary.get('train_loss', float('nan')))}")
        self.pred_err_label.setText(
            f"Pred Err: {self._fmt(summary.get('predicted_reconstruction_error', float('nan')))}"
        )
        self.active_label.setText(f"Active Slots: {self._fmt(summary.get('active_slots', '-'))}")
        self.align_label.setText(f"Prediction Aligned: {bool(packet.get('prediction_aligned', False))}")
        self.details.setPlainText(self._summary_to_text(summary))

    def _apply_logging_to_policy(self):
        if self.policy is None or not hasattr(self.policy, "set_debug_logging"):
            return
        enabled = bool(self.logging_enabled_cb.isChecked())
        every = int(self.every_spin.value())
        out_dir = str(self.out_dir_edit.text()).strip()
        try:
            self.policy.set_debug_logging(enabled, every=every, out_dir=out_dir)
        except Exception:
            return
