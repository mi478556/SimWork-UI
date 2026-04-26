from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CodaTrainingPanel(QWidget):
    run_compare_requested = pyqtSignal(dict)
    apply_candidate_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._last_results: List[Dict[str, Any]] = []
        self._current_source: Dict[str, Any] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("CODA Training Lab")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #e8eef7;")
        root.addWidget(title)

        subtitle = QLabel("Use a saved playback run as a fixed training source, compare CODA presets, then apply the best one.")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #9aa3ad;")
        root.addWidget(subtitle)

        source_group = QGroupBox("Training Source")
        source_layout = QGridLayout(source_group)
        source_layout.setHorizontalSpacing(14)
        source_layout.setVerticalSpacing(8)
        self.source_session_label = QLabel("No saved run selected")
        self.source_session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_clip_label = QLabel("-")
        self.source_step_label = QLabel("-")
        self.source_note_label = QLabel("Select a saved run in Playback Mode to use it as a CODA training source.")
        self.source_note_label.setWordWrap(True)
        source_layout.addWidget(QLabel("Session"), 0, 0)
        source_layout.addWidget(self.source_session_label, 0, 1)
        source_layout.addWidget(QLabel("Clip"), 1, 0)
        source_layout.addWidget(self.source_clip_label, 1, 1)
        source_layout.addWidget(QLabel("Cursor Step"), 2, 0)
        source_layout.addWidget(self.source_step_label, 2, 1)
        source_layout.addWidget(self.source_note_label, 3, 0, 1, 2)
        root.addWidget(source_group)

        control_group = QGroupBox("Compare Presets")
        control_layout = QGridLayout(control_group)
        control_layout.setHorizontalSpacing(14)
        control_layout.setVerticalSpacing(8)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Baseline", "baseline")
        self.preset_combo.addItem("Contextual Spawn Prior", "contextual_spawn_prior")
        self.preset_combo.addItem("Zero Spawn Prior", "zero_spawn_prior")
        self.preset_combo.addItem("Contextual Warm64", "contextual_warm64")
        self.preset_combo.addItem("Zero Prior Warm64", "zero_prior_warm64")
        self.preset_combo.addItem("Blanket Prior Warm64", "blanket_prior_warm64")
        self.preset_combo.addItem("Lookahead Contextual Warm64", "lookahead_contextual_warm64")

        self.compare_set_combo = QComboBox()
        self.compare_set_combo.addItem("Top 3", "baseline,contextual_spawn_prior,zero_spawn_prior")
        self.compare_set_combo.addItem("Warm64 Trio", "contextual_warm64,zero_prior_warm64,blanket_prior_warm64")
        self.compare_set_combo.addItem("Lookahead Trio", "contextual_warm64,blanket_prior_warm64,lookahead_contextual_warm64")
        self.compare_set_combo.addItem("All Presets", "baseline,contextual_spawn_prior,zero_spawn_prior,contextual_warm64,zero_prior_warm64,blanket_prior_warm64,lookahead_contextual_warm64")

        self.offline_updates_spin = QSpinBox()
        self.offline_updates_spin.setRange(1, 5000)
        self.offline_updates_spin.setValue(64)

        self.eval_samples_spin = QSpinBox()
        self.eval_samples_spin.setRange(16, 50000)
        self.eval_samples_spin.setValue(256)

        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, 256)
        self.cpu_threads_spin.setValue(70)

        self.run_compare_btn = QPushButton("Train + Compare")
        self.apply_best_btn = QPushButton("Apply Best To CODA")
        self.apply_selected_btn = QPushButton("Apply Selected Preset")
        self.apply_best_btn.setEnabled(False)
        self.apply_selected_btn.setEnabled(True)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addWidget(self.run_compare_btn)
        button_row.addWidget(self.apply_best_btn)
        button_row.addWidget(self.apply_selected_btn)
        button_row.addStretch(1)

        control_layout.addWidget(QLabel("Compare Set"), 0, 0)
        control_layout.addWidget(self.compare_set_combo, 0, 1)
        control_layout.addWidget(QLabel("Selected Preset"), 1, 0)
        control_layout.addWidget(self.preset_combo, 1, 1)
        control_layout.addWidget(QLabel("Offline Updates"), 0, 2)
        control_layout.addWidget(self.offline_updates_spin, 0, 3)
        control_layout.addWidget(QLabel("Eval Samples"), 1, 2)
        control_layout.addWidget(self.eval_samples_spin, 1, 3)
        control_layout.addWidget(QLabel("CPU Threads"), 2, 2)
        control_layout.addWidget(self.cpu_threads_spin, 2, 3)
        control_layout.addLayout(button_row, 3, 0, 1, 4)

        root.addWidget(control_group)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: 600; color: #dbe3ed;")
        root.addWidget(self.status_label)

        self.results_edit = QPlainTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText("Comparison results will appear here.")
        root.addWidget(self.results_edit, stretch=1)

        self.run_compare_btn.clicked.connect(self._emit_run_compare)
        self.apply_best_btn.clicked.connect(self._emit_apply_best)
        self.apply_selected_btn.clicked.connect(self._emit_apply_selected)

    def set_source(self, *, session_id: Optional[str], clip_id: Optional[str], step_index: Optional[int]) -> None:
        self._current_source = {
            "session_id": session_id,
            "clip_id": clip_id,
            "step_index": step_index,
        }
        self.source_session_label.setText(str(session_id or "No saved run selected"))
        self.source_clip_label.setText(str(clip_id or "-"))
        self.source_step_label.setText(str(step_index if step_index is not None else "-"))
        if session_id:
            self.source_note_label.setText("This saved session will be used as the fixed CODA training source for comparison.")
        else:
            self.source_note_label.setText("Select a saved run in Playback Mode to use it as a CODA training source.")

    def set_running(self, running: bool, message: str = "") -> None:
        self.run_compare_btn.setEnabled(not running and bool(self._current_source.get("session_id")))
        self.apply_best_btn.setEnabled((not running) and bool(self._last_results))
        self.apply_selected_btn.setEnabled(not running)
        self.status_label.setText(message or ("Running..." if running else "Idle"))

    def set_results(self, rows: Sequence[Dict[str, Any]]) -> None:
        self._last_results = list(rows or [])
        self.apply_best_btn.setEnabled(bool(self._last_results))
        if not self._last_results:
            self.results_edit.setPlainText("No comparison results yet.")
            return

        lines: List[str] = []
        for row in self._last_results:
            lines.append(f"{row.get('label', '?')}")
            lines.append(
                "  "
                + f"spawn_contrast={self._fmt(row.get('spawn_prob_pod_prev_off_spawn_contrast'))} "
                + f"spawn_tpr={self._fmt(row.get('spawn_tpr'))} "
                + f"spawn_fpr={self._fmt(row.get('spawn_fpr'))}"
            )
            lines.append(
                "  "
                + f"near={self._fmt(row.get('spawn_prob_pod_prev_off_near_spawn_mean'))} "
                + f"far={self._fmt(row.get('spawn_prob_pod_prev_off_far_spawn_mean'))} "
                + f"mean={self._fmt(row.get('spawn_prob_pod_prev_off_mean'))}"
            )
            lines.append(
                "  "
                + f"updates={row.get('applied_updates', 0)} "
                + f"buffer={row.get('buffer_size', 0)} "
                + f"session={row.get('session_id', '-')}"
            )
            lines.append("")
        self.results_edit.setPlainText("\n".join(lines).strip())

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(str(text))

    def _emit_run_compare(self) -> None:
        session_id = self._current_source.get("session_id")
        if not session_id:
            self.set_status_text("Select a playback session first.")
            return
        candidates = [tok.strip() for tok in str(self.compare_set_combo.currentData() or "").split(",") if tok.strip()]
        self.run_compare_requested.emit(
            {
                "session_id": session_id,
                "clip_id": self._current_source.get("clip_id"),
                "step_index": self._current_source.get("step_index"),
                "offline_updates": int(self.offline_updates_spin.value()),
                "eval_samples": int(self.eval_samples_spin.value()),
                "cpu_threads": int(self.cpu_threads_spin.value()),
                "candidates": candidates,
            }
        )

    def _emit_apply_best(self) -> None:
        if not self._last_results:
            self.set_status_text("Run a comparison first.")
            return
        best = dict(self._last_results[0])
        self.apply_candidate_requested.emit(
            {
                "label": best.get("label"),
                "cfg_updates": dict(best.get("cfg_updates") or {}),
                "source": "best_result",
            }
        )

    def _emit_apply_selected(self) -> None:
        label = str(self.preset_combo.currentData() or "")
        self.apply_candidate_requested.emit(
            {
                "label": label,
                "cfg_updates": None,
                "source": "selected_preset",
            }
        )

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            x = float(value)
        except Exception:
            return "nan"
        if x != x or x == float("inf") or x == float("-inf"):
            return "nan"
        return f"{x:.4f}"
