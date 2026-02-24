from __future__ import annotations

from typing import Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QHBoxLayout, QFrame, QSpinBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QFont


class Sparkline(QWidget):
    def __init__(self, title: str, color: str):
        super().__init__()
        self.title = title
        self.color = QColor(color)
        self.values = []
        self.setMinimumHeight(130)

    def set_values(self, values):
        self.values = list(values or [])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()

        painter.fillRect(rect, QColor("#1f242b"))

        painter.setPen(QPen(QColor("#3b4350"), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        title_font = QFont()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#dbe3ed"))
        painter.drawText(rect.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.title)

        if not self.values:
            painter.setPen(QColor("#8c97a6"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No data yet")
            return

        vals = self.values[-120:]
        vmin = min(vals)
        vmax = max(vals)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        left = 10
        right = max(left + 1, rect.width() - 10)
        top = 28
        bottom = max(top + 1, rect.height() - 10)
        w = max(1, right - left)
        h = max(1, bottom - top)

        painter.setPen(QPen(QColor("#2b313a"), 1))
        painter.drawLine(left, bottom, right, bottom)

        pen = QPen(self.color, 2)
        painter.setPen(pen)
        n = len(vals)
        for i in range(1, n):
            x0 = left + int(((i - 1) / max(1, n - 1)) * w)
            x1 = left + int((i / max(1, n - 1)) * w)
            y0 = bottom - int(((vals[i - 1] - vmin) / (vmax - vmin)) * h)
            y1 = bottom - int(((vals[i] - vmin) / (vmax - vmin)) * h)
            painter.drawLine(x0, y0, x1, y1)

        painter.setPen(QColor("#aeb7c5"))
        painter.setFont(QFont("", 8))
        painter.drawText(left, top + 10, f"max {vmax:.3f}")
        painter.drawText(left, bottom - 2, f"min {vmin:.3f}")


class RLTrainingPanel(QWidget):
    """Compact dashboard for RL training progress and health metrics."""

    def __init__(self):
        super().__init__()
        self.graphs_paused = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("RL Training Dashboard")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #e8eef7;")
        root.addWidget(title)

        self.subtitle = QLabel("Monitoring online actor-critic learning (phase-1 capped episodes)")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("font-size: 12px; color: #9aa3ad;")
        root.addWidget(self.subtitle)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        max_steps_label = QLabel("Max Episode Steps")
        max_steps_label.setStyleSheet("font-weight: 600;")
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 1_000_000)
        self.max_steps_spin.setSingleStep(50)
        self.max_steps_spin.setValue(1000)
        self.max_steps_spin.setMinimumWidth(120)
        controls.addWidget(max_steps_label)
        controls.addWidget(self.max_steps_spin)
        controls.addStretch(1)
        root.addLayout(controls)

        top_cards = QGridLayout()
        top_cards.setHorizontalSpacing(24)
        top_cards.setVerticalSpacing(8)
        root.addLayout(top_cards)

        self.header_labels: Dict[str, QLabel] = {}
        summary_rows = [
            ("training_status", "Training"),
            ("policy_name", "Policy"),
            ("episode_count", "Episodes"),
            ("steps_total", "Steps (Total)"),
            ("updates_total", "Policy Updates"),
            ("best_episode_reward", "Best Reward"),
            ("steps_per_sec", "Steps/Sec"),
            ("episodes_per_min", "Episodes/Min"),
        ]
        for idx, (key, label_text) in enumerate(summary_rows):
            box = QFrame()
            box.setFrameShape(QFrame.Shape.StyledPanel)
            box.setStyleSheet("QFrame { background: #1f242b; border: 1px solid #3b4350; border-radius: 6px; }")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 8, 10, 8)
            box_layout.setSpacing(2)
            k = QLabel(label_text)
            k.setStyleSheet("font-size: 11px; color: #9aa3ad;")
            v = QLabel("-")
            v.setStyleSheet("font-size: 16px; font-weight: 700; color: #e8eef7;")
            box_layout.addWidget(k)
            box_layout.addWidget(v)
            top_cards.addWidget(box, idx // 4, idx % 4)
            self.header_labels[key] = v

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        root.addLayout(grid)

        self.value_labels: Dict[str, QLabel] = {}

        rows = [
            ("steps_episode", "Steps (Episode)"),
            ("phase", "Current Phase"),
            ("learning_enabled", "Learning Enabled"),
            ("stomach", "Stomach"),
            ("last_reward", "Last Reward"),
            ("episode_reward", "Episode Reward"),
            ("reward_avg_20", "Avg Reward (Last 20 Ep)"),
            ("steps_avg_20", "Avg Steps (Last 20 Ep)"),
            ("phase1_stop_phase2", "Stops @ Phase 2"),
            ("phase1_stop_death", "Stops @ Death"),
            ("phase1_stop_max_steps", "Stops @ Max Steps"),
            ("autosave_every", "Autosave Every"),
            ("last_autosave_episode", "Last Autosave Ep"),
            ("checkpoint_path", "Checkpoint"),
        ]

        for r, (key, label_text) in enumerate(rows):
            k = QLabel(f"{label_text}:")
            k.setStyleSheet("font-weight: 600;")
            v = QLabel("-")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if key == "checkpoint_path":
                v.setWordWrap(True)
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            self.value_labels[key] = v

        charts = QHBoxLayout()
        charts.setSpacing(10)
        root.addLayout(charts)
        self.reward_chart = Sparkline("Episode Reward Trend", "#5cc8ff")
        self.steps_chart = Sparkline("Episode Length Trend", "#8ee28e")
        charts.addWidget(self.reward_chart, 1)
        charts.addWidget(self.steps_chart, 1)

        root.addStretch(1)

    def update_metrics(self, metrics: Dict[str, Any]):
        for key, label in self.header_labels.items():
            val = metrics.get(key, "-")
            label.setText(str(val))
        for key, label in self.value_labels.items():
            val = metrics.get(key, "-")
            label.setText(str(val))
        if not self.graphs_paused:
            self.reward_chart.set_values(metrics.get("reward_history", []))
            self.steps_chart.set_values(metrics.get("steps_history", []))

    def set_max_episode_steps(self, value: int):
        self.max_steps_spin.setValue(int(value))

    def set_graphs_paused(self, paused: bool):
        self.graphs_paused = bool(paused)
