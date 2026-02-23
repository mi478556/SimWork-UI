# app_controller.py

from __future__ import annotations
import sys
import os
import time
import inspect

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QPushButton,
    QComboBox,
    QSizePolicy,
    QSlider,
    QStackedWidget,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal

import qdarkstyle
from typing import Dict, Any
import numpy as np

from viewer.event_bus import EventBus
from engine.environment_runtime import EnvironmentRuntime
from engine.snapshot_state import snapshot_to_runtime_dict
from engine.injection_bridge import InjectionBridge
from engine.oracle_tools import EnvDistanceTool
from agent.agent_controller import AgentController
from agent.starter_policy import StarterWanderPolicy, NoOpPolicy, OscillatingPolicy
from agent.rl_actor_critic_policy import RLActorCriticFoodPolicy

from runner.live_runner import LiveRunner
from runner.playback_runner import PlaybackRunner
from runner.eval_runner import EvalRunner
from runner.observation_builder import build_observation

 
from viewer.trace_browser_panel import TraceBrowserPanel
from viewer.trace_player_panel import TracePlayerPanel
from viewer.snapshot_panel import SnapshotPanel
from viewer.live_env_panel import LiveEnvPanel
from viewer.session_preview_panel import SessionPreviewPanel
from viewer.rl_training_panel import RLTrainingPanel

from dataset.session_store import SessionStore
from dataset.finalize_manager import FinalizeManager


class AppController(QMainWindow):

    submit_finalize_job = pyqtSignal(object)
    take_captured_qt = pyqtSignal(dict)

    # Recording intent constants
    REC_OFF = "OFF"
    REC_ARMED = "ARMED"
    REC_REQUESTED = "REQUESTED"
    REC_CAPTURING = "CAPTURING"

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------
        # Core systems
        # ------------------------------------------------------------------

        self.bus = EventBus()
        self.env = EnvironmentRuntime()
        self.bridge = InjectionBridge(self.env)

        here = os.path.dirname(os.path.abspath(__file__))   # viewer/
        repo_root = os.path.dirname(here)                   # project root
        self.repo_root = repo_root
        data_root = os.path.join(repo_root, "data")

        self.store = SessionStore(data_root)

        # Recording pipeline manager
        self.record_intent = self.REC_OFF
        self.finalize_mgr = FinalizeManager()

        # Ensure finalize submissions are marshalled onto the Qt thread
        self.submit_finalize_job.connect(self._submit_finalize_job)

        # Ensure TakeCaptured events from EventBus are marshalled onto Qt thread
        self.take_captured_qt.connect(self._on_take_captured_qt)

        self.agent_controller = AgentController(
            policies={
                "Wander": StarterWanderPolicy(),
                "Oscillate": OscillatingPolicy(),
                "NoOp": NoOpPolicy(),
                "RL agent": RLActorCriticFoodPolicy(),
            },
            active_policy_name="RL agent",
            tools={"distance": EnvDistanceTool(self.env)},
        )
        self.rl_checkpoint_path = os.path.join(data_root, "checkpoints", "rl_agent_latest.pt")
        self.rl_training_enabled = False
        self.rl_episode_count = 0
        self.rl_steps_in_episode = 0
        self.rl_total_steps = 0
        self.rl_autosave_every = 20
        self.rl_max_episode_steps = 1000
        self._rl_last_step_seen = 0
        self.rl_last_autosave_episode = 0
        self.rl_training_start_time = None
        self.rl_episode_rewards = []
        self.rl_episode_steps_history = []
        self.rl_stop_reasons = {"phase_transition": 0, "death": 0, "max_steps": 0}

        self.live_runner = LiveRunner(
            env=self.env,
            agent_controller=self.agent_controller,
            bridge=self.bridge,
            store=self.store,
            bus=self.bus,
        )

        self.playback_runner = PlaybackRunner(
            store=self.store,
            bus=self.bus,
        )

        self.eval_runner = EvalRunner(
            env=self.env,
            bridge=self.bridge,
            agent_controller=self.agent_controller,
            store=self.store,
            bus=self.bus,
        )

        # ------------------------------------------------------------------
        # Window + root layout
        # ------------------------------------------------------------------

        self.setWindowTitle("Agent Research Workstation")

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)

        # ------------------------------------------------------------------
        # Toolbar
        # ------------------------------------------------------------------

        self.toolbar = QToolBar("Modes")
        self.addToolBar(self.toolbar)

        self.live_btn = QPushButton("Live Mode")
        self.play_btn = QPushButton("Playback Mode")
        self.eval_btn = QPushButton("Eval Mode")

        self.toolbar.addWidget(self.live_btn)
        self.toolbar.addWidget(self.play_btn)
        self.toolbar.addWidget(self.eval_btn)

        self.live_btn.clicked.connect(lambda: self._switch_mode("live"))
        self.play_btn.clicked.connect(lambda: self._switch_mode("playback"))
        self.eval_btn.clicked.connect(lambda: self._switch_mode("eval"))

        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_toggle_record)

        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.record_btn)

        self.freeze_btn = QPushButton("Freeze Agent")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.clicked.connect(
            lambda checked: self.bus.publish(
                "AgentFreezeToggled",
                {"enabled": checked}
            )
        )
        self.toolbar.addWidget(self.freeze_btn)

        # Human control toggle
        self.human_btn = QPushButton("Human Control")
        self.human_btn.setCheckable(True)
        # Publish intent; AppController will enforce invariants and publish final toggle
        self.human_btn.clicked.connect(
            lambda checked: self.bus.publish("HumanModeRequested", {"enabled": checked})
        )
        self.toolbar.addWidget(self.human_btn)

        # Edit mode toggle (god-mode)
        self.edit_mode_btn = QPushButton("Edit Mode")
        self.edit_mode_btn.setCheckable(True)
        # Publish intent; AppController will ensure edit=>pause invariant
        self.edit_mode_btn.clicked.connect(
            lambda checked: self.bus.publish("EditModeRequested", {"enabled": checked})
        )
        self.toolbar.addWidget(self.edit_mode_btn)

        self.toolbar.addSeparator()
        self.agent_selector = QComboBox()
        self.agent_selector.setMinimumWidth(140)
        self.agent_selector.addItems(self.agent_controller.list_policies())
        if self.agent_controller.active_policy_name is not None:
            self.agent_selector.setCurrentText(self.agent_controller.active_policy_name)
        self.agent_selector.currentTextChanged.connect(
            lambda name: self.bus.publish("AgentPolicyRequested", {"name": name})
        )
        self.toolbar.addWidget(self.agent_selector)

        self.train_rl_btn = QPushButton("Train RL")
        self.train_rl_btn.setCheckable(True)
        self.train_rl_btn.clicked.connect(self._on_train_rl_toggled)
        self.toolbar.addWidget(self.train_rl_btn)

        self.save_rl_btn = QPushButton("Save RL")
        self.save_rl_btn.clicked.connect(lambda: self._save_rl_checkpoint(auto=False))
        self.toolbar.addWidget(self.save_rl_btn)

        self.load_rl_btn = QPushButton("Load RL")
        self.load_rl_btn.clicked.connect(self._load_rl_checkpoint)
        self.toolbar.addWidget(self.load_rl_btn)

        self.play_control_btn = QPushButton("Play")
        self.play_control_btn.setCheckable(True)
        self.pause_control_btn = QPushButton("Pause")
        self.pause_control_btn.setCheckable(True)
        self.step_control_btn = QPushButton("Step")

        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.play_control_btn)
        self.toolbar.addWidget(self.pause_control_btn)
        self.toolbar.addWidget(self.step_control_btn)

        # Publish play/pause intents; AppController centralizes transitions
        self.play_control_btn.clicked.connect(lambda: self.bus.publish("PlayRequested", {}))
        self.pause_control_btn.clicked.connect(lambda: self.bus.publish("PauseRequested", {}))
        self.step_control_btn.clicked.connect(
            lambda: self.bus.publish("LiveStep", {})
        )

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(8)
        self.speed_slider.setValue(1)

        self.toolbar.addWidget(self.speed_slider)

        self.speed_slider.valueChanged.connect(
            lambda v: self.bus.publish(
                "LiveSpeedChanged",
                {"factor": v}
            )
        )

        # ------------------------------------------------------------------
        # TOP ROW: Trace browser | Trace player | Snapshot panel
        # ------------------------------------------------------------------

        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        root_layout.addLayout(top_row)

        self.trace_browser_panel = TraceBrowserPanel(self.store, self.bus)
        self.trace_player_panel = TracePlayerPanel(self.store, self.bus)
        self.snapshot_panel = SnapshotPanel(self.bus)

        # SnapshotPanel is only visible in edit (god) mode; keep it hidden by default
        self.snapshot_panel.setVisible(False)
        # Show/hide when edit mode toggles
        self.bus.subscribe("EditModeToggled", lambda p: self.snapshot_panel.setVisible(bool(p.get("enabled", False))))
        # Also hide while human-as-agent control is active; restore visibility based on edit_mode_btn
        self.bus.subscribe(
            "HumanAsAgentToggled",
            lambda p: self.snapshot_panel.setVisible(False)
            if p.get("enabled", False)
            else self.snapshot_panel.setVisible(bool(self.edit_mode_btn.isChecked())),
        )

        top_row.addWidget(self.trace_browser_panel, stretch=1)
        top_row.addWidget(self.trace_player_panel, stretch=2)
        top_row.addWidget(self.snapshot_panel, stretch=1)


        from PyQt6.QtWidgets import QStackedWidget

        self.viewport = QStackedWidget()
        root_layout.addWidget(self.viewport, stretch=1)

        self.live_env_panel = LiveEnvPanel(self.bus)
        self.session_preview_panel = SessionPreviewPanel(
            store=self.store,
            bus=self.bus,
        )
        self.rl_training_panel = RLTrainingPanel()
        self.rl_training_panel.set_max_episode_steps(self.rl_max_episode_steps)
        self.rl_training_panel.max_steps_spin.valueChanged.connect(self._on_rl_max_steps_changed)

        self.viewport.addWidget(self.live_env_panel)        # index 0
        self.viewport.addWidget(self.session_preview_panel) # index 1
        self.viewport.addWidget(self.rl_training_panel)     # index 2

        self.viewport.setCurrentWidget(self.live_env_panel)

        # ------------------------------------------------------------------
        # State + subscriptions
        # ------------------------------------------------------------------

        self.current_mode = "live"
        self.bus.publish("ModeChanged", {"mode": "live"})
        # edit mode state (god-mode)
        self.edit_mode = False
        self.bus.subscribe("EditModeToggled", lambda p: setattr(self, "edit_mode", bool(p.get("enabled", False))))

        self.bus.subscribe("InjectRequested", self._on_inject_request)
        self.bus.subscribe("AgentStepRequested", self._on_agent_step)

        self.bus.subscribe("PlaybackSelectClip", self._on_playback_select)
        self.bus.subscribe("PlaybackSeekStep", self._on_playback_seek)
        self.bus.subscribe("PlaybackPlay", lambda _: self.playback_runner.play())
        self.bus.subscribe("PlaybackPause", lambda _: self.playback_runner.pause())

        self.bus.subscribe("SnapshotEdited", self._on_snapshot_edited)
        self.bus.subscribe("EvalRunRequested", self._on_eval_run)
        self.bus.subscribe("EvalPauseRequested", lambda _: self.eval_runner.pause())
        self.bus.subscribe("EvalStepRequested", lambda _: self.eval_runner.step_once())

        self.bus.subscribe("ModeChanged", self._on_mode_changed)
        # Live rendering now uses EnvRenderPacket (frame+snapshot+telemetry)
        self.bus.subscribe("EnvRenderPacket", self._on_env_telemetry)
        self.bus.subscribe("EnvEditRequested", self._on_env_edit_requested)
        self.bus.subscribe("EnvResetRequested", self._on_env_reset)
        # recording pipeline events: marshal EventBus->Qt via signal
        self.bus.subscribe(
            "TakeCaptured",
            lambda payload: self.take_captured_qt.emit(payload),
        )
        # listen for pipeline depth changes from finalize manager
        try:
            self.finalize_mgr.pipeline_depth_changed.connect(self._on_pipeline_depth)
        except Exception:
            pass
        try:
            self.finalize_mgr.busy_changed.connect(self._on_finalize_busy)
        except Exception:
            pass
        # Centralized control intents
        self.bus.subscribe("PlayRequested", self._on_play_requested)
        self.bus.subscribe("PauseRequested", self._on_pause_requested)
        self.bus.subscribe("EditModeRequested", self._on_edit_mode_requested)
        self.bus.subscribe("HumanModeRequested", self._on_human_mode_requested)
        self.bus.subscribe("AgentPolicyRequested", self._on_agent_policy_requested)
        # UI should disable edit mode while human control is active
        self.bus.subscribe("HumanAsAgentToggled", self._on_human_control_toggled)
        self.bus.subscribe("RecordMark", self._on_record_mark)
        self.bus.subscribe("AgentFreezeToggled", self._on_agent_freeze)

        # Focus request: when the runner asks for focus shift to the live env
        self.bus.subscribe("RequestFocusLiveEnv", lambda p: QTimer.singleShot(0, self.live_env_panel.setFocus))

        # Finalize cancellation signal (if provided by FinalizeManager)
        try:
            self.finalize_mgr.finalize_canceled.connect(self._on_finalize_canceled)
        except Exception:
            pass
        # Capture lifecycle events
        self.bus.subscribe("CaptureStarted", self._on_capture_started)
        self.bus.subscribe("CaptureStopped", self._on_capture_stopped)
        try:
            self.finalize_mgr.finalize_failed.connect(self._on_finalize_failed)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # GUI tick
        # ------------------------------------------------------------------

        self.timer = QTimer()
        self.timer.timeout.connect(self._on_gui_tick)
        self.timer.start(16)

        self.last_tick_time = time.time()

        # Canonical controller state
        self.sim_state = "RUNNING"
        self.control_mode = "AGENT"

        # Preview hooks

        try:
            self.bus.subscribe("SessionPreviewActivated", self._on_session_preview_activated)
        except Exception:
            pass

        try:
            self.bus.subscribe("SessionPreviewThumbnailSet", self._on_session_thumbnail_set)
        except Exception:
            pass

        try:
            self.finalize_mgr.job_finished.connect(self._on_finalize_job_finished)
        except Exception:
            pass

        try:
            # UI panels refresh on finalize completion via EventBus (deferred)
            self.bus.subscribe(
                "FinalizeFinished",
                lambda p: QTimer.singleShot(0, self.session_preview_panel.refresh),
            )
        except Exception:
            pass

        try:
            self.bus.subscribe(
                "FinalizeFinished",
                lambda p: QTimer.singleShot(0, self.trace_browser_panel.refresh_sessions),
            )
        except Exception:
            pass

                                                           
    def _switch_mode(self, mode: str):
        # Ensure any active capture stops when leaving live mode
        if self.current_mode == "live" and mode != "live":
            try:
                if self.record_intent == self.REC_CAPTURING:
                    self.bus.publish("CaptureStopRequested", {})
            except Exception:
                pass
            self.record_intent = self.REC_OFF
            try:
                self.record_btn.setChecked(False)
            except Exception:
                pass

        self.current_mode = mode
        self.bus.publish("ModeChanged", {"mode": mode})

                                                                
        self.bus.publish("PanelModePermissions", {
            "mode": mode,
            "live_env_editable": mode in ("live", "eval"),
            "snapshot_injection_enabled": mode != "playback",
            "trace_player_editable": mode == "playback",
        })

        # ensure live panel receives keyboard focus when switching to live
        if mode == "live":
            try:
                self.live_env_panel.setFocus()
            except Exception:
                pass

    def _on_mode_changed(self, payload):
        mode = payload["mode"]

        # Viewport switching: live/eval show live view, playback shows session preview
        try:
            if mode == "live" or mode == "eval":
                if mode == "live" and self.rl_training_enabled:
                    self._show_rl_training_view()
                else:
                    self._show_live_view()
            elif mode == "playback":
                self._show_session_preview()
        except Exception:
            pass

        # Leaving live mode must always stop capture immediately
        if mode != "live":
            if getattr(self, "rl_training_enabled", False):
                self._set_rl_training(False)
            try:
                if self.record_intent == self.REC_CAPTURING:
                    self.bus.publish("CaptureStopRequested", {})
            except Exception:
                pass
            self.record_intent = self.REC_OFF
            try:
                self.record_btn.setChecked(False)
            except Exception:
                pass

    # ------------------------------------------------------------
    # Centralized control intent handlers
    # ------------------------------------------------------------
    def _on_play_requested(self, payload):
        # Play -> running; enforce edit_mode OFF
        self.sim_state = "RUNNING"
        self.edit_mode = False
        try:
            self.edit_mode_btn.setChecked(False)
        except Exception:
            pass
        try:
            self.play_control_btn.setChecked(True)
            self.pause_control_btn.setChecked(False)
        except Exception:
            pass
        # publish legacy event consumed by runners
        self.bus.publish("LivePlay", {})
        # If user armed recording while paused, start capture on play — check pipeline
        if self.record_intent == self.REC_ARMED:
            try:
                # Use pipeline depth policy rather than is_busy() to decide
                if self.finalize_mgr.pipeline_depth() < 2:
                    # disk preflight
                    if not self._has_disk_space():
                        try:
                            self.record_btn.setChecked(False)
                        except Exception:
                            pass
                        self._notify_user("Not enough disk space to start recording.")
                        return
                    self.record_intent = self.REC_REQUESTED
                    self.bus.publish("CaptureStartRequested", {})
                else:
                    # Backpressure: disarm and pause
                    self.record_intent = self.REC_OFF
                    try:
                        self.record_btn.setChecked(False)
                    except Exception:
                        pass
                    self._pause_internal("Recording backlog full")
            except Exception:
                pass

    def _on_pause_requested(self, payload):
        self.sim_state = "PAUSED"
        try:
            self.play_control_btn.setChecked(False)
            self.pause_control_btn.setChecked(True)
        except Exception:
            pass
        self.bus.publish("LivePause", {})

    def _on_edit_mode_requested(self, payload):
        enabled = bool(payload.get("enabled", False))
        if enabled:
            # Editing forces pause
            self.sim_state = "PAUSED"
            try:
                self.play_control_btn.setChecked(False)
                self.pause_control_btn.setChecked(True)
            except Exception:
                pass
            self.edit_mode = True
            try:
                self.edit_mode_btn.setChecked(True)
            except Exception:
                pass
            self.bus.publish("EditModeToggled", {"enabled": True})
            # Ensure runtime is paused so edits apply immediately
            self.bus.publish("LivePause", {})
        else:
            self.edit_mode = False
            try:
                self.edit_mode_btn.setChecked(False)
            except Exception:
                pass
            self.bus.publish("EditModeToggled", {"enabled": False})

    def _on_human_mode_requested(self, payload):
        enabled = bool(payload.get("enabled", False))
        self.control_mode = "HUMAN" if enabled else "AGENT"
        try:
            self.human_btn.setChecked(enabled)
        except Exception:
            pass
        # publish the legacy toggle for downstream consumers
        self.bus.publish("HumanAsAgentToggled", {"enabled": enabled})

    def _on_agent_policy_requested(self, payload):
        name = str(payload.get("name", "")).strip()
        if not name:
            return

        if self.rl_training_enabled and name != "RL agent":
            self._set_rl_training(False)
            self._notify_user("RL training stopped because active policy changed.")

        if not self.agent_controller.set_active_policy(name):
            self._notify_user(f"Unknown agent policy: {name}")
            return

        try:
            self.agent_selector.blockSignals(True)
            self.agent_selector.setCurrentText(name)
        except Exception:
            pass
        finally:
            try:
                self.agent_selector.blockSignals(False)
            except Exception:
                pass

        try:
            self.bus.publish("AgentPolicyChanged", {"name": name})
        except Exception:
            pass

    def _get_rl_policy(self):
        policy = self.agent_controller.policies.get("RL agent")
        if policy is None:
            self._notify_user("RL agent policy is not registered.")
            return None
        return policy

    def _set_agent_selector_text(self, name: str):
        try:
            self.agent_selector.blockSignals(True)
            self.agent_selector.setCurrentText(name)
        except Exception:
            pass
        finally:
            try:
                self.agent_selector.blockSignals(False)
            except Exception:
                pass

    def _set_rl_training(self, enabled: bool):
        self.rl_training_enabled = bool(enabled)
        try:
            self.train_rl_btn.blockSignals(True)
            self.train_rl_btn.setChecked(self.rl_training_enabled)
        except Exception:
            pass
        finally:
            try:
                self.train_rl_btn.blockSignals(False)
            except Exception:
                pass
        # Training mode runs with no fixed GUI tick interval.
        try:
            self.timer.start(0 if self.rl_training_enabled else 16)
        except Exception:
            pass
        if self.rl_training_enabled and self.current_mode == "live":
            self._show_rl_training_view()
            self._update_rl_training_dashboard()
        elif (not self.rl_training_enabled) and self.current_mode == "live":
            self._show_live_view()

    def _on_train_rl_toggled(self, checked: bool):
        if not checked:
            self._set_rl_training(False)
            return

        policy = self._get_rl_policy()
        if policy is None:
            self._set_rl_training(False)
            return

        if self.current_mode != "live":
            self._switch_mode("live")

        if self.agent_controller.active_policy_name != "RL agent":
            ok = self.agent_controller.set_active_policy("RL agent")
            if not ok:
                self._notify_user("Could not activate RL agent policy.")
                self._set_rl_training(False)
                return
            self._set_agent_selector_text("RL agent")

        # Training should run with agent control, not human control.
        self.bus.publish("HumanModeRequested", {"enabled": False})
        self.bus.publish("PlayRequested", {})

        try:
            policy.reset_episode()
        except Exception:
            pass

        self.rl_episode_count = 0
        self.rl_steps_in_episode = 0
        self.rl_total_steps = 0
        self._rl_last_step_seen = int(getattr(self.env, "global_step", 0))
        self.rl_last_autosave_episode = 0
        self.rl_training_start_time = time.time()
        self.rl_episode_rewards = []
        self.rl_episode_steps_history = []
        self.rl_stop_reasons = {"phase_transition": 0, "death": 0, "max_steps": 0}
        self._set_rl_training(True)

    def _on_rl_max_steps_changed(self, value: int):
        self.rl_max_episode_steps = int(value)
        self._update_rl_training_dashboard()

    def _save_rl_checkpoint(self, *, auto: bool):
        policy = self._get_rl_policy()
        if policy is None:
            return
        try:
            policy.save(self.rl_checkpoint_path)
            if not auto:
                self._notify_user(f"Saved RL checkpoint:\n{self.rl_checkpoint_path}")
            else:
                self.rl_last_autosave_episode = int(self.rl_episode_count)
        except Exception as e:
            self._notify_user(f"Failed to save RL checkpoint:\n{e}")

    def _load_rl_checkpoint(self):
        policy = self._get_rl_policy()
        if policy is None:
            return
        if not os.path.exists(self.rl_checkpoint_path):
            self._notify_user(f"No checkpoint found:\n{self.rl_checkpoint_path}")
            return
        try:
            policy.load(self.rl_checkpoint_path)
            if self.agent_controller.active_policy_name != "RL agent":
                self.agent_controller.set_active_policy("RL agent")
                self._set_agent_selector_text("RL agent")
            self._notify_user(f"Loaded RL checkpoint:\n{self.rl_checkpoint_path}")
            self._update_rl_training_dashboard()
        except Exception as e:
            self._notify_user(f"Failed to load RL checkpoint:\n{e}")

    def _update_rl_training_dashboard(self):
        policy = self._get_rl_policy()
        snap = self.env.snapshot_state()
        phase = int(getattr(snap, "phase", 1))
        stomach = float(getattr(snap, "stomach", 0.0))

        learning_enabled = "-"
        total_updates = 0
        last_reward = 0.0
        episode_reward = 0.0
        best_episode_reward = 0.0

        if policy is not None:
            learning_enabled = bool(getattr(policy, "learning_enabled", False))
            total_updates = int(getattr(policy, "total_updates", 0))
            last_reward = float(getattr(policy, "last_reward", 0.0))
            episode_reward = float(getattr(policy, "episode_reward", 0.0))
            best_episode_reward = float(getattr(policy, "best_episode_reward", 0.0))

        if best_episode_reward == float("-inf"):
            best_episode_reward = 0.0

        elapsed = 0.0
        if self.rl_training_start_time is not None:
            elapsed = max(1e-6, time.time() - float(self.rl_training_start_time))
        steps_per_sec = (float(self.rl_total_steps) / elapsed) if elapsed > 0.0 else 0.0
        episodes_per_min = (float(self.rl_episode_count) / elapsed * 60.0) if elapsed > 0.0 else 0.0

        recent_rewards = self.rl_episode_rewards[-20:]
        recent_steps = self.rl_episode_steps_history[-20:]
        reward_avg_20 = (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
        steps_avg_20 = (sum(recent_steps) / len(recent_steps)) if recent_steps else 0.0

        metrics = {
            "training_status": "ON" if self.rl_training_enabled else "OFF",
            "policy_name": str(self.agent_controller.active_policy_name or "-"),
            "episode_count": int(self.rl_episode_count),
            "steps_episode": int(self.rl_steps_in_episode),
            "steps_total": int(self.rl_total_steps),
            "updates_total": int(total_updates),
            "steps_per_sec": f"{steps_per_sec:.2f}",
            "episodes_per_min": f"{episodes_per_min:.2f}",
            "phase": int(phase),
            "learning_enabled": learning_enabled,
            "stomach": f"{stomach:.4f}",
            "last_reward": f"{last_reward:.6f}",
            "episode_reward": f"{episode_reward:.6f}",
            "best_episode_reward": f"{best_episode_reward:.6f}",
            "reward_avg_20": f"{reward_avg_20:.6f}",
            "steps_avg_20": f"{steps_avg_20:.2f}",
            "phase1_stop_phase2": int(self.rl_stop_reasons.get("phase_transition", 0)),
            "phase1_stop_death": int(self.rl_stop_reasons.get("death", 0)),
            "phase1_stop_max_steps": int(self.rl_stop_reasons.get("max_steps", 0)),
            "autosave_every": f"{int(self.rl_autosave_every)} episodes",
            "last_autosave_episode": int(self.rl_last_autosave_episode),
            "checkpoint_path": self.rl_checkpoint_path,
            "reward_history": self.rl_episode_rewards[-120:],
            "steps_history": self.rl_episode_steps_history[-120:],
        }
        self.rl_training_panel.update_metrics(metrics)

    def _maybe_handle_rl_training_step(self):
        if not self.rl_training_enabled:
            return
        if self.current_mode != "live":
            return

        step_now = int(getattr(self.env, "global_step", 0))
        if step_now <= self._rl_last_step_seen:
            return

        step_delta = step_now - self._rl_last_step_seen
        self._rl_last_step_seen = step_now
        self.rl_steps_in_episode += step_delta
        self.rl_total_steps += step_delta

        snap = self.env.snapshot_state()
        phase = int(getattr(snap, "phase", 1))
        death = bool(self.env.check_death()) if hasattr(self.env, "check_death") else False
        max_steps_hit = self.rl_steps_in_episode >= int(self.rl_max_episode_steps)

        # Hard cap training episodes at phase 1.
        if phase >= 2 or death or max_steps_hit:
            self.rl_episode_count += 1
            if max_steps_hit:
                reason = "max_steps"
            elif phase >= 2:
                reason = "phase_transition"
            else:
                reason = "death"
            self.rl_stop_reasons[reason] = int(self.rl_stop_reasons.get(reason, 0)) + 1

            policy = self._get_rl_policy()
            if reason == "death" and policy is not None:
                try:
                    policy.apply_terminal_reward(-100.0)
                except Exception:
                    pass
            episode_reward = 0.0
            if policy is not None:
                episode_reward = float(getattr(policy, "episode_reward", 0.0))
            self.rl_episode_rewards.append(episode_reward)
            self.rl_episode_steps_history.append(int(self.rl_steps_in_episode))
            self.bus.publish(
                "RLTrainingEpisodeFinished",
                {
                    "episode": self.rl_episode_count,
                    "steps": self.rl_steps_in_episode,
                    "reason": reason,
                },
            )

            if self.rl_episode_count % self.rl_autosave_every == 0:
                self._save_rl_checkpoint(auto=True)

            self.rl_steps_in_episode = 0

            # Reset to a fresh phase-1 episode and continue training forever
            self.env.reset()
            if policy is not None:
                try:
                    policy.reset_episode()
                except Exception:
                    pass

            self._rl_last_step_seen = int(getattr(self.env, "global_step", 0))
            try:
                self._emit_render_packet()
            except Exception:
                pass
            try:
                self.bus.publish("EnvStateUpdated", self.env.snapshot_state())
            except Exception:
                pass

        self._update_rl_training_dashboard()

                                                           
    def _on_toggle_record(self, checked: bool):
        # Focus live env for recording
        try:
            self.bus.publish("RequestFocusLiveEnv", {})
        except Exception:
            pass

        # If enabling recording
        if checked:
            # Note: admission control is based exclusively on pipeline depth
            # handled in _on_pipeline_depth. Do not gate here using is_busy().

            # disk preflight
            if not self._has_disk_space():
                try:
                    self.record_btn.setChecked(False)
                except Exception:
                    pass
                self._notify_user("Not enough disk space to start recording.")
                return

            if self.sim_state == "RUNNING":
                # guard duplicate requests
                if self.record_intent in (self.REC_REQUESTED, self.REC_CAPTURING):
                    return
                # request the runner to start capture
                self.record_intent = self.REC_REQUESTED
                self.bus.publish("CaptureStartRequested", {})
            else:
                self.record_intent = self.REC_ARMED
        else:
            if self.record_intent in (self.REC_CAPTURING, self.REC_REQUESTED):
                self.bus.publish("CaptureStopRequested", {})
            self.record_intent = self.REC_OFF

    def _on_take_captured_qt(self, payload: Dict[str, Any]):
        from dataset.finalize_worker import FinalizeJob

        job = FinalizeJob(
            pending_dir=payload.get("pending_dir"),
            output_dir=self.store.root_dir,
        )

        # Marshal submission back onto the Qt thread
        try:
            self.submit_finalize_job.emit(job)
        except Exception:
            # Fallback: ensure UI and intent are safe
            try:
                self.record_intent = self.REC_OFF
                self.record_btn.setChecked(False)
            except Exception:
                pass
            try:
                self._pause_internal("Finalize submission failed")
            except Exception:
                pass

    def _on_pipeline_depth(self, depth: int):
        """
        depth = active + queued finalize jobs

        Policy:
        - depth < 2 -> recording allowed
        - depth >= 2 -> recording disabled + pause
        """
        try:
            if depth >= 2:
                self.record_btn.setEnabled(False)

                # Disarm any pending intent
                if self.record_intent in (self.REC_ARMED, self.REC_REQUESTED):
                    self.record_intent = self.REC_OFF
                    try:
                        self.record_btn.setChecked(False)
                    except Exception:
                        pass

                self._pause_internal("Recording backlog full")
            else:
                self.record_btn.setEnabled(True)
        except Exception:
            pass

    def _on_finalize_busy(self, busy: bool):
        """
        Busy means a save is currently running.
        This must NOT gate recording. Recording is gated exclusively by pipeline depth.
        """
        # Optional: visual hint only
        try:
            self.record_btn.setProperty("saving", bool(busy))
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
        except Exception:
            pass

    def _on_record_mark(self, payload):


        self.bus.publish("RecordMarkCaptured", payload)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _pause_internal(self, reason: str = ""):
        self.sim_state = "PAUSED"
        try:
            self.play_control_btn.setChecked(False)
            self.pause_control_btn.setChecked(True)
        except Exception:
            pass
        try:
            self.bus.publish("LivePause", {})
        except Exception:
            pass
        if reason:
            self._notify_user(reason)

    def _notify_user(self, msg: str):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Notice", msg)
        except Exception:
            try:
                print("NOTICE:", msg)
            except Exception:
                pass

    def _has_disk_space(self, min_gb: float = 1.0) -> bool:
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.store.root_dir)
            return (free / (1024 ** 3)) >= float(min_gb)
        except Exception:
            return True

    def _on_capture_started(self, payload: Dict[str, Any]):
        try:
            if self.record_intent == self.REC_REQUESTED:
                self.record_intent = self.REC_CAPTURING
        except Exception:
            pass

    def _on_capture_stopped(self, payload: Dict[str, Any]):
        try:
            if self.record_intent == self.REC_CAPTURING:
                self.record_intent = self.REC_OFF
                try:
                    self.record_btn.setChecked(False)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_finalize_failed(self, take_id: str, error: str):
        try:
            self._notify_user(f"Recording save failed:\n{error}")
            self.record_btn.setEnabled(True)
        except Exception:
            pass

    def _submit_finalize_job(self, job):
        """
        Slot that runs on the Qt main thread to submit finalize jobs safely.
        """
        try:
            ok = self.finalize_mgr.submit(job)
            if not ok:
                self.record_intent = self.REC_OFF
                try:
                    self.record_btn.setChecked(False)
                except Exception:
                    pass
                self._pause_internal("Finalize submission failed")
        except Exception:
            try:
                self.record_intent = self.REC_OFF
                try:
                    self.record_btn.setChecked(False)
                except Exception:
                    pass
                self._pause_internal("Finalize submission failed")
            except Exception:
                pass

    # -----------------
    # Viewport helpers
    # -----------------
    def _show_live_view(self):
        try:
            self.viewport.setCurrentWidget(self.live_env_panel)
            try:
                self.live_env_panel.setFocus()
            except Exception:
                pass
        except Exception:
            pass

    def _show_session_preview(self):
        try:
            self.viewport.setCurrentWidget(self.session_preview_panel)
            self.session_preview_panel.refresh()
            try:
                self.session_preview_panel.setFocus()
            except Exception:
                pass
        except Exception:
            pass

    def _show_rl_training_view(self):
        try:
            self.viewport.setCurrentWidget(self.rl_training_panel)
        except Exception:
            pass

    def _on_session_preview_activated(self, payload: Dict[str, Any]):
        try:
            sid = payload.get("session_id")
            if not sid:
                return

            # switch mode first
            self.enter_playback_mode()

            # choose default clip (first available)
            clips = self.store.clips.load(sid)
            clip_id = next(iter(clips.keys()), None)
            if clip_id is None:
                return

            self.bus.publish(
                "PlaybackSelectClip",
                {
                    "session_id": sid,
                    "clip_id": clip_id,
                },
            )
        except Exception:
            pass

    def enter_playback_mode(self):
        try:
            self._switch_mode("playback")
            self._show_session_preview()
        except Exception:
            pass

    def _on_session_thumbnail_set(self, payload: Dict[str, Any]):
        try:
            sid = payload.get("session_id")
            step = int(payload.get("step_index", 0))
            if not sid:
                return
            # Persist thumbnail metadata and invalidate caches
            try:
                self.store.clips.save_session_meta(sid, {"thumbnail_step": step})
            except Exception:
                pass
            try:
                self.session_preview_panel.thumb_cache.invalidate(sid)
            except Exception:
                pass
            try:
                self.session_preview_panel.refresh()
            except Exception:
                pass
            try:
                self.trace_browser_panel.refresh_sessions()
            except Exception:
                pass
        except Exception:
            pass

    def _on_finalize_finished(self, session_id: str):
        # DEPRECATED: Use `_on_finalize_job_finished` bridge to EventBus instead.
        try:
            self.session_preview_panel.refresh()
        except Exception:
            pass

    def _on_finalize_job_finished(self, session_id: str):
        """
        Bridge finalize completion into the EventBus.
        This is the ONLY place that knows about both FinalizeManager and EventBus.
        """
        try:
            self.bus.publish(
                "FinalizeFinished",
                {"session_id": session_id},
            )
        except Exception:
            pass

                                                           
    def _on_gui_tick(self):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now

        if self.current_mode == "live":
            if self.rl_training_enabled:
                self.live_runner._on_step_request({"mode": "live", "training_mode": True})
            else:
                self.live_runner.tick(dt=dt)
            self._maybe_handle_rl_training_step()

        elif self.current_mode == "playback":
                        
                                                                               
            self.playback_runner.tick()

        elif self.current_mode == "eval":
            pass

        # Rendering is event-driven via EnvRenderPacket; no per-tick refresh needed

                                                           
    def _on_inject_request(self, payload):
        raw_snapshot = payload["snapshot"]

        # In live mode, only allow injection when paused to avoid surprising overwrites
        if self.current_mode == "live" and not getattr(self.live_runner, "is_paused", False):
            return

        normalized = self.bridge.normalize_snapshot(raw_snapshot)

                                                          
        self.bridge.apply_snapshot(normalized)

        # After applying an injected snapshot to the runtime, emit render packet
        try:
            self._emit_render_packet()
        except Exception:
            pass

                                            
        incoming_prov = payload.get("provenance", {}) or {}

        provenance = {
                                          
            "source": incoming_prov.get("source", "manual"),

            "session_id": incoming_prov.get("session_id"),
            "clip_id": incoming_prov.get("clip_id"),
            "step_index": incoming_prov.get("step_index"),

                                                        
            "branch_parent_id": incoming_prov.get("branch_parent_id"),
            "branch_depth": int(incoming_prov.get("branch_depth", 0)),

                                                
            "modifications": incoming_prov.get("modifications", {}),
        }

                               
        self.bus.publish("EnvProvenanceUpdated", provenance)
        self.bus.publish("EnvStateUpdated", normalized)

                                                           
    def _on_snapshot_edited(self, payload):


        candidate = payload["snapshot"]

        normalized, advisories = self.bridge.validate_snapshot_preview(candidate)

        self.bus.publish("SnapshotValidationResult", {
            "normalized": normalized,
            "advisories": advisories,                                 
        })

                                                           
    def _on_agent_step(self, payload):
                         
        if self.current_mode == "playback":
            return

        # Build observation via LiveRunner's agent renderer (strict)
        snapshot = self.env.snapshot_state()
        agent_renderer = self.live_runner.agent_renderer
        try:
            frame = agent_renderer.render(snapshot)
        except Exception:
            frame = None

        obs = build_observation(snapshot, frame)
        action, _ = self.agent_controller.run_step(obs)
        self.env.step(action)

        # After stepping the authoritative runtime, emit render packet
        try:
            self._emit_render_packet()
        except Exception:
            pass

        state = self.env.snapshot_state()
        self.bus.publish("EnvStateUpdated", state)

                                                           
    def _on_playback_select(self, payload):
        print(f"[AppController] _on_playback_select: session={payload.get('session_id')} clip={payload.get('clip_id')}")
        self.playback_runner.load_clip(
            payload["session_id"],
            payload["clip_id"]
        )

    def _on_playback_seek(self, payload):
                                               
        self.playback_runner.seek_step(payload["step"])

                                                           
    def _on_eval_run(self, payload):
        steps = payload.get("num_steps", 50)

                                                           
        if "snapshot" in payload:
                                              
            snapshot = payload["snapshot"]
        else:
                                       
            snapshot = self.store.get_step_snapshot(
                payload.get("session_id"),
                payload.get("step_index"),
            )

                                                           
        normalized = self.bridge.normalize_snapshot(snapshot)
        self.bridge.apply_snapshot(normalized)

                                                           
        parent_prov = payload.get("provenance", {}) or {}

        parent_session = parent_prov.get("session_id", payload.get("session_id"))
        parent_clip = parent_prov.get("clip_id", payload.get("clip_id"))
        parent_step = parent_prov.get("step_index", payload.get("step_index"))

        parent_depth = int(parent_prov.get("branch_depth", 0))

                                                           
        child_depth = parent_depth + 1

        provenance = {
            "source": "eval",

                                    
            "session_id": parent_session,
            "clip_id": parent_clip,
            "step_index": parent_step,

                     
            "branch_parent_id": parent_session,
            "branch_depth": child_depth,

            "modifications": payload.get("modifications", {}),
        }

                                                           
        self.bus.publish("EnvProvenanceUpdated", provenance)

                                                           
        self.eval_runner.run_eval_from_state(
            snapshot=normalized,
            num_steps=steps,
            agent_id=payload.get("agent_id", None),

            branch_parent_session=parent_session,
            branch_parent_clip=parent_clip,
            branch_parent_step=parent_step,

            branch_depth=child_depth,

            on_step=lambda step_idx, state: self.bus.publish(
                "EvalStepCompleted",
                {
                    "step_idx": step_idx,
                    "state": state,
                    "branch_parent_session": parent_session,
                    "branch_depth": child_depth,
                },
            ),
            on_finish=lambda final, reason: self.bus.publish(
                "EvalFinished",
                {
                    "final_state": final,
                    "termination_reason": reason,
                    "branch_parent_session": parent_session,
                    "branch_parent_clip": parent_clip,
                    "branch_parent_step": parent_step,
                    "branch_depth": child_depth,
                },
            ),
            on_interrupt=lambda reason: self.bus.publish(
                "EvalInterrupted",
                {
                    "reason": reason,
                    "branch_parent_session": parent_session,
                    "branch_parent_clip": parent_clip,
                    "branch_parent_step": parent_step,
                    "branch_depth": child_depth,
                },
            ),
        )

    def _emit_render_packet(self):
        """Create and publish an EnvRenderPacket from the authoritative runtime state."""
        snap = self.env.snapshot_state()

        # Use the runner's configured UI renderer (strict)
        renderer = self.live_runner.ui_renderer

        # Always produce a frame. Never publish frame=None.
        try:
            frame = renderer.render(snap, overlays={"rooms": True})
        except Exception as e:
            # Development-friendly: print once per failure
            print("[_emit_render_packet] render failed:", repr(e))
            # Fallback to blank frame at the renderer output size
            out_size = getattr(renderer, "frame_size", 84)
            frame = np.zeros((out_size, out_size, 3), dtype=np.float32)

        try:
            telemetry = self.env.build_telemetry()
            if telemetry is None:
                telemetry = {}
        except Exception as e:
            print("[_emit_render_packet] telemetry failed:", repr(e))
            telemetry = {}

        packet = {
            "frame": frame,
            "snapshot": snap,
            "telemetry": telemetry,
            "sim_time": float(getattr(self.env, "sim_time", 0.0)),
            "step_index": int(getattr(self.env, "global_step", 0)),
        }

        self.bus.publish("EnvRenderPacket", packet)

                                                           
    def _on_env_telemetry(self, payload):
        # payload is EnvRenderPacket; forward telemetry to visual panel
        telemetry = payload.get("telemetry", {}) if isinstance(payload, dict) else {}
        self.live_env_panel.update_overlay(telemetry)

                                                           
    def _on_agent_freeze(self, payload):
        if payload["enabled"]:
            self.agent_controller.freeze_policy()
        else:
            self.agent_controller.unfreeze_policy()

    def _on_env_reset(self, payload: Dict[str, Any]):
        # Only allow UI-triggered resets in live mode
        if self.current_mode != "live":
            return
        # Reset the authoritative runtime
        self.env.reset()

        # Emit render packet so UI and panels update from authoritative state
        try:
            self._emit_render_packet()
        except Exception:
            pass

        # Also publish state update for legacy consumers
        try:
            snap = self.env.snapshot_state()
            self.bus.publish("EnvStateUpdated", snap)
        except Exception:
            pass

    def _on_human_control_toggled(self, payload):
        enabled = bool(payload.get("enabled", False))
        try:
            # disable edit mode while human-as-agent is active
            self.edit_mode_btn.setEnabled(not enabled)
            if enabled and self.edit_mode_btn.isChecked():
                # turn off edit mode if human control is enabled
                self.edit_mode_btn.setChecked(False)
                self.bus.publish("EditModeToggled", {"enabled": False})
            # ensure live panel has focus so it receives key events
            if enabled:
                try:
                    self.live_env_panel.setFocus()
                except Exception:
                    pass
        except Exception:
            pass

    def _can_edit_live_env(self) -> bool:
        # Only allow edits when in live mode and live runner is paused
        if self.current_mode != "live":
            return False
        return getattr(self.live_runner, "is_paused", False) and getattr(self, "edit_mode", False)

    def _on_env_edit_requested(self, payload):
        # Payload expected to contain a 'mutation' dict or full snapshot
        if not self._can_edit_live_env():
            return

        mutation = payload.get("mutation", {}) or {}
        phase_was_edited = "phase" in mutation

        # Get current canonical snapshot
        current = self.env.snapshot_state()

        # Convert current snapshot to a plain runtime dict, merge mutation
        try:
            candidate = snapshot_to_runtime_dict(current)
        except Exception:
            # fallback: try dict-like conversion
            try:
                candidate = dict(current)
            except Exception:
                candidate = {}

        # deep merge for nested fields so patches like {"wall": {"enabled": False}} work
        def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(base)
            for k, v in (patch or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_update(out.get(k, {}), v)
                else:
                    out[k] = v
            return out

        candidate = _deep_update(candidate, mutation)

        # normalize into EnvStateSnapshot and validate preview
        normalized_snap = self.bridge.normalize_snapshot(candidate)
        normalized, advisories = self.bridge.validate_snapshot_preview(normalized_snap)

        # Publish validation results to UI
        try:
            self.bus.publish("SnapshotValidationResult", {
                "normalized": normalized,
                "advisories": advisories,
            })
        except Exception:
            pass

        # Apply the normalized snapshot to the environment (edit-origin)
        self.bridge.apply_snapshot(normalized, from_edit=True)

        if phase_was_edited:
            try:
                self.env._configure_phase_state()
            except Exception:
                pass

        # After applying an edit to the runtime, emit render packet
        try:
            self._emit_render_packet()
        except Exception:
            pass

        # Notify other systems of the new state
        self.bus.publish("EnvStateUpdated", normalized)

    def closeEvent(self, event):
        try:
            busy = self.finalize_mgr.is_busy()
        except Exception:
            try:
                state = self.finalize_mgr.snapshot()
                depth = int(state.get("active", 0)) + int(state.get("queued", 0))
                busy = depth > 0
            except Exception:
                busy = False

        if busy:
            from PyQt6.QtWidgets import QMessageBox

            r = QMessageBox.warning(
                self,
                "Save in progress",
                "A recording is still being saved.\n\n"
                "Yes = kill save and exit\n"
                "No = cancel exit",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r == QMessageBox.StandardButton.Yes:
                try:
                    self.finalize_mgr.kill_active()
                except Exception:
                    pass
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _on_finalize_canceled(self, take_id: str):
        try:
            print(f"Finalize canceled for take {take_id}")
        except Exception:
            pass


def run_app():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    win = AppController()
    win.resize(1400, 900)
    win.show()

    sys.exit(app.exec())
