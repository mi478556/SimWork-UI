# live_runner.py

from __future__ import annotations

from typing import Optional, Dict, Any

from viewer.event_bus import EventBus
from engine.environment_runtime import EnvironmentRuntime
from engine.injection_bridge import InjectionBridge
from agent.agent_controller import AgentController
from agent.logging_agent import LoggingAgent
from engine.snapshot_state import EnvStateSnapshot                          
from engine.env_renderer import EnvRenderer
from runner.observation_builder import build_observation
import time

from dataset.session_store import SessionStore
from dataset.recorder import SessionRecorder
from agent.human_controller import HumanController


class LiveRunner:


    def __init__(
        self,
        env: EnvironmentRuntime,
        agent_controller: AgentController,
        bridge: InjectionBridge,
        store: SessionStore,
        bus: EventBus,
        logging_agent: Optional[LoggingAgent] = None,
    ):
        self.env = env
        self.agent_controller = agent_controller
        self.bridge = bridge
        self.store = store
        self.bus = bus
        self.logging_agent = logging_agent

                                                             
        self.recorder: Optional[SessionRecorder] = None

                                                         
        self.session_ids: Dict[str, Any] = {}

        # paused state for edit gating and UI
        self.is_paused: bool = False
        # Human control wiring
        self.human_enabled: bool = False
        self.frozen: bool = False
        self.human_controller = HumanController(bus)

        # offscreen renderers: low-res for agent obs, high-res for UI
        self.agent_renderer = EnvRenderer(frame_size=84)
        self.ui_renderer = EnvRenderer(frame_size=256)

        # UI render throttle (frames per second)
        self.render_fps = 60.0
        self._last_render_time = 0.0

        bus.subscribe("HumanAsAgentToggled", self._on_human_toggle)
        bus.subscribe("AgentFreezeToggled", self._on_freeze_toggle)
        bus.subscribe("LivePause", lambda _: self.pause())
        bus.subscribe("LivePlay", lambda _: self.play())
        bus.subscribe("LiveStep", lambda _: self.step_once())
        bus.subscribe("CaptureStartRequested", lambda _: self.start_capture())
        bus.subscribe("CaptureStopRequested", lambda _: self.stop_capture())
        bus.subscribe("LiveSpeedChanged", self._on_speed_changed)
        # Cancel captures on environment resets or mode change away from live
        bus.subscribe("EnvResetRequested", lambda _: self.cancel_capture("env_reset"))
        bus.subscribe("ModeChanged", lambda p: self.cancel_capture("mode_change") if p.get("mode") != "live" else None)

                                                        
        bus.subscribe("AgentStepRequested", self._on_step_request)
        bus.subscribe("InjectRequested", self._on_inject_request)

                                                          
    # Legacy recording session management removed. LiveRunner is capture-only;
    # start_capture()/stop_capture() are the supported fast start/stop API.

    # ------------------------------------------------------------
    # Capture-only API (fast start/stop)
    # ------------------------------------------------------------
    def start_capture(self):
        # Already capturing → ACK but do not restart
        if self.recorder is not None:
            try:
                self.bus.publish("CaptureStarted", {"already_active": True})
            except Exception:
                pass
            return

        # Refuse if paused
        if getattr(self, "is_paused", False):
            try:
                self.bus.publish("CaptureStartRefused", {"reason": "paused"})
            except Exception:
                pass
            return

        from dataset.recorder import SessionRecorder

        self.recorder = SessionRecorder(self.store)
        try:
            self.bus.publish("CaptureStarted", {"session_id": self.recorder.session_id})
        except Exception:
            pass

    def stop_capture(self):
        # Nothing to stop → ACK idempotently
        if self.recorder is None:
            try:
                self.bus.publish("CaptureStopped", {"already_stopped": True})
            except Exception:
                pass
            return

        # stop_capture must be fast and return a PendingTake descriptor
        pending = None
        recorder_ref = None
        try:
            import time as _time
            start = _time.monotonic()
            recorder_ref = self.recorder
            pending = recorder_ref.stop_capture()
            dt = _time.monotonic() - start
            if dt > 0.1:
                try:
                    self.bus.publish("CaptureWarning", {"message": "stop_capture exceeded time budget", "dt": dt})
                except Exception:
                    pass
        except Exception:
            pending = None

        # Clear recorder quickly (background handoff will reference recorder_ref)
        self.recorder = None

        # Notify capture stopped (lightweight)
        try:
            self.bus.publish("CaptureStopped", {})
        except Exception:
            pass
        # Discard empty takes to avoid pipeline clogging
        try:
            if pending is not None and getattr(pending, "frame_count", 0) == 0:
                try:
                    import shutil
                    shutil.rmtree(pending.pending_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    self.bus.publish("CaptureDiscarded", {"reason": "empty_take"})
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Handoff to finalize: wait for writer completion off the UI thread,
        # then publish TakeCaptured. This prevents races between writer and
        # finalize while keeping stop_capture() fast.
        try:
            # recorder_ref was captured above before stop_capture() was invoked
            # and will be closed over by the handoff thread below.
            recorder_ref = recorder_ref

            # If we don't have a recorder_ref, try to use the pending take's
            # originating recorder by relying on closure semantics: when stop
            # returned, `self.recorder` has been cleared; instead, callers should
            # capture recorder before calling stop_capture. To be robust, if no
            # recorder_ref is available, emit TakeCaptured immediately.
            import threading

            def _handoff():
                try:
                    # Prefer waiting on the recorder instance if available
                    ok = False
                    if recorder_ref is not None:
                        try:
                            ok = recorder_ref.wait_writer_done(timeout=5.0)
                        except Exception:
                            ok = False
                    else:
                        # No recorder ref: assume writes completed
                        ok = True

                    if not ok:
                        try:
                            self.bus.publish("CaptureFailed", {"reason": "writer_timeout"})
                        except Exception:
                            pass
                        return

                    try:
                        if pending is not None:
                            self.bus.publish(
                                "TakeCaptured",
                                {
                                    "pending_dir": pending.pending_dir,
                                    "take_id": pending.take_id,
                                    "session_id": pending.session_id,
                                    "metadata": pending.metadata,
                                },
                            )
                    except Exception:
                        pass
                except Exception:
                    pass

            threading.Thread(target=_handoff, daemon=True).start()
        except Exception:
            pass

    def cancel_capture(self, reason: str):
        # Explicit cancellation path separate from normal stop
        if self.recorder is None:
            return

        pending = None
        try:
            pending = self.recorder.stop_capture(cancelled=True)
        except Exception:
            pending = None

        self.recorder = None

        try:
            self.bus.publish("CaptureCanceled", {"reason": reason})
        except Exception:
            pass

    def _on_human_toggle(self, payload: Dict[str, Any]):
        enabled = bool(payload.get("enabled", False))
        self.human_enabled = enabled
        # CRITICAL: forward state to HumanController
        try:
            if self.human_controller is not None:
                self.human_controller.set_enabled(enabled)
        except Exception:
            pass

    def _on_freeze_toggle(self, payload: Dict[str, Any]):
        self.frozen = bool(payload.get("enabled", False))

    def _on_speed_changed(self, payload: Dict[str, Any]):
        # placeholder: speed handling could scale dt or pacing
        return

                                                          
    def _on_inject_request(self, payload: Dict[str, Any]):


        snapshot = payload["snapshot"]

        normalized = self.bridge.normalize_snapshot(snapshot)
        # apply to the runtime
        self.bridge.apply_snapshot_to_env(self.env, normalized)

        # snapshot after apply
        new_snapshot = self.env.snapshot_state()

        # render a UI-quality frame for display
        try:
            frame = self.ui_renderer.render(new_snapshot)
        except Exception:
            frame = None

        packet = {
            "frame": frame,
            "snapshot": new_snapshot,
            "telemetry": self.env.build_telemetry() if hasattr(self.env, "build_telemetry") else {},
            "sim_time": getattr(self.env, "sim_time", 0.0),
            "step_index": getattr(self.env, "global_step", 0),
        }

        # keep legacy event for non-render consumers
        self.bus.publish("EnvStateUpdated", new_snapshot)
        self.bus.publish("EnvRenderPacket", packet)

                                                          
    def _build_interaction_event(
        self,
        *,
        step_index: int,
        obs: Dict[str, Any],
        env_state: EnvStateSnapshot,
        action,
        oracle_distance,
    ) -> Dict[str, Any]:


        if hasattr(obs, "get"):
            frame = obs.get("frame")
        else:
            frame = getattr(obs, "frame", None)

                                                                                   
        if hasattr(env_state, "get"):
            wall_state = env_state.get("wall_state", {})
            oracle_query = env_state.get("oracle_query", None)

            if oracle_query is None:
                oracle_query_a = None
                oracle_query_b = None
            else:
                oracle_query_a = oracle_query[0]
                oracle_query_b = oracle_query[1]

            agent_pos = list(env_state.get("agent_pos", []))
            pods_list = [list(p) for p in env_state.get("pods", [])]
            stomach = float(env_state.get("stomach"))
            phase = int(env_state.get("phase"))
            seq_idx = int(env_state.get("sequence_index", 0))
            wall_enabled = bool(wall_state.get("enabled", False))
            wall_blocking = bool(wall_state.get("blocking", False))
        else:
            oracle_query_a = None
            oracle_query_b = None
            agent_pos = list(env_state.agent_pos)
            pods_list = [ [float(p.pos[0]), float(p.pos[1]), float(p.spawn[0]), float(p.spawn[1])] if hasattr(p, 'pos') else list(p) for p in env_state.pods]
            stomach = float(env_state.stomach)
            phase = int(env_state.phase)
            seq_idx = int(env_state.sequence_index)
            wall_enabled = bool(env_state.wall.enabled)
            wall_blocking = bool(env_state.wall.blocking)

        return dict(
            step_index=step_index,
            frame=frame,
            stomach=float(stomach),
            oracle_distance=(float(oracle_distance) if oracle_distance is not None else float('nan')),
            agent_pos=agent_pos,
            action=list(action),
            phase=int(phase),
            food_positions=pods_list,
            wall_state=dict(
                enabled=wall_enabled,
                blocking=wall_blocking,
            ),
            sequence_index=int(seq_idx),
            oracle_query_a=oracle_query_a,
            oracle_query_b=oracle_query_b,
            mode="live",
        )

    def _record_interaction(self, event: Dict[str, Any]):

        # Capture is presence-based only: if no recorder is active, do nothing.
        if self.recorder is None:
            return
        # heartbeat for watchdog
        try:
            import time as _time
            self._last_capture_time = _time.monotonic()
        except Exception:
            pass

        oracle_query = None
        if event["oracle_query_a"] is not None and event["oracle_query_b"] is not None:
            oracle_query = [
                event["oracle_query_a"],
                event["oracle_query_b"],
            ]

        self.recorder.append(
            frame=event["frame"],
            stomach=event["stomach"],
            agent_pos=event["agent_pos"],
            action=event["action"],
            phase=event["phase"],
            food_positions=event["food_positions"],
            wall_enabled=event["wall_state"]["enabled"],
            wall_blocking=event["wall_state"]["blocking"],
            sequence_index=event["sequence_index"],
            oracle_query=oracle_query,
            oracle_distance=event["oracle_distance"],
        )

    def _on_step_request(self, payload: Dict[str, Any]):

        # Build low-resolution agent observation from snapshot
        env_state: EnvStateSnapshot = self.env.snapshot_state()
        try:
            agent_frame = self.agent_renderer.render(env_state)
        except Exception:
            agent_frame = None

        obs = build_observation(env_state, agent_frame)

        action, oracle_distance = self._choose_action(obs)

        if self.logging_agent is not None:
            self.logging_agent.act(
                observation=obs,
                env_state=env_state,
                session_ids=self.session_ids,
            )


        self.env.step(action)

        new_state: EnvStateSnapshot = self.env.snapshot_state()

        # publish both legacy state for loggers and a render packet for UI
        self.bus.publish("EnvStateUpdated", new_state)

        # Throttle UI rendering to render_fps
        now = time.perf_counter()
        dt_render = now - getattr(self, "_last_render_time", 0.0)
        if dt_render >= (1.0 / float(self.render_fps)):
            try:
                ui_frame = self.ui_renderer.render(new_state)
            except Exception:
                ui_frame = None

            packet = {
                "frame": ui_frame,
                "snapshot": new_state,
                "telemetry": self.env.build_telemetry() if hasattr(self.env, "build_telemetry") else {},
                "sim_time": getattr(self.env, "sim_time", 0.0),
                "step_index": getattr(self.env, "global_step", 0),
            }

            self.bus.publish("EnvRenderPacket", packet)
            self._last_render_time = now

        step_index = 0
        if self.recorder is not None:
            try:
                step_index = int(self.recorder.num_frames())
            except Exception:
                step_index = 0

        event = self._build_interaction_event(
            step_index=step_index,
            obs=obs,
            env_state=env_state,
            action=action,
            oracle_distance=oracle_distance,
        )
        self._record_interaction(event)

                                                          
    def run_forever(self):
        while True:
            self._on_step_request({"mode": "live"})

    def tick(self, dt: float = 0.0):
        # Do not advance the environment while paused; step_once may still be used
        if getattr(self, "is_paused", False):
            return

        try:
            self._on_step_request({"mode": "live"})
        except Exception:
            raise

    # ------------------------------------------------------------
    # Action selection / human arbitration
    # ------------------------------------------------------------

    def _choose_action(self, obs):
        # frozen -> no action
        if self.frozen:
            import numpy as _np

            return _np.array([0.0, 0.0], dtype=_np.float32), 0.0

        if self.human_enabled and self.human_controller is not None:
            return self.human_controller.run_step(obs)

        return self.agent_controller.run_step(obs)

    def get_last_capture_time(self):
        return getattr(self, "_last_capture_time", None)

    # Control helpers for Play/Pause/Step
    def play(self):
        # start/resume play
        self.playing = True
        self.is_paused = False
        # IMPORTANT: unfreeze agent on resume
        self.frozen = False
        try:
            self.bus.publish("AgentFreezeToggled", {"enabled": False})
        except Exception:
            pass

        try:
            self.bus.publish("LivePausedStateChanged", {"paused": False})
        except Exception:
            pass
        # Re-arm human controller if selected
        try:
            if self.human_enabled and self.human_controller is not None:
                self.human_controller.set_enabled(True)
        except Exception:
            pass
        # Request UI focus for the live env view
        try:
            self.bus.publish("RequestFocusLiveEnv", {})
        except Exception:
            pass

    def pause(self):
        # pause stepping
        self.playing = False
        self.is_paused = True
        # Freeze agent while paused
        self.frozen = True
        try:
            self.bus.publish("AgentFreezeToggled", {"enabled": True})
        except Exception:
            pass

        try:
            self.bus.publish("LivePausedStateChanged", {"paused": True})
        except Exception:
            pass

    def step_once(self):
        # single step
        try:
            self._on_step_request({"mode": "manual"})
        except Exception:
            raise
