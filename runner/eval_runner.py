# eval_runner.py

from __future__ import annotations

from typing import Optional, Callable, Dict, Any

from viewer.event_bus import EventBus
from env.injection_bridge import InjectionBridge
from env.environment_runtime import EnvironmentRuntime
from env.snapshot_state import EnvStateSnapshot
from agent.agent_controller import AgentController
from agent.logging_agent import LoggingAgent
from env.env_renderer import EnvRenderer
from runner.observation_builder import build_observation

from dataset.session_store import SessionStore
from dataset.recorder import SessionRecorder
from dataset.session_types import ProvenanceMeta


class EvalRunner:


    def __init__(
        self,
        env: EnvironmentRuntime,
        bridge: InjectionBridge,
        agent_controller: AgentController,
        store: SessionStore,
        bus: EventBus,
        logging_agent: Optional[LoggingAgent] = None,
    ):
        self.env = env
        self.bridge = bridge
        self.agent_controller = agent_controller
        self.store = store
        self.bus = bus
        self.logging_agent = logging_agent

        self.recorder: Optional[SessionRecorder] = None
        # offscreen renderer for eval UI packets
        self.renderer = EnvRenderer(frame_size=84)

                                                                  
    def run_eval_from_state(
        self,
        *,
        snapshot: EnvStateSnapshot,
        num_steps: int = 50,
        agent_id: Optional[str] = None,

                              
        branch_parent_session: Optional[str] = None,
        branch_parent_clip: Optional[str] = None,
        branch_parent_step: Optional[int] = None,
        branch_depth: int = 0,                 

                                                                       
        parent_provenance: Optional[Dict[str, Any]] = None,

        probe_config: Optional[Dict[str, Any]] = None,

        on_step: Optional[Callable[[int, EnvStateSnapshot], None]] = None,
        on_finish: Optional[Callable[[EnvStateSnapshot, str], None]] = None,
        on_interrupt: Optional[Callable[[str], None]] = None,
    ):


        parent_depth = int(branch_depth) if branch_depth is not None else 0

        if parent_provenance is not None:
            branch_parent_session = parent_provenance.get(
                "session_id", branch_parent_session
            )
            branch_parent_clip = parent_provenance.get(
                "clip_id", branch_parent_clip
            )
            branch_parent_step = parent_provenance.get(
                "step_index", branch_parent_step
            )
            parent_depth = int(
                parent_provenance.get("branch_depth", parent_depth)
            )

        eval_depth = parent_depth + 1

                                                              
        normalized = self.bridge.normalize_snapshot(snapshot)
        self.bridge.apply_snapshot(normalized)

        # publish a render packet for UI consumption (initial state)
        try:
            frame = self.renderer.render(normalized)
        except Exception:
            frame = None

        packet = {
            "frame": frame,
            "snapshot": normalized,
            "telemetry": self.env.build_telemetry() if hasattr(self.env, "build_telemetry") else {},
            "sim_time": getattr(self.env, "sim_time", 0.0),
            "step_index": getattr(self.env, "global_step", 0),
        }

        # keep legacy event for non-render consumers
        self.bus.publish("EnvStateUpdated", normalized)
        self.bus.publish("EnvRenderPacket", packet)

                                                              
        self.recorder = SessionRecorder(self.store)

        self.recorder.provenance = ProvenanceMeta(
            origin="eval",

            branch_parent_id=branch_parent_session,
            branch_depth=eval_depth,

            source_session_id=branch_parent_session,
            source_clip_id=branch_parent_clip,
            source_step_index=branch_parent_step,

            agent_id=agent_id,
            notes=str(probe_config) if probe_config is not None else None,
        )

        eval_run_id = self.recorder.session_id

                                                              
        self.bus.publish(
            "EvalRunStarted",
            {
                "eval_run_id": eval_run_id,
                "branch_parent_session": branch_parent_session,
                "branch_parent_clip": branch_parent_clip,
                "branch_parent_step": branch_parent_step,
                "branch_depth": eval_depth,
            },
        )

        termination_reason = "max_steps"
        final_state: EnvStateSnapshot = normalized

                                                               
        try:
            for step_idx in range(num_steps):

                env_state: EnvStateSnapshot = self.env.snapshot_state()
                try:
                    frame = self.renderer.render(env_state)
                except Exception:
                    frame = None

                obs = build_observation(env_state, frame)

                action, oracle_distance = self.agent_controller.run_step(obs)

                if self.logging_agent is not None:
                    self.logging_agent.act(
                        observation=obs,
                        env_state=env_state,
                        session_ids={"mode": "eval", "eval_run_id": eval_run_id},
                    )

                self.env.step(action)

                new_state: EnvStateSnapshot = self.env.snapshot_state()
                final_state = new_state

                self._append_eval_step(
                    step_index=step_idx,
                    obs=obs,
                    env_state=env_state,
                    action=action,
                    oracle_distance=oracle_distance,
                )

                self.bus.publish(
                    "EvalStepCompleted",
                    {
                        "eval_run_id": eval_run_id,
                        "step_idx": step_idx,
                        "state": new_state,
                        "branch_parent_session": branch_parent_session,
                        "branch_parent_clip": branch_parent_clip,
                        "branch_parent_step": branch_parent_step,
                        "branch_depth": eval_depth,
                    },
                )

                # publish UI render packet for this eval step
                try:
                    frame = self.renderer.render(new_state)
                except Exception:
                    frame = None

                packet = {
                    "frame": frame,
                    "snapshot": new_state,
                    "telemetry": self.env.build_telemetry() if hasattr(self.env, "build_telemetry") else {},
                    "sim_time": getattr(self.env, "sim_time", 0.0),
                    "step_index": getattr(self.env, "global_step", 0),
                }

                self.bus.publish("EnvRenderPacket", packet)

                if on_step is not None:
                    on_step(step_idx, new_state)

                if new_state.get("death", False):
                    termination_reason = "death"
                    break

        except Exception as e:
            termination_reason = f"exception:{type(e).__name__}"

            if on_interrupt is not None:
                on_interrupt(termination_reason)

            self.bus.publish(
                "EvalInterrupted",
                {
                    "eval_run_id": eval_run_id,
                    "reason": termination_reason,
                    "exception": repr(e),
                },
            )

        finally:
            if self.recorder is not None:
                session = self.recorder.finalize()
                self.recorder = None

                self.bus.publish(
                    "EvalSessionFinished",
                    {
                        "eval_run_id": eval_run_id,
                        "session_id": session.session_id,
                        "termination_reason": termination_reason,
                    },
                )

                                                              
        self.bus.publish(
            "EvalFinished",
            {
                "eval_run_id": eval_run_id,
                "final_state": final_state,
                "termination_reason": termination_reason,
            },
        )

        if on_finish is not None:
            on_finish(final_state, termination_reason)

                                                                  
    def _append_eval_step(
        self,
        *,
        step_index: int,
        obs: Dict[str, Any],
        env_state: EnvStateSnapshot,
        action,
        oracle_distance,
    ):
        if self.recorder is None:
            return

                                                           
        if hasattr(obs, "get"):
            frame = obs.get("frame")
        else:
            frame = getattr(obs, "frame", None)

        wall_state = env_state.get("wall_state", {})
        oracle_query = env_state.get("oracle_query", None)

        if oracle_query is None:
            oracle_query_a = None
            oracle_query_b = None
            oracle_query_packed = None
        else:
            oracle_query_a = oracle_query[0]
            oracle_query_b = oracle_query[1]
            oracle_query_packed = [oracle_query_a, oracle_query_b]

        self.recorder.append(
            frame=frame,
            stomach=float(env_state["stomach"]),
            agent_pos=list(env_state["agent_pos"]),
            action=list(action),
            phase=int(env_state["phase"]),
            food_positions=[list(p) for p in env_state.get("pods", [])],
            wall_enabled=bool(wall_state.get("enabled", False)),
            wall_blocking=bool(wall_state.get("blocking", False)),
            sequence_index=int(env_state.get("sequence_index", 0)),
            oracle_query=oracle_query_packed,
            oracle_distance=float(oracle_distance),
        )
