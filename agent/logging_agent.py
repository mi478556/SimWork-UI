# logging_agent.py

from typing import Optional, Dict, Any, Tuple

from agent.policy_base import Observation, AgentPolicy
from agent.execution_context import AgentExecutionContext
from engine.snapshot_state import EnvStateSnapshot


def validate_observation_contract(obs: Observation):


    if obs.frame is None:
        raise RuntimeError("Observation missing frame")

    if obs.stomach is None:
        raise RuntimeError("Observation missing stomach value")

                                                          
class LoggingAgent:


    REQUIRED_PROVENANCE_FIELDS = {
        "canonicalized",
        "branch_id",
        "branch_parent_id",
        "branch_depth",
        "origin_mode",                                       
        "origin_clip_step",
    }

    def __init__(
        self,
        base_agent: AgentPolicy,
        recorder,
        tools: Dict[str, Any],
        *,
        policy_introspector=None,                                 
        oracle_enabled: bool = True,
        oracle_budget_per_episode: Optional[int] = None,
    ):
        self.base_agent = base_agent
        self.recorder = recorder
        self.tools = tools

        self.policy_introspector = policy_introspector

        self.oracle_enabled = oracle_enabled
        self.oracle_budget_per_episode = oracle_budget_per_episode
        self.oracle_used_this_episode = 0

        self.context: AgentExecutionContext = AgentExecutionContext.LIVE

                                                              
    def on_session_start(self, session_meta=None):
        self.recorder.record_event({
            "type": "SessionStart",
            "meta": session_meta or {},
        })

    def on_session_close(self):
        self.recorder.record_event({"type": "SessionClosed"})
        self.recorder.flush()

    def on_clip_start(self, clip_meta=None):
        self.recorder.record_event({
            "type": "ClipStart",
            "meta": clip_meta or {},
        })

    def on_clip_close(self):
        self.recorder.record_event({"type": "ClipClosed"})

    def on_episode_reset(self):
        self.oracle_used_this_episode = 0
        self.recorder.record_event({"type": "LifeResetEvent"})

    def on_phase_transition(self, new_phase: int):
        self.recorder.record_event({
            "type": "PhaseTransitionEvent",
            "phase": new_phase,
        })

    def on_agent_died(self):
        self.recorder.record_event({"type": "AgentDiedEvent"})

                                                              
    def _validate_snapshot_provenance(self, env_state: EnvStateSnapshot):

        prov = env_state.get("provenance_meta", {})

        missing = self.REQUIRED_PROVENANCE_FIELDS - set(prov.keys())
        if missing:
            raise RuntimeError(
                f"Snapshot provenance missing required fields: {missing}"
            )

        if not prov["canonicalized"]:
            raise RuntimeError(
                "LoggingAgent refused non-canonical snapshot."
            )

                                            
        if "step_index" not in env_state:
            raise RuntimeError("EnvState missing step_index")

                                                              
    def act(
        self,
        observation: Observation,
        env_state: EnvStateSnapshot,
        session_ids: Dict[str, str],
        *,
        no_action: bool,
        policy_mode: str,
    ) -> Tuple[Optional[list], Optional[float]]:

        validate_observation_contract(observation)
        self._validate_snapshot_provenance(env_state)

                                           
        if self.context == AgentExecutionContext.PLAYBACK:
            raise RuntimeError(
                "LoggingAgent refused execution in PLAYBACK mode."
            )

        step_index = env_state["step_index"]

                                                      
        if no_action:
            event = {
                "type": "NoActionFrame",

                "policy_mode": policy_mode,                    

                "session_id": session_ids.get("session_id"),
                "clip_id": session_ids.get("clip_id"),
                "step_index": step_index,

                "execution_context": self.context.value,

                "observation": {
                    "frame": observation.frame,
                    "stomach": observation.stomach,
                    "oracle_distance": None,
                },

                "action": None,

                "oracle_context": None,
                "oracle_query_a": None,
                "oracle_query_b": None,

                "env_state": env_state,
            }

            self.recorder.record_step(event)
            return None, None

                                                      
        action, oracle_query = self.base_agent.act(observation, self.tools)
        oracle_distance = None
        qa = qb = None

                                                      
        oracle_context = None

        if oracle_query is not None:

            if not self.oracle_enabled:
                raise RuntimeError("Oracle query attempted but oracle disabled")

            if (
                self.oracle_budget_per_episode is not None
                and self.oracle_used_this_episode >= self.oracle_budget_per_episode
            ):
                raise RuntimeError("Oracle query budget exceeded")

            qa, qb = oracle_query
            oracle_distance = self.tools["distance"].query(qa, qb)
            self.oracle_used_this_episode += 1

            oracle_context = {
                "execution_mode": self.context.value,
                "episode_query_index": self.oracle_used_this_episode,
            }

                                                      
        event = {
            "type": "ActionFrame",

            "policy_mode": policy_mode,                             

            "session_id": session_ids.get("session_id"),
            "clip_id": session_ids.get("clip_id"),
            "step_index": step_index,

            "execution_context": self.context.value,

            "observation": {
                "frame": observation.frame,
                "stomach": observation.stomach,
                "oracle_distance": oracle_distance,
            },

            "action": action,

            "oracle_query_a": qa,
            "oracle_query_b": qb,
            "oracle_distance": oracle_distance,
            "oracle_context": oracle_context,

            "oracle_budget_remaining": (
                None if self.oracle_budget_per_episode is None
                else self.oracle_budget_per_episode - self.oracle_used_this_episode
            ),

            "env_state": env_state,
        }

        self.recorder.record_step(event)

                                                      
        if self.policy_introspector is not None:
            self.policy_introspector.log_debug_signal({
                "step_index": step_index,
                "execution_mode": self.context.value,
                "policy_mode": policy_mode,
            })

        return action, oracle_distance
