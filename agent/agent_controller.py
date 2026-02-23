# agent_controller.py

from typing import Dict, Any, Optional, Tuple

from agent.policy_base import Observation
from agent.execution_context import AgentExecutionContext
from agent.logging_agent import LoggingAgent       


class AgentController:


    def __init__(
        self,
        policy: Optional[Any] = None,
        policies: Optional[Dict[str, Any]] = None,
        active_policy_name: Optional[str] = None,
        logging_agent: Optional[LoggingAgent] = None,
        tools: Optional[Dict[str, Any]] = None,
    ):
        self.policy = policy
        self.policies: Dict[str, Any] = {}
        self.active_policy_name: Optional[str] = None
        self.logging_agent = logging_agent
        self.tools = tools or {}

        self.paused: bool = False
        self.policy_frozen: bool = False

                                            
        self.execution_enabled: bool = True

                                      
        self.context: AgentExecutionContext = AgentExecutionContext.LIVE

        if policy is not None:
            self.register_policy("default", policy, set_active=True)

        if policies:
            for name, policy_obj in policies.items():
                self.register_policy(name, policy_obj, set_active=False)

        if active_policy_name is not None:
            self.set_active_policy(active_policy_name)

    def register_policy(self, name: str, policy: Any, *, set_active: bool = False):
        if not name:
            raise ValueError("Policy name must be a non-empty string.")
        self.policies[name] = policy
        if set_active or self.active_policy_name is None:
            self.active_policy_name = name
            self.policy = policy

    def list_policies(self):
        return list(self.policies.keys())

    def set_active_policy(self, name: str) -> bool:
        policy = self.policies.get(name)
        if policy is None:
            return False
        self.active_policy_name = name
        self.policy = policy
        return True

                                                              
    def set_execution_enabled(self, flag: bool):


        self.execution_enabled = flag
        if self.logging_agent is not None:
            self.logging_agent.set_context(self.context)

    def set_context(self, context: AgentExecutionContext):


        self.context = context
        if self.logging_agent is not None:
            self.logging_agent.set_context(context)

    def set_policy_frozen(self, frozen: bool):


        self.policy_frozen = frozen

    def freeze_policy(self):

        self.set_policy_frozen(True)

    def unfreeze_policy(self):

        self.set_policy_frozen(False)

                                                              
    def run_step_logged(
        self,
        observation: Observation,
        env_state,
        session_ids: Dict[str, str],
    ) -> Optional[Tuple[list, Optional[float]]]:

                                                     
        if not self.execution_enabled:
            raise RuntimeError(
                "AgentController refused execution — execution disabled "
                "(this should occur only in PLAYBACK mode)."
            )

                                    
        if self.paused:
            return self.logging_agent.act(
                observation=observation,
                env_state=env_state,
                session_ids=session_ids,
                no_action=True,
                policy_mode="paused_no_action",
            )

                                            
        if self.policy_frozen:
            return self.logging_agent.act(
                observation=observation,
                env_state=env_state,
                session_ids=session_ids,
                no_action=False,
                policy_mode="frozen_manual_step",
            )

                                      
        return self.logging_agent.act(
            observation=observation,
            env_state=env_state,
            session_ids=session_ids,
            no_action=False,
            policy_mode="live",
        )

    def run_step(self, observation: Observation):


        if self.policy is not None:
            try:
                action, oracle_query = self.policy.act(observation, self.tools)
            except Exception:
                return [0.0, 0.0], None

            if oracle_query is None:
                return action, None

            qa, qb = oracle_query
            if "distance" in self.tools:
                try:
                    dist = self.tools["distance"].query(qa, qb)
                except Exception:
                    dist = None
                return action, dist
            return action, None

        if self.logging_agent is not None and hasattr(self.logging_agent, "base_agent"):
            try:
                action, oracle_query = self.logging_agent.base_agent.act(observation, self.tools)
            except Exception:
                return [0.0, 0.0], None

            if oracle_query is None:
                return action, None

            qa, qb = oracle_query
            if "distance" in self.tools:
                try:
                    dist = self.tools["distance"].query(qa, qb)
                except Exception:
                    dist = None
                return action, dist
            return action, None

                      
        return [0.0, 0.0], None

                                                              
    def step_manual_logged(
        self,
        observation: Observation,
        env_state,
        session_ids: Dict[str, str],
    ):


        if not self.execution_enabled:
            raise RuntimeError(
                "Manual probe attempted while execution disabled."
            )

        return self.logging_agent.act(
            observation=observation,
            env_state=env_state,
            session_ids=session_ids,
            no_action=False,
            policy_mode="frozen_manual_step" if self.policy_frozen else "live",
        )

                                                              
    def set_paused(self, flag: bool):
        self.paused = flag

    def toggle_pause(self):
        self.paused = not self.paused
