from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from agent.policy_base import AgentPolicy
from agent.execution_context import AgentExecutionContext


class StarterWanderPolicy(AgentPolicy):
    """
    Simple movement policy for smoke-testing the agent loop.
    Produces bounded 2D actions with mild persistence and random drift.
    """

    def __init__(
        self,
        *,
        speed: float = 0.7,
        turn_every: int = 18,
        jitter: float = 0.2,
        seed: int = 7,
    ):
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.turn_every = max(1, int(turn_every))
        self.jitter = float(max(0.0, jitter))
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.direction = self._sample_unit_vec()

    def _sample_unit_vec(self) -> np.ndarray:
        v = self.rng.normal(size=2).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return v / n

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return self._sample_unit_vec()
        return (v / n).astype(np.float32)

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        self.step_count += 1

        if self.step_count % self.turn_every == 0:
            self.direction = self._sample_unit_vec()
        else:
            noise = self.rng.normal(scale=self.jitter, size=2).astype(np.float32)
            self.direction = self._normalize(self.direction + noise)

        action = (self.direction * self.speed).astype(np.float32)
        return action, None


class NoOpPolicy(AgentPolicy):
    """Always emits zero action. Useful as a baseline."""

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        return np.array([0.0, 0.0], dtype=np.float32), None


class OscillatingPolicy(AgentPolicy):
    """
    Deterministic cyclical movement for repeatable behavior checks.
    """

    def __init__(self, *, speed: float = 0.8, angular_step: float = 0.2):
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.angular_step = float(max(0.001, angular_step))
        self.t = 0.0

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        self.t += self.angular_step
        action = np.array(
            [np.cos(self.t), np.sin(self.t)],
            dtype=np.float32,
        ) * self.speed
        return action, None
