from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from viewer.event_bus import EventBus

class HumanController:
    """
    Mirrors AgentController's run_step API:
      run_step(obs) -> (action, oracle_distance)
    Stores the latest intent vector from HumanInputUpdated.
    """

    def __init__(
        self,
        bus: EventBus,
        action_scale: float = 1.0,
        clamp: float = 1.0,
    ):
        self.bus = bus
        self.action_scale = float(action_scale)
        self.clamp = float(clamp)

        # cache last human action as a numpy vector
        self._last_action = np.zeros(2, dtype=np.float32)

        # subscribe to both legacy and preferred event names
        bus.subscribe("HumanInputUpdated", self._on_input)
        bus.subscribe("HumanInputEvent", self._on_input)

    def _on_input(self, payload: Dict[str, Any]):
        # Accept either {'action': np.array([...])} or {'dx':..., 'dy':...}
        try:
            if "action" in payload:
                a = payload.get("action")
                self._last_action = np.array(a, dtype=np.float32)
                return
        except Exception:
            pass

        dx = float(payload.get("dx", 0.0))
        dy = float(payload.get("dy", 0.0))
        self._last_action = np.array([dx, dy], dtype=np.float32)

    def run_step(self, obs) -> Tuple[np.ndarray, float]:
        # Return a copy of the cached last action (do not poll hardware here)
        a = self._last_action.copy() * self.action_scale
        if self.clamp > 0:
            a = np.clip(a, -self.clamp, self.clamp)
        return a, 0.0

    def set_enabled(self, enabled: bool):
        # Enable/disable human control; reset cached input when enabling
        try:
            self.enabled = bool(enabled)
        except Exception:
            self.enabled = False
        if self.enabled:
            # reset last action so stale inputs are not applied
            self._last_action = np.zeros(2, dtype=np.float32)
