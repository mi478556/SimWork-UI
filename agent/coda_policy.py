from __future__ import annotations

import os
import time
import copy
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agent.execution_context import AgentExecutionContext
from agent.coda_causal_model import CodaCausalModel, CodaCausalTransition, build_causal_config
from agent.coda_executive_model import CodaExecutiveInput, CodaExecutiveModel, build_executive_config
from agent.coda_forward_model import CodaDynamicsConfig, CodaForwardModelComponent, CodaTokenSchema
from agent.policy_base import AgentPolicy

ACTION_MODE_WANDER = "wander"
ACTION_MODE_RL = "rl"
ACTION_MODE_WALL_CONDITIONAL = "wall_conditional"
ACTION_MODE_HUMAN = "human"
ACTION_MODE_EXECUTIVE = "executive"
VALID_ACTION_MODES = {ACTION_MODE_WANDER, ACTION_MODE_RL, ACTION_MODE_WALL_CONDITIONAL, ACTION_MODE_HUMAN, ACTION_MODE_EXECUTIVE}


class CODAPolicy(AgentPolicy):
    """
    CODA: Causal Online Discovery Agent.
    Current bootstrap behavior is a wandering controller so the agent can be
    selected, run, saved, and loaded while the core design evolves.
    """

    def __init__(
        self,
        *,
        speed: float = 0.72,
        turn_every: int = 22,
        jitter: float = 0.15,
        seed: int = 101,
        enable_forward_model: bool = True,
        train_forward_model: bool = True,
        forward_buffer_size: int = 4096,
        forward_batch_size: int = 16,
        forward_warmup: int = 128,
        forward_update_every: int = 8,
        forward_lr: float = 3e-4,
        forward_num_condition_slots: int = 0,
        fast_mode: bool = True,
        fast_train_stride: int = 1,
        action_mode: str = ACTION_MODE_WANDER,
        debug_dump_every: int = 200,
        debug_logging_enabled: bool = False,
        debug_dir: str = "data/coda_debug",
    ):
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.turn_every = max(1, int(turn_every))
        self.jitter = float(max(0.0, jitter))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.step_count = 0
        self.direction = self._sample_unit_vec()

        self.enable_forward_model = bool(enable_forward_model)
        self.train_forward_model = bool(train_forward_model)
        self._base_forward_batch_size = int(max(1, forward_batch_size))
        self._base_forward_warmup = int(max(1, forward_warmup))
        self._base_forward_update_every = int(max(1, forward_update_every))
        self.fast_mode = bool(fast_mode)
        self.fast_train_stride = int(max(1, fast_train_stride))
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in VALID_ACTION_MODES:
            self.action_mode = ACTION_MODE_WANDER
        self.debug_dump_every = int(max(0, debug_dump_every))
        self.debug_dir = os.path.abspath(str(debug_dir))
        self.debug_logging_enabled = bool(debug_logging_enabled)
        self.debug_view_enabled = False
        self._debug_force_dump_once = False
        self.last_debug_dump_path: Optional[str] = None
        self.last_debug_dump_error: Optional[str] = None

        token_schema = CodaTokenSchema(
            num_agent_slots=1,
            num_pod_slots=2,
            num_wall_slots=14,
            num_condition_slots=int(max(0, forward_num_condition_slots)),
            num_slack_slots=2,
        )
        dynamics_cfg = CodaDynamicsConfig(
            buffer_capacity=int(max(32, forward_buffer_size)),
            batch_size=self._base_forward_batch_size,
            warmup_transitions=self._base_forward_warmup,
            update_every=self._base_forward_update_every,
            lr=float(max(1e-6, forward_lr)),
        )
        self.forward_component = CodaForwardModelComponent(
            schema=token_schema,
            dynamics_config=dynamics_cfg,
            action_dim=2,
            seed=self.seed + 31,
            training_enabled=self.train_forward_model,
            device="cpu",
        )
        self._apply_fast_mode_profile()

        self.prev_tokens: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None
        self.prev_phase: Optional[int] = None
        self.prev_stomach: Optional[float] = None
        self.prev_agent_pos: Optional[np.ndarray] = None
        self.prev_nearest_pod_dist: Optional[float] = None
        self.prev_far_pod_dist: Optional[float] = None
        self.prev_region_active: Optional[bool] = None
        self.prev_region_signature: Optional[float] = None
        self.prev_region_proximity: Optional[float] = None
        self.prev_region_dwell: int = 0
        self.prev_region_entry_trace: float = 0.0
        self.prev_wall_blocking: Optional[bool] = None
        self.pending_predicted_tokens: Optional[np.ndarray] = None

        self.debug_history: deque = deque(maxlen=512)
        self.last_debug_info: Dict[str, Any] = {}
        self.latest_debug_packet: Dict[str, Any] = {}
        self._last_action_source: str = "wander_forced"
        self._last_rl_diag: Dict[str, Any] = {
            "found": False,
            "class": "",
            "total_updates": float("nan"),
            "learning_enabled": float("nan"),
            "inference_path": "",
        }
        self.causal_model = CodaCausalModel(build_causal_config())
        self.executive_model = CodaExecutiveModel(build_executive_config("pain_aware_comfort_controller"))
        self._causal_history: list = []
        self._last_executive_diag: Dict[str, Any] = {}
        self._last_causal_interface_signals: Dict[str, float] = {}

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

    def _obs_to_dict(self, observation: Any) -> Dict[str, Any]:
        if isinstance(observation, dict):
            return observation
        return {}

    def _detect_wall_visible_from_obs(self, obs: Dict[str, Any]) -> bool:
        # Prefer authoritative wall state from observation when present.
        wall = obs.get("wall", {}) if isinstance(obs, dict) else {}
        try:
            if hasattr(wall, "get") and ("blocking" in wall):
                return bool(wall.get("blocking", False))
            if hasattr(wall, "blocking"):
                return bool(getattr(wall, "blocking", False))
        except Exception:
            pass

        frame = obs.get("frame", None)
        if frame is None:
            return False
        arr = np.asarray(frame)
        if arr.size == 0:
            return False
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, repeats=3, axis=2)
        elif arr.ndim != 3 or arr.shape[2] < 3:
            return False
        rgb = arr[:, :, :3].astype(np.float32, copy=False)
        if float(np.max(rgb)) > 1.5:
            rgb = rgb / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        wall_mask = (r > 0.65) & (g > 0.65) & (b > 0.65)
        return bool(np.sum(wall_mask) >= 12)

    def _wander_action(self) -> np.ndarray:
        if self.step_count % self.turn_every == 0:
            self.direction = self._sample_unit_vec()
        else:
            noise = self.rng.normal(scale=self.jitter, size=2).astype(np.float32)
            self.direction = self._normalize(self.direction + noise)
        return (self.direction * self.speed).astype(np.float32)

    def _condition_scalar_signals(self, prev_tokens: np.ndarray, curr_tokens: np.ndarray) -> Dict[str, float]:
        schema = self.forward_component.schema
        if int(getattr(schema, "num_condition_slots", 0)) < 2:
            return {
                "stomach_level": 0.0,
                "stomach_delta": 0.0,
                "stomach_stress": 0.0,
                "pain_level": 0.0,
                "pain_delta": 0.0,
                "pain_from_stomach": 0.0,
            }
        go = 1 + schema.num_types
        stomach_slot = int(schema.condition_start)
        pain_slot = stomach_slot + 1
        prev_arr = np.asarray(prev_tokens, dtype=np.float32)
        curr_arr = np.asarray(curr_tokens, dtype=np.float32)
        stomach_level = float(curr_arr[stomach_slot, go + 0])
        stomach_delta = float(curr_arr[stomach_slot, go + 1] - prev_arr[stomach_slot, go + 1])
        stomach_low = float(curr_arr[stomach_slot, go + 2])
        stomach_high = float(curr_arr[stomach_slot, go + 3])
        stomach_stress = float(max(stomach_low, stomach_high))
        pain_level = float(curr_arr[pain_slot, go + 0])
        pain_delta = float(curr_arr[pain_slot, go + 0] - prev_arr[pain_slot, go + 0])
        pain_from_stomach = float(curr_arr[pain_slot, go + 1])
        return {
            "stomach_level": float(np.clip(stomach_level, 0.0, 1.0)),
            "stomach_delta": float(np.clip(stomach_delta, -1.0, 1.0)),
            "stomach_stress": float(np.clip(stomach_stress, 0.0, 1.0)),
            "pain_level": float(np.clip(pain_level, 0.0, 1.0)),
            "pain_delta": float(np.clip(pain_delta, -1.0, 1.0)),
            "pain_from_stomach": float(np.clip(pain_from_stomach, 0.0, 1.0)),
        }

    def _nearest_pod_info(self, obs: Dict[str, Any], *, far_only: bool = False) -> Tuple[Optional[np.ndarray], float]:
        agent_pos = np.asarray(obs.get("agent_pos", [-0.5, 0.0]), dtype=np.float32).reshape(-1)
        if agent_pos.shape[0] < 2:
            agent_xy = np.array([-0.5, 0.0], dtype=np.float32)
        else:
            agent_xy = agent_pos[:2].astype(np.float32, copy=False)
        best_pos: Optional[np.ndarray] = None
        best_dist = float("inf")
        for pod in obs.get("pods", []) or []:
            try:
                active = bool(pod.get("active", False))
                if not active:
                    continue
                pos = np.asarray(pod.get("pos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
                if pos.shape[0] < 2:
                    continue
                pod_xy = pos[:2].astype(np.float32, copy=False)
                if far_only:
                    if agent_xy[0] < 0.0 and float(pod_xy[0]) <= 0.0:
                        continue
                    if agent_xy[0] >= 0.0 and float(pod_xy[0]) >= 0.0:
                        continue
                dist = float(np.linalg.norm(pod_xy - agent_xy))
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pod_xy
            except Exception:
                continue
        return best_pos, best_dist

    def _anonymous_region_state(self, obs: Dict[str, Any]) -> Dict[str, float]:
        agent_pos = np.asarray(obs.get("agent_pos", [-0.5, 0.0]), dtype=np.float32).reshape(-1)
        agent_xy = agent_pos[:2].astype(np.float32, copy=False) if agent_pos.shape[0] >= 2 else np.array([-0.5, 0.0], dtype=np.float32)

        rooms = list(obs.get("rooms", []) or [])
        if (not rooms) and obs.get("frame", None) is not None and bool(obs.get("wall", {}).get("blocking", False)):
            arr = np.asarray(obs.get("frame"), dtype=np.float32)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, repeats=3, axis=2)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                rgb = arr[:, :, :3]
                if float(np.max(rgb)) > 1.5:
                    rgb = rgb / 255.0
                white = (rgb[:, :, 0] > 0.65) & (rgb[:, :, 1] > 0.65) & (rgb[:, :, 2] > 0.65)
                ys, xs = np.nonzero(white)
                if xs.size > 0:
                    wall_x = int(np.median(xs))
                    h, w = white.shape
                    agent_left = bool(float(agent_xy[0]) < 0.0)
                    if agent_left:
                        side_mask = white[:, : max(0, wall_x - 3)]
                    else:
                        side_mask = white[:, min(w, wall_x + 3) :]
                    y_hits = np.where(np.sum(side_mask, axis=1) >= 3)[0].tolist()
                    y_levels: list[int] = []
                    for y in y_hits:
                        if not y_levels or abs(y - y_levels[-1]) > 3:
                            y_levels.append(int(y))
                    if len(y_levels) >= 2:
                        x0 = -1.0 if agent_left else 0.0
                        x1 = (float(wall_x) / max(1.0, float(w))) * 2.0 - 1.0 if agent_left else 1.0
                        for y0, y1 in zip(y_levels[:-1], y_levels[1:]):
                            if (y1 - y0) < 4:
                                continue
                            wy0 = (float(y0) / max(1.0, float(h))) * 2.0 - 1.0
                            wy1 = (float(y1) / max(1.0, float(h))) * 2.0 - 1.0
                            rx = min(x0, x1)
                            rw = abs(x1 - x0)
                            ry = min(wy0, wy1)
                            rh = abs(wy1 - wy0)
                            rooms.append(np.asarray([rx, ry, rw, rh], dtype=np.float32))

        inside_sig: Optional[float] = None
        best_dist = float("inf")
        best_sig = 0.0
        for room in rooms:
            try:
                rr = np.asarray(room, dtype=np.float32).reshape(-1)
                if rr.shape[0] < 4:
                    continue
                rx, ry, rw, rh = [float(v) for v in rr[:4]]
                cx = rx + 0.5 * rw
                cy = ry + 0.5 * rh
                dist = float(np.linalg.norm(np.asarray([cx, cy], dtype=np.float32) - agent_xy))
                sig = float(np.clip(0.5 + 0.5 * (cy / 0.8), 0.0, 1.0))
                if (rx <= float(agent_xy[0]) <= rx + rw) and (ry <= float(agent_xy[1]) <= ry + rh):
                    inside_sig = sig
                if dist < best_dist:
                    best_dist = dist
                    best_sig = sig
            except Exception:
                continue
        occupancy = 1.0 if inside_sig is not None else 0.0
        signature = float(inside_sig if inside_sig is not None else best_sig)
        proximity = float(np.exp(-6.0 * best_dist)) if np.isfinite(best_dist) else 0.0
        return {
            "occupancy": float(occupancy),
            "signature": float(np.clip(signature, 0.0, 1.0)),
            "proximity": float(np.clip(proximity, 0.0, 1.0)),
        }

    def _sequence_state_proxy(self, obs: Dict[str, Any]) -> float:
        phase = int(obs.get("phase", self.prev_phase if self.prev_phase is not None else 1))
        if phase < 3:
            return 0.0
        best = 0.0
        for pod in obs.get("pods", []) or []:
            try:
                if not bool(pod.get("active", False)):
                    continue
                pos = np.asarray(pod.get("pos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
                if pos.shape[0] < 2:
                    continue
                stage = float(np.clip((0.95 - abs(float(pos[0]))) / 0.95, 0.0, 1.0))
                best = max(best, stage)
            except Exception:
                continue
        return float(np.clip(best, 0.0, 1.0))

    def _executive_interface_signals(
        self,
        *,
        obs: Dict[str, Any],
        condition_signals: Dict[str, float],
        world_surprise: float,
    ) -> Dict[str, float]:
        _, nearest_dist = self._nearest_pod_info(obs, far_only=False)
        _, far_dist = self._nearest_pod_info(obs, far_only=True)
        wall_blocking = 1.0 if bool(obs.get("wall", {}).get("blocking", False)) else 0.0
        effective_wall_blocking = float(wall_blocking)

        dist_delta = 0.0 if (self.prev_nearest_pod_dist is None or not np.isfinite(nearest_dist)) else float(self.prev_nearest_pod_dist - nearest_dist)
        far_dist_delta = 0.0 if (self.prev_far_pod_dist is None or not np.isfinite(far_dist)) else float(self.prev_far_pod_dist - far_dist)

        comfort_now = float(
            np.clip(
                1.0
                - 0.65 * float(np.clip(condition_signals.get("stomach_stress", 0.0), 0.0, 1.0))
                - 0.85 * float(np.clip(condition_signals.get("pain_level", 0.0), 0.0, 1.0)),
                0.0,
                1.0,
            )
        )
        prev_stress = float(max(0.0, 1.0 - float(self.prev_stomach if self.prev_stomach is not None else 0.4)))
        prev_pain = float(max(0.0, float(self.prev_stomach if self.prev_stomach is not None else 0.4) - 1.0))
        comfort_prev = float(np.clip(1.0 - 0.65 * prev_stress - 0.85 * prev_pain, 0.0, 1.0))
        condition_progress = float(np.clip(0.5 + 0.5 * (comfort_now - comfort_prev), 0.0, 1.0))
        distance_progress = float(np.clip(0.5 + 0.5 * np.clip(dist_delta, -1.0, 1.0), 0.0, 1.0)) if np.isfinite(nearest_dist) else 0.0
        direct_reachability = float(np.exp(-2.5 * nearest_dist)) if np.isfinite(nearest_dist) else 0.0
        region_state = self._anonymous_region_state(obs)
        region_active = bool(region_state["occupancy"] > 0.5)
        region_signature = float(region_state["signature"])
        region_proximity = float(region_state["proximity"])
        if (
            wall_blocking < 0.5
            and bool(self.prev_wall_blocking)
            and self.prev_region_active
            and self.prev_region_signature is not None
        ):
            effective_wall_blocking = 1.0
            region_active = True
            region_signature = float(self.prev_region_signature)
            region_proximity = float(
                max(
                    region_proximity,
                    float(self.prev_region_proximity if self.prev_region_proximity is not None else 0.0),
                )
            )
        region_entry = 0.0
        region_exit = 0.0
        if self.prev_region_active is not None:
            if (not self.prev_region_active) and region_active:
                region_entry = 1.0
            elif self.prev_region_active and (not region_active):
                region_exit = 1.0
            elif region_active and self.prev_region_signature is not None and abs(region_signature - float(self.prev_region_signature)) > 0.08:
                region_entry = 1.0
        region_dwell = float(self.prev_region_dwell + 1 if region_active else 0)
        region_entry_trace = float(np.clip(0.92 * float(self.prev_region_entry_trace) + region_entry, 0.0, 1.0))

        blocking_signal = float(np.clip(effective_wall_blocking * (1.0 if np.isfinite(far_dist) else 0.0), 0.0, 1.0))

        control_ineffectiveness = 0.0
        if self.prev_action is not None and self.prev_agent_pos is not None:
            prev_action = np.asarray(self.prev_action, dtype=np.float32).reshape(-1)
            curr_pos = np.asarray(obs.get("agent_pos", self.prev_agent_pos), dtype=np.float32).reshape(-1)
            if prev_action.shape[0] >= 2 and curr_pos.shape[0] >= 2:
                move_delta = curr_pos[:2] - self.prev_agent_pos[:2]
                prev_mag = float(np.linalg.norm(prev_action[:2]))
                if prev_mag > 0.05:
                    projected = float(np.dot(move_delta[:2], prev_action[:2] / max(prev_mag, 1e-6)))
                    repeated = 0.0
                    if self._last_executive_diag.get("action") is not None:
                        last_exec_action = np.asarray(self._last_executive_diag.get("action", [0.0, 0.0]), dtype=np.float32).reshape(-1)
                        if last_exec_action.shape[0] >= 2 and float(np.linalg.norm(last_exec_action[:2])) > 0.05:
                            repeated = float(
                                np.clip(
                                    0.5
                                    + 0.5
                                    * np.dot(
                                        prev_action[:2] / max(prev_mag, 1e-6),
                                        last_exec_action[:2] / max(float(np.linalg.norm(last_exec_action[:2])), 1e-6),
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                    proj_fail = float(np.clip((0.05 + prev_mag - max(0.0, projected) * 20.0) / 1.5, 0.0, 1.0))
                    control_ineffectiveness = float(np.clip(0.65 * max(repeated, 0.5 if effective_wall_blocking > 0.5 else 0.0) + 0.35 * proj_fail, 0.0, 1.0))
        means_feasibility = float(np.clip(direct_reachability * (1.0 - blocking_signal), 0.0, 1.0))
        trajectory_progress = float(np.clip((0.65 * condition_progress + 0.35 * distance_progress) * (0.6 + 0.4 * means_feasibility), 0.0, 1.0))
        sequence_state = float(self._sequence_state_proxy(obs))

        rollout_mismatch = 0.0
        if self.pending_predicted_tokens is not None:
            pred_arr = np.asarray(self.pending_predicted_tokens, dtype=np.float32)
            curr_tokens, _ = self.forward_component.tokenize_observation(obs)
            curr_arr = np.asarray(curr_tokens, dtype=np.float32)
            num_types = int(self.forward_component.schema.num_types)
            gates_pred = np.asarray(pred_arr[:, 0], dtype=np.float32)
            gates_curr = np.asarray(curr_arr[:, 0], dtype=np.float32)
            type_pred = np.argmax(np.asarray(pred_arr[:, 1 : 1 + num_types], dtype=np.float32), axis=-1)
            type_curr = np.argmax(np.asarray(curr_arr[:, 1 : 1 + num_types], dtype=np.float32), axis=-1)
            pred_candidates = (gates_pred >= float(self.forward_component.config.gate_present_threshold)) & (type_pred != 0) & (type_pred != (num_types - 2)) & (type_pred != (num_types - 1))
            curr_candidates = (gates_curr >= float(self.forward_component.config.gate_present_threshold)) & (type_curr != 0) & (type_curr != (num_types - 2)) & (type_curr != (num_types - 1))
            rollout_mismatch = float(np.clip(abs(float(np.sum(pred_candidates)) - float(np.sum(curr_candidates))) / 4.0, 0.0, 1.0))

        return {
            "region_occupancy": float(1.0 if region_active else 0.0),
            "region_entry": float(region_entry),
            "region_exit": float(region_exit),
            "region_signature": float(region_signature),
            "region_proximity": float(region_proximity),
            "region_dwell": float(np.clip(region_dwell / 32.0, 0.0, 1.0)),
            "region_entry_trace": float(region_entry_trace),
            "trajectory_progress": float(trajectory_progress),
            "blocking_signal": float(blocking_signal),
            "control_ineffectiveness": float(control_ineffectiveness),
            "means_feasibility": float(means_feasibility),
            "sequence_state": float(sequence_state),
            "rollout_mismatch": float(rollout_mismatch),
            "trajectory_progress_teacher": 0.0,
            "means_feasibility_teacher": 0.0,
        }

    def _executive_action(self, obs: Dict[str, Any], tools: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        tokens, _ = self.forward_component.tokenize_observation(obs)
        causal_beliefs: Dict[str, float] = {}
        causal_surprise: Dict[str, float] = {}
        causal_memory: Dict[str, float] = {}
        interface_signals: Dict[str, float] = {}
        if self.prev_tokens is not None and self.prev_action is not None:
            predicted = (
                np.asarray(self.pending_predicted_tokens, dtype=np.float32)
                if self.pending_predicted_tokens is not None
                else np.asarray(tokens, dtype=np.float32)
            )
            curr_arr = np.asarray(tokens, dtype=np.float32)
            pred_arr = np.asarray(predicted, dtype=np.float32)
            gate_err = float(np.mean(np.abs(pred_arr[:, 0] - curr_arr[:, 0])))
            geom_start = 1 + self.forward_component.schema.num_types
            geom_err = float(np.mean(np.abs(pred_arr[:, geom_start:] - curr_arr[:, geom_start:])))
            world_surprise = float(np.clip(0.75 * gate_err + 0.25 * geom_err, 0.0, 1.0))
            cond = self._condition_scalar_signals(np.asarray(self.prev_tokens, dtype=np.float32), curr_arr)
            interface_signals = self._executive_interface_signals(
                obs=obs,
                condition_signals=cond,
                world_surprise=world_surprise,
            )
            consumed = float(max(0.0, cond.get("stomach_delta", 0.0)) > 1e-4)
            transition = CodaCausalTransition(
                prev_tokens=np.asarray(self.prev_tokens, dtype=np.float32),
                action=np.asarray(self.prev_action, dtype=np.float32),
                curr_tokens=curr_arr,
                predicted_tokens=pred_arr,
                scalar_signals={
                    "world_surprise": float(world_surprise),
                    "coherence": float(np.clip(1.0 - world_surprise, 0.0, 1.0)),
                    "regime_signal": 0.0,
                    "env_change_observed": 0.0,
                    "interaction_event": float(consumed),
                    "consumed_token": float(consumed),
                    "delayed_change_target": 0.0,
                    "belief_probe_target": 0.0,
                    **cond,
                    **interface_signals,
                },
                metadata={"step": int(self.step_count)},
            )
            causal_out = self.causal_model.step(transition)
            self._causal_history.append(
                {
                    "consumed_token": float(consumed),
                    "stomach_delta": float(causal_out.evidence_dict.get("stomach_delta", 0.0)),
                    "pain_delta": float(causal_out.evidence_dict.get("pain_delta", 0.0)),
                    "stomach_stress": float(causal_out.evidence_dict.get("stomach_stress", 0.0)),
                }
            )
            if len(self._causal_history) > 256:
                self._causal_history = self._causal_history[-256:]
            causal_beliefs = dict(causal_out.belief_dict)
            causal_surprise = dict(causal_out.surprise_dict)
            causal_memory = dict(causal_out.memory_hints)
            self._last_causal_interface_signals = {
                key: float(causal_out.evidence_dict.get(key, 0.0))
                for key in (
                    "trajectory_progress",
                    "blocking_signal",
                    "control_ineffectiveness",
                    "means_feasibility",
                    "sequence_state",
                    "rollout_mismatch",
                    "trajectory_progress_teacher",
                    "means_feasibility_teacher",
                )
            }
        executive_input = CodaExecutiveInput(
            tokens=np.asarray(tokens, dtype=np.float32),
            causal_beliefs=causal_beliefs,
            causal_surprise=causal_surprise,
            causal_memory=dict(causal_memory),
            interface_signals=dict(self._last_causal_interface_signals if self._last_causal_interface_signals else interface_signals),
            metadata={
                "num_types": int(self.forward_component.schema.num_types),
                "condition_start_index": int(self.forward_component.schema.condition_start),
                "step": int(self.step_count),
            },
        )
        executive_out = self.executive_model.step(executive_input)
        self._last_executive_diag = {
            "wander_drive": float(executive_out.wander_drive),
            "commitment": float(executive_out.commitment),
            "progress_score": float(executive_out.progress_score),
            "selected_focus": str(executive_out.selected_focus),
            "focus_scores": dict(executive_out.focus_scores),
            "signals": dict(executive_out.debug.get("signals", {})),
            "interface_signals": dict(executive_out.debug.get("interface_signals", {})),
            "causal_memory": dict(executive_out.debug.get("causal_memory", {})),
            "action": np.asarray(executive_out.action, dtype=np.float32).tolist(),
        }
        return np.asarray(executive_out.action, dtype=np.float32), "executive_baseline"

    def _rl_action(self, obs: Dict[str, Any], tools: Dict[str, Any]) -> Optional[np.ndarray]:
        self._last_rl_diag = {
            "found": False,
            "class": "",
            "total_updates": float("nan"),
            "learning_enabled": float("nan"),
            "inference_path": "",
        }
        policies = tools.get("policies", None)
        if not isinstance(policies, dict):
            return None
        rl_policy = policies.get("RL agent", None)
        if rl_policy is None:
            return None
        if rl_policy is self:
            return None
        if not hasattr(rl_policy, "act"):
            return None
        self._last_rl_diag["found"] = True
        self._last_rl_diag["class"] = str(type(rl_policy).__name__)
        self._last_rl_diag["total_updates"] = float(getattr(rl_policy, "total_updates", float("nan")))
        self._last_rl_diag["learning_enabled"] = float(bool(getattr(rl_policy, "learning_enabled", False)))

        # Preferred path: deterministic forward pass without mutating RL rollout state.
        if all(hasattr(rl_policy, k) for k in ("_build_features", "_policy_forward", "device", "model")):
            try:
                import torch

                features = rl_policy._build_features(obs)  # type: ignore[attr-defined]
                obs_t = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(rl_policy.device).unsqueeze(0)  # type: ignore[attr-defined]
                rl_policy.model.eval()  # type: ignore[attr-defined]
                with torch.no_grad():
                    dist, _ = rl_policy._policy_forward(obs_t)  # type: ignore[attr-defined]
                    raw_action = dist.mean
                    action = torch.tanh(raw_action)
                out = action.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                if out.shape[0] >= 2:
                    self._last_rl_diag["inference_path"] = "direct_deterministic_forward"
                    return np.clip(out[:2], -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass

        # Fallback path: call policy.act, but temporarily freeze learning behavior.
        old_learning = getattr(rl_policy, "learning_enabled", None)
        try:
            if old_learning is not None:
                rl_policy.learning_enabled = False
            out = rl_policy.act(obs, tools, context=AgentExecutionContext.PLAYBACK)
            if isinstance(out, tuple) and len(out) >= 1:
                action = np.asarray(out[0], dtype=np.float32).reshape(-1)
            else:
                action = np.asarray(out, dtype=np.float32).reshape(-1)
            if action.shape[0] < 2:
                return None
            self._last_rl_diag["inference_path"] = "policy_act_fallback"
            return np.clip(action[:2], -1.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            return None
        finally:
            if old_learning is not None:
                try:
                    rl_policy.learning_enabled = old_learning
                except Exception:
                    pass

    def _human_action(self, obs: Dict[str, Any], tools: Dict[str, Any]) -> Optional[np.ndarray]:
        hc = tools.get("human_controller", None)
        if hc is not None and hasattr(hc, "run_step"):
            try:
                out = hc.run_step(obs)
                if isinstance(out, tuple) and len(out) >= 1:
                    action = np.asarray(out[0], dtype=np.float32).reshape(-1)
                else:
                    action = np.asarray(out, dtype=np.float32).reshape(-1)
                if action.shape[0] >= 2:
                    return np.clip(action[:2], -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass
        raw = tools.get("human_action", None)
        if raw is not None:
            try:
                action = np.asarray(raw, dtype=np.float32).reshape(-1)
                if action.shape[0] >= 2:
                    return np.clip(action[:2], -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:
                pass
        return None

    def _select_action(self, obs: Dict[str, Any], tools: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        mode = str(self.action_mode).strip().lower()
        if mode not in VALID_ACTION_MODES:
            mode = ACTION_MODE_WANDER

        wall_visible = self._detect_wall_visible_from_obs(obs)
        if mode == ACTION_MODE_WALL_CONDITIONAL:
            if wall_visible:
                return self._wander_action(), "wander_wall_present"
            rl_act = self._rl_action(obs, tools)
            if rl_act is not None:
                return rl_act, "rl_wall_absent"
            return self._wander_action(), "wander_fallback_no_rl"

        if mode == ACTION_MODE_RL:
            rl_act = self._rl_action(obs, tools)
            if rl_act is not None:
                return rl_act, "rl_forced"
            return self._wander_action(), "wander_fallback_no_rl"

        if mode == ACTION_MODE_HUMAN:
            human_act = self._human_action(obs, tools)
            if human_act is not None:
                return human_act, "human_forced"
            return self._wander_action(), "wander_fallback_no_human"

        if mode == ACTION_MODE_EXECUTIVE:
            try:
                return self._executive_action(obs, tools)
            except Exception:
                return self._wander_action(), "wander_fallback_executive_error"

        return self._wander_action(), "wander_forced"

    def _detect_reset(self, obs: Dict[str, Any]) -> bool:
        if not obs:
            return False

        phase = int(obs.get("phase", self.prev_phase if self.prev_phase is not None else 1))
        stomach = float(obs.get("stomach", self.prev_stomach if self.prev_stomach is not None else 0.4))
        pos_raw = np.asarray(
            obs.get("agent_pos", self.prev_agent_pos if self.prev_agent_pos is not None else [-0.5, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if pos_raw.shape[0] < 2:
            pos = np.array([-0.5, 0.0], dtype=np.float32)
        else:
            pos = pos_raw[:2]

        if self.prev_phase is not None and phase < self.prev_phase:
            return True

        if self.prev_agent_pos is not None:
            jump = float(np.linalg.norm(pos - self.prev_agent_pos))
            if jump > 1.25:
                return True

        if self.prev_stomach is not None and self.prev_stomach > 1.2 and stomach < 0.55:
            if float(np.linalg.norm(pos - np.array([-0.5, 0.0], dtype=np.float32))) < 0.35:
                return True

        return False

    def _update_obs_cache(self, obs: Dict[str, Any]):
        if not obs:
            return
        self.prev_phase = int(obs.get("phase", self.prev_phase if self.prev_phase is not None else 1))
        self.prev_stomach = float(obs.get("stomach", self.prev_stomach if self.prev_stomach is not None else 0.0))
        pos_raw = np.asarray(
            obs.get("agent_pos", self.prev_agent_pos if self.prev_agent_pos is not None else [-0.5, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if pos_raw.shape[0] >= 2:
            self.prev_agent_pos = pos_raw[:2].astype(np.float32, copy=False)
        _, nearest_dist = self._nearest_pod_info(obs, far_only=False)
        _, far_dist = self._nearest_pod_info(obs, far_only=True)
        region_state = self._anonymous_region_state(obs)
        self.prev_nearest_pod_dist = float(nearest_dist) if np.isfinite(nearest_dist) else None
        self.prev_far_pod_dist = float(far_dist) if np.isfinite(far_dist) else None
        region_active = bool(region_state["occupancy"] > 0.5)
        region_signature = float(region_state["signature"])
        region_entry = 0.0
        if self.prev_region_active is not None:
            if (not self.prev_region_active) and region_active:
                region_entry = 1.0
            elif region_active and self.prev_region_signature is not None and abs(region_signature - float(self.prev_region_signature)) > 0.08:
                region_entry = 1.0
        self.prev_region_active = region_active
        self.prev_region_signature = region_signature
        self.prev_region_proximity = float(region_state["proximity"])
        self.prev_region_dwell = (self.prev_region_dwell + 1) if self.prev_region_active else 0
        self.prev_region_entry_trace = float(np.clip(0.92 * float(self.prev_region_entry_trace) + region_entry, 0.0, 1.0))
        self.prev_wall_blocking = bool(obs.get("wall", {}).get("blocking", False))

    def _dump_debug_snapshot(
        self,
        *,
        frame: Any,
        tokens: np.ndarray,
        predicted_tokens: np.ndarray,
        prediction_aligned: bool,
        debug_views: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> bool:
        if self.debug_dump_every <= 0:
            return False

        try:
            os.makedirs(self.debug_dir, exist_ok=True)
        except Exception:
            self.last_debug_dump_error = "mkdir_failed"
            return False

        ts_ms = int(time.time() * 1000)
        stem = f"coda_step_{int(self.step_count):07d}_{ts_ms}"
        npz_path = os.path.join(self.debug_dir, f"{stem}.npz")

        frame_arr = None
        try:
            frame_arr = np.asarray(frame, dtype=np.float32) if frame is not None else None
        except Exception:
            frame_arr = None

        try:
            np.savez_compressed(
                npz_path,
                frame=frame_arr if frame_arr is not None else np.zeros((0, 0, 3), dtype=np.float32),
                tokens=np.asarray(tokens, dtype=np.float32),
                predicted_tokens=np.asarray(predicted_tokens, dtype=np.float32),
                reconstruction=np.asarray(debug_views.get("reconstruction"), dtype=np.float32),
                predicted_reconstruction=(
                    np.asarray(debug_views.get("predicted_reconstruction"), dtype=np.float32)
                    if debug_views.get("predicted_reconstruction") is not None
                    else np.zeros((0, 0, 3), dtype=np.float32)
                ),
            )
        except Exception:
            self.last_debug_dump_error = "npz_write_failed"
            return False

        json_path = os.path.join(self.debug_dir, f"{stem}.json")
        try:
            serializable = {}
            for key, value in summary.items():
                if isinstance(value, (float, int, str, bool)) or value is None:
                    serializable[key] = value
                elif isinstance(value, dict):
                    serializable[key] = {
                        str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in value.items()
                    }
                else:
                    serializable[key] = str(value)
            serializable["prediction_aligned"] = bool(prediction_aligned)
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, sort_keys=True)
        except Exception:
            self.last_debug_dump_error = "json_write_failed"
            return False
        self.last_debug_dump_path = json_path
        self.last_debug_dump_error = None
        return True

    def _run_forward_component(self, obs: Dict[str, Any], action: np.ndarray):
        if not self.enable_forward_model:
            return

        t0 = time.perf_counter()
        reset_detected = self._detect_reset(obs)
        t1 = time.perf_counter()
        tokens, token_diag = self.forward_component.tokenize_observation(obs)
        t2 = time.perf_counter()

        if reset_detected:
            self.pending_predicted_tokens = None

        if self.prev_tokens is not None and self.prev_action is not None and not reset_detected:
            self.forward_component.observe_transition(self.prev_tokens, self.prev_action, tokens)

        if reset_detected:
            self.forward_component.mark_reset()

        prediction_aligned = self.pending_predicted_tokens is not None
        aligned_predicted_tokens = (
            np.asarray(self.pending_predicted_tokens, dtype=np.float32)
            if prediction_aligned
            else None
        )

        should_try_update = True
        if self.fast_mode and self.fast_train_stride > 1:
            should_try_update = (self.step_count % self.fast_train_stride) == 0
        train_metrics = self.forward_component.maybe_update() if should_try_update else None
        t3 = time.perf_counter()
        need_debug_views = bool(self.debug_view_enabled or self.debug_logging_enabled)
        if need_debug_views:
            debug_views = self.forward_component.build_debug_views(
                frame_t=obs.get("frame"),
                tokens_t=tokens,
                predicted_tokens_for_target=aligned_predicted_tokens,
                frame_target_for_prediction=(obs.get("frame") if prediction_aligned else None),
            )
        else:
            debug_views = {
                "reconstruction": None,
                "predicted_reconstruction": None,
                "reconstruction_error": float("nan"),
                "predicted_reconstruction_error": float("nan"),
                "center_pixel": {},
                "predicted_center_contributors": [],
                "reconstruction_center_contributors": [],
            }
        t4 = time.perf_counter()
        next_predicted_tokens = self.forward_component.predict_next_tokens(tokens, action, use_ema=True)
        t5 = time.perf_counter()
        predicted_tokens_for_dump = (
            aligned_predicted_tokens if aligned_predicted_tokens is not None else next_predicted_tokens
        )

        token_summary = self.forward_component.summarize_tokens(tokens)
        summary = {
            "step": int(self.step_count),
            "action_mode": str(self.action_mode),
            "action_source": str(self._last_action_source),
            "rl_found": bool(self._last_rl_diag.get("found", False)),
            "rl_class": str(self._last_rl_diag.get("class", "")),
            "rl_total_updates": float(self._last_rl_diag.get("total_updates", float("nan"))),
            "rl_learning_enabled": float(self._last_rl_diag.get("learning_enabled", float("nan"))),
            "rl_inference_path": str(self._last_rl_diag.get("inference_path", "")),
            "executive_wander_drive": float(self._last_executive_diag.get("wander_drive", float("nan"))),
            "executive_commitment": float(self._last_executive_diag.get("commitment", float("nan"))),
            "executive_progress_score": float(self._last_executive_diag.get("progress_score", float("nan"))),
            "executive_selected_focus": str(self._last_executive_diag.get("selected_focus", "")),
            "reset_detected": bool(reset_detected),
            "prediction_aligned": bool(prediction_aligned),
            "token_source": token_diag.get("source", "unknown"),
            "component_counts": token_diag.get("component_counts", {}),
            "candidate_source": token_diag.get("candidate_source", {}),
            "wall_visible": bool(token_diag.get("wall_visible", False)),
            "active_slots": int(token_summary.get("active_slots", 0)),
            "active_per_type": token_summary.get("active_per_type", {}),
            "gate_mean": float(token_summary.get("gate_mean", 0.0)),
            "reconstruction_error": float(debug_views.get("reconstruction_error", float("nan"))),
            "predicted_reconstruction_error": (
                float(debug_views.get("predicted_reconstruction_error", float("nan")))
                if prediction_aligned
                else float("nan")
            ),
            "buffer_size": int(len(self.forward_component.buffer)),
            "total_updates": int(self.forward_component.total_updates),
            "logging_enabled": bool(self.debug_logging_enabled),
            "debug_dump_every": int(self.debug_dump_every),
            "train_updated": bool(train_metrics is not None),
            "debug_dir": str(self.debug_dir),
            "debug_last_dump_path": str(self.last_debug_dump_path) if self.last_debug_dump_path else "",
            "debug_last_dump_error": str(self.last_debug_dump_error) if self.last_debug_dump_error else "",
            "center_pixel": dict(debug_views.get("center_pixel", {})),
            "predicted_center_contributors": list(debug_views.get("predicted_center_contributors", [])),
            "reconstruction_center_contributors": list(debug_views.get("reconstruction_center_contributors", [])),
            "timing_detect_reset_ms": float((t1 - t0) * 1000.0),
            "timing_tokenize_ms": float((t2 - t1) * 1000.0),
            "timing_train_update_ms": float((t3 - t2) * 1000.0),
            "timing_debug_views_ms": float((t4 - t3) * 1000.0),
            "timing_predict_ms": float((t5 - t4) * 1000.0),
            "timing_forward_total_ms": float((t5 - t0) * 1000.0),
            "fast_mode": bool(self.fast_mode),
            "fast_train_stride": int(self.fast_train_stride),
            "effective_batch_size": int(self.forward_component.config.batch_size),
            "effective_update_every": int(self.forward_component.config.update_every),
            "effective_warmup_transitions": int(self.forward_component.config.warmup_transitions),
            "effective_event_sample_fraction": float(self.forward_component.config.event_sample_fraction),
            "effective_event_heavy_every": int(self.forward_component.config.event_heavy_every),
            "effective_event_window_steps": int(self.forward_component.config.event_window_steps),
            "effective_event_transition_weight": float(self.forward_component.config.event_transition_weight),
        }
        if hasattr(self.forward_component, "get_transition_stats"):
            try:
                tstats = dict(self.forward_component.get_transition_stats())
            except Exception:
                tstats = {}
            for k, v in tstats.items():
                try:
                    summary[f"data_{k}"] = float(v)
                except Exception:
                    summary[f"data_{k}"] = v
        metrics_src = train_metrics if train_metrics is not None else self.forward_component.last_train_metrics
        if metrics_src:
            summary["train_loss"] = float(metrics_src.get("loss", float("nan")))
            summary["train_gate_loss"] = float(metrics_src.get("gate_loss", float("nan")))
            summary["train_type_loss"] = float(metrics_src.get("type_loss", float("nan")))
            summary["train_geom_loss"] = float(metrics_src.get("geom_loss", float("nan")))
            summary["train_hazard_loss"] = float(metrics_src.get("hazard_loss", float("nan")))
            summary["train_despawn_loss"] = float(metrics_src.get("despawn_loss", float("nan")))
            summary["train_collision_loss"] = float(metrics_src.get("collision_loss", float("nan")))
            summary["train_contact_drop_loss"] = float(metrics_src.get("contact_drop_loss", float("nan")))
            summary["per_token_loss_mean"] = float(metrics_src.get("per_token_loss_mean", float("nan")))
            summary["per_token_loss_max"] = float(metrics_src.get("per_token_loss_max", float("nan")))
            summary["per_token_loss_argmax"] = int(metrics_src.get("per_token_loss_argmax", -1))
            summary["edge_attr_max"] = float(metrics_src.get("edge_attr_max", float("nan")))
            summary["geom_logvar_mean"] = float(metrics_src.get("geom_logvar_mean", float("nan")))
            summary["edge_attribution_max_per_token"] = list(metrics_src.get("edge_attribution_max_per_token", []))
            summary["contract_penetration_rate"] = float(metrics_src.get("contract_penetration_rate", float("nan")))
            summary["contract_pod_drop_recall"] = float(metrics_src.get("contract_pod_drop_recall", float("nan")))
            summary["contract_wall_drop_recall"] = float(metrics_src.get("contract_wall_drop_recall", float("nan")))
            summary["split_normal_pos_error"] = float(metrics_src.get("split_normal_pos_error", float("nan")))
            summary["split_wall_contact_pos_error"] = float(metrics_src.get("split_wall_contact_pos_error", float("nan")))
            summary["split_pod_contact_pos_error"] = float(metrics_src.get("split_pod_contact_pos_error", float("nan")))
            summary["split_event_pos_error"] = float(metrics_src.get("split_event_pos_error", float("nan")))
            summary["split_event_gate_accuracy"] = float(metrics_src.get("split_event_gate_accuracy", float("nan")))

        self.last_debug_info = summary
        self.debug_history.append(summary)
        self.latest_debug_packet = {
            "step": int(self.step_count),
            "summary": dict(summary),
            "frame": (
                np.asarray(obs.get("frame"), dtype=np.float32).copy()
                if (self.debug_view_enabled and obs.get("frame") is not None)
                else None
            ),
            "reconstruction": (
                np.asarray(debug_views.get("reconstruction"), dtype=np.float32).copy()
                if debug_views.get("reconstruction") is not None
                else None
            ),
            "predicted_reconstruction": (
                np.asarray(debug_views.get("predicted_reconstruction"), dtype=np.float32).copy()
                if debug_views.get("predicted_reconstruction") is not None
                else None
            ),
            "prediction_aligned": bool(prediction_aligned),
        }

        should_dump = self.debug_logging_enabled and self.debug_dump_every > 0 and (
            self._debug_force_dump_once or (self.step_count % self.debug_dump_every == 0)
        )
        if should_dump:
            dumped = self._dump_debug_snapshot(
                frame=obs.get("frame"),
                tokens=tokens,
                predicted_tokens=predicted_tokens_for_dump,
                prediction_aligned=prediction_aligned,
                debug_views=debug_views,
                summary=summary,
            )
            if dumped:
                self._debug_force_dump_once = False

        self.prev_tokens = np.asarray(tokens, dtype=np.float32).copy()
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.pending_predicted_tokens = np.asarray(next_predicted_tokens, dtype=np.float32).copy()

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        self.step_count += 1

        obs = self._obs_to_dict(observation)
        action, action_source = self._select_action(obs, tools)
        self._last_action_source = str(action_source)
        self._run_forward_component(obs, action)
        self._update_obs_cache(obs)
        return action, None

    def reset_episode(self):
        self.prev_tokens = None
        self.prev_action = None
        self.prev_phase = None
        self.prev_stomach = None
        self.prev_agent_pos = None
        self.prev_nearest_pod_dist = None
        self.prev_far_pod_dist = None
        self.prev_region_active = None
        self.prev_region_signature = None
        self.prev_region_proximity = None
        self.prev_region_dwell = 0
        self.prev_region_entry_trace = 0.0
        self.prev_wall_blocking = None
        self.pending_predicted_tokens = None
        self._last_action_source = "wander_forced"
        self._last_rl_diag = {
            "found": False,
            "class": "",
            "total_updates": float("nan"),
            "learning_enabled": float("nan"),
            "inference_path": "",
        }
        self._causal_history = []
        self._last_executive_diag = {}
        self._last_causal_interface_signals = {}
        self.causal_model = CodaCausalModel(build_causal_config())
        self.executive_model.reset()
        self.latest_debug_packet = {}
        self.forward_component.mark_reset()

    def on_environment_reset(self):
        self.reset_episode()

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "last": dict(self.last_debug_info),
            "history": list(self.debug_history),
        }

    def get_latest_debug_packet(self) -> Dict[str, Any]:
        if not self.latest_debug_packet:
            return {}
        packet = {
            "step": int(self.latest_debug_packet.get("step", 0)),
            "summary": dict(self.latest_debug_packet.get("summary", {})),
            "prediction_aligned": bool(self.latest_debug_packet.get("prediction_aligned", False)),
        }
        frame = self.latest_debug_packet.get("frame")
        recon = self.latest_debug_packet.get("reconstruction")
        pred = self.latest_debug_packet.get("predicted_reconstruction")
        packet["frame"] = np.asarray(frame, dtype=np.float32).copy() if frame is not None else None
        packet["reconstruction"] = np.asarray(recon, dtype=np.float32).copy() if recon is not None else None
        packet["predicted_reconstruction"] = np.asarray(pred, dtype=np.float32).copy() if pred is not None else None
        return packet

    def set_debug_logging(self, enabled: bool, *, every: Optional[int] = None, out_dir: Optional[str] = None):
        prev_enabled = bool(self.debug_logging_enabled)
        self.debug_logging_enabled = bool(enabled)
        if every is not None:
            self.debug_dump_every = int(max(1, every))
        if out_dir is not None and str(out_dir).strip():
            self.debug_dir = os.path.abspath(str(out_dir).strip())
        if self.debug_logging_enabled and not prev_enabled:
            self._debug_force_dump_once = True

    def set_debug_view_enabled(self, enabled: bool):
        self.debug_view_enabled = bool(enabled)

    def _apply_fast_mode_profile(self):
        cfg = self.forward_component.config
        if self.fast_mode:
            cfg.batch_size = int(max(4, self._base_forward_batch_size // 2))
            cfg.update_every = int(max(self._base_forward_update_every * 2, self._base_forward_update_every + 1))
            cfg.warmup_transitions = int(max(self._base_forward_warmup, 256))
        else:
            cfg.batch_size = int(self._base_forward_batch_size)
            cfg.update_every = int(self._base_forward_update_every)
            cfg.warmup_transitions = int(self._base_forward_warmup)
        # Event-focused defaults (formerly stress profile) are now always on.
        cfg.batch_size = int(max(cfg.batch_size, 16))
        cfg.update_every = int(max(1, min(cfg.update_every, 2)))
        cfg.warmup_transitions = int(min(cfg.warmup_transitions, 128))
        cfg.event_sample_fraction = 0.7
        cfg.event_heavy_every = 2
        cfg.event_heavy_fraction = 0.9
        cfg.event_window_steps = 4
        cfg.event_transition_weight = 4.0
        cfg.loss_hazard_weight = 2.0
        cfg.loss_despawn_weight = 2.0
        cfg.loss_contact_drop_weight = 2.0
        cfg.contact_condition_weight = 2.5
        cfg.focal_alpha = 0.7
        cfg.focal_gamma = 2.0
        cfg.gate_transition_weighted = True

    def set_fast_mode(self, enabled: bool, *, train_stride: Optional[int] = None):
        self.fast_mode = bool(enabled)
        if train_stride is not None:
            self.fast_train_stride = int(max(1, train_stride))
        self._apply_fast_mode_profile()

    def set_action_mode(self, mode: str):
        m = str(mode).strip().lower()
        if m in VALID_ACTION_MODES:
            self.action_mode = m

    def get_action_mode(self) -> str:
        return str(self.action_mode)

    def auto_tune_forward_settings(
        self,
        *,
        trials: int = 8,
        train_updates_per_trial: int = 128,
        eval_samples: int = 1024,
        eval_horizon: int = 12,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        from utils.coda_eval_dashboard import evaluate_policy

        fc = self.forward_component
        if len(fc.buffer) < int(max(32, fc.config.warmup_transitions)):
            return {
                "ok": False,
                "reason": "not_enough_buffer_data",
                "buffer_size": int(len(fc.buffer)),
                "warmup_required": int(fc.config.warmup_transitions),
            }

        rng = np.random.default_rng(int(self.seed + 17011) if seed is None else int(seed))
        base_state = copy.deepcopy(fc.state_dict())
        base_cfg = copy.deepcopy(fc.config)
        base_fast_mode = bool(self.fast_mode)
        base_stride = int(self.fast_train_stride)

        def _sample_candidate() -> Dict[str, Any]:
            return {
                "batch_size": int(rng.choice([8, 12, 16, 24])),
                "update_every": int(rng.choice([1, 2, 4])),
                "event_sample_fraction": float(rng.choice([0.5, 0.7, 0.85])),
                "event_heavy_every": int(rng.choice([1, 2, 4])),
                "event_heavy_fraction": float(rng.choice([0.8, 0.9, 1.0])),
                "event_window_steps": int(rng.choice([2, 4, 6])),
                "event_transition_weight": float(rng.choice([2.0, 4.0, 6.0])),
                "loss_hazard_weight": float(rng.choice([1.0, 2.0, 3.0])),
                "loss_despawn_weight": float(rng.choice([1.0, 2.0, 3.0])),
                "loss_contact_drop_weight": float(rng.choice([1.0, 2.0, 3.0])),
                "contact_condition_weight": float(rng.choice([1.5, 2.5, 3.5])),
                "focal_alpha": float(rng.choice([0.55, 0.7, 0.85])),
                "focal_gamma": float(rng.choice([1.0, 1.5, 2.0])),
                "gate_transition_weighted": bool(rng.choice([True, True, False])),
            }

        def _apply_candidate(cfg_obj: CodaDynamicsConfig, cand: Dict[str, Any]):
            for k, v in cand.items():
                if hasattr(cfg_obj, k):
                    setattr(cfg_obj, k, v)
            cfg_obj.batch_size = int(max(4, cfg_obj.batch_size))
            cfg_obj.update_every = int(max(1, cfg_obj.update_every))
            cfg_obj.event_sample_fraction = float(np.clip(cfg_obj.event_sample_fraction, 0.0, 1.0))
            cfg_obj.event_heavy_fraction = float(np.clip(cfg_obj.event_heavy_fraction, 0.0, 1.0))
            cfg_obj.event_heavy_every = int(max(1, cfg_obj.event_heavy_every))
            cfg_obj.event_window_steps = int(max(0, cfg_obj.event_window_steps))
            cfg_obj.event_transition_weight = float(max(1.0, cfg_obj.event_transition_weight))
            cfg_obj.contact_condition_weight = float(max(1.0, cfg_obj.contact_condition_weight))
            cfg_obj.focal_alpha = float(np.clip(cfg_obj.focal_alpha, 0.01, 0.99))
            cfg_obj.focal_gamma = float(max(0.0, cfg_obj.focal_gamma))

        def _score(metrics: Dict[str, Any]) -> float:
            tr = metrics.get("event_detection_transition", {})
            disappear = tr.get("disappear", {})
            spawn = tr.get("spawn", {})
            wall = tr.get("wall_open", {})
            drift = metrics.get("rollout_drift_by_horizon", [])
            drift10 = float(drift[9]) if isinstance(drift, list) and len(drift) > 9 else float("nan")
            score = 0.0
            score += 2.5 * float(disappear.get("tpr", 0.0))
            score += 2.5 * float(wall.get("tpr", 0.0))
            score += 1.2 * float(spawn.get("tpr", 0.0))
            score -= 2.0 * float(disappear.get("fpr", 0.0))
            score -= 2.0 * float(wall.get("fpr", 0.0))
            score -= 0.8 * float(spawn.get("fpr", 0.0))
            score += 0.01 * (
                float(disappear.get("tp", 0.0))
                + float(wall.get("tp", 0.0))
                + 0.5 * float(spawn.get("tp", 0.0))
            )
            if np.isfinite(drift10):
                score -= 0.4 * drift10
            return float(score)

        trial_rows: list[Dict[str, Any]] = []
        best_idx = -1
        best_score = -1e18
        best_cfg: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None

        try:
            for trial_i in range(int(max(1, trials))):
                fc.load_state_dict(copy.deepcopy(base_state))
                cand = _sample_candidate()
                _apply_candidate(fc.config, cand)
                self.fast_mode = False
                self.fast_train_stride = 1
                n_updates = int(max(8, train_updates_per_trial))
                for _ in range(n_updates):
                    _ = fc.maybe_update(force=True)

                metrics = evaluate_policy(
                    self,
                    max_samples=int(max(64, eval_samples)),
                    horizon=int(max(4, eval_horizon)),
                )
                sc = _score(metrics)
                row = {
                    "trial": int(trial_i + 1),
                    "score": float(sc),
                    "config": dict(cand),
                    "event_detection_transition": metrics.get("event_detection_transition", {}),
                    "event_prediction_activity": metrics.get("event_prediction_activity", {}),
                }
                trial_rows.append(row)
                if sc > best_score:
                    best_score = sc
                    best_idx = trial_i
                    best_cfg = dict(cand)
                    best_metrics = metrics
        finally:
            fc.load_state_dict(copy.deepcopy(base_state))
            fc.config = copy.deepcopy(base_cfg)
            self.fast_mode = base_fast_mode
            self.fast_train_stride = base_stride
            self._apply_fast_mode_profile()

        if best_cfg is None:
            return {"ok": False, "reason": "no_trials"}

        _apply_candidate(fc.config, best_cfg)
        self.fast_mode = False
        self.fast_train_stride = 1
        for _ in range(int(max(16, train_updates_per_trial))):
            _ = fc.maybe_update(force=True)

        return {
            "ok": True,
            "best_trial_index": int(best_idx + 1),
            "best_score": float(best_score),
            "best_config": dict(best_cfg),
            "best_metrics": best_metrics if isinstance(best_metrics, dict) else {},
            "trials": trial_rows,
            "applied": True,
            "post_profile": {
                "fast_mode": bool(self.fast_mode),
                "fast_train_stride": int(self.fast_train_stride),
                "batch_size": int(fc.config.batch_size),
                "update_every": int(fc.config.update_every),
            },
        }

    def save(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "format": "coda_forward_v1",
            "meta": {
                "speed": float(self.speed),
                "turn_every": int(self.turn_every),
                "jitter": float(self.jitter),
                "seed": int(self.seed),
                "step_count": int(self.step_count),
                "direction": np.asarray(self.direction, dtype=np.float32),
                "rng_state": self.rng.bit_generator.state,
                "enable_forward_model": bool(self.enable_forward_model),
                "train_forward_model": bool(self.train_forward_model),
                "fast_mode": bool(self.fast_mode),
                "fast_train_stride": int(self.fast_train_stride),
                "base_forward_batch_size": int(self._base_forward_batch_size),
                "base_forward_warmup": int(self._base_forward_warmup),
                "base_forward_update_every": int(self._base_forward_update_every),
                "action_mode": str(self.action_mode),
                "debug_logging_enabled": bool(self.debug_logging_enabled),
                "debug_dump_every": int(self.debug_dump_every),
                "debug_dir": str(self.debug_dir),
            },
            "forward_component": self.forward_component.state_dict(),
            "debug": {"last": dict(self.last_debug_info)},
        }
        torch.save(payload, path)

    def load(self, path: str):
        loaded_new_format = False
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")

            if isinstance(payload, dict) and payload.get("format") == "coda_forward_v1":
                meta = payload.get("meta", {})
                self.speed = float(meta.get("speed", self.speed))
                self.turn_every = int(meta.get("turn_every", self.turn_every))
                self.jitter = float(meta.get("jitter", self.jitter))
                self.seed = int(meta.get("seed", self.seed))
                self.step_count = int(meta.get("step_count", self.step_count))

                direction = np.asarray(meta.get("direction", self.direction), dtype=np.float32).reshape(-1)
                if direction.shape[0] >= 2:
                    self.direction = self._normalize(direction[:2])
                else:
                    self.direction = self._sample_unit_vec()

                self.enable_forward_model = bool(meta.get("enable_forward_model", self.enable_forward_model))
                self.train_forward_model = bool(meta.get("train_forward_model", self.train_forward_model))
                self.fast_mode = bool(meta.get("fast_mode", self.fast_mode))
                self.fast_train_stride = int(meta.get("fast_train_stride", self.fast_train_stride))
                self._base_forward_batch_size = int(meta.get("base_forward_batch_size", self._base_forward_batch_size))
                self._base_forward_warmup = int(meta.get("base_forward_warmup", self._base_forward_warmup))
                self._base_forward_update_every = int(
                    meta.get("base_forward_update_every", self._base_forward_update_every)
                )
                self.set_action_mode(str(meta.get("action_mode", self.action_mode)))
                self.debug_logging_enabled = bool(meta.get("debug_logging_enabled", self.debug_logging_enabled))
                self.debug_dump_every = int(meta.get("debug_dump_every", self.debug_dump_every))
                self.debug_dir = os.path.abspath(str(meta.get("debug_dir", self.debug_dir)))
                self._apply_fast_mode_profile()

                self.rng = np.random.default_rng(self.seed)
                try:
                    rng_state = meta.get("rng_state", None)
                    if rng_state is not None:
                        self.rng.bit_generator.state = rng_state
                except Exception:
                    pass

                self.forward_component.training_enabled = self.train_forward_model
                self.forward_component.load_state_dict(payload.get("forward_component", {}))
                debug_block = payload.get("debug", {})
                self.last_debug_info = dict(debug_block.get("last", self.last_debug_info))

                self.reset_episode()
                loaded_new_format = True
        except Exception:
            loaded_new_format = False

        if loaded_new_format:
            return

        # Backward compatibility: old CODA checkpoint saved via np.savez.
        data = np.load(path, allow_pickle=False)
        self.speed = float(np.array(data.get("speed", np.array([self.speed]))).reshape(-1)[0])
        self.turn_every = int(np.array(data.get("turn_every", np.array([self.turn_every]))).reshape(-1)[0])
        self.jitter = float(np.array(data.get("jitter", np.array([self.jitter]))).reshape(-1)[0])
        self.seed = int(np.array(data.get("seed", np.array([self.seed]))).reshape(-1)[0])
        self.step_count = int(np.array(data.get("step_count", np.array([0]))).reshape(-1)[0])
        direction = np.array(data.get("direction", self.direction), dtype=np.float32).reshape(-1)
        if direction.shape[0] >= 2:
            self.direction = self._normalize(direction[:2])
        else:
            self.direction = self._sample_unit_vec()
        self.rng = np.random.default_rng(self.seed)
        self.reset_episode()
