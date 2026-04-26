from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-6


@dataclass
class CodaExecutiveConfig:
    """
    Iteration-friendly executive scaffold.

    The first job of the executive is intentionally narrow:
    - wander early
    - gradually reduce wandering
    - act through the lens of world-model tokens and causal-model signals
    - avoid hardwiring specific object classes as goals

    This file is designed to be easy to ablate, probe, and extend before we
    wire it deeply into live CODA behavior.
    """

    num_types: int = 5
    type_agent: int = 0
    type_pod: int = 1
    type_wall: int = 2
    type_condition: int = 3
    type_empty: int = 4
    condition_start_index: Optional[int] = None
    wander_speed: float = 0.42
    goal_speed: float = 0.85
    action_smoothing: float = 0.25
    stop_drift: float = 0.08
    distance_scale: float = 2.5
    novelty_gain: float = 1.1
    uncertainty_gain: float = 0.9
    memory_support_gain: float = 1.0
    repeated_failure_gain: float = 0.8
    goal_causal_support_gain: float = 0.4
    goal_support_context_gate_gain: float = 1.0
    low_need_gain: float = 1.0
    high_need_food_penalty: float = 1.4
    satiety_center: float = 0.58
    satiety_width: float = 0.10
    progress_gain: float = 1.2
    pain_gain: float = 0.9
    commitment_comfort_suppression: float = 0.0
    commitment_high_need_suppression: float = 0.0
    commitment_bias: float = 0.0
    variant: str = "wander_heavy"

    def with_overrides(self, **overrides: Any) -> "CodaExecutiveConfig":
        return replace(self, **overrides)


@dataclass
class CodaExecutiveInput:
    tokens: np.ndarray
    causal_beliefs: Dict[str, float] = field(default_factory=dict)
    causal_surprise: Dict[str, float] = field(default_factory=dict)
    causal_memory: Dict[str, float] = field(default_factory=dict)
    interface_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodaExecutiveOutput:
    action: np.ndarray
    wander_drive: float
    commitment: float
    progress_score: float
    selected_focus: str
    focus_scores: Dict[str, float]
    debug: Dict[str, Any]


@dataclass
class CodaExecutiveState:
    timestep: int = 0
    wander_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))
    prev_stomach_level: float = 0.0
    prev_pain_level: float = 0.0
    stalled_food_steps: float = 0.0
    prev_region_signature: float = -1.0
    prev2_region_signature: float = -1.0
    general_hold: float = 0.0
    general_signature: float = -1.0
    general_kind: str = ""
    general_xy: Optional[np.ndarray] = None
    reacquire_mode: int = 0
    reacquire_goal_idx: int = -1
    trap_hold_count: int = 0
    negative_affordance_penalties: Dict[str, float] = field(default_factory=dict)


@dataclass
class CodaLearnedGoalValue:
    variant: str
    weights: np.ndarray
    bias: float = 0.0
    feature_mean: Optional[np.ndarray] = None
    feature_scale: Optional[np.ndarray] = None


@dataclass
class CodaLearnedCommitment:
    variant: str
    weights: np.ndarray
    bias: float = 0.0
    feature_mean: Optional[np.ndarray] = None
    feature_scale: Optional[np.ndarray] = None


EXECUTIVE_PRESET_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "wander_heavy": {
        "variant": "wander_heavy",
        "novelty_gain": 1.35,
        "uncertainty_gain": 1.0,
        "memory_support_gain": 0.7,
        "repeated_failure_gain": 0.45,
        "goal_causal_support_gain": 0.20,
        "goal_support_context_gate_gain": 1.10,
        "progress_gain": 0.85,
        "pain_gain": 0.55,
        "commitment_comfort_suppression": 0.35,
        "commitment_high_need_suppression": 0.20,
        "commitment_bias": -0.15,
    },
    "progress_mixed_goal_pursuit": {
        "variant": "progress_mixed_goal_pursuit",
        "novelty_gain": 1.0,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 0.95,
        "repeated_failure_gain": 0.75,
        "goal_causal_support_gain": 0.50,
        "goal_support_context_gate_gain": 1.00,
        "progress_gain": 1.15,
        "pain_gain": 0.85,
        "commitment_comfort_suppression": 0.55,
        "commitment_high_need_suppression": 0.30,
        "commitment_bias": 0.0,
    },
    "pain_aware_comfort_controller": {
        "variant": "pain_aware_comfort_controller",
        "novelty_gain": 0.85,
        "uncertainty_gain": 0.75,
        "memory_support_gain": 1.10,
        "repeated_failure_gain": 0.90,
        "goal_causal_support_gain": 0.65,
        "goal_support_context_gate_gain": 1.20,
        "progress_gain": 1.35,
        "pain_gain": 1.20,
        "high_need_food_penalty": 1.8,
        "commitment_comfort_suppression": 0.85,
        "commitment_high_need_suppression": 0.45,
        "commitment_bias": 0.10,
    },
    "learned_goal_value_pooled": {
        "variant": "learned_goal_value_pooled",
        "novelty_gain": 1.0,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 0.95,
        "repeated_failure_gain": 0.75,
        "goal_causal_support_gain": 0.50,
        "goal_support_context_gate_gain": 1.00,
        "progress_gain": 1.15,
        "pain_gain": 0.85,
        "commitment_comfort_suppression": 0.55,
        "commitment_high_need_suppression": 0.30,
        "commitment_bias": 0.0,
    },
    "learned_goal_value_relational": {
        "variant": "learned_goal_value_relational",
        "novelty_gain": 0.95,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 1.0,
        "repeated_failure_gain": 0.80,
        "goal_causal_support_gain": 0.55,
        "goal_support_context_gate_gain": 1.05,
        "progress_gain": 1.20,
        "pain_gain": 0.95,
        "commitment_comfort_suppression": 0.60,
        "commitment_high_need_suppression": 0.35,
        "commitment_bias": 0.0,
    },
    "learned_goal_value_with_memory": {
        "variant": "learned_goal_value_with_memory",
        "novelty_gain": 0.90,
        "uncertainty_gain": 0.80,
        "memory_support_gain": 1.10,
        "repeated_failure_gain": 0.85,
        "goal_causal_support_gain": 0.65,
        "goal_support_context_gate_gain": 1.10,
        "progress_gain": 1.25,
        "pain_gain": 1.05,
        "commitment_comfort_suppression": 0.65,
        "commitment_high_need_suppression": 0.35,
        "commitment_bias": 0.05,
    },
    "learned_commitment_pooled": {
        "variant": "learned_commitment_pooled",
        "novelty_gain": 0.95,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 1.00,
        "repeated_failure_gain": 0.80,
        "goal_causal_support_gain": 0.65,
        "goal_support_context_gate_gain": 1.20,
        "progress_gain": 1.35,
        "pain_gain": 1.20,
        "high_need_food_penalty": 1.8,
        "commitment_comfort_suppression": 0.85,
        "commitment_high_need_suppression": 0.45,
        "commitment_bias": 0.10,
    },
    "learned_commitment_comfort": {
        "variant": "learned_commitment_comfort",
        "novelty_gain": 0.95,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 1.00,
        "repeated_failure_gain": 0.80,
        "goal_causal_support_gain": 0.65,
        "goal_support_context_gate_gain": 1.20,
        "progress_gain": 1.35,
        "pain_gain": 1.20,
        "high_need_food_penalty": 1.8,
        "commitment_comfort_suppression": 0.95,
        "commitment_high_need_suppression": 0.55,
        "commitment_bias": 0.10,
    },
    "learned_commitment_with_memory": {
        "variant": "learned_commitment_with_memory",
        "novelty_gain": 0.95,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 1.05,
        "repeated_failure_gain": 0.85,
        "goal_causal_support_gain": 0.70,
        "goal_support_context_gate_gain": 1.20,
        "progress_gain": 1.35,
        "pain_gain": 1.20,
        "high_need_food_penalty": 1.8,
        "commitment_comfort_suppression": 0.90,
        "commitment_high_need_suppression": 0.50,
        "commitment_bias": 0.10,
    },
    # Backward-compatible alias while we transition the conceptual language
    # from direct pod pursuit to general goal pursuit.
    "progress_mixed_pod_pursuit": {
        "variant": "progress_mixed_goal_pursuit",
        "novelty_gain": 1.0,
        "uncertainty_gain": 0.85,
        "memory_support_gain": 0.95,
        "repeated_failure_gain": 0.75,
        "goal_causal_support_gain": 0.50,
        "progress_gain": 1.15,
        "pain_gain": 0.85,
        "commitment_bias": 0.0,
    },
}


def build_executive_config(preset: str = "wander_heavy", **overrides: Any) -> CodaExecutiveConfig:
    cfg = CodaExecutiveConfig()
    preset_overrides = EXECUTIVE_PRESET_OVERRIDES.get(str(preset).strip().lower(), {})
    if preset_overrides:
        cfg = cfg.with_overrides(**preset_overrides)
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


class CodaExecutiveModel:
    """
    Lightweight executive layer for early CODA experiments.

    Current behavior is intentionally narrow:
    - derive smooth wander pressure
    - estimate early candidate-means usefulness from internal state + causal support
    - mix wandering and selected-means pursuit continuously

    This should be treated as a scaffold for repeated iteration, not as the
    final executive architecture.
    """

    def __init__(self, config: Optional[CodaExecutiveConfig] = None, *, seed: int = 17):
        self.config = config or CodaExecutiveConfig()
        self.rng = np.random.default_rng(int(seed))
        self.state = CodaExecutiveState()
        self.last_debug: Dict[str, Any] = {}
        self.learned_goal_value: Optional[CodaLearnedGoalValue] = None
        self.learned_commitment: Optional[CodaLearnedCommitment] = None

    def reset(self) -> None:
        self.state = CodaExecutiveState(wander_dir=self._sample_unit_vec())
        self.last_debug = {}

    def clone_config(self, **overrides: Any) -> CodaExecutiveConfig:
        return self.config.with_overrides(**overrides)

    def clear_learned_goal_value(self) -> None:
        self.learned_goal_value = None

    def clear_learned_commitment(self) -> None:
        self.learned_commitment = None

    def set_learned_goal_value(
        self,
        *,
        variant: str,
        weights: np.ndarray,
        bias: float = 0.0,
        feature_mean: Optional[np.ndarray] = None,
        feature_scale: Optional[np.ndarray] = None,
    ) -> None:
        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        mean_arr = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        scale_arr = None if feature_scale is None else np.asarray(feature_scale, dtype=np.float32).reshape(-1)
        self.learned_goal_value = CodaLearnedGoalValue(
            variant=str(variant),
            weights=weights_arr,
            bias=float(bias),
            feature_mean=mean_arr,
            feature_scale=scale_arr,
        )

    def set_learned_commitment(
        self,
        *,
        variant: str,
        weights: np.ndarray,
        bias: float = 0.0,
        feature_mean: Optional[np.ndarray] = None,
        feature_scale: Optional[np.ndarray] = None,
    ) -> None:
        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        mean_arr = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        scale_arr = None if feature_scale is None else np.asarray(feature_scale, dtype=np.float32).reshape(-1)
        self.learned_commitment = CodaLearnedCommitment(
            variant=str(variant),
            weights=weights_arr,
            bias=float(bias),
            feature_mean=mean_arr,
            feature_scale=scale_arr,
        )

    def step(self, executive_input: CodaExecutiveInput) -> CodaExecutiveOutput:
        tokens = np.asarray(executive_input.tokens, dtype=np.float32)
        entity = self._extract_entities(tokens, executive_input.metadata)
        entity = self._augment_memory_means(entity, executive_input)
        signals = self._derive_internal_signals(entity, executive_input)
        self._update_stall_trace(signals)

        wander_drive = self._compute_wander_drive(signals, executive_input)
        goal_scores = self._score_goals(entity, signals, executive_input)
        progress_score, best_goal_idx = self._compute_progress_score(goal_scores)
        self.last_debug["_signals_for_commitment"] = dict(signals)
        commitment = self._compute_commitment(progress_score, wander_drive)

        wander_action = self._wander_action()
        goal_action = self._goal_action(entity, best_goal_idx)
        raw_action = (1.0 - commitment) * wander_action + commitment * goal_action
        if best_goal_idx is None and signals["comfort_band"] >= 0.75:
            raw_action = (1.0 - wander_drive) * self.config.stop_drift * wander_action
        action = self._smooth_action(raw_action)
        stable_band_score = self._stable_band_score(executive_input)
        surprise = float(np.clip(executive_input.causal_surprise.get("structured_surprise", 0.0), 0.0, 1.0))
        if stable_band_score > 0.0:
            sig = float(np.clip(executive_input.interface_signals.get("region_signature", 0.0), 0.0, 1.0))
            self.state.general_signature = sig
            self.state.general_xy = np.asarray(
                [self._wall_x_for_agent(entity.get("agent_xy", None)), self._signature_to_world_y(sig)],
                dtype=np.float32,
            )
            self.state.general_kind = "region"
            self.state.general_hold = float(np.clip(0.65 * self.state.general_hold + 0.35 * stable_band_score, 0.0, 1.0))
        else:
            visible_scores: List[float] = []
            visible_candidates: List[Dict[str, Any]] = []
            for idx, candidate in enumerate(entity.get("candidates", [])):
                if idx >= len(goal_scores):
                    continue
                if str(candidate.get("source", "")) in {"memory_affordance", "memory_region"}:
                    continue
                visible_scores.append(float(goal_scores[idx]))
                visible_candidates.append(candidate)
            if visible_scores and visible_candidates:
                best_visible_idx = int(np.argmax(np.asarray(visible_scores, dtype=np.float32)))
                best_visible_score = float(visible_scores[best_visible_idx])
                best_visible_candidate = visible_candidates[best_visible_idx]
                self_evidence = self._self_target_evidence(signals)
                if best_visible_score > 0.30 and self_evidence > 0.10:
                    self.state.general_xy = np.asarray(
                        best_visible_candidate.get("xy", np.zeros((2,), dtype=np.float32)),
                        dtype=np.float32,
                    )
                    self.state.general_signature = 0.5
                    self.state.general_kind = "visible"
                    self.state.general_hold = float(np.clip(0.70 * self.state.general_hold + 0.30 * best_visible_score, 0.0, 1.0))
                else:
                    self.state.general_hold = float(np.clip(0.65 * self.state.general_hold, 0.0, 1.0))
            else:
                self.state.general_hold = float(np.clip(0.65 * self.state.general_hold, 0.0, 1.0))

        blocked_stall = (
            float(np.clip(executive_input.interface_signals.get("blocking_signal", 0.0), 0.0, 1.0)) > 0.5
            and float(np.clip(executive_input.interface_signals.get("control_ineffectiveness", 0.0), 0.0, 1.0)) > 0.72
            and float(np.clip(executive_input.interface_signals.get("means_feasibility", 0.0), 0.0, 1.0)) < 0.25
        )
        if blocked_stall:
            self.state.general_hold = float(np.clip(0.35 * self.state.general_hold, 0.0, 1.0))
            if self.state.general_hold < 0.08:
                self.state.general_xy = None
                self.state.general_signature = -1.0
                self.state.general_kind = ""
            commitment = float(np.clip(0.35 * commitment, 0.0, 1.0))
            action = self._escape_blocked_action(entity)

        blocked = float(np.clip(executive_input.interface_signals.get("blocking_signal", 0.0), 0.0, 1.0))
        feasible = float(np.clip(executive_input.interface_signals.get("means_feasibility", 0.0), 0.0, 1.0))
        control_bad = float(np.clip(executive_input.interface_signals.get("control_ineffectiveness", 0.0), 0.0, 1.0))
        region_occupancy = float(np.clip(executive_input.interface_signals.get("region_occupancy", 0.0), 0.0, 1.0))
        region_proximity = float(np.clip(executive_input.interface_signals.get("region_proximity", 0.0), 0.0, 1.0))
        region_entry_trace = float(np.clip(executive_input.interface_signals.get("region_entry_trace", 0.0), 0.0, 1.0))
        bridge_gate = bool(
            (region_occupancy > 0.50 or region_proximity > 0.22 or region_entry_trace > 0.12)
            and control_bad < 0.50
        )
        bridge_trigger = bool(blocked > 0.80 and feasible < 0.11 and control_bad > 0.20 and bridge_gate)
        if bridge_trigger:
            self.state.reacquire_mode = max(int(self.state.reacquire_mode), 4)
        elif self.state.reacquire_mode > 0:
            self.state.reacquire_mode = max(int(self.state.reacquire_mode) - 1, 0)

        if self.state.reacquire_mode > 0:
            candidates = list(entity.get("candidates", []))
            best_bridge_idx = None
            best_bridge_score = -1e9
            best_direct_idx = None
            best_direct_score = -1e9
            for idx, candidate in enumerate(candidates):
                source = str(candidate.get("source", ""))
                if source in {"memory_region", "memory_affordance"}:
                    bridge_score = (
                        1.10 * float(candidate.get("memory_support", 0.0))
                        + 0.95 * float(candidate.get("memory_match", 0.0))
                        + (0.20 if source == "memory_region" else 0.10)
                        + 0.06 * region_entry_trace
                        + 0.04 * max(0.0, region_occupancy - region_proximity)
                    )
                    if bridge_score > best_bridge_score:
                        best_bridge_score = bridge_score
                        best_bridge_idx = idx
                else:
                    direct_score = float(goal_scores[idx]) if idx < len(goal_scores) else 0.0
                    if direct_score > best_direct_score:
                        best_direct_score = direct_score
                        best_direct_idx = idx

            direct_allowed = bool(
                blocked < 0.70
                and feasible > 0.08
                and int(self.state.trap_hold_count) <= 0
            )
            use_direct = bool(
                direct_allowed
                and best_direct_idx is not None
                and (best_direct_score + 0.16 * region_entry_trace) >= (best_bridge_score + 0.04)
            )
            chosen_idx = best_direct_idx if use_direct else best_bridge_idx
            if chosen_idx is not None:
                self.state.reacquire_goal_idx = int(chosen_idx)
                if int(chosen_idx) >= len(goal_scores):
                    goal_scores.extend([0.0] * (int(chosen_idx) + 1 - len(goal_scores)))
                if not use_direct:
                    goal_scores[int(chosen_idx)] = float(max(float(goal_scores[int(chosen_idx)]), float(progress_score) + 0.45 + best_bridge_score))
                best_goal_idx = int(chosen_idx)
                goal_action = self._goal_action(entity, best_goal_idx)
                if float(np.linalg.norm(goal_action)) > 1e-6:
                    mix = 0.76 if use_direct else 0.82
                    action = np.clip((1.0 - mix) * action + mix * goal_action, -1.0, 1.0).astype(np.float32, copy=False)
                    commitment = float(np.clip(commitment + (0.20 if use_direct else 0.18), 0.0, 1.0))
                    progress_score = float(np.clip(progress_score + 0.08, 0.0, 1.0))
                    wander_drive = float(np.clip(wander_drive * (0.46 if use_direct else 0.43), 0.0, 1.0))
            else:
                self.state.reacquire_goal_idx = -1
        else:
            self.state.reacquire_goal_idx = -1

        if self.state.general_hold > 0.0:
            progress_score = float(np.clip(progress_score + 0.14 * self.state.general_hold, 0.0, 1.0))
            progress_score = float(np.clip(progress_score + 0.05 * surprise, 0.0, 1.0))
            commitment = float(np.clip(commitment + 0.12 * self.state.general_hold, 0.0, 1.0))
            action = np.clip((1.0 + 0.12 * self.state.general_hold) * action, -1.0, 1.0).astype(np.float32, copy=False)

        focus_scores: Dict[str, float] = {"wander": float(wander_drive)}
        for idx, score in enumerate(goal_scores):
            focus_scores[f"means_{idx}"] = float(score)
        selected_focus = max(focus_scores.items(), key=lambda kv: kv[1])[0] if focus_scores else "wander"

        trap = float(np.clip(blocked * control_bad * (1.0 - feasible), 0.0, 1.0))
        trap_correction_active = bool(
            trap > 0.16
            and self.state.reacquire_mode <= 0
            and not bridge_trigger
        )
        penalties = {
            key: float(val) * 0.86
            for key, val in dict(self.state.negative_affordance_penalties).items()
            if float(val) > 0.04
        }
        if trap_correction_active and str(selected_focus).startswith("means_"):
            self.state.trap_hold_count = min(int(self.state.trap_hold_count) + 1, 4)
            context_weight = float(np.clip(0.50 * blocked + 0.35 * control_bad + 0.15 * (1.0 - feasible), 0.0, 1.0))
            penalties[str(selected_focus)] = float(np.clip(penalties.get(str(selected_focus), 0.0) + 0.60 * context_weight, 0.0, 1.0))
            commitment = float(np.clip(commitment * (1.0 - 0.60 * context_weight), 0.0, 1.0))
            progress_score = float(np.clip(progress_score * (1.0 - 0.30 * context_weight), 0.0, 1.0))
            self.state.general_hold = float(np.clip(0.30 * float(self.state.general_hold), 0.0, 1.0))
        else:
            self.state.trap_hold_count = max(int(self.state.trap_hold_count) - 1, 0)
        for key, penalty in penalties.items():
            if key in focus_scores:
                focus_scores[key] = float(focus_scores[key]) * (1.0 - 0.68 * float(np.clip(penalty, 0.0, 1.0)))
        if focus_scores:
            selected_focus = max(focus_scores.items(), key=lambda kv: kv[1])[0]
        self.state.negative_affordance_penalties = penalties

        if trap_correction_active and action.shape[0] >= 2:
            agent_xy_arr = np.asarray(entity.get("agent_xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            sign = -1.0 if float(agent_xy_arr[0]) < 0.0 else 1.0
            normal = np.asarray([sign, 0.0], dtype=np.float32)
            proj = float(np.dot(action[:2], normal))
            threshold = 0.12 + 0.06 * trap
            if proj > threshold:
                action[:2] = action[:2] - proj * normal
                action[:2] += -0.12 * normal
        if int(self.state.trap_hold_count) >= 2 and action.shape[0] >= 2:
            agent_xy_arr = np.asarray(entity.get("agent_xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            sign = -1.0 if float(agent_xy_arr[0]) < 0.0 else 1.0
            normal = np.asarray([sign, 0.0], dtype=np.float32)
            action[:2] += -0.10 * normal
            action[0] = min(float(action[0]), -0.03 * sign)
            action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

        self.state.prev_stomach_level = float(signals["stomach_level"])
        self.state.prev_pain_level = float(signals["pain_level"])
        self.state.prev2_region_signature = float(self.state.prev_region_signature)
        self.state.prev_region_signature = float(np.clip(executive_input.interface_signals.get("region_signature", 0.0), 0.0, 1.0))
        self.state.prev_action = action.astype(np.float32, copy=True)
        self.state.timestep += 1

        debug = {
            "variant": str(self.config.variant),
            "signals": dict(signals),
            "interface_signals": dict(executive_input.interface_signals),
            "causal_memory": dict(executive_input.causal_memory),
            "focus_scores": dict(focus_scores),
            "best_goal_idx": None if best_goal_idx is None else int(best_goal_idx),
            "entity": {
                "agent_xy": None if entity["agent_xy"] is None else np.asarray(entity["agent_xy"], dtype=np.float32).tolist(),
                "num_candidates": int(len(entity["candidates"])),
            },
            "wander_action": np.asarray(wander_action, dtype=np.float32).tolist(),
            "goal_action": np.asarray(goal_action, dtype=np.float32).tolist(),
            "raw_action": np.asarray(raw_action, dtype=np.float32).tolist(),
            "stable_band_score": float(stable_band_score),
            "general_hold": float(self.state.general_hold),
            "general_signature": float(self.state.general_signature),
            "general_kind": str(self.state.general_kind),
        }
        self.last_debug = debug
        return CodaExecutiveOutput(
            action=action.astype(np.float32, copy=False),
            wander_drive=float(wander_drive),
            commitment=float(commitment),
            progress_score=float(progress_score),
            selected_focus=str(selected_focus),
            focus_scores=focus_scores,
            debug=debug,
        )

    def export_debug_snapshot(self) -> Dict[str, Any]:
        return {
            "config": {
                "variant": str(self.config.variant),
                "wander_speed": float(self.config.wander_speed),
                "goal_speed": float(self.config.goal_speed),
                "distance_scale": float(self.config.distance_scale),
                "learned_goal_value": None
                if self.learned_goal_value is None
                else {
                    "variant": str(self.learned_goal_value.variant),
                    "num_features": int(self.learned_goal_value.weights.shape[0]),
                },
                "learned_commitment": None
                if self.learned_commitment is None
                else {
                    "variant": str(self.learned_commitment.variant),
                    "num_features": int(self.learned_commitment.weights.shape[0]),
                },
            },
            "state": {
                "timestep": int(self.state.timestep),
                "wander_dir": np.asarray(self.state.wander_dir, dtype=np.float32).tolist(),
                "prev_action": np.asarray(self.state.prev_action, dtype=np.float32).tolist(),
                "prev_stomach_level": float(self.state.prev_stomach_level),
                "prev_pain_level": float(self.state.prev_pain_level),
                "stalled_food_steps": float(self.state.stalled_food_steps),
            },
            "last_debug": dict(self.last_debug),
        }

    def _sample_unit_vec(self) -> np.ndarray:
        v = self.rng.normal(size=2).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < EPS:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (v / n).astype(np.float32, copy=False)

    def _sigmoid(self, x: float) -> float:
        x_clip = float(np.clip(x, -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(-x_clip)))

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, max(0, 2 - arr.shape[0])))
        n = float(np.linalg.norm(arr[:2]))
        if n < EPS:
            return self._sample_unit_vec()
        return (arr[:2] / n).astype(np.float32, copy=False)

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.config.action_smoothing, 0.0, 1.0))
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            arr = np.pad(arr, (0, max(0, 2 - arr.shape[0])))
        smoothed = alpha * self.state.prev_action + (1.0 - alpha) * arr[:2]
        return np.clip(smoothed, -1.0, 1.0).astype(np.float32, copy=False)

    def _wander_action(self) -> np.ndarray:
        noise = 0.22 * self.rng.normal(size=2).astype(np.float32)
        self.state.wander_dir = self._normalize(0.92 * self.state.wander_dir + noise)
        return (self.state.wander_dir * float(self.config.wander_speed)).astype(np.float32, copy=False)

    def _goal_action(self, entity: Dict[str, Any], best_goal_idx: Optional[int]) -> np.ndarray:
        if best_goal_idx is None:
            return np.zeros((2,), dtype=np.float32)
        agent_xy = entity["agent_xy"]
        if agent_xy is None:
            return np.zeros((2,), dtype=np.float32)
        candidates: List[Dict[str, Any]] = entity["candidates"]
        if not (0 <= int(best_goal_idx) < len(candidates)):
            return np.zeros((2,), dtype=np.float32)
        candidate_xy = np.asarray(candidates[int(best_goal_idx)]["xy"], dtype=np.float32)
        delta = candidate_xy - np.asarray(agent_xy, dtype=np.float32)
        return (self._normalize(delta) * float(self.config.goal_speed)).astype(np.float32, copy=False)

    def _escape_blocked_action(self, entity: Dict[str, Any]) -> np.ndarray:
        agent_xy = entity.get("agent_xy", None)
        if agent_xy is None:
            return self._wander_action()
        agent_xy_arr = np.asarray(agent_xy, dtype=np.float32).reshape(-1)
        away_x = -1.0 if float(agent_xy_arr[0]) < 0.0 else 1.0
        vertical = 0.35 * float(self.rng.normal())
        return self._normalize(np.asarray([away_x, vertical], dtype=np.float32)) * float(self.config.goal_speed)

    def _stable_band_score(self, executive_input: CodaExecutiveInput) -> float:
        signals = dict(executive_input.interface_signals)
        blocking = float(np.clip(signals.get("blocking_signal", 0.0), 0.0, 1.0))
        occupancy = float(np.clip(signals.get("region_occupancy", 0.0), 0.0, 1.0))
        proximity = float(np.clip(signals.get("region_proximity", 0.0), 0.0, 1.0))
        signature = float(np.clip(signals.get("region_signature", 0.0), 0.0, 1.0))
        if blocking <= 0.5 or occupancy <= 0.5 or proximity >= 0.09:
            return 0.0
        same1 = 1.0 if self.state.prev_region_signature >= 0.0 and abs(signature - self.state.prev_region_signature) < 0.03 else 0.0
        same2 = 1.0 if self.state.prev2_region_signature >= 0.0 and abs(signature - self.state.prev2_region_signature) < 0.03 else 0.0
        extreme = 1.0 if (signature < 0.1 or signature > 0.9) else 0.0
        mid_trigger = 1.0 if (0.30 <= signature <= 0.45) else 0.0
        return float(np.clip(0.45 * same1 + 0.25 * same2 + 0.20 * extreme + 0.10 * mid_trigger, 0.0, 1.0))

    def _wall_x_for_agent(self, agent_xy: Any) -> float:
        if agent_xy is None:
            return -0.08
        arr = np.asarray(agent_xy, dtype=np.float32).reshape(-1)
        x = float(arr[0]) if arr.size > 0 else -0.08
        return -0.08 if x < 0.0 else 0.08

    def _compute_wander_drive(self, signals: Dict[str, float], executive_input: CodaExecutiveInput) -> float:
        novelty = float(np.clip(executive_input.causal_surprise.get("structured_surprise", 0.0), 0.0, 1.0))
        uncertainty = float(
            np.clip(
                executive_input.causal_surprise.get("belief_surprise", 0.0)
                + 0.5 * executive_input.causal_surprise.get("world_surprise", 0.0),
                0.0,
                1.0,
            )
        )
        memory_support = float(np.clip(executive_input.causal_beliefs.get("memory_support", 0.0), 0.0, 1.0))
        repeated_failure = float(np.clip(self.state.stalled_food_steps / 8.0, 0.0, 1.0))
        logit = (
            self.config.novelty_gain * novelty
            + self.config.uncertainty_gain * uncertainty
            - self.config.memory_support_gain * memory_support
            - self.config.repeated_failure_gain * repeated_failure
        )
        return self._sigmoid(logit)

    def _compute_progress_score(self, goal_scores: List[float]) -> Tuple[float, Optional[int]]:
        if not goal_scores:
            return 0.0, None
        arr = np.asarray(goal_scores, dtype=np.float32)
        idx = int(np.argmax(arr))
        return float(arr[idx]), idx

    def _self_target_evidence(self, signals: Dict[str, float]) -> float:
        return float(
            np.clip(
                max(
                    float(signals.get("affordance_stomach_support", 0.0)),
                    float(signals.get("affordance_stomach_confidence", 0.0)),
                    float(signals.get("affordance_stomach_match", 0.0)),
                    float(signals.get("affordance_pain_support", 0.0)),
                    float(signals.get("affordance_pain_confidence", 0.0)),
                    float(signals.get("affordance_pain_match", 0.0)),
                ),
                0.0,
                1.0,
            )
        )

    def _compute_commitment(self, progress_score: float, wander_drive: float) -> float:
        if self.learned_commitment is not None and self.last_debug.get("_commitment_features") is not None:
            feat = np.asarray(self.last_debug["_commitment_features"], dtype=np.float32).reshape(-1)
            if self.learned_commitment.feature_mean is not None and self.learned_commitment.feature_scale is not None:
                scale = np.where(np.abs(self.learned_commitment.feature_scale) < 1e-6, 1.0, self.learned_commitment.feature_scale)
                feat = (feat - self.learned_commitment.feature_mean) / scale
            linear = float(np.dot(self.learned_commitment.weights, feat) + self.learned_commitment.bias)
            return self._sigmoid(linear)
        signals = self.last_debug.get("_signals_for_commitment", {})
        comfort_band = float(np.clip(signals.get("comfort_band", 0.0), 0.0, 1.0))
        high_need = float(np.clip(signals.get("high_need", 0.0), 0.0, 1.0))
        logit = (
            self.config.progress_gain * float(progress_score)
            - float(wander_drive)
            - self.config.commitment_comfort_suppression * comfort_band
            - self.config.commitment_high_need_suppression * high_need
            + float(self.config.commitment_bias)
        )
        return self._sigmoid(logit)

    def _update_stall_trace(self, signals: Dict[str, float]) -> None:
        low_need = float(signals["low_need"])
        prev_low_need = float(max(0.0, 1.0 - self.state.prev_stomach_level))
        improving = low_need < max(0.0, prev_low_need - 0.01)
        if improving:
            self.state.stalled_food_steps = float(max(0.0, 0.6 * self.state.stalled_food_steps - 0.5))
        else:
            self.state.stalled_food_steps = float(np.clip(self.state.stalled_food_steps + 1.0, 0.0, 64.0))

    def _derive_internal_signals(
        self,
        entity: Dict[str, Any],
        executive_input: CodaExecutiveInput,
    ) -> Dict[str, float]:
        stomach = entity["stomach"]
        pain = entity["pain"]
        stomach_level = float(stomach.get("level", executive_input.causal_surprise.get("stomach_level", 0.0)))
        low_need = float(stomach.get("low_drive", max(0.0, 1.0 - stomach_level)))
        high_need = float(stomach.get("high_drive", 0.0))
        comfort_band = float(stomach.get("comfort_band", 0.0))
        stomach_stress = float(stomach.get("stress", max(low_need, high_need)))
        pain_level = float(pain.get("level", 0.0))
        pain_from_stomach = float(pain.get("from_stomach", executive_input.causal_surprise.get("pain_from_stomach", 0.0)))
        pain_trend = float(np.clip(pain_level - self.state.prev_pain_level, -1.0, 1.0))
        stomach_delta = float(np.clip(stomach_level - self.state.prev_stomach_level, -1.0, 1.0))
        relief_causal_support = float(
            np.clip(
                executive_input.causal_beliefs.get("memory_support", 0.0)
                + self.config.goal_causal_support_gain * pain_from_stomach,
                0.0,
                1.0,
            )
        )
        support_context = float(
            np.clip(
                self.config.goal_support_context_gate_gain
                * (
                    0.65 * low_need
                    + 0.35 * pain_level * pain_from_stomach
                    + 0.25 * max(0.0, 1.0 - comfort_band)
                    - 0.60 * comfort_band
                    - 0.35 * high_need
                ),
                0.0,
                1.0,
            )
        )
        effective_relief_causal_support = float(np.clip(relief_causal_support * support_context, 0.0, 1.0))
        affordance_support = float(
            np.clip(
                executive_input.causal_memory.get(
                    "affordance_support",
                    executive_input.causal_memory.get("wall_method_support", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        best_affordance_signature = float(
            np.clip(
                executive_input.causal_memory.get(
                    "best_affordance_signature",
                    executive_input.causal_memory.get("best_known_region_signature", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        best_affordance_confidence = float(
            np.clip(
                executive_input.causal_memory.get(
                    "best_affordance_confidence",
                    executive_input.causal_memory.get("best_known_region_support", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        best_affordance_match = float(
            np.clip(
                executive_input.causal_memory.get(
                    "best_affordance_match",
                    executive_input.causal_memory.get("best_known_region_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_env_support = float(
            np.clip(
                executive_input.causal_beliefs.get("current_env_support", executive_input.causal_memory.get("best_affordance_env_support", 0.0)),
                0.0,
                1.0,
            )
        )
        affordance_env_confidence = float(
            np.clip(
                executive_input.causal_beliefs.get("current_env_confidence", executive_input.causal_memory.get("best_affordance_env_confidence", 0.0)),
                0.0,
                1.0,
            )
        )
        affordance_env_match = float(
            np.clip(
                executive_input.causal_beliefs.get("current_env_match", executive_input.causal_memory.get("best_affordance_env_match", 0.0)),
                0.0,
                1.0,
            )
        )
        affordance_stomach_support = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_stomach_support",
                    executive_input.causal_memory.get("best_affordance_self_stomach_support", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_stomach_confidence = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_stomach_confidence",
                    executive_input.causal_memory.get("best_affordance_self_stomach_confidence", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_stomach_match = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_stomach_match",
                    executive_input.causal_memory.get("best_affordance_self_stomach_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_pain_support = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_pain_support",
                    executive_input.causal_memory.get("best_affordance_self_pain_support", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_pain_confidence = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_pain_confidence",
                    executive_input.causal_memory.get("best_affordance_self_pain_confidence", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        affordance_pain_match = float(
            np.clip(
                executive_input.causal_beliefs.get(
                    "current_self_pain_match",
                    executive_input.causal_memory.get("best_affordance_self_pain_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        satiety_room = 1.0 - self._sigmoid((stomach_level - self.config.satiety_center) / max(self.config.satiety_width, 1e-3))
        relief_need = float(
            np.clip(
                self.config.low_need_gain * low_need
                + 0.35 * pain_level * pain_from_stomach
                - self.config.high_need_food_penalty * high_need,
                0.0,
                1.0,
            )
        )
        return {
            "stomach_level": float(np.clip(stomach_level, 0.0, 1.0)),
            "stomach_delta": float(stomach_delta),
            "low_need": float(np.clip(low_need, 0.0, 1.0)),
            "high_need": float(np.clip(high_need, 0.0, 1.0)),
            "comfort_band": float(np.clip(comfort_band, 0.0, 1.0)),
            "stomach_stress": float(np.clip(stomach_stress, 0.0, 1.0)),
            "pain_level": float(np.clip(pain_level, 0.0, 1.0)),
            "pain_trend": float(pain_trend),
            "pain_from_stomach": float(np.clip(pain_from_stomach, 0.0, 1.0)),
            "relief_causal_support": float(relief_causal_support),
            "support_context": float(support_context),
            "effective_relief_causal_support": float(effective_relief_causal_support),
            "affordance_support": float(affordance_support),
            "best_affordance_signature": float(best_affordance_signature),
            "best_affordance_confidence": float(best_affordance_confidence),
            "best_affordance_match": float(best_affordance_match),
            "affordance_env_support": float(affordance_env_support),
            "affordance_env_confidence": float(affordance_env_confidence),
            "affordance_env_match": float(affordance_env_match),
            "affordance_stomach_support": float(affordance_stomach_support),
            "affordance_stomach_confidence": float(affordance_stomach_confidence),
            "affordance_stomach_match": float(affordance_stomach_match),
            "affordance_pain_support": float(affordance_pain_support),
            "affordance_pain_confidence": float(affordance_pain_confidence),
            "affordance_pain_match": float(affordance_pain_match),
            "wall_method_support": float(affordance_support),
            "best_known_region_signature": float(best_affordance_signature),
            "best_known_region_support": float(best_affordance_confidence),
            "best_known_region_match": float(best_affordance_match),
            "satiety_room": float(np.clip(satiety_room, 0.0, 1.0)),
            "relief_need": float(relief_need),
        }

    def commitment_feature_vector(
        self,
        *,
        progress_score: float,
        wander_drive: float,
        signals: Dict[str, float],
        executive_input: CodaExecutiveInput,
        variant: str = "pooled",
    ) -> np.ndarray:
        base = [
            float(np.clip(progress_score, 0.0, 1.0)),
            float(np.clip(wander_drive, 0.0, 1.0)),
            float(np.clip(signals.get("relief_need", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("comfort_band", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("high_need", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("pain_level", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("pain_from_stomach", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("effective_relief_causal_support", signals.get("relief_causal_support", 0.0)), 0.0, 1.0)),
        ]
        key = str(variant).strip().lower()
        if key == "pooled":
            return np.asarray(base, dtype=np.float32)
        if key == "comfort":
            return np.asarray(
                base
                + [
                    float(np.clip(signals.get("low_need", 0.0), 0.0, 1.0)),
                    float(np.clip(signals.get("satiety_room", 0.0), 0.0, 1.0)),
                    float(np.clip(signals.get("stomach_stress", 0.0), 0.0, 1.0)),
                    float(np.clip(signals.get("support_context", 0.0), 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
        if key == "with_memory":
            return np.asarray(
                base
                + [
                    float(np.clip(executive_input.causal_beliefs.get("memory_support", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_beliefs.get("delayed_env_change", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_surprise.get("structured_surprise", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_surprise.get("belief_surprise", 0.0), 0.0, 1.0)),
                    float(np.clip(signals.get("support_context", 0.0), 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
        return np.asarray(base, dtype=np.float32)

    def _score_goals(
        self,
        entity: Dict[str, Any],
        signals: Dict[str, float],
        executive_input: CodaExecutiveInput,
    ) -> List[float]:
        agent_xy = entity["agent_xy"]
        candidates: List[Dict[str, Any]] = entity["candidates"]
        if agent_xy is None or not candidates:
            return []
        agent_xy_arr = np.asarray(agent_xy, dtype=np.float32)
        if self.learned_goal_value is not None:
            out: List[float] = []
            for candidate in candidates:
                feature_vec = self.goal_feature_vector(
                    entity=entity,
                    goal=candidate,
                    signals=signals,
                    executive_input=executive_input,
                    variant=str(self.learned_goal_value.variant),
                )
                feat = np.asarray(feature_vec, dtype=np.float32).reshape(-1)
                if self.learned_goal_value.feature_mean is not None and self.learned_goal_value.feature_scale is not None:
                    scale = np.where(np.abs(self.learned_goal_value.feature_scale) < 1e-6, 1.0, self.learned_goal_value.feature_scale)
                    feat = (feat - self.learned_goal_value.feature_mean) / scale
                linear = float(np.dot(self.learned_goal_value.weights, feat) + self.learned_goal_value.bias)
                out.append(self._sigmoid(linear))
            return out
        out: List[float] = []
        self_evidence = self._self_target_evidence(signals)
        for candidate in candidates:
            candidate_xy = np.asarray(candidate["xy"], dtype=np.float32)
            dist = float(np.linalg.norm(candidate_xy - agent_xy_arr))
            reachability = float(np.exp(-float(self.config.distance_scale) * dist))
            memory_region_bonus = 0.0
            source = str(candidate.get("source", ""))
            is_memory_candidate = source in {"memory_region", "memory_affordance"}
            if is_memory_candidate:
                memory_region_bonus = float(
                    np.clip(
                        0.45 * float(candidate.get("memory_support", 0.0))
                        + 0.55 * float(candidate.get("memory_match", 0.0)),
                        0.0,
                        1.0,
                    )
                )
            usefulness = float(
                np.clip(
                    signals["relief_need"] * signals["satiety_room"]
                    + signals.get("effective_relief_causal_support", signals["relief_causal_support"])
                    + memory_region_bonus
                    + self.config.pain_gain * max(0.0, signals["pain_level"] * signals["pain_from_stomach"]),
                    0.0,
                    1.5,
                )
            )
            variant_bonus = 0.0
            if str(self.config.variant) == "wander_heavy":
                variant_bonus = 0.10 * executive_input.causal_surprise.get("structured_surprise", 0.0)
            elif str(self.config.variant) == "progress_mixed_goal_pursuit":
                variant_bonus = 0.15 * executive_input.causal_beliefs.get("delayed_env_change", 0.0)
            elif str(self.config.variant) == "pain_aware_comfort_controller":
                variant_bonus = 0.20 * signals["pain_from_stomach"] + 0.15 * signals["low_need"]
            score = float(np.clip(reachability * (usefulness + variant_bonus), 0.0, 1.0))
            if not is_memory_candidate:
                score *= float(np.clip((self_evidence - 0.10) / 0.25, 0.0, 1.0))
            out.append(score)
        return out

    def goal_feature_vector(
        self,
        *,
        entity: Dict[str, Any],
        goal: Dict[str, Any],
        signals: Dict[str, float],
        executive_input: CodaExecutiveInput,
        variant: str = "pooled",
    ) -> np.ndarray:
        agent_xy = entity.get("agent_xy", None)
        goal_xy = np.asarray(goal.get("xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if agent_xy is None:
            delta = np.zeros((2,), dtype=np.float32)
            dist = 0.0
        else:
            delta = goal_xy[:2] - np.asarray(agent_xy, dtype=np.float32).reshape(-1)[:2]
            dist = float(np.linalg.norm(delta))
        inv_dist = 1.0 / (1.0 + dist)
        reachability = float(np.exp(-float(self.config.distance_scale) * dist))
        unit_delta = self._normalize(delta if delta.shape[0] >= 2 else np.zeros((2,), dtype=np.float32))
        prev_action = np.asarray(self.state.prev_action, dtype=np.float32).reshape(-1)
        prev_mag = float(np.linalg.norm(prev_action[:2])) if prev_action.size >= 2 else 0.0
        prev_align = 0.0
        if prev_mag > EPS:
            prev_align = float(np.dot(unit_delta[:2], prev_action[:2] / max(prev_mag, EPS)))
        base = [
            reachability,
            inv_dist,
            float(np.clip(signals.get("relief_need", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("satiety_room", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("relief_causal_support", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("pain_level", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("pain_from_stomach", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("low_need", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("high_need", 0.0), 0.0, 1.0)),
            float(np.clip(signals.get("comfort_band", 0.0), 0.0, 1.0)),
        ]
        variant_key = str(variant).strip().lower()
        if variant_key == "pooled":
            return np.asarray(base, dtype=np.float32)
        if variant_key == "relational":
            return np.asarray(
                base
                + [
                    float(np.clip(unit_delta[0], -1.0, 1.0)),
                    float(np.clip(unit_delta[1], -1.0, 1.0)),
                    float(np.clip(abs(delta[0]), 0.0, 4.0)),
                    float(np.clip(abs(delta[1]), 0.0, 4.0)),
                    float(np.clip(prev_align, -1.0, 1.0)),
                    float(np.clip(prev_mag, 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
        if variant_key == "with_memory":
            return np.asarray(
                base
                + [
                    float(np.clip(executive_input.causal_beliefs.get("memory_support", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_beliefs.get("delayed_env_change", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_surprise.get("structured_surprise", 0.0), 0.0, 1.0)),
                    float(np.clip(executive_input.causal_surprise.get("belief_surprise", 0.0), 0.0, 1.0)),
                    float(np.clip(prev_align, -1.0, 1.0)),
                ],
                dtype=np.float32,
            )
        return np.asarray(base, dtype=np.float32)

    def _extract_entities(self, tokens: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        arr = np.asarray(tokens, dtype=np.float32)
        out: Dict[str, Any] = {
            "agent_xy": None,
            "candidates": [],
            "stomach": {},
            "pain": {},
        }
        if arr.ndim != 2 or arr.shape[0] == 0:
            return out
        num_types = int(metadata.get("num_types", self.config.num_types))
        geom_start = 1 + max(1, num_types)
        if arr.shape[1] < geom_start:
            return out
        gates = np.clip(arr[:, 0], 0.0, 1.0)
        type_logits = arr[:, 1 : 1 + num_types]
        type_idx = np.argmax(type_logits, axis=-1) if type_logits.size > 0 else np.zeros((arr.shape[0],), dtype=np.int32)
        present = gates >= 0.5

        agent_mask = present & (type_idx == int(self.config.type_agent))
        if np.any(agent_mask) and arr.shape[1] >= geom_start + 2:
            best_agent = int(np.argmax(gates * agent_mask.astype(np.float32)))
            out["agent_xy"] = arr[best_agent, geom_start : geom_start + 2].astype(np.float32, copy=False)

        candidate_mask = (
            present
            & (type_idx != int(self.config.type_agent))
            & (type_idx != int(self.config.type_condition))
            & (type_idx != int(self.config.type_empty))
        )
        for idx in np.where(candidate_mask)[0].tolist():
            xy = arr[idx, geom_start : geom_start + 2] if arr.shape[1] >= geom_start + 2 else np.zeros((2,), dtype=np.float32)
            out["candidates"].append(
                {
                    "slot": int(idx),
                    "xy": np.asarray(xy, dtype=np.float32),
                    "type_idx": int(type_idx[idx]),
                }
            )

        cond_mask = present & (type_idx == int(self.config.type_condition))
        cond_slots = [int(i) for i in np.where(cond_mask)[0].tolist()]
        cond_slots.sort()
        if self.config.condition_start_index is not None:
            base = int(self.config.condition_start_index)
            cond_slots = [base + i for i in range(len(cond_slots))]
        if cond_slots:
            stomach_slot = int(cond_slots[0])
            if arr.shape[1] >= geom_start + 6:
                out["stomach"] = {
                    "slot": stomach_slot,
                    "level": float(np.clip(arr[stomach_slot, geom_start + 0], 0.0, 1.0)),
                    "delta": float(np.clip(arr[stomach_slot, geom_start + 1], -1.0, 1.0)),
                    "low_drive": float(np.clip(arr[stomach_slot, geom_start + 2], 0.0, 1.0)),
                    "high_drive": float(np.clip(arr[stomach_slot, geom_start + 3], 0.0, 1.0)),
                    "comfort_band": float(np.clip(arr[stomach_slot, geom_start + 4], 0.0, 1.0)),
                    "stress": float(np.clip(arr[stomach_slot, geom_start + 5], 0.0, 1.0)),
                }
        if len(cond_slots) > 1:
            pain_slot = int(cond_slots[1])
            if arr.shape[1] >= geom_start + 6:
                out["pain"] = {
                    "slot": pain_slot,
                    "level": float(np.clip(arr[pain_slot, geom_start + 0], 0.0, 1.0)),
                    "from_stomach": float(np.clip(arr[pain_slot, geom_start + 1], 0.0, 1.0)),
                    "from_food": float(np.clip(arr[pain_slot, geom_start + 2], 0.0, 1.0)),
                    "from_context": float(np.clip(arr[pain_slot, geom_start + 3], 0.0, 1.0)),
                    "unknown": float(np.clip(arr[pain_slot, geom_start + 4], 0.0, 1.0)),
                    "persistence": float(np.clip(arr[pain_slot, geom_start + 5], 0.0, 1.0)),
                }
        return out

    def _augment_memory_means(
        self,
        entity: Dict[str, Any],
        executive_input: CodaExecutiveInput,
    ) -> Dict[str, Any]:
        agent_xy = entity.get("agent_xy", None)
        if agent_xy is None:
            return entity
        affordance_support = float(
            np.clip(
                executive_input.causal_memory.get(
                    "affordance_support",
                    executive_input.causal_memory.get("wall_method_support", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        agent_xy_arr = np.asarray(agent_xy, dtype=np.float32).reshape(-1)
        wall_x = -0.08 if float(agent_xy_arr[0]) < 0.0 else 0.08
        augmented = dict(entity)
        candidates = list(entity.get("candidates", []))
        next_slot = -1
        if affordance_support > 0.05:
            for rank in range(2):
                support = float(
                    np.clip(
                        executive_input.causal_memory.get(
                            f"known_affordance_confidence_{rank}",
                            executive_input.causal_memory.get(f"known_region_support_{rank}", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                signature = float(
                    np.clip(
                        executive_input.causal_memory.get(
                            f"known_affordance_signature_{rank}",
                            executive_input.causal_memory.get(f"known_region_signature_{rank}", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                match = float(
                    np.clip(
                        executive_input.causal_memory.get(
                            f"known_affordance_match_{rank}",
                            executive_input.causal_memory.get(f"known_region_match_{rank}", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                if support <= 0.05:
                    continue
                candidates.append(
                    {
                        "slot": int(next_slot),
                        "xy": np.asarray([wall_x, self._signature_to_world_y(signature)], dtype=np.float32),
                        "type_idx": -1,
                        "source": "memory_affordance",
                        "memory_support": float(support * affordance_support),
                        "memory_match": float(match),
                        "memory_signature": float(signature),
                    }
                )
                next_slot -= 1
        if self.state.general_xy is not None and self.state.general_hold > 0.08:
            candidates.append(
                {
                    "slot": int(-100),
                    "xy": np.asarray(self.state.general_xy, dtype=np.float32),
                    "type_idx": -1,
                    "source": "memory_region",
                    "memory_support": float(np.clip(0.65 * self.state.general_hold + 0.25, 0.0, 1.0)),
                    "memory_match": float(np.clip(self.state.general_hold, 0.0, 1.0)),
                    "memory_signature": float(np.clip(self.state.general_signature, 0.0, 1.0)),
                    "memory_kind": str(self.state.general_kind),
                }
            )
        augmented["candidates"] = candidates
        return augmented

    def _signature_to_world_y(self, signature: float) -> float:
        sig = float(np.clip(signature, 0.0, 1.0))
        return float(np.clip((2.0 * sig - 1.0) * 0.8, -0.8, 0.8))
