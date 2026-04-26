from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-6
SEQUENCE_TRACE_DIM = 16
AFFORDANCE_BIN_COUNT = 8
AFFORDANCE_TARGETS: Tuple[str, ...] = ("env", "self_stomach", "self_pain")


@dataclass
class CodaCausalConfig:
    """
    Live CODA causal-model configuration.

    The live model should run a single implementation path. This config is for
    stable numeric parameters, not strategy switching.
    """

    gate_present_threshold: float = 0.5
    sequence_window: int = 12
    max_memories: int = 256
    retrieval_topk: int = 4
    memory_decay: float = 0.995
    belief_decay: float = 0.96
    novelty_weight: float = 0.35
    significance_threshold: float = 0.42
    novelty_threshold: float = 0.28
    memory_similarity: str = "cosine"
    interaction_trace_decay: float = 0.985
    regime_trace_decay: float = 0.97
    cause_progress_decay: float = 0.995
    interaction_trace_gain: float = 0.28
    regime_trace_gain: float = 0.14
    memory_trace_gain: float = 0.08
    surprise_trace_gain: float = 0.06
    inverse_xy_scale: float = 8.0
    inverse_rls_forgetting: float = 0.995
    inverse_rls_init_scale: float = 5.0
    coherent_write_retrieval_penalty: float = 0.35
    coherent_write_force_coherent_threshold: float = 0.20
    coherent_write_force_coverage_threshold: float = 0.18
    coherent_write_force_harm_threshold: float = 0.20
    coherent_write_force_change_threshold: float = 0.08
    coherent_write_force_retrieved_max: float = 0.90
    reserved_scalar_keys: Tuple[str, ...] = (
        "world_surprise",
        "coherence",
        "regime_signal",
        "env_change_observed",
        "interaction_event",
        "consumed_token",
        "delayed_change_target",
        "belief_probe_target",
        "inv_actual_error",
        "inv_pred_error",
        "coherence_score",
        "coverage_score",
        "coherent_surprise_score",
        "stomach_level",
        "stomach_delta",
        "stomach_stress",
        "pain_level",
        "pain_delta",
        "pain_from_stomach",
        "region_occupancy",
        "region_entry",
        "region_exit",
        "region_signature",
        "region_proximity",
        "region_dwell",
        "region_entry_trace",
        "trajectory_progress",
        "blocking_signal",
        "control_ineffectiveness",
        "means_feasibility",
        "sequence_state",
        "rollout_mismatch",
        "trajectory_progress_teacher",
        "means_feasibility_teacher",
    )

    def with_overrides(self, **overrides: Any) -> "CodaCausalConfig":
        return replace(self, **overrides)


@dataclass
class CodaCausalTransition:
    prev_tokens: np.ndarray
    action: np.ndarray
    curr_tokens: np.ndarray
    predicted_tokens: Optional[np.ndarray] = None
    scalar_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodaCausalMemoryEntry:
    key: np.ndarray
    value: np.ndarray
    strength: float
    age: int = 0
    write_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodaCausalStepOutput:
    evidence_vector: np.ndarray
    evidence_dict: Dict[str, float]
    belief_dict: Dict[str, float]
    surprise_dict: Dict[str, float]
    retrieval_scores: np.ndarray
    memory_hints: Dict[str, float]
    write_decision: bool


@dataclass
class CodaCausalState:
    belief_vector: np.ndarray
    recent_evidence: Deque[np.ndarray]
    interaction_trace: float = 0.0
    regime_trace: float = 0.0
    cause_progress: float = 0.0
    timestep: int = 0


BELIEF_CHANNELS: Tuple[str, ...] = (
    "novelty",
    "interaction_relevance",
    "immediate_env_change",
    "delayed_env_change",
    "regime_shift",
    "causal_significance",
    "memory_support",
    "belief_probe",
)

CURRENT_CAUSAL_STATE_KEYS: Tuple[str, ...] = (
    "current_env_confidence",
    "current_env_match",
    "current_env_support",
    "current_self_stomach_confidence",
    "current_self_stomach_match",
    "current_self_stomach_support",
    "current_self_pain_confidence",
    "current_self_pain_match",
    "current_self_pain_support",
)

MEMORY_STATE_KEYS: Tuple[str, ...] = (
    "env_confidence",
    "env_match",
    "env_support",
    "self_stomach_confidence",
    "self_stomach_match",
    "self_stomach_support",
    "self_pain_confidence",
    "self_pain_match",
    "self_pain_support",
    "causal_strength",
)


def build_causal_config(_preset: Optional[str] = None, **overrides: Any) -> CodaCausalConfig:
    cfg = CodaCausalConfig()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


class CodaCausalModel:
    """
    Lightweight causal-model scaffold centered on:
    - transition evidence
    - sparse causal memories
    - evolving belief state
    - structured surprise

    Reserved scalar signals:
    - world_surprise
    - coherence
    - regime_signal
    - env_change_observed
    - interaction_event
    - consumed_token
    - delayed_change_target
    - belief_probe_target

    These are optional. The scaffold stays usable if only token transitions and
    actions are available, and it becomes more informative as more signals are fed.
    """

    def __init__(self, config: Optional[CodaCausalConfig] = None):
        self.config = config or CodaCausalConfig()
        self.memory_bank: List[CodaCausalMemoryEntry] = []
        self.affordance_bins: Dict[str, np.ndarray] = {
            target: np.zeros((AFFORDANCE_BIN_COUNT,), dtype=np.float32) for target in AFFORDANCE_TARGETS
        }
        self.inverse_weights: Optional[np.ndarray] = None
        self.inverse_cov: Optional[np.ndarray] = None
        self.state = CodaCausalState(
            belief_vector=np.zeros((len(BELIEF_CHANNELS),), dtype=np.float32),
            recent_evidence=deque(maxlen=max(1, self.config.sequence_window)),
            timestep=0,
        )

    def reset(self) -> None:
        self.memory_bank.clear()
        for target in AFFORDANCE_TARGETS:
            self.affordance_bins[target].fill(0.0)
        self.inverse_weights = None
        self.inverse_cov = None
        self.state.belief_vector.fill(0.0)
        self.state.recent_evidence.clear()
        self.state.interaction_trace = 0.0
        self.state.regime_trace = 0.0
        self.state.cause_progress = 0.0
        self.state.timestep = 0

    def _memory_state_from_current(
        self,
        current_state: Dict[str, float],
        evidence_dict: Dict[str, float],
    ) -> np.ndarray:
        value = np.zeros((len(MEMORY_STATE_KEYS),), dtype=np.float32)
        value[0] = float(np.clip(current_state.get("current_env_confidence", 0.0), 0.0, 1.0))
        value[1] = float(np.clip(current_state.get("current_env_match", 0.0), 0.0, 1.0))
        value[2] = float(np.clip(current_state.get("current_env_support", 0.0), 0.0, 1.0))
        value[3] = float(np.clip(current_state.get("current_self_stomach_confidence", 0.0), 0.0, 1.0))
        value[4] = float(np.clip(current_state.get("current_self_stomach_match", 0.0), 0.0, 1.0))
        value[5] = float(np.clip(current_state.get("current_self_stomach_support", 0.0), 0.0, 1.0))
        value[6] = float(np.clip(current_state.get("current_self_pain_confidence", 0.0), 0.0, 1.0))
        value[7] = float(np.clip(current_state.get("current_self_pain_match", 0.0), 0.0, 1.0))
        value[8] = float(np.clip(current_state.get("current_self_pain_support", 0.0), 0.0, 1.0))
        value[9] = float(
            np.clip(
                max(
                    value[0] * value[1],
                    value[3] * value[4],
                    value[6] * value[7],
                    evidence_dict.get("coherent_surprise_score", 0.0),
                    evidence_dict.get("condition_harm", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        return value

    def _empty_memory_state(self) -> np.ndarray:
        return np.zeros((len(MEMORY_STATE_KEYS),), dtype=np.float32)

    def _current_state_from_memory(self, retrieved_value: np.ndarray) -> Dict[str, float]:
        if retrieved_value.size < len(MEMORY_STATE_KEYS):
            return {}
        return {
            "current_env_confidence": float(np.clip(retrieved_value[0], 0.0, 1.0)),
            "current_env_match": float(np.clip(retrieved_value[1], 0.0, 1.0)),
            "current_env_support": float(np.clip(retrieved_value[2], 0.0, 1.0)),
            "current_self_stomach_confidence": float(np.clip(retrieved_value[3], 0.0, 1.0)),
            "current_self_stomach_match": float(np.clip(retrieved_value[4], 0.0, 1.0)),
            "current_self_stomach_support": float(np.clip(retrieved_value[5], 0.0, 1.0)),
            "current_self_pain_confidence": float(np.clip(retrieved_value[6], 0.0, 1.0)),
            "current_self_pain_match": float(np.clip(retrieved_value[7], 0.0, 1.0)),
            "current_self_pain_support": float(np.clip(retrieved_value[8], 0.0, 1.0)),
            "memory_causal_strength": float(np.clip(retrieved_value[9], 0.0, 1.0)),
        }

    def clone_config(self, **overrides: Any) -> CodaCausalConfig:
        return self.config.with_overrides(**overrides)

    def step(self, transition: CodaCausalTransition) -> CodaCausalStepOutput:
        evidence_dict, evidence_vector = self._encode_transition_evidence(transition)
        retrieval_scores, retrieved_value = self._retrieve(evidence_vector)
        belief_vector = self._update_beliefs(evidence_dict, retrieved_value)
        surprise_dict = self._compute_structured_surprise(evidence_dict, belief_vector, retrieval_scores)
        write_decision = self._should_write_memory(evidence_dict, surprise_dict, retrieval_scores)
        if write_decision:
            self._write_memory(evidence_vector, evidence_dict, belief_vector, transition)
        self._update_affordance_memory(evidence_dict)
        memory_hints = self._derive_memory_hints(evidence_dict)
        self._update_inverse_model(transition)
        self._advance_time(evidence_vector)
        belief_dict = {name: float(belief_vector[idx]) for idx, name in enumerate(BELIEF_CHANNELS)}
        belief_dict.update(self._derive_current_causal_state(evidence_dict))
        return CodaCausalStepOutput(
            evidence_vector=evidence_vector,
            evidence_dict=evidence_dict,
            belief_dict=belief_dict,
            surprise_dict=surprise_dict,
            retrieval_scores=retrieval_scores,
            memory_hints=memory_hints,
            write_decision=bool(write_decision),
        )

    def export_debug_snapshot(self) -> Dict[str, Any]:
        memories = []
        for entry in self.memory_bank:
            memories.append(
                {
                    "strength": float(entry.strength),
                    "age": int(entry.age),
                    "write_count": int(entry.write_count),
                    "metadata": dict(entry.metadata),
                    "value": entry.value.astype(np.float32, copy=False).tolist(),
                }
            )
        return {
            "belief": {
                name: float(self.state.belief_vector[idx])
                for idx, name in enumerate(BELIEF_CHANNELS)
            },
            "memory_count": int(len(self.memory_bank)),
            "memories": memories,
            "recent_evidence_count": int(len(self.state.recent_evidence)),
            "timestep": int(self.state.timestep),
            "config": {
                "sequence_window": int(self.config.sequence_window),
                "max_memories": int(self.config.max_memories),
                "retrieval_topk": int(self.config.retrieval_topk),
            },
        }

    def _encode_transition_evidence(
        self,
        transition: CodaCausalTransition,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        prev_tokens = np.asarray(transition.prev_tokens, dtype=np.float32)
        curr_tokens = np.asarray(transition.curr_tokens, dtype=np.float32)
        pred_tokens = (
            np.asarray(transition.predicted_tokens, dtype=np.float32)
            if transition.predicted_tokens is not None
            else None
        )
        action = np.asarray(transition.action, dtype=np.float32).reshape(-1)
        prev_summary = self._summarize_tokens(prev_tokens)
        curr_summary = self._summarize_tokens(curr_tokens)

        overlap_mask = prev_summary["present_mask"] & curr_summary["present_mask"]
        geom_delta_mean = 0.0
        geom_delta_xy = 0.0
        if np.any(overlap_mask):
            prev_geom = prev_tokens[overlap_mask, prev_summary["geom_slice"]]
            curr_geom = curr_tokens[overlap_mask, curr_summary["geom_slice"]]
            geom_delta = np.abs(curr_geom - prev_geom)
            geom_delta_mean = float(np.mean(geom_delta))
            geom_delta_xy = float(np.mean(np.linalg.norm(curr_geom[:, :2] - prev_geom[:, :2], axis=1)))

        appear_count = float(np.sum((~prev_summary["present_mask"]) & curr_summary["present_mask"]))
        disappear_count = float(np.sum(prev_summary["present_mask"] & (~curr_summary["present_mask"])))
        presence_delta = float(np.mean(np.abs(curr_tokens[:, 0] - prev_tokens[:, 0])))
        type_mass_delta = np.abs(curr_summary["type_mass"] - prev_summary["type_mass"])
        action_norm = float(np.linalg.norm(action)) if action.size > 0 else 0.0
        action_mean = float(np.mean(action)) if action.size > 0 else 0.0
        action_abs_mean = float(np.mean(np.abs(action))) if action.size > 0 else 0.0
        scalar_values = self._collect_scalar_signals(transition.scalar_signals)
        inverse_actual_pred = self._infer_inverse_action(prev_tokens, curr_tokens, scalar_values)
        inverse_pred_pred = (
            self._infer_inverse_action(prev_tokens, pred_tokens, scalar_values)
            if pred_tokens is not None
            else np.zeros_like(action)
        )
        inv_actual_error = self._action_error(inverse_actual_pred, action)
        inv_pred_error = self._action_error(inverse_pred_pred, action)
        coherence_score = float(np.clip(1.0 - inv_pred_error, 0.0, 1.0))
        coverage_score = float(
            np.clip(
                0.65 * scalar_values.get("world_surprise", 0.0)
                + 0.35 * inv_actual_error,
                0.0,
                1.0,
            )
        )
        coherent_surprise_score = float(
            np.clip(
                0.6 * scalar_values.get("world_surprise", 0.0)
                + 0.4 * max(inv_actual_error - inv_pred_error, 0.0),
                0.0,
                1.0,
            )
        )
        scalar_values["inv_actual_error"] = float(inv_actual_error)
        scalar_values["inv_pred_error"] = float(inv_pred_error)
        scalar_values["coherence_score"] = float(coherence_score)
        scalar_values["coverage_score"] = float(coverage_score)
        scalar_values["coherent_surprise_score"] = float(coherent_surprise_score)
        condition_harm = float(
            np.clip(
                0.45 * scalar_values.get("stomach_stress", 0.0)
                + 0.35 * scalar_values.get("pain_level", 0.0)
                + 0.20 * max(scalar_values.get("pain_delta", 0.0), 0.0),
                0.0,
                1.0,
            )
        )
        condition_change = float(
            np.clip(
                0.5 * abs(scalar_values.get("stomach_delta", 0.0))
                + 0.5 * abs(scalar_values.get("pain_delta", 0.0)),
                0.0,
                1.0,
            )
        )
        scalar_values["condition_harm"] = float(condition_harm)
        scalar_values["condition_change"] = float(condition_change)

        evidence_dict: Dict[str, float] = {
            "prev_present_mass": float(prev_summary["present_mass"]),
            "curr_present_mass": float(curr_summary["present_mass"]),
            "presence_delta": float(presence_delta),
            "appear_count": float(appear_count),
            "disappear_count": float(disappear_count),
            "geom_delta_mean": float(geom_delta_mean),
            "geom_delta_xy": float(geom_delta_xy),
            "action_norm": float(action_norm),
            "action_mean": float(action_mean),
            "action_abs_mean": float(action_abs_mean),
            "overlap_ratio": float(prev_summary["overlap_ratio"]),
            "type_delta_mean": float(np.mean(type_mass_delta)) if type_mass_delta.size > 0 else 0.0,
            "condition_harm": float(condition_harm),
            "condition_change": float(condition_change),
        }
        evidence_dict.update(scalar_values)
        for idx, value in enumerate(np.asarray(inverse_actual_pred, dtype=np.float32).reshape(-1)):
            evidence_dict[f"inverse_actual_action_pred_{idx}"] = float(value)
        for idx, value in enumerate(np.asarray(inverse_pred_pred, dtype=np.float32).reshape(-1)):
            evidence_dict[f"inverse_predicted_action_pred_{idx}"] = float(value)

        for idx, value in enumerate(prev_summary["type_mass"]):
            evidence_dict[f"prev_type_mass_{idx}"] = float(value)
        for idx, value in enumerate(curr_summary["type_mass"]):
            evidence_dict[f"curr_type_mass_{idx}"] = float(value)
        for idx, value in enumerate(type_mass_delta):
            evidence_dict[f"type_delta_{idx}"] = float(value)

        seq_features = np.zeros((SEQUENCE_TRACE_DIM,), dtype=np.float32)
        if self.state.recent_evidence:
            seq_stack = np.stack(list(self.state.recent_evidence), axis=0).astype(np.float32, copy=False)
            seq_mean = np.mean(seq_stack, axis=0).astype(np.float32, copy=False).reshape(-1)
            take = min(int(seq_mean.shape[0]), int(SEQUENCE_TRACE_DIM))
            if take > 0:
                seq_features[:take] = seq_mean[:take]

        vector_parts = [
            np.array(
                [
                    evidence_dict["prev_present_mass"],
                    evidence_dict["curr_present_mass"],
                    evidence_dict["presence_delta"],
                    evidence_dict["appear_count"],
                    evidence_dict["disappear_count"],
                    evidence_dict["geom_delta_mean"],
                    evidence_dict["geom_delta_xy"],
                    evidence_dict["action_norm"],
                    evidence_dict["action_mean"],
                    evidence_dict["action_abs_mean"],
                    evidence_dict["overlap_ratio"],
                    evidence_dict["type_delta_mean"],
                ],
                dtype=np.float32,
            ),
            prev_summary["type_mass"].astype(np.float32, copy=False),
            curr_summary["type_mass"].astype(np.float32, copy=False),
            type_mass_delta.astype(np.float32, copy=False),
            np.array([float(scalar_values[k]) for k in self.config.reserved_scalar_keys], dtype=np.float32),
            seq_features,
        ]
        evidence_vector = np.concatenate(vector_parts, axis=0).astype(np.float32, copy=False)
        evidence_vector = self._normalize(evidence_vector)
        return evidence_dict, evidence_vector

    def _infer_inverse_action(
        self,
        prev_tokens: np.ndarray,
        next_tokens: Optional[np.ndarray],
        scalar_values: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        if next_tokens is None:
            return np.zeros((2,), dtype=np.float32)
        prev_agent = self._extract_agent_xy(prev_tokens)
        next_agent = self._extract_agent_xy(next_tokens)
        if prev_agent is None or next_agent is None:
            return np.zeros((2,), dtype=np.float32)
        delta = np.asarray(next_agent, dtype=np.float32) - np.asarray(prev_agent, dtype=np.float32)
        est = float(max(1e-3, self.config.inverse_xy_scale)) * delta
        return np.clip(est.reshape(-1)[:2], -1.0, 1.0).astype(np.float32, copy=False)

    def _build_inverse_features(
        self,
        prev_tokens: np.ndarray,
        next_tokens: np.ndarray,
        scalar_values: Dict[str, float],
        strategy: str,
    ) -> np.ndarray:
        prev_summary = self._summarize_tokens(prev_tokens)
        next_summary = self._summarize_tokens(next_tokens)
        type_delta = np.abs(next_summary["type_mass"] - prev_summary["type_mass"]).astype(np.float32, copy=False)
        overlap_mask = prev_summary["present_mask"] & next_summary["present_mask"]
        geom_delta_mean = 0.0
        geom_delta_xy = 0.0
        if np.any(overlap_mask):
            prev_geom = np.asarray(prev_tokens, dtype=np.float32)[overlap_mask, prev_summary["geom_slice"]]
            next_geom = np.asarray(next_tokens, dtype=np.float32)[overlap_mask, next_summary["geom_slice"]]
            geom_delta = np.abs(next_geom - prev_geom)
            geom_delta_mean = float(np.mean(geom_delta))
            geom_delta_xy = float(np.mean(np.linalg.norm(next_geom[:, :2] - prev_geom[:, :2], axis=1)))
        appear_count = float(np.sum((~prev_summary["present_mask"]) & next_summary["present_mask"]))
        disappear_count = float(np.sum(prev_summary["present_mask"] & (~next_summary["present_mask"])))
        presence_delta = float(
            np.mean(np.abs(np.asarray(next_tokens, dtype=np.float32)[:, 0] - np.asarray(prev_tokens, dtype=np.float32)[:, 0]))
        )
        prev_agent = self._extract_agent_xy(prev_tokens)
        next_agent = self._extract_agent_xy(next_tokens)
        agent_dx = 0.0
        agent_dy = 0.0
        if prev_agent is not None and next_agent is not None:
            delta = np.asarray(next_agent, dtype=np.float32) - np.asarray(prev_agent, dtype=np.float32)
            agent_dx = float(delta[0])
            if delta.shape[0] > 1:
                agent_dy = float(delta[1])
        pooled = [
            float(prev_summary["present_mass"]),
            float(next_summary["present_mass"]),
            float(presence_delta),
            float(appear_count),
            float(disappear_count),
            float(geom_delta_mean),
            float(geom_delta_xy),
            float(prev_summary["overlap_ratio"]),
            float(np.mean(type_delta)) if type_delta.size > 0 else 0.0,
            float(agent_dx),
            float(agent_dy),
        ]
        prev_type = prev_summary["type_mass"].astype(np.float32, copy=False)
        next_type = next_summary["type_mass"].astype(np.float32, copy=False)
        if strategy == "learned_inverse_delta_only":
            parts = [
                np.asarray(
                    [
                        presence_delta,
                        appear_count,
                        disappear_count,
                        geom_delta_mean,
                        geom_delta_xy,
                        agent_dx,
                        agent_dy,
                    ],
                    dtype=np.float32,
                ),
                type_delta.astype(np.float32, copy=False),
            ]
        elif strategy == "learned_inverse_pooled_regime":
            parts = [
                np.asarray(pooled, dtype=np.float32),
                prev_type,
                next_type,
                type_delta.astype(np.float32, copy=False),
                np.asarray(
                    [
                        float(scalar_values.get("regime_signal", 0.0)),
                        float(scalar_values.get("condition_harm", 0.0)),
                        float(scalar_values.get("condition_change", 0.0)),
                        float(scalar_values.get("stomach_level", 0.0)),
                        float(scalar_values.get("pain_level", 0.0)),
                        float(scalar_values.get("pain_from_stomach", 0.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        else:
            parts = [
                np.asarray(pooled, dtype=np.float32),
                prev_type,
                next_type,
                type_delta.astype(np.float32, copy=False),
            ]
        feat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(feat))
        if norm > EPS:
            feat = feat / norm
        return feat

    def _predict_inverse_from_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros((2,), dtype=np.float32)
        self._ensure_inverse_state(int(x.shape[0]))
        assert self.inverse_weights is not None
        pred = np.matmul(self.inverse_weights, x.reshape(-1, 1)).reshape(-1)
        return np.clip(pred[:2], -1.0, 1.0).astype(np.float32, copy=False)

    def _ensure_inverse_state(self, feat_dim: int) -> None:
        dim = max(1, int(feat_dim))
        if self.inverse_weights is not None and self.inverse_cov is not None:
            if int(self.inverse_weights.shape[1]) == dim:
                return
        self.inverse_weights = np.zeros((2, dim), dtype=np.float32)
        self.inverse_cov = (
            np.eye(dim, dtype=np.float32) * float(max(1e-3, self.config.inverse_rls_init_scale))
        )

    def _update_inverse_model(self, transition: CodaCausalTransition) -> None:
        return

    def _extract_agent_xy(self, tokens: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(tokens, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        num_types = min(4, max(0, arr.shape[1] - 1))
        if num_types <= 0 or arr.shape[1] <= (1 + num_types):
            return None
        gates = np.clip(arr[:, 0], 0.0, 1.0)
        type_idx = np.argmax(arr[:, 1 : 1 + num_types], axis=-1)
        agent_mask = (type_idx == 0) & (gates >= float(self.config.gate_present_threshold))
        if not np.any(agent_mask):
            return None
        idx = int(np.argmax(gates * agent_mask.astype(np.float32)))
        geom_start = 1 + num_types
        if arr.shape[1] < geom_start + 2:
            return None
        return arr[idx, geom_start : geom_start + 2].astype(np.float32, copy=False)

    def _action_error(self, pred_action: np.ndarray, true_action: np.ndarray) -> float:
        pa = np.asarray(pred_action, dtype=np.float32).reshape(-1)
        ta = np.asarray(true_action, dtype=np.float32).reshape(-1)
        n = max(int(pa.shape[0]), int(ta.shape[0]), 1)
        if pa.shape[0] != n:
            pa = np.pad(pa, (0, n - pa.shape[0]))
        if ta.shape[0] != n:
            ta = np.pad(ta, (0, n - ta.shape[0]))
        l2 = float(np.linalg.norm(pa - ta) / np.sqrt(float(n)))
        return float(np.clip(l2, 0.0, 1.0))

    def _summarize_tokens(self, tokens: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(tokens, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
            return {
                "present_mask": np.zeros((0,), dtype=bool),
                "present_mass": 0.0,
                "type_mass": np.zeros((0,), dtype=np.float32),
                "overlap_ratio": 0.0,
                "geom_slice": slice(0, 0),
            }
        num_types = max(0, arr.shape[1] - 1)
        geom_start = 1
        if num_types >= 4:
            num_types = 4
            geom_start = 1 + num_types
        else:
            num_types = max(1, arr.shape[1] // 2)
            geom_start = min(arr.shape[1], 1 + num_types)
        type_logits = arr[:, 1 : 1 + num_types]
        gates = np.clip(arr[:, 0], 0.0, 1.0)
        present_mask = gates >= float(self.config.gate_present_threshold)
        type_mass = np.sum(gates[:, None] * type_logits, axis=0) if type_logits.size > 0 else np.zeros((0,), dtype=np.float32)
        overlap_ratio = float(np.mean(present_mask.astype(np.float32))) if present_mask.size > 0 else 0.0
        return {
            "present_mask": present_mask,
            "present_mass": float(np.sum(gates)),
            "type_mass": type_mass.astype(np.float32, copy=False),
            "overlap_ratio": overlap_ratio,
            "geom_slice": slice(geom_start, arr.shape[1]),
        }

    def _collect_scalar_signals(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in self.config.reserved_scalar_keys:
            try:
                out[key] = float(raw_signals.get(key, 0.0))
            except Exception:
                out[key] = 0.0
        return out

    def _retrieve(self, evidence_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.memory_bank:
            return np.zeros((0,), dtype=np.float32), self._empty_memory_state()
        scores = []
        for entry in self.memory_bank:
            scores.append(self._similarity(entry.key, evidence_vector) * float(max(EPS, entry.strength)))
        score_arr = np.asarray(scores, dtype=np.float32)
        topk = min(len(score_arr), max(1, int(self.config.retrieval_topk)))
        if topk <= 0:
            return np.zeros((0,), dtype=np.float32), self._empty_memory_state()
        order = np.argsort(score_arr)[-topk:]
        top_scores = score_arr[order].astype(np.float32, copy=False)
        weights = np.maximum(top_scores, 0.0)
        if float(np.sum(weights)) <= EPS:
            retrieved = self._empty_memory_state()
        else:
            weights = weights / float(np.sum(weights))
            retrieved = np.sum(
                [float(weights[idx]) * self.memory_bank[int(mem_idx)].value for idx, mem_idx in enumerate(order)],
                axis=0,
                dtype=np.float32,
            )
        return top_scores, retrieved.astype(np.float32, copy=False)

    def _update_beliefs(self, evidence_dict: Dict[str, float], retrieved_value: np.ndarray) -> np.ndarray:
        prev = self.state.belief_vector.astype(np.float32, copy=True)
        target = np.zeros_like(prev)
        current_state = self._derive_current_causal_state(evidence_dict)
        retrieved_state = self._current_state_from_memory(retrieved_value)
        current_env_drive = float(
            np.clip(
                max(
                    current_state.get("current_env_confidence", 0.0) * current_state.get("current_env_match", 0.0),
                    retrieved_state.get("current_env_confidence", 0.0) * retrieved_state.get("current_env_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        current_stomach_drive = float(
            np.clip(
                max(
                    current_state.get("current_self_stomach_confidence", 0.0)
                    * current_state.get("current_self_stomach_match", 0.0),
                    retrieved_state.get("current_self_stomach_confidence", 0.0)
                    * retrieved_state.get("current_self_stomach_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        current_pain_drive = float(
            np.clip(
                max(
                    current_state.get("current_self_pain_confidence", 0.0)
                    * current_state.get("current_self_pain_match", 0.0),
                    retrieved_state.get("current_self_pain_confidence", 0.0)
                    * retrieved_state.get("current_self_pain_match", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        self_effect_drive = float(
            np.clip(
                max(
                    max(evidence_dict.get("stomach_delta", 0.0), 0.0),
                    max(-evidence_dict.get("pain_delta", 0.0), 0.0),
                ),
                0.0,
                1.0,
            )
        )
        target[0] = float(np.clip(evidence_dict.get("presence_delta", 0.0) + evidence_dict.get("world_surprise", 0.0), 0.0, 1.0))
        target[1] = float(
            np.clip(
                0.25 * evidence_dict.get("interaction_event", 0.0)
                + 0.20 * evidence_dict.get("consumed_token", 0.0)
                + 0.15 * self_effect_drive
                + 0.15 * max(current_stomach_drive, current_pain_drive)
                + 0.15 * current_env_drive
                + 0.10 * evidence_dict.get("action_abs_mean", 0.0),
                0.0,
                1.0,
            )
        )
        target[2] = float(
            np.clip(
                evidence_dict.get("env_change_observed", 0.0)
                + 0.40 * current_env_drive
                + 0.5 * evidence_dict.get("regime_signal", 0.0)
                + 0.5 * evidence_dict.get("world_surprise", 0.0),
                0.0,
                1.0,
            )
        )
        target[3] = float(self._compute_delayed_belief_target(evidence_dict, retrieved_value, prev))
        target[4] = float(np.clip(evidence_dict.get("regime_signal", 0.0), 0.0, 1.0))
        target[5] = float(
            np.clip(
                max(
                    target[2],
                    target[3],
                    target[1],
                    current_env_drive,
                    current_stomach_drive,
                    current_pain_drive,
                    evidence_dict.get("condition_harm", 0.0),
                    0.6 * evidence_dict.get("condition_change", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        target[6] = float(np.clip(retrieved_state.get("memory_causal_strength", 0.0), 0.0, 1.0))
        target[7] = float(np.clip(evidence_dict.get("belief_probe_target", 0.0), 0.0, 1.0))

        updated = self.config.belief_decay * prev + (1.0 - self.config.belief_decay) * target
        if retrieved_value.size == updated.size:
            updated = np.clip(0.8 * updated + 0.2 * retrieved_value, 0.0, 1.0)
        self.state.belief_vector = updated.astype(np.float32, copy=False)
        return self.state.belief_vector

    def _compute_delayed_belief_target(
        self,
        evidence_dict: Dict[str, float],
        retrieved_value: np.ndarray,
        prev_belief: np.ndarray,
    ) -> float:
        retrieved_state = self._current_state_from_memory(retrieved_value)
        direct = float(
            np.clip(
                evidence_dict.get("delayed_change_target", 0.0)
                + 0.6 * evidence_dict.get("consumed_token", 0.0)
                + 0.3 * evidence_dict.get("interaction_event", 0.0),
                0.0,
                1.0,
            )
        )
        interaction_drive = float(
            np.clip(
                0.7 * evidence_dict.get("consumed_token", 0.0)
                + 0.15 * self._derive_current_causal_state(evidence_dict).get("current_self_stomach_support", 0.0)
                + 0.15 * self._derive_current_causal_state(evidence_dict).get("current_self_pain_support", 0.0)
                + 0.3 * evidence_dict.get("interaction_event", 0.0),
                0.0,
                1.0,
            )
        )
        regime_drive = float(
            np.clip(
                0.7 * evidence_dict.get("regime_signal", 0.0)
                + 0.20 * self._derive_current_causal_state(evidence_dict).get("current_env_support", 0.0)
                + 0.3 * evidence_dict.get("env_change_observed", 0.0),
                0.0,
                1.0,
            )
        )
        memory_drive = float(
            np.clip(
                max(
                    retrieved_state.get("current_env_support", 0.0),
                    retrieved_state.get("current_self_stomach_support", 0.0),
                    retrieved_state.get("current_self_pain_support", 0.0),
                    retrieved_state.get("memory_causal_strength", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        surprise_drive = float(np.clip(evidence_dict.get("world_surprise", 0.0), 0.0, 1.0))

        self.state.interaction_trace = float(
            np.clip(
                self.config.interaction_trace_decay * self.state.interaction_trace
                + self.config.interaction_trace_gain * interaction_drive,
                0.0,
                1.0,
            )
        )
        self.state.regime_trace = float(
            np.clip(
                self.config.regime_trace_decay * self.state.regime_trace
                + self.config.regime_trace_gain * regime_drive,
                0.0,
                1.0,
            )
        )
        self.state.cause_progress = float(
            np.clip(
                self.config.cause_progress_decay * self.state.cause_progress
                + self.config.interaction_trace_gain * interaction_drive
                + self.config.regime_trace_gain * regime_drive
                + self.config.memory_trace_gain * memory_drive
                + self.config.surprise_trace_gain * surprise_drive,
                0.0,
                1.0,
            )
        )

        accum = float(np.clip(self.state.cause_progress, 0.0, 1.0))
        blended = 0.82 * accum + 0.18 * direct
        # Keep some inertia so repeated relevant events can compound over time.
        blended = max(blended, float(prev_belief[3]) * float(self.config.cause_progress_decay))
        return float(np.clip(blended, 0.0, 1.0))

    def _compute_structured_surprise(
        self,
        evidence_dict: Dict[str, float],
        belief_vector: np.ndarray,
        retrieval_scores: np.ndarray,
    ) -> Dict[str, float]:
        world_surprise = float(evidence_dict.get("world_surprise", 0.0))
        regime_signal = float(evidence_dict.get("regime_signal", 0.0))
        novelty = float(
            np.clip(
                evidence_dict.get("presence_delta", 0.0)
                + evidence_dict.get("geom_delta_mean", 0.0)
                + evidence_dict.get("type_delta_mean", 0.0),
                0.0,
                1.0,
            )
        )
        memory_relevance = float(np.max(retrieval_scores)) if retrieval_scores.size > 0 else 0.0
        condition_harm = float(np.clip(evidence_dict.get("condition_harm", 0.0), 0.0, 1.0))
        condition_change = float(np.clip(evidence_dict.get("condition_change", 0.0), 0.0, 1.0))
        coverage_score = float(np.clip(evidence_dict.get("coverage_score", 0.0), 0.0, 1.0))
        coherence_score = float(np.clip(evidence_dict.get("coherence_score", 0.0), 0.0, 1.0))
        coherent_surprise_score = float(np.clip(evidence_dict.get("coherent_surprise_score", 0.0), 0.0, 1.0))
        current_state = self._derive_current_causal_state(evidence_dict)
        current_env_drive = float(
            np.clip(current_state.get("current_env_confidence", 0.0) * current_state.get("current_env_match", 0.0), 0.0, 1.0)
        )
        current_stomach_drive = float(
            np.clip(
                current_state.get("current_self_stomach_confidence", 0.0)
                * current_state.get("current_self_stomach_match", 0.0),
                0.0,
                1.0,
            )
        )
        current_pain_drive = float(
            np.clip(
                current_state.get("current_self_pain_confidence", 0.0)
                * current_state.get("current_self_pain_match", 0.0),
                0.0,
                1.0,
            )
        )
        belief_surprise = float(
            np.clip(
                abs(evidence_dict.get("env_change_observed", 0.0) - current_env_drive)
                + abs(max(evidence_dict.get("stomach_delta", 0.0), 0.0) - current_stomach_drive)
                + abs(max(-evidence_dict.get("pain_delta", 0.0), 0.0) - current_pain_drive)
                + abs(evidence_dict.get("delayed_change_target", 0.0) - belief_vector[3]),
                0.0,
                1.0,
            )
        )
        causal_significance = float(
            np.clip(
                max(
                    max(current_env_drive, current_stomach_drive, current_pain_drive, belief_vector[3], regime_signal, world_surprise),
                    0.55
                    * max(current_env_drive, current_stomach_drive, current_pain_drive, belief_vector[3], regime_signal, world_surprise)
                    + 0.45
                    * max(
                        condition_harm,
                        0.7 * condition_change,
                        coherent_surprise_score,
                    ),
                    0.5 * coherent_surprise_score + 0.5 * condition_harm,
                    0.65 * condition_harm + 0.35 * condition_change,
                ),
                0.0,
                1.0,
            )
        )
        novelty_weight = float(np.clip(self.config.novelty_weight, 0.0, 1.0))
        condition_weight = 0.10
        leftover = max(0.0, 1.0 - (0.35 + 0.25 + novelty_weight + condition_weight))
        structured = float(
            np.clip(
                0.35 * world_surprise
                + 0.25 * belief_surprise
                + novelty_weight * novelty
                + condition_weight * condition_harm
                + leftover * max(causal_significance - memory_relevance, 0.0),
                0.0,
                1.0,
            )
        )
        return {
            "world_surprise": float(world_surprise),
            "belief_surprise": float(belief_surprise),
            "novelty": float(novelty),
            "memory_relevance": float(memory_relevance),
            "coverage_score": float(coverage_score),
            "coherence_score": float(coherence_score),
            "coherent_surprise_score": float(coherent_surprise_score),
            "condition_harm": float(condition_harm),
            "condition_change": float(condition_change),
            "causal_significance": float(causal_significance),
            "structured_surprise": float(structured),
        }

    def _should_write_memory(
        self,
        evidence_dict: Dict[str, float],
        surprise_dict: Dict[str, float],
        retrieval_scores: np.ndarray,
    ) -> bool:
        significance = float(surprise_dict.get("causal_significance", 0.0))
        novelty = float(surprise_dict.get("novelty", 0.0))
        coherent_surprise = float(surprise_dict.get("coherent_surprise_score", 0.0))
        condition_harm = float(surprise_dict.get("condition_harm", 0.0))
        retrieved = float(np.max(retrieval_scores)) if retrieval_scores.size > 0 else 0.0
        condition_change = float(surprise_dict.get("condition_change", 0.0))
        coverage = float(surprise_dict.get("coverage_score", 0.0))
        write_drive = float(
            np.clip(
                max(
                    significance,
                    0.55 * coherent_surprise
                    + 0.45 * condition_harm
                    + 0.20 * condition_change,
                    0.65 * condition_harm + 0.25 * condition_change + 0.15 * coverage,
                )
                - float(self.config.coherent_write_retrieval_penalty) * retrieved,
                0.0,
                1.0,
            )
        )
        return bool(
            write_drive >= float(self.config.significance_threshold)
            or (
                coherent_surprise >= float(self.config.coherent_write_force_coherent_threshold)
                and coverage >= float(self.config.coherent_write_force_coverage_threshold)
                and retrieved < float(self.config.coherent_write_force_retrieved_max)
            )
            or (
                condition_harm >= float(self.config.coherent_write_force_harm_threshold)
                and condition_change >= float(self.config.coherent_write_force_change_threshold)
                and retrieved < float(self.config.coherent_write_force_retrieved_max)
            )
            or (
                novelty >= float(self.config.novelty_threshold)
                and coherent_surprise >= 0.12
                and retrieved < 0.70
            )
        )

    def _write_memory(
        self,
        evidence_vector: np.ndarray,
        evidence_dict: Dict[str, float],
        belief_vector: np.ndarray,
        transition: CodaCausalTransition,
    ) -> None:
        current_state = self._derive_current_causal_state(evidence_dict)
        value = self._memory_state_from_current(current_state, evidence_dict)
        entry = CodaCausalMemoryEntry(
            key=evidence_vector.astype(np.float32, copy=True),
            value=value,
            strength=float(max(self.config.significance_threshold, value[9])),
            age=0,
            write_count=1,
            metadata={
                "timestep": int(self.state.timestep),
                "metadata": dict(transition.metadata),
                "evidence_summary": {
                    "env_change_observed": float(evidence_dict.get("env_change_observed", 0.0)),
                    "blocking_signal": float(evidence_dict.get("blocking_signal", 0.0)),
                    "region_signature": float(evidence_dict.get("region_signature", 0.0)),
                    "region_proximity": float(evidence_dict.get("region_proximity", 0.0)),
                    "region_dwell": float(evidence_dict.get("region_dwell", 0.0)),
                    "region_occupancy": float(evidence_dict.get("region_occupancy", 0.0)),
                    "region_entry_trace": float(evidence_dict.get("region_entry_trace", 0.0)),
                    "control_ineffectiveness": float(evidence_dict.get("control_ineffectiveness", 0.0)),
                    "means_feasibility": float(evidence_dict.get("means_feasibility", 0.0)),
                    "rollout_mismatch": float(evidence_dict.get("rollout_mismatch", 0.0)),
                    "trajectory_progress": float(evidence_dict.get("trajectory_progress", 0.0)),
                },
            },
        )
        if len(self.memory_bank) >= int(max(1, self.config.max_memories)):
            weakest_idx = int(np.argmin([max(EPS, mem.strength) / float(1 + mem.age) for mem in self.memory_bank]))
            self.memory_bank[weakest_idx] = entry
        else:
            self.memory_bank.append(entry)

    def _derive_memory_hints(self, evidence_dict: Dict[str, float]) -> Dict[str, float]:
        hints: Dict[str, float] = {
            "affordance_support": 0.0,
            "best_affordance_signature": 0.0,
            "best_affordance_confidence": 0.0,
            "best_affordance_match": 0.0,
            "known_affordance_signature_0": 0.0,
            "known_affordance_confidence_0": 0.0,
            "known_affordance_match_0": 0.0,
            "known_affordance_effect_env_0": 0.0,
            "known_affordance_effect_self_stomach_0": 0.0,
            "known_affordance_effect_self_pain_0": 0.0,
            "known_affordance_signature_1": 0.0,
            "known_affordance_confidence_1": 0.0,
            "known_affordance_match_1": 0.0,
            "known_affordance_effect_env_1": 0.0,
            "known_affordance_effect_self_stomach_1": 0.0,
            "known_affordance_effect_self_pain_1": 0.0,
            "wall_method_support": 0.0,
            "best_known_region_signature": 0.0,
            "best_known_region_support": 0.0,
            "best_known_region_match": 0.0,
            "known_region_signature_0": 0.0,
            "known_region_support_0": 0.0,
            "known_region_match_0": 0.0,
            "known_region_signature_1": 0.0,
            "known_region_support_1": 0.0,
            "known_region_match_1": 0.0,
        }
        for target in AFFORDANCE_TARGETS:
            hints[f"best_affordance_{target}_signature"] = 0.0
            hints[f"best_affordance_{target}_confidence"] = 0.0
            hints[f"best_affordance_{target}_match"] = 0.0
            hints[f"best_affordance_{target}_support"] = 0.0
        num_bins = AFFORDANCE_BIN_COUNT

        target_supports: Dict[str, float] = {}
        target_orders: Dict[str, List[int]] = {}
        for target in AFFORDANCE_TARGETS:
            bins = self.affordance_bins[target].astype(np.float32, copy=False)
            total = float(np.sum(bins))
            if total <= EPS:
                continue
            target_supports[target] = total
            target_orders[target] = [int(idx) for idx in np.argsort(bins)[::-1].tolist() if float(bins[int(idx)]) > EPS]

        if not target_supports:
            return hints

        overall_total = float(sum(target_supports.values()))
        hints["affordance_support"] = float(np.clip(overall_total / 6.0, 0.0, 1.0))

        ranked: List[Tuple[float, str, int]] = []
        for target, order in target_orders.items():
            bins = self.affordance_bins[target].astype(np.float32, copy=False)
            total = max(target_supports[target], EPS)
            current_signature = self._affordance_signature_for_target(target, evidence_dict)
            best_idx = int(order[0])
            best_center = float((float(best_idx) + 0.5) / float(num_bins))
            best_support = float(np.clip(float(bins[best_idx]) / total, 0.0, 1.0))
            best_match = float(np.clip(1.0 - abs(current_signature - best_center) / (1.0 / float(num_bins)), 0.0, 1.0))
            hints[f"best_affordance_{target}_signature"] = best_center
            hints[f"best_affordance_{target}_confidence"] = best_support
            hints[f"best_affordance_{target}_match"] = best_match
            hints[f"best_affordance_{target}_support"] = float(np.clip(target_supports[target] / 6.0, 0.0, 1.0))
            for bin_idx in order[:2]:
                conf = float(np.clip(float(bins[bin_idx]) / total, 0.0, 1.0))
                ranked.append((conf, target, int(bin_idx)))
        ranked.sort(key=lambda item: item[0], reverse=True)

        for rank, (_, target, bin_idx) in enumerate(ranked[:2]):
            bins = self.affordance_bins[target].astype(np.float32, copy=False)
            total = max(target_supports[target], EPS)
            current_signature = self._affordance_signature_for_target(target, evidence_dict)
            center = float((float(bin_idx) + 0.5) / float(num_bins))
            support = float(np.clip(float(bins[bin_idx]) / total, 0.0, 1.0))
            match = float(np.clip(1.0 - abs(current_signature - center) / (1.0 / float(num_bins)), 0.0, 1.0))
            hints[f"known_affordance_signature_{rank}"] = center
            hints[f"known_affordance_confidence_{rank}"] = support
            hints[f"known_affordance_match_{rank}"] = match
            hints[f"known_affordance_effect_env_{rank}"] = 1.0 if target == "env" else 0.0
            hints[f"known_affordance_effect_self_stomach_{rank}"] = 1.0 if target == "self_stomach" else 0.0
            hints[f"known_affordance_effect_self_pain_{rank}"] = 1.0 if target == "self_pain" else 0.0
            if rank == 0:
                hints["best_affordance_signature"] = center
                hints["best_affordance_confidence"] = support
                hints["best_affordance_match"] = match

        env_total = target_supports.get("env", 0.0)
        if env_total > EPS:
            env_bins = self.affordance_bins["env"].astype(np.float32, copy=False)
            hints["wall_method_support"] = float(np.clip(env_total / 6.0, 0.0, 1.0))
            current_signature = self._affordance_signature_for_target("env", evidence_dict)
            for rank, bin_idx in enumerate(target_orders.get("env", [])[:2]):
                center = float((float(bin_idx) + 0.5) / float(num_bins))
                support = float(np.clip(float(env_bins[bin_idx]) / max(env_total, EPS), 0.0, 1.0))
                match = float(np.clip(1.0 - abs(current_signature - center) / (1.0 / float(num_bins)), 0.0, 1.0))
                hints[f"known_region_signature_{rank}"] = center
                hints[f"known_region_support_{rank}"] = support
                hints[f"known_region_match_{rank}"] = match
                if rank == 0:
                    hints["best_known_region_signature"] = center
                    hints["best_known_region_support"] = support
                    hints["best_known_region_match"] = match
        return hints

    def _current_affordance_state_for_target(self, target: str, evidence_dict: Dict[str, float]) -> Tuple[float, float, float]:
        bins = self.affordance_bins.get(target)
        if bins is None:
            return 0.0, 0.0, 0.0
        total = float(np.sum(bins))
        if total <= EPS:
            return 0.0, 0.0, 0.0
        signature = self._affordance_signature_for_target(target, evidence_dict)
        context_gate, _ = self._affordance_context_for_target(target, evidence_dict)
        num_bins = AFFORDANCE_BIN_COUNT
        best_idx = int(np.argmax(bins))
        center = float((float(best_idx) + 0.5) / float(num_bins))
        confidence = float(np.clip(float(bins[best_idx]) / total, 0.0, 1.0))
        match = float(np.clip(1.0 - abs(signature - center) / (1.0 / float(num_bins)), 0.0, 1.0))
        support = float(np.clip((total / 6.0) * (0.5 + 0.5 * context_gate), 0.0, 1.0))
        if str(target).strip().lower() == "env":
            local_idx = int(np.clip(np.floor(signature * float(num_bins)), 0, num_bins - 1))
            local_mass = float(bins[local_idx])
            if local_idx > 0:
                local_mass += 0.35 * float(bins[local_idx - 1])
            if local_idx + 1 < num_bins:
                local_mass += 0.35 * float(bins[local_idx + 1])
            local_ratio = float(np.clip(local_mass / total, 0.0, 1.0))
            confidence = float(np.clip(0.6 * confidence + 0.4 * local_ratio, 0.0, 1.0))
            support = float(np.clip((0.25 + 0.75 * local_ratio) * (0.35 + 0.65 * context_gate), 0.0, 1.0))
            blocking = float(np.clip(evidence_dict.get("blocking_signal", 0.0), 0.0, 1.0))
            confidence = float(np.clip(confidence * (0.55 + 0.45 * blocking), 0.0, 1.0))
            support = float(np.clip(support * (0.25 + 0.75 * blocking), 0.0, 1.0))
        return confidence, match, support

    def _derive_current_causal_state(self, evidence_dict: Dict[str, float]) -> Dict[str, float]:
        state: Dict[str, float] = {}
        for target in AFFORDANCE_TARGETS:
            conf, match, support = self._current_affordance_state_for_target(target, evidence_dict)
            state[f"current_{target}_confidence"] = conf
            state[f"current_{target}_match"] = match
            state[f"current_{target}_support"] = support
        return state

    def _affordance_signature_for_target(self, target: str, evidence_dict: Dict[str, float]) -> float:
        key = str(target).strip().lower()
        if key == "env":
            return float(np.clip(evidence_dict.get("region_signature", 0.0), 0.0, 1.0))
        if key == "self_stomach":
            stomach_level = float(np.clip(evidence_dict.get("stomach_level", 0.0), 0.0, 1.0))
            consumed = float(np.clip(evidence_dict.get("consumed_token", 0.0), 0.0, 1.0))
            interaction = float(np.clip(evidence_dict.get("interaction_event", 0.0), 0.0, 1.0))
            pod_delta = float(np.clip(abs(evidence_dict.get("type_delta_1", 0.0)), 0.0, 1.0))
            means_signature = float(np.clip(max(consumed, interaction, pod_delta), 0.0, 1.0))
            need_signature = float(np.clip(1.0 - stomach_level, 0.0, 1.0))
            return float(np.clip(0.65 * means_signature + 0.35 * need_signature, 0.0, 1.0))
        if key == "self_pain":
            pain_level = float(np.clip(evidence_dict.get("pain_level", 0.0), 0.0, 1.0))
            stomach_level = float(np.clip(evidence_dict.get("stomach_level", 0.0), 0.0, 1.0))
            pain_from_stomach = float(np.clip(evidence_dict.get("pain_from_stomach", 0.0), 0.0, 1.0))
            stomach_recovery = float(np.clip(max(evidence_dict.get("stomach_delta", 0.0), 0.0), 0.0, 1.0))
            return float(
                np.clip(
                    0.45 * pain_from_stomach
                    + 0.25 * pain_level
                    + 0.20 * float(np.clip(1.0 - stomach_level, 0.0, 1.0))
                    + 0.10 * stomach_recovery,
                    0.0,
                    1.0,
                )
            )
        return 0.0

    def _affordance_context_for_target(self, target: str, evidence_dict: Dict[str, float]) -> Tuple[float, float]:
        key = str(target).strip().lower()
        if key == "env":
            occupancy = float(np.clip(evidence_dict.get("region_occupancy", 0.0), 0.0, 1.0))
            proximity = float(np.clip(evidence_dict.get("region_proximity", 0.0), 0.0, 1.0))
            dwell = float(np.clip(evidence_dict.get("region_dwell", 0.0), 0.0, 1.0))
            entry_trace = float(np.clip(evidence_dict.get("region_entry_trace", 0.0), 0.0, 1.0))
            context_gate = float(np.clip(max(occupancy, proximity, entry_trace), 0.0, 1.0))
            context_weight = float(
                np.clip((0.4 + 0.6 * context_gate) * (0.8 + 0.7 * proximity + 0.5 * dwell), 0.0, 3.0)
            )
            return context_gate, context_weight
        if key == "self_stomach":
            stomach_level = float(np.clip(evidence_dict.get("stomach_level", 0.0), 0.0, 1.0))
            stomach_stress = float(np.clip(evidence_dict.get("stomach_stress", 0.0), 0.0, 1.0))
            consumed_token = float(np.clip(evidence_dict.get("consumed_token", 0.0), 0.0, 1.0))
            interaction_event = float(np.clip(evidence_dict.get("interaction_event", 0.0), 0.0, 1.0))
            pod_delta = float(np.clip(abs(evidence_dict.get("type_delta_1", 0.0)), 0.0, 1.0))
            disappear_count = float(np.clip(evidence_dict.get("disappear_count", 0.0), 0.0, 1.0))
            stomach_delta = float(np.clip(abs(evidence_dict.get("stomach_delta", 0.0)), 0.0, 1.0))
            means_gate = float(np.clip(max(consumed_token, interaction_event, pod_delta, disappear_count), 0.0, 1.0))
            context_gate = float(np.clip(max(means_gate, stomach_stress, stomach_delta), 0.0, 1.0))
            context_weight = float(
                np.clip(
                    (0.5 + 0.5 * context_gate)
                    * (0.7 + 0.8 * means_gate + 0.5 * stomach_stress + 0.4 * (1.0 - stomach_level)),
                    0.0,
                    3.0,
                )
            )
            return context_gate, context_weight
        if key == "self_pain":
            pain_level = float(np.clip(evidence_dict.get("pain_level", 0.0), 0.0, 1.0))
            pain_from_stomach = float(np.clip(evidence_dict.get("pain_from_stomach", 0.0), 0.0, 1.0))
            stomach_stress = float(np.clip(evidence_dict.get("stomach_stress", 0.0), 0.0, 1.0))
            pain_relief = float(np.clip(max(-evidence_dict.get("pain_delta", 0.0), 0.0), 0.0, 1.0))
            stomach_recovery = float(np.clip(max(evidence_dict.get("stomach_delta", 0.0), 0.0), 0.0, 1.0))
            context_gate = float(np.clip(max(pain_level, pain_from_stomach, pain_relief, stomach_recovery), 0.0, 1.0))
            context_weight = float(
                np.clip(
                    (0.5 + 0.5 * context_gate)
                    * (0.7 + 0.7 * pain_from_stomach + 0.6 * stomach_recovery + 0.4 * stomach_stress),
                    0.0,
                    3.0,
                )
            )
            return context_gate, context_weight
        return 0.0, 0.0

    def _update_affordance_memory(self, evidence_dict: Dict[str, float]) -> None:
        if not self.affordance_bins:
            return
        target_values = {
            "env": float(np.clip(evidence_dict.get("env_change_observed", 0.0), 0.0, 1.0)),
            "self_stomach": float(np.clip(max(evidence_dict.get("stomach_delta", 0.0), 0.0), 0.0, 1.0)),
            "self_pain": float(np.clip(max(-evidence_dict.get("pain_delta", 0.0), 0.0), 0.0, 1.0)),
        }
        for target in AFFORDANCE_TARGETS:
            self.affordance_bins[target] *= np.float32(0.995)
            signature = self._affordance_signature_for_target(target, evidence_dict)
            context_gate, context_weight = self._affordance_context_for_target(target, evidence_dict)
            if context_gate <= EPS:
                continue
            effect_value = target_values[target]
            if effect_value <= EPS:
                continue
            bin_idx = int(np.clip(np.floor(signature * float(AFFORDANCE_BIN_COUNT)), 0, AFFORDANCE_BIN_COUNT - 1))
            weight = float(np.clip(context_weight * effect_value, 0.0, 3.0))
            self.affordance_bins[target][bin_idx] = np.float32(
                min(12.0, float(self.affordance_bins[target][bin_idx]) + weight)
            )

    def _advance_time(self, evidence_vector: np.ndarray) -> None:
        self.state.recent_evidence.append(evidence_vector.astype(np.float32, copy=True))
        self.state.timestep += 1
        for idx, entry in enumerate(self.memory_bank):
            self.memory_bank[idx] = replace(
                entry,
                strength=float(np.clip(entry.strength * self.config.memory_decay, 0.05, 1.0)),
                age=int(entry.age + 1),
            )

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.config.memory_similarity == "cosine":
            a_n = self._normalize(a)
            b_n = self._normalize(b)
            return float(np.dot(a_n, b_n))
        return float(-np.mean(np.abs(a - b)))

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(arr))
        if denom <= EPS:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / denom).astype(np.float32, copy=False)
