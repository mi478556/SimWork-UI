from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from agent.coda_forward_model import TYPE_AGENT, TYPE_POD
from agent.coda_policy import CODAPolicy
from dataset.session_store import SessionStore
from runner.observation_builder import build_observation


def configure_cpu_threads(num_threads: int) -> None:
    n = int(max(1, num_threads))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(n, 2)))
    except Exception:
        pass


def candidate_specs() -> List[Dict[str, Any]]:
    return [
        {"label": "baseline", "cfg_updates": {}, "bootstrap_kwargs": {}},
        {
            "label": "contextual_spawn_prior",
            "cfg_updates": {
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "event_window_steps": 6,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "zero_spawn_prior",
            "cfg_updates": {
                "loss_spawn_prior_weight": 0.0,
                "event_window_steps": 6,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "contextual_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "elapsed_time_scale": 4.0,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "zero_prior_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.0,
                "elapsed_time_scale": 4.0,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "blanket_prior_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 0.0,
                "elapsed_time_scale": 4.0,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "lookahead_contextual_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "lookahead_contextual_hungarian_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
            },
            "bootstrap_kwargs": {},
        },
        {
            "label": "lookahead_contextual_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_contextual_hungarian_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_identity_memory_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_identity_memory_low_weight_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.15,
                "match_memory_decay_scale": 12.0,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_identity_memory_fast_decay_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 4.0,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_identity_memory_reappear_only_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
                "match_memory_reappearance_only": True,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_learned_correspondence_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
                "match_memory_reappearance_only": True,
                "match_learned_embed_weight": 0.25,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_learned_corr_low_weight_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
                "match_memory_reappearance_only": True,
                "match_learned_embed_weight": 0.10,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_learned_corr_fast_decay_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
                "match_memory_reappearance_only": True,
                "match_learned_embed_weight": 0.25,
                "match_learned_embed_decay_scale": 4.0,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
        {
            "label": "lookahead_learned_corr_reappear_only_state_only_warm64",
            "cfg_updates": {
                "batch_size": 16,
                "warmup_transitions": 64,
                "update_every": 1,
                "event_sample_fraction": 1.0,
                "event_heavy_fraction": 1.0,
                "event_heavy_every": 1,
                "event_window_steps": 6,
                "loss_hazard_weight": 3.0,
                "loss_spawn_prior_weight": 0.05,
                "spawn_prior_context_relief": 1.0,
                "pod_spawn_window_steps": 6,
                "spawn_lookahead_steps": 12,
                "loss_spawn_lookahead_weight": 1.0,
                "elapsed_time_scale": 4.0,
                "use_hungarian_matching": True,
                "match_memory_weight": 0.35,
                "match_memory_decay_scale": 12.0,
                "match_memory_reappearance_only": True,
                "match_learned_embed_weight": 0.25,
                "match_learned_embed_reappearance_only": True,
            },
            "bootstrap_kwargs": {"observation_mode": "state_only"},
        },
    ]


def apply_cfg_overrides(policy: CODAPolicy, cfg_updates: Dict[str, Any]) -> None:
    fc = policy.forward_component
    for key, value in cfg_updates.items():
        if hasattr(fc.config, key):
            setattr(fc.config, key, value)
    if "gate_present_threshold" in cfg_updates:
        thr = float(cfg_updates["gate_present_threshold"])
        fc.model.gate_present_threshold = thr
        fc.ema_model.gate_present_threshold = thr
    fc.model.absent_gate_decay = float(np.clip(fc.config.absent_gate_decay, 0.0, 1.0))
    fc.ema_model.absent_gate_decay = float(np.clip(fc.config.absent_gate_decay, 0.0, 1.0))


def _predict_hazard_for_rows(
    policy: CODAPolicy,
    rows: Sequence[Any],
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    fc = policy.forward_component
    model = fc.ema_model
    model.eval()
    all_pred: List[np.ndarray] = []
    all_hazard: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            t = torch.from_numpy(np.stack([np.asarray(r[0], dtype=np.float32) for r in chunk], axis=0)).to(fc.device)
            a = torch.from_numpy(np.stack([np.asarray(r[1], dtype=np.float32) for r in chunk], axis=0)).to(fc.device)
            e = torch.from_numpy(np.stack([np.asarray(r[3], dtype=np.float32) for r in chunk], axis=0)).to(fc.device)
            pred, _, hazard, _ = model.forward_with_aux(t, a, e)
            all_pred.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            all_hazard.append(hazard.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(all_pred, axis=0), np.concatenate(all_hazard, axis=0)


def _cls_stats(true_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(true_mask & pred_mask))
    fp = int(np.sum((~true_mask) & pred_mask))
    fn = int(np.sum(true_mask & (~pred_mask)))
    tn = int(np.sum((~true_mask) & (~pred_mask)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": float(tp / max(1, tp + fn)),
        "fpr": float(fp / max(1, fp + tn)),
    }


def quick_eval_policy(
    policy: CODAPolicy,
    *,
    max_samples: int,
    gate_threshold: float = 0.5,
    pod_spawn_window_steps: int = 6,
) -> Dict[str, Any]:
    raw = policy.forward_component.buffer.state_dict().get("data", [])
    if not raw:
        return {
            "event_detection_transition": {},
            "hazard_diagnostics": {},
            "num_transitions_total": 0,
        }

    n = len(raw)
    sample_n = min(max(1, int(max_samples)), n)
    sample_idx = np.linspace(0, n - 1, num=sample_n, dtype=np.int64)
    rows = [raw[int(i)] for i in sample_idx]
    raw_event_code = np.asarray([int(entry[5]) if len(entry) >= 6 else 0 for entry in raw], dtype=np.int32)
    raw_pod_spawn = ((raw_event_code & 2) != 0).astype(np.bool_)
    raw_pod_spawn_window = raw_pod_spawn.copy()
    w = int(max(0, pod_spawn_window_steps))
    if w > 0 and np.any(raw_pod_spawn):
        pod_idx = np.flatnonzero(raw_pod_spawn)
        for idx in pod_idx:
            lo = max(0, int(idx - w))
            hi = min(n, int(idx + w + 1))
            raw_pod_spawn_window[lo:hi] = True

    pred, hazard = _predict_hazard_for_rows(policy, rows)
    t = np.stack([np.asarray(r[0], dtype=np.float32) for r in rows], axis=0)
    tp1 = np.stack([np.asarray(r[2], dtype=np.float32) for r in rows], axis=0)

    schema = policy.forward_component.schema
    go = 1 + schema.num_types
    prev_gate = t[..., 0]
    true_gate = tp1[..., 0]
    pred_gate = pred[..., 0]
    tgt_type = np.argmax(tp1[..., 1:go], axis=-1)

    true_spawn = (prev_gate <= gate_threshold) & (true_gate > gate_threshold)
    pred_spawn = (prev_gate <= gate_threshold) & (pred_gate > gate_threshold)
    true_off = (prev_gate > gate_threshold) & (true_gate <= gate_threshold)
    pred_off = (prev_gate > gate_threshold) & (pred_gate <= gate_threshold)

    event_metrics_transition = {
        "spawn": _cls_stats(np.any(true_spawn, axis=1), np.any(pred_spawn, axis=1)),
        "disappear": _cls_stats(np.any(true_off, axis=1), np.any(pred_off, axis=1)),
    }

    hazard_spawn_logits = hazard[..., 0] if hazard.shape[-1] >= 1 else np.zeros_like(prev_gate)
    hazard_spawn_prob = 1.0 / (1.0 + np.exp(-np.clip(hazard_spawn_logits, -30.0, 30.0)))
    pod_mask = tgt_type == TYPE_POD
    agent_mask = tgt_type == TYPE_AGENT
    prev_off_mask = prev_gate <= gate_threshold
    sampled_pod_spawn_window = raw_pod_spawn_window[sample_idx].reshape(-1, 1)

    def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(arr[mask])) if np.any(mask) else float("nan")

    hazard_diag = {
        "spawn_prob_pod_prev_off_mean": _masked_mean(hazard_spawn_prob, pod_mask & prev_off_mask),
        "spawn_prob_agent_prev_off_mean": _masked_mean(hazard_spawn_prob, agent_mask & prev_off_mask),
        "spawn_prob_pod_prev_off_near_spawn_mean": _masked_mean(hazard_spawn_prob, pod_mask & prev_off_mask & sampled_pod_spawn_window),
        "spawn_prob_pod_prev_off_far_spawn_mean": _masked_mean(hazard_spawn_prob, pod_mask & prev_off_mask & (~sampled_pod_spawn_window)),
    }
    near = hazard_diag["spawn_prob_pod_prev_off_near_spawn_mean"]
    far = hazard_diag["spawn_prob_pod_prev_off_far_spawn_mean"]
    hazard_diag["spawn_prob_pod_prev_off_spawn_contrast"] = float(near - far) if np.isfinite(near) and np.isfinite(far) else float("nan")
    return {
        "event_detection_transition": event_metrics_transition,
        "hazard_diagnostics": hazard_diag,
        "num_transitions_total": n,
    }


def bootstrap_policy_from_session(
    policy: CODAPolicy,
    *,
    store: SessionStore,
    session_id: str,
    offline_updates: int,
    observation_mode: str = "frame_state",
) -> Dict[str, Any]:
    session = store.load_session(session_id, mmap=True)
    fc = policy.forward_component
    try:
        fc.buffer._data.clear()
    except Exception:
        pass
    fc.mark_reset()
    policy.reset_episode()

    transitions = 0
    obs_mode = str(observation_mode).strip().lower()
    for step in range(int(session.T) - 1):
        snap_t = store.build_step_snapshot(session, step)
        snap_tp1 = store.build_step_snapshot(session, step + 1)
        frame_t = None if obs_mode == "state_only" else session.frames[step]
        frame_tp1 = None if obs_mode == "state_only" else session.frames[step + 1]
        obs_t = build_observation(snap_t, frame_t)
        obs_tp1 = build_observation(snap_tp1, frame_tp1)
        tokens_t, _ = fc.tokenize_observation(obs_t)
        tokens_tp1, _ = fc.tokenize_observation(obs_tp1)
        action = np.asarray(session.actions[step], dtype=np.float32).reshape(-1)
        if action.shape[0] < 2:
            padded = np.zeros((2,), dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded
        else:
            action = action[:2]
        fc.observe_transition(tokens_t, action, tokens_tp1)
        transitions += 1

    applied_updates = 0
    prev_training = bool(fc.training_enabled)
    fc.training_enabled = True
    for _ in range(int(max(0, offline_updates))):
        if fc.maybe_update(force=True) is not None:
            applied_updates += 1
    fc.training_enabled = prev_training
    return {
        "transitions": int(transitions),
        "applied_updates": int(applied_updates),
        "buffer_size": int(len(fc.buffer)),
        "observation_mode": obs_mode,
    }


def compare_on_saved_run(
    *,
    data_root: str,
    session_id: str,
    offline_updates: int = 64,
    eval_samples: int = 192,
    candidates: Sequence[str] | None = None,
    cpu_threads: int = 1,
    policy_seed: int = 17,
) -> List[Dict[str, Any]]:
    configure_cpu_threads(int(cpu_threads))
    store = SessionStore(str(data_root))
    wanted = {str(x).strip() for x in (candidates or []) if str(x).strip()}
    specs = candidate_specs()
    if wanted:
        specs = [cand for cand in specs if cand["label"] in wanted]
    if not specs:
        raise RuntimeError("No matching CODA candidate presets selected.")

    rows: List[Dict[str, Any]] = []
    for cand in specs:
        policy = CODAPolicy(enable_forward_model=True, train_forward_model=False, debug_logging_enabled=False, seed=int(policy_seed))
        apply_cfg_overrides(policy, dict(cand["cfg_updates"]))
        boot = bootstrap_policy_from_session(
            policy,
            store=store,
            session_id=session_id,
            offline_updates=int(offline_updates),
            **dict(cand.get("bootstrap_kwargs", {})),
        )
        metrics = quick_eval_policy(
            policy,
            max_samples=int(eval_samples),
            pod_spawn_window_steps=int(
                max(
                    int(policy.forward_component.config.event_window_steps),
                    int(getattr(policy.forward_component.config, "pod_spawn_window_steps", 0)),
                    6,
                )
            ),
        )
        tr = metrics.get("event_detection_transition", {})
        hz = metrics.get("hazard_diagnostics", {})
        rows.append(
            {
                "label": str(cand["label"]),
                "cfg_updates": copy.deepcopy(cand["cfg_updates"]),
                "bootstrap_kwargs": copy.deepcopy(cand.get("bootstrap_kwargs", {})),
                "session_id": session_id,
                "transitions": int(boot["transitions"]),
                "buffer_size": int(boot["buffer_size"]),
                "applied_updates": int(boot["applied_updates"]),
                "observation_mode": str(boot.get("observation_mode", "frame_state")),
                "spawn_tpr": float(tr.get("spawn", {}).get("tpr", float("nan"))),
                "spawn_fpr": float(tr.get("spawn", {}).get("fpr", float("nan"))),
                "disappear_tpr": float(tr.get("disappear", {}).get("tpr", float("nan"))),
                "spawn_prob_pod_prev_off_mean": float(hz.get("spawn_prob_pod_prev_off_mean", float("nan"))),
                "spawn_prob_pod_prev_off_near_spawn_mean": float(hz.get("spawn_prob_pod_prev_off_near_spawn_mean", float("nan"))),
                "spawn_prob_pod_prev_off_far_spawn_mean": float(hz.get("spawn_prob_pod_prev_off_far_spawn_mean", float("nan"))),
                "spawn_prob_pod_prev_off_spawn_contrast": float(hz.get("spawn_prob_pod_prev_off_spawn_contrast", float("nan"))),
            }
        )

    rows.sort(
        key=lambda row: (
            -np.nan_to_num(float(row.get("spawn_prob_pod_prev_off_spawn_contrast", float("nan"))), nan=-1e9),
            -np.nan_to_num(float(row.get("spawn_tpr", float("nan"))), nan=-1e9),
            -np.nan_to_num(float(row.get("disappear_tpr", float("nan"))), nan=-1e9),
        )
    )
    return rows


def default_cpu_threads(leave_free: int = 2) -> int:
    try:
        total = int(os.cpu_count() or 1)
    except Exception:
        total = 1
    return max(1, total - int(max(0, leave_free)))
