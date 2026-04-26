from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.coda_forward_model import TYPE_AGENT, TYPE_NAMES, TYPE_POD, TYPE_WALL
from agent.coda_policy import CODAPolicy


EPS = 1e-8
TWO_PI = 2.0 * np.pi


@dataclass
class EvalConfig:
    checkpoint: str
    max_samples: int
    horizon: int
    gate_threshold: float
    interaction_eps: float
    event_window: int
    no_plot: bool
    output_json: str


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % TWO_PI - np.pi


def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.nanmean(x)),
        "median": float(np.nanmedian(x)),
        "p95": float(np.nanpercentile(x, 95)),
    }


def _safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(arr[finite]))


def _stack_transitions(
    data: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.stack([np.asarray(r[0], dtype=np.float32) for r in data], axis=0)
    a = np.stack([np.asarray(r[1], dtype=np.float32) for r in data], axis=0)
    tp1 = np.stack([np.asarray(r[2], dtype=np.float32) for r in data], axis=0)
    e = np.stack([np.asarray(r[3], dtype=np.float32) for r in data], axis=0)
    return t, a, tp1, e


def _predict_tokens_and_attention(
    policy: CODAPolicy,
    tokens_t: np.ndarray,
    actions: np.ndarray,
    elapsed_t: np.ndarray,
    *,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    fc = policy.forward_component
    model = fc.ema_model
    model.eval()
    all_preds: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, tokens_t.shape[0], batch_size):
            j = min(tokens_t.shape[0], i + batch_size)
            t = torch.from_numpy(tokens_t[i:j]).to(fc.device)
            a = torch.from_numpy(actions[i:j]).to(fc.device)
            e = torch.from_numpy(elapsed_t[i:j]).to(fc.device)
            out = model.forward_with_aux(t, a, e)
            pred = out[0]
            all_preds.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            attn_t = model.get_last_attention()
            if attn_t is None:
                all_attn.append(np.zeros((j - i, t.shape[1], t.shape[1]), dtype=np.float32))
            else:
                all_attn.append(attn_t.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(all_preds, axis=0), np.concatenate(all_attn, axis=0)


def _predict_tokens_attention_hazard(
    policy: CODAPolicy,
    tokens_t: np.ndarray,
    actions: np.ndarray,
    elapsed_t: np.ndarray,
    *,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc = policy.forward_component
    model = fc.ema_model
    model.eval()
    all_preds: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    all_hazard: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, tokens_t.shape[0], batch_size):
            j = min(tokens_t.shape[0], i + batch_size)
            t = torch.from_numpy(tokens_t[i:j]).to(fc.device)
            a = torch.from_numpy(actions[i:j]).to(fc.device)
            e = torch.from_numpy(elapsed_t[i:j]).to(fc.device)
            out = model.forward_with_aux(t, a, e)
            pred = out[0]
            hazard = out[2] if len(out) >= 3 else None
            all_preds.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            if hazard is None:
                all_hazard.append(np.zeros((j - i, t.shape[1], 2), dtype=np.float32))
            else:
                all_hazard.append(hazard.detach().cpu().numpy().astype(np.float32, copy=False))
            attn_t = model.get_last_attention()
            if attn_t is None:
                all_attn.append(np.zeros((j - i, t.shape[1], t.shape[1]), dtype=np.float32))
            else:
                all_attn.append(attn_t.detach().cpu().numpy().astype(np.float32, copy=False))
    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_attn, axis=0),
        np.concatenate(all_hazard, axis=0),
    )


def _point_to_oriented_rect_dist(
    px: float, py: float, cx: float, cy: float, phi: float, sx: float, sy: float
) -> float:
    dx = px - cx
    dy = py - cy
    c = math.cos(phi)
    s = math.sin(phi)
    rx = c * dx + s * dy
    ry = -s * dx + c * dy
    qx = max(abs(rx) - max(sx, 1e-6), 0.0)
    qy = max(abs(ry) - max(sy, 1e-6), 0.0)
    return float(math.sqrt(qx * qx + qy * qy))


def _event_timing_error(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    true_idx = np.flatnonzero(true_mask)
    pred_idx = np.flatnonzero(pred_mask)
    if true_idx.size == 0 or pred_idx.size == 0:
        return float("nan")
    d = np.abs(true_idx[:, None] - pred_idx[None, :])
    return float(np.mean(np.min(d, axis=1)))


def evaluate(config: EvalConfig) -> Dict[str, Any]:
    policy = CODAPolicy()
    policy.load(config.checkpoint)
    return evaluate_policy(
        policy,
        max_samples=config.max_samples,
        horizon=config.horizon,
        gate_threshold=config.gate_threshold,
        interaction_eps=config.interaction_eps,
        event_window=config.event_window,
    )


def evaluate_policy(
    policy: CODAPolicy,
    *,
    max_samples: int = 1024,
    horizon: int = 20,
    gate_threshold: float = 0.5,
    interaction_eps: float = 0.06,
    event_window: int = 4,
) -> Dict[str, Any]:
    config = EvalConfig(
        checkpoint="",
        max_samples=int(max(1, max_samples)),
        horizon=int(max(1, horizon)),
        gate_threshold=float(gate_threshold),
        interaction_eps=float(max(0.0, interaction_eps)),
        event_window=int(max(0, event_window)),
        no_plot=True,
        output_json="",
    )
    schema = policy.forward_component.schema
    go = 1 + schema.num_types
    wall_start = schema.wall_start
    wall_end = wall_start + schema.num_wall_slots
    agent_start = 0
    agent_end = schema.num_agent_slots

    raw = policy.forward_component.buffer.state_dict().get("data", [])
    if not raw:
        raise RuntimeError("Checkpoint contains no CODA transitions in replay buffer.")

    full_t, full_a, full_tp1, full_e = _stack_transitions(raw)
    n = full_t.shape[0]
    sample_n = min(max(1, config.max_samples), n)
    sample_idx = np.linspace(0, n - 1, num=sample_n, dtype=np.int64)
    t = full_t[sample_idx]
    a = full_a[sample_idx]
    tp1 = full_tp1[sample_idx]
    e = full_e[sample_idx]

    pred, _, _ = _predict_tokens_attention_hazard(policy, t, a, e)
    full_pred, full_attn, full_hazard = _predict_tokens_attention_hazard(policy, full_t, full_a, full_e)

    prev_gate = t[..., 0]
    true_gate = tp1[..., 0]
    pred_gate = pred[..., 0]
    tgt_type = np.argmax(tp1[..., 1:go], axis=-1)
    pred_type = np.argmax(pred[..., 1:go], axis=-1)

    true_geom = tp1[..., go:]
    pred_geom = pred[..., go:]
    prev_geom = t[..., go:]

    pos_err = np.linalg.norm(pred_geom[..., 0:2] - true_geom[..., 0:2], axis=-1)
    vel_true = true_geom[..., 0:2] - prev_geom[..., 0:2]
    vel_pred = pred_geom[..., 0:2] - prev_geom[..., 0:2]
    vel_err = np.linalg.norm(vel_pred - vel_true, axis=-1)
    phi_err = np.abs(_wrap_angle(pred_geom[..., 2] - true_geom[..., 2]))
    scale_err = np.abs(pred_geom[..., 3] - true_geom[..., 3]) + np.abs(pred_geom[..., 4] - true_geom[..., 4])
    geom_l2 = np.linalg.norm(pred_geom - true_geom, axis=-1)

    active = true_gate > config.gate_threshold
    per_type: Dict[str, Dict[str, Dict[str, float]]] = {}
    for t_id in (TYPE_AGENT, TYPE_POD, TYPE_WALL):
        m = active & (tgt_type == t_id)
        per_type[TYPE_NAMES[t_id]] = {
            "position": _safe_stats(pos_err[m]),
            "velocity": _safe_stats(vel_err[m]),
            "orientation": _safe_stats(phi_err[m]),
            "scale": _safe_stats(scale_err[m]),
        }

    true_spawn = (prev_gate <= config.gate_threshold) & (true_gate > config.gate_threshold)
    pred_spawn = (prev_gate <= config.gate_threshold) & (pred_gate > config.gate_threshold)
    true_off = (prev_gate > config.gate_threshold) & (true_gate <= config.gate_threshold)
    pred_off = (prev_gate > config.gate_threshold) & (pred_gate <= config.gate_threshold)
    wall_true_open = true_off & (tgt_type == TYPE_WALL)
    wall_pred_open = pred_off & (pred_type == TYPE_WALL)

    def _cls_stats(tmask: np.ndarray, pmask: np.ndarray) -> Dict[str, float]:
        tp = float(np.sum(tmask & pmask))
        fp = float(np.sum((~tmask) & pmask))
        fn = float(np.sum(tmask & (~pmask)))
        tn = float(np.sum((~tmask) & (~pmask)))
        tpr = tp / max(tp + fn, EPS)
        fpr = fp / max(fp + tn, EPS)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": tpr, "fpr": fpr}

    event_metrics_sample = {
        "spawn": _cls_stats(true_spawn, pred_spawn),
        "disappear": _cls_stats(true_off, pred_off),
        "wall_open": _cls_stats(wall_true_open, wall_pred_open),
    }
    full_prev_gate = full_t[..., 0]
    full_true_gate = full_tp1[..., 0]
    full_pred_gate = full_pred[..., 0]
    full_tgt_type = np.argmax(full_tp1[..., 1:go], axis=-1)
    full_pred_type = np.argmax(full_pred[..., 1:go], axis=-1)
    full_true_spawn = (full_prev_gate <= config.gate_threshold) & (full_true_gate > config.gate_threshold)
    full_pred_spawn = (full_prev_gate <= config.gate_threshold) & (full_pred_gate > config.gate_threshold)
    full_true_off = (full_prev_gate > config.gate_threshold) & (full_true_gate <= config.gate_threshold)
    full_pred_off = (full_prev_gate > config.gate_threshold) & (full_pred_gate <= config.gate_threshold)
    full_wall_true_open = full_true_off & (full_tgt_type == TYPE_WALL)
    full_wall_pred_open = full_pred_off & (full_pred_type == TYPE_WALL)
    event_metrics = {
        "spawn": _cls_stats(full_true_spawn, full_pred_spawn),
        "disappear": _cls_stats(full_true_off, full_pred_off),
        "wall_open": _cls_stats(full_wall_true_open, full_wall_pred_open),
    }
    # Per-transition (any-slot) event stats complement per-slot counts.
    full_true_spawn_step = np.any(full_true_spawn, axis=1)
    full_pred_spawn_step = np.any(full_pred_spawn, axis=1)
    full_true_off_step = np.any(full_true_off, axis=1)
    full_pred_off_step = np.any(full_pred_off, axis=1)
    full_true_wall_open_step = np.any(full_wall_true_open, axis=1)
    full_pred_wall_open_step = np.any(full_wall_pred_open, axis=1)
    event_metrics_transition = {
        "spawn": _cls_stats(full_true_spawn_step, full_pred_spawn_step),
        "disappear": _cls_stats(full_true_off_step, full_pred_off_step),
        "wall_open": _cls_stats(full_true_wall_open_step, full_pred_wall_open_step),
    }
    token_supervision = {
        "sample_pod_drop_count": int(np.sum(true_off & (tgt_type == TYPE_POD))),
        "sample_wall_drop_count": int(np.sum(true_off & (tgt_type == TYPE_WALL))),
        "sample_pod_spawn_count": int(np.sum(true_spawn & (tgt_type == TYPE_POD))),
        "sample_wall_spawn_count": int(np.sum(true_spawn & (tgt_type == TYPE_WALL))),
        "full_pod_drop_count": int(np.sum(full_true_off & (full_tgt_type == TYPE_POD))),
        "full_wall_drop_count": int(np.sum(full_true_off & (full_tgt_type == TYPE_WALL))),
        "full_pod_spawn_count": int(np.sum(full_true_spawn & (full_tgt_type == TYPE_POD))),
        "full_wall_spawn_count": int(np.sum(full_true_spawn & (full_tgt_type == TYPE_WALL))),
        "full_spawn_transition_count": int(np.sum(full_true_spawn_step)),
        "full_disappear_transition_count": int(np.sum(full_true_off_step)),
        "full_wall_open_transition_count": int(np.sum(full_true_wall_open_step)),
    }

    # Hazard diagnostics: helps detect event-head saturation.
    hazard_spawn_logits = full_hazard[..., 0] if full_hazard.shape[-1] >= 1 else np.zeros_like(full_prev_gate)
    if full_hazard.shape[-1] >= 2:
        hazard_despawn_logits = full_hazard[..., 1]
    else:
        hazard_despawn_logits = -hazard_spawn_logits
    hazard_spawn_prob = 1.0 / (1.0 + np.exp(-np.clip(hazard_spawn_logits, -30.0, 30.0)))
    hazard_despawn_prob = 1.0 / (1.0 + np.exp(-np.clip(hazard_despawn_logits, -30.0, 30.0)))

    def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(arr[mask]))

    pod_mask = full_tgt_type == TYPE_POD
    wall_mask = full_tgt_type == TYPE_WALL
    agent_mask = full_tgt_type == TYPE_AGENT
    prev_on_mask = full_prev_gate > config.gate_threshold
    prev_off_mask = ~prev_on_mask
    hazard_diag = {
        "spawn_logit_pod_prev_off_mean": _masked_mean(hazard_spawn_logits, pod_mask & prev_off_mask),
        "spawn_prob_pod_prev_off_mean": _masked_mean(hazard_spawn_prob, pod_mask & prev_off_mask),
        "despawn_logit_pod_prev_on_mean": _masked_mean(hazard_despawn_logits, pod_mask & prev_on_mask),
        "despawn_prob_pod_prev_on_mean": _masked_mean(hazard_despawn_prob, pod_mask & prev_on_mask),
        "spawn_logit_wall_prev_off_mean": _masked_mean(hazard_spawn_logits, wall_mask & prev_off_mask),
        "spawn_prob_wall_prev_off_mean": _masked_mean(hazard_spawn_prob, wall_mask & prev_off_mask),
        "despawn_logit_wall_prev_on_mean": _masked_mean(hazard_despawn_logits, wall_mask & prev_on_mask),
        "despawn_prob_wall_prev_on_mean": _masked_mean(hazard_despawn_prob, wall_mask & prev_on_mask),
        "spawn_prob_agent_prev_off_mean": _masked_mean(hazard_spawn_prob, agent_mask & prev_off_mask),
        "despawn_prob_agent_prev_on_mean": _masked_mean(hazard_despawn_prob, agent_mask & prev_on_mask),
    }

    timing_by_slot = []
    for s in range(schema.max_slots):
        timing_by_slot.append(
            {
                "spawn": _event_timing_error(true_spawn[:, s], pred_spawn[:, s]),
                "disappear": _event_timing_error(true_off[:, s], pred_off[:, s]),
            }
        )
    event_timing_error = {
        "spawn_mean_steps": _safe_nanmean([r["spawn"] for r in timing_by_slot]),
        "disappear_mean_steps": _safe_nanmean([r["disappear"] for r in timing_by_slot]),
    }

    # Interaction-conditioned errors (agent vs wall proximity)
    agent_true = tp1[:, agent_start:agent_end, go:]
    agent_pred = pred[:, agent_start:agent_end, go:]
    agent_gate = tp1[:, agent_start:agent_end, 0] > config.gate_threshold
    wall_gate = t[:, wall_start:wall_end, 0] > config.gate_threshold
    wall_geom = t[:, wall_start:wall_end, go:]
    near = np.zeros((sample_n,), dtype=bool)
    epos_agent = np.full((sample_n,), np.nan, dtype=np.float32)
    for i in range(sample_n):
        if not np.any(agent_gate[i]):
            continue
        a_idx = int(np.argmax(t[i, agent_start:agent_end, 0]))
        ax, ay = agent_true[i, a_idx, 0], agent_true[i, a_idx, 1]
        px, py = agent_pred[i, a_idx, 0], agent_pred[i, a_idx, 1]
        epos_agent[i] = float(np.linalg.norm([px - ax, py - ay]))
        dmin = float("inf")
        for w in range(schema.num_wall_slots):
            if not wall_gate[i, w]:
                continue
            x, y, phi, sx, sy, wt = wall_geom[i, w]
            d = _point_to_oriented_rect_dist(float(ax), float(ay), float(x), float(y), float(phi), float(sx), float(min(sy, wt)))
            dmin = min(dmin, d)
        near[i] = dmin <= config.interaction_eps if np.isfinite(dmin) else False
    interaction_metrics = {
        "interaction_pos_error": _safe_stats(epos_agent[near & np.isfinite(epos_agent)]),
        "no_interaction_pos_error": _safe_stats(epos_agent[(~near) & np.isfinite(epos_agent)]),
        "interaction_fraction": float(np.mean(near.astype(np.float32))),
    }

    # Counterfactual wall-present vs wall-absent on sampled transitions.
    t_present = t.copy()
    t_absent = t.copy()
    t_present[:, wall_start:wall_end, 0] = 1.0
    t_absent[:, wall_start:wall_end, 0] = 0.0
    pred_present, _ = _predict_tokens_and_attention(policy, t_present, a, e)
    pred_absent, _ = _predict_tokens_and_attention(policy, t_absent, a, e)
    cf_agent = pred_present[:, agent_start:agent_end, go : go + 2] - pred_absent[:, agent_start:agent_end, go : go + 2]
    cf_delta = np.linalg.norm(cf_agent, axis=-1).reshape(-1)
    counterfactual = _safe_stats(cf_delta[np.isfinite(cf_delta)])

    # Multi-step rollout drift.
    max_h = max(1, config.horizon)
    starts = np.linspace(0, max(0, n - max_h - 1), num=min(24, max(1, n - max_h)), dtype=np.int64)
    drift = np.full((max_h,), np.nan, dtype=np.float32)
    if starts.size > 0:
        m_batch = full_t[starts].copy()
        elapsed_batch = full_e[starts].copy()
        for h in range(max_h):
            idx = starts + h
            actions_h = full_a[idx]
            pred_batch, _ = _predict_tokens_and_attention(policy, m_batch, actions_h, elapsed_batch, batch_size=64)
            tgt_batch = full_tp1[idx]
            g = tgt_batch[..., 0] > config.gate_threshold
            step_vals: List[float] = []
            for r in range(pred_batch.shape[0]):
                if np.any(g[r]):
                    step_vals.append(float(np.mean(np.linalg.norm((pred_batch[r, :, go:] - tgt_batch[r, :, go:])[g[r]], axis=-1))))
            if step_vals:
                drift[h] = float(np.mean(step_vals))

            prev_on = m_batch[..., 0] > config.gate_threshold
            curr_on = pred_batch[..., 0] > config.gate_threshold
            elapsed_batch = elapsed_batch.copy()
            elapsed_batch[prev_on & curr_on] = 0.0
            elapsed_batch[(~prev_on) & curr_on] = 0.0
            elapsed_batch[prev_on & (~curr_on)] = 1.0
            elapsed_batch[(~prev_on) & (~curr_on)] += 1.0
            m_batch = pred_batch

    # Identity consistency (non-identity assignment rate).
    # In fixed-slot mode, non-identity assignment indicates frequent slot flip pressure.
    non_identity = []
    change_rate = []
    prev_perm = None
    for i in range(sample_n):
        p = pred_geom[i]
        q = true_geom[i]
        cost = np.linalg.norm(p[:, None, 0:2] - q[None, :, 0:2], axis=-1)
        perm = np.argmin(cost, axis=1)
        non_identity.append(float(np.mean(perm != np.arange(schema.max_slots))))
        if prev_perm is not None:
            change_rate.append(float(np.mean(perm != prev_perm)))
        prev_perm = perm
    identity_metrics = {
        "non_identity_rate": float(np.mean(non_identity)) if non_identity else float("nan"),
        "assignment_change_rate": float(np.mean(change_rate)) if change_rate else float("nan"),
    }

    # Rare event robustness (sampled subset).
    event_geom = {
        "wall_disappear_geom_error": _safe_stats(geom_l2[wall_true_open]),
        "pod_spawn_geom_error": _safe_stats(geom_l2[true_spawn & (tgt_type == TYPE_POD)]),
        "pod_disappear_geom_error": _safe_stats(geom_l2[true_off & (tgt_type == TYPE_POD)]),
    }
    # Rare event robustness on full buffer to avoid sample sparsity hiding rare pods.
    full_true_geom = full_tp1[..., go:]
    full_pred_geom = full_pred[..., go:]
    full_geom_l2 = np.linalg.norm(full_pred_geom - full_true_geom, axis=-1)
    full_event_geom = {
        "wall_disappear_geom_error": _safe_stats(full_geom_l2[full_wall_true_open]),
        "pod_spawn_geom_error": _safe_stats(full_geom_l2[full_true_spawn & (full_tgt_type == TYPE_POD)]),
        "pod_disappear_geom_error": _safe_stats(full_geom_l2[full_true_off & (full_tgt_type == TYPE_POD)]),
    }

    # Attention attribution around wall disappearance.
    agent_to_wall = np.mean(full_attn[:, agent_start:agent_end, wall_start:wall_end], axis=(1, 2))
    wall_event_frames = np.any(
        ((full_t[..., 0] > config.gate_threshold) & (full_tp1[..., 0] <= config.gate_threshold) & (np.argmax(full_tp1[..., 1:go], axis=-1) == TYPE_WALL)),
        axis=1,
    )
    if np.any(wall_event_frames):
        event_idx = np.flatnonzero(wall_event_frames)
        event_mask = np.zeros_like(wall_event_frames)
        for idx in event_idx:
            lo = max(0, int(idx - config.event_window))
            hi = min(n, int(idx + config.event_window + 1))
            event_mask[lo:hi] = True
        base_mask = ~event_mask
    else:
        event_mask = np.zeros_like(wall_event_frames)
        base_mask = np.ones_like(wall_event_frames)
    attention_metrics = {
        "agent_to_wall_baseline": float(np.mean(agent_to_wall[base_mask])) if np.any(base_mask) else float("nan"),
        "agent_to_wall_event_window": float(np.mean(agent_to_wall[event_mask])) if np.any(event_mask) else float("nan"),
        "agent_to_wall_delta": (
            float(np.mean(agent_to_wall[event_mask]) - np.mean(agent_to_wall[base_mask]))
            if np.any(event_mask) and np.any(base_mask)
            else float("nan")
        ),
    }

    # Architectural displacement cap check.
    disp_true = np.linalg.norm(full_tp1[..., go : go + 2] - full_t[..., go : go + 2], axis=-1)
    active_disp = (full_t[..., 0] > config.gate_threshold) & (full_tp1[..., 0] > config.gate_threshold)
    max_true_disp = float(np.max(disp_true[active_disp])) if np.any(active_disp) else float("nan")
    model = policy.forward_component.model
    if hasattr(model, "jump_head"):
        # With jump head + absolute XY pathway, effective per-step XY range is the world range.
        max_model_disp = float(2.0 * np.sqrt(2.0))
        model_disp_mode = "jump_abs_xy"
    else:
        max_model_disp = float(getattr(model, "max_xy_step_scale", 0.25))
        model_disp_mode = "bounded_delta_xy"
    arch = {
        "max_true_disp": max_true_disp,
        "max_model_disp": max_model_disp,
        "model_disp_mode": model_disp_mode,
        "constraint_violated": bool(np.isfinite(max_true_disp) and max_true_disp > max_model_disp),
    }

    # Data collection sanity checks: distinguish transition motion from
    # trajectory continuity.
    if n >= 2:
        transition_delta = np.linalg.norm((full_tp1[..., go : go + 2] - full_t[..., go : go + 2]).reshape(n, -1), axis=-1)
        link_delta = np.linalg.norm((full_tp1[:-1] - full_t[1:]).reshape(n - 1, -1), axis=-1)
        data_quality = {
            "transition_l2_mean": float(np.mean(transition_delta)),
            "transition_l2_p95": float(np.percentile(transition_delta, 95)),
            "transition_close_frac_lt_0p25": float(np.mean(transition_delta < 0.25)),
            "adjacent_link_l2_mean": float(np.mean(link_delta)),
            "adjacent_link_l2_p95": float(np.percentile(link_delta, 95)),
            "adjacent_link_close_frac_lt_0p25": float(np.mean(link_delta < 0.25)),
        }
    else:
        data_quality = {
            "transition_l2_mean": float("nan"),
            "transition_l2_p95": float("nan"),
            "transition_close_frac_lt_0p25": float("nan"),
            "adjacent_link_l2_mean": float("nan"),
            "adjacent_link_l2_p95": float("nan"),
            "adjacent_link_close_frac_lt_0p25": float("nan"),
        }

    result: Dict[str, Any] = {
        "num_transitions_total": int(n),
        "num_transitions_sampled": int(sample_n),
        "token_supervision": token_supervision,
        "one_step_errors": per_type,
        "event_detection": event_metrics,
        "event_detection_transition": event_metrics_transition,
        "event_detection_sample": event_metrics_sample,
        "event_prediction_activity": {
            "full_spawn_pred_positive": int(np.sum(full_pred_spawn)),
            "full_disappear_pred_positive": int(np.sum(full_pred_off)),
            "full_wall_open_pred_positive": int(np.sum(full_wall_pred_open)),
            "full_spawn_pred_transition_positive": int(np.sum(full_pred_spawn_step)),
            "full_disappear_pred_transition_positive": int(np.sum(full_pred_off_step)),
            "full_wall_open_pred_transition_positive": int(np.sum(full_pred_wall_open_step)),
        },
        "hazard_diagnostics": hazard_diag,
        "event_timing_error": event_timing_error,
        "interaction_sensitivity": interaction_metrics,
        "counterfactual_agent_delta": counterfactual,
        "rollout_drift_by_horizon": [float(v) for v in drift],
        "identity_consistency": identity_metrics,
        "rare_event_robustness": event_geom,
        "rare_event_robustness_full": full_event_geom,
        "attention_attribution": attention_metrics,
        "architecture_constraint": arch,
        "data_quality": data_quality,
    }
    return result


def _plot_dashboard(metrics: Dict[str, Any], title: str):
    drift = np.asarray(metrics.get("rollout_drift_by_horizon", []), dtype=np.float32)
    per = metrics.get("one_step_errors", {})
    ev = metrics.get("event_detection", {})
    inter = metrics.get("interaction_sensitivity", {})
    cf = metrics.get("counterfactual_agent_delta", {})
    rare = metrics.get("rare_event_robustness", {})
    attn = metrics.get("attention_attribution", {})
    arch = metrics.get("architecture_constraint", {})

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(title)

    types = ["agent", "pod", "wall"]
    pos_mean = [per.get(t, {}).get("position", {}).get("mean", np.nan) for t in types]
    pos_p95 = [per.get(t, {}).get("position", {}).get("p95", np.nan) for t in types]
    ax = axes[0, 0]
    x = np.arange(len(types))
    ax.bar(x - 0.15, pos_mean, width=0.3, label="mean")
    ax.bar(x + 0.15, pos_p95, width=0.3, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_title("Per-Type Position Error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 1]
    labels = ["spawn", "disappear", "wall_open"]
    tprs = [ev.get(k, {}).get("tpr", np.nan) for k in labels]
    fprs = [ev.get(k, {}).get("fpr", np.nan) for k in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, tprs, width=0.3, label="TPR")
    ax.bar(x + 0.15, fprs, width=0.3, label="FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Event Detection")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 2]
    ax.bar(["no-int", "int"], [inter.get("no_interaction_pos_error", {}).get("mean", np.nan), inter.get("interaction_pos_error", {}).get("mean", np.nan)])
    ax.set_title("Interaction-Conditioned Agent Pos Error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if drift.size > 0:
        ax.plot(np.arange(1, drift.size + 1), drift, marker="o")
    ax.set_title("Rollout Drift vs Horizon")
    ax.set_xlabel("Horizon k")
    ax.set_ylabel("Mean geom drift")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(["CF delta"], [cf.get("mean", np.nan)])
    ax.set_title("Counterfactual Agent Delta (Wall On vs Off)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.bar(
        ["wall_disappear", "pod_spawn", "pod_disappear"],
        [
            rare.get("wall_disappear_geom_error", {}).get("mean", np.nan),
            rare.get("pod_spawn_geom_error", {}).get("mean", np.nan),
            rare.get("pod_disappear_geom_error", {}).get("mean", np.nan),
        ],
    )
    ax.set_xticklabels(["wall_disappear", "pod_spawn", "pod_disappear"], rotation=20)
    ax.set_title("Rare Event Geom Error")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.bar(["baseline", "event"], [attn.get("agent_to_wall_baseline", np.nan), attn.get("agent_to_wall_event_window", np.nan)])
    ax.set_title("Attention Agent->Wall")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.axis("off")
    text = (
        f"num transitions: {metrics.get('num_transitions_total', 0)}\n"
        f"arch max_true_disp: {arch.get('max_true_disp', np.nan):.4f}\n"
        f"arch max_model_disp: {arch.get('max_model_disp', np.nan):.4f}\n"
        f"constraint violated: {arch.get('constraint_violated', False)}\n"
        f"identity non-id rate: {metrics.get('identity_consistency', {}).get('non_identity_rate', np.nan):.4f}\n"
        f"assignment change rate: {metrics.get('identity_consistency', {}).get('assignment_change_rate', np.nan):.4f}\n"
    )
    ax.text(0.01, 0.95, text, va="top", ha="left", family="monospace")
    ax.set_title("Architecture / Identity Checks")

    ax = axes[2, 2]
    ax.axis("off")
    ax.text(
        0.01,
        0.95,
        json.dumps(metrics.get("event_timing_error", {}), indent=2),
        va="top",
        ha="left",
        family="monospace",
    )
    ax.set_title("Event Timing Error")

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CODA forward model behavior from checkpoint replay buffer.")
    parser.add_argument("--checkpoint", default="data/checkpoints/coda_latest.npz", help="Path to CODA checkpoint.")
    parser.add_argument("--max-samples", type=int, default=1024, help="Max transitions used for one-step metrics.")
    parser.add_argument("--horizon", type=int, default=20, help="Rollout horizon for drift.")
    parser.add_argument("--gate-threshold", type=float, default=0.5, help="Gate threshold for presence/event decisions.")
    parser.add_argument("--interaction-eps", type=float, default=0.06, help="Distance threshold for interaction grouping.")
    parser.add_argument("--event-window", type=int, default=4, help="Window around event frames for attention aggregation.")
    parser.add_argument("--no-plot", action="store_true", help="Skip dashboard plot.")
    parser.add_argument("--output-json", default="", help="Optional path to save metrics JSON.")
    args = parser.parse_args()

    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        max_samples=int(max(1, args.max_samples)),
        horizon=int(max(1, args.horizon)),
        gate_threshold=float(args.gate_threshold),
        interaction_eps=float(max(0.0, args.interaction_eps)),
        event_window=int(max(0, args.event_window)),
        no_plot=bool(args.no_plot),
        output_json=str(args.output_json).strip(),
    )
    metrics = evaluate(cfg)

    print(json.dumps(metrics, indent=2))
    if cfg.output_json:
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if not cfg.no_plot:
        _plot_dashboard(metrics, title="CODA Forward Model Evaluation")


if __name__ == "__main__":
    main()
