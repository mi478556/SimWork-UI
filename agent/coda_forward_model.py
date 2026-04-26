from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
from contextlib import nullcontext
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6
TWO_PI = 2.0 * np.pi
GEOM_HUBER_DELTA = 0.05

TYPE_AGENT = 0
TYPE_POD = 1
TYPE_WALL = 2
TYPE_CONDITION = 3
TYPE_EMPTY = 4

TYPE_NAMES = {
    TYPE_AGENT: "agent",
    TYPE_POD: "pod",
    TYPE_WALL: "wall",
    TYPE_CONDITION: "condition",
    TYPE_EMPTY: "empty",
}


@dataclass(frozen=True)
class CodaTokenSchema:
    num_agent_slots: int = 1
    num_pod_slots: int = 2
    num_wall_slots: int = 4
    num_condition_slots: int = 0
    num_slack_slots: int = 3
    num_types: int = 5
    geom_dim: int = 6

    @property
    def max_slots(self) -> int:
        return self.num_agent_slots + self.num_pod_slots + self.num_wall_slots + self.num_condition_slots + self.num_slack_slots

    @property
    def token_dim(self) -> int:
        return 1 + self.num_types + self.geom_dim

    @property
    def wall_start(self) -> int:
        return self.num_agent_slots + self.num_pod_slots

    @property
    def condition_start(self) -> int:
        return self.wall_start + self.num_wall_slots


@dataclass
class CodaDynamicsConfig:
    buffer_capacity: int = 4096
    batch_size: int = 32
    warmup_transitions: int = 128
    update_every: int = 4
    hidden_dim: int = 128
    num_layers: int = 3
    lr: float = 3e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    loss_gate_weight: float = 1.0
    loss_type_weight: float = 0.75
    loss_geom_weight: float = 4.0
    loss_hazard_weight: float = 1.0
    track_inertia_weight: float = 0.25
    use_uncertainty_head: bool = True
    geom_logvar_min: float = -6.0
    geom_logvar_max: float = 3.0
    gate_present_threshold: float = 0.5
    elapsed_time_scale: float = 10.0
    spawn_geom_weight_scale: float = 0.25
    use_hungarian_matching: bool = False
    absent_gate_decay: float = 0.5
    loss_despawn_weight: float = 1.0
    loss_collision_weight: float = 2.0
    loss_contact_drop_weight: float = 2.0
    event_transition_weight: float = 3.0
    min_xy_step_scale: float = 0.05
    max_xy_step_scale: float = 1.0
    min_phi_step_scale: float = 0.2
    max_phi_step_scale: float = 3.141592653589793
    min_size_step_scale: float = 0.02
    max_size_step_scale: float = 0.6
    min_wt_step_scale: float = 0.01
    max_wt_step_scale: float = 0.4
    event_sample_fraction: float = 0.45
    event_heavy_every: int = 4
    event_heavy_fraction: float = 0.7
    event_window_steps: int = 3
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    contact_condition_weight: float = 2.0
    loss_spawn_prior_weight: float = 0.05
    spawn_prior_context_relief: float = 0.0
    pod_spawn_window_steps: int = 0
    spawn_lookahead_steps: int = 0
    loss_spawn_lookahead_weight: float = 0.0
    match_memory_weight: float = 0.0
    match_memory_decay_scale: float = 12.0
    match_memory_reappearance_only: bool = False
    match_learned_embed_weight: float = 0.0
    match_learned_embed_decay_scale: float = 12.0
    match_learned_embed_reappearance_only: bool = False
    loss_jump_weight: float = 1.0
    jump_disp_threshold: float = 0.45
    collision_use_target_wall: bool = True
    train_type_dynamics: bool = False
    gate_transition_weighted: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"


def _to_rgb_frame(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None

    arr = np.asarray(frame)
    if arr.size == 0:
        return None

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, repeats=3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] < 3:
        return None

    arr = arr[:, :, :3].astype(np.float32, copy=False)
    mx = float(arr.max()) if arr.size > 0 else 1.0
    if mx > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _world_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    xw = ((xx.astype(np.float32) + 0.5) / float(max(1, width))) * 2.0 - 1.0
    yw = ((yy.astype(np.float32) + 0.5) / float(max(1, height))) * 2.0 - 1.0
    return xw, yw


def _argmax_type(tokens: np.ndarray, schema: CodaTokenSchema) -> np.ndarray:
    return np.argmax(tokens[:, 1 : 1 + schema.num_types], axis=1)


def _pack_token(
    schema: CodaTokenSchema,
    gate: float,
    token_type: int,
    geometry: np.ndarray,
) -> np.ndarray:
    token = np.zeros((schema.token_dim,), dtype=np.float32)
    type_vec = np.zeros((schema.num_types,), dtype=np.float32)
    type_vec[int(np.clip(token_type, 0, schema.num_types - 1))] = 1.0

    token[0] = float(np.clip(gate, 0.0, 1.0))
    token[1 : 1 + schema.num_types] = type_vec
    token[1 + schema.num_types :] = geometry.astype(np.float32, copy=False)
    return token


class CodaConditionTokenizer:
    """
    Translates internal/body state into condition tokens.

    Slot 0: stomach token
    Slot 1: pain token
    """

    def __init__(self, schema: CodaTokenSchema):
        self.schema = schema
        self.prev_stomach = 0.4
        self.recent_food_trace = 0.0
        self.prev_pain = 0.0
        self.food_trace_decay = 0.96
        self.stomach_baseline = 0.4
        self.stomach_max = 1.6
        self.pain_profile_variant = "comfort_well_distance"

    def set_pain_profile_variant(self, variant: str) -> None:
        name = str(variant).strip().lower()
        if not name:
            name = "comfort_well_distance"
        self.pain_profile_variant = name

    def reset_state(self):
        self.prev_stomach = float(self.stomach_baseline)
        self.recent_food_trace = 0.0
        self.prev_pain = 0.0

    def tokenize(self, observation: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if int(self.schema.num_condition_slots) <= 0:
            return [], {"active_slots": 0}

        stomach = float(observation.get("stomach", self.prev_stomach))
        stomach_delta = float(stomach - self.prev_stomach)
        food_event = 1.0 if stomach_delta > 1e-4 else 0.0
        self.recent_food_trace = float(self.food_trace_decay * self.recent_food_trace + food_event)

        stomach_norm = float(np.clip((stomach - self.stomach_baseline) / max(1e-6, self.stomach_max - self.stomach_baseline), 0.0, 1.0))
        delta_scaled = float(np.clip(0.5 + 1.25 * stomach_delta, 0.0, 1.0))
        center = 0.875
        span = 0.225
        low_drive = float(np.clip((0.70 - stomach) / max(1e-6, 0.70 - 0.20), 0.0, 1.0))
        high_drive = float(np.clip((stomach - 1.05) / max(1e-6, 1.55 - 1.05), 0.0, 1.0))
        dist = abs((stomach - center) / max(1e-6, span))
        comfort_pain = float(np.clip((dist - 1.0) / 1.6, 0.0, 1.0))
        comfort_membership = float(np.clip(1.0 - comfort_pain, 0.0, 1.0))

        stomach_token = np.array(
            [stomach_norm, delta_scaled, low_drive, high_drive, comfort_membership, comfort_pain],
            dtype=np.float32,
        )

        side_emphasis = float(np.clip(max(low_drive, high_drive), 0.0, 1.0))
        variant = str(getattr(self, "pain_profile_variant", "comfort_well_distance")).strip().lower()
        if variant == "fullness_quadratic":
            high_term = float(np.clip(high_drive * high_drive, 0.0, 1.0))
            low_term = float(np.clip(low_drive, 0.0, 1.0))
            side_term = float(np.clip(max(low_term, high_term), 0.0, 1.0))
            pain_drive = float(np.clip(comfort_pain * (0.60 + 0.40 * side_term), 0.0, 1.0))
        elif variant == "fullness_sigmoid":
            fullness = float(1.0 / (1.0 + np.exp(-(stomach - 1.22) / 0.08)))
            side_term = float(np.clip(max(low_drive, fullness), 0.0, 1.0))
            pain_drive = float(np.clip(comfort_pain * (0.55 + 0.45 * side_term), 0.0, 1.0))
        elif variant == "fullness_food_trace":
            high_term = float(np.clip(0.55 * high_drive + 0.45 * high_drive * np.clip(self.recent_food_trace, 0.0, 1.0), 0.0, 1.0))
            side_term = float(np.clip(max(low_drive, high_term), 0.0, 1.0))
            pain_drive = float(np.clip(comfort_pain * (0.60 + 0.40 * side_term), 0.0, 1.0))
        else:
            pain_drive = float(np.clip(comfort_pain * (0.65 + 0.35 * side_emphasis), 0.0, 1.0))
        emitted_links = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        pain_intensity = float(
            np.clip(
                pain_drive,
                0.0,
                1.0,
            )
        )
        pain_persistence = float(np.clip(0.7 * self.prev_pain + 0.3 * pain_intensity, 0.0, 1.0))
        pain_token = np.array(
            [
                pain_intensity,
                float(emitted_links[0]),
                float(emitted_links[1]),
                float(emitted_links[2]),
                float(emitted_links[3]),
                pain_persistence,
            ],
            dtype=np.float32,
        )

        self.prev_stomach = float(stomach)
        self.prev_pain = float(pain_intensity)

        candidates = [stomach_token]
        if int(self.schema.num_condition_slots) >= 2:
            candidates.append(pain_token)
        while len(candidates) < int(self.schema.num_condition_slots):
            candidates.append(np.zeros((self.schema.geom_dim,), dtype=np.float32))

        diag = {
            "active_slots": int(min(len(candidates), self.schema.num_condition_slots)),
            "stomach_norm": stomach_norm,
            "stomach_delta": stomach_delta,
            "recent_food_trace": float(self.recent_food_trace),
            "pain_intensity": pain_intensity,
            "pain_links": {
                "stomach": float(emitted_links[0]),
                "recent_food": float(emitted_links[1]),
                "agent_context": float(emitted_links[2]),
                "unknown": float(emitted_links[3]),
            },
            "pain_profile_variant": str(variant),
        }
        return candidates[: int(self.schema.num_condition_slots)], diag


class CodaGeometryTokenizer:
    def __init__(self, schema: CodaTokenSchema):
        self.schema = schema
        self.condition_tokenizer = CodaConditionTokenizer(schema)

        self.min_scale = 0.01
        self.max_scale = 1.2
        self.min_thickness = 0.005
        self.max_thickness = 0.4
        self.default_wall_thickness = 0.015

        self.min_pixels_agent = 4
        self.min_pixels_pod = 4
        self.min_pixels_wall = 12

        self._slot_types = (
            [TYPE_AGENT] * self.schema.num_agent_slots
            + [TYPE_POD] * self.schema.num_pod_slots
            + [TYPE_WALL] * self.schema.num_wall_slots
            + [TYPE_CONDITION] * self.schema.num_condition_slots
            + [TYPE_EMPTY] * self.schema.num_slack_slots
        )
        self._last_geometry = np.zeros((self.schema.max_slots, self.schema.geom_dim), dtype=np.float32)
        self._initialize_default_geometries()

    def _initialize_default_geometries(self):
        for idx, t in enumerate(self._slot_types):
            if t == TYPE_AGENT:
                self._last_geometry[idx] = np.array([-0.5, 0.0, 0.0, 0.04, 0.04, 0.04], dtype=np.float32)
            elif t == TYPE_POD:
                self._last_geometry[idx] = np.array([0.0, 0.0, 0.0, 0.06, 0.06, 0.06], dtype=np.float32)
            elif t == TYPE_WALL:
                self._last_geometry[idx] = np.array([0.0, 0.0, 1.5708, 0.02, 0.7, 0.02], dtype=np.float32)
            elif t == TYPE_CONDITION:
                self._last_geometry[idx] = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05], dtype=np.float32)
            else:
                self._last_geometry[idx] = np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.02], dtype=np.float32)

    def reset_state(self):
        self._last_geometry.fill(0.0)
        self._initialize_default_geometries()
        try:
            self.condition_tokenizer.reset_state()
        except Exception:
            pass

    def _clamp_geometry(self, geometry: np.ndarray) -> np.ndarray:
        g = np.asarray(geometry, dtype=np.float32).reshape(-1)
        if g.shape[0] < self.schema.geom_dim:
            pad = np.zeros((self.schema.geom_dim,), dtype=np.float32)
            pad[: g.shape[0]] = g
            g = pad
        g = g[: self.schema.geom_dim].astype(np.float32, copy=True)

        g[0] = float(np.clip(g[0], -1.0, 1.0))
        g[1] = float(np.clip(g[1], -1.0, 1.0))
        g[2] = float(np.clip(g[2], -np.pi, np.pi))
        g[3] = float(np.clip(abs(g[3]), self.min_scale, self.max_scale))
        g[4] = float(np.clip(abs(g[4]), self.min_scale, self.max_scale))
        g[5] = float(np.clip(abs(g[5]), self.min_thickness, self.max_thickness))
        return g

    def _extract_masks(self, rgb_frame: np.ndarray) -> Dict[str, np.ndarray]:
        r = rgb_frame[:, :, 0]
        g = rgb_frame[:, :, 1]
        b = rgb_frame[:, :, 2]

        agent_mask = (g > 0.65) & (r < 0.35) & (b < 0.35)
        pod_mask = (r > 0.65) & (g > 0.65) & (b < 0.35)
        wall_mask = (r > 0.65) & (g > 0.65) & (b > 0.65)

        return {
            "agent": agent_mask,
            "pod": pod_mask,
            "wall": wall_mask,
        }

    def _connected_components(self, mask: np.ndarray, min_pixels: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if mask.ndim != 2:
            return []

        h, w = mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.nonzero(mask)
        comps: List[Tuple[np.ndarray, np.ndarray]] = []

        for idx in range(len(ys)):
            y0 = int(ys[idx])
            x0 = int(xs[idx])
            if visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = 1
            cy: List[int] = []
            cx: List[int] = []

            while stack:
                y, x = stack.pop()
                cy.append(y)
                cx.append(x)

                y_up = y - 1
                y_dn = y + 1
                x_lt = x - 1
                x_rt = x + 1

                if y_up >= 0 and mask[y_up, x] and not visited[y_up, x]:
                    visited[y_up, x] = 1
                    stack.append((y_up, x))
                if y_dn < h and mask[y_dn, x] and not visited[y_dn, x]:
                    visited[y_dn, x] = 1
                    stack.append((y_dn, x))
                if x_lt >= 0 and mask[y, x_lt] and not visited[y, x_lt]:
                    visited[y, x_lt] = 1
                    stack.append((y, x_lt))
                if x_rt < w and mask[y, x_rt] and not visited[y, x_rt]:
                    visited[y, x_rt] = 1
                    stack.append((y, x_rt))

            if len(cy) >= int(min_pixels):
                comps.append(
                    (
                        np.asarray(cy, dtype=np.int32),
                        np.asarray(cx, dtype=np.int32),
                    )
                )

        return comps

    def _component_to_geometry(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        *,
        height: int,
        width: int,
        token_type: int,
    ) -> np.ndarray:
        x = ((float(xs.mean()) + 0.5) / float(max(1, width))) * 2.0 - 1.0
        y = ((float(ys.mean()) + 0.5) / float(max(1, height))) * 2.0 - 1.0

        span_x = ((float(xs.max()) - float(xs.min()) + 1.0) / float(max(1, width))) * 2.0
        span_y = ((float(ys.max()) - float(ys.min()) + 1.0) / float(max(1, height))) * 2.0

        sx = max(self.min_scale, 0.5 * span_x)
        sy = max(self.min_scale, 0.5 * span_y)
        phi = 0.0

        if token_type == TYPE_WALL:
            phi = float(np.pi * 0.5) if span_y >= span_x else 0.0
        else:
            radius = max(sx, sy)
            sx = radius
            sy = radius

        thickness = float(np.clip(min(sx, sy), self.min_thickness, self.max_thickness))
        return self._clamp_geometry(np.array([x, y, phi, sx, sy, thickness], dtype=np.float32))

    def _extract_state_candidates(self, observation: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        candidates = {"agent": [], "pod": [], "wall": [], "condition": []}

        agent_pos = np.asarray(observation.get("agent_pos", [-0.5, 0.0]), dtype=np.float32).reshape(-1)
        if agent_pos.shape[0] >= 2:
            candidates["agent"].append(
                self._clamp_geometry(np.array([agent_pos[0], agent_pos[1], 0.0, 0.04, 0.04, 0.04], dtype=np.float32))
            )

        for pod in observation.get("pods", []) or []:
            try:
                is_active = bool(pod.get("active", False)) if hasattr(pod, "get") else bool(pod.active)
                if not is_active:
                    continue
                pos = pod.get("pos", [0.0, 0.0]) if hasattr(pod, "get") else pod.pos
                food = float(pod.get("food", 0.2)) if hasattr(pod, "get") else float(pod.food)
                pos = np.asarray(pos, dtype=np.float32).reshape(-1)
                if pos.shape[0] >= 2:
                    radius = float(np.clip(0.05 + food * 0.05, self.min_scale, self.max_scale))
                    candidates["pod"].append(
                        self._clamp_geometry(np.array([pos[0], pos[1], 0.0, radius, radius, radius], dtype=np.float32))
                    )
            except Exception:
                continue

        wall_present = self._state_wall_present(observation)

        if wall_present:
            rooms = observation.get("rooms", []) or []
            side = str(observation.get("bucket_side", "left"))
            candidates["wall"].extend(self._wall_segments_from_rooms(rooms, side))
        else:
                                                                                        
                                                     
            candidates["wall"] = []

        if int(self.schema.num_condition_slots) > 0:
            condition_candidates, _ = self.condition_tokenizer.tokenize(observation)
            candidates["condition"] = condition_candidates

        return candidates

    @staticmethod
    def _state_wall_present(observation: Dict[str, Any]) -> bool:
        wall_state = observation.get("wall", {}) if isinstance(observation, dict) else {}
        try:
            if hasattr(wall_state, "get"):
                # Prefer explicit instantaneous state when available.
                if "blocking" in wall_state:
                    return bool(wall_state.get("blocking", False))
                if "enabled" in wall_state:
                    return bool(wall_state.get("enabled", False))
                return bool(wall_state.get("open_until", 0.0) > 0.0)
            if hasattr(wall_state, "blocking"):
                return bool(getattr(wall_state, "blocking", False))
            if hasattr(wall_state, "enabled"):
                return bool(getattr(wall_state, "enabled", False))
            return bool(getattr(wall_state, "open_until", 0.0) > 0.0)
        except Exception:
            return False

    def _segment_to_geometry(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        thickness: float,
    ) -> np.ndarray:
        p0 = np.asarray(p0, dtype=np.float32).reshape(-1)
        p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
        if p0.shape[0] < 2 or p1.shape[0] < 2:
            return self._clamp_geometry(np.array([0.0, 0.0, 0.0, 0.02, 0.02, thickness], dtype=np.float32))

        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        dx = x1 - x0
        dy = y1 - y0
        seg_len = float(np.sqrt(dx * dx + dy * dy))
        phi = float(np.arctan2(dy, dx))

        sx = max(self.min_scale, 0.5 * seg_len)
        sy = max(self.min_scale, 0.5 * float(thickness))
        return self._clamp_geometry(np.array([cx, cy, phi, sx, sy, thickness], dtype=np.float32))

    def _wall_segments_from_rooms(self, rooms: Sequence[Any], side: str) -> List[np.ndarray]:
        geoms: List[np.ndarray] = []
        t = float(np.clip(self.default_wall_thickness, self.min_thickness, self.max_thickness))

                                                                           
        geoms.append(self._segment_to_geometry(np.array([0.0, -1.0]), np.array([0.0, 1.0]), thickness=t))

        side_norm = "left" if str(side).strip().lower() == "left" else "right"
                                                                                
                                                                   
        y_boundaries: List[float] = []
        x_min = None
        x_max = None

        for r in rooms:
            try:
                rr = np.array(r, dtype=np.float32).reshape(-1)
                if rr.shape[0] < 4:
                    continue
                rx, ry, rw, rh = [float(v) for v in rr[:4]]
                room_x0 = rx
                room_x1 = rx + rw
                x_min = room_x0 if x_min is None else min(x_min, room_x0)
                x_max = room_x1 if x_max is None else max(x_max, room_x1)
                y_boundaries.append(ry)
                y_boundaries.append(ry + rh)
            except Exception:
                continue

        if y_boundaries and x_min is not None and x_max is not None:
                                                                                   
            y_sorted = sorted(y_boundaries)
            y_unique: List[float] = []
            for y in y_sorted:
                if not y_unique or abs(y - y_unique[-1]) > 1e-4:
                    y_unique.append(y)

            if side_norm == "left":
                x0, x1 = float(x_min), 0.0
            else:
                x0, x1 = 0.0, float(x_max)

            for y in y_unique:
                geoms.append(
                    self._segment_to_geometry(
                        np.array([x0, y], dtype=np.float32),
                        np.array([x1, y], dtype=np.float32),
                        thickness=t,
                    )
                )

        return geoms

    def _sort_geometry_candidates(self, geoms: Sequence[np.ndarray], token_type: int) -> List[np.ndarray]:
        if not geoms:
            return []
        if token_type == TYPE_WALL:
                                    
            return sorted(geoms, key=lambda g: float(max(g[3], g[4])), reverse=True)
        return sorted(geoms, key=lambda g: float(g[3] * g[4]), reverse=True)

    def tokenize(self, observation: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = observation if isinstance(observation, dict) else {}
        rgb = _to_rgb_frame(obs.get("frame"))

        geometry_candidates: Dict[str, List[np.ndarray]]
        diag: Dict[str, Any] = {}

        if rgb is None:
            geometry_candidates = self._extract_state_candidates(obs)
            diag["source"] = "state_fallback"
            diag["component_counts"] = {
                "agent": len(geometry_candidates["agent"]),
                "pod": len(geometry_candidates["pod"]),
                "wall": len(geometry_candidates["wall"]),
                "condition": len(geometry_candidates["condition"]),
            }
            diag["wall_visible"] = bool(len(geometry_candidates["wall"]) > 0)
            diag["candidate_source"] = {"agent": "state", "pod": "state", "wall": "state", "condition": "condition"}
        else:
            h, w = rgb.shape[:2]
            masks = self._extract_masks(rgb)

            agent_comps = self._connected_components(masks["agent"], self.min_pixels_agent)
            pod_comps = self._connected_components(masks["pod"], self.min_pixels_pod)
            wall_comps = self._connected_components(masks["wall"], self.min_pixels_wall)

            agent_geoms = [
                self._component_to_geometry(ys, xs, height=h, width=w, token_type=TYPE_AGENT)
                for ys, xs in agent_comps
            ]
            pod_geoms = [
                self._component_to_geometry(ys, xs, height=h, width=w, token_type=TYPE_POD)
                for ys, xs in pod_comps
            ]
            wall_geoms = [
                self._component_to_geometry(ys, xs, height=h, width=w, token_type=TYPE_WALL)
                for ys, xs in wall_comps
            ]

            state_candidates = self._extract_state_candidates(obs)
            wall_present_state = self._state_wall_present(obs)
            # State authority: if wall is absent in state, suppress frame-derived wall candidates.
            if not wall_present_state:
                wall_geoms = []
            elif state_candidates["wall"]:
                wall_geoms = state_candidates["wall"]

            agent_sorted = self._sort_geometry_candidates(agent_geoms, TYPE_AGENT)
            pod_sorted = self._sort_geometry_candidates(pod_geoms, TYPE_POD)
            wall_sorted = self._sort_geometry_candidates(wall_geoms, TYPE_WALL)

            geometry_candidates = {
                "agent": agent_sorted or state_candidates["agent"],
                "pod": pod_sorted or state_candidates["pod"],
                "wall": wall_sorted,
                "condition": state_candidates["condition"],
            }

            if wall_present_state and state_candidates["wall"]:
                diag["source"] = "frame+state_wall"
            elif not wall_present_state:
                diag["source"] = "frame+state_no_wall"
            else:
                diag["source"] = "frame"
            diag["component_counts"] = {
                "agent": len(agent_comps),
                "pod": len(pod_comps),
                "wall": len(wall_comps),
                "condition": len(geometry_candidates["condition"]),
            }
            diag["wall_visible"] = bool(len(wall_sorted) > 0)
            diag["candidate_source"] = {
                "agent": "frame" if len(agent_sorted) > 0 else "state",
                "pod": "frame" if len(pod_sorted) > 0 else "state",
                "wall": "state" if (wall_present_state and len(state_candidates["wall"]) > 0) else ("frame" if len(wall_sorted) > 0 else "none"),
                "condition": "condition",
            }

        tokens = np.zeros((self.schema.max_slots, self.schema.token_dim), dtype=np.float32)
        slot_idx = 0

        slot_idx = self._write_slot_group(
            tokens=tokens,
            slot_idx=slot_idx,
            num_slots=self.schema.num_agent_slots,
            token_type=TYPE_AGENT,
            candidates=geometry_candidates["agent"],
        )
        slot_idx = self._write_slot_group(
            tokens=tokens,
            slot_idx=slot_idx,
            num_slots=self.schema.num_pod_slots,
            token_type=TYPE_POD,
            candidates=geometry_candidates["pod"],
        )
        slot_idx = self._write_slot_group(
            tokens=tokens,
            slot_idx=slot_idx,
            num_slots=self.schema.num_wall_slots,
            token_type=TYPE_WALL,
            candidates=geometry_candidates["wall"],
        )
        slot_idx = self._write_slot_group(
            tokens=tokens,
            slot_idx=slot_idx,
            num_slots=self.schema.num_condition_slots,
            token_type=TYPE_CONDITION,
            candidates=geometry_candidates["condition"],
        )
        slot_idx = self._write_slot_group(
            tokens=tokens,
            slot_idx=slot_idx,
            num_slots=self.schema.num_slack_slots,
            token_type=TYPE_EMPTY,
            candidates=[],
        )

        gates = tokens[:, 0]
        types = _argmax_type(tokens, self.schema)
        diag["active_slots"] = int(np.sum(gates > 0.5))
        diag["active_per_type"] = {
            TYPE_NAMES[t]: int(np.sum((types == t) & (gates > 0.5)))
            for t in (TYPE_AGENT, TYPE_POD, TYPE_WALL, TYPE_CONDITION, TYPE_EMPTY)
        }
        return tokens, diag

    def _write_slot_group(
        self,
        *,
        tokens: np.ndarray,
        slot_idx: int,
        num_slots: int,
        token_type: int,
        candidates: Sequence[np.ndarray],
    ) -> int:
        for i in range(num_slots):
            idx = slot_idx + i
            if i < len(candidates):
                geom = self._clamp_geometry(np.asarray(candidates[i], dtype=np.float32))
                gate = 1.0
                self._last_geometry[idx] = geom
            else:
                geom = self._last_geometry[idx]
                gate = 0.0

            tokens[idx] = _pack_token(self.schema, gate=gate, token_type=token_type, geometry=geom)
        return slot_idx + num_slots


class CodaTokenRenderer:
    def __init__(self, schema: CodaTokenSchema):
        self.schema = schema
        self.colors = {
            TYPE_AGENT: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            TYPE_POD: np.array([1.0, 1.0, 0.0], dtype=np.float32),
            TYPE_WALL: np.array([1.0, 1.0, 1.0], dtype=np.float32),
            TYPE_CONDITION: np.array([1.0, 0.2, 1.0], dtype=np.float32),
            TYPE_EMPTY: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

    def render(self, tokens: np.ndarray, *, height: int, width: int) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.float32)
        xw, yw = _world_grid(height, width)

        tokens = np.asarray(tokens, dtype=np.float32)
        token_types = _argmax_type(tokens, self.schema)
        geom_offset = 1 + self.schema.num_types

        for i in range(tokens.shape[0]):
            gate = float(np.clip(tokens[i, 0], 0.0, 1.0))
            if gate < 0.05:
                continue

            token_type = int(token_types[i])
            if token_type in (TYPE_EMPTY, TYPE_CONDITION):
                continue

            geom = tokens[i, geom_offset : geom_offset + self.schema.geom_dim]
            x, y, phi, sx, sy, wt = [float(v) for v in geom]
            sx = max(0.005, abs(sx))
            sy = max(0.005, abs(sy))
            wt = max(0.005, abs(wt))

            dx = xw - x
            dy = yw - y
            c = float(np.cos(phi))
            s = float(np.sin(phi))
            rx = c * dx + s * dy
            ry = -s * dx + c * dy

            if token_type in (TYPE_AGENT, TYPE_POD):
                mask = (rx / sx) ** 2 + (ry / sy) ** 2 <= 1.0
            else:
                                                                                            
                half_thickness = min(sy, wt)
                mask = (np.abs(rx) <= sx) & (np.abs(ry) <= half_thickness)

            color = self.colors.get(token_type, self.colors[TYPE_EMPTY]) * gate
            img[mask] = np.maximum(img[mask], color)

        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

    def explain_pixel(self, tokens: np.ndarray, *, height: int, width: int, py: int, px: int) -> List[Dict[str, float]]:
        tokens = np.asarray(tokens, dtype=np.float32)
        token_types = _argmax_type(tokens, self.schema)
        geom_offset = 1 + self.schema.num_types
        xw, yw = _world_grid(height, width)
        x0 = float(xw[int(py), int(px)])
        y0 = float(yw[int(py), int(px)])

        hits: List[Dict[str, float]] = []
        for i in range(tokens.shape[0]):
            gate = float(np.clip(tokens[i, 0], 0.0, 1.0))
            if gate < 0.01:
                continue
            token_type = int(token_types[i])
            if token_type in (TYPE_EMPTY, TYPE_CONDITION):
                continue

            geom = tokens[i, geom_offset : geom_offset + self.schema.geom_dim]
            x, y, phi, sx, sy, wt = [float(v) for v in geom]
            sx = max(0.005, abs(sx))
            sy = max(0.005, abs(sy))
            wt = max(0.005, abs(wt))

            dx = x0 - x
            dy = y0 - y
            c = float(np.cos(phi))
            s = float(np.sin(phi))
            rx = c * dx + s * dy
            ry = -s * dx + c * dy

            if token_type in (TYPE_AGENT, TYPE_POD):
                inside = (rx / sx) ** 2 + (ry / sy) ** 2 <= 1.0
            else:
                half_thickness = min(sy, wt)
                inside = (abs(rx) <= sx) and (abs(ry) <= half_thickness)

            if inside:
                hits.append(
                    {
                        "slot": float(i),
                        "type_id": float(token_type),
                        "gate": gate,
                        "x": x,
                        "y": y,
                        "phi": phi,
                        "sx": sx,
                        "sy": sy,
                        "wt": wt,
                    }
                )
        hits.sort(key=lambda d: float(d["gate"]), reverse=True)
        return hits


class _RelationalBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_types: int, geom_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_types = int(num_types)
        self.geom_dim = int(geom_dim)

                                                                                                               
        self.geom_rel_dim = 12
        self.gate_rel_dim = 2
        self.type_rel_dim = 2 * self.num_types
        self.rel_dim = self.geom_rel_dim + self.gate_rel_dim + self.type_rel_dim

        in_dim = (2 * self.hidden_dim) + self.rel_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geom_score_mlp = nn.Sequential(
            nn.Linear((2 * self.hidden_dim) + self.geom_rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_score_mlp = nn.Sequential(
            nn.Linear((2 * self.hidden_dim) + self.type_rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_score_mlp = nn.Sequential(
            nn.Linear((2 * self.hidden_dim) + self.gate_rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_norm = nn.LayerNorm(self.rel_dim)
        self.geom_rel_norm = nn.LayerNorm(self.geom_rel_dim)
        self.gate_rel_norm = nn.LayerNorm(self.gate_rel_dim)
        self.type_rel_norm = nn.LayerNorm(self.type_rel_dim)
        self.distance_bias_gamma = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor, tokens: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        bsz, slots, hdim = h.shape
        if hdim != self.hidden_dim:
            raise RuntimeError(f"Unexpected hidden dim: got {hdim}, expected {self.hidden_dim}")

        gates = gates.clamp(0.0, 1.0)
        gate_i = gates.unsqueeze(2)           
        gate_j = gates.unsqueeze(1)           

        type_probs = tokens[..., 1 : 1 + self.num_types].clamp(EPS, 1.0)
        geom = tokens[..., 1 + self.num_types : 1 + self.num_types + self.geom_dim]

        x = geom[..., 0:1]
        y = geom[..., 1:2]
        phi = geom[..., 2:3]
        sx = geom[..., 3:4].abs()
        sy = geom[..., 4:5].abs()

        dx = x.unsqueeze(1) - x.unsqueeze(2)                        
        dy = y.unsqueeze(1) - y.unsqueeze(2)
        dist = torch.sqrt(dx * dx + dy * dy + EPS)

        dphi = phi.unsqueeze(1) - phi.unsqueeze(2)
        cos_dphi = torch.cos(dphi)
        sin_dphi = torch.sin(dphi)

        sx_i = sx.unsqueeze(2).expand(bsz, slots, slots, 1)
        sy_i = sy.unsqueeze(2).expand(bsz, slots, slots, 1)
        sx_j = sx.unsqueeze(1).expand(bsz, slots, slots, 1)
        sy_j = sy.unsqueeze(1).expand(bsz, slots, slots, 1)
        overlap_x = F.relu(sx_i + sx_j - torch.abs(dx))
        overlap_y = F.relu(sy_i + sy_j - torch.abs(dy))
        collision_flag = ((overlap_x > 0.0) & (overlap_y > 0.0)).to(h.dtype)

        type_i = type_probs.unsqueeze(2).expand(bsz, slots, slots, self.num_types)
        type_j = type_probs.unsqueeze(1).expand(bsz, slots, slots, self.num_types)

        gate_i_exp = gate_i.expand(bsz, slots, slots, 1)
        gate_j_exp = gate_j.expand(bsz, slots, slots, 1)

                                                                                   
        geom_rel = torch.cat(
            [
                dx / 2.0,
                dy / 2.0,
                dist / 2.0,
                cos_dphi,
                sin_dphi,
                sx_i,
                sy_i,
                sx_j,
                sy_j,
                overlap_x / 2.0,
                overlap_y / 2.0,
                collision_flag,
            ],
            dim=-1,
        )
        gate_rel = torch.cat([gate_i_exp, gate_j_exp], dim=-1)
        type_rel = torch.cat([type_i, type_j], dim=-1)

        geom_rel = self.geom_rel_norm(geom_rel)
        gate_rel = self.gate_rel_norm(gate_rel)
        type_rel = self.type_rel_norm(type_rel)
        rel = torch.cat([geom_rel, gate_rel, type_rel], dim=-1)
        rel = self.rel_norm(rel)

        h_i = h.unsqueeze(2).expand(bsz, slots, slots, hdim)
        h_j = h.unsqueeze(1).expand(bsz, slots, slots, hdim)
        pair_feat = torch.cat([h_i, h_j, rel], dim=-1)

        m_ij = self.message_mlp(pair_feat)
        geom_score = self.geom_score_mlp(torch.cat([h_i, h_j, geom_rel], dim=-1)).squeeze(-1)
        type_score = self.type_score_mlp(torch.cat([h_i, h_j, type_rel], dim=-1)).squeeze(-1)
        gate_score = self.gate_score_mlp(torch.cat([h_i, h_j, gate_rel], dim=-1)).squeeze(-1)
        scores = geom_score + type_score + gate_score         

                                                             
        gamma = F.softplus(self.distance_bias_gamma)
        scores = scores - gamma * (dist.squeeze(-1) / 2.0)

        gate_j_s = gates.squeeze(-1).unsqueeze(1)         
                                                                 
        self_mask = torch.eye(slots, device=h.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(self_mask, -1e9)
        scores = scores.masked_fill(gate_j_s <= 0.0, -1e9)

        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

                                                                                                  
        alpha = alpha * gate_j_s
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + EPS)

                                                              
        gate_i_s = gates.squeeze(-1).unsqueeze(-1)         
        alpha = alpha * gate_i_s

        m_i = (alpha.unsqueeze(-1) * m_ij).sum(dim=2)         
        out = self.update_mlp(torch.cat([h, m_i], dim=-1))
        self.last_attention = alpha.detach()
        return self.norm(h + out)


class _SetDynamicsModel(nn.Module):
    def __init__(
        self,
        schema: CodaTokenSchema,
        action_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        use_uncertainty_head: bool = True,
        geom_logvar_min: float = -6.0,
        geom_logvar_max: float = 3.0,
        gate_present_threshold: float = 0.5,
        elapsed_time_scale: float = 10.0,
        min_xy_step_scale: float = 0.05,
        max_xy_step_scale: float = 1.0,
        min_phi_step_scale: float = 0.2,
        max_phi_step_scale: float = np.pi,
        min_size_step_scale: float = 0.02,
        max_size_step_scale: float = 0.6,
        min_wt_step_scale: float = 0.01,
        max_wt_step_scale: float = 0.4,
    ):
        super().__init__()
        self.schema = schema
        self.token_dim = schema.token_dim
        self.geom_offset = 1 + schema.num_types

        self.token_embed = nn.Sequential(
            nn.Linear(self.token_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        slot_types = (
            [TYPE_AGENT] * self.schema.num_agent_slots
            + [TYPE_POD] * self.schema.num_pod_slots
            + [TYPE_WALL] * self.schema.num_wall_slots
            + [TYPE_CONDITION] * self.schema.num_condition_slots
            + [TYPE_EMPTY] * self.schema.num_slack_slots
        )
        type_template = torch.zeros(self.schema.max_slots, self.schema.num_types, dtype=torch.float32)
        for idx, t in enumerate(slot_types):
            type_template[idx, int(t)] = 1.0
                                                                            
        self.register_buffer("slot_type_template", type_template, persistent=True)
                                                                                     
        spawn_eligible = (torch.tensor(slot_types, dtype=torch.long) != TYPE_EMPTY).to(torch.float32).unsqueeze(-1)
        self.register_buffer("spawn_eligible_mask", spawn_eligible, persistent=True)
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _RelationalBlock(
                    hidden_dim=hidden_dim,
                    num_types=self.schema.num_types,
                    geom_dim=self.schema.geom_dim,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.use_uncertainty_head = bool(use_uncertainty_head)
        self.geom_logvar_min = float(geom_logvar_min)
        self.geom_logvar_max = float(geom_logvar_max)
        self.gate_present_threshold = float(gate_present_threshold)
        self.elapsed_time_scale = float(max(1e-3, elapsed_time_scale))
        if self.use_uncertainty_head:
            self.geom_logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.schema.geom_dim),
            )
        else:
            self.geom_logvar_head = None
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
                                                                 
        self.jump_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.min_scale = 0.01
        self.max_scale = 1.2
        self.min_thickness = 0.005
        self.max_thickness = 0.4
        self.absent_gate_decay = 0.5
        self.min_xy_step_scale = float(min_xy_step_scale)
        self.max_xy_step_scale = float(max_xy_step_scale)
        self.min_phi_step_scale = float(min_phi_step_scale)
        self.max_phi_step_scale = float(max_phi_step_scale)
        self.min_size_step_scale = float(min_size_step_scale)
        self.max_size_step_scale = float(max_size_step_scale)
        self.min_wt_step_scale = float(min_wt_step_scale)
        self.max_wt_step_scale = float(max_wt_step_scale)

        self.xy_step_scale_raw = nn.Parameter(
            torch.full((self.schema.num_types,), self._target_to_logit(0.25, self.min_xy_step_scale, self.max_xy_step_scale))
        )
        self.phi_step_scale_raw = nn.Parameter(
            torch.full((self.schema.num_types,), self._target_to_logit(1.0, self.min_phi_step_scale, self.max_phi_step_scale))
        )
        self.size_step_scale_raw = nn.Parameter(
            torch.full((self.schema.num_types,), self._target_to_logit(0.15, self.min_size_step_scale, self.max_size_step_scale))
        )
        self.wt_step_scale_raw = nn.Parameter(
            torch.full((self.schema.num_types,), self._target_to_logit(0.10, self.min_wt_step_scale, self.max_wt_step_scale))
        )

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        *,
        actions: Optional[torch.Tensor] = None,
        elapsed_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if elapsed_time is None:
            elapsed_time = torch.zeros(tokens.shape[0], tokens.shape[1], 1, dtype=tokens.dtype, device=tokens.device)
        elif elapsed_time.dim() == 2:
            elapsed_time = elapsed_time.unsqueeze(-1)
        elapsed_scaled = torch.tanh(elapsed_time / self.elapsed_time_scale)
        token_in = torch.cat([tokens, elapsed_scaled], dim=-1)
        h = self.token_embed(token_in)
        if actions is not None:
            h = h + self.action_embed(actions).unsqueeze(1)
        return h

    @staticmethod
    def _target_to_logit(target: float, lo: float, hi: float) -> float:
        lo_f = float(lo)
        hi_f = float(max(lo + 1e-6, hi))
        t = float(np.clip((float(target) - lo_f) / (hi_f - lo_f), 1e-4, 1.0 - 1e-4))
        return float(np.log(t / (1.0 - t)))

    @staticmethod
    def _bounded_scale(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        lo_t = float(lo)
        hi_t = float(max(lo + 1e-6, hi))
        return lo_t + (hi_t - lo_t) * torch.sigmoid(raw)

    def forward_with_aux(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        elapsed_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        gates = tokens[..., :1].clamp(0.0, 1.0)
        if elapsed_time is None:
            elapsed_time = torch.zeros(tokens.shape[0], tokens.shape[1], 1, dtype=tokens.dtype, device=tokens.device)
        elif elapsed_time.dim() == 2:
            elapsed_time = elapsed_time.unsqueeze(-1)
        elapsed_scaled = torch.tanh(elapsed_time / self.elapsed_time_scale)

        h = self.encode_tokens(tokens, actions=actions, elapsed_time=elapsed_time)
        for block in self.blocks:
            h = block(h, tokens, gates)
        raw = self.head(h)
        hazard_logits = self.hazard_head(h)
        jump_raw = self.jump_head(h)
        out_tokens = self._stabilize(raw, tokens, hazard_logits, jump_raw)

        geom_logvar: Optional[torch.Tensor] = None
        if self.geom_logvar_head is not None:
            lv_raw = self.geom_logvar_head(h)
            geom_logvar = lv_raw.clamp(self.geom_logvar_min, self.geom_logvar_max)
        return out_tokens, geom_logvar, hazard_logits, jump_raw

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out_tokens, _, _, _ = self.forward_with_aux(tokens, actions)
        return out_tokens

    def get_last_attention(self) -> Optional[torch.Tensor]:
        if not self.blocks:
            return None
        last = self.blocks[-1]
        if hasattr(last, "last_attention"):
            return last.last_attention
        return None

    def _stabilize(self, raw: torch.Tensor, tokens: torch.Tensor, hazard_logits: torch.Tensor, jump_raw: torch.Tensor) -> torch.Tensor:
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = 4.0 * torch.tanh(raw / 4.0)
        hazard_logits = torch.nan_to_num(hazard_logits, nan=0.0, posinf=0.0, neginf=0.0)

        gate_prev = tokens[..., :1].clamp(EPS, 1.0 - EPS)
        gate_logit_prev = torch.log(gate_prev) - torch.log1p(-gate_prev)
        gate_persist = torch.sigmoid(gate_logit_prev + raw[..., :1])
        if hazard_logits.shape[-1] >= 2:
            spawn_logits = hazard_logits[..., 0:1]
            despawn_logits = hazard_logits[..., 1:2]
        else:
            spawn_logits = hazard_logits[..., :1]
            despawn_logits = -hazard_logits[..., :1]
        gate_spawn = torch.sigmoid(spawn_logits)
        gate_despawn = torch.sigmoid(despawn_logits)
        present_mask = (gate_prev > self.gate_present_threshold).to(gate_prev.dtype)                            

                                                                                                
        spawn_eligible = self.spawn_eligible_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        spawn_mask = (1.0 - present_mask) * spawn_eligible
        stay_absent_mask = torch.clamp(1.0 - present_mask - spawn_mask, min=0.0, max=1.0)
        gate_absent = gate_prev * self.absent_gate_decay
        gate_present = (1.0 - gate_despawn) * gate_persist

        gate = (
            present_mask * gate_present
            + spawn_mask * gate_spawn
            + stay_absent_mask * gate_absent
        )

                                                                  
        type_template = self.slot_type_template.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        type_probs = type_template.expand(tokens.shape[0], -1, -1)

        xy_scale_types = self._bounded_scale(self.xy_step_scale_raw, self.min_xy_step_scale, self.max_xy_step_scale)
        phi_scale_types = self._bounded_scale(self.phi_step_scale_raw, self.min_phi_step_scale, self.max_phi_step_scale)
        size_scale_types = self._bounded_scale(self.size_step_scale_raw, self.min_size_step_scale, self.max_size_step_scale)
        wt_scale_types = self._bounded_scale(self.wt_step_scale_raw, self.min_wt_step_scale, self.max_wt_step_scale)
        xy_scale = torch.sum(type_probs * xy_scale_types.view(1, 1, -1), dim=-1, keepdim=True)
        phi_scale = torch.sum(type_probs * phi_scale_types.view(1, 1, -1), dim=-1, keepdim=True)
        size_scale = torch.sum(type_probs * size_scale_types.view(1, 1, -1), dim=-1, keepdim=True)
        wt_scale = torch.sum(type_probs * wt_scale_types.view(1, 1, -1), dim=-1, keepdim=True)

        base_geom = tokens[..., self.geom_offset :]
        delta_geom = raw[..., self.geom_offset :]

        x_normal = (base_geom[..., 0:1] + xy_scale * torch.tanh(delta_geom[..., 0:1])).clamp(-1.0, 1.0)
        y_normal = (base_geom[..., 1:2] + xy_scale * torch.tanh(delta_geom[..., 1:2])).clamp(-1.0, 1.0)
        jump_logits = torch.nan_to_num(jump_raw[..., 0:1], nan=0.0, posinf=0.0, neginf=0.0)
        jump_xy = torch.tanh(torch.nan_to_num(jump_raw[..., 1:3], nan=0.0, posinf=0.0, neginf=0.0))
        jump_prob = torch.sigmoid(jump_logits)
        x = ((1.0 - jump_prob) * x_normal + jump_prob * jump_xy[..., 0:1]).clamp(-1.0, 1.0)
        y = ((1.0 - jump_prob) * y_normal + jump_prob * jump_xy[..., 1:2]).clamp(-1.0, 1.0)
        phi = (base_geom[..., 2:3] + phi_scale * torch.tanh(delta_geom[..., 2:3])).clamp(-np.pi, np.pi)
        sx = (base_geom[..., 3:4] + size_scale * torch.tanh(delta_geom[..., 3:4])).clamp(self.min_scale, self.max_scale)
        sy = (base_geom[..., 4:5] + size_scale * torch.tanh(delta_geom[..., 4:5])).clamp(self.min_scale, self.max_scale)
        wt = (base_geom[..., 5:6] + wt_scale * torch.tanh(delta_geom[..., 5:6])).clamp(
            self.min_thickness, self.max_thickness
        )
        geom = torch.cat([x, y, phi, sx, sy, wt], dim=-1)
                                                          
                                                                                                   
        active_or_spawning = torch.clamp((gate > self.gate_present_threshold).to(tokens.dtype) + spawn_mask, 0.0, 1.0)
        geom = active_or_spawning * geom + (1.0 - active_or_spawning) * base_geom
        return torch.cat([gate, type_probs, geom], dim=-1)


class CodaTransitionBuffer:
    def __init__(self, capacity: int):
        self._data: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32, np.int32, np.float32]] = deque(
            maxlen=int(max(1, capacity))
        )

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        tokens_t: np.ndarray,
        action: np.ndarray,
        tokens_tp1: np.ndarray,
        elapsed_t: np.ndarray,
        *,
        is_event: bool = False,
        event_code: int = 0,
        contact_candidate: bool = False,
    ):
        t0 = np.asarray(tokens_t, dtype=np.float32).copy()
        a = np.asarray(action, dtype=np.float32).copy()
        t1 = np.asarray(tokens_tp1, dtype=np.float32).copy()
        e = np.asarray(elapsed_t, dtype=np.float32).copy()
        if (not np.isfinite(t0).all()) or (not np.isfinite(a).all()) or (not np.isfinite(t1).all()) or (not np.isfinite(e).all()):
            return
        self._data.append(
            (
                t0,
                a,
                t1,
                e,
                np.float32(1.0 if is_event else 0.0),
                np.int32(int(event_code)),
                np.float32(1.0 if contact_candidate else 0.0),
            )
        )

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._data)
        if n <= 0:
            raise RuntimeError("Cannot sample from empty transition buffer.")

        bsz = int(max(1, batch_size))
        replace = n < bsz
        indices = rng.choice(n, size=bsz, replace=replace)

        tokens_t = np.stack([self._data[int(i)][0] for i in indices], axis=0).astype(np.float32, copy=False)
        actions = np.stack([self._data[int(i)][1] for i in indices], axis=0).astype(np.float32, copy=False)
        tokens_tp1 = np.stack([self._data[int(i)][2] for i in indices], axis=0).astype(np.float32, copy=False)
        elapsed_t = np.stack([self._data[int(i)][3] for i in indices], axis=0).astype(np.float32, copy=False)
        return tokens_t, actions, tokens_tp1, elapsed_t

    def sample_stratified(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator,
        event_fraction: float,
        event_only: bool = False,
        event_window_steps: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._data)
        if n <= 0:
            raise RuntimeError("Cannot sample from empty transition buffer.")

        bsz = int(max(1, batch_size))
        raw_event = np.asarray([float(entry[4]) for entry in self._data], dtype=np.float32)
        contact = np.asarray([float(entry[6]) for entry in self._data], dtype=np.float32)
        window = raw_event.copy()
        w = int(max(0, event_window_steps))
        if w > 0 and np.any(raw_event > 0.5):
            event_idx = np.flatnonzero(raw_event > 0.5)
            for idx in event_idx:
                lo = max(0, int(idx - w))
                hi = min(n, int(idx + w + 1))
                window[lo:hi] = 1.0

        event_idx = np.flatnonzero(window > 0.5)
        non_idx = np.flatnonzero(window <= 0.5)
        pod_spawn_raw = np.asarray(
            [1.0 if (int(entry[5]) & 2) != 0 else 0.0 for entry in self._data],
            dtype=np.float32,
        )
        pod_spawn_window = pod_spawn_raw.copy()
        if w > 0 and np.any(pod_spawn_raw > 0.5):
            pod_spawn_idx = np.flatnonzero(pod_spawn_raw > 0.5)
            for idx in pod_spawn_idx:
                lo = max(0, int(idx - w))
                hi = min(n, int(idx + w + 1))
                pod_spawn_window[lo:hi] = 1.0
        pod_spawn_future = np.zeros((n,), dtype=np.float32)
        if w > 0 and np.any(pod_spawn_raw > 0.5):
            pod_spawn_idx = np.flatnonzero(pod_spawn_raw > 0.5)
            for idx in pod_spawn_idx:
                lo = max(0, int(idx - w))
                hi = int(idx)
                if hi > lo:
                    pod_spawn_future[lo:hi] = 1.0

        if event_only and event_idx.size > 0:
            if event_idx.size >= bsz:
                picked = rng.choice(event_idx, size=bsz, replace=False)
            else:
                picked = rng.choice(event_idx, size=bsz, replace=True)
        else:
            frac = float(np.clip(event_fraction, 0.0, 1.0))
            n_event = int(round(frac * bsz))
            n_event = max(0, min(bsz, n_event))
            n_non = bsz - n_event

            picked_event: np.ndarray
            picked_non: np.ndarray
            if event_idx.size <= 0:
                picked_event = np.zeros((0,), dtype=np.int64)
                n_non = bsz
            else:
                picked_event = rng.choice(event_idx, size=n_event, replace=(event_idx.size < n_event)).astype(np.int64, copy=False)

            if non_idx.size <= 0:
                picked_non = rng.choice(event_idx if event_idx.size > 0 else np.arange(n), size=n_non, replace=True).astype(np.int64, copy=False)
            else:
                picked_non = rng.choice(non_idx, size=n_non, replace=(non_idx.size < n_non)).astype(np.int64, copy=False)

            picked = np.concatenate([picked_event, picked_non], axis=0) if picked_event.size > 0 else picked_non
            if picked.size < bsz:
                extra = rng.choice(np.arange(n), size=(bsz - picked.size), replace=True).astype(np.int64, copy=False)
                picked = np.concatenate([picked, extra], axis=0)
            rng.shuffle(picked)

        indices = picked.astype(np.int64, copy=False)
        tokens_t = np.stack([self._data[int(i)][0] for i in indices], axis=0).astype(np.float32, copy=False)
        actions = np.stack([self._data[int(i)][1] for i in indices], axis=0).astype(np.float32, copy=False)
        tokens_tp1 = np.stack([self._data[int(i)][2] for i in indices], axis=0).astype(np.float32, copy=False)
        elapsed_t = np.stack([self._data[int(i)][3] for i in indices], axis=0).astype(np.float32, copy=False)
        event_t = np.asarray([self._data[int(i)][4] for i in indices], dtype=np.float32)
        event_code_t = np.asarray([self._data[int(i)][5] for i in indices], dtype=np.int32)
        contact_t = np.asarray([self._data[int(i)][6] for i in indices], dtype=np.float32)
        window_t = window[indices].astype(np.float32, copy=False)
        pod_spawn_window_t = pod_spawn_window[indices].astype(np.float32, copy=False)
        pod_spawn_future_t = pod_spawn_future[indices].astype(np.float32, copy=False)
        return tokens_t, actions, tokens_tp1, elapsed_t, event_t, event_code_t, contact_t, window_t, pod_spawn_window_t, pod_spawn_future_t

    def state_dict(self) -> Dict[str, Any]:
        return {"data": list(self._data)}

    def load_state_dict(self, payload: Dict[str, Any]):
        self._data.clear()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for entry in data:
            if not isinstance(entry, (list, tuple)):
                continue
            if len(entry) == 4:
                self.add(entry[0], entry[1], entry[2], entry[3], is_event=False, event_code=0, contact_candidate=False)
            elif len(entry) >= 7:
                self.add(
                    entry[0],
                    entry[1],
                    entry[2],
                    entry[3],
                    is_event=bool(float(entry[4])),
                    event_code=int(entry[5]),
                    contact_candidate=bool(float(entry[6])),
                )
            elif len(entry) == 3:
                                                                                     
                t = np.asarray(entry[0], dtype=np.float32)
                self.add(
                    entry[0],
                    entry[1],
                    entry[2],
                    np.zeros((t.shape[0],), dtype=np.float32),
                    is_event=False,
                    event_code=0,
                    contact_candidate=False,
                )


def _hungarian_assignment(cost: np.ndarray) -> np.ndarray:
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("Hungarian assignment requires a square cost matrix.")

    n = c.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=np.bool_)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.zeros(n, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


class CodaForwardModelComponent:
    def __init__(
        self,
        *,
        schema: Optional[CodaTokenSchema] = None,
        dynamics_config: Optional[CodaDynamicsConfig] = None,
        action_dim: int = 2,
        seed: int = 0,
        device: str = "cpu",
        training_enabled: bool = True,
    ):
        self.schema = schema or CodaTokenSchema()
        self.config = dynamics_config or CodaDynamicsConfig()
        self.training_enabled = bool(training_enabled)

        self.rng = np.random.default_rng(int(seed))
        self.device = torch.device(device)

        self.tokenizer = CodaGeometryTokenizer(self.schema)
        self.renderer = CodaTokenRenderer(self.schema)
        self.buffer = CodaTransitionBuffer(self.config.buffer_capacity)

        self.model = _SetDynamicsModel(
            self.schema,
            action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            use_uncertainty_head=self.config.use_uncertainty_head,
            geom_logvar_min=self.config.geom_logvar_min,
            geom_logvar_max=self.config.geom_logvar_max,
            gate_present_threshold=self.config.gate_present_threshold,
            elapsed_time_scale=self.config.elapsed_time_scale,
            min_xy_step_scale=self.config.min_xy_step_scale,
            max_xy_step_scale=self.config.max_xy_step_scale,
            min_phi_step_scale=self.config.min_phi_step_scale,
            max_phi_step_scale=self.config.max_phi_step_scale,
            min_size_step_scale=self.config.min_size_step_scale,
            max_size_step_scale=self.config.max_size_step_scale,
            min_wt_step_scale=self.config.min_wt_step_scale,
            max_wt_step_scale=self.config.max_wt_step_scale,
        ).to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        mp_dtype_name = str(self.config.mixed_precision_dtype).strip().lower()
        self._amp_dtype = torch.bfloat16 if mp_dtype_name == "bfloat16" else torch.float16
        self._amp_enabled = bool(self.config.use_mixed_precision and self.device.type == "cuda")
        self._grad_scaler = torch.cuda.amp.GradScaler(
            enabled=bool(self._amp_enabled and self._amp_dtype == torch.float16)
        )

        self.total_updates = 0
        self.total_seen_transitions = 0
        self._update_clock = 0
        self.last_train_metrics: Dict[str, float] = {
            "loss": 0.0,
            "gate_loss": 0.0,
            "type_loss": 0.0,
            "geom_loss": 0.0,
            "hazard_loss": 0.0,
            "despawn_loss": 0.0,
            "jump_loss": 0.0,
            "collision_loss": 0.0,
            "contact_drop_loss": 0.0,
            "buffer_size": 0.0,
            "total_updates": 0.0,
            "per_token_loss_mean": 0.0,
            "per_token_loss_max": 0.0,
            "edge_attr_max": 0.0,
            "geom_logvar_mean": 0.0,
            "adjacent_link_l2_mean": 0.0,
            "adjacent_link_l2_p95": 0.0,
            "close_frac_lt_0p25": 0.0,
            "token_gate_flip_frac": 0.0,
            "token_gate_flip_frac_event_window": 0.0,
            "token_gate_flip_frac_non_event_window": 0.0,
            "token_gate_flip_frac_event_true": 0.0,
            "token_gate_flip_frac_non_event_true": 0.0,
            "mixed_precision_enabled": float(1.0 if self._amp_enabled else 0.0),
        }
        self.last_per_token_loss: Optional[np.ndarray] = None
        self.last_attention: Optional[np.ndarray] = None
        self.last_edge_attribution: Optional[np.ndarray] = None
                                                                 
        self.elapsed_time = np.zeros((self.schema.max_slots,), dtype=np.float32)
        self._slot_types = np.array(
            [TYPE_AGENT] * self.schema.num_agent_slots
            + [TYPE_POD] * self.schema.num_pod_slots
            + [TYPE_WALL] * self.schema.num_wall_slots
            + [TYPE_CONDITION] * self.schema.num_condition_slots
            + [TYPE_EMPTY] * self.schema.num_slack_slots,
            dtype=np.int64,
        )
        self.transition_stats: Dict[str, float] = {
            "total_transitions": 0.0,
            "event_transitions": 0.0,
            "contact_candidate_transitions": 0.0,
            "pod_drop_events": 0.0,
            "wall_drop_events": 0.0,
            "pod_spawn_events": 0.0,
            "wall_spawn_events": 0.0,
            "slot_type_mismatch_steps": 0.0,
        }
        self.model.absent_gate_decay = float(np.clip(self.config.absent_gate_decay, 0.0, 1.0))
        self.ema_model.absent_gate_decay = float(np.clip(self.config.absent_gate_decay, 0.0, 1.0))

    def tokenize_observation(self, observation: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.tokenizer.tokenize(observation)

    def _contact_candidate_transition(self, tokens_t: np.ndarray, gate_threshold: float) -> bool:
        t = np.asarray(tokens_t, dtype=np.float32)
        go = 1 + self.schema.num_types
        a0 = 0
        a1 = self.schema.num_agent_slots
        p0 = a1
        p1 = p0 + self.schema.num_pod_slots
        w0 = p1
        w1 = w0 + self.schema.num_wall_slots

        if t.shape[0] < w1:
            return False

        ag = t[a0:a1, go:]
        pg = t[p0:p1, go:]
        wg = t[w0:w1, go:]
        a_on = t[a0:a1, 0] > gate_threshold
        p_on = t[p0:p1, 0] > gate_threshold
        w_on = t[w0:w1, 0] > gate_threshold
        if not np.any(a_on):
            return False

                             
        for ai in range(ag.shape[0]):
            if not a_on[ai]:
                continue
            ax, ay, _, asx, asy, _ = ag[ai]
            ar = max(abs(asx), abs(asy))
            for pi in range(pg.shape[0]):
                if not p_on[pi]:
                    continue
                px, py, _, psx, psy, _ = pg[pi]
                pr = max(abs(psx), abs(psy))
                d = float(np.hypot(ax - px, ay - py))
                if d <= (ar + pr):
                    return True

                                  
            for wi in range(wg.shape[0]):
                if not w_on[wi]:
                    continue
                wx, wy, phi, sx, sy, wt = wg[wi]
                dx = float(ax - wx)
                dy = float(ay - wy)
                c = float(np.cos(phi))
                s = float(np.sin(phi))
                rx = c * dx + s * dy
                ry = -s * dx + c * dy
                half_th = min(abs(sy), abs(wt))
                qx = max(abs(rx) - abs(sx), 0.0)
                qy = max(abs(ry) - half_th, 0.0)
                dist = float(np.hypot(qx, qy))
                if dist <= ar:
                    return True
        return False

    def observe_transition(self, tokens_t: np.ndarray, action: np.ndarray, tokens_tp1: np.ndarray):
        elapsed_t = self.elapsed_time.copy()
        prev_gate = np.asarray(tokens_t, dtype=np.float32)[:, 0]
        curr_gate = np.asarray(tokens_tp1, dtype=np.float32)[:, 0]
        prev_on = prev_gate > 0.5
        curr_on = curr_gate > 0.5
        off_to_on = (~prev_on) & curr_on
        on_to_off = prev_on & (~curr_on)
        pod_mask = self._slot_types == TYPE_POD
        wall_mask = self._slot_types == TYPE_WALL
        event_code = 0
        if np.any(on_to_off & pod_mask):
            event_code |= 1
        if np.any(off_to_on & pod_mask):
            event_code |= 2
        if np.any(on_to_off & wall_mask):
            event_code |= 4
        if np.any(off_to_on & wall_mask):
            event_code |= 8
        is_event = event_code != 0
        contact_candidate = self._contact_candidate_transition(tokens_t, gate_threshold=0.5)
        self.buffer.add(
            tokens_t=tokens_t,
            action=action,
            tokens_tp1=tokens_tp1,
            elapsed_t=elapsed_t,
            is_event=is_event,
            event_code=event_code,
            contact_candidate=contact_candidate,
        )

        curr_types = np.argmax(np.asarray(tokens_tp1, dtype=np.float32)[:, 1 : 1 + self.schema.num_types], axis=1)
        prev_types = np.argmax(np.asarray(tokens_t, dtype=np.float32)[:, 1 : 1 + self.schema.num_types], axis=1)

        next_elapsed = self.elapsed_time.copy()
        both_on = prev_on & curr_on
        both_off = (~prev_on) & (~curr_on)

        next_elapsed[both_on] = 0.0
        next_elapsed[off_to_on] = 0.0
        next_elapsed[on_to_off] = 1.0
        next_elapsed[both_off] = next_elapsed[both_off] + 1.0
        self.elapsed_time = np.asarray(next_elapsed, dtype=np.float32)
        self.total_seen_transitions += 1

        self.transition_stats["total_transitions"] += 1.0
        self.transition_stats["event_transitions"] += float(1.0 if is_event else 0.0)
        self.transition_stats["contact_candidate_transitions"] += float(1.0 if contact_candidate else 0.0)
        self.transition_stats["pod_drop_events"] += float(np.sum(on_to_off & pod_mask))
        self.transition_stats["wall_drop_events"] += float(np.sum(on_to_off & wall_mask))
        self.transition_stats["pod_spawn_events"] += float(np.sum(off_to_on & pod_mask))
        self.transition_stats["wall_spawn_events"] += float(np.sum(off_to_on & wall_mask))
        if (not np.all(curr_types == self._slot_types)) or (not np.all(prev_types == self._slot_types)):
            self.transition_stats["slot_type_mismatch_steps"] += 1.0

    def mark_reset(self):
                                                                                            
        try:
            self.tokenizer.reset_state()
        except Exception:
            pass
        self.elapsed_time.fill(0.0)

    def predict_next_tokens(self, tokens: np.ndarray, action: np.ndarray, *, use_ema: bool = True) -> np.ndarray:
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(self.device).unsqueeze(0)
            a = torch.from_numpy(np.asarray(action, dtype=np.float32).reshape(1, -1)).to(self.device)
            elapsed = torch.from_numpy(np.asarray(self.elapsed_time, dtype=np.float32)).to(self.device).unsqueeze(0)
            amp_ctx = (
                torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
                if self._amp_enabled
                else nullcontext()
            )
            with amp_ctx:
                pred, _, _, _ = model.forward_with_aux(t, a, elapsed)
        return pred.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def maybe_update(self, *, force: bool = False) -> Optional[Dict[str, float]]:
        self._update_clock += 1
        if not self.training_enabled:
            return None
        if len(self.buffer) < int(max(1, self.config.warmup_transitions)):
            return None
        if not force and (self._update_clock % int(max(1, self.config.update_every)) != 0):
            return None

        heavy_every = int(max(1, self.config.event_heavy_every))
        next_update_id = int(self.total_updates + 1)
        event_heavy = (next_update_id % heavy_every) == 0
        event_frac = float(self.config.event_heavy_fraction if event_heavy else self.config.event_sample_fraction)
        (
            tokens_t_np,
            actions_np,
            tokens_tp1_np,
            elapsed_t_np,
            event_t_np,
            event_code_t_np,
            contact_t_np,
            event_window_t_np,
            pod_spawn_window_t_np,
            pod_spawn_future_t_np,
        ) = self.buffer.sample_stratified(
            self.config.batch_size,
            rng=self.rng,
            event_fraction=event_frac,
            event_only=False,
            event_window_steps=int(max(self.config.event_window_steps, self.config.pod_spawn_window_steps, self.config.spawn_lookahead_steps, 0)),
        )

        tokens_t = torch.from_numpy(tokens_t_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        tokens_tp1 = torch.from_numpy(tokens_tp1_np).to(self.device)
        elapsed_t = torch.from_numpy(elapsed_t_np).to(self.device)
        event_window_t = torch.from_numpy(event_window_t_np).to(self.device)
        pod_spawn_window_t = torch.from_numpy(pod_spawn_window_t_np).to(self.device)
        pod_spawn_future_t = torch.from_numpy(pod_spawn_future_t_np).to(self.device)
        contact_t = torch.from_numpy(contact_t_np).to(self.device)

        self.model.train()
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else nullcontext()
        )
        with amp_ctx:
            pred_tokens, pred_geom_logvar, hazard_logits, jump_aux = self.model.forward_with_aux(tokens_t, actions, elapsed_t)
            pred_tokens = torch.nan_to_num(pred_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
            if pred_geom_logvar is not None:
                pred_geom_logvar = torch.nan_to_num(
                    pred_geom_logvar,
                    nan=0.0,
                    posinf=self.config.geom_logvar_max,
                    neginf=self.config.geom_logvar_min,
                )
            matched_target = self._set_match_targets(pred_tokens, tokens_tp1, tokens_t, elapsed_t)

            (
                loss,
                gate_loss,
                type_loss,
                geom_loss,
                hazard_loss,
                despawn_loss,
                jump_loss,
                collision_loss,
                contact_drop_loss,
                per_token,
                contract_metrics,
            ) = self._dynamics_loss(
                pred_tokens,
                matched_target,
                tokens_t,
                hazard_logits,
                jump_aux,
                pred_geom_logvar,
                event_window_t,
                pod_spawn_window_t,
                pod_spawn_future_t,
                contact_t,
                elapsed_t,
            )

            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
            gate_loss = torch.nan_to_num(gate_loss, nan=0.0, posinf=1e6, neginf=0.0)
            type_loss = torch.nan_to_num(type_loss, nan=0.0, posinf=1e6, neginf=0.0)
            geom_loss = torch.nan_to_num(geom_loss, nan=0.0, posinf=1e6, neginf=0.0)
            hazard_loss = torch.nan_to_num(hazard_loss, nan=0.0, posinf=1e6, neginf=0.0)
            despawn_loss = torch.nan_to_num(despawn_loss, nan=0.0, posinf=1e6, neginf=0.0)
            jump_loss = torch.nan_to_num(jump_loss, nan=0.0, posinf=1e6, neginf=0.0)
            collision_loss = torch.nan_to_num(collision_loss, nan=0.0, posinf=1e6, neginf=0.0)
            contact_drop_loss = torch.nan_to_num(contact_drop_loss, nan=0.0, posinf=1e6, neginf=0.0)

        self.optimizer.zero_grad(set_to_none=True)
        if self._grad_scaler.is_enabled():
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max(1e-6, self.config.grad_clip)))
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max(1e-6, self.config.grad_clip)))
            self.optimizer.step()
        self._update_ema()

        self.total_updates += 1
        per_token_np = per_token.detach().cpu().numpy().astype(np.float32, copy=False)
        per_token_mean = np.mean(per_token_np, axis=0) if per_token_np.size > 0 else np.zeros((pred_tokens.shape[1],), dtype=np.float32)

        attn_t = self.model.get_last_attention()
        edge_attr_max = 0.0
        if attn_t is not None:
            attn_np = attn_t.detach().cpu().numpy().astype(np.float32, copy=False)
            attn_mean = np.mean(attn_np, axis=0) if attn_np.size > 0 else np.zeros((pred_tokens.shape[1], pred_tokens.shape[1]), dtype=np.float32)
            edge_attr = attn_mean * per_token_mean[:, None]
            self.last_attention = attn_mean
            self.last_edge_attribution = edge_attr
            edge_attr_max = float(np.max(edge_attr)) if edge_attr.size > 0 else 0.0
        else:
            self.last_attention = None
            self.last_edge_attribution = None
        self.last_per_token_loss = per_token_mean

        self.last_train_metrics = {
            "loss": float(loss.detach().cpu().item()),
            "gate_loss": float(gate_loss.detach().cpu().item()),
            "type_loss": float(type_loss.detach().cpu().item()),
            "geom_loss": float(geom_loss.detach().cpu().item()),
            "despawn_loss": float(despawn_loss.detach().cpu().item()),
            "jump_loss": float(jump_loss.detach().cpu().item()),
            "collision_loss": float(collision_loss.detach().cpu().item()),
            "contact_drop_loss": float(contact_drop_loss.detach().cpu().item()),
            "buffer_size": float(len(self.buffer)),
            "total_updates": float(self.total_updates),
            "per_token_loss_mean": float(np.mean(per_token_mean)) if per_token_mean.size > 0 else 0.0,
            "per_token_loss_max": float(np.max(per_token_mean)) if per_token_mean.size > 0 else 0.0,
            "edge_attr_max": float(edge_attr_max),
            "geom_logvar_mean": float(pred_geom_logvar.mean().detach().cpu().item()) if pred_geom_logvar is not None else 0.0,
            "hazard_loss": float(hazard_loss.detach().cpu().item()),
            "batch_event_fraction": float(np.mean(event_t_np)) if event_t_np.size > 0 else 0.0,
            "batch_contact_fraction": float(np.mean(contact_t_np)) if contact_t_np.size > 0 else 0.0,
            "batch_event_heavy": float(1.0 if event_heavy else 0.0),
            "mixed_precision_enabled": float(1.0 if self._amp_enabled else 0.0),
        }
        go = 1 + self.schema.num_types
        try:
            xy_delta = tokens_tp1_np[:, :, go : go + 2] - tokens_t_np[:, :, go : go + 2]
            xy_l2 = np.linalg.norm(xy_delta, axis=-1)
            gate_flip = ((tokens_t_np[:, :, 0] > 0.5) != (tokens_tp1_np[:, :, 0] > 0.5)).astype(np.float32)
            event_window_mask = (event_window_t_np.reshape(-1, 1) > 0.5)
            event_true_mask = (event_t_np.reshape(-1, 1) > 0.5)
            non_event_window_mask = ~event_window_mask
            non_event_true_mask = ~event_true_mask
            self.last_train_metrics["adjacent_link_l2_mean"] = float(np.mean(xy_l2))
            self.last_train_metrics["adjacent_link_l2_p95"] = float(np.percentile(xy_l2, 95))
            self.last_train_metrics["close_frac_lt_0p25"] = float(np.mean((xy_l2 < 0.25).astype(np.float32)))
            self.last_train_metrics["token_gate_flip_frac"] = float(np.mean(gate_flip))
            self.last_train_metrics["token_gate_flip_frac_event_window"] = (
                float(np.mean(gate_flip[event_window_mask])) if np.any(event_window_mask) else float("nan")
            )
            self.last_train_metrics["token_gate_flip_frac_non_event_window"] = (
                float(np.mean(gate_flip[non_event_window_mask])) if np.any(non_event_window_mask) else float("nan")
            )
            self.last_train_metrics["token_gate_flip_frac_event_true"] = (
                float(np.mean(gate_flip[event_true_mask])) if np.any(event_true_mask) else float("nan")
            )
            self.last_train_metrics["token_gate_flip_frac_non_event_true"] = (
                float(np.mean(gate_flip[non_event_true_mask])) if np.any(non_event_true_mask) else float("nan")
            )
        except Exception:
            self.last_train_metrics["adjacent_link_l2_mean"] = float("nan")
            self.last_train_metrics["adjacent_link_l2_p95"] = float("nan")
            self.last_train_metrics["close_frac_lt_0p25"] = float("nan")
            self.last_train_metrics["token_gate_flip_frac"] = float("nan")
            self.last_train_metrics["token_gate_flip_frac_event_window"] = float("nan")
            self.last_train_metrics["token_gate_flip_frac_non_event_window"] = float("nan")
            self.last_train_metrics["token_gate_flip_frac_event_true"] = float("nan")
            self.last_train_metrics["token_gate_flip_frac_non_event_true"] = float("nan")
        self.last_train_metrics.update(contract_metrics)
        self.last_train_metrics.update(self.get_transition_stats())
        if per_token_mean.size > 0:
            self.last_train_metrics["per_token_loss_argmax"] = int(np.argmax(per_token_mean))
            self.last_train_metrics["edge_attribution_max_per_token"] = [
                float(v) for v in np.max(self.last_edge_attribution, axis=1)
            ] if self.last_edge_attribution is not None else []
        return dict(self.last_train_metrics)

    def _set_match_targets(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        previous_tokens: torch.Tensor,
        elapsed_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not bool(self.config.use_hungarian_matching):
            return target

        pred_np = predicted.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        prev_np = previous_tokens.detach().cpu().numpy()
        elapsed_np = elapsed_t.detach().cpu().numpy() if elapsed_t is not None else None
        prev_embed_np: Optional[np.ndarray] = None
        tgt_embed_np: Optional[np.ndarray] = None
        if float(max(0.0, self.config.match_learned_embed_weight)) > 0.0:
            with torch.no_grad():
                zero_elapsed_prev = torch.zeros_like(previous_tokens[..., 0])
                zero_elapsed_tgt = torch.zeros_like(target[..., 0])
                prev_embed = self.model.encode_tokens(previous_tokens, elapsed_time=zero_elapsed_prev)
                tgt_embed = self.model.encode_tokens(target, elapsed_time=zero_elapsed_tgt)
                prev_embed_np = prev_embed.detach().cpu().numpy()
                tgt_embed_np = tgt_embed.detach().cpu().numpy()

        bsz, slots, _ = pred_np.shape
        perms = np.zeros((bsz, slots), dtype=np.int64)

        for b in range(bsz):
            cost = self._pairwise_cost_matrix(
                pred_np[b],
                tgt_np[b],
                prev_np[b],
                None if elapsed_np is None else elapsed_np[b],
                None if prev_embed_np is None else prev_embed_np[b],
                None if tgt_embed_np is None else tgt_embed_np[b],
            )
            perms[b] = _hungarian_assignment(cost)

        perms_t = torch.from_numpy(perms).to(predicted.device)
        batch_idx = torch.arange(bsz, device=predicted.device).unsqueeze(1)
        return target[batch_idx, perms_t]

    def _pairwise_cost_matrix(
        self,
        pred: np.ndarray,
        tgt: np.ndarray,
        prev: np.ndarray,
        elapsed: Optional[np.ndarray] = None,
        prev_embed: Optional[np.ndarray] = None,
        tgt_embed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        tdim = self.schema.num_types
        go = 1 + tdim

        pred_gate = pred[:, None, 0]
        tgt_gate = tgt[None, :, 0]
        gate_cost = (pred_gate - tgt_gate) ** 2

        pred_type = pred[:, None, 1:go]
        tgt_type = tgt[None, :, 1:go]
        type_cost = np.sum((pred_type - tgt_type) ** 2, axis=-1)

        pred_geom = pred[:, None, go:]
        tgt_geom = tgt[None, :, go:]
        geom_diff = pred_geom - tgt_geom
                                                                                 
        geom_diff[..., 2] = (geom_diff[..., 2] + np.pi) % TWO_PI - np.pi
        geom_w = np.asarray([1.0, 1.0, 0.1, 1.0, 1.0, 0.5], dtype=np.float32).reshape(1, 1, -1)
        ad = np.abs(geom_diff)
        geom_huber = np.where(
            ad <= GEOM_HUBER_DELTA,
            0.5 * (ad ** 2) / GEOM_HUBER_DELTA,
            ad - 0.5 * GEOM_HUBER_DELTA,
        )
        geom_weight = np.minimum(pred_gate, tgt_gate)
        geom_cost = geom_weight * np.sum(geom_huber * geom_w, axis=-1)

        prev_gate = prev[:, None, 0]
        prev_geom = prev[:, None, go:]
        track_diff = prev_geom - tgt_geom
        track_diff[..., 2] = (track_diff[..., 2] + np.pi) % TWO_PI - np.pi
        at = np.abs(track_diff)
        track_huber = np.where(
            at <= GEOM_HUBER_DELTA,
            0.5 * (at ** 2) / GEOM_HUBER_DELTA,
            at - 0.5 * GEOM_HUBER_DELTA,
        )
        tgt_type_is_pod = tgt[None, :, 1 + TYPE_POD] > 0.5
        spawn_mask = (
            (prev_gate <= self.config.gate_present_threshold)
            & (tgt_gate > self.config.gate_present_threshold)
            & tgt_type_is_pod
        )
        geom_cost = np.where(spawn_mask, geom_cost * float(self.config.spawn_geom_weight_scale), geom_cost)
        track_weight = np.minimum(prev_gate, tgt_gate)
        track_cost = track_weight * np.sum(track_huber * geom_w, axis=-1)

        memory_cost = 0.0
        if float(max(0.0, self.config.match_memory_weight)) > 0.0:
            prev_type = np.argmax(prev[:, 1:go], axis=-1)
            tgt_type_idx = np.argmax(tgt[:, 1:go], axis=-1)
            same_type = (prev_type[:, None] == tgt_type_idx[None, :]).astype(np.float32)
            non_empty = ((prev_type[:, None] != TYPE_EMPTY) & (tgt_type_idx[None, :] != TYPE_EMPTY)).astype(np.float32)
            elapsed_arr = np.asarray(elapsed if elapsed is not None else np.zeros((prev.shape[0],), dtype=np.float32), dtype=np.float32).reshape(-1)
            decay_scale = float(max(1e-3, self.config.match_memory_decay_scale))
            memory_decay = np.exp(-np.clip(elapsed_arr, 0.0, 1e6)[:, None] / decay_scale).astype(np.float32)
            tgt_on = (tgt_gate > self.config.gate_present_threshold).astype(np.float32)
            memory_weight = same_type * non_empty * memory_decay * tgt_on
            if bool(self.config.match_memory_reappearance_only):
                prev_off = (prev_gate <= self.config.gate_present_threshold).astype(np.float32)
                memory_weight = memory_weight * prev_off
            memory_cost = memory_weight * np.sum(track_huber * geom_w, axis=-1)

        embed_cost = 0.0
        if (
            float(max(0.0, self.config.match_learned_embed_weight)) > 0.0
            and prev_embed is not None
            and tgt_embed is not None
        ):
            pe = np.asarray(prev_embed, dtype=np.float32)
            te = np.asarray(tgt_embed, dtype=np.float32)
            pe = pe / np.clip(np.linalg.norm(pe, axis=-1, keepdims=True), 1e-6, None)
            te = te / np.clip(np.linalg.norm(te, axis=-1, keepdims=True), 1e-6, None)
            sim = np.einsum("id,jd->ij", pe, te).astype(np.float32, copy=False)
            embed_cost = 0.5 * (1.0 - np.clip(sim, -1.0, 1.0))
            prev_type = np.argmax(prev[:, 1:go], axis=-1)
            tgt_type_idx = np.argmax(tgt[:, 1:go], axis=-1)
            same_type = (prev_type[:, None] == tgt_type_idx[None, :]).astype(np.float32)
            embed_weight = same_type.astype(np.float32)
            elapsed_arr = np.asarray(elapsed if elapsed is not None else np.zeros((prev.shape[0],), dtype=np.float32), dtype=np.float32).reshape(-1)
            decay_scale = float(max(1e-3, self.config.match_learned_embed_decay_scale))
            embed_decay = np.exp(-np.clip(elapsed_arr, 0.0, 1e6)[:, None] / decay_scale).astype(np.float32)
            embed_weight = embed_weight * embed_decay
            if bool(self.config.match_learned_embed_reappearance_only):
                prev_off = (prev_gate <= self.config.gate_present_threshold).astype(np.float32)
                tgt_on = (tgt_gate > self.config.gate_present_threshold).astype(np.float32)
                embed_weight = embed_weight * prev_off * tgt_on
            embed_cost = embed_cost * embed_weight

        total = (
            float(self.config.loss_gate_weight) * gate_cost
            + float(self.config.loss_type_weight) * type_cost
            + float(self.config.loss_geom_weight) * geom_cost
            + float(self.config.track_inertia_weight) * track_cost
            + float(self.config.match_memory_weight) * memory_cost
            + float(self.config.match_learned_embed_weight) * embed_cost
        )
        total = np.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
        return total.astype(np.float64, copy=False)

    @staticmethod
    def _agent_pod_penetration(agent_geom: torch.Tensor, pod_geom: torch.Tensor) -> torch.Tensor:
                                                                   
        ax = agent_geom[..., 0:1].unsqueeze(2)
        ay = agent_geom[..., 1:2].unsqueeze(2)
        px = pod_geom[..., 0:1].unsqueeze(1)
        py = pod_geom[..., 1:2].unsqueeze(1)
        dist = torch.sqrt((ax - px) ** 2 + (ay - py) ** 2 + EPS)

        ar = torch.maximum(agent_geom[..., 3:4].abs(), agent_geom[..., 4:5].abs()).unsqueeze(2)
        pr = torch.maximum(pod_geom[..., 3:4].abs(), pod_geom[..., 4:5].abs()).unsqueeze(1)
        return F.relu(ar + pr - dist).squeeze(-1)

    @staticmethod
    def _agent_wall_penetration(agent_geom: torch.Tensor, wall_geom: torch.Tensor) -> torch.Tensor:
                                                                    
        ax = agent_geom[..., 0:1].unsqueeze(2)
        ay = agent_geom[..., 1:2].unsqueeze(2)
        wx = wall_geom[..., 0:1].unsqueeze(1)
        wy = wall_geom[..., 1:2].unsqueeze(1)
        phi = wall_geom[..., 2:3].unsqueeze(1)
        w_sx = wall_geom[..., 3:4].abs().unsqueeze(1)
        w_sy = wall_geom[..., 4:5].abs().unsqueeze(1)
        w_wt = wall_geom[..., 5:6].abs().unsqueeze(1)
        half_th = torch.minimum(w_sy, w_wt)

        dx = ax - wx
        dy = ay - wy
        c = torch.cos(phi)
        s = torch.sin(phi)
        rx = c * dx + s * dy
        ry = -s * dx + c * dy

        qx = F.relu(torch.abs(rx) - w_sx)
        qy = F.relu(torch.abs(ry) - half_th)
        dist = torch.sqrt(qx * qx + qy * qy + EPS)

        ar = torch.maximum(agent_geom[..., 3:4].abs(), agent_geom[..., 4:5].abs()).unsqueeze(2)
        return F.relu(ar - dist).squeeze(-1)

    def _dynamics_loss(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        prev_tokens: torch.Tensor,
        hazard_logits: torch.Tensor,
        jump_aux: torch.Tensor,
        pred_geom_logvar: Optional[torch.Tensor] = None,
        event_window_t: Optional[torch.Tensor] = None,
        pod_spawn_window_t: Optional[torch.Tensor] = None,
        pod_spawn_future_t: Optional[torch.Tensor] = None,
        contact_candidate_t: Optional[torch.Tensor] = None,
        elapsed_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        tdim = self.schema.num_types
        go = 1 + tdim

        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=-1.0)
        prev_tokens = torch.nan_to_num(prev_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
        hazard_logits = torch.nan_to_num(hazard_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        jump_aux = torch.nan_to_num(jump_aux, nan=0.0, posinf=20.0, neginf=-20.0)

        pred_gate = pred[..., 0].clamp(0.0, 1.0)
        tgt_gate = tgt[..., 0].clamp(0.0, 1.0)
        prev_gate = prev_tokens[..., 0].clamp(0.0, 1.0)
        gate_loss_raw = (pred_gate - tgt_gate) ** 2
        transition_mask = ((prev_gate > self.config.gate_present_threshold) != (tgt_gate > self.config.gate_present_threshold)).to(pred_gate.dtype)
        if bool(self.config.gate_transition_weighted):
            event_w = float(max(1.0, self.config.event_transition_weight))
            gate_loss_per_token = gate_loss_raw * (1.0 + (event_w - 1.0) * transition_mask)
        else:
            gate_loss_per_token = gate_loss_raw
        gate_loss = gate_loss_per_token.mean()

        pred_type = pred[..., 1:go].clamp(EPS, 1.0)
        tgt_type = tgt[..., 1:go]
        type_loss_per_token = -(tgt_type * torch.log(pred_type)).sum(dim=-1)
        type_loss = type_loss_per_token.mean()

        pred_geom = pred[..., go:]
        tgt_geom = tgt[..., go:]
        geom_diff = pred_geom - tgt_geom
        geom_diff_phi = torch.remainder(geom_diff[..., 2:3] + np.pi, TWO_PI) - np.pi
        geom_diff = torch.cat([geom_diff[..., :2], geom_diff_phi, geom_diff[..., 3:]], dim=-1)
        geom_w = torch.tensor([1.0, 1.0, 0.1, 1.0, 1.0, 0.5], device=pred.device, dtype=pred.dtype).view(1, 1, -1)
        ad = torch.abs(geom_diff)
        geom_huber = torch.where(
            ad <= GEOM_HUBER_DELTA,
            0.5 * (ad ** 2) / GEOM_HUBER_DELTA,
            ad - 0.5 * GEOM_HUBER_DELTA,
        )
                                                                               
                                                                  
        geom_weight = tgt_gate
        tgt_type = tgt[..., 1:go]
        tgt_type_is_pod = (tgt_type[..., TYPE_POD] > 0.5).to(pred_gate.dtype)
        spawn_mask = (
            ((prev_gate <= self.config.gate_present_threshold) & (tgt_gate > self.config.gate_present_threshold)).to(pred_gate.dtype)
            * tgt_type_is_pod
        )
        geom_weight = geom_weight * (1.0 - spawn_mask) + geom_weight * spawn_mask * float(self.config.spawn_geom_weight_scale)

        if pred_geom_logvar is not None:
            lv = pred_geom_logvar.clamp(self.config.geom_logvar_min, self.config.geom_logvar_max)
            inv_var = torch.exp(-lv)
                                                                               
                                                                          
            geom_nll_dim = geom_huber * inv_var * geom_w + 0.1 * F.softplus(lv) * geom_w
            geom_nll_token = geom_nll_dim.sum(dim=-1)
            geom_loss_per_token = geom_weight * geom_nll_token
        else:
            geom_sq = (geom_huber * geom_w).sum(dim=-1)
            geom_loss_per_token = geom_weight * geom_sq

                                                                                            
        geom_loss = geom_loss_per_token.mean()

        if hazard_logits.shape[-1] >= 2:
            spawn_logits = hazard_logits[..., 0]
            despawn_logits = hazard_logits[..., 1]
        else:
            spawn_logits = hazard_logits[..., 0]
            despawn_logits = -hazard_logits[..., 0]
        spawn_target = ((prev_gate <= self.config.gate_present_threshold) & (tgt_gate > self.config.gate_present_threshold)).to(
            spawn_logits.dtype
        )
        despawn_target = ((prev_gate > self.config.gate_present_threshold) & (tgt_gate <= self.config.gate_present_threshold)).to(
            despawn_logits.dtype
        )
        nonempty_mask = (1.0 - tgt_type[..., TYPE_EMPTY]).clamp(0.0, 1.0)
        gamma = float(max(0.0, self.config.focal_gamma))
        alpha = float(np.clip(self.config.focal_alpha, 0.01, 0.99))
        hazard_bce = F.binary_cross_entropy_with_logits(spawn_logits, spawn_target, reduction="none")
        despawn_bce = F.binary_cross_entropy_with_logits(despawn_logits, despawn_target, reduction="none")
                                                                    
        spawn_pt = torch.exp(-hazard_bce).clamp(EPS, 1.0)
        despawn_pt = torch.exp(-despawn_bce).clamp(EPS, 1.0)
        spawn_alpha = spawn_target * alpha + (1.0 - spawn_target) * (1.0 - alpha)
        despawn_alpha = despawn_target * alpha + (1.0 - despawn_target) * (1.0 - alpha)
        hazard_loss_per_token = spawn_alpha * ((1.0 - spawn_pt).clamp_min(EPS) ** gamma) * hazard_bce * nonempty_mask
        despawn_loss_per_token = despawn_alpha * ((1.0 - despawn_pt).clamp_min(EPS) ** gamma) * despawn_bce * nonempty_mask

        if pod_spawn_future_t is not None and float(max(0.0, self.config.loss_spawn_lookahead_weight)) > 0.0:
            future_window = pod_spawn_future_t.reshape(-1, 1).to(pred.dtype)
            pod_slot_mask = (tgt_type[..., TYPE_POD] > 0.5).to(pred.dtype)
            absent_prev = (prev_gate <= self.config.gate_present_threshold).to(pred.dtype)
            lookahead_target = future_window * pod_slot_mask * absent_prev
            lookahead_bce = F.binary_cross_entropy_with_logits(spawn_logits, lookahead_target, reduction="none")
            lookahead_pt = torch.exp(-lookahead_bce).clamp(EPS, 1.0)
            lookahead_alpha = lookahead_target * alpha + (1.0 - lookahead_target) * (1.0 - alpha)
            lookahead_loss_per_token = lookahead_alpha * ((1.0 - lookahead_pt).clamp_min(EPS) ** gamma) * lookahead_bce * pod_slot_mask
            hazard_loss_per_token = hazard_loss_per_token + float(max(0.0, self.config.loss_spawn_lookahead_weight)) * lookahead_loss_per_token

        if elapsed_t is not None:
            elapsed_local = elapsed_t.to(pred.dtype)
            if elapsed_local.dim() == 3:
                elapsed_local = elapsed_local.squeeze(-1)
            early_factor = torch.exp(-elapsed_local / float(max(1e-3, self.config.elapsed_time_scale)))
            absent_prev = (prev_gate <= self.config.gate_present_threshold).to(pred.dtype)
            spawn_prob = torch.sigmoid(spawn_logits)
            spawn_prior_context = torch.zeros_like(spawn_prob)
            if pod_spawn_window_t is not None:
                psw = pod_spawn_window_t.reshape(-1, 1).to(pred.dtype)
                tgt_type_local = tgt[..., 1:go]
                pod_slot_mask = (tgt_type_local[..., TYPE_POD] > 0.5).to(pred.dtype)
                spawn_prior_context = torch.maximum(spawn_prior_context, psw * pod_slot_mask)
            context_relief = float(np.clip(self.config.spawn_prior_context_relief, 0.0, 1.0))
            blind_factor = 1.0 - context_relief * spawn_prior_context
            spawn_prior_per_token = spawn_prob * early_factor * absent_prev * nonempty_mask * blind_factor
            hazard_loss_per_token = hazard_loss_per_token + float(max(0.0, self.config.loss_spawn_prior_weight)) * spawn_prior_per_token

                                          
        a0 = 0
        a1 = self.schema.num_agent_slots
        p0 = a1
        p1 = p0 + self.schema.num_pod_slots
        w0 = p1
        w1 = w0 + self.schema.num_wall_slots

        pred_geom_agent = pred_geom[:, a0:a1, :]
        pred_geom_pod = pred_geom[:, p0:p1, :]
        pred_geom_wall = pred_geom[:, w0:w1, :]
        prev_geom_agent = prev_tokens[:, a0:a1, go:]
        prev_geom_pod = prev_tokens[:, p0:p1, go:]
        prev_geom_wall = prev_tokens[:, w0:w1, go:]
        tgt_geom_wall = tgt[:, w0:w1, go:]

        pred_gate_agent = pred_gate[:, a0:a1]
        pred_gate_pod = pred_gate[:, p0:p1]
        pred_gate_wall = pred_gate[:, w0:w1]
        prev_gate_pod = prev_gate[:, p0:p1]
        prev_gate_wall = prev_gate[:, w0:w1]
        tgt_gate_pod = tgt_gate[:, p0:p1]
        tgt_gate_wall = tgt_gate[:, w0:w1]

        if bool(self.config.collision_use_target_wall):
            wall_geom_for_collision = tgt_geom_wall
            wall_gate_for_collision = tgt_gate_wall
        else:
            wall_geom_for_collision = prev_geom_wall
            wall_gate_for_collision = prev_gate_wall
        pen_agent_wall = self._agent_wall_penetration(pred_geom_agent, wall_geom_for_collision)         
        wall_present = (wall_gate_for_collision > self.config.gate_present_threshold).to(pred.dtype)
        wall_pen_w = pred_gate_agent.unsqueeze(2) * wall_present.unsqueeze(1)
        collision_loss = (pen_agent_wall * wall_pen_w).sum() / (wall_pen_w.sum() + EPS)

        pen_prev_pod = self._agent_pod_penetration(prev_geom_agent, prev_geom_pod)         
        pen_prev_wall = self._agent_wall_penetration(prev_geom_agent, prev_geom_wall)         
                                                                                      
        prev_agent_on = (prev_gate[:, a0:a1] > self.config.gate_present_threshold).to(pred.dtype)       
        prev_pod_on = (prev_gate_pod > self.config.gate_present_threshold).to(pred.dtype)       
        prev_wall_on = (prev_gate_wall > self.config.gate_present_threshold).to(pred.dtype)       
        pen_prev_pod = pen_prev_pod * prev_agent_on.unsqueeze(2) * prev_pod_on.unsqueeze(1)
        pen_prev_wall = pen_prev_wall * prev_agent_on.unsqueeze(2) * prev_wall_on.unsqueeze(1)
        contact_prev_pod = (torch.amax(pen_prev_pod, dim=1) > 0.0).to(pred.dtype)
        contact_prev_wall = (torch.amax(pen_prev_wall, dim=1) > 0.0).to(pred.dtype)

        pod_drop_true = (
            (prev_gate_pod > self.config.gate_present_threshold)
            & (tgt_gate_pod <= self.config.gate_present_threshold)
        ).to(pred.dtype)
        wall_drop_true = (
            (prev_gate_wall > self.config.gate_present_threshold)
            & (tgt_gate_wall <= self.config.gate_present_threshold)
        ).to(pred.dtype)
        contact_drop_target_pod = contact_prev_pod * pod_drop_true
        contact_drop_target_wall = contact_prev_wall * wall_drop_true
        contact_w = float(max(1.0, self.config.contact_condition_weight))
        slot_contact_weight = torch.ones_like(despawn_loss_per_token)
        slot_contact_weight[:, p0:p1] = 1.0 + (contact_w - 1.0) * contact_prev_pod
        slot_contact_weight[:, w0:w1] = 1.0 + (contact_w - 1.0) * contact_prev_wall
        if contact_candidate_t is not None:
            cglob = contact_candidate_t.reshape(-1, 1).to(pred.dtype)
            slot_contact_weight = slot_contact_weight * (1.0 + (contact_w - 1.0) * cglob)
        despawn_loss_per_token = despawn_loss_per_token * slot_contact_weight
        event_scale = None
        if event_window_t is not None:
            ew = event_window_t.reshape(-1, 1).to(pred.dtype)
            event_scale = 1.0 + (float(max(1.0, self.config.event_transition_weight)) - 1.0) * torch.maximum(transition_mask, ew)
        else:
            event_scale = 1.0 + (float(max(1.0, self.config.event_transition_weight)) - 1.0) * transition_mask
        hazard_loss_per_token = hazard_loss_per_token * event_scale

        hazard_norm = torch.sum(nonempty_mask) + EPS
        despawn_norm = torch.sum(nonempty_mask * slot_contact_weight) + EPS
        hazard_loss = hazard_loss_per_token.sum() / hazard_norm
        despawn_loss = despawn_loss_per_token.sum() / despawn_norm

                                                                                                             
        prev_geom = prev_tokens[..., go:]
        disp_true = torch.linalg.norm(tgt_geom[..., 0:2] - prev_geom[..., 0:2], dim=-1)
        jump_target = (disp_true > float(max(1e-3, self.config.jump_disp_threshold))).to(pred.dtype)
        jump_logits = jump_aux[..., 0]
        jump_xy = torch.tanh(jump_aux[..., 1:3])
        jump_bce = F.binary_cross_entropy_with_logits(jump_logits, jump_target, reduction="none")
        tgt_xy = tgt_geom[..., 0:2].clamp(-1.0, 1.0)
        jump_xy_l1 = F.smooth_l1_loss(jump_xy, tgt_xy, reduction="none").sum(dim=-1)
        jump_loss_per_token = jump_bce + jump_target * jump_xy_l1
        jump_loss = jump_loss_per_token.mean()
                                                                                                  
        contact_drop_loss_pod = (contact_drop_target_pod * pred_gate_pod).sum() / (contact_drop_target_pod.sum() + EPS)
        contact_drop_loss_wall = (contact_drop_target_wall * pred_gate_wall).sum() / (contact_drop_target_wall.sum() + EPS)
        contact_drop_loss = 0.5 * (contact_drop_loss_pod + contact_drop_loss_wall)

        if bool(self.config.train_type_dynamics):
            type_train_per_token = type_loss_per_token
            type_train = type_loss
        else:
            type_train_per_token = torch.zeros_like(type_loss_per_token)
            type_train = type_loss.new_zeros(())

        per_token_total = (
            float(self.config.loss_gate_weight) * gate_loss_per_token
            + float(self.config.loss_type_weight) * type_train_per_token
            + float(self.config.loss_geom_weight) * geom_loss_per_token
            + float(self.config.loss_hazard_weight) * hazard_loss_per_token
            + float(self.config.loss_despawn_weight) * despawn_loss_per_token
            + float(self.config.loss_jump_weight) * jump_loss_per_token
        )

        loss = (
            float(self.config.loss_gate_weight) * gate_loss
            + float(self.config.loss_type_weight) * type_train
            + float(self.config.loss_geom_weight) * geom_loss
            + float(self.config.loss_hazard_weight) * hazard_loss
            + float(self.config.loss_despawn_weight) * despawn_loss
            + float(self.config.loss_jump_weight) * jump_loss
            + float(self.config.loss_collision_weight) * collision_loss
            + float(self.config.loss_contact_drop_weight) * contact_drop_loss
        )
        pod_drop_pred = (
            (prev_gate_pod > self.config.gate_present_threshold)
            & (pred_gate_pod <= self.config.gate_present_threshold)
        ).to(pred.dtype)
        wall_drop_pred = (
            (prev_gate_wall > self.config.gate_present_threshold)
            & (pred_gate_wall <= self.config.gate_present_threshold)
        ).to(pred.dtype)
        pod_drop_recall = (pod_drop_pred * contact_drop_target_pod).sum() / (contact_drop_target_pod.sum() + EPS)
        wall_drop_recall = (wall_drop_pred * contact_drop_target_wall).sum() / (contact_drop_target_wall.sum() + EPS)

        pos_err = torch.linalg.norm(pred_geom[..., 0:2] - tgt_geom[..., 0:2], dim=-1)
        event_step = torch.any(transition_mask > 0.0, dim=1)
        wall_contact_step = torch.any(contact_prev_wall > 0.0, dim=1)
        pod_contact_step = torch.any(contact_prev_pod > 0.0, dim=1)
        normal_step = ~(event_step | wall_contact_step | pod_contact_step)
        agent_pos_err = pos_err[:, a0:a1].mean(dim=1)
        pred_on = pred_gate > self.config.gate_present_threshold
        tgt_on = tgt_gate > self.config.gate_present_threshold
        event_slot_mask = transition_mask > 0.0
        if torch.any(event_slot_mask):
            gate_event_acc = torch.mean((pred_on[event_slot_mask] == tgt_on[event_slot_mask]).to(pred.dtype))
        else:
            gate_event_acc = pred.new_tensor(float("nan"))
        pen_rate = torch.mean((pen_agent_wall > 0.0).to(pred.dtype) * wall_pen_w)
        pen_ind = (pen_agent_wall > 0.0).to(pred.dtype)
        wall_present_3d = wall_present.unsqueeze(1)
        wall_absent_3d = 1.0 - wall_present_3d
        pen_w_present = pred_gate_agent.unsqueeze(2) * wall_present_3d
        pen_w_absent = pred_gate_agent.unsqueeze(2) * wall_absent_3d
        pen_rate_present = (pen_ind * pen_w_present).sum() / (pen_w_present.sum() + EPS)
        pen_rate_absent = (pen_ind * pen_w_absent).sum() / (pen_w_absent.sum() + EPS)
        contract_metrics = {
            "contract_penetration_rate": float(pen_rate.detach().cpu().item()),
            "contract_penetration_rate_wall_present": float(pen_rate_present.detach().cpu().item()),
            "contract_penetration_rate_wall_absent": float(pen_rate_absent.detach().cpu().item()),
            "contract_pod_drop_recall": float(pod_drop_recall.detach().cpu().item()),
            "contract_wall_drop_recall": float(wall_drop_recall.detach().cpu().item()),
            "jump_target_rate": float(torch.mean(jump_target).detach().cpu().item()),
            "jump_pred_rate": float(torch.mean((torch.sigmoid(jump_logits) > 0.5).to(pred.dtype)).detach().cpu().item()),
            "jump_rate_gap_abs": float(torch.abs(torch.mean((torch.sigmoid(jump_logits) > 0.5).to(pred.dtype)) - torch.mean(jump_target)).detach().cpu().item()),
            "split_normal_pos_error": float(torch.mean(agent_pos_err[normal_step]).detach().cpu().item()) if torch.any(normal_step) else float("nan"),
            "split_wall_contact_pos_error": float(torch.mean(agent_pos_err[wall_contact_step]).detach().cpu().item()) if torch.any(wall_contact_step) else float("nan"),
            "split_pod_contact_pos_error": float(torch.mean(agent_pos_err[pod_contact_step]).detach().cpu().item()) if torch.any(pod_contact_step) else float("nan"),
            "split_event_pos_error": float(torch.mean(agent_pos_err[event_step]).detach().cpu().item()) if torch.any(event_step) else float("nan"),
            "split_event_gate_accuracy": (
                float(gate_event_acc.detach().cpu().item())
                if bool(torch.isfinite(gate_event_acc).detach().cpu().item())
                else float("nan")
            ),
        }

        return (
            loss,
            gate_loss,
            type_loss,
            geom_loss,
            hazard_loss,
            despawn_loss,
            jump_loss,
            collision_loss,
            contact_drop_loss,
            per_token_total,
            contract_metrics,
        )

    @torch.no_grad()
    def _update_ema(self):
        decay = float(np.clip(self.config.ema_decay, 0.0, 0.99999))
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.copy_(b)

    def build_debug_views(
        self,
        *,
        frame_t: Any,
        tokens_t: np.ndarray,
        predicted_tokens_for_target: Optional[np.ndarray] = None,
        frame_target_for_prediction: Any = None,
    ) -> Dict[str, Any]:
        rgb_t = _to_rgb_frame(frame_t)
        if rgb_t is None:
            h, w = 84, 84
        else:
            h, w = int(rgb_t.shape[0]), int(rgb_t.shape[1])

        recon_t = self.renderer.render(tokens_t, height=h, width=w)
        out: Dict[str, Any] = {
            "reconstruction": recon_t,
            "predicted_reconstruction": None,
            "reconstruction_error": float("nan"),
            "predicted_reconstruction_error": float("nan"),
            "center_pixel": {"x": int(w // 2), "y": int(h // 2)},
            "reconstruction_center_contributors": self.renderer.explain_pixel(
                tokens_t, height=h, width=w, py=int(h // 2), px=int(w // 2)
            ),
            "predicted_center_contributors": [],
        }

        if predicted_tokens_for_target is not None:
            out["predicted_reconstruction"] = self.renderer.render(predicted_tokens_for_target, height=h, width=w)
            out["predicted_center_contributors"] = self.renderer.explain_pixel(
                predicted_tokens_for_target, height=h, width=w, py=int(h // 2), px=int(w // 2)
            )

        if rgb_t is not None:
            err = np.abs(rgb_t - recon_t)
            out["reconstruction_error"] = float(np.mean(err))

        rgb_target = _to_rgb_frame(frame_target_for_prediction)
        if rgb_target is not None and out["predicted_reconstruction"] is not None:
            pred_err = np.abs(rgb_target - out["predicted_reconstruction"])
            out["predicted_reconstruction_error"] = float(np.mean(pred_err))

        return out

    def summarize_tokens(self, tokens: np.ndarray) -> Dict[str, Any]:
        t = np.asarray(tokens, dtype=np.float32)
        gates = t[:, 0]
        types = _argmax_type(t, self.schema)
        geom = t[:, 1 + self.schema.num_types :]

        summary = {
            "active_slots": int(np.sum(gates > 0.5)),
            "gate_mean": float(np.mean(gates)),
            "gate_max": float(np.max(gates)) if gates.size > 0 else 0.0,
            "gate_min": float(np.min(gates)) if gates.size > 0 else 0.0,
            "active_per_type": {
                TYPE_NAMES[t_id]: int(np.sum((types == t_id) & (gates > 0.5)))
                for t_id in (TYPE_AGENT, TYPE_POD, TYPE_WALL, TYPE_CONDITION, TYPE_EMPTY)
            },
            "geom_ranges": {
                "x": [float(np.min(geom[:, 0])), float(np.max(geom[:, 0]))] if geom.size > 0 else [0.0, 0.0],
                "y": [float(np.min(geom[:, 1])), float(np.max(geom[:, 1]))] if geom.size > 0 else [0.0, 0.0],
                "sx": [float(np.min(geom[:, 3])), float(np.max(geom[:, 3]))] if geom.size > 0 else [0.0, 0.0],
                "sy": [float(np.min(geom[:, 4])), float(np.max(geom[:, 4]))] if geom.size > 0 else [0.0, 0.0],
            },
        }
        return summary

    def get_relational_debug(self) -> Dict[str, Any]:
        return {
            "per_token_loss": (
                np.asarray(self.last_per_token_loss, dtype=np.float32).copy()
                if self.last_per_token_loss is not None
                else None
            ),
            "attention": (
                np.asarray(self.last_attention, dtype=np.float32).copy()
                if self.last_attention is not None
                else None
            ),
            "edge_attribution": (
                np.asarray(self.last_edge_attribution, dtype=np.float32).copy()
                if self.last_edge_attribution is not None
                else None
            ),
        }

    def get_transition_stats(self) -> Dict[str, float]:
        out = dict(self.transition_stats)
        total = float(max(1.0, out.get("total_transitions", 0.0)))
        out["pod_drop_rate"] = float(out.get("pod_drop_events", 0.0) / total)
        out["wall_drop_rate"] = float(out.get("wall_drop_events", 0.0) / total)
        out["pod_spawn_rate"] = float(out.get("pod_spawn_events", 0.0) / total)
        out["wall_spawn_rate"] = float(out.get("wall_spawn_events", 0.0) / total)
        out["event_transition_rate"] = float(out.get("event_transitions", 0.0) / total)
        out["contact_candidate_rate"] = float(out.get("contact_candidate_transitions", 0.0) / total)
        out["slot_type_mismatch_rate"] = float(out.get("slot_type_mismatch_steps", 0.0) / total)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": {
                "num_agent_slots": self.schema.num_agent_slots,
                "num_pod_slots": self.schema.num_pod_slots,
                "num_wall_slots": self.schema.num_wall_slots,
                "num_condition_slots": self.schema.num_condition_slots,
                "num_slack_slots": self.schema.num_slack_slots,
                "num_types": self.schema.num_types,
                "geom_dim": self.schema.geom_dim,
            },
            "config": {
                "buffer_capacity": self.config.buffer_capacity,
                "batch_size": self.config.batch_size,
                "warmup_transitions": self.config.warmup_transitions,
                "update_every": self.config.update_every,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "lr": self.config.lr,
                "grad_clip": self.config.grad_clip,
                "ema_decay": self.config.ema_decay,
                "loss_gate_weight": self.config.loss_gate_weight,
                "loss_type_weight": self.config.loss_type_weight,
                "loss_geom_weight": self.config.loss_geom_weight,
                "loss_hazard_weight": self.config.loss_hazard_weight,
                "track_inertia_weight": self.config.track_inertia_weight,
                "use_uncertainty_head": self.config.use_uncertainty_head,
                "geom_logvar_min": self.config.geom_logvar_min,
                "geom_logvar_max": self.config.geom_logvar_max,
                "gate_present_threshold": self.config.gate_present_threshold,
                "elapsed_time_scale": self.config.elapsed_time_scale,
                "spawn_geom_weight_scale": self.config.spawn_geom_weight_scale,
                "use_hungarian_matching": self.config.use_hungarian_matching,
                "absent_gate_decay": self.config.absent_gate_decay,
                "loss_despawn_weight": self.config.loss_despawn_weight,
                "loss_collision_weight": self.config.loss_collision_weight,
                "loss_contact_drop_weight": self.config.loss_contact_drop_weight,
                "event_transition_weight": self.config.event_transition_weight,
                "min_xy_step_scale": self.config.min_xy_step_scale,
                "max_xy_step_scale": self.config.max_xy_step_scale,
                "min_phi_step_scale": self.config.min_phi_step_scale,
                "max_phi_step_scale": self.config.max_phi_step_scale,
                "min_size_step_scale": self.config.min_size_step_scale,
                "max_size_step_scale": self.config.max_size_step_scale,
                "min_wt_step_scale": self.config.min_wt_step_scale,
                "max_wt_step_scale": self.config.max_wt_step_scale,
                "event_sample_fraction": self.config.event_sample_fraction,
                "event_heavy_every": self.config.event_heavy_every,
                "event_heavy_fraction": self.config.event_heavy_fraction,
                "event_window_steps": self.config.event_window_steps,
                "focal_gamma": self.config.focal_gamma,
                "focal_alpha": self.config.focal_alpha,
                "contact_condition_weight": self.config.contact_condition_weight,
                "loss_spawn_prior_weight": self.config.loss_spawn_prior_weight,
                "spawn_prior_context_relief": self.config.spawn_prior_context_relief,
                "pod_spawn_window_steps": self.config.pod_spawn_window_steps,
                "spawn_lookahead_steps": self.config.spawn_lookahead_steps,
                "loss_spawn_lookahead_weight": self.config.loss_spawn_lookahead_weight,
                "match_memory_weight": self.config.match_memory_weight,
                "match_memory_decay_scale": self.config.match_memory_decay_scale,
                "match_memory_reappearance_only": self.config.match_memory_reappearance_only,
                "match_learned_embed_weight": self.config.match_learned_embed_weight,
                "match_learned_embed_decay_scale": self.config.match_learned_embed_decay_scale,
                "match_learned_embed_reappearance_only": self.config.match_learned_embed_reappearance_only,
                "loss_jump_weight": self.config.loss_jump_weight,
                "jump_disp_threshold": self.config.jump_disp_threshold,
                "collision_use_target_wall": self.config.collision_use_target_wall,
                "train_type_dynamics": self.config.train_type_dynamics,
                "gate_transition_weighted": self.config.gate_transition_weighted,
            },
            "training_enabled": bool(self.training_enabled),
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "total_updates": int(self.total_updates),
            "total_seen_transitions": int(self.total_seen_transitions),
            "update_clock": int(self._update_clock),
            "last_train_metrics": dict(self.last_train_metrics),
            "transition_stats": dict(self.transition_stats),
        }

    def load_state_dict(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return

        cfg = payload.get("config", {})
        if isinstance(cfg, dict):
            self.config = CodaDynamicsConfig(
                buffer_capacity=int(cfg.get("buffer_capacity", self.config.buffer_capacity)),
                batch_size=int(cfg.get("batch_size", self.config.batch_size)),
                warmup_transitions=int(cfg.get("warmup_transitions", self.config.warmup_transitions)),
                update_every=int(cfg.get("update_every", self.config.update_every)),
                hidden_dim=int(cfg.get("hidden_dim", self.config.hidden_dim)),
                num_layers=int(cfg.get("num_layers", self.config.num_layers)),
                lr=float(cfg.get("lr", self.config.lr)),
                grad_clip=float(cfg.get("grad_clip", self.config.grad_clip)),
                ema_decay=float(cfg.get("ema_decay", self.config.ema_decay)),
                loss_gate_weight=float(cfg.get("loss_gate_weight", self.config.loss_gate_weight)),
                loss_type_weight=float(cfg.get("loss_type_weight", self.config.loss_type_weight)),
                loss_geom_weight=float(cfg.get("loss_geom_weight", self.config.loss_geom_weight)),
                loss_hazard_weight=float(cfg.get("loss_hazard_weight", self.config.loss_hazard_weight)),
                track_inertia_weight=float(cfg.get("track_inertia_weight", self.config.track_inertia_weight)),
                use_uncertainty_head=bool(cfg.get("use_uncertainty_head", self.config.use_uncertainty_head)),
                geom_logvar_min=float(cfg.get("geom_logvar_min", self.config.geom_logvar_min)),
                geom_logvar_max=float(cfg.get("geom_logvar_max", self.config.geom_logvar_max)),
                gate_present_threshold=float(cfg.get("gate_present_threshold", self.config.gate_present_threshold)),
                elapsed_time_scale=float(cfg.get("elapsed_time_scale", self.config.elapsed_time_scale)),
                spawn_geom_weight_scale=float(cfg.get("spawn_geom_weight_scale", self.config.spawn_geom_weight_scale)),
                use_hungarian_matching=bool(cfg.get("use_hungarian_matching", self.config.use_hungarian_matching)),
                absent_gate_decay=float(cfg.get("absent_gate_decay", self.config.absent_gate_decay)),
                loss_despawn_weight=float(cfg.get("loss_despawn_weight", self.config.loss_despawn_weight)),
                loss_collision_weight=float(cfg.get("loss_collision_weight", self.config.loss_collision_weight)),
                loss_contact_drop_weight=float(cfg.get("loss_contact_drop_weight", self.config.loss_contact_drop_weight)),
                event_transition_weight=float(cfg.get("event_transition_weight", self.config.event_transition_weight)),
                min_xy_step_scale=float(cfg.get("min_xy_step_scale", self.config.min_xy_step_scale)),
                max_xy_step_scale=float(cfg.get("max_xy_step_scale", self.config.max_xy_step_scale)),
                min_phi_step_scale=float(cfg.get("min_phi_step_scale", self.config.min_phi_step_scale)),
                max_phi_step_scale=float(cfg.get("max_phi_step_scale", self.config.max_phi_step_scale)),
                min_size_step_scale=float(cfg.get("min_size_step_scale", self.config.min_size_step_scale)),
                max_size_step_scale=float(cfg.get("max_size_step_scale", self.config.max_size_step_scale)),
                min_wt_step_scale=float(cfg.get("min_wt_step_scale", self.config.min_wt_step_scale)),
                max_wt_step_scale=float(cfg.get("max_wt_step_scale", self.config.max_wt_step_scale)),
                event_sample_fraction=float(cfg.get("event_sample_fraction", self.config.event_sample_fraction)),
                event_heavy_every=int(cfg.get("event_heavy_every", self.config.event_heavy_every)),
                event_heavy_fraction=float(cfg.get("event_heavy_fraction", self.config.event_heavy_fraction)),
                event_window_steps=int(cfg.get("event_window_steps", self.config.event_window_steps)),
                focal_gamma=float(cfg.get("focal_gamma", self.config.focal_gamma)),
                focal_alpha=float(cfg.get("focal_alpha", self.config.focal_alpha)),
                contact_condition_weight=float(cfg.get("contact_condition_weight", self.config.contact_condition_weight)),
                loss_spawn_prior_weight=float(cfg.get("loss_spawn_prior_weight", self.config.loss_spawn_prior_weight)),
                spawn_prior_context_relief=float(cfg.get("spawn_prior_context_relief", self.config.spawn_prior_context_relief)),
                pod_spawn_window_steps=int(cfg.get("pod_spawn_window_steps", self.config.pod_spawn_window_steps)),
                spawn_lookahead_steps=int(cfg.get("spawn_lookahead_steps", self.config.spawn_lookahead_steps)),
                loss_spawn_lookahead_weight=float(cfg.get("loss_spawn_lookahead_weight", self.config.loss_spawn_lookahead_weight)),
                match_memory_weight=float(cfg.get("match_memory_weight", self.config.match_memory_weight)),
                match_memory_decay_scale=float(cfg.get("match_memory_decay_scale", self.config.match_memory_decay_scale)),
                match_memory_reappearance_only=bool(cfg.get("match_memory_reappearance_only", self.config.match_memory_reappearance_only)),
                match_learned_embed_weight=float(cfg.get("match_learned_embed_weight", self.config.match_learned_embed_weight)),
                match_learned_embed_decay_scale=float(cfg.get("match_learned_embed_decay_scale", self.config.match_learned_embed_decay_scale)),
                match_learned_embed_reappearance_only=bool(cfg.get("match_learned_embed_reappearance_only", self.config.match_learned_embed_reappearance_only)),
                loss_jump_weight=float(cfg.get("loss_jump_weight", self.config.loss_jump_weight)),
                jump_disp_threshold=float(cfg.get("jump_disp_threshold", self.config.jump_disp_threshold)),
                collision_use_target_wall=bool(cfg.get("collision_use_target_wall", self.config.collision_use_target_wall)),
                train_type_dynamics=bool(cfg.get("train_type_dynamics", self.config.train_type_dynamics)),
                gate_transition_weighted=bool(cfg.get("gate_transition_weighted", self.config.gate_transition_weighted)),
            )
            self.model.absent_gate_decay = float(np.clip(self.config.absent_gate_decay, 0.0, 1.0))
            self.ema_model.absent_gate_decay = float(np.clip(self.config.absent_gate_decay, 0.0, 1.0))
            for m in (self.model, self.ema_model):
                m.min_xy_step_scale = float(self.config.min_xy_step_scale)
                m.max_xy_step_scale = float(self.config.max_xy_step_scale)
                m.min_phi_step_scale = float(self.config.min_phi_step_scale)
                m.max_phi_step_scale = float(self.config.max_phi_step_scale)
                m.min_size_step_scale = float(self.config.min_size_step_scale)
                m.max_size_step_scale = float(self.config.max_size_step_scale)
                m.min_wt_step_scale = float(self.config.min_wt_step_scale)
                m.max_wt_step_scale = float(self.config.max_wt_step_scale)

        try:
            self.model.load_state_dict(payload.get("model", {}), strict=False)
        except Exception:
            pass
        try:
            self.ema_model.load_state_dict(payload.get("ema_model", {}), strict=False)
        except Exception:
            pass
        try:
            self.optimizer.load_state_dict(payload.get("optimizer", {}))
        except Exception:
            pass

        try:
            self.buffer.load_state_dict(payload.get("buffer", {}))
        except Exception:
            pass

        rng_state = payload.get("rng_state", None)
        if rng_state is not None:
            try:
                self.rng.bit_generator.state = rng_state
            except Exception:
                pass

        self.training_enabled = bool(payload.get("training_enabled", self.training_enabled))
        self.total_updates = int(payload.get("total_updates", self.total_updates))
        self.total_seen_transitions = int(payload.get("total_seen_transitions", self.total_seen_transitions))
        self._update_clock = int(payload.get("update_clock", self._update_clock))
        self.last_train_metrics = dict(payload.get("last_train_metrics", self.last_train_metrics))
        if isinstance(payload.get("transition_stats", None), dict):
            ts = payload.get("transition_stats", {})
            for k in self.transition_stats.keys():
                if k in ts:
                    try:
                        self.transition_stats[k] = float(ts[k])
                    except Exception:
                        pass
