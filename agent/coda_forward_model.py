from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6

TYPE_AGENT = 0
TYPE_POD = 1
TYPE_WALL = 2
TYPE_EMPTY = 3

TYPE_NAMES = {
    TYPE_AGENT: "agent",
    TYPE_POD: "pod",
    TYPE_WALL: "wall",
    TYPE_EMPTY: "empty",
}


@dataclass(frozen=True)
class CodaTokenSchema:
    num_agent_slots: int = 1
    num_pod_slots: int = 2
    num_wall_slots: int = 4
    num_slack_slots: int = 3
    num_types: int = 4
    geom_dim: int = 6

    @property
    def max_slots(self) -> int:
        return self.num_agent_slots + self.num_pod_slots + self.num_wall_slots + self.num_slack_slots

    @property
    def token_dim(self) -> int:
        return 1 + self.num_types + self.geom_dim

    @property
    def wall_start(self) -> int:
        return self.num_agent_slots + self.num_pod_slots


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
    ema_decay: float = 0.995
    loss_gate_weight: float = 1.0
    loss_type_weight: float = 0.75
    loss_geom_weight: float = 4.0


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


class CodaGeometryTokenizer:
    def __init__(self, schema: CodaTokenSchema):
        self.schema = schema

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
            else:
                self._last_geometry[idx] = np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.02], dtype=np.float32)

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
        candidates = {"agent": [], "pod": [], "wall": []}

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

        wall_state = observation.get("wall", {}) if isinstance(observation, dict) else {}
        wall_blocking = False
        try:
            if hasattr(wall_state, "get"):
                wall_blocking = bool(wall_state.get("blocking", False))
            else:
                wall_blocking = bool(getattr(wall_state, "blocking", False))
        except Exception:
            wall_blocking = False

        if wall_blocking:
            rooms = observation.get("rooms", []) or []
            side = str(observation.get("bucket_side", "left"))
            candidates["wall"].extend(self._wall_segments_from_rooms(rooms, side))
        else:
            # Keep at least one latent wall primitive around with gate=0 to stabilize slots.
            candidates["wall"] = []

        return candidates

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

        # Central vertical divider at x=0 present whenever blocking wall is visible.
        geoms.append(self._segment_to_geometry(np.array([0.0, -1.0]), np.array([0.0, 1.0]), thickness=t))

        side_norm = "left" if str(side).strip().lower() == "left" else "right"
        # Use unique y-boundaries across rooms so all comb teeth are represented
        # with a compact, non-redundant set of horizontal segments.
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
            # Snap close boundaries together to avoid duplicates from float jitter.
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
            # Longer segments first.
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
            diag["component_counts"] = {"agent": len(geometry_candidates["agent"]), "pod": len(geometry_candidates["pod"]), "wall": 0}
            diag["wall_visible"] = False
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
            # Prefer authoritative wall geometry from state when available.
            if state_candidates["wall"]:
                wall_geoms = state_candidates["wall"]

            geometry_candidates = {
                "agent": self._sort_geometry_candidates(agent_geoms, TYPE_AGENT) or state_candidates["agent"],
                "pod": self._sort_geometry_candidates(pod_geoms, TYPE_POD) or state_candidates["pod"],
                "wall": self._sort_geometry_candidates(wall_geoms, TYPE_WALL),
            }

            diag["source"] = "frame+state_wall" if state_candidates["wall"] else "frame"
            diag["component_counts"] = {
                "agent": len(agent_comps),
                "pod": len(pod_comps),
                "wall": len(wall_comps),
            }
            diag["wall_visible"] = bool(len(wall_comps) > 0 or len(state_candidates["wall"]) > 0)

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
            num_slots=self.schema.num_slack_slots,
            token_type=TYPE_EMPTY,
            candidates=[],
        )

        gates = tokens[:, 0]
        types = _argmax_type(tokens, self.schema)
        diag["active_slots"] = int(np.sum(gates > 0.5))
        diag["active_per_type"] = {
            TYPE_NAMES[t]: int(np.sum((types == t) & (gates > 0.5)))
            for t in (TYPE_AGENT, TYPE_POD, TYPE_WALL, TYPE_EMPTY)
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
            if token_type == TYPE_EMPTY:
                continue

            geom = tokens[i, geom_offset : geom_offset + self.schema.geom_dim]
            x, y, phi, sx, sy, _ = [float(v) for v in geom]
            sx = max(0.005, abs(sx))
            sy = max(0.005, abs(sy))

            dx = xw - x
            dy = yw - y
            c = float(np.cos(phi))
            s = float(np.sin(phi))
            rx = c * dx + s * dy
            ry = -s * dx + c * dy

            if token_type in (TYPE_AGENT, TYPE_POD):
                mask = (rx / sx) ** 2 + (ry / sy) ** 2 <= 1.0
            else:
                mask = (np.abs(rx) <= sx) & (np.abs(ry) <= sy)

            color = self.colors.get(token_type, self.colors[TYPE_EMPTY]) * gate
            img[mask] = np.maximum(img[mask], color)

        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


class _SetBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        denom = gates.sum(dim=1, keepdim=True).clamp_min(EPS)
        pooled = (h * gates).sum(dim=1, keepdim=True) / denom
        pooled = pooled.expand_as(h)
        out = self.mlp(torch.cat([h, pooled], dim=-1))
        return self.norm(h + out)


class _SetDynamicsModel(nn.Module):
    def __init__(
        self,
        schema: CodaTokenSchema,
        action_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.schema = schema
        self.token_dim = schema.token_dim
        self.geom_offset = 1 + schema.num_types

        self.token_embed = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([_SetBlock(hidden_dim) for _ in range(max(1, num_layers))])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

        self.min_scale = 0.01
        self.max_scale = 1.2
        self.min_thickness = 0.005
        self.max_thickness = 0.4

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        gates = tokens[..., :1].clamp(0.0, 1.0)
        h = self.token_embed(tokens)
        h = h + self.action_embed(actions).unsqueeze(1)
        for block in self.blocks:
            h = block(h, gates)
        raw = self.head(h)
        return self._stabilize(raw, tokens)

    def _stabilize(self, raw: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = 4.0 * torch.tanh(raw / 4.0)

        gate_prev = tokens[..., :1].clamp(EPS, 1.0 - EPS)
        gate_logit_prev = torch.log(gate_prev) - torch.log1p(-gate_prev)
        gate = torch.sigmoid(gate_logit_prev + raw[..., :1])

        type_prev = tokens[..., 1 : 1 + self.schema.num_types].clamp(EPS, 1.0)
        type_logits = torch.log(type_prev) + raw[..., 1 : 1 + self.schema.num_types]
        type_logits = type_logits.clamp(-20.0, 20.0)
        type_probs = torch.softmax(type_logits, dim=-1)

        base_geom = tokens[..., self.geom_offset :]
        delta_geom = raw[..., self.geom_offset :]

        x = (base_geom[..., 0:1] + 0.25 * torch.tanh(delta_geom[..., 0:1])).clamp(-1.0, 1.0)
        y = (base_geom[..., 1:2] + 0.25 * torch.tanh(delta_geom[..., 1:2])).clamp(-1.0, 1.0)
        phi = (base_geom[..., 2:3] + 1.0 * torch.tanh(delta_geom[..., 2:3])).clamp(-np.pi, np.pi)
        sx = (base_geom[..., 3:4] + 0.15 * torch.tanh(delta_geom[..., 3:4])).clamp(self.min_scale, self.max_scale)
        sy = (base_geom[..., 4:5] + 0.15 * torch.tanh(delta_geom[..., 4:5])).clamp(self.min_scale, self.max_scale)
        wt = (base_geom[..., 5:6] + 0.1 * torch.tanh(delta_geom[..., 5:6])).clamp(
            self.min_thickness, self.max_thickness
        )
        geom = torch.cat([x, y, phi, sx, sy, wt], dim=-1)
        return torch.cat([gate, type_probs, geom], dim=-1)


class CodaTransitionBuffer:
    def __init__(self, capacity: int):
        self._data: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=int(max(1, capacity)))

    def __len__(self) -> int:
        return len(self._data)

    def add(self, tokens_t: np.ndarray, action: np.ndarray, tokens_tp1: np.ndarray):
        t0 = np.asarray(tokens_t, dtype=np.float32).copy()
        a = np.asarray(action, dtype=np.float32).copy()
        t1 = np.asarray(tokens_tp1, dtype=np.float32).copy()
        if (not np.isfinite(t0).all()) or (not np.isfinite(a).all()) or (not np.isfinite(t1).all()):
            return
        self._data.append((t0, a, t1))

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._data)
        if n <= 0:
            raise RuntimeError("Cannot sample from empty transition buffer.")

        bsz = int(max(1, batch_size))
        replace = n < bsz
        indices = rng.choice(n, size=bsz, replace=replace)

        tokens_t = np.stack([self._data[int(i)][0] for i in indices], axis=0).astype(np.float32, copy=False)
        actions = np.stack([self._data[int(i)][1] for i in indices], axis=0).astype(np.float32, copy=False)
        tokens_tp1 = np.stack([self._data[int(i)][2] for i in indices], axis=0).astype(np.float32, copy=False)
        return tokens_t, actions, tokens_tp1

    def state_dict(self) -> Dict[str, Any]:
        return {"data": list(self._data)}

    def load_state_dict(self, payload: Dict[str, Any]):
        self._data.clear()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for entry in data:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            self.add(entry[0], entry[1], entry[2])


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
        ).to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))

        self.total_updates = 0
        self.total_seen_transitions = 0
        self._update_clock = 0
        self.last_train_metrics: Dict[str, float] = {
            "loss": 0.0,
            "gate_loss": 0.0,
            "type_loss": 0.0,
            "geom_loss": 0.0,
            "buffer_size": 0.0,
            "total_updates": 0.0,
        }

    def tokenize_observation(self, observation: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.tokenizer.tokenize(observation)

    def observe_transition(self, tokens_t: np.ndarray, action: np.ndarray, tokens_tp1: np.ndarray):
        self.buffer.add(tokens_t=tokens_t, action=action, tokens_tp1=tokens_tp1)
        self.total_seen_transitions += 1

    def mark_reset(self):
        # Transition boundaries are handled by callers not appending cross-reset pairs.
        return

    def predict_next_tokens(self, tokens: np.ndarray, action: np.ndarray, *, use_ema: bool = True) -> np.ndarray:
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(self.device).unsqueeze(0)
            a = torch.from_numpy(np.asarray(action, dtype=np.float32).reshape(1, -1)).to(self.device)
            pred = model(t, a)
        return pred.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def maybe_update(self, *, force: bool = False) -> Optional[Dict[str, float]]:
        self._update_clock += 1
        if not self.training_enabled:
            return None
        if len(self.buffer) < int(max(1, self.config.warmup_transitions)):
            return None
        if not force and (self._update_clock % int(max(1, self.config.update_every)) != 0):
            return None

        tokens_t_np, actions_np, tokens_tp1_np = self.buffer.sample(
            self.config.batch_size,
            rng=self.rng,
        )

        tokens_t = torch.from_numpy(tokens_t_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        tokens_tp1 = torch.from_numpy(tokens_tp1_np).to(self.device)

        self.model.train()
        pred_tokens = self.model(tokens_t, actions)
        pred_tokens = torch.nan_to_num(pred_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
        matched_target = self._set_match_targets(pred_tokens, tokens_tp1)

        loss, gate_loss, type_loss, geom_loss = self._dynamics_loss(pred_tokens, matched_target)

        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
        gate_loss = torch.nan_to_num(gate_loss, nan=0.0, posinf=1e6, neginf=0.0)
        type_loss = torch.nan_to_num(type_loss, nan=0.0, posinf=1e6, neginf=0.0)
        geom_loss = torch.nan_to_num(geom_loss, nan=0.0, posinf=1e6, neginf=0.0)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max(1e-6, self.config.grad_clip)))
        self.optimizer.step()
        self._update_ema()

        self.total_updates += 1
        self.last_train_metrics = {
            "loss": float(loss.detach().cpu().item()),
            "gate_loss": float(gate_loss.detach().cpu().item()),
            "type_loss": float(type_loss.detach().cpu().item()),
            "geom_loss": float(geom_loss.detach().cpu().item()),
            "buffer_size": float(len(self.buffer)),
            "total_updates": float(self.total_updates),
        }
        return dict(self.last_train_metrics)

    def _set_match_targets(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_np = predicted.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        bsz, slots, _ = pred_np.shape
        perms = np.zeros((bsz, slots), dtype=np.int64)

        for b in range(bsz):
            cost = self._pairwise_cost_matrix(pred_np[b], tgt_np[b])
            perms[b] = _hungarian_assignment(cost)

        perms_t = torch.from_numpy(perms).to(predicted.device)
        batch_idx = torch.arange(bsz, device=predicted.device).unsqueeze(1)
        return target[batch_idx, perms_t]

    def _pairwise_cost_matrix(self, pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
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
        geom_cost = tgt_gate * np.sum((pred_geom - tgt_geom) ** 2, axis=-1)

        total = (
            float(self.config.loss_gate_weight) * gate_cost
            + float(self.config.loss_type_weight) * type_cost
            + float(self.config.loss_geom_weight) * geom_cost
        )
        total = np.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
        return total.astype(np.float64, copy=False)

    def _dynamics_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tdim = self.schema.num_types
        go = 1 + tdim

        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=-1.0)

        pred_gate = pred[..., 0].clamp(0.0, 1.0)
        tgt_gate = tgt[..., 0].clamp(0.0, 1.0)
        gate_loss = F.mse_loss(pred_gate, tgt_gate)

        pred_type = pred[..., 1:go].clamp(EPS, 1.0)
        tgt_type = tgt[..., 1:go]
        type_loss = -(tgt_type * torch.log(pred_type)).sum(dim=-1).mean()

        pred_geom = pred[..., go:]
        tgt_geom = tgt[..., go:]
        geom_sq = ((pred_geom - tgt_geom) ** 2).sum(dim=-1)
        geom_loss = (geom_sq * tgt_gate).sum() / (tgt_gate.sum() + EPS)

        loss = (
            float(self.config.loss_gate_weight) * gate_loss
            + float(self.config.loss_type_weight) * type_loss
            + float(self.config.loss_geom_weight) * geom_loss
        )
        return loss, gate_loss, type_loss, geom_loss

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
        frame: Any,
        tokens: np.ndarray,
        predicted_next_tokens: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        rgb = _to_rgb_frame(frame)
        if rgb is None:
            h, w = 84, 84
        else:
            h, w = int(rgb.shape[0]), int(rgb.shape[1])

        recon = self.renderer.render(tokens, height=h, width=w)
        out: Dict[str, Any] = {
            "reconstruction": recon,
            "predicted_reconstruction": None,
            "reconstruction_error": float("nan"),
            "predicted_reconstruction_error": float("nan"),
        }

        if predicted_next_tokens is not None:
            out["predicted_reconstruction"] = self.renderer.render(predicted_next_tokens, height=h, width=w)

        if rgb is not None:
            err = np.abs(rgb - recon)
            out["reconstruction_error"] = float(np.mean(err))
            if out["predicted_reconstruction"] is not None:
                pred_err = np.abs(rgb - out["predicted_reconstruction"])
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
                for t_id in (TYPE_AGENT, TYPE_POD, TYPE_WALL, TYPE_EMPTY)
            },
            "geom_ranges": {
                "x": [float(np.min(geom[:, 0])), float(np.max(geom[:, 0]))] if geom.size > 0 else [0.0, 0.0],
                "y": [float(np.min(geom[:, 1])), float(np.max(geom[:, 1]))] if geom.size > 0 else [0.0, 0.0],
                "sx": [float(np.min(geom[:, 3])), float(np.max(geom[:, 3]))] if geom.size > 0 else [0.0, 0.0],
                "sy": [float(np.min(geom[:, 4])), float(np.max(geom[:, 4]))] if geom.size > 0 else [0.0, 0.0],
            },
        }
        return summary

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": {
                "num_agent_slots": self.schema.num_agent_slots,
                "num_pod_slots": self.schema.num_pod_slots,
                "num_wall_slots": self.schema.num_wall_slots,
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
        }

    def load_state_dict(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return

        try:
            self.model.load_state_dict(payload.get("model", {}))
        except Exception:
            pass
        try:
            self.ema_model.load_state_dict(payload.get("ema_model", {}))
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
