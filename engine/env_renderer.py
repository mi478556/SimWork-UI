# env/env_renderer.py

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

WORLD_MIN = -1.0
WORLD_MAX = 1.0

CANONICAL_RENDER_SIZE = 256
DEFAULT_FRAME_SIZE = 84

AGENT_RADIUS = 0.04
POD_BASE_RADIUS = 0.05
POD_FOOD_RADIUS_SCALE = 0.05

WALL_THICKNESS_PX = 2
ROOM_LINE_THICKNESS_PX = 1


def _world_to_pixel(p: np.ndarray, size: int) -> tuple[int, int]:
    """
    Map continuous world coordinates [-1,1] to pixel coordinates.
    Matches pygame continuous_to_screen(), but clamps to valid indices
    so numpy rasterization never OOBs.
    """
    fx = (float(p[0]) - WORLD_MIN) / (WORLD_MAX - WORLD_MIN) * size
    fy = (float(p[1]) - WORLD_MIN) / (WORLD_MAX - WORLD_MIN) * size

    x = int(fx)
    y = int(fy)

    # Clamp to [0, size-1]
    if x < 0:
        x = 0
    elif x >= size:
        x = size - 1

    if y < 0:
        y = 0
    elif y >= size:
        y = size - 1

    return x, y


def _draw_circle(img, center, radius, color):
    h, w, _ = img.shape
    cx, cy = center
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    mask = x * x + y * y <= radius * radius
    img[mask] = color


def _draw_vertical_line(img, x, color, thickness):
    h, w, _ = img.shape
    x0 = max(0, x - thickness // 2)
    x1 = min(w, x + thickness // 2 + 1)
    img[:, x0:x1] = color


def _draw_rect_outline(img, rect, size, color):
    x, y, w, h = rect
    p0 = _world_to_pixel(np.array([x, y]), size)
    p1 = _world_to_pixel(np.array([x + w, y + h]), size)
    x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
    x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])

    img[y0:y1, x0] = color
    img[y0:y1, x1] = color
    img[y0, x0:x1] = color
    img[y1, x0:x1] = color


def _draw_wall_segment(
    img: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    size: int,
    color: tuple[int, int, int],
    thickness: int,
):
    x0, y0 = _world_to_pixel(p0, size)
    x1, y1 = _world_to_pixel(p1, size)

    # vertical segment
    if x0 == x1:
        x = x0
        y0_, y1_ = sorted([y0, y1])
        img[y0_:y1_, x - thickness // 2 : x + thickness // 2 + 1] = color
    else:
        # horizontal segment
        y = y0
        x0_, x1_ = sorted([x0, x1])
        img[y - thickness // 2 : y + thickness // 2 + 1, x0_:x1_] = color


def _downsample(img: np.ndarray, target_size: int) -> np.ndarray:
    scale = img.shape[0] // target_size
    img = img.reshape(
        target_size, scale,
        target_size, scale,
        3
    ).mean(axis=(1, 3))
    return img.astype(np.float32)


def _sget(obj, key, default=None):
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


class EnvRenderer:

    def __init__(
        self,
        frame_size: int = DEFAULT_FRAME_SIZE,
        canonical_size: int = CANONICAL_RENDER_SIZE,
        background_color=(0, 0, 0),
    ):
        self.frame_size = frame_size
        self.canonical_size = canonical_size
        self.background_color = background_color

    def render(self, snapshot: Dict[str, Any], *, overlays=None) -> np.ndarray:
        if overlays is None:
            overlays = {}

        # ---------------- canonical render ----------------
        size = self.canonical_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = self.background_color

        agent_pos = np.asarray(_sget(snapshot, "agent_pos", [0.0, 0.0]))
        ax, ay = _world_to_pixel(agent_pos, size)
        _draw_circle(img, (ax, ay), int(AGENT_RADIUS * size / 2), (0, 255, 0))

        for pod in _sget(snapshot, "pods", []):
            if not _sget(pod, "active", False):
                continue
            pos = np.asarray(_sget(pod, "pos"))
            food = float(_sget(pod, "food", 0.0))
            px, py = _world_to_pixel(pos, size)
            r = POD_BASE_RADIUS + food * POD_FOOD_RADIUS_SCALE
            _draw_circle(img, (px, py), int(r * size / 2), (255, 255, 0))

        wall = _sget(snapshot, "wall", {})
        wall_blocking = bool(_sget(wall, "blocking", False))
        if wall_blocking:
            wx = _world_to_pixel(np.array([0.0, 0.0]), size)[0]
            _draw_vertical_line(img, wx, (255, 255, 255), WALL_THICKNESS_PX)

        # (No downsampling here — renderer is canonical 256×256)

        # ---------------- draw rooms AFTER downsample ----------------
        # --------------------------------------------------------------
        # Comb wall rooms (pygame-equivalent: rooms ARE the wall)
        # --------------------------------------------------------------
        if overlays.get("rooms", True) and wall_blocking:
            rooms = _sget(snapshot, "rooms", None)

            if rooms is not None:
                side = _sget(snapshot, "bucket_side", "left")

                for r in rooms:
                    rx, ry, rw, rh = r

                    # corners in world coords
                    p00 = np.array([rx, ry], dtype=np.float32)
                    p10 = np.array([rx + rw, ry], dtype=np.float32)
                    p01 = np.array([rx, ry + rh], dtype=np.float32)
                    p11 = np.array([rx + rw, ry + rh], dtype=np.float32)

                    # top edge
                    _draw_wall_segment(
                        img, p00, p10, size,
                        color=(255, 255, 255),
                        thickness=WALL_THICKNESS_PX,
                    )

                    # bottom edge
                    _draw_wall_segment(
                        img, p01, p11, size,
                        color=(255, 255, 255),
                        thickness=WALL_THICKNESS_PX,
                    )

                    if side == "left":
                        # draw right edge only (wall face)
                        _draw_wall_segment(
                            img, p10, p11, size,
                            color=(255, 255, 255),
                            thickness=WALL_THICKNESS_PX,
                        )
                    else:
                        # draw left edge only (wall face)
                        _draw_wall_segment(
                            img, p00, p01, size,
                            color=(255, 255, 255),
                            thickness=WALL_THICKNESS_PX,
                        )

        return img.astype(np.float32) / 255.0
