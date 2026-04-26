# clip_index.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import json
import os
import zipfile
from datetime import datetime

import numpy as np


@dataclass
class ClipRef:
    session_id: str
    clip_id: str

    # Span in absolute timestep coordinates
    start: int
    end: int

    label: Optional[str]
    bookmarked_at: str

    meta: Dict[str, Any]

    # provenance is not serialized — provided by caller
    provenance: Optional[object] = field(default=None, repr=False)

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    @property
    def is_empty(self) -> bool:
        return self.length == 0


class ClipIndex:

    DEFAULT_CLIP_ID = "full_run"

    def __init__(self, root_dir: str):
        # root_dir now points to: dataset/saves/
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    # NEW — clip index lives next to session.npz
    def _clip_path(self, session_id: str) -> str:
        return os.path.join(self.root_dir, session_id, "clips.json")

    def _load_raw(self, session_id: str) -> dict:
        path = self._clip_path(session_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _session_npz_path(self, session_id: str) -> str:
        return os.path.join(self.root_dir, session_id, "session.npz")

    def _session_frame_count(self, session_id: str) -> int:
        """Read frames.npy shape from session.npz without loading frame data."""
        path = self._session_npz_path(session_id)
        if not os.path.exists(path):
            return 0

        try:
            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("frames.npy", "r") as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    elif version in ((2, 0), (3, 0)):
                        shape, _, _ = np.lib.format.read_array_header_2_0(f)
                    else:
                        return 0
            return int(shape[0]) if shape else 0
        except Exception:
            return 0

    def _default_clip(self, session_id: str) -> Dict[str, ClipRef]:
        frame_count = self._session_frame_count(session_id)
        if frame_count <= 0:
            return {}

        return {
            self.DEFAULT_CLIP_ID: ClipRef(
                session_id=session_id,
                clip_id=self.DEFAULT_CLIP_ID,
                start=0,
                end=frame_count,
                label="Full Run",
                bookmarked_at=datetime.fromtimestamp(
                    os.path.getmtime(self._session_npz_path(session_id))
                ).strftime("%Y-%m-%d_%H-%M-%S"),
                meta={"kind": "full_run", "generated": True},
            )
        }

    def _save_raw(self, session_id: str, data: dict):
        path = self._clip_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ----------------------------------------------------
    # Load index
    # ----------------------------------------------------
    def load(self, session_id: str, provenance=None) -> Dict[str, ClipRef]:

        raw = self._load_raw(session_id)

        # Backward compatible: old files store clips as top-level mapping
        if not raw:
            return self._default_clip(session_id)

        clips_data = raw.get("clips") if isinstance(raw, dict) and "clips" in raw else raw

        clips: Dict[str, ClipRef] = {}
        for cid, entry in (clips_data.items() if isinstance(clips_data, dict) else []):
            c = ClipRef(
                session_id=session_id,
                clip_id=cid,
                start=int(entry["start"]),
                end=int(entry["end"]),
                label=entry.get("label"),
                bookmarked_at=entry.get("bookmarked_at", ""),
                meta=entry.get("meta", {}),
            )

            # provenance injected at load time
            c.provenance = provenance

            clips[cid] = c

        if not clips:
            return self._default_clip(session_id)

        return clips

    # ----------------------------------------------------
    # Save index
    # ----------------------------------------------------
    def save(self, session_id: str, clips: Dict[str, ClipRef]):
        serializable = {}
        for cid, c in clips.items():
            serializable[cid] = dict(
                start=int(c.start),
                end=int(c.end),
                label=c.label,
                bookmarked_at=c.bookmarked_at,
                meta=c.meta,
            )

        # Preserve existing raw structure when possible
        raw = self._load_raw(session_id)
        if raw and isinstance(raw, dict) and ("clips" in raw or raw.keys()):
            # Use the new schema
            raw.setdefault("session", {})
            raw["clips"] = serializable
            raw.setdefault("version", 1)
            self._save_raw(session_id, raw)
        else:
            # Backward-compatible: write top-level mapping
            path = self._clip_path(session_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(serializable, f, indent=2)

    # ----------------------------------------------------
    # Session-level metadata helpers
    # ----------------------------------------------------
    def load_session_meta(self, session_id: str) -> dict:
        raw = self._load_raw(session_id)
        return raw.get("session", {}) if isinstance(raw, dict) else {}

    def save_session_meta(self, session_id: str, meta: dict):
        raw = self._load_raw(session_id)
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("session", {}).update(meta)
        raw.setdefault("version", 1)
        self._save_raw(session_id, raw)

    # ----------------------------------------------------
    # Convenience helper
    # ----------------------------------------------------
    def list_clips_for_session(self, session_id: str):
        return list(self.load(session_id).values())
