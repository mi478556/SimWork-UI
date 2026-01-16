# clip_index.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import json
import os


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
            return {}

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
