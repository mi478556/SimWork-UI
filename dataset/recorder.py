# recorder.py

from __future__ import annotations

import uuid
import time
import os
import json
import numpy as np
import threading
import queue
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from dataset.session_types import ProvenanceMeta
from dataset.clip_index import ClipRef
from dataset.session_store import SessionStore


# ------------------------------------------------------------
# Pending take descriptor (handoff to finalize worker)
# ------------------------------------------------------------

@dataclass
class PendingTake:
    pending_dir: str
    take_id: str
    session_id: str
    created_at: float
    frame_count: int
    metadata: Dict[str, Any]


# ------------------------------------------------------------
# SessionRecorder — CAPTURE ONLY
# ------------------------------------------------------------

class SessionRecorder:
    """
    Fast, capture-only recorder.

    Guarantees:
    - append() is amortized O(1)
    - stop_capture() is bounded time
    - no heavy I/O or compression during capture stop
    - all data required for finalize is written to disk
    """

    # --------------------------------------------------------
    # Init
    # --------------------------------------------------------

    def __init__(self, store: SessionStore):
        self.store = store

        self.session_id = str(uuid.uuid4())
        self.start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Per-take staging dir (create immediately so writes never need moves)
        self.take_id = str(uuid.uuid4())
        self.take_dir = os.path.join(
            self.store.root_dir,
            "pending",
            self.session_id,
            self.take_id,
        )
        os.makedirs(self.take_dir, exist_ok=True)

        # Provenance (passed through to finalize)
        self.provenance = ProvenanceMeta(
            origin="live",
            branch_parent_id=None,
            branch_depth=0,
            source_session_id=None,
            source_clip_id=None,
            source_step_index=None,
            agent_id=None,
            notes=None,
        )

        # Chunking parameters
        self.chunk_size = 256
        self.chunk_index = 0

        # In-memory buffers (bounded)
        self._buffers: Dict[str, list] = {
            "frames": [],
            "stomach": [],
            "agent_pos": [],
            "actions": [],
            "phases": [],
            "food_positions": [],
            "wall_enabled": [],
            "wall_blocking": [],
            "sequence_index": [],
            "oracle_queries": [],
            "oracle_distances": [],
        }

        # On-disk chunk index
        self._chunk_files: Dict[str, List[str]] = {
            k: [] for k in self._buffers
        }

        # Bookmarks (small, safe to keep in memory)
        self.bookmarks: List[ClipRef] = []

        self.total_deaths = 0
        self.total_food_eaten = 0

        self._finalized = False

        # Total number of appended frames (bounded-time accounting)
        self.frame_count: int = 0
        # Background writer queue and thread (avoid blocking append)
        self._write_q: "queue.Queue[Tuple[int, str, Dict[str, np.ndarray]]]" = queue.Queue(maxsize=8)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        # In-memory completion event set when writer finishes (no filesystem sentinel)
        self._writer_done = threading.Event()
        self._writer_stop = threading.Event()

    def num_frames(self) -> int:
        """Number of frames appended so far (cheap, does not touch disk)."""
        return int(self.frame_count)

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _flush_chunk(self) -> None:
        """
        Flush buffered data to disk as one chunk into the per-take directory.

        This version pre-declares filenames on the sim thread and hands
        Python lists (shallow-copied) to the writer thread so heavy numpy
        conversions happen off the sim thread.
        """
        if not self._buffers["frames"]:
            return

        idx = int(self.chunk_index)
        self.chunk_index += 1

        # --------------------------------------------------
        # 1. Predeclare filenames NOW (sim thread)
        # --------------------------------------------------
        for name in self._buffers.keys():
            filename = f"{name}_{idx:06d}.npy"
            self._chunk_files[name].append(filename)

        # --------------------------------------------------
        # 2. Hand raw Python lists to writer thread (shallow copy)
        # --------------------------------------------------
        payload: Dict[str, list] = {}
        for name, buf in self._buffers.items():
            payload[name] = buf[:]
            buf.clear()

        try:
            self._write_q.put_nowait((idx, self.take_dir, payload))
        except queue.Full:
            raise RuntimeError("Recorder writer backlog full")

        return

    def _writer_loop(self) -> None:
        """Background writer that drains _write_q and performs np.save calls."""
        while True:
            try:
                item = self._write_q.get()
            except Exception:
                continue

            try:
                if item is None:
                    # Writer completion sentinel: set in-memory event only
                    try:
                        self._writer_done.set()
                    except Exception:
                        pass
                    try:
                        self._write_q.task_done()
                    except Exception:
                        pass
                    break

                chunk_index, target_dir, payload = item
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception:
                    pass

                for name, seq in payload.items():
                    try:
                        arr = np.asarray(seq)
                        filename = f"{name}_{chunk_index:06d}.npy"
                        path = os.path.join(target_dir, filename)
                        np.save(path, arr)
                    except Exception as e:
                        print(
                            "[writer] failed saving",
                            name,
                            chunk_index,
                            "err:",
                            repr(e),
                        )
                        raise
                # note: filename bookkeeping is done on sim thread in _flush_chunk
                
            finally:
                try:
                    self._write_q.task_done()
                except Exception:
                    pass

    # --------------------------------------------------------
    # Append step (hot path)
    # --------------------------------------------------------

    def append(
        self,
        *,
        frame,
        stomach,
        agent_pos,
        action,
        phase,
        food_positions,
        wall_enabled,
        wall_blocking,
        sequence_index,
        oracle_query=None,
        oracle_distance=np.nan,
    ):
        if self._finalized:
            raise RuntimeError("append() called after stop_capture()")

        self._buffers["frames"].append(frame)
        self._buffers["stomach"].append(stomach)
        self._buffers["agent_pos"].append(agent_pos)
        self._buffers["actions"].append(action)
        self._buffers["phases"].append(phase)
        self._buffers["food_positions"].append(food_positions)
        self._buffers["wall_enabled"].append(wall_enabled)
        self._buffers["wall_blocking"].append(wall_blocking)
        self._buffers["sequence_index"].append(sequence_index)

        if oracle_query is None:
            self._buffers["oracle_queries"].append(
                np.full((2, 2), np.nan)
            )
        else:
            self._buffers["oracle_queries"].append(oracle_query)

        self._buffers["oracle_distances"].append(oracle_distance)

        # bounded-time accounting
        self.frame_count += 1

        if len(self._buffers["frames"]) >= self.chunk_size:
            self._flush_chunk()

    # --------------------------------------------------------
    # Bookmark (micro-clip marker only)
    # --------------------------------------------------------

    def bookmark(
        self,
        step_index: int,
        label: Optional[str] = None,
        note: Optional[str] = None,
        *,
        kind: str = "bookmark",
    ):
        if self._finalized:
            raise RuntimeError("bookmark() called after stop_capture()")

        clip_id = str(uuid.uuid4())

        start = max(0, step_index - 32)
        end = step_index + 32

        self.bookmarks.append(
            ClipRef(
                session_id=self.session_id,
                clip_id=clip_id,
                start=start,
                end=end,
                label=label,
                bookmarked_at=self.start_time,
                meta=dict(note=note, kind=kind),
            )
        )

    # --------------------------------------------------------
    # Stop capture (FAST, bounded)
    # --------------------------------------------------------

    def stop_capture(self, cancelled: bool = False) -> PendingTake:
        """
        Stop capture immediately.
        Writes pending.json and returns PendingTake.
        No compression. No concatenation. No blocking.
        """
        if self._finalized:
            raise RuntimeError("stop_capture() called twice")

        # Simplified stop: flush remaining buffers (enqueue only), write metadata,
        # and return PendingTake. Do not block on writer thread or move files.
        if self._finalized:
            raise RuntimeError("stop_capture() called twice")

        # Flush remaining buffers into the take_dir (enqueue only)
        try:
            if self._buffers["frames"]:
                self._flush_chunk()
        except Exception:
            pass

        self._finalized = True

        # Signal writer completion (non-blocking)
        try:
            self._write_q.put_nowait(None)
        except Exception:
            try:
                self._write_q.put(None, timeout=0.1)
            except Exception:
                pass

        # Bounded-time frame count (do not touch disk here)
        frame_count = int(self.frame_count)

        pending_meta: Dict[str, Any] = {
            "schema": "pending-v1",
            "take_id": self.take_id,
            "session_id": self.session_id,
            "created_at": time.time(),
            "frame_count": int(frame_count),
            "columns": self._chunk_files,
            "provenance": self.provenance.__dict__,
            "bookmarks": [c.__dict__ for c in self.bookmarks],
            "total_deaths": int(self.total_deaths),
            "total_food_eaten": int(self.total_food_eaten),
            "cancelled": bool(cancelled),
        }

        meta_path = os.path.join(self.take_dir, "pending.json")
        with open(meta_path, "w") as f:
            json.dump(pending_meta, f, indent=2)

        return PendingTake(
            pending_dir=self.take_dir,
            take_id=self.take_id,
            session_id=self.session_id,
            created_at=pending_meta["created_at"],
            frame_count=frame_count,
            metadata=pending_meta,
        )

    def wait_writer_done(self, timeout: float = 5.0) -> bool:
        """Wait for the background writer to finish writing enqueued chunks.

        Returns True if writer finished within `timeout` seconds.
        """
        try:
            return bool(self._writer_done.wait(timeout=timeout))
        except Exception:
            return False
