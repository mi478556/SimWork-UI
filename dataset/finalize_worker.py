# dataset/finalize_worker.py

from __future__ import annotations

import os
import json
import time
import shutil
import traceback
from multiprocessing import Process, Pipe
from typing import Dict, Any

import numpy as np

from dataset.session_store import SessionStore
from dataset.session_types import SessionData, SessionMeta, ProvenanceMeta
from dataset.clip_index import ClipRef


def _wait_for_files(pending_dir, columns, timeout=5.0):
    deadline = time.time() + timeout
    expected = []

    for files in columns.values():
        expected.extend(files)

    while time.time() < deadline:
        missing = [
            fn for fn in expected
            if not os.path.exists(os.path.join(pending_dir, fn))
        ]
        if not missing:
            return True
        time.sleep(0.05)

    return False


# ------------------------------------------------------------
# Subprocess entrypoint
# ------------------------------------------------------------

def finalize_entrypoint(pending_dir: str, output_root: str, status_conn):
    """
    Finalize a captured take.

    Runs in a subprocess.
    Must NEVER touch Qt, EventBus, or any UI objects.
    Must be kill-safe and recovery-safe.
    """

    job_path = os.path.join(pending_dir, "job.json")

    def _write_job(state: str, **extra):
        payload = {
            "state": state,
            "pid": os.getpid(),
            "timestamp": time.time(),
        }
        payload.update(extra)
        try:
            tmp_path = job_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            try:
                os.replace(tmp_path, job_path)
            except Exception:
                # best-effort fallback
                try:
                    os.remove(job_path)
                except Exception:
                    pass
                try:
                    os.replace(tmp_path, job_path)
                except Exception:
                    pass
        except Exception:
            pass

    try:
        try:
            status_conn.send({"state": "running", "pending_dir": pending_dir})
        except Exception:
            pass
        _write_job("running")

        # --------------------------------------------------
        # Load pending metadata
        # --------------------------------------------------
        meta_path = os.path.join(pending_dir, "pending.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # --------------------------------------------------
        # Handle cancelled takes explicitly
        # --------------------------------------------------
        if meta.get("cancelled", False):
            _write_job("canceled")
            try:
                status_conn.send({
                    "state": "canceled",
                    "pending_dir": pending_dir,
                })
            except Exception:
                pass
            try:
                status_conn.close()
            except Exception:
                pass
            return

        session_id = meta["session_id"]
        columns = meta["columns"]

        # --------------------------------------------------
        # Wait for writer to create the expected chunk files.
        # This ensures we never read missing or half-written .npy files.
        # --------------------------------------------------
        if not _wait_for_files(pending_dir, columns, timeout=5.0):
            raise RuntimeError(
                "Timed out waiting for chunk files to appear"
            )

        # --------------------------------------------------
        # Load and concatenate chunked columns
        # --------------------------------------------------
        arrays: Dict[str, Any] = {}

        for name, files in columns.items():
            parts = []
            for fn in files:
                path = os.path.join(pending_dir, fn)
                parts.append(np.load(path))
            arrays[name] = np.concatenate(parts, axis=0)

        # --------------------------------------------------
        # Guard against empty finalization
        # --------------------------------------------------
        if arrays["frames"].shape[0] == 0:
            raise RuntimeError("Refusing to finalize empty take")

        # --------------------------------------------------
        # Reconstruct provenance
        # --------------------------------------------------
        prov = ProvenanceMeta(**meta["provenance"])

        created_at_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S",
            time.localtime(meta["created_at"]),
        )

        session_meta = SessionMeta(
            session_id=session_id,
            created_at=created_at_str,
            snapshot_schema_version="snapshot-v1",
            fps=60,
            dt=1.0 / 60.0,
            env_version="env-v1",
            pod_count=int(arrays["food_positions"].shape[1]),
            wall_mode="comb",
            total_deaths=int(meta.get("total_deaths", 0)),
            total_food_eaten=int(meta.get("total_food_eaten", 0)),
            provenance=prov,
        )

        session = SessionData(
            session_id=session_id,
            meta=session_meta,

            frames=arrays["frames"],
            stomach=arrays["stomach"],
            agent_pos=arrays["agent_pos"],
            actions=arrays["actions"],

            phases=arrays["phases"],
            food_positions=arrays["food_positions"],

            wall_enabled=arrays["wall_enabled"],
            wall_blocking=arrays["wall_blocking"],

            sequence_index=arrays["sequence_index"],

            oracle_queries=arrays["oracle_queries"],
            oracle_distances=arrays["oracle_distances"],
        )

        # --------------------------------------------------
        # Atomic save into dataset/saves/<session_id>/
        # NOTE: output_root MUST be the dataset root
        # --------------------------------------------------
        store = SessionStore(output_root)
        store.save_session(session)

        # Save a small thumbnail artifact to make preview loads cheap
        try:
            session_dir = os.path.join(output_root, "saves", session_id)
            os.makedirs(session_dir, exist_ok=True)
            thumb_path = os.path.join(session_dir, "thumbnail.npy")
            try:
                thumb_step = 0
                frame = arrays["frames"][thumb_step]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                np.save(thumb_path, frame)
            except Exception:
                pass
        except Exception:
            pass

        # --------------------------------------------------
        # Save clip bookmarks
        # --------------------------------------------------
        clips = store.clips.load(session_id, provenance=prov)
        for c in meta.get("bookmarks", []):
            clips[c["clip_id"]] = ClipRef(**c)
        store.clips.save(session_id, clips)

        shutil.rmtree(pending_dir, ignore_errors=True)

        # Cleanup empty session dir if no more takes exist
        session_dir = os.path.dirname(pending_dir)
        try:
            if os.path.isdir(session_dir) and not os.listdir(session_dir):
                os.rmdir(session_dir)
        except Exception:
            pass

        _write_job("finished")

        try:
            status_conn.send({
                "state": "finished",
                "session_id": session_id,
            })
        except Exception:
            pass
        try:
            status_conn.close()
        except Exception:
            pass

    except Exception as e:
        _write_job(
            "failed",
            error=repr(e),
            traceback=traceback.format_exc(),
        )

        try:
            status_conn.send({
                "state": "failed",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        try:
            status_conn.close()
        except Exception:
            pass



class FinalizeJob:
    def __init__(self, pending_dir: str, output_dir: str):
        """
        output_dir MUST be the dataset root directory,
        not dataset/saves.
        """
        self.pending_dir = pending_dir
        self.output_dir = output_dir
        self.take_id = os.path.basename(pending_dir)

        parent_conn, child_conn = Pipe(duplex=False)
        self.conn = parent_conn
        # keep child conn ref so parent can close it after start
        self._child_conn = child_conn
        self.process = Process(
            target=finalize_entrypoint,
            args=(pending_dir, output_dir, child_conn),
            daemon=True,
        )

    def start(self):
        self.process.start()
        # Parent no longer needs the child end; close it to allow EOF semantics
        try:
            if getattr(self, "_child_conn", None):
                try:
                    self._child_conn.close()
                except Exception:
                    pass
                self._child_conn = None
        except Exception:
            pass

    def is_alive(self) -> bool:
        try:
            return bool(self.process.is_alive())
        except Exception:
            return False

    def terminate(self):
        try:
            if self.process.is_alive():
                try:
                    self.process.terminate()
                except Exception:
                    pass
                self.process.join(timeout=1.0)
                try:
                    if self.process.is_alive():
                        try:
                            self.process.kill()
                        except Exception:
                            pass
                        self.process.join(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass
        # Close IPC endpoints
        try:
            if getattr(self, "conn", None):
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
        except Exception:
            pass
        try:
            if getattr(self, "_child_conn", None):
                try:
                    self._child_conn.close()
                except Exception:
                    pass
                self._child_conn = None
        except Exception:
            pass
