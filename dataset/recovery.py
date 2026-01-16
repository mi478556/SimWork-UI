# dataset/recovery.py
from __future__ import annotations

import os
import json
import shutil
from typing import List, Dict, Any, Tuple


def find_pending_takes(root_dir: str) -> List[Dict[str, Any]]:
    """
    Scan the dataset root for pending (unfinalized) takes.

    Expected layout:
        <root_dir>/pending/<session_id>/<take_id>/pending.json

    Returns a list of descriptors with enough information for the UI
    to decide whether to finalize or delete.
    """
    pending_root = os.path.join(root_dir, "pending")
    results: List[Dict[str, Any]] = []

    if not os.path.isdir(pending_root):
        return results

    for session_id in os.listdir(pending_root):
        session_dir = os.path.join(pending_root, session_id)
        if not os.path.isdir(session_dir):
            continue

        for take_id in os.listdir(session_dir):
            take_dir = os.path.join(session_dir, take_id)
            if not os.path.isdir(take_dir):
                continue

            meta_path = os.path.join(take_dir, "pending.json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

            results.append({
                "session_id": session_id,
                "take_id": take_id,
                "pending_dir": take_dir,
                "meta": meta,
            })

    return results


def load_job_state(pending_dir: str) -> Dict[str, Any]:
    """
    Load job.json if present. This describes the last known
    finalize state (queued / running / failed / canceled).
    """
    job_path = os.path.join(pending_dir, "job.json")
    if not os.path.isfile(job_path):
        return {"state": "unknown"}

    try:
        with open(job_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "corrupt"}


def delete_pending_take(pending_dir: str) -> bool:
    """
    Permanently delete a pending take from disk.
    This is irreversible and should only be called after user confirmation.
    """
    try:
        shutil.rmtree(pending_dir, ignore_errors=False)
        return True
    except Exception:
        return False


def recoverable_takes(root_dir: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Convenience helper that returns (pending_take, job_state) pairs.

    Intended for UI use:
      - show take metadata
      - show last known job state
      - offer finalize or delete
    """
    takes = find_pending_takes(root_dir)
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for t in takes:
        job_state = load_job_state(t["pending_dir"])
        out.append((t, job_state))

    return out


def cleanup_all_pending(root_dir: str) -> int:
    """
    Emergency cleanup utility.
    Deletes ALL pending takes without confirmation.

    Returns number of deleted take directories.
    """
    count = 0
    pending_root = os.path.join(root_dir, "pending")
    if not os.path.isdir(pending_root):
        return 0

    for session_id in os.listdir(pending_root):
        session_dir = os.path.join(pending_root, session_id)
        if not os.path.isdir(session_dir):
            continue

        try:
            shutil.rmtree(session_dir, ignore_errors=False)
            count += 1
        except Exception:
            pass

    return count
