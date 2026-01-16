# finalize_manager.py

from __future__ import annotations
from collections import deque
from typing import Optional, Dict, Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from dataset.finalize_worker import FinalizeJob


class FinalizeManager(QObject):
    # Signals
    job_started = pyqtSignal(str)              # pending_dir
    job_finished = pyqtSignal(str)             # session_id
    job_failed = pyqtSignal(str, str)          # pending_dir, error
    job_canceled = pyqtSignal(str)             # pending_dir
    pipeline_depth_changed = pyqtSignal(int)
    busy_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        self.active: Optional[FinalizeJob] = None
        self.queue = deque()

        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_active)
        self.timer.start(100)

    # ------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------
    def pipeline_depth(self) -> int:
        return (1 if self.active else 0) + len(self.queue)

    def is_busy(self) -> bool:
        """True if an active finalize process is running (not merely queued)."""
        try:
            return (self.active is not None) and self.active.is_alive()
        except Exception:
            return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active": 1 if self.active else 0,
            "queued": len(self.queue),
            "active_pending_dir": (
                self.active.pending_dir if self.active else None
            ),
            "queued_pending_dirs": [j.pending_dir for j in self.queue],
        }

    # ------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------
    def submit(self, job: FinalizeJob) -> bool:
        # Reject if pipeline already at capacity (active + queued >= 2)
        if self.pipeline_depth() >= 2:
            return False

        if self.active is None:
            self.active = job
            job.start()
            self.job_started.emit(job.pending_dir)
        else:
            self.queue.append(job)
        self.pipeline_depth_changed.emit(self.pipeline_depth())
        self.busy_changed.emit(self.is_busy())
        return True

    # ------------------------------------------------------------
    # Polling active worker
    # ------------------------------------------------------------
    def _poll_active(self):
        if self.active is None:
            return
        # Recovery: if child died unexpectedly, advance and emit failure
        try:
            if not self.active.is_alive():
                try:
                    self.job_failed.emit(self.active.pending_dir, "finalize process exited unexpectedly")
                except Exception:
                    pass
                self._advance()
                return
        except Exception:
            pass

        # Drain any available messages without blocking the Qt event loop
        try:
            while self.active is not None and getattr(self.active, "conn", None) and self.active.conn.poll(0):
                try:
                    msg = self.active.conn.recv()
                except Exception:
                    break

                state = msg.get("state")

                if state == "finished":
                    session_id = msg.get("session_id", "unknown")
                    try:
                        self.job_finished.emit(session_id)
                    except Exception:
                        pass
                    self._advance()
                    break

                elif state == "failed":
                    error = msg.get("error", "unknown error")
                    try:
                        self.job_failed.emit(self.active.pending_dir, error)
                    except Exception:
                        pass
                    self._advance()
                    break

                elif state == "canceled":
                    try:
                        self.job_canceled.emit(self.active.pending_dir)
                    except Exception:
                        pass
                    self._advance()
                    break
                # loop to drain remaining messages
        except Exception:
            return

    # ------------------------------------------------------------
    # Pipeline advancement
    # ------------------------------------------------------------
    def _advance(self):
        # Cleanup IPC for the active job before clearing it
        try:
            self._cleanup_active_ipc()
        except Exception:
            pass

        # Clear active
        self.active = None

        # Promote queued job if present
        if self.queue:
            self.active = self.queue.popleft()
            self.active.start()
            self.job_started.emit(self.active.pending_dir)
        self.pipeline_depth_changed.emit(self.pipeline_depth())
        self.busy_changed.emit(self.is_busy())

    def _cleanup_active_ipc(self):
        try:
            if self.active and getattr(self.active, "conn", None):
                try:
                    self.active.conn.close()
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------
    def kill_active(self):
        if not self.active:
            return

        pending_dir = self.active.pending_dir

        try:
            self.active.terminate()
        except Exception:
            pass

        try:
            self._cleanup_active_ipc()
        except Exception:
            pass

        self.active = None
        try:
            self.job_canceled.emit(pending_dir)
        except Exception:
            pass

        # Promote next job if any
        if self.queue:
            self.active = self.queue.popleft()
            self.active.start()
            try:
                self.job_started.emit(self.active.pending_dir)
            except Exception:
                pass
        self.pipeline_depth_changed.emit(self.pipeline_depth())
        self.busy_changed.emit(self.is_busy())

    def kill_all(self):
        # Kill active job
        if self.active:
            try:
                self.active.terminate()
            except Exception:
                pass
            try:
                self._cleanup_active_ipc()
            except Exception:
                pass
            try:
                self.job_canceled.emit(self.active.pending_dir)
            except Exception:
                pass
            self.active = None

        # Clear queued jobs
        while self.queue:
            job = self.queue.popleft()
            try:
                if getattr(job, "conn", None):
                    try:
                        job.conn.close()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self.job_canceled.emit(job.pending_dir)
            except Exception:
                pass
        self.pipeline_depth_changed.emit(self.pipeline_depth())
        self.busy_changed.emit(self.is_busy())
