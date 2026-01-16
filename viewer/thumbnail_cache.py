# viewer/thumbnail_cache.py

from collections import OrderedDict
from typing import Optional
import os
import numpy as np

from PyQt6.QtGui import QImage
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from dataset.session_store import SessionStore


# Local copy to avoid circular import with session_preview_panel
def frame_to_qimage(frame: np.ndarray) -> Optional[QImage]:
    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    h, w, _ = frame.shape
    return QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _ThumbnailLoadTask(QRunnable):
    def __init__(self, store: SessionStore, session_id: str, emit_loaded):
        super().__init__()
        self.store = store
        self.session_id = session_id
        self.emit_loaded = emit_loaded

    def run(self):
        img = None
        try:
            thumb_path = os.path.join(
                self.store.saves_dir,
                self.session_id,
                "thumbnail.npy",
            )

            if os.path.exists(thumb_path):
                frame = np.load(thumb_path, allow_pickle=False)
                img = frame_to_qimage(frame)

        except Exception:
            img = None

        try:
            self.emit_loaded(self.session_id, img)
        except Exception:
            pass


class ThumbnailCache(QObject):
    thumbnail_ready = pyqtSignal(str)
    _thumb_loaded = pyqtSignal(str, object)

    def __init__(self, store: SessionStore, max_items: int = 32):
        super().__init__()
        self.store = store
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, Optional[QImage]] = OrderedDict()
        self._loading: set[str] = set()
        self._pool = QThreadPool.globalInstance()
        self._thumb_loaded.connect(self._on_thumb_loaded)

    def get(self, session_id: str) -> Optional[QImage]:
        # Fast path: cache hit
        if session_id in self._cache:
            img = self._cache.pop(session_id)
            self._cache[session_id] = img
            return img

        # Already loading → return immediately
        if session_id in self._loading:
            return None

        # Mark as loading
        self._loading.add(session_id)

        # Schedule background load of precomputed thumbnail artifact
        task = _ThumbnailLoadTask(
            self.store,
            session_id,
            self._thumb_loaded.emit,
        )
        self._pool.start(task)

        # Immediate return — no blocking
        return None

    @pyqtSlot(str, object)
    def _on_thumb_loaded(self, session_id: str, img_obj):
        img = img_obj

        try:
            self._loading.discard(session_id)
            self._cache[session_id] = img

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

            self.thumbnail_ready.emit(session_id)
        except Exception:
            pass

    def invalidate(self, session_id: Optional[str] = None):
        if session_id is None:
            self._cache.clear()
        else:
            self._cache.pop(session_id, None)
