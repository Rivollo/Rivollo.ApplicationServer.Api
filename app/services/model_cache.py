"""On-disk cache for source 3D models.

Every colourway is baked from the product's original GLB, and those files are
large — a typical Tripo mesh is 40–80 MB, which takes ~35s to pull from the CDN.
Without a cache, saving three colourways for one product downloads the same
60 MB three times and the seller waits minutes for work that takes seconds.

The cache is:

  * content-addressed by source URL, so a re-uploaded model (new blob path)
    naturally misses and re-downloads rather than serving stale geometry
  * size-capped with least-recently-used eviction, so it cannot fill the disk
  * disposable — every entry can be deleted at any moment and the only cost is
    one re-download

Entries live in the system temp directory, so a container restart clears them.
That is intentional: this is a latency optimisation, not durable storage.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = "rivollo-model-cache"
# Roughly 20 models at typical size. Tuned to stay well inside a container's
# ephemeral disk while covering a seller's working session on one product.
DEFAULT_MAX_BYTES = 1_500_000_000


class ModelCache:
    """LRU disk cache for downloaded source models."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._dir = Path(tempfile.gettempdir()) / CACHE_DIR_NAME
        # Guards eviction so two concurrent bakes can't delete each other's
        # freshly written entry mid-write.
        self._lock = threading.Lock()

    # ---------- Paths ----------

    @staticmethod
    def _key(source_url: str) -> str:
        return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:32]

    def _path_for(self, source_url: str) -> Path:
        return self._dir / f"{self._key(source_url)}.glb"

    # ---------- Public API ----------

    def get(self, source_url: str) -> Optional[Path]:
        """Path to the cached model, or None on a miss.

        Touches the file's mtime on a hit so eviction sees it as recently used.
        """
        path = self._path_for(source_url)
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return None
            os.utime(path, None)
            return path
        except OSError:
            return None

    def put(self, source_url: str, content: bytes) -> Optional[Path]:
        """Store a downloaded model. Returns its path, or None if caching failed.

        Written to a temp file and then atomically replaced, so a crash or a
        concurrent reader never observes a half-written model.
        """
        path = self._path_for(source_url)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(self._dir), suffix=".part")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                os.replace(tmp_name, path)
            except BaseException:
                # Never leave a .part file behind on failure.
                with suppress(OSError):
                    os.unlink(tmp_name)
                raise
            self._evict_if_needed()
            return path
        except OSError as exc:
            # A cache failure must never fail a bake — fall back to no caching.
            logger.warning("Could not cache source model: %s", exc)
            return None

    def clear(self) -> None:
        """Drop every entry. Used by tests and by maintenance tasks."""
        with self._lock:
            if not self._dir.is_dir():
                return
            for entry in self._dir.iterdir():
                with suppress(OSError):
                    entry.unlink()

    # ---------- Eviction ----------

    def _evict_if_needed(self) -> None:
        with self._lock:
            if not self._dir.is_dir():
                return
            try:
                entries = [
                    (p, p.stat()) for p in self._dir.glob("*.glb") if p.is_file()
                ]
            except OSError:
                return

            total = sum(stat.st_size for _, stat in entries)
            if total <= self._max_bytes:
                return

            # Oldest access first — the model a seller is actively working on
            # keeps being touched by get(), so it survives.
            entries.sort(key=lambda item: item[1].st_atime)
            for path, stat in entries:
                if total <= self._max_bytes:
                    break
                with suppress(OSError):
                    path.unlink()
                    total -= stat.st_size
                    logger.info("Evicted cached model %s", path.name)


model_cache = ModelCache()
