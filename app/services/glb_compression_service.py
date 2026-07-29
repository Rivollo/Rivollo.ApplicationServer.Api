"""GLB Draco compression via the glTF-Transform Node.js toolchain.

glTF-Transform has no Python bindings, so the actual compression runs in
scripts/glb_compress/compress.mjs (Node.js, using the real glTF-Transform
API) and this module shells out to it. `compress()` always raises on
failure — it never silently falls back — so callers stay in control of the
fallback-to-original-bytes behavior.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "glb_compress" / "compress.mjs"
_TIMEOUT_SECONDS = 120


class GLBCompressionService:
    def compress(self, glb_bytes: bytes) -> bytes:
        """Draco-compress GLB bytes. Raises RuntimeError on any failure."""
        start = time.monotonic()
        original_size = len(glb_bytes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / "input.glb"
            out_path = Path(tmp_dir) / "output.glb"
            in_path.write_bytes(glb_bytes)

            try:
                result = subprocess.run(
                    ["node", str(_SCRIPT_PATH), str(in_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("node executable not found — is Node.js installed?") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"gltf-transform draco compression timed out after {_TIMEOUT_SECONDS}s") from exc

            if result.returncode != 0:
                raise RuntimeError(
                    f"gltf-transform draco compression failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            if not out_path.exists():
                raise RuntimeError("gltf-transform did not produce an output file")

            compressed_bytes = out_path.read_bytes()

        duration = time.monotonic() - start
        compressed_size = len(compressed_bytes)
        ratio = (1 - (compressed_size / original_size)) if original_size else 0.0

        logger.info(
            "Draco compression: original=%d bytes  compressed=%d bytes  "
            "ratio=%.1f%%  duration=%.2fs",
            original_size, compressed_size, ratio * 100, duration,
        )

        return compressed_bytes


glb_compression_service = GLBCompressionService()
