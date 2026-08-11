"""GLB Draco compression via the glTF-Transform Node.js toolchain.

glTF-Transform has no Python bindings, so the actual compression runs in
scripts/glb_compress/compress.mjs (Node.js, using the real glTF-Transform
API) and this module shells out to it. Both entry points always raise on
failure — they never silently fall back — so callers stay in control of the
fallback-to-original-bytes behavior.

Two entry points:

  compress()          Draco-compressed GLB bytes. The original behaviour,
                      unchanged.

  compress_package()  The same Draco-compressed GLB *plus* a Draco-compressed
                      glTF package (model.gltf + model.bin + texture files).
                      Both come from ONE draco() transform of the ORIGINAL GLB,
                      so the glTF is never produced by decoding the compressed
                      GLB, and the two describe identical geometry.
"""

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "glb_compress" / "compress.mjs"
_TIMEOUT_SECONDS = 120
# Serialising a second container costs extra wall-clock on top of the encode,
# so the package path gets a longer budget than the GLB-only path.
_PACKAGE_TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class CompressedMeshPackage:
    """A Draco-compressed GLB and the equivalent Draco-compressed glTF package."""

    glb: bytes
    # filename -> bytes. Names are assigned by glTF-Transform and referenced
    # from model.gltf by RELATIVE uri, so they must be preserved verbatim when
    # uploaded or the package breaks.
    gltf_files: dict[str, bytes]
    gltf_entry: str

    @property
    def gltf_total_bytes(self) -> int:
        return sum(len(b) for b in self.gltf_files.values())


class GLBCompressionService:
    def compress(self, glb_bytes: bytes) -> bytes:
        """Draco-compress GLB bytes. Raises RuntimeError on any failure."""
        start = time.monotonic()
        original_size = len(glb_bytes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / "input.glb"
            out_path = Path(tmp_dir) / "output.glb"
            in_path.write_bytes(glb_bytes)

            self._run_node([str(in_path), str(out_path)], _TIMEOUT_SECONDS)

            if not out_path.exists():
                raise RuntimeError("gltf-transform did not produce an output file")

            compressed_bytes = out_path.read_bytes()

        self._log_ratio("Draco compression", original_size, len(compressed_bytes), start)
        return compressed_bytes

    def compress_package(self, glb_bytes: bytes) -> CompressedMeshPackage:
        """Draco-compress into BOTH a GLB and a glTF package.

        The glTF is serialised from the same in-memory document as the GLB,
        after a single draco() transform of the original bytes. It is never
        derived by converting the compressed GLB — that would decode
        KHR_draco_mesh_compression and produce uncompressed geometry.

        Raises RuntimeError on any failure.
        """
        start = time.monotonic()
        original_size = len(glb_bytes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path = Path(tmp_dir) / "input.glb"
            out_path = Path(tmp_dir) / "output.glb"
            gltf_dir = Path(tmp_dir) / "gltf"
            in_path.write_bytes(glb_bytes)

            result = self._run_node(
                [str(in_path), str(out_path), str(gltf_dir)], _PACKAGE_TIMEOUT_SECONDS
            )

            if not out_path.exists():
                raise RuntimeError("gltf-transform did not produce a GLB output file")
            if not gltf_dir.is_dir():
                raise RuntimeError("gltf-transform did not produce a glTF package directory")

            compressed_bytes = out_path.read_bytes()

            # Read back whatever the writer actually emitted rather than
            # assuming filenames — glTF-Transform names texture files itself
            # (baseColor_1.png, ...) and model.gltf references those names.
            gltf_files = {
                child.name: child.read_bytes()
                for child in sorted(gltf_dir.iterdir())
                if child.is_file()
            }

        entry = self._parse_entry_name(result.stdout)
        if entry not in gltf_files:
            raise RuntimeError(
                f"glTF package is missing its entry file {entry!r} "
                f"(found: {sorted(gltf_files)})"
            )

        package = CompressedMeshPackage(
            glb=compressed_bytes, gltf_files=gltf_files, gltf_entry=entry
        )

        self._log_ratio("Draco compression (glb)", original_size, len(compressed_bytes), start)
        logger.info(
            "Draco glTF package: entry=%s  files=%d (%s)  total=%d bytes",
            entry, len(gltf_files), ", ".join(sorted(gltf_files)), package.gltf_total_bytes,
        )
        return package

    # ---------- internals ----------

    def _run_node(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                ["node", str(_SCRIPT_PATH), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("node executable not found — is Node.js installed?") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"gltf-transform draco compression timed out after {timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"gltf-transform draco compression failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result

    @staticmethod
    def _parse_entry_name(stdout: str) -> str:
        """Entry filename from the script's JSON summary line."""
        for line in reversed((stdout or "").strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            entry = payload.get("gltf_entry")
            if entry:
                return str(entry)
        raise RuntimeError("gltf-transform did not report a glTF entry filename")

    @staticmethod
    def _log_ratio(label: str, original: int, compressed: int, start: float) -> None:
        ratio = (1 - (compressed / original)) if original else 0.0
        logger.info(
            "%s: original=%d bytes  compressed=%d bytes  ratio=%.1f%%  duration=%.2fs",
            label, original, compressed, ratio * 100, time.monotonic() - start,
        )


glb_compression_service = GLBCompressionService()
