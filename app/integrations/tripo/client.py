"""Async HTTP client for Tripo's own API (openapi.tripo3d.ai/v3).

This is NOT the Tripo model we call through fal. fal wraps a single
image-to-3D task and exposes none of Tripo's part segmentation; only the direct
API has ``generate_parts``. The two integrations coexist deliberately — see
``app/integrations/fal`` for the other one.

Protocol, which differs from fal's in every respect:

    1. POST <endpoint>  with  Authorization: Bearer <key>
       -> {"code": 0, "data": {"task_id": "..."}}
    2. GET /tasks/{task_id} until status is terminal  (plural — /task/ 404s)
       -> {"code": 0, "data": {"status", "progress", "output", ...}}
    3. Download the GLB from output.model_url

The response envelope always carries a ``code``; **0 means success** and any
other value is an error even when the HTTP status is 200. Checking only the
HTTP status would silently treat failures as successes.

``progress`` (0-100) is the reason this client exists in this shape: unlike
fal's opaque IN_QUEUE/IN_PROGRESS, it lets callers report real progress and
derive a self-correcting ETA.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Submit / status calls are small; generation happens server-side.
_REQUEST_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)
# Model downloads can be large.
_DOWNLOAD_TIMEOUT = httpx.Timeout(timeout=600.0, connect=10.0)

_POLL_INITIAL_BACKOFF = 2.0
_POLL_MAX_BACKOFF = 10.0

# Tripo's terminal task states.
_SUCCESS_STATES = {"success"}
_FAILURE_STATES = {"failed", "cancelled", "banned", "expired", "unknown"}

# Progress callback: (progress_0_to_100, status) -> awaitable
ProgressCallback = Callable[[int, str], Awaitable[None]]


class TripoError(RuntimeError):
    """A Tripo API call failed. Carries the task id when one exists."""

    def __init__(self, message: str, task_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.task_id = task_id


@dataclass
class TripoTaskResult:
    """A completed Tripo task."""

    task_id: str
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    credits_consumed: Optional[int] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def model_url(self) -> Optional[str]:
        """Download URL of the generated model, if the task produced one."""
        url = self.output.get("model_url")
        return url if isinstance(url, str) and url else None

    @property
    def rendered_image_url(self) -> Optional[str]:
        url = self.output.get("rendered_image_url")
        return url if isinstance(url, str) and url else None


class TripoClient:
    """Client for Tripo's direct API."""

    @staticmethod
    def _headers() -> dict[str, str]:
        key = (settings.TRIPO_API_KEY or "").strip()
        if not key:
            raise TripoError(
                "TRIPO_API_KEY is not configured. Set it in the environment "
                "(Container App secret) to call Tripo's direct API."
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _base_url() -> str:
        return (settings.TRIPO_API_BASE_URL or "https://openapi.tripo3d.ai/v3").rstrip("/")

    @staticmethod
    def _unwrap(payload: dict, context: str, task_id: Optional[str] = None) -> dict:
        """Validate Tripo's {code, data} envelope and return ``data``.

        ``code == 0`` is success. Any other code is a failure even on HTTP 200,
        so this must be checked on every response.
        """
        if not isinstance(payload, dict):
            raise TripoError(f"{context}: response was not a JSON object", task_id)

        code = payload.get("code")
        if code != 0:
            message = payload.get("message") or payload.get("suggestion") or "no message"
            raise TripoError(f"{context}: Tripo returned code {code} — {message}", task_id)

        data = payload.get("data")
        if not isinstance(data, dict):
            raise TripoError(f"{context}: response had no data object", task_id)
        return data

    # ---------- Task lifecycle ----------

    @staticmethod
    async def submit(path: str, body: dict) -> str:
        """Create a task. Returns its ``task_id``.

        ``path`` is relative to the API base, e.g. "/generation/image-to-model".
        """
        url = f"{TripoClient._base_url()}{path}"
        headers = TripoClient._headers()

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=body, headers=headers)
        except httpx.RequestError as exc:
            raise TripoError(f"Tripo submit request failed ({path}): {exc}") from exc

        if response.status_code != 200:
            raise TripoError(
                f"Tripo submit {path} returned HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        data = TripoClient._unwrap(response.json(), f"submit {path}")
        task_id = data.get("task_id")
        if not task_id:
            raise TripoError(f"Tripo submit {path} returned no task_id")

        logger.info("Tripo task submitted  path=%s  task_id=%s", path, task_id)
        return str(task_id)

    @staticmethod
    async def wait(
        task_id: str,
        *,
        max_wait_seconds: Optional[float] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TripoTaskResult:
        """Poll a task until it succeeds, fails, or the deadline passes.

        ``on_progress`` is invoked only when the percentage actually changes, so
        callers can broadcast to a WebSocket without flooding it with duplicates.
        """
        base = TripoClient._base_url()
        headers = TripoClient._headers()
        deadline_budget = max_wait_seconds or settings.TRIPO_MAX_WAIT_SECONDS

        loop = asyncio.get_event_loop()
        deadline = loop.time() + deadline_budget
        backoff = _POLL_INITIAL_BACKOFF
        last_progress = -1

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            while True:
                try:
                    response = await client.get(
                        f"{base}/tasks/{task_id}", headers=headers
                    )
                except httpx.RequestError as exc:
                    # Transient network trouble should retry, not abort a task
                    # that may well be progressing fine server-side.
                    logger.warning(
                        "Tripo poll request failed (will retry)  task_id=%s  %s",
                        task_id, exc,
                    )
                    response = None

                if response is not None and response.status_code == 200:
                    data = TripoClient._unwrap(
                        response.json(), "task status", task_id
                    )
                    status = str(data.get("status") or "").lower()
                    progress = int(data.get("progress") or 0)

                    if on_progress and progress != last_progress:
                        last_progress = progress
                        try:
                            await on_progress(progress, status)
                        except Exception:  # noqa: BLE001 - reporting must not kill the task
                            logger.warning(
                                "Tripo progress callback raised (ignored)  task_id=%s",
                                task_id, exc_info=True,
                            )

                    if status in _SUCCESS_STATES:
                        logger.info(
                            "Tripo task succeeded  task_id=%s  credits=%s",
                            task_id, data.get("credits_consumed"),
                        )
                        return TripoTaskResult(
                            task_id=task_id,
                            status=status,
                            output=data.get("output") or {},
                            credits_consumed=data.get("credits_consumed"),
                            created_at=data.get("created_at"),
                            completed_at=data.get("completed_at"),
                        )

                    if status in _FAILURE_STATES:
                        raise TripoError(
                            f"Tripo task {status}: "
                            f"{data.get('message') or 'no detail provided'}",
                            task_id,
                        )
                elif response is not None:
                    # 404 means the URL is wrong or the task does not exist —
                    # retrying cannot fix either, and doing so silently burns
                    # the entire wait budget on a task that may have succeeded.
                    # Fail immediately and keep the task id so the caller can
                    # still resume or inspect it.
                    if response.status_code == 404:
                        raise TripoError(
                            f"Tripo task status endpoint returned 404 "
                            f"({base}/tasks/{task_id}). The task may still be "
                            f"running server-side.",
                            task_id,
                        )
                    logger.warning(
                        "Tripo poll returned HTTP %s  task_id=%s",
                        response.status_code, task_id,
                    )

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TripoError(
                        f"Tripo task did not finish within {deadline_budget:.0f}s",
                        task_id,
                    )

                await asyncio.sleep(min(backoff, remaining))
                backoff = min(backoff * 1.5, _POLL_MAX_BACKOFF)

    @staticmethod
    async def download(url: str) -> tuple[bytes, str]:
        """Fetch a generated model. Returns (bytes, content_type)."""
        try:
            async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as client:
                response = await client.get(url)
        except httpx.RequestError as exc:
            raise TripoError(f"Tripo model download failed: {exc}") from exc

        if response.status_code != 200 or not response.content:
            raise TripoError(
                f"Tripo model download returned HTTP {response.status_code}"
            )

        content_type = response.headers.get("content-type") or "model/gltf-binary"
        logger.info("Tripo model downloaded  bytes=%d  url=%s", len(response.content), url)
        return response.content, content_type


tripo_client = TripoClient()
