"""CDN URL rewrite middleware.

Transparently replaces Azure Blob Storage base URLs with the CDN base URL in
every outgoing JSON response. This requires zero changes to any route or schema
and works retroactively for URLs already stored as blob URLs in the database.

URL structure
-------------
Blob:  https://<account>.blob.core.windows.net/<container>/<blob_path>
CDN:   https://<cdn-endpoint>/<container>/<blob_path>

The host is the only thing that differs. The middleware replaces only the
host prefix, so the container and path are preserved exactly.

Configuration (env vars)
------------------------
AZURE_STORAGE_ACCOUNT  — e.g. "rivollodevstg"
CDN_BASE_URL           — e.g. "https://rivollo-dev-cdn-xxx.z01.azurefd.net"
                         (no trailing slash, no container prefix)

The middleware is a complete no-op when either value is missing, so local
and CI environments without a CDN continue to work unchanged.

Implementation note
-------------------
We use a pure ASGI middleware class rather than Starlette's BaseHTTPMiddleware.
BaseHTTPMiddleware has a documented issue where it silently drops BackgroundTasks
attached to responses. The pure ASGI approach avoids this entirely.
"""
from __future__ import annotations

from typing import Callable

from app.core.config import settings


class BlobToCdnMiddleware:
    """Pure ASGI middleware — rewrites Azure Blob URLs to CDN URLs in JSON responses.

    Supports multiple storage accounts: configure AZURE_BLOB_BASE_URL for the
    primary account and AZURE_BLOB_BASE_URLS (comma-separated) for any additional
    accounts.  All matching blob URLs in every JSON response are replaced with the
    single CDN base URL.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

        cdn_base = (settings.CDN_BASE_URL or "").rstrip("/")
        cdn_bytes = cdn_base.encode() if cdn_base else b""

        # Build one (from, to) pair per distinct blob base URL.
        # Pre-encoded once at startup — zero per-request overhead.
        self._replacements: list[tuple[bytes, bytes]] = [
            (blob_base.encode(), cdn_bytes)
            for blob_base in settings.all_blob_base_urls()
            if blob_base and cdn_bytes and blob_base.encode() != cdn_bytes
        ]
        self._active = bool(self._replacements)

    async def __call__(self, scope, receive, send) -> None:
        if not self._active or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Intercept send to inspect and rewrite the response.
        sender = _CdnRewriteSender(send, self._replacements)
        await self.app(scope, receive, sender)


class _CdnRewriteSender:
    """Wraps the ASGI send callable to rewrite blob URLs in JSON response bodies."""

    __slots__ = ("_send", "_replacements", "_is_json", "_headers_message", "_body_parts")

    def __init__(self, send: Callable, replacements: list[tuple[bytes, bytes]]) -> None:
        self._send = send
        self._replacements = replacements
        self._is_json = False
        self._headers_message: dict = {}
        # Collect body chunks — JSON responses are always fully in memory already.
        self._body_parts: list[bytes] = []

    async def __call__(self, message: dict) -> None:
        if message["type"] == "http.response.start":
            content_type = ""
            for k, v in message.get("headers", []):
                if k.lower() == b"content-type":
                    content_type = v.decode("latin-1")
                    break
            self._is_json = "application/json" in content_type
            if not self._is_json:
                # Not JSON — pass through as-is from here on.
                await self._send(message)
                return
            # For JSON: hold the headers until we know the final body length.
            self._headers_message = message

        elif message["type"] == "http.response.body":
            if not self._is_json:
                await self._send(message)
                return

            chunk = message.get("body", b"")
            if chunk:
                self._body_parts.append(chunk)

            more_body = message.get("more_body", False)
            if not more_body:
                # All chunks received — apply all blob→CDN replacements and flush.
                rewritten = b"".join(self._body_parts)
                for from_bytes, to_bytes in self._replacements:
                    rewritten = rewritten.replace(from_bytes, to_bytes)

                # Patch content-length to match the rewritten body size.
                raw_headers: list[tuple[bytes, bytes]] = list(self._headers_message.get("headers", []))
                patched_headers = [
                    (k, str(len(rewritten)).encode() if k.lower() == b"content-length" else v)
                    for k, v in raw_headers
                ]

                await self._send({**self._headers_message, "headers": patched_headers})
                await self._send({**message, "body": rewritten, "more_body": False})

        else:
            # Lifecycle messages (e.g. http.response.trailers) — pass through.
            await self._send(message)
