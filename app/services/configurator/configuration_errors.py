"""Structured errors for configuration dimensions (ADR-017).

Raised as HTTPException like every other configurator error, with a dict
detail, so the body is {"detail": {"code", "message", "fields"}}: a stable code
the editor can switch on, a sentence for the seller, and the request paths at
fault. Q2 (the api_error envelope) stays open; this keeps to the repo-wide
HTTPException convention.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, status


def invalid(
    code: str,
    message: str,
    fields: Optional[list[str]] = None,
    *,
    status_code: int = status.HTTP_400_BAD_REQUEST,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "fields": list(fields or [])},
    )
