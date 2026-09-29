"""API key management routes (docs/api_keys.md).

    POST   /api-keys              create a key (JWT) — the raw key is returned ONCE
    GET    /api-keys              list the caller's keys (JWT)
    GET    /api-keys/current      describe the key making the request (API key)
    GET    /api-keys/{key_id}     one key (JWT)
    DELETE /api-keys/{key_id}     revoke a key (JWT) — the row is kept

Management is for a signed-in seller (portal JWT). ``/current`` is for the
integration holding a key: it is authenticated BY that key, so it can only ever
describe its own caller.

A router of its own so the feature can be removed with one include_router line,
without opening auth.py.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from app.api.deps import DB, ApiKeyAuth, CurrentUser
from app.models.api_key import ApiKey
from app.schemas.api_keys import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyCredits,
    ApiKeyOwner,
    ApiKeyResponse,
    CurrentApiKeyResponse,
)
from app.services.api_key_service import ApiKeyService
from app.utils.envelopes import api_success

router = APIRouter(tags=["api-keys"])


def _parse_key_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid API key id.")


def _key_response(api_key: ApiKey, now: Optional[datetime] = None) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        scopes=list(api_key.scopes or []),
        status=ApiKeyService.status_of(api_key, now),
        created_at=api_key.created_date,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        revoked_at=api_key.revoked_at,
    )


@router.post("/api-keys", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    request: Request,
    current_user: CurrentUser,
    db: DB,
):
    """Create an API key. The full key is in ``data.key`` and is never shown again."""
    created = await ApiKeyService.create(
        db,
        current_user.id,
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
        request=request,
    )
    body = ApiKeyCreatedResponse(
        **_key_response(created.api_key).model_dump(),
        key=created.raw_key,
    )
    return api_success(body.model_dump(mode="json"))


@router.get("/api-keys", response_model=dict)
async def list_api_keys(current_user: CurrentUser, db: DB):
    """Every key the caller has created, newest first. Revoked keys are included."""
    now = datetime.now(timezone.utc)
    keys = await ApiKeyService.list_for_user(db, current_user.id)
    return api_success([_key_response(k, now).model_dump(mode="json") for k in keys])


# Declared before /api-keys/{key_id} so "current" is never parsed as an id.
@router.get("/api-keys/current", response_model=dict)
async def get_current_api_key(principal: ApiKeyAuth, db: DB):
    """Who this key belongs to, and the AI credits available to it.

    For an integration's "Connected as ..." screen. Send the key as
    ``Authorization: Bearer riv_live_...``.
    """
    user = principal.user
    credits = await ApiKeyService.credits_for(db, user.id)
    body = CurrentApiKeyResponse(
        api_key=_key_response(principal.api_key),
        user=ApiKeyOwner(id=user.id, name=user.name, email=user.email, avatar_url=user.avatar_url),
        credits=ApiKeyCredits(**credits),
    )
    return api_success(body.model_dump(mode="json"))


@router.get("/api-keys/{key_id}", response_model=dict)
async def get_api_key(key_id: str, current_user: CurrentUser, db: DB):
    """One of the caller's keys. Another user's key is 404."""
    api_key = await ApiKeyService.get_for_user(db, current_user.id, _parse_key_id(key_id))
    return api_success(_key_response(api_key).model_dump(mode="json"))


@router.delete("/api-keys/{key_id}", response_model=dict)
async def revoke_api_key(key_id: str, request: Request, current_user: CurrentUser, db: DB):
    """Revoke a key. It stops working immediately; the record is kept. Idempotent."""
    api_key = await ApiKeyService.revoke(
        db, current_user.id, _parse_key_id(key_id), request=request
    )
    return api_success(_key_response(api_key).model_dump(mode="json"))
