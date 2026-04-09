"""AI suggestion routes — GPT-4o Vision powered auto-suggestions."""

import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator, model_validator

from app.api.deps import CurrentUser, DB, get_current_user
from app.core.config import settings
from app.services.ai_suggestion_service import (
    ai_suggestion_service,
    _USER_PROMPT_MAX_CHARS,
    _USER_INPUT_NAME_MAX_CHARS,
    _USER_INPUT_DESC_MAX_CHARS,
)
from app.utils.envelopes import api_success

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    dependencies=[Depends(get_current_user)],
)


# ---------------------------------------------------------------------------
# Per-user rate limiter
#
# Current implementation: in-process memory (defaultdict keyed by
# (user_id, "YYYY-MM-DDTHH:MM")).  Stale minute-buckets are evicted on
# every check to prevent unbounded memory growth.
#
# KNOWN LIMITATION: this counter is per-process.  When the app runs on
# multiple replicas (e.g. Azure Container Apps scaled-out), each replica
# has its own counter, so a user can effectively make
#   OPENAI_RATE_LIMIT_PER_MINUTE × replica_count
# calls per minute.
#
# To enforce the limit across replicas, replace _RateLimiter.check() with
# a Redis INCR/EXPIRE implementation — the call site (_rate_limiter.check)
# is unchanged.
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self) -> None:
        self._counter: dict[tuple[str, str], int] = defaultdict(int)

    def check(self, user_id: str, count: int = 1) -> None:
        """
        Raises HTTP 429 if the user has exceeded OPENAI_RATE_LIMIT_PER_MINUTE
        AI calls within the current UTC minute.  Pass count > 1 for batch calls.
        """
        limit = settings.OPENAI_RATE_LIMIT_PER_MINUTE
        current_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

        stale_keys = [k for k in self._counter if k[1] != current_bucket]
        for k in stale_keys:
            del self._counter[k]

        key = (user_id, current_bucket)
        self._counter[key] += count

        if self._counter[key] > limit:
            logger.warning(
                "AI rate limit exceeded for user %s (count=%d, limit=%d)",
                user_id, self._counter[key], limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"AI suggestion limit reached. "
                    f"You can make {limit} requests per minute. "
                    "Please wait and try again."
                ),
                headers={"Retry-After": "60"},
            )


_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ProductSuggestRequest(BaseModel):
    productId: str
    imageUrl: str        # product IMAGE asset CDN URL
    glbUrl: Optional[str] = None  # 3D GLB model CDN URL
    mode: Literal["ai_suggest", "user_prompt", "user_input"]
    userPrompt: Optional[str] = None       # required when mode == "user_prompt"
    userName: Optional[str] = None         # required when mode == "user_input"
    userDescription: Optional[str] = None  # required when mode == "user_input"

    @field_validator("imageUrl")
    @classmethod
    def image_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("imageUrl cannot be empty")
        return v.strip()

    @field_validator("glbUrl")
    @classmethod
    def glb_url_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            return v if v else None
        return v

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "ProductSuggestRequest":
        if self.mode == "user_prompt":
            if not self.userPrompt or not self.userPrompt.strip():
                raise ValueError("userPrompt is required when mode is 'user_prompt'")
            if len(self.userPrompt.strip()) > _USER_PROMPT_MAX_CHARS:
                raise ValueError(
                    f"userPrompt exceeds maximum length of {_USER_PROMPT_MAX_CHARS} characters"
                )
            self.userPrompt = self.userPrompt.strip()
        if self.mode == "user_input":
            if not self.userName or not self.userName.strip():
                raise ValueError("userName is required when mode is 'user_input'")
            if len(self.userName.strip()) > _USER_INPUT_NAME_MAX_CHARS:
                raise ValueError(f"userName exceeds maximum length of {_USER_INPUT_NAME_MAX_CHARS} characters")
            if not self.userDescription or not self.userDescription.strip():
                raise ValueError("userDescription is required when mode is 'user_input'")
            if len(self.userDescription.strip()) > _USER_INPUT_DESC_MAX_CHARS:
                raise ValueError(f"userDescription exceeds maximum length of {_USER_INPUT_DESC_MAX_CHARS} characters")
        return self


class HotspotSuggestRequest(BaseModel):
    productId: str
    imageUrl: str
    mode: Literal["ai_suggest", "user_prompt", "user_input"]
    userPrompt: Optional[str] = None        # required when mode == "user_prompt"
    userTitle: Optional[str] = None         # required when mode == "user_input"
    userDescription: Optional[str] = None  # required when mode == "user_input"

    @field_validator("imageUrl")
    @classmethod
    def image_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("imageUrl cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "HotspotSuggestRequest":
        if self.mode == "user_prompt":
            if not self.userPrompt or not self.userPrompt.strip():
                raise ValueError("userPrompt is required when mode is 'user_prompt'")
            if len(self.userPrompt.strip()) > _USER_PROMPT_MAX_CHARS:
                raise ValueError(
                    f"userPrompt exceeds maximum length of {_USER_PROMPT_MAX_CHARS} characters"
                )
            self.userPrompt = self.userPrompt.strip()
        if self.mode == "user_input":
            if not self.userTitle or not self.userTitle.strip():
                raise ValueError("userTitle is required when mode is 'user_input'")
            if len(self.userTitle.strip()) > _USER_INPUT_NAME_MAX_CHARS:
                raise ValueError(f"userTitle exceeds maximum length of {_USER_INPUT_NAME_MAX_CHARS} characters")
            if not self.userDescription or not self.userDescription.strip():
                raise ValueError("userDescription is required when mode is 'user_input'")
            if len(self.userDescription.strip()) > _USER_INPUT_DESC_MAX_CHARS:
                raise ValueError(f"userDescription exceeds maximum length of {_USER_INPUT_DESC_MAX_CHARS} characters")
        return self


class LinkSuggestRequest(BaseModel):
    productId: str
    linkUrl: str                            # the actual URL being annotated
    imageUrl: Optional[str] = None         # optional visual context (product image)
    mode: Literal["ai_suggest", "user_prompt", "user_input"]
    userPrompt: Optional[str] = None       # required when mode == "user_prompt"
    userName: Optional[str] = None         # required when mode == "user_input"
    userDescription: Optional[str] = None  # required when mode == "user_input"

    @field_validator("linkUrl")
    @classmethod
    def link_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("linkUrl cannot be empty")
        return v.strip()

    @field_validator("imageUrl")
    @classmethod
    def image_url_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            return v if v else None
        return v

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "LinkSuggestRequest":
        if self.mode == "user_prompt":
            if not self.userPrompt or not self.userPrompt.strip():
                raise ValueError("userPrompt is required when mode is 'user_prompt'")
            if len(self.userPrompt.strip()) > _USER_PROMPT_MAX_CHARS:
                raise ValueError(
                    f"userPrompt exceeds maximum length of {_USER_PROMPT_MAX_CHARS} characters"
                )
            self.userPrompt = self.userPrompt.strip()
        if self.mode == "user_input":
            if not self.userName or not self.userName.strip():
                raise ValueError("userName is required when mode is 'user_input'")
            if len(self.userName.strip()) > _USER_INPUT_NAME_MAX_CHARS:
                raise ValueError(f"userName exceeds maximum length of {_USER_INPUT_NAME_MAX_CHARS} characters")
            if not self.userDescription or not self.userDescription.strip():
                raise ValueError("userDescription is required when mode is 'user_input'")
            if len(self.userDescription.strip()) > _USER_INPUT_DESC_MAX_CHARS:
                raise ValueError(f"userDescription exceeds maximum length of {_USER_INPUT_DESC_MAX_CHARS} characters")
        return self


# ---------------------------------------------------------------------------
# Feature 1 — Product name + description
# No DB connection needed — only calls OpenAI
# ---------------------------------------------------------------------------

@router.post("/product-suggest", response_model=dict)
async def product_suggest(
    payload: ProductSuggestRequest,
    current_user: CurrentUser,
):
    """
    Suggest a product name and description from image + optional GLB 3D model.

    Three modes:
      - ai_suggest   : AI auto-suggests based on image (+ GLB if provided).
      - user_prompt  : User provides a hint; AI refines the suggestion.
      - user_input   : User supplies name + description directly; no AI call.
    """
    try:
        uuid.UUID(payload.productId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid productId format")

    # Option 3: user provides name + description directly — no AI call needed
    if payload.mode == "user_input":
        return api_success({
            "name": payload.userName.strip(),
            "description": payload.userDescription.strip(),
        })

    _rate_limiter.check(str(current_user.id), count=2)

    results = await ai_suggestion_service.suggest_product(
        image_url=payload.imageUrl,
        glb_url=payload.glbUrl,
        mode=payload.mode,
        user_prompt=payload.userPrompt,
    )
    return api_success(results)


# ---------------------------------------------------------------------------
# Feature 2 — Hotspot title + description
# DB needed to fetch product name + existing hotspot titles
# ---------------------------------------------------------------------------

@router.post("/hotspot-suggest", response_model=dict)
async def hotspot_suggest(
    payload: HotspotSuggestRequest,
    current_user: CurrentUser,
    db: DB,
):
    """
    Suggest a hotspot title and description from a product image.

    Three modes:
      - ai_suggest   : AI auto-suggests based on image + product context.
      - user_prompt  : User provides a hint about the hotspot; AI refines.
      - user_input   : User supplies title + description directly; no AI call.
    """
    try:
        product_uuid = uuid.UUID(payload.productId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid productId format")

    # Option 3: user provides title + description directly — no AI call needed
    if payload.mode == "user_input":
        return api_success({
            "title": payload.userTitle.strip(),
            "description": payload.userDescription.strip(),
        })

    _rate_limiter.check(str(current_user.id), count=2)

    results = await ai_suggestion_service.suggest_hotspot(
        db=db,
        product_id=product_uuid,
        user_id=current_user.id,
        image_url=payload.imageUrl,
        mode=payload.mode,
        user_prompt=payload.userPrompt,
    )
    return api_success(results)


# ---------------------------------------------------------------------------
# Feature 3 — Product link name + description
# DB needed to fetch product name + existing link names (avoid duplicates)
# ---------------------------------------------------------------------------

@router.post("/link-suggest", response_model=dict)
async def link_suggest(
    payload: LinkSuggestRequest,
    current_user: CurrentUser,
    db: DB,
):
    """
    Suggest a name and description for a product link (URL).

    Three modes:
      - ai_suggest   : AI analyses the URL + product context to suggest.
      - user_prompt  : User provides a hint; AI refines the suggestion.
      - user_input   : User supplies name + description directly; no AI call.
    """
    try:
        product_uuid = uuid.UUID(payload.productId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid productId format")

    # Option 3: user provides name + description directly — no AI call needed
    if payload.mode == "user_input":
        return api_success({
            "name": payload.userName.strip(),
            "description": payload.userDescription.strip(),
        })

    _rate_limiter.check(str(current_user.id), count=2)

    results = await ai_suggestion_service.suggest_link(
        db=db,
        product_id=product_uuid,
        user_id=current_user.id,
        link_url=payload.linkUrl,
        image_url=payload.imageUrl,
        mode=payload.mode,
        user_prompt=payload.userPrompt,
    )
    return api_success(results)
