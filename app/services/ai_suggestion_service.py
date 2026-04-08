"""AI suggestion service — GPT-4o Vision with retry, validation, and usage logging."""

import json
import logging
import posixpath
import uuid
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
)

from app.core.config import settings
from app.models.models import Hotspot, Product, ProductLink

logger = logging.getLogger(__name__)

# Input length caps — driven by config (override via .env)
_USER_PROMPT_MAX_CHARS = settings.AI_USER_PROMPT_MAX_CHARS
_USER_INPUT_NAME_MAX_CHARS = settings.AI_USER_INPUT_NAME_MAX_CHARS
_USER_INPUT_DESC_MAX_CHARS = settings.AI_USER_INPUT_DESC_MAX_CHARS
_LINK_URL_MAX_CHARS = settings.AI_LINK_URL_MAX_CHARS


# ---------------------------------------------------------------------------
# GLB helper — GPT-4o Vision cannot decode binary GLB files,
# but the filename often carries product-relevant context (e.g.
# "leather-sofa-v2.glb" → "leather sofa v2").  We extract it and
# pass it as a text hint rather than pretending the model can see the file.
# ---------------------------------------------------------------------------

def _glb_display_name(glb_url: str) -> Optional[str]:
    """Return a human-readable name derived from a GLB CDN URL, or None."""
    try:
        path = urlparse(glb_url).path
        filename = posixpath.basename(path)
        stem = posixpath.splitext(filename)[0].strip()
        if not stem:
            return None
        return stem.replace("-", " ").replace("_", " ")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal sentinel — raised inside _http_post to trigger tenacity retry
# ---------------------------------------------------------------------------
class _OpenAIRetryable(Exception):
    """Raised when OpenAI returns a retryable error (429, 5xx, timeout)."""

    def __init__(self, status_code: int, body: str, retry_after: int = 0):
        self.status_code = status_code
        self.body = body
        # Seconds to wait before next attempt (from Retry-After header, if present)
        self.retry_after = retry_after
        super().__init__(f"OpenAI {status_code}")


# ---------------------------------------------------------------------------
# Custom tenacity wait — respects Retry-After header from OpenAI 429
# ---------------------------------------------------------------------------
def _openai_wait(retry_state) -> float:
    """
    If OpenAI sent a Retry-After header, honour it (capped at 60s).
    Otherwise fall back to exponential backoff: 2s, 4s, 8s, …, max 20s.
    """
    exc = retry_state.outcome.exception()
    if isinstance(exc, _OpenAIRetryable) and exc.retry_after > 0:
        wait = min(exc.retry_after, 60)
        logger.info("Respecting Retry-After: waiting %ds before next attempt", wait)
        return float(wait)
    # exponential: 2^(attempt-1) * 2, capped at 20s
    return min(2 ** (retry_state.attempt_number - 1) * 2, 20)


# ---------------------------------------------------------------------------
# Response validators
# ---------------------------------------------------------------------------

def _validate_product_response(data: dict) -> dict:
    """Ensure the parsed JSON has non-empty name + description strings."""
    name = data.get("name")
    description = data.get("description")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Missing or empty 'name' in OpenAI response: {data}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Missing or empty 'description' in OpenAI response: {data}")

    # Soft-trim to word limits — never fail the user for a slightly wordy response
    return {
        "name": " ".join(name.strip().split()[:6]),
        "description": " ".join(description.strip().split()[:50]),
    }


def _validate_link_response(data: dict) -> dict:
    """Ensure the parsed JSON has non-empty name + description strings."""
    name = data.get("name")
    description = data.get("description")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Missing or empty 'name' in OpenAI response: {data}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Missing or empty 'description' in OpenAI response: {data}")

    return {
        "name": " ".join(name.strip().split()[:6]),
        "description": " ".join(description.strip().split()[:30]),
    }


def _validate_hotspot_response(data: dict) -> dict:
    """Ensure the parsed JSON has non-empty title + description strings."""
    title = data.get("title")
    description = data.get("description")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Missing or empty 'title' in OpenAI response: {data}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Missing or empty 'description' in OpenAI response: {data}")

    return {
        "title": " ".join(title.strip().split()[:5]),
        "description": " ".join(description.strip().split()[:30]),
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AiSuggestionService:

    # ------------------------------------------------------------------
    # Feature 1 — Product name + description
    # ------------------------------------------------------------------

    @staticmethod
    async def suggest_product(
        image_url: str,
        glb_url: Optional[str] = None,
        mode: str = "ai_suggest",
        user_prompt: Optional[str] = None,
    ) -> dict:
        """
        Suggest a product name (max 6 words) and description (max 50 words).

        Modes:
          - ai_suggest  : AI analyses the image (+ notes the GLB if provided).
          - user_prompt : User hint is prepended so AI can tailor the suggestion.

        Returns {"name": "...", "description": "..."}
        """
        json_format = (
            '{"name": "<product name, max 6 words>", '
            '"description": "<product description, max 50 words>"}'
        )

        # GPT-4o Vision cannot read binary GLB files, but the filename
        # often carries useful context (e.g. "leather-sofa-v2.glb").
        # We extract it and pass it as a text hint.
        glb_note = ""
        if glb_url:
            glb_name = _glb_display_name(glb_url)
            glb_note = (
                f" The product also has a 3D model (GLB) named '{glb_name}'."
                if glb_name
                else " The product also has a 3D model (GLB format)."
            )

        if mode == "user_prompt" and user_prompt:
            safe_prompt = user_prompt.strip()[:_USER_PROMPT_MAX_CHARS].replace('"', "'")
            prompt_text = (
                f"The user describes the product as: '{safe_prompt}'\n\n"
                f"Look at the product image{glb_note} and return JSON in exactly this format:\n"
                f"{json_format}"
            )
        else:
            prompt_text = (
                f"Look at the product image{glb_note} and return JSON in exactly this format:\n"
                f"{json_format}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional product copywriter. "
                    "You always respond with ONLY a valid JSON object — no markdown, no extra text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                ],
            },
        ]

        raw = await AiSuggestionService._call_openai(messages, context="product-suggest")
        try:
            return _validate_product_response(raw)
        except ValueError as exc:
            logger.error("product-suggest response validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI returned an unexpected response. Please try again.",
            )

    # ------------------------------------------------------------------
    # Feature 2 — Hotspot title + description
    # ------------------------------------------------------------------

    @staticmethod
    async def suggest_hotspot(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        image_url: str,
        mode: str = "ai_suggest",
        user_prompt: Optional[str] = None,
    ) -> dict:
        """
        Suggest a hotspot title (max 5 words) and description (max 30 words).

        Modes:
          - ai_suggest  : AI analyses the image in product context.
          - user_prompt : User hint is prepended so AI can tailor the suggestion.

        Fetches product name + existing hotspot labels from DB to avoid duplicates.
        Returns {"title": "...", "description": "..."}
        """
        json_format = (
            '{"title": "<hotspot title, max 5 words>", '
            '"description": "<hotspot description, max 30 words>"}'
        )

        # Fetch product name — also enforces ownership (created_by = user_id)
        product_result = await db.execute(
            select(Product.name).where(
                Product.id == product_id,
                Product.created_by == user_id,
            )
        )
        product_name: Optional[str] = product_result.scalar_one_or_none()
        if product_name is None:
            # Return 404 in both "not found" and "not owned" cases —
            # never leak whether the product exists to a different user.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found",
            )

        # Fetch existing hotspot labels to avoid duplicates
        hotspot_result = await db.execute(
            select(Hotspot.label).where(Hotspot.product_id == product_id)
        )
        existing_titles: list[str] = [row[0] for row in hotspot_result.fetchall()]

        # Build system context
        context_parts = [f'Product name: "{product_name}"']
        if existing_titles:
            titles_str = ", ".join(f'"{t}"' for t in existing_titles)
            context_parts.append(
                f"Existing hotspot titles — do NOT suggest any of these: {titles_str}"
            )
        system_context = "\n".join(context_parts)

        # Build user prompt text based on mode
        if mode == "user_prompt" and user_prompt:
            safe_prompt = user_prompt.strip()[:_USER_PROMPT_MAX_CHARS].replace('"', "'")
            prompt_text = (
                f"The user points to a part of the image and says: '{safe_prompt}'\n\n"
                f"Look at the image and return JSON in exactly this format:\n{json_format}"
            )
        else:
            prompt_text = (
                f"Look at the image and suggest a title and description for a notable feature or part.\n"
                f"Return JSON in exactly this format:\n{json_format}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional product annotator. "
                    "You always respond with ONLY a valid JSON object — no markdown, no extra text.\n"
                    f"{system_context}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                ],
            },
        ]

        raw = await AiSuggestionService._call_openai(messages, context="hotspot-suggest")
        try:
            return _validate_hotspot_response(raw)
        except ValueError as exc:
            logger.error("hotspot-suggest response validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI returned an unexpected response. Please try again.",
            )

    # ------------------------------------------------------------------
    # Feature 3 — Product link name + description
    # ------------------------------------------------------------------

    @staticmethod
    async def suggest_link(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        link_url: str,
        image_url: Optional[str] = None,
        mode: str = "ai_suggest",
        user_prompt: Optional[str] = None,
    ) -> dict:
        """
        Suggest a link name (max 6 words) and description (max 30 words)
        for a ProductLink URL being added to a product.

        Modes:
          - ai_suggest  : AI analyses the URL + product context.
          - user_prompt : User hint is prepended to tailor the suggestion.

        Fetches product name + existing link names from DB to avoid duplicates.
        Optionally accepts an imageUrl for additional visual context.
        Returns {"name": "...", "description": "..."}
        """
        json_format = (
            '{"name": "<link display name, max 6 words>", '
            '"description": "<link description, max 30 words>"}'
        )

        # Fetch product name — also enforces ownership (created_by = user_id)
        product_result = await db.execute(
            select(Product.name).where(
                Product.id == product_id,
                Product.created_by == user_id,
            )
        )
        product_name: Optional[str] = product_result.scalar_one_or_none()
        if product_name is None:
            # Return 404 in both "not found" and "not owned" cases —
            # never leak whether the product exists to a different user.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found",
            )

        # Sanitise link_url before embedding in prompt — cap length and
        # escape quotes to prevent prompt injection via a crafted URL.
        safe_link_url = link_url[:_LINK_URL_MAX_CHARS].replace('"', "'")

        # Fetch existing link names for this product to avoid duplicates
        link_result = await db.execute(
            select(ProductLink.name).where(
                ProductLink.productid == str(product_id),
                ProductLink.isactive.is_(True),
            )
        )
        existing_names: list[str] = [row[0] for row in link_result.fetchall()]

        # Build system context
        context_parts = [f'Product name: "{product_name}"']
        if existing_names:
            names_str = ", ".join(f'"{n}"' for n in existing_names)
            context_parts.append(
                f"Existing link names — do NOT suggest any of these: {names_str}"
            )
        system_context = "\n".join(context_parts)

        # Build prompt text based on mode
        if mode == "user_prompt" and user_prompt:
            safe_prompt = user_prompt.strip()[:_USER_PROMPT_MAX_CHARS].replace('"', "'")
            prompt_text = (
                f"The user describes this link as: '{safe_prompt}'\n"
                f"Link URL: {safe_link_url}\n\n"
                f"Return JSON in exactly this format:\n{json_format}"
            )
        else:
            prompt_text = (
                f"Suggest a display name and short description for this link.\n"
                f"Link URL: {safe_link_url}\n\n"
                f"Return JSON in exactly this format:\n{json_format}"
            )

        # Build message content — include image if provided
        user_content: list[dict] = [{"type": "text", "text": prompt_text}]
        if image_url:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional product copywriter. "
                    "You always respond with ONLY a valid JSON object — no markdown, no extra text.\n"
                    f"{system_context}"
                ),
            },
            {"role": "user", "content": user_content},
        ]

        raw = await AiSuggestionService._call_openai(messages, context="link-suggest")
        try:
            return _validate_link_response(raw)
        except ValueError as exc:
            logger.error("link-suggest response validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI returned an unexpected response. Please try again.",
            )

    # ------------------------------------------------------------------
    # Core OpenAI call — retry, Retry-After, usage logging, choices guard
    # ------------------------------------------------------------------

    @staticmethod
    async def _call_openai(messages: list[dict], context: str = "") -> dict:
        """
        POST to OpenAI chat completions with:
          - Shared httpx client across all retry attempts (connection pool reuse)
          - Up to OPENAI_MAX_RETRIES attempts with Retry-After-aware backoff
          - Immediate fail on 400/401/403 (config errors, not transient)
          - finish_reason guard against truncated responses
          - Usage token logging on success
        """
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API key is not configured",
            )

        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": messages,
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 1.0,
            "response_format": {"type": "json_object"},
        }

        # One shared client across ALL retry attempts — connection pool is preserved.
        # Timeout is per-attempt (not total); configure via OPENAI_TIMEOUT_SECONDS.
        async with httpx.AsyncClient(timeout=float(settings.OPENAI_TIMEOUT_SECONDS)) as client:
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(_OpenAIRetryable),
                    stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES),
                    wait=_openai_wait,
                    reraise=False,
                ):
                    with attempt:
                        response = await AiSuggestionService._http_post(
                            client, api_key, payload, context
                        )

            except RetryError as exc:
                last: _OpenAIRetryable = exc.last_attempt.exception()
                logger.error(
                    "[%s] OpenAI exhausted %d retries — last status %d",
                    context,
                    settings.OPENAI_MAX_RETRIES,
                    last.status_code,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI service is temporarily unavailable. Please try again shortly.",
                )

        # --- Parse response ---
        data = response.json()

        # Log token usage for cost tracking
        usage = data.get("usage", {})
        logger.info(
            "[%s] OpenAI usage — prompt=%s completion=%s total=%s",
            context,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            usage.get("total_tokens", "?"),
        )

        # Guard: choices may be empty (e.g. content policy filter)
        choices = data.get("choices")
        if not choices:
            logger.error("[%s] OpenAI returned empty choices: %s", context, data)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI service returned no result (possible content filter). Please try again.",
            )

        choice = choices[0]

        # Guard: truncated response due to max_tokens
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            logger.warning(
                "[%s] OpenAI response truncated (finish_reason=length) — "
                "consider increasing OPENAI_MAX_TOKENS",
                context,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI response was truncated. Please try again.",
            )

        raw_content: str = choice["message"]["content"]

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            logger.error(
                "[%s] OpenAI returned non-JSON content: %s",
                context,
                raw_content[:300],
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI service returned unexpected response format. Please try again.",
            )

    @staticmethod
    async def _http_post(
        client: httpx.AsyncClient,
        api_key: str,
        payload: dict,
        context: str,
    ) -> httpx.Response:
        """
        Execute one HTTP POST to OpenAI using the shared client.
        Raises _OpenAIRetryable for 429/5xx/timeout (tenacity will retry).
        Raises HTTPException immediately for 400/401/403 (no retry).
        """
        try:
            auth_headers = (
                {"api-key": api_key}
                if settings.OPENAI_USE_AZURE
                else {"Authorization": f"Bearer {api_key}"}
            )
            response = await client.post(
                settings.OPENAI_CHAT_URL,
                headers={"Content-Type": "application/json", **auth_headers},
                json=payload,
            )
        except httpx.TimeoutException:
            logger.warning("[%s] OpenAI request timed out — will retry", context)
            raise _OpenAIRetryable(408, "timeout")
        except httpx.RequestError as exc:
            logger.warning("[%s] OpenAI network error: %s — will retry", context, exc)
            raise _OpenAIRetryable(0, str(exc))

        if response.status_code == 200:
            return response

        body_preview = response.text[:200]

        if response.status_code == 429:
            # Parse Retry-After and pass it through for _openai_wait to honour
            retry_after_str = response.headers.get("Retry-After", "0")
            try:
                retry_after = int(retry_after_str)
            except ValueError:
                retry_after = 10  # safe default if header is non-numeric
            logger.warning(
                "[%s] OpenAI 429 rate limit — Retry-After: %ds",
                context,
                retry_after,
            )
            raise _OpenAIRetryable(429, body_preview, retry_after=retry_after)

        if response.status_code >= 500:
            logger.warning(
                "[%s] OpenAI %d server error — will retry: %s",
                context,
                response.status_code,
                body_preview,
            )
            raise _OpenAIRetryable(response.status_code, body_preview)

        # 400/401/403 — configuration error, no point retrying
        if response.status_code in (400, 401, 403):
            logger.error(
                "[%s] OpenAI %d non-retryable error: %s",
                context,
                response.status_code,
                body_preview,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI service configuration error ({response.status_code}). Contact support.",
            )

        logger.error(
            "[%s] OpenAI unexpected status %d: %s",
            context,
            response.status_code,
            body_preview,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service returned error {response.status_code}.",
        )


ai_suggestion_service = AiSuggestionService()
