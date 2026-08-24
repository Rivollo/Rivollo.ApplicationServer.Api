"""Image content moderation: screen every upload BEFORE it reaches blob storage.

``ImageModerationService.screen`` is called by each image-accepting route with
the raw bytes. It classifies the image and either returns (allowed) or raises
``ContentPolicyViolation`` (rejected, rendered as HTTP 422 by the handler in
``app.main``). Nothing is stored for a rejected image, and nothing is held
against the user — there is no strike count or suspension here.

Providers (IMAGE_MODERATION_PROVIDER):

  azure  Azure AI Content Safety ``image:analyze``. Returns a severity of
         0 / 2 / 4 / 6 for each of four categories — Sexual, Violence, Hate,
         SelfHarm. AZURE_CONTENT_SAFETY_BLOCK_RULES says which categories are
         blocked and from what severity, so the operator decides what kind of
         images are allowed. This is the default.
  fal    fal.ai ``imageutils/nsfw``. One number, ``nsfw_probability``, with no
         categories — only a single threshold can be tuned. Used when Azure is
         not configured, or when explicitly selected.

Failure policy mirrors the email-domain check: fail OPEN on provider/network
trouble (IMAGE_MODERATION_FAIL_OPEN) so an outage does not block product
creation, but log loudly. Fail CLOSED only on a definitive "not allowed".
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Only these are screened. GLB/USDZ/etc. are not images and go straight through.
IMAGE_CONTENT_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif", "image/bmp", "image/tiff",
})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tiff", ".tif"})

ERROR_CODE = "CONTENT_POLICY_VIOLATION"

# Azure Content Safety image limits (2024-09-01): <= 4 MB, 50..7200 px a side,
# JPEG/PNG/GIF/BMP/TIFF/WEBP. Larger inputs are re-encoded to fit.
_AZURE_MAX_BYTES = 4 * 1024 * 1024
_AZURE_MAX_SIDE = 7200
_AZURE_CATEGORIES = ("Hate", "SelfHarm", "Sexual", "Violence")

REJECTION_MESSAGE = (
    "This image isn't allowed on Rivollo. Uploads must be product photos — "
    "explicit, adult, violent or hateful content is blocked by our content "
    "policy. Please choose a different image."
)
UNAVAILABLE_MESSAGE = "We could not verify this image right now. Please try again in a moment."


class ContentPolicyViolation(Exception):
    """Raised when an image is rejected. Rendered by the handler in app.main."""

    status_code = 422
    code = ERROR_CODE

    def __init__(self, message: str, *, reasons: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.reasons = reasons or {}

    def details(self) -> dict:
        # Category -> severity that tripped the rule (azure) or {"nsfw": p} (fal).
        return {"flagged": self.reasons}


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    checked: bool                 # False when skipped (disabled / not image / fail-open)
    provider: Optional[str] = None
    scores: dict = field(default_factory=dict)


def _is_image(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type and content_type.split(";")[0].strip().lower() in IMAGE_CONTENT_TYPES:
        return True
    if filename:
        dot = filename.rfind(".")
        if dot != -1 and filename[dot:].lower() in IMAGE_EXTENSIONS:
            return True
    return False


def _guess_mime(filename: Optional[str], content_type: Optional[str]) -> str:
    if content_type and content_type.startswith("image/"):
        return content_type.split(";")[0].strip()
    guessed = mimetypes.guess_type(filename or "")[0]
    return guessed or "image/png"


def _fit_for_azure(image_bytes: bytes) -> Optional[bytes]:
    """Return bytes Azure will accept, re-encoding/downscaling when needed.

    Returns None if the image cannot be decoded (Pillow missing or bad file),
    in which case the caller treats it as a provider failure.
    """
    if len(image_bytes) <= _AZURE_MAX_BYTES:
        return image_bytes
    try:
        from PIL import Image  # lazy: only needed for oversized uploads
    except ImportError:
        logger.warning("Pillow not available; cannot downscale %d-byte image for Azure", len(image_bytes))
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Halve until under the size cap; JPEG q=85 is plenty for classification.
        for _ in range(6):
            w, h = img.size
            if max(w, h) > _AZURE_MAX_SIDE:
                scale = _AZURE_MAX_SIDE / max(w, h)
                img = img.resize((max(50, int(w * scale)), max(50, int(h * scale))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            if buf.tell() <= _AZURE_MAX_BYTES:
                return buf.getvalue()
            img = img.resize((max(50, img.size[0] // 2), max(50, img.size[1] // 2)))
        return buf.getvalue()
    except Exception as exc:  # decode error, truncated file, etc.
        logger.warning("Could not prepare image for Azure Content Safety: %s", exc)
        return None


class ImageModerationService:
    """Stateless facade; everything it needs comes from settings."""

    # ------------------------------------------------------------------ #
    # Azure AI Content Safety
    # ------------------------------------------------------------------ #
    @staticmethod
    def azure_configured() -> bool:
        return bool(settings.AZURE_CONTENT_SAFETY_ENDPOINT and settings.AZURE_CONTENT_SAFETY_KEY)

    @staticmethod
    async def azure_severities(image_bytes: bytes) -> Optional[dict[str, int]]:
        """Return {category_lower: severity} or None if the service was unavailable."""
        prepared = _fit_for_azure(image_bytes)
        if prepared is None:
            return None

        url = (
            settings.AZURE_CONTENT_SAFETY_ENDPOINT.rstrip("/")
            + f"/contentsafety/image:analyze?api-version={settings.AZURE_CONTENT_SAFETY_API_VERSION}"
        )
        body = {
            "image": {"content": base64.b64encode(prepared).decode("ascii")},
            "categories": list(_AZURE_CATEGORIES),
            "outputType": "FourSeverityLevels",
        }
        timeout = httpx.Timeout(timeout=settings.IMAGE_MODERATION_TIMEOUT_SECONDS, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    headers={
                        "Ocp-Apim-Subscription-Key": settings.AZURE_CONTENT_SAFETY_KEY,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.RequestError as exc:
            logger.warning("Azure Content Safety unreachable: %s", exc)
            return None

        if resp.status_code != 200:
            logger.warning("Azure Content Safety returned HTTP %s: %s", resp.status_code, resp.text[:300])
            return None

        try:
            out: dict[str, int] = {}
            for entry in resp.json().get("categoriesAnalysis", []):
                out[str(entry["category"]).lower()] = int(entry.get("severity", 0))
            return out
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Azure Content Safety returned an unexpected body: %s", exc)
            return None

    @staticmethod
    def azure_violations(severities: dict[str, int]) -> dict[str, int]:
        """Apply AZURE_CONTENT_SAFETY_BLOCK_RULES; returns the categories that tripped."""
        rules = settings.get_content_safety_block_rules()
        return {
            cat: sev
            for cat, sev in severities.items()
            if cat in rules and sev >= rules[cat]
        }

    # ------------------------------------------------------------------ #
    # fal.ai NSFW (single probability, no categories)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def fal_nsfw_score(image_bytes: bytes, mime: str) -> Optional[float]:
        """Return P(nsfw) in [0, 1], or None if the classifier was unavailable."""
        if not settings.FAL_KEY:
            logger.error("Image moderation fell back to fal but FAL_KEY is not set")
            return None

        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        url = f"https://fal.run/{settings.IMAGE_MODERATION_FAL_ENDPOINT_ID}"
        timeout = httpx.Timeout(timeout=settings.IMAGE_MODERATION_TIMEOUT_SECONDS, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Key {settings.FAL_KEY}", "Content-Type": "application/json"},
                    json={"image_url": data_uri},
                )
        except httpx.RequestError as exc:
            logger.warning("fal NSFW classifier unreachable: %s", exc)
            return None

        if resp.status_code != 200:
            logger.warning("fal NSFW classifier returned HTTP %s: %s", resp.status_code, resp.text[:300])
            return None

        try:
            score = float(resp.json()["nsfw_probability"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("fal NSFW classifier returned an unexpected body: %s", exc)
            return None
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    @staticmethod
    def _provider() -> str:
        wanted = (settings.IMAGE_MODERATION_PROVIDER or "azure").strip().lower()
        if wanted == "azure" and not ImageModerationService.azure_configured():
            logger.warning(
                "IMAGE_MODERATION_PROVIDER=azure but AZURE_CONTENT_SAFETY_ENDPOINT/KEY "
                "are not set — falling back to fal"
            )
            return "fal"
        return wanted

    @staticmethod
    async def screen(
        image_bytes: bytes,
        *,
        user_id=None,
        source: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> ModerationResult:
        """Screen an image about to be stored. Raises ContentPolicyViolation."""
        if not settings.IMAGE_MODERATION_ENABLED:
            return ModerationResult(allowed=True, checked=False)
        if not image_bytes or not _is_image(filename, content_type):
            return ModerationResult(allowed=True, checked=False)

        provider = ImageModerationService._provider()
        flagged: dict = {}
        scores: dict = {}
        unavailable = False

        if provider == "azure":
            severities = await ImageModerationService.azure_severities(image_bytes)
            if severities is None:
                unavailable = True
            else:
                scores = severities
                flagged = ImageModerationService.azure_violations(severities)
        else:
            score = await ImageModerationService.fal_nsfw_score(
                image_bytes, _guess_mime(filename, content_type)
            )
            if score is None:
                unavailable = True
            else:
                scores = {"nsfw": score}
                if score >= settings.IMAGE_MODERATION_NSFW_THRESHOLD:
                    flagged = {"nsfw": score}

        if unavailable:
            if settings.IMAGE_MODERATION_FAIL_OPEN:
                logger.warning(
                    "Image moderation (%s) inconclusive for user=%s source=%s — allowing (fail-open)",
                    provider, user_id, source,
                )
                return ModerationResult(allowed=True, checked=False, provider=provider)
            raise ContentPolicyViolation(UNAVAILABLE_MESSAGE)

        if flagged:
            logger.warning(
                "Rejected image user=%s source=%s provider=%s flagged=%s",
                user_id, source, provider, flagged,
            )
            raise ContentPolicyViolation(REJECTION_MESSAGE, reasons=flagged)

        return ModerationResult(allowed=True, checked=True, provider=provider, scores=scores)


image_moderation_service = ImageModerationService()
