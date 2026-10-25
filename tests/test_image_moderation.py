"""ImageModerationService: provider selection, Azure category rules, fail-open."""

import uuid

import pytest

from app.core.config import settings
from app.services.image_moderation_service import (
    ContentPolicyViolation,
    ImageModerationService,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MODERATION_ENABLED", True)
    monkeypatch.setattr(settings, "IMAGE_MODERATION_PROVIDER", "azure")
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_ENDPOINT", "https://x.cognitiveservices.azure.com")
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_KEY", "k")
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_BLOCK_RULES", "Sexual:2,Violence:4,Hate:4,SelfHarm:4")
    monkeypatch.setattr(settings, "IMAGE_MODERATION_FAIL_OPEN", True)


def _azure(monkeypatch, value):
    async def fake(image_bytes):
        return value
    monkeypatch.setattr(ImageModerationService, "azure_severities", staticmethod(fake))


def _fal(monkeypatch, value):
    async def fake(image_bytes, mime):
        return value
    monkeypatch.setattr(ImageModerationService, "fal_nsfw_score", staticmethod(fake))


async def test_non_image_is_skipped(monkeypatch):
    _azure(monkeypatch, {"sexual": 6})
    r = await ImageModerationService.screen(b"glTF....", source="t", filename="m.glb")
    assert r.allowed and not r.checked


async def test_safe_image_allowed(monkeypatch):
    _azure(monkeypatch, {"hate": 0, "selfharm": 0, "sexual": 0, "violence": 2})
    r = await ImageModerationService.screen(PNG, source="t", filename="a.png")
    assert r.allowed and r.checked and r.provider == "azure"


async def test_sexual_low_severity_rejected(monkeypatch):
    _azure(monkeypatch, {"hate": 0, "selfharm": 0, "sexual": 2, "violence": 0})
    with pytest.raises(ContentPolicyViolation) as ei:
        await ImageModerationService.screen(PNG, user_id=uuid.uuid4(), source="t", filename="a.png")
    assert ei.value.status_code == 422 and ei.value.code == "CONTENT_POLICY_VIOLATION"
    assert ei.value.details() == {"flagged": {"sexual": 2}}


async def test_violence_below_rule_allowed_above_rejected(monkeypatch):
    _azure(monkeypatch, {"violence": 2})
    assert (await ImageModerationService.screen(PNG, source="t", filename="a.png")).allowed
    _azure(monkeypatch, {"violence": 4})
    with pytest.raises(ContentPolicyViolation):
        await ImageModerationService.screen(PNG, source="t", filename="a.png")


async def test_category_not_in_rules_is_allowed(monkeypatch):
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_BLOCK_RULES", "Sexual:2")
    _azure(monkeypatch, {"violence": 6, "sexual": 0})
    assert (await ImageModerationService.screen(PNG, source="t", filename="a.png")).allowed


def test_block_rules_parsing(monkeypatch):
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_BLOCK_RULES", " sexual , Violence:6,bogus:x ,Hate:")
    assert settings.get_content_safety_block_rules() == {"sexual": 2, "violence": 6, "hate": 2}


async def test_azure_unconfigured_falls_back_to_fal(monkeypatch):
    monkeypatch.setattr(settings, "AZURE_CONTENT_SAFETY_KEY", "")
    monkeypatch.setattr(settings, "IMAGE_MODERATION_NSFW_THRESHOLD", 0.7)
    _fal(monkeypatch, 0.9)
    with pytest.raises(ContentPolicyViolation) as ei:
        await ImageModerationService.screen(PNG, source="t", content_type="image/png")
    assert ei.value.details() == {"flagged": {"nsfw": 0.9}}
    _fal(monkeypatch, 0.1)
    r = await ImageModerationService.screen(PNG, source="t", content_type="image/png")
    assert r.allowed and r.provider == "fal"


async def test_provider_down_fails_open(monkeypatch):
    _azure(monkeypatch, None)
    r = await ImageModerationService.screen(PNG, source="t", filename="a.png")
    assert r.allowed and not r.checked


async def test_provider_down_fails_closed_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MODERATION_FAIL_OPEN", False)
    _azure(monkeypatch, None)
    with pytest.raises(ContentPolicyViolation):
        await ImageModerationService.screen(PNG, source="t", filename="a.png")


async def test_disabled_skips_everything(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MODERATION_ENABLED", False)
    _azure(monkeypatch, {"sexual": 6})
    r = await ImageModerationService.screen(PNG, source="t", filename="a.png")
    assert r.allowed and not r.checked
