"""Contracts for the Shopify integration (docs/shopify-integration/spec.md).

snake_case, no aliases. Prices travel as decimal STRINGS ("499.00"), never
floats. Every image URL a client sends must be https on cdn.shopify.com; those
URLs are stored, and fetched server-side only through the host-locked importer.

The public payload (``PublicShopifyProduct``) is a separate shopper schema, not
the seller one with fields dropped: it cannot carry inventory, SKU, GIDs, the
Shopify status, connection or user ids.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.shopify import ROLE_COLOUR, ROLE_INFO, ROLE_LAYOUT

SHOPIFY_IMAGE_HOST = "cdn.shopify.com"
_SHOP_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]{0,98}\.myshopify\.com$")
_PRICE = re.compile(r"^\d{1,10}(\.\d{1,2})?$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")

MAX_VARIANTS = 100
MAX_IMAGES = 50
MAX_IMAGES_PER_VARIANT = 20
MAX_OPTIONS = 3  # Shopify's own limit per product
TEXT_MAX = 255


def parse_shopify_id(value: Any, resource: str) -> int:
    """``gid://shopify/<resource>/<n>`` or ``<n>`` -> n. Raises ValueError."""
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    else:
        text = str(value or "").strip()
        prefix = f"gid://shopify/{resource}/"
        if text.startswith("gid://"):
            if not text.startswith(prefix):
                raise ValueError(f"expected a gid://shopify/{resource}/ id")
            text = text[len(prefix):]
        if not text.isdigit():
            raise ValueError(f"expected a numeric Shopify {resource} id")
        number = int(text)
    if number <= 0 or number >= 2**63:
        raise ValueError(f"Shopify {resource} id out of range")
    return number


def validate_shopify_image_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != SHOPIFY_IMAGE_HOST:
        raise ValueError(f"image URLs must be https://{SHOPIFY_IMAGE_HOST}/...")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("image URLs must not carry credentials or a port")
    if len(url) > 2000:
        raise ValueError("image URL is too long")
    return url


def normalise_shop_domain(value: str) -> str:
    domain = (value or "").strip().lower()
    if domain.startswith("https://"):
        domain = domain[len("https://"):]
    domain = domain.rstrip("/")
    if not _SHOP_DOMAIN.match(domain):
        raise ValueError("shop_domain must look like my-store.myshopify.com")
    return domain


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
class ShopifyConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shop_domain: str = Field(..., max_length=120)

    @field_validator("shop_domain")
    @classmethod
    def _domain(cls, value: str) -> str:
        return normalise_shop_domain(value)


class ShopifyConnectionResponse(BaseModel):
    id: uuid.UUID
    shop_domain: str
    api_key_id: uuid.UUID
    isactive: bool
    connected_at: datetime
    disconnected_at: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #
class ShopifyImageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = Field(default=None, max_length=TEXT_MAX)
    url: str
    alt_text: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        return validate_shopify_image_url(value)


class ShopifyOptionValueIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, max_length=TEXT_MAX)
    value: str = Field(..., min_length=1, max_length=TEXT_MAX)


class ShopifyVariantIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shopify_variant_id: int
    title: str = Field(..., min_length=1, max_length=TEXT_MAX)
    sku: Optional[str] = Field(default=None, max_length=TEXT_MAX)
    price: str
    compare_at_price: Optional[str] = None
    inventory: Optional[int] = Field(default=None, ge=-(2**31), le=2**31 - 1)
    available: bool
    image_urls: list[str] = Field(default_factory=list, max_length=MAX_IMAGES_PER_VARIANT)
    # Older plugin builds send a single image_url; folded into image_urls.
    image_url: Optional[str] = None
    options: list[ShopifyOptionValueIn] = Field(default_factory=list, max_length=MAX_OPTIONS)

    @field_validator("shopify_variant_id", mode="before")
    @classmethod
    def _variant_id(cls, value: Any) -> int:
        return parse_shopify_id(value, "ProductVariant")

    @field_validator("price")
    @classmethod
    def _price(cls, value: str) -> str:
        if not _PRICE.match(str(value)):
            raise ValueError('price must be a decimal string such as "499.00"')
        return str(value)

    @field_validator("compare_at_price")
    @classmethod
    def _compare_at(cls, value: Optional[str]) -> Optional[str]:
        if value in (None, ""):
            return None
        if not _PRICE.match(str(value)):
            raise ValueError('compare_at_price must be a decimal string such as "599.00"')
        return str(value)

    @field_validator("image_urls")
    @classmethod
    def _urls(cls, value: list[str]) -> list[str]:
        return [validate_shopify_image_url(u) for u in value]

    @model_validator(mode="after")
    def _fold_single_image(self) -> "ShopifyVariantIn":
        if self.image_url:
            url = validate_shopify_image_url(self.image_url)
            if url not in self.image_urls:
                self.image_urls = [url, *self.image_urls][:MAX_IMAGES_PER_VARIANT]
        self.image_url = None
        names = [o.name for o in self.options]
        if len(names) != len(set(names)):
            raise ValueError("a variant cannot repeat an option name")
        return self


class ShopifySyncRequest(BaseModel):
    """``POST /integrations/shopify/products/sync``.

    Unknown fields are ignored rather than rejected, so a newer plugin build
    sending extra Shopify data does not break sync.
    """

    model_config = ConfigDict(extra="ignore")

    shopify_product_id: int
    title: str = Field(..., min_length=1, max_length=TEXT_MAX)
    handle: str = Field(..., min_length=1, max_length=TEXT_MAX)
    description: Optional[str] = Field(default=None, max_length=100_000)
    vendor: Optional[str] = Field(default=None, max_length=TEXT_MAX)
    product_type: Optional[str] = Field(default=None, max_length=TEXT_MAX)
    tags: list[str] = Field(default_factory=list, max_length=250)
    status: str = Field(..., max_length=20)
    currency: str
    images: list[ShopifyImageIn] = Field(default_factory=list, max_length=MAX_IMAGES)
    variants: list[ShopifyVariantIn] = Field(..., min_length=1, max_length=MAX_VARIANTS)

    @field_validator("shopify_product_id", mode="before")
    @classmethod
    def _product_id(cls, value: Any) -> int:
        return parse_shopify_id(value, "Product")

    @field_validator("status")
    @classmethod
    def _status(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not re.match(r"^[A-Z_]{1,20}$", cleaned):
            raise ValueError("status must be a Shopify product status such as ACTIVE")
        return cleaned

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not _CURRENCY.match(cleaned):
            raise ValueError("currency must be an ISO 4217 code such as INR")
        return cleaned

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str]) -> list[str]:
        return [t[:TEXT_MAX] for t in (tag.strip() for tag in value) if t]

    @model_validator(mode="after")
    def _unique_variants(self) -> "ShopifySyncRequest":
        ids = [v.shopify_variant_id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("variants must not repeat a shopify_variant_id")
        return self


class ShopifySyncResponse(BaseModel):
    id: uuid.UUID
    shopify_product_id: str
    rivollo_product_id: uuid.UUID
    rivollo_status: Optional[str] = None
    synced_at: datetime
    variants_synced: int
    created: bool


# --------------------------------------------------------------------------- #
# Options, GLB requests
# --------------------------------------------------------------------------- #
ShopifyOptionRole = Literal["layout", "info", "colour"]


class ShopifyOptionsRequest(BaseModel):
    """``PUT /integrations/shopify/products/{id}/options``."""

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, ShopifyOptionRole] = Field(default_factory=dict, max_length=MAX_OPTIONS)
    original_layout_value: Optional[str] = Field(default=None, min_length=1, max_length=TEXT_MAX)


class ShopifyMainGlbRequest(BaseModel):
    """``POST /integrations/shopify/products/{id}/glb``."""

    model_config = ConfigDict(extra="forbid")

    image_url: str
    model: Optional[str] = Field(default=None, max_length=100)
    # Restart a main-GLB generation that has been stuck longer than
    # GENERATION_STALE_AFTER_SECONDS. Charged again.
    retry: bool = False

    @field_validator("image_url")
    @classmethod
    def _url(cls, value: str) -> str:
        return validate_shopify_image_url(value)


class ShopifyLayoutGlbRequest(BaseModel):
    """``POST /integrations/shopify/products/{id}/layouts/{layout_id}/glb``."""

    model_config = ConfigDict(extra="forbid")

    image_url: str
    model: Optional[str] = Field(default=None, max_length=100)
    auto_accept: bool = False

    @field_validator("image_url")
    @classmethod
    def _url(cls, value: str) -> str:
        return validate_shopify_image_url(value)


# --------------------------------------------------------------------------- #
# State (GET /integrations/shopify/products/{id}) — the merchant's own data
# --------------------------------------------------------------------------- #
class ShopifyVariantOut(BaseModel):
    shopify_variant_id: str
    title: str
    sku: Optional[str] = None
    price: str
    compare_at_price: Optional[str] = None
    inventory: Optional[int] = None
    available: bool
    image_urls: list[str] = Field(default_factory=list)
    options: dict[str, str] = Field(default_factory=dict)


class ShopifyLayoutModelOut(BaseModel):
    id: str  # "original" or the model variant id, as in the shopper payload
    name: str
    glb_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ShopifyLayoutOut(BaseModel):
    id: uuid.UUID
    option_value: str
    is_original: bool
    position: int
    # "none" | "generating" | "ready_for_review" | "ready" | "failed"
    state: str
    model: Optional[ShopifyLayoutModelOut] = None
    candidate_image_urls: list[str] = Field(default_factory=list)
    generations: list[dict[str, Any]] = Field(default_factory=list)


class RivolloProductOut(BaseModel):
    id: uuid.UUID
    status: str
    # "none" | "generating" | "stalled" | "ready"
    main_glb_state: str
    glb_url: Optional[str] = None
    usdz_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    public_id: Optional[str] = None
    viewer_url: Optional[str] = None


class ShopifyProductStateResponse(BaseModel):
    id: uuid.UUID
    shopify_product_id: str
    shop_domain: str
    title: str
    handle: str
    currency: str
    shopify_status: str
    synced_at: datetime
    images: list[dict[str, Any]] = Field(default_factory=list)
    options: list[dict[str, Any]] = Field(default_factory=list)
    option_roles: dict[str, str] = Field(default_factory=dict)
    variants: list[ShopifyVariantOut] = Field(default_factory=list)
    # None when the linked Rivollo product was deleted; the next sync recreates it.
    rivollo_product: Optional[RivolloProductOut] = None
    layouts: list[ShopifyLayoutOut] = Field(default_factory=list)


class ShopifyProductSummary(BaseModel):
    id: uuid.UUID
    shopify_product_id: str
    title: str
    rivollo_product_id: uuid.UUID
    synced_at: datetime


# --------------------------------------------------------------------------- #
# Public shopper payload (Phase 3)
# --------------------------------------------------------------------------- #
class PublicShopifyOption(BaseModel):
    name: str
    role: Literal["layout", "info"]
    values: list[str]


class PublicShopifyLayout(BaseModel):
    value: str
    model: str  # "original" or a model variant id, as in the configurator payload


class PublicShopifyVariant(BaseModel):
    id: str
    title: str
    options: dict[str, str]
    price: str
    compare_at_price: Optional[str] = None
    available: bool
    image_url: Optional[str] = None
    add_to_cart_url: str


class PublicShopifyProduct(BaseModel):
    title: str
    currency: str
    product_url: str
    layout_option: Optional[str] = None
    options: list[PublicShopifyOption]
    layouts: list[PublicShopifyLayout]
    variants: list[PublicShopifyVariant]


ROLES_ACCEPTED_IN_V1 = (ROLE_LAYOUT, ROLE_INFO)
ROLE_RESERVED = ROLE_COLOUR
