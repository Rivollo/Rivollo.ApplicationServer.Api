"""ORM models for the Shopify integration (docs/shopify-integration/spec.md, ADR-016).

An isolated module: four tables, all prefixed ``tbl_shopify_``. The code never
modifies a core table's rows except through the services it calls.

FOREIGN KEYS (standard ownership rules; all CASCADE)
  tbl_shopify_connections.user_id       -> tbl_users
  tbl_shopify_connections.api_key_id    -> tbl_api_keys   (revoke is soft; a hard
                                                           delete takes the binding)
  tbl_shopify_products.user_id          -> tbl_users
  tbl_shopify_products.rivollo_product_id -> tbl_products
  variants / layouts .shopify_product_ref -> tbl_shopify_products
The account purge deletes products (step 6) and then the user (step 9); these
cascades remove every Shopify row. Rivollo.AccountPurge.Job must allow-list the
tbl_users and tbl_products FKs first — docs/account-purge-job-changes.md.
created_by / updated_by stay plain UUIDs (AuditMixin).

Products are SOFT-deleted by the app, so a live link can still point at a
product the seller deleted; services treat that as "not there".

    tbl_shopify_connections        an API key bound to one shop
    tbl_shopify_products           one synced Shopify product -> one Rivollo product
    tbl_shopify_product_variants   the variants the merchant selected (Shopify's
                                   commerce data: price, stock, options)
    tbl_shopify_layouts            one row per value of the product's "layout"
                                   option — each value is a 3D model

"Variant" here always means a SHOPIFY variant. Rivollo's own "model variant"
(ADR-014) and "colour variant" are different things; code says
``shopify_variant`` to keep them apart.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base
from app.models.models import AuditMixin, UUIDMixin

# Option roles (tbl_shopify_products.option_roles values).
ROLE_LAYOUT = "layout"  # each value is a 3D model (the Original or a model variant)
ROLE_INFO = "info"      # a plain selector: price and cart only
ROLE_COLOUR = "colour"  # reserved for the colour phase; rejected in v1


class ShopifyConnection(UUIDMixin, AuditMixin, Base):
    """An API key bound to a shop. The shop for every Shopify call comes from here."""

    __tablename__ = "tbl_shopify_connections"
    __table_args__ = (
        # One live connection per shop, and one live shop per key.
        Index(
            "ux_shopify_connections_active_shop",
            "shop_domain",
            unique=True,
            postgresql_where=text("isactive"),
        ),
        Index(
            "ux_shopify_connections_active_key",
            "api_key_id",
            unique=True,
            postgresql_where=text("isactive"),
        ),
        Index("ix_shopify_connections_user", "user_id"),
        # Full index for the api_key -> connection cascade (the unique one is partial).
        Index("ix_shopify_connections_api_key", "api_key_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_api_keys.id", ondelete="CASCADE"), nullable=False
    )
    shop_domain: Mapped[str] = mapped_column(Text, nullable=False)
    isactive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"), default=True
    )
    connected_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    disconnected_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class ShopifyProduct(UUIDMixin, AuditMixin, Base):
    """A Shopify product as last synced, linked to the Rivollo product sync created."""

    __tablename__ = "tbl_shopify_products"
    __table_args__ = (
        UniqueConstraint("shop_domain", "shopify_product_id", name="uq_shopify_products_shop_product"),
        Index("ix_shopify_products_user", "user_id"),
        Index("ix_shopify_products_rivollo_product", "rivollo_product_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    shop_domain: Mapped[str] = mapped_column(Text, nullable=False)
    # The numeric part of gid://shopify/Product/<id>.
    shopify_product_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    handle: Mapped[str] = mapped_column(Text, nullable=False)
    description_html: Mapped[Optional[str]] = mapped_column(Text)
    vendor: Mapped[Optional[str]] = mapped_column(Text)
    product_type: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # ACTIVE / DRAFT / ARCHIVED. Never written to tbl_products.status, which is
    # the 3D pipeline state.
    shopify_status: Mapped[str] = mapped_column(Text, nullable=False)
    currency: Mapped[str] = mapped_column(Text, nullable=False)
    # [{id, url, alt_text}] — Shopify CDN URLs, stored, fetched only on demand.
    images: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # [{name, values[]}], derived from the synced variants.
    options: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # {"Layout": "layout", "Color": "info"}; unlisted options are "info".
    option_roles: Mapped[dict[str, str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Rivollo configuration dimensions (ADR-017) -> this product's options, by
    # stable codes: {"capacity": {"option_name": "Capacity",
    # "values": {"3_seater": "3 Seater"}}}. NULL = not mapped. Written only by
    # PUT /products/{id}/configurator/configuration, which validates it against
    # the synced options; a later sync that renames an option leaves it as is,
    # and the affected Shopify variants then resolve to no model.
    dimension_mapping: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)

    # The draft product sync created.
    rivollo_product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    # When the main GLB was last requested; drives "stalled" detection, since
    # the product pipeline itself records no start time.
    main_glb_requested_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    synced_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    shopify_variants: Mapped[list["ShopifyProductVariant"]] = relationship(
        "ShopifyProductVariant",
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="ShopifyProductVariant.position",
        lazy="selectin",
    )
    layouts: Mapped[list["ShopifyLayout"]] = relationship(
        "ShopifyLayout",
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="ShopifyLayout.position",
        lazy="selectin",
    )


class ShopifyProductVariant(UUIDMixin, AuditMixin, Base):
    """One Shopify variant the merchant chose to sync."""

    __tablename__ = "tbl_shopify_product_variants"
    __table_args__ = (
        UniqueConstraint(
            "shopify_product_ref", "shopify_variant_id", name="uq_shopify_variants_product_variant"
        ),
    )

    shopify_product_ref: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_shopify_products.id", ondelete="CASCADE"),
        nullable=False,
    )
    shopify_variant_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    sku: Mapped[Optional[str]] = mapped_column(Text)
    # Exact decimals, never floats.
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    compare_at_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    # Merchant-private: never in a public payload.
    inventory_quantity: Mapped[Optional[int]] = mapped_column(Integer)
    available: Mapped[bool] = mapped_column(Boolean, nullable=False)
    image_urls: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # [{name, value}]
    options: Mapped[list[dict[str, str]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    product: Mapped[ShopifyProduct] = relationship("ShopifyProduct", back_populates="shopify_variants")


class ShopifyLayout(UUIDMixin, AuditMixin, Base):
    """One value of the product's layout option, i.e. one 3D model.

    The Original's model is the Rivollo product's main GLB. Every other layout's
    model is the model variant its accepted generation became; generations are
    linked through ``client_ref = "shopify-layout:<id>"`` on
    tbl_model_variant_generations, so no configurator table carries a Shopify
    column and auto-accepted generations need no extra bookkeeping here.
    """

    __tablename__ = "tbl_shopify_layouts"
    __table_args__ = (
        UniqueConstraint("shopify_product_ref", "option_value", name="uq_shopify_layouts_value"),
        Index(
            "ux_shopify_layouts_one_original",
            "shopify_product_ref",
            unique=True,
            postgresql_where=text("is_original"),
        ),
    )

    shopify_product_ref: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_shopify_products.id", ondelete="CASCADE"),
        nullable=False,
    )
    option_value: Mapped[str] = mapped_column(Text, nullable=False)
    is_original: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    product: Mapped[ShopifyProduct] = relationship("ShopifyProduct", back_populates="layouts")

    @property
    def client_ref(self) -> str:
        return layout_client_ref(self.id)


def layout_client_ref(layout_id: uuid.UUID) -> str:
    return f"shopify-layout:{layout_id}"
