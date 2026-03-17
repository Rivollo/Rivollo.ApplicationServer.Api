from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, column_property, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.types import TIMESTAMP
from geoalchemy2 import Geometry, WKTElement

from app.models.base import Base


class UUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class AuditMixin:
    """Audit fields that exist in all tables: created_by, created_date, updated_by, updated_date"""
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))
    created_date: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))
    updated_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class CreatedAtMixin:
    """For tables that have created_at column directly (in addition to audit fields)"""
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class TimestampMixin(AuditMixin):
    """Map created_at/updated_at to created_date/updated_date for backward compatibility"""
    @property
    def created_at(self) -> datetime:
        return self.created_date

    @property
    def updated_at(self) -> Optional[datetime]:
        return self.updated_date


class SoftDeleteMixin:
    deleted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class OrgRole(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"




class AssetType(str, enum.Enum):
    IMAGE = "image"
    MODEL = "model"
    MASK = "mask"
    THUMBNAIL = "thumbnail"


class ProductStatus(str, enum.Enum):
    DRAFT = "draft"
    QUEUE = "queue"
    PROCESSING = "processing"
    READY = "ready"
    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"
    ARCHIVED = "archived"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HotspotActionType(str, enum.Enum):
    NONE = "none"
    LINK = "link"
    MATERIAL_SWITCH = "material-switch"
    VARIANT_SWITCH = "variant-switch"
    TEXT_ONLY = "text-only"


class NotificationChannel(str, enum.Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"


class AuthProvider(str, enum.Enum):
    GOOGLE = "google"
    EMAIL = "email"


class Organization(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tbl_organizations"

    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    # branding is TEXT in database, storing JSON as string
    branding: Mapped[Optional[str]] = mapped_column(Text)
    # Virtual column - organizations table doesn't have deleted_at in database
    deleted_at = column_property(literal_column("NULL::timestamptz"))

    __table_args__ = (
        Index(
            "ix_organizations_slug_unique",
            "slug",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    members: Mapped[list["OrgMember"]] = relationship("OrgMember", back_populates="organization")
    assets: Mapped[list["Asset"]] = relationship("Asset", back_populates="organization")


class User(UUIDMixin, CreatedAtMixin, AuditMixin, Base):
    """User model - has BOTH created_at AND audit fields (created_date, etc.)"""
    __tablename__ = "tbl_users"

    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(Text)
    name: Mapped[Optional[str]] = mapped_column(Text)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text)
    bio: Mapped[Optional[str]] = mapped_column(Text)

    # Virtual column - users table doesn't have deleted_at in database
    deleted_at = column_property(literal_column("NULL::timestamptz"))

    # Property for backward compatibility
    @property
    def updated_at(self) -> Optional[datetime]:
        return self.updated_date

    subscriptions: Mapped[list["Subscription"]] = relationship("Subscription", back_populates="user")
    licenses: Mapped[list["LicenseAssignment"]] = relationship("LicenseAssignment", back_populates="user")
    identities: Mapped[list["AuthIdentity"]] = relationship("AuthIdentity", back_populates="user")


class OrgMember(AuditMixin, Base):
    __tablename__ = "tbl_org_members"
    __table_args__ = (UniqueConstraint("org_id", "user_id", name="uq_org_user"),)

    org_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[OrgRole] = mapped_column(Enum(OrgRole, name="org_role", native_enum=False), nullable=False)

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    organization: Mapped[Organization] = relationship("Organization", back_populates="members")
    user: Mapped[User] = relationship("User")




class Asset(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_assets"
    __table_args__ = (Index("ix_assets_org_type", "org_id", "type"),)

    org_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[AssetType] = mapped_column(Enum(AssetType, name="asset_type", native_enum=False), nullable=False)
    storage: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'azure_blob'"))
    url: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String)

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    organization: Mapped[Organization] = relationship("Organization", back_populates="assets")


class Product(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tbl_products"
    __table_args__ = ()

    # No org_id column in current DB snapshot; expose virtual NULL
    org_id = column_property(literal_column("NULL::uuid"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ProductStatus] = mapped_column(
        Enum(ProductStatus, name="product_status", native_enum=False), nullable=False, server_default=text("'draft'"),
    )
    # cover_asset_id column no longer exists in some database snapshots; keep virtual
    cover_asset_id = column_property(literal_column("NULL::uuid"))
    model_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_assets.id", ondelete="SET NULL")
    )
    # tags column absent in legacy snapshot; expose as virtual empty array
    tags = column_property(literal_column("'{}'::text[]"))
    # Note: products table doesn't have metadata column in actual DB
    # Keeping as virtual column for backward compatibility
    product_metadata = column_property(literal_column("'{}'::jsonb"))
    published_at = column_property(literal_column("NULL::timestamptz"))
    # New columns added to tbl_products
    description: Mapped[Optional[str]] = mapped_column(Text)
    price: Mapped[Optional[float]] = mapped_column(BigInteger)  # Price stored as integer (cents) or float value
    currency_type: Mapped[Optional[int]] = mapped_column(Integer)  # Currency type ID (integer)
    background_type: Mapped[Optional[int]] = mapped_column(Integer)  # Background ID (integer)
    # created_by, updated_by from TimestampMixin -> AuditMixin
    # Virtual column - products table doesn't have deleted_at in database
    deleted_at = column_property(literal_column("NULL::timestamptz"))

    # No organization relationship without org_id FK
    configurator: Mapped[Optional["Configurator"]] = relationship(
        "Configurator", back_populates="product", uselist=False
    )
    hotspots: Mapped[list["Hotspot"]] = relationship(
        "Hotspot", back_populates="product", cascade="all, delete-orphan"
    )
    dimensions: Mapped[list["ProductDimensions"]] = relationship(
        "ProductDimensions", back_populates="product", cascade="all, delete-orphan"
    )
    dimension_groups: Mapped[list["ProductDimensionGroup"]] = relationship(
        "ProductDimensionGroup", back_populates="product", cascade="all, delete-orphan"
    )
    publish_links: Mapped[list["PublishLink"]] = relationship(
        "PublishLink", back_populates="product", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="product")


class Configurator(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_configurators"

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    # settings is TEXT in database, storing JSON as string
    settings: Mapped[Optional[str]] = mapped_column(Text)

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    product: Mapped[Product] = relationship("Product", back_populates="configurator")


class HotspotType(AuditMixin, Base):
    __tablename__ = "tbl_hotspot_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    hotspots: Mapped[list["Hotspot"]] = relationship("Hotspot", back_populates="hotspot_type")


class Hotspot(UUIDMixin, CreatedAtMixin, AuditMixin, Base):
    """Hotspot - has BOTH created_at AND audit fields"""
    __tablename__ = "tbl_hotspots"
    __table_args__ = (
        Index("ix_hotspots_product_order", "product_id", "order_index"),
        CheckConstraint("pos_x BETWEEN -1.0 AND 1.0", name="ck_hotspot_pos_x"),
        CheckConstraint("pos_y BETWEEN -1.0 AND 1.0", name="ck_hotspot_pos_y"),
        CheckConstraint("pos_z BETWEEN -1.0 AND 1.0", name="ck_hotspot_pos_z"),
    )

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # Legacy columns for backward compatibility (kept for migration period)
    pos_x: Mapped[float] = mapped_column(nullable=False)
    pos_y: Mapped[float] = mapped_column(nullable=False)
    pos_z: Mapped[float] = mapped_column(nullable=False)
    # PostGIS geometry column for 3D position (PointZ)
    position_3d: Mapped[Optional[Any]] = mapped_column(
        Geometry("POINTZ", srid=0, spatial_index=True), nullable=True
    )
    text_font: Mapped[Optional[str]] = mapped_column(String)
    text_color: Mapped[Optional[str]] = mapped_column(String)
    bg_color: Mapped[Optional[str]] = mapped_column(String)
    action_type: Mapped[HotspotActionType] = mapped_column(
        Enum(HotspotActionType, name="hotspot_action", native_enum=False),
        nullable=False,
        server_default=text("'none'"),
    )
    # action_payload is TEXT in database, storing JSON as string
    action_payload: Mapped[Optional[str]] = mapped_column(Text)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    hotspot_type_id: Mapped[Optional[int]] = mapped_column(
        "hotspot_type", Integer, ForeignKey("tbl_hotspot_type.id", ondelete="SET NULL")
    )

    # Property for backward compatibility
    @property
    def updated_at(self) -> Optional[datetime]:
        return self.updated_date

    def get_position_from_geometry(self) -> tuple[float, float, float]:
        """Extract x, y, z coordinates from PostGIS geometry."""
        if self.position_3d is not None:
            try:
                # Try to extract coordinates from PostGIS geometry
                # This will be handled by SQL queries using ST_X, ST_Y, ST_Z
                # For now, fallback to legacy columns
                return (self.pos_x, self.pos_y, self.pos_z)
            except Exception:
                return (self.pos_x, self.pos_y, self.pos_z)
        return (self.pos_x, self.pos_y, self.pos_z)

    def set_position_to_geometry(self, x: float, y: float, z: float) -> None:
        """Set PostGIS geometry from x, y, z coordinates."""
        # Create a PostGIS PointZ geometry using WKT
        # Format: POINTZ(x y z)
        wkt = f"POINTZ({x} {y} {z})"
        self.position_3d = WKTElement(wkt, srid=0)
        # Also update legacy columns for backward compatibility
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z

    hotspot_type: Mapped[Optional[HotspotType]] = relationship("HotspotType", back_populates="hotspots")
    product: Mapped[Product] = relationship("Product", back_populates="hotspots")


class ProductDimensionGroup(AuditMixin, Base):
    """Dimension group - a collection of dimension parameters."""
    __tablename__ = "tbl_product_dimension_groups"
    __table_args__ = (
        Index("ix_dimension_groups_product_order", "product_id", "order_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)  # Name of the dimension group
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="dimension_groups")
    dimensions: Mapped[list["ProductDimensions"]] = relationship(
        "ProductDimensions", back_populates="dimension_group", cascade="all, delete-orphan"
    )


class ProductDimensions(AuditMixin, Base):
    """Product dimensions - allows multiple dimensions per product."""
    __tablename__ = "tbl_product_dimensions"
    __table_args__ = (
        Index("ix_product_dimensions_product_type", "product_id", "dimension_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )

    # Dimension identification
    dimension_type: Mapped[str] = mapped_column(String, nullable=True)  # 'width', 'height', 'depth', etc.
    dimension_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # Optional display name

    # Dimension values
    value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'cm'"))

    # Associated hotspots
    start_hotspot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_hotspots.id", ondelete="SET NULL"), nullable=True
    )
    end_hotspot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_hotspots.id", ondelete="SET NULL"), nullable=True
    )

    # Ordering for multiple dimensions of the same type
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    
    # Link to dimension group
    dimension_group_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_product_dimension_groups.id", ondelete="CASCADE"), nullable=True
    )

    # Relationships
    product: Mapped["Product"] = relationship("Product", back_populates="dimensions")
    dimension_group: Mapped[Optional["ProductDimensionGroup"]] = relationship(
        "ProductDimensionGroup", back_populates="dimensions"
    )
    start_hotspot: Mapped[Optional["Hotspot"]] = relationship(
        "Hotspot", foreign_keys=[start_hotspot_id], post_update=True
    )
    end_hotspot: Mapped[Optional["Hotspot"]] = relationship(
        "Hotspot", foreign_keys=[end_hotspot_id], post_update=True
    )


class Job(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_jobs"
    __table_args__ = (
        Index("ix_jobs_product_status", "product_id", "status"),
    )

    # Note: org_id not in actual database, made virtual for backward compatibility
    @property
    def org_id(self) -> Optional[uuid.UUID]:
        return self.product.org_id if hasattr(self, 'product') and self.product else None

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    image_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_assets.id")
    )
    model_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tbl_assets.id"))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    engine: Mapped[Optional[str]] = mapped_column(Text)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Virtual columns - these don't exist in actual database
    gpu_type = column_property(literal_column("NULL::text"))
    credits_used = column_property(literal_column("1::integer"))
    started_at = column_property(literal_column("NULL::timestamptz"))
    error = column_property(literal_column("'{}'::jsonb"))

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    product: Mapped[Product] = relationship("Product", back_populates="jobs")


class PublishLink(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tbl_publish_links"
    __table_args__ = (
        Index(
            "ix_publish_links_product_enabled",
            "product_id",
            postgresql_where=text("is_enabled"),
        ),
    )

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    public_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    password_hash: Mapped[Optional[str]] = mapped_column(Text)
    # iframe_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # view_count: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    # last_viewed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    product: Mapped[Product] = relationship("Product", back_populates="publish_links")


class Gallery(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tbl_galleries"
    __table_args__ = (UniqueConstraint("org_id", "slug", name="uq_gallery_org_slug"),)

    org_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    # settings column doesn't exist in DB snapshot; expose virtual empty object
    settings = column_property(literal_column("'{}'::jsonb"))
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="SET NULL")
    )
    # Virtual column - galleries table doesn't have deleted_at in database
    deleted_at = column_property(literal_column("NULL::timestamptz"))

    items: Mapped[list["GalleryItem"]] = relationship(
        "GalleryItem", back_populates="gallery", cascade="all, delete-orphan"
    )


class GalleryItem(UUIDMixin, CreatedAtMixin, Base):
    __tablename__ = "tbl_gallery_items"
    __table_args__ = (
        UniqueConstraint("gallery_id", "product_id", name="uq_gallery_product"),
        Index("ix_gallery_items_order", "gallery_id", "order_index"),
    )

    gallery_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_galleries.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    gallery: Mapped[Gallery] = relationship("Gallery", back_populates="items")
    product: Mapped[Product] = relationship("Product")


class AnalyticsEvent(Base):
    __tablename__ = "tbl_analytics_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    org_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="SET NULL")
    )
    publish_link_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_publish_links.id", ondelete="SET NULL")
    )
    session_id: Mapped[Optional[str]] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    ip_hash: Mapped[Optional[str]] = mapped_column(String)
    occurred_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), nullable=False)


Index("ix_analytics_events_org_time", AnalyticsEvent.org_id, AnalyticsEvent.occurred_at)
Index("ix_analytics_events_product_time", AnalyticsEvent.product_id, AnalyticsEvent.occurred_at)
Index(
    "ix_analytics_events_payload_gin",
    AnalyticsEvent.payload,
    postgresql_using="gin",
)


class AnalyticsDailyProduct(Base):
    __tablename__ = "tbl_analytics_daily_product"
    __table_args__ = (
        PrimaryKeyConstraint("day", "org_id", "product_id", name="pk_analytics_daily_product"),
    )

    day: Mapped[date] = mapped_column(Date, nullable=False)
    org_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    views: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    engaged: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))
    adds_from_3d: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("0"))


class AuthIdentity(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_auth_identities"
    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uq_auth_identity_provider"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[AuthProvider] = mapped_column(Enum(AuthProvider, name="auth_provider", native_enum=False), nullable=False)
    provider_user_id: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(CITEXT)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    # meta is TEXT in database, storing JSON as string
    meta: Mapped[Optional[str]] = mapped_column(Text)

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    user: Mapped[User] = relationship("User", back_populates="identities")


class EmailVerification(UUIDMixin, Base):
    __tablename__ = "tbl_email_verifications"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class PasswordReset(UUIDMixin, Base):
    __tablename__ = "tbl_password_resets"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class ActivityLog(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_activity_logs"
    __table_args__ = (Index("ix_activity_logs_org_occurred_at", "org_id", "occurred_at"),)

    actor_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="SET NULL")
    )
    org_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_organizations.id", ondelete="SET NULL")
    )
    target_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))
    action: Mapped[str] = mapped_column(Text, nullable=False)
    ip: Mapped[Optional[str]] = mapped_column(Text)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    # metadata is TEXT in database, storing JSON as string
    activity_metadata: Mapped[Optional[str]] = mapped_column("metadata", Text)

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    # Alias for backward compatibility
    @property
    def user_id(self) -> Optional[uuid.UUID]:
        return self.actor_user_id

    @user_id.setter
    def user_id(self, value: Optional[uuid.UUID]) -> None:
        self.actor_user_id = value

    @property
    def ip_address(self) -> Optional[str]:
        return self.ip

    @ip_address.setter
    def ip_address(self, value: Optional[str]) -> None:
        self.ip = value


class Notification(UUIDMixin, CreatedAtMixin, Base):
    __tablename__ = "tbl_notifications"
    __table_args__ = (
        Index(
            "ix_notifications_user_unread",
            "user_id",
            postgresql_where=text("read_at IS NULL"),
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # 'channel' column does not exist in DB; expose default as virtual
    channel = column_property(literal_column("'in_app'::text"))
    # DB stores 'data' as TEXT; services are responsible for JSON serialization
    data: Mapped[Optional[str]] = mapped_column(Text)
    read_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class UserNotificationPreference(Base):
    __tablename__ = "tbl_user_notification_prefs"

    # Table has only user_id as key-like column; model it as PK for ORM
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), primary_key=True
    )
    # Stored as TEXT in DB; service parses as JSON/CSV
    channels: Mapped[Optional[str]] = mapped_column(Text)
    muted_types: Mapped[Optional[str]] = mapped_column(Text)
    # Keep audit updated_date mapping for convenience
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        "updated_date",
        TIMESTAMP(timezone=True),
    )


# Legacy models for backwards compatibility with existing routes
class Upload(UUIDMixin, CreatedAtMixin, Base):
    __tablename__ = "uploads"

    filename: Mapped[str] = mapped_column(String, nullable=False)
    upload_url: Mapped[str] = mapped_column(Text, nullable=False)
    file_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String, nullable=False)


class AssetPart(UUIDMixin, CreatedAtMixin, Base):
    __tablename__ = "asset_parts"

    asset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_assets.id", ondelete="CASCADE"), nullable=False
    )
    part_name: Mapped[str] = mapped_column(String, nullable=False)
    storage: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'azure_blob'"))
    url: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)


# Alias for backwards compatibility
JobStatusEnum = JobStatus


class ProductAsset(AuditMixin, Base):
    __tablename__ = "tbl_product_assets"

    # Database has id as UUID with default gen_random_uuid()
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    image: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)


class ProductAssetMapping(AuditMixin, Base):
    __tablename__ = "tbl_product_asset_mapping"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    productid: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    product_asset_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False  # Changed to UUID to match tbl_product_assets.id
        # PGUUID(as_uuid=True), ForeignKey("tbl_product_assets.id", ondelete="CASCADE"), nullable=False
    )
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class AssetStatic(AuditMixin, Base):
    """Static asset reference table (tbl_asset)."""
    __tablename__ = "tbl_asset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assetid: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class CurrencyType(AuditMixin, Base):
    """Currency type reference table (tbl_currencytype)."""
    __tablename__ = "tbl_currencytype"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class BackgroundType(AuditMixin, Base):
    """Background type reference table (tbl_background_type)."""
    __tablename__ = "tbl_background_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class Background(AuditMixin, Base):
    """Background reference table (tbl_background)."""
    __tablename__ = "tbl_background"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    background_type_id: Mapped[int] = mapped_column(Integer, ForeignKey("tbl_background_type.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    image: Mapped[str] = mapped_column(Text, nullable=False)


class ProductLinkType(AuditMixin, Base):
    """Product link types table (tbl_product_link_type)."""
    __tablename__ = "tbl_product_link_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class ProductLink(AuditMixin, Base):
    """Product links table (tbl_product_links)."""
    __tablename__ = "tbl_product_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    productid: Mapped[str] = mapped_column(
        String, ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    link: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    link_type: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("tbl_product_link_type.id"), nullable=True
    )
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


# ============================================================================
# Subscription-related models and enums
# ============================================================================
# These are defined in separate files but imported here for backward
# compatibility. All existing imports from app.models.models will continue to work.
from app.models.license_assignment import LicenseAssignment
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus, SubscriptionStatus
from app.models.payment import Payment  # registers tbl_payments with SQLAlchemy metadata
class Support(AuditMixin, Base):
    """Support contact table (tbl_support)."""
    __tablename__ = "tbl_support"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fullname: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date
