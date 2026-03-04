"""Product schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing import List, Optional


# === Core Product Schemas ===


class ProductBase(BaseModel):
    """Base product fields."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    brand: Optional[str] = Field(None, max_length=100)
    accent_color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    accent_overlay: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    tags: Optional[list[str]] = Field(None, max_items=20)


class ProductCreate(ProductBase):
    """Product creation request."""

    pass


class ProductUpdate(BaseModel):
    """Product update request (all fields optional)."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    brand: Optional[str] = Field(None, max_length=100)
    accent_color: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    accent_overlay: Optional[str] = Field(None, pattern="^#[0-9A-Fa-f]{6}$")
    tags: Optional[list[str]] = Field(None, max_items=20)



class ProductLinkCreate(BaseModel):
    """Product link create/update model."""

    name: str = Field(..., min_length=1, max_length=200)
    link: str = Field(..., min_length=1)
    description: Optional[str] = Field(None, max_length=2000)


class BackgroundInput(BaseModel):
    """Background input from frontend (color or image)."""
    
    type: str = Field(..., description="Background type: 'Color' or 'Image'")
    value: str = Field(..., description="Hex color code (e.g., '#AB902B') or image URL")


class ProductDetailsUpdate(BaseModel):
    """Product details update request for insert/update API.
    
    Note: When links are provided, they are ADDED to existing links (not replaced).
    To update or delete existing links, use the dedicated endpoints:
    - POST /products/{product_id}/links - Add new links
    - PATCH /links/{link_id} - Update an existing link
    - DELETE /links/{link_id} - Delete a link
    """

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    price: Optional[float] = Field(None, ge=0)
    currency_type: Optional[int] = Field(None, description="Currency type ID (integer)")
    background: Optional[BackgroundInput] = Field(None, description="Background with type and value (Color/Image)")
    backgroundid: Optional[int] = Field(None, description="Background ID from tbl_background (deprecated, use 'background' instead)")
    links: Optional[list[ProductLinkCreate]] = Field(None, description="Links to ADD to the product (existing links are preserved)")



class ConfiguratorSettings(BaseModel):
    """3D product configurator settings."""

    materials: Optional[list[dict[str, Any]]] = None
    variants: Optional[list[dict[str, Any]]] = None
    hotspots: Optional[list[dict[str, Any]]] = None
    links: Optional[list[dict[str, Any]]] = None
    settings: Optional[dict[str, Any]] = None
    notes: Optional[str] = None


class ProductResponse(ProductBase):
    """Product response model."""

    id: str
    status: str
    ready_metric: Optional[str] = None
    processing_progress: Optional[int] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    configurator: Optional[ConfiguratorSettings] = None

    class Config:
        from_attributes = True


class ProductListResponse(BaseModel):
    """Paginated product list response."""

    items: list[ProductResponse]
    meta: dict[str, Any]


class PublishProductRequest(BaseModel):
    """Publish/unpublish request."""

    publish: bool


class PublishProductResponse(BaseModel):
    """Response after publishing/unpublishing."""

    published: bool
    published_at: Optional[datetime] = None


class ProductImageItem(BaseModel):
    """Image item in product assets response."""

    asset_id: int
    url: str
    type: str


class ProductMeshItem(BaseModel):
    """Mesh / 3D asset item in product assets response (assetid = 2)."""

    asset_id: int
    url: str


class ProductAssetsData(BaseModel):
    """Product assets data."""

    id: str
    name: str
    description: Optional[str] = None
    price: Optional[float] = None
    currency_type: Optional[int] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    mesh: list["ProductMeshItem"] = Field(default_factory=list)
    images: list[ProductImageItem] = Field(default_factory=list)
    background: Optional[dict] = None  # Background data with type
    links: Optional[list[dict]] = None  # Product links
    hotspots: list["ProductAssetsHotspot"] = Field(default_factory=list)
    model: Optional[dict] = None  # Model data including dimensions
    public_id: Optional[str] = None  # Public ID for published products


class ProductAssetsResponse(BaseModel):
    """Response containing product assets."""

    data: ProductAssetsData


class ProductStatusData(BaseModel):
    """Product status data when not ready."""

    id: str
    name: str
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProductStatusResponse(BaseModel):
    """Response containing product status."""

    data: ProductStatusData


class ProductWithPrimaryAsset(BaseModel):
    """Product with primary asset (asset_id = 1)."""

    id: str
    name: str
    status: str
    image: Optional[str] = None
    asset_type: Optional[str] = None
    asset_type_id: Optional[int] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency_type: Optional[str] = None
    background_type: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    public_id: Optional[str] = None


class ProductsByUserResponse(BaseModel):
    """Response containing list of products with primary assets for a user."""

    items: list[ProductWithPrimaryAsset]


class CurrencyTypeResponse(BaseModel):
    """Currency type response model."""

    id: int
    code: str
    name: str
    symbol: str
    description: Optional[str] = None
    isactive: bool
    created_by: Optional[str] = None
    created_date: datetime
    updated_by: Optional[str] = None
    updated_date: Optional[datetime] = None


class CurrencyTypesResponse(BaseModel):
    """Response containing list of currency types."""

    items: list[CurrencyTypeResponse]


class BackgroundTypeResponse(BaseModel):
    """Background type response model."""

    id: int
    name: str
    description: Optional[str] = None
    isactive: bool
    created_by: Optional[str] = None
    created_date: datetime
    updated_by: Optional[str] = None
    updated_date: Optional[datetime] = None


class BackgroundTypesResponse(BaseModel):
    """Response containing list of background types."""

    items: list[BackgroundTypeResponse]


class HotspotTypeResponse(BaseModel):
    """Hotspot type response model."""

    id: int
    name: str
    description: Optional[str] = None
    isactive: bool
    created_by: Optional[str] = None
    created_date: datetime
    updated_by: Optional[str] = None
    updated_date: Optional[datetime] = None


class HotspotTypesResponse(BaseModel):
    """Response containing list of hotspot types."""

    items: list[HotspotTypeResponse]


class BackgroundResponse(BaseModel):
    """Background response model."""

    id: int
    background_type_id: int
    name: str
    description: Optional[str] = None
    isactive: bool
    image: str
    created_by: Optional[str] = None
    created_date: datetime
    updated_by: Optional[str] = None
    updated_date: Optional[datetime] = None


class BackgroundsResponse(BaseModel):
    """Response containing list of backgrounds."""

    items: list[BackgroundResponse]


class ProductLinkResponse(BaseModel):
    """Product link response model."""

    id: int
    productid: str
    name: str
    link: str
    description: Optional[str] = None
    isactive: bool
    created_by: Optional[str] = None
    created_date: datetime
    updated_by: Optional[str] = None
    updated_date: Optional[datetime] = None


class ProductLinksResponse(BaseModel):
    """Response containing list of product links."""

    items: list[ProductLinkResponse]


# === Hotspot Schemas ===


class HotspotPosition(BaseModel):
    """3D position for hotspot."""

    x: float = Field(..., ge=-1.0, le=1.0)
    y: float = Field(..., ge=-1.0, le=1.0)
    z: float = Field(..., ge=-1.0, le=1.0)


class ProductAssetsHotspot(BaseModel):
    """Hotspot item included in product assets."""

    id: str
    title: str
    description: Optional[str] = None
    position: HotspotPosition
    hotspot_type: Optional[int] = None
    order_index: int


class HotspotCreate(BaseModel):
    """Create hotspot request."""

    title: str
    description: str
    position: HotspotPosition
    hotspot_type: Optional[int] = Field(
        None, description="Hotspot type ID from tbl_hotspot_type (optional)"
    )
    text_font: Optional[str] = None
    text_color: Optional[str] = None
    bg_color: Optional[str] = None
    action_type: str = "none"
    action_payload: dict[str, Any] = Field(default_factory=dict)


class HotspotResponse(BaseModel):
    """Hotspot response model."""

    id: str
    title: str
    description: str
    position: HotspotPosition
    text_font: Optional[str] = None
    text_color: Optional[str] = None
    bg_color: Optional[str] = None
    action_type: str
    action_payload: dict[str, Any]
    hotspot_type: Optional[int] = None
    order_index: int
    created_at: datetime

    class Config:
        from_attributes = True


# === Dimension Schemas ===
class DimensionPosition(BaseModel):
    """3D position for dimension hotspot"""

    x: float
    y: float
    z: float


class DimensionHotspot(BaseModel):
    """Start / End hotspot for a dimension"""

    type: str          # "start" | "end"
    title: str
    position: DimensionPosition


class DimensionItem(BaseModel):
    """Single dynamic dimension"""

    name: str          # e.g. "Seat Width", "Back Height"
    value: float
    unit: Optional[str] = "cm"
    hotspots: List[DimensionHotspot]