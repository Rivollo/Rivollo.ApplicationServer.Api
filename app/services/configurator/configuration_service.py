"""Configuration dimensions: Capacity x Layout over a product's models (ADR-017).

A configured product names its dimensions and values, and each model (the
original — id "original", no variant row — or an extra variant) takes one
value per dimension. The viewer resolves a shopper's picks to exactly one model
from those selections; it never builds a Cartesian product.

Rules, all checked BEFORE any write, so a rejected PUT changes nothing:
  1. a required dimension has at least one active value
  2. every listed model has a value for every required dimension
  3. a selection uses a value of that dimension, active, of this product
  4. no two models share a complete combination (under the product row lock,
     as ADR-012 does for material indices: it spans rows)
  5. exactly one listed model is the default
  6. every listed model has a GLB
  7-8. a Shopify mapping is exact and complete, and the default combination is
     a real Shopify variant (app/services/shopify/mapping_service.py)

The default is stored as one ``is_default`` value per dimension, written from
the model the seller marks as default; a product with no dimensions keeps the
original model as its default (ADR-014 unchanged there). Reads are live, like
every configurator table: there is no draft/publish snapshot (ADR-017).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import configurator_repository as repo
from app.models.configurator import (
    CONFIGURATION_CODE_PATTERN,
    CONFIGURATION_VALUE_CODE_PATTERN,
    ConfigurationDimension,
    ConfigurationValue,
    ModelConfigurationValue,
    ProductModelVariant,
)
from app.schemas.configurator import ConfigurationUpdateRequest
from app.services.configurator.configuration_errors import invalid
from app.services.configurator.recipe import validate_image_url_ownership
from app.services.shopify import mapping_service

logger = logging.getLogger(__name__)

ORIGINAL = "original"
ORIGINAL_NAME = "Default"
THUMBNAIL_ASSET_ID = 1
PRODUCT_NOT_FOUND = "Product not found"

_CODE = re.compile(CONFIGURATION_CODE_PATTERN)
_VALUE_CODE = re.compile(CONFIGURATION_VALUE_CODE_PATTERN)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LiveModel:
    """A model the product has now: the original, or an active extra variant."""

    id: str
    variant_id: Optional[uuid.UUID]
    name: str
    glb_url: Optional[str]
    thumbnail_url: Optional[str]


@dataclass
class LoadedConfiguration:
    dimensions: list[ConfigurationDimension] = field(default_factory=list)
    # model id ("original" or str(variant id)) -> {dimension code: value code}
    selections: dict[str, dict[str, str]] = field(default_factory=dict)
    # dimension code -> default value code
    default_selection: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.dimensions)

    def default_model_id(self, live_ids) -> Optional[str]:
        for model_id, selection in self.selections.items():
            if model_id in live_ids and selection == self.default_selection:
                return model_id
        return None


class ConfigurationService:
    @staticmethod
    def enabled() -> bool:
        return settings.ENABLE_CONFIGURATION_DIMENSIONS and settings.ENABLE_MODEL_VARIANTS

    @staticmethod
    def require_enabled() -> None:
        """404 while the feature is off, so the routes look absent."""
        if not ConfigurationService.enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    @staticmethod
    async def load(db: AsyncSession, product_id: uuid.UUID) -> LoadedConfiguration:
        """The stored configuration. Empty (unconfigured) while the feature is off."""
        if not ConfigurationService.enabled():
            return LoadedConfiguration()
        dimensions = await repo.get_dimensions(db, product_id)
        if not dimensions:
            return LoadedConfiguration()
        values = {v.id: (d.code, v.code) for d in dimensions for v in d.values}
        selections: dict[str, dict[str, str]] = {}
        for row in await repo.get_model_configuration_values(db, product_id):
            pair = values.get(row.value_id)
            if pair is None:
                continue
            model_id = str(row.model_variant_id) if row.model_variant_id else ORIGINAL
            selections.setdefault(model_id, {})[pair[0]] = pair[1]
        default = {d.code: v.code for d in dimensions for v in d.values if v.is_default}
        return LoadedConfiguration(dimensions=dimensions, selections=selections, default_selection=default)

    @staticmethod
    async def live_models(db: AsyncSession, product_id: uuid.UUID) -> list[LiveModel]:
        """The original first, then the active extra variants, as every reader orders them."""
        mesh = await repo.get_product_mesh_asset(db, product_id)
        thumbnail = await repo.get_product_asset(db, product_id, THUMBNAIL_ASSET_ID)
        models = [
            LiveModel(
                id=ORIGINAL,
                variant_id=None,
                name=ORIGINAL_NAME,
                glb_url=mesh.image if mesh is not None and mesh.image else None,
                thumbnail_url=thumbnail.image if thumbnail is not None else None,
            )
        ]
        variants = await repo.get_model_variants(db, product_id)
        assets = await repo.get_assets_by_ids(db, [v.glb_asset_id for v in variants])
        for v in variants:
            glb = assets.get(v.glb_asset_id)
            models.append(
                LiveModel(
                    id=str(v.id),
                    variant_id=v.id,
                    name=v.name,
                    glb_url=glb.image if glb is not None and glb.image else None,
                    thumbnail_url=v.thumbnail_url,
                )
            )
        return models

    @staticmethod
    async def get(db: AsyncSession, product_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, Any]:
        ConfigurationService.require_enabled()
        if await repo.get_owned_product(db, product_id, user_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)
        shopify_product = await mapping_service.linked_product(db, product_id, user_id)
        return await ConfigurationService._view(db, product_id, shopify_product)

    @staticmethod
    async def _view(db: AsyncSession, product_id: uuid.UUID, shopify_product) -> dict[str, Any]:
        config = await ConfigurationService.load(db, product_id)
        models = await ConfigurationService.live_models(db, product_id)
        live_ids = {m.id for m in models}
        default_id = config.default_model_id(live_ids) if config.configured else ORIGINAL
        return {
            "dimensions": [
                {
                    "id": d.id,
                    "code": d.code,
                    "label": d.label,
                    "display_type": d.display_type,
                    "order_index": d.order_index,
                    "is_required": d.is_required,
                    "values": [
                        {
                            "id": v.id,
                            "code": v.code,
                            "label": v.label,
                            "order_index": v.order_index,
                            "thumbnail_url": v.thumbnail_url,
                            "is_active": v.isactive,
                            "is_default": v.is_default,
                        }
                        for v in d.values
                    ],
                }
                for d in config.dimensions
            ],
            "variants": [
                {
                    "id": m.id,
                    "name": m.name,
                    "glb_url": m.glb_url,
                    "thumbnail_url": m.thumbnail_url,
                    "is_default": m.id == default_id,
                    "selections": config.selections.get(m.id),
                }
                for m in models
            ],
            "shopify_linked": shopify_product is not None,
            "shopify_mapping": shopify_product.dimension_mapping if shopify_product is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Write: the whole configuration, all or nothing
    # ------------------------------------------------------------------ #
    @staticmethod
    async def update(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: ConfigurationUpdateRequest,
    ) -> dict[str, Any]:
        ConfigurationService.require_enabled()
        # Ownership and the lock in one query: concurrent PUTs serialise here.
        product = await repo.get_owned_product(db, product_id, user_id, for_update=True)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)

        models = {m.id: m for m in await ConfigurationService.live_models(db, product_id)}
        plan = ConfigurationService._plan(payload, models, user_id)

        shopify_product = await mapping_service.linked_product(db, product_id, user_id)
        mapping_given = "shopify_mapping" in payload.model_fields_set
        if mapping_given:
            mapping = (
                {k: v.model_dump() for k, v in payload.shopify_mapping.items()}
                if payload.shopify_mapping
                else None
            )
        else:
            # Kept, but re-checked against the new configuration.
            mapping = shopify_product.dimension_mapping if shopify_product is not None else None
        if not plan.dimensions:
            if mapping and mapping_given:
                raise invalid(
                    "SHOPIFY_UNKNOWN_DIMENSION",
                    "A Shopify mapping needs configuration dimensions.",
                    ["shopify_mapping"],
                )
            mapping = None
        if mapping:
            if shopify_product is None:
                raise invalid(
                    "SHOPIFY_NOT_LINKED",
                    "This product is not linked to a Shopify product.",
                    ["shopify_mapping"],
                )
            mapping_service.validate(
                mapping,
                dimensions={d.code: {v.code: v.label for v in d.values} for d in plan.dimensions},
                used_values=plan.used_values(),
                default_selection=plan.default_selection(),
                product=shopify_product,
            )

        try:
            await ConfigurationService._apply(db, product_id, user_id, plan)
            if shopify_product is not None and (mapping_given or mapping != shopify_product.dimension_mapping):
                shopify_product.dimension_mapping = mapping
            await db.commit()
        except IntegrityError:
            await db.rollback()
            logger.warning("Configuration write for product %s hit a constraint", product_id, exc_info=True)
            raise invalid(
                "CONFIGURATION_CONFLICT",
                "The configuration changed while it was being saved. Reload and try again.",
                status_code=status.HTTP_409_CONFLICT,
            )
        # expire_on_commit is off: the rows in the session are what was written.
        return await ConfigurationService._view(db, product_id, shopify_product)

    @staticmethod
    def _plan(
        payload: ConfigurationUpdateRequest, models: dict[str, LiveModel], user_id: uuid.UUID
    ) -> "_Plan":
        dims = payload.dimensions
        if len(dims) > settings.CONFIGURATION_MAX_DIMENSIONS:
            raise invalid(
                "TOO_MANY_DIMENSIONS",
                f"A product can have at most {settings.CONFIGURATION_MAX_DIMENSIONS} dimensions.",
                ["dimensions"],
            )
        dim_index: dict[str, Any] = {}
        for i, dim in enumerate(dims):
            path = f"dimensions[{i}]"
            if not _CODE.match(dim.code):
                raise invalid(
                    "INVALID_CODE",
                    f"'{dim.code}' is not a valid code: use lowercase letters, digits and _, starting with a letter.",
                    [f"{path}.code"],
                )
            if dim.code in dim_index:
                raise invalid("DUPLICATE_DIMENSION", f"Dimension '{dim.code}' is listed twice.", [f"{path}.code"])
            dim_index[dim.code] = dim
            if len(dim.values) > settings.CONFIGURATION_MAX_VALUES:
                raise invalid(
                    "TOO_MANY_VALUES",
                    f"Dimension '{dim.label}' has more than {settings.CONFIGURATION_MAX_VALUES} values.",
                    [f"{path}.values"],
                )
            codes: set[str] = set()
            labels: set[str] = set()
            for j, value in enumerate(dim.values):
                vpath = f"{path}.values[{j}]"
                if not _VALUE_CODE.match(value.code):
                    raise invalid(
                        "INVALID_CODE",
                        f"'{value.code}' is not a valid value code: use lowercase letters, digits and _.",
                        [f"{vpath}.code"],
                    )
                if value.code in codes:
                    raise invalid("DUPLICATE_VALUE", f"Value '{value.code}' is listed twice in '{dim.label}'.", [f"{vpath}.code"])
                if value.label.strip().lower() in labels:
                    raise invalid("DUPLICATE_VALUE", f"Label '{value.label}' is used twice in '{dim.label}'.", [f"{vpath}.label"])
                codes.add(value.code)
                labels.add(value.label.strip().lower())
                if value.thumbnail_url:
                    try:
                        validate_image_url_ownership(value.thumbnail_url, user_id)
                    except HTTPException:
                        raise invalid(
                            "INVALID_THUMBNAIL_URL",
                            "A value's thumbnail must be an image you uploaded via POST /uploads/content.",
                            [f"{vpath}.thumbnail_url"],
                        )
            if dim.is_required and not any(v.is_active for v in dim.values):
                raise invalid(
                    "DIMENSION_WITHOUT_VALUES",
                    f"Required dimension '{dim.label}' has no active values.",
                    [f"{path}.values"],
                )

        variants = payload.variants
        if not dims:
            if variants:
                raise invalid(
                    "VARIANTS_WITHOUT_DIMENSIONS",
                    "Models can only be assigned once the product has dimensions.",
                    ["variants"],
                )
            return _Plan(dimensions=[], selections={}, default_id=None)
        if not variants:
            raise invalid("NO_MODELS", "Assign at least one model to the configuration.", ["variants"])
        if len(variants) > settings.CONFIGURATION_MAX_MODELS:
            raise invalid(
                "TOO_MANY_MODELS",
                f"A configuration can have at most {settings.CONFIGURATION_MAX_MODELS} models.",
                ["variants"],
            )

        selections: dict[str, dict[str, str]] = {}
        combos: dict[tuple, str] = {}
        defaults: list[str] = []
        for i, model in enumerate(variants):
            path = f"variants.{model.id}"
            live = models.get(model.id)
            if live is None:
                raise invalid("UNKNOWN_MODEL", f"'{model.id}' is not a model of this product.", [f"variants[{i}].id"])
            if model.id in selections:
                raise invalid("DUPLICATE_MODEL", f"Model '{live.name}' is listed twice.", [f"variants[{i}].id"])
            if not live.glb_url:
                raise invalid("MODEL_WITHOUT_GLB", f"Model '{live.name}' has no 3D model file.", [f"{path}.id"])
            for dim_code, value_code in model.selections.items():
                dim = dim_index.get(dim_code)
                if dim is None:
                    raise invalid("UNKNOWN_DIMENSION", f"'{dim_code}' is not a dimension of this configuration.", [f"{path}.selections.{dim_code}"])
                value = next((v for v in dim.values if v.code == value_code), None)
                if value is None:
                    raise invalid("UNKNOWN_VALUE", f"'{value_code}' is not a value of '{dim.label}'.", [f"{path}.selections.{dim_code}"])
                if not value.is_active:
                    raise invalid("INACTIVE_VALUE", f"'{value.label}' is inactive and cannot be assigned.", [f"{path}.selections.{dim_code}"])
            for dim in dims:
                if dim.is_required and dim.code not in model.selections:
                    raise invalid(
                        "MISSING_SELECTION",
                        f"Model '{live.name}' has no {dim.label}.",
                        [f"{path}.selections.{dim.code}"],
                    )
            key = tuple(sorted(model.selections.items()))
            if key in combos:
                other = combos[key]
                described = " and ".join(
                    f"{dim_index[c].label}={next(v.label for v in dim_index[c].values if v.code == vc)}"
                    for c, vc in sorted(model.selections.items(), key=lambda kv: dim_index[kv[0]].order_index)
                )
                raise invalid(
                    "DUPLICATE_VARIANT_SELECTION",
                    f"Two models use {described}.",
                    [f"variants.{other}.selections", f"{path}.selections"],
                )
            combos[key] = model.id
            selections[model.id] = dict(model.selections)
            if model.is_default:
                defaults.append(model.id)

        if not defaults:
            raise invalid("DEFAULT_REQUIRED", "Mark one model as the default.", ["variants"])
        if len(defaults) > 1:
            raise invalid(
                "MULTIPLE_DEFAULTS",
                "Only one model can be the default.",
                [f"variants.{d}.is_default" for d in defaults],
            )
        return _Plan(dimensions=list(dims), selections=selections, default_id=defaults[0])

    @staticmethod
    async def _apply(db: AsyncSession, product_id: uuid.UUID, user_id: uuid.UUID, plan: "_Plan") -> None:
        """Make the stored configuration equal the plan. Dimensions and values keep
        their ids when their code is kept. Flushes in phases so the unique
        indexes (one default per dimension, case-insensitive labels) never see a
        half-written state."""
        now = _utcnow()
        existing = {d.code: d for d in await repo.get_dimensions(db, product_id)}
        wanted = {d.code: d for d in plan.dimensions}
        default_selection = plan.default_selection()

        # Phase 1: clear assignments and defaults, retire what goes, park changing labels.
        await repo.delete_model_configuration_values(db, product_id)
        for code, row in existing.items():
            if code not in wanted:
                await repo.delete(db, row)
                continue
            spec_values = {v.code: v for v in wanted[code].values}
            for value in list(row.values):
                value.is_default = False
                spec = spec_values.get(value.code)
                if spec is None:
                    row.values.remove(value)  # delete-orphan
                elif spec.label != value.label:
                    value.label = f"~{value.id}"
        await repo.flush(db)

        # Phase 2: upsert dimensions and values to their final state.
        rows: dict[str, ConfigurationDimension] = {}
        value_ids: dict[tuple[str, str], uuid.UUID] = {}
        for spec in plan.dimensions:
            row = existing.get(spec.code)
            if row is None:
                row = ConfigurationDimension(
                    id=uuid.uuid4(), product_id=product_id, code=spec.code, created_by=user_id, created_date=now
                )
                row.values = []
                repo.add(db, row)
            else:
                row.updated_by, row.updated_date = user_id, now
            row.label = spec.label
            row.display_type = spec.display_type
            row.order_index = spec.order_index
            row.is_required = spec.is_required
            current = {v.code: v for v in row.values}
            for vspec in spec.values:
                value = current.get(vspec.code)
                if value is None:
                    value = ConfigurationValue(
                        id=uuid.uuid4(), code=vspec.code, created_by=user_id, created_date=now
                    )
                    row.values.append(value)
                else:
                    value.updated_by, value.updated_date = user_id, now
                value.label = vspec.label
                value.order_index = vspec.order_index
                value.thumbnail_url = vspec.thumbnail_url
                value.isactive = vspec.is_active
                value.is_default = False
                value_ids[(spec.code, vspec.code)] = value.id
            rows[spec.code] = row
        await repo.flush(db)

        # Phase 3: the default values and the model assignments.
        for code, value_code in default_selection.items():
            for value in rows[code].values:
                if value.code == value_code:
                    value.is_default = True
        for model_id, selection in plan.selections.items():
            variant_id = None if model_id == ORIGINAL else uuid.UUID(model_id)
            for dim_code, value_code in selection.items():
                repo.add(
                    db,
                    ModelConfigurationValue(
                        id=uuid.uuid4(),
                        product_id=product_id,
                        model_variant_id=variant_id,
                        dimension_id=rows[dim_code].id,
                        value_id=value_ids[(dim_code, value_code)],
                        created_by=user_id,
                        created_date=now,
                    ),
                )
        await repo.flush(db)

    # ------------------------------------------------------------------ #
    # Hooks for the rest of the configurator
    # ------------------------------------------------------------------ #
    @staticmethod
    async def release_variant(db: AsyncSession, variant: ProductModelVariant) -> None:
        """Before a variant is soft-deleted: refuse if it is the configured default
        (the configuration would have none), else drop its assignments. No commit."""
        if not ConfigurationService.enabled():
            return
        config = await ConfigurationService.load(db, variant.product_id)
        if not config.configured:
            return
        if config.selections.get(str(variant.id)) == config.default_selection:
            raise invalid(
                "MODEL_IS_DEFAULT",
                "This model is the configuration's default. Choose another default first.",
                status_code=status.HTTP_409_CONFLICT,
            )
        await repo.delete_model_configuration_values(db, variant.product_id, model_variant_id=variant.id)


@dataclass
class _Plan:
    dimensions: list  # list[ConfigurationDimensionIn]
    selections: dict[str, dict[str, str]]
    default_id: Optional[str]

    def default_selection(self) -> dict[str, str]:
        return dict(self.selections.get(self.default_id, {})) if self.default_id else {}

    def used_values(self) -> dict[str, set[str]]:
        used: dict[str, set[str]] = {}
        for selection in self.selections.values():
            for code, value in selection.items():
                used.setdefault(code, set()).add(value)
        return used


configuration_service = ConfigurationService()
