"""Product Configurator repository — persistence only.

Follows the repository shape every other domain in this project uses: stateless
``@staticmethod``s taking the session first, no commits, a module-level
singleton at the bottom. Transaction boundaries belong to the service layer, so
nothing here calls ``commit()``.

OWNERSHIP IS RESOLVED HERE, NOT ASSERTED BY THE CALLER
------------------------------------------------------
Every lookup that can reach a seller's data takes ``user_id`` and resolves it
through the owning product:

    option -> part -> product.created_by

A client-supplied ``product_id`` is never trusted when operating on an existing
part or option — the parent is walked from the child's own foreign keys. That is
why there is no ``get_part_by_id(part_id)`` without a user: an unscoped getter
is exactly the shape that leaks into a route and becomes a cross-tenant read.

The neighbouring hotspot and colour-variant repositories deliberately are NOT
the model here. Their ``get_product_by_id`` is a bare ``db.get(Product, id)``
with no owner filter, and their services check only that the product exists —
so any authenticated user can read and modify any product's hotspots and
colourways. See ADR-008.

Reference: docs/configurator/data-model.md, docs/configurator/api-spec.md.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.models import (
    PartOption,
    PartOptionTexture,
    Product,
    ProductAsset,
    ProductAssetMapping,
    ProductPart,
)

# A product's uploaded GLB is stored in tbl_product_assets with asset_id 9.
#
# Duplicated from app/database/color_variant_repo.py:27 on purpose: extracting a
# shared constant means editing colour-variant code, which is out of scope. The
# planned extraction is recorded in architecture.md section 9 — when it happens,
# both modules should import one definition.
MESH_ASSET_ID = 9
# USDZ, for the shopper payload's AR model URL.
USDZ_ASSET_ID = 11


class ConfiguratorRepository:
    """Data access for parts, options and baked option textures."""

    # ------------------------------------------------------------------ #
    # Product / ownership
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_owned_product(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> Optional[Product]:
        """The product, only if this user owns it and it is not soft-deleted.

        ``for_update`` takes a row lock, which is how the material-index overlap
        rule is made safe against two concurrent writers without a database
        constraint (ADR-012). The ownership check and the lock are the same
        query, so guarding a write costs no extra round trip.

        Returns None rather than raising: the service turns that into 404, never
        403 — a 403 confirms the product exists and belongs to someone else,
        which is an enumeration oracle over every seller's catalogue.
        """
        stmt = select(Product).where(
            Product.id == product_id,
            Product.created_by == user_id,
            Product.deleted_at.is_(None),
        )
        if for_update:
            stmt = stmt.with_for_update()
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_published_product(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[Product]:
        """A product a shopper is allowed to see: published and not deleted.

        No owner filter — this backs the public endpoint, whose caller is a
        shopper, not a seller. The published check is what stops a draft
        product's configuration leaking (api-spec.md section 9).
        """
        result = await db.execute(
            select(Product).where(
                Product.id == product_id,
                Product.status == "published",
                Product.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_product_asset(
        db: AsyncSession,
        product_id: uuid.UUID,
        asset_id: int,
    ) -> Optional[ProductAsset]:
        """Newest active product asset of one format, or None.

        asset_id is the FORMAT id: 9 = GLB, 11 = USDZ, 17 = Draco glTF zip.
        """
        result = await db.execute(
            select(ProductAsset)
            .join(
                ProductAssetMapping,
                ProductAsset.id == ProductAssetMapping.product_asset_id,
            )
            .where(
                ProductAssetMapping.productid == product_id,
                ProductAsset.asset_id == asset_id,
                ProductAssetMapping.isactive.is_(True),
            )
            .order_by(ProductAssetMapping.created_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_product_mesh_asset(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[ProductAsset]:
        """The product's current source GLB asset row, or None.

        Resolved through tbl_product_assets / tbl_product_asset_mapping the way
        the rest of the app does it, newest active mesh first — so a re-uploaded
        model wins over the one it replaced.

        Returns the ROW, not just the URL, because the Configurator needs both:
        ``image`` is the URL to inspect, and ``id`` is the Phase-1 ``glb_version``
        value (ADR-006 stores it prefixed, ``asset:<uuid>``).
        """
        result = await db.execute(
            select(ProductAsset)
            .join(
                ProductAssetMapping,
                ProductAsset.id == ProductAssetMapping.product_asset_id,
            )
            .where(
                ProductAssetMapping.productid == product_id,
                ProductAsset.asset_id == MESH_ASSET_ID,
                ProductAssetMapping.isactive.is_(True),
            )
            .order_by(ProductAssetMapping.created_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------ #
    # Parts — read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_parts_for_product(
        db: AsyncSession,
        product_id: uuid.UUID,
        *,
        active_only: bool = False,
    ) -> list[ProductPart]:
        """All parts of a product in display order, options eagerly loaded.

        Caller must already have established ownership of ``product_id``.
        """
        stmt = select(ProductPart).where(ProductPart.product_id == product_id)
        if active_only:
            stmt = stmt.where(ProductPart.isactive.is_(True))
        stmt = stmt.order_by(ProductPart.order_index.asc(), ProductPart.created_date.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_owned_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[ProductPart]:
        """A part, resolved to its owner through its own product_id.

        The product is joined from the part's foreign key, never from anything
        the client sent.
        """
        result = await db.execute(
            select(ProductPart)
            .join(Product, Product.id == ProductPart.product_id)
            .where(
                ProductPart.id == part_id,
                Product.created_by == user_id,
                Product.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_sibling_parts(
        db: AsyncSession,
        product_id: uuid.UUID,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> list[ProductPart]:
        """Active parts of a product, for the material-index overlap check.

        Must be called inside the transaction that holds the product row lock
        from ``get_owned_product(for_update=True)``, or the check races.
        """
        stmt = select(ProductPart).where(
            ProductPart.product_id == product_id,
            ProductPart.isactive.is_(True),
        )
        if exclude_id is not None:
            stmt = stmt.where(ProductPart.id != exclude_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def part_slug_exists(
        db: AsyncSession,
        product_id: uuid.UUID,
        slug: str,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> bool:
        stmt = select(ProductPart.id).where(
            ProductPart.product_id == product_id,
            ProductPart.slug == slug,
        )
        if exclude_id is not None:
            stmt = stmt.where(ProductPart.id != exclude_id)
        result = await db.execute(stmt.limit(1))
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def get_next_part_order_index(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> int:
        result = await db.execute(
            select(func.max(ProductPart.order_index)).where(
                ProductPart.product_id == product_id
            )
        )
        current_max = result.scalar()
        return 0 if current_max is None else current_max + 1

    # ------------------------------------------------------------------ #
    # Options — read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_options_for_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        *,
        active_only: bool = False,
    ) -> list[PartOption]:
        stmt = select(PartOption).where(PartOption.part_id == part_id)
        if active_only:
            stmt = stmt.where(PartOption.isactive.is_(True))
        stmt = stmt.order_by(PartOption.order_index.asc(), PartOption.created_date.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_owned_option(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[PartOption]:
        """An option, resolved to its owner through option -> part -> product.

        Two joins rather than a trusted product_id: this is the whole chain the
        security rules require, walked from the option's own foreign keys.
        """
        result = await db.execute(
            select(PartOption)
            .join(ProductPart, ProductPart.id == PartOption.part_id)
            .join(Product, Product.id == ProductPart.product_id)
            .where(
                PartOption.id == option_id,
                Product.created_by == user_id,
                Product.deleted_at.is_(None),
            )
            .options(selectinload(PartOption.part))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_option_for_bake(
        db: AsyncSession,
        option_id: uuid.UUID,
    ) -> Optional[PartOption]:
        """An option by id, WITHOUT an ownership check. Internal to baking only.

        🔴 Never call this from a request handler. Every other getter in this
        module takes a ``user_id`` on purpose — an unscoped lookup is exactly the
        shape that leaks into a route and becomes a cross-tenant read (see the
        module docstring).

        This one exists because the bake pipeline runs DETACHED from the request
        that authorised it: by the time a bake finishes there is no current user,
        the seller's token may have expired, and the result must still be
        recorded. Ownership was established by ``BakeService.request_bake``
        before the work was ever enqueued.

        Loads ``part`` and ``textures`` because every caller needs both to decide
        what is stale.
        """
        result = await db.execute(
            select(PartOption)
            .where(PartOption.id == option_id)
            .options(selectinload(PartOption.part), selectinload(PartOption.textures))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def option_slug_exists(
        db: AsyncSession,
        part_id: uuid.UUID,
        slug: str,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> bool:
        stmt = select(PartOption.id).where(
            PartOption.part_id == part_id,
            PartOption.slug == slug,
        )
        if exclude_id is not None:
            stmt = stmt.where(PartOption.id != exclude_id)
        result = await db.execute(stmt.limit(1))
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def get_next_option_order_index(
        db: AsyncSession,
        part_id: uuid.UUID,
    ) -> int:
        result = await db.execute(
            select(func.max(PartOption.order_index)).where(PartOption.part_id == part_id)
        )
        current_max = result.scalar()
        return 0 if current_max is None else current_max + 1

    @staticmethod
    async def count_options_for_part(db: AsyncSession, part_id: uuid.UUID) -> int:
        result = await db.execute(
            select(func.count()).select_from(PartOption).where(PartOption.part_id == part_id)
        )
        return int(result.scalar() or 0)

    @staticmethod
    async def get_default_option(
        db: AsyncSession,
        part_id: uuid.UUID,
    ) -> Optional[PartOption]:
        """The part's default option, if it has one.

        At most one row can match — ux_part_options_one_default is a partial
        unique index, so this is a database guarantee rather than a convention.
        """
        result = await db.execute(
            select(PartOption).where(
                PartOption.part_id == part_id,
                PartOption.is_default.is_(True),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_default_promotion_candidate(
        db: AsyncSession,
        part_id: uuid.UUID,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> Optional[PartOption]:
        """The option that should become default when the current one goes.

        Lowest order_index among options that are active AND completed — an
        option with no baked texture cannot be the look a shopper loads first.
        """
        stmt = select(PartOption).where(
            PartOption.part_id == part_id,
            PartOption.isactive.is_(True),
            PartOption.bake_status == "completed",
        )
        if exclude_id is not None:
            stmt = stmt.where(PartOption.id != exclude_id)
        result = await db.execute(
            stmt.order_by(PartOption.order_index.asc(), PartOption.created_date.asc()).limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def clear_default(
        db: AsyncSession,
        part_id: uuid.UUID,
        *,
        keep_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Unset is_default across a part, optionally sparing one row.

        Issued as a bulk UPDATE so the partial unique index is never transiently
        violated by a flush ordering that sets the new default before clearing
        the old one.
        """
        stmt = (
            update(PartOption)
            .where(PartOption.part_id == part_id, PartOption.is_default.is_(True))
            .values(is_default=False)
        )
        if keep_id is not None:
            stmt = stmt.where(PartOption.id != keep_id)
        await db.execute(stmt)

    # ------------------------------------------------------------------ #
    # Option textures
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_stale_baking_options(
        db: AsyncSession,
        older_than: datetime,
        limit: int,
    ) -> list[PartOption]:
        """Options stuck in `baking` since before ``older_than``.

        Served by the partial index ix_options_stale, so this scans only the
        handful of rows actually in flight rather than the whole table.
        """
        result = await db.execute(
            select(PartOption)
            .where(
                PartOption.bake_status == "baking",
                PartOption.bake_started_at.is_not(None),
                PartOption.bake_started_at < older_than,
            )
            .order_by(PartOption.bake_started_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def claim_stale_bake(
        db: AsyncSession,
        option_id: uuid.UUID,
        recipe_hash: str,
        older_than: datetime,
    ) -> bool:
        """Atomically move one lost bake back to `pending`. False if someone else did.

        A single guarded UPDATE, not read-then-write: two application replicas
        sweeping at the same moment must not both re-enqueue the same option. The
        WHERE clause repeats every condition the caller checked, so only one
        statement can win.

        ``recipe_hash`` is in the predicate as well — if the recipe changed while
        the row sat stuck, a newer bake already owns it and this must not touch it.
        """
        result = await db.execute(
            update(PartOption)
            .where(
                PartOption.id == option_id,
                PartOption.recipe_hash == recipe_hash,
                PartOption.bake_status == "baking",
                PartOption.bake_started_at.is_not(None),
                PartOption.bake_started_at < older_than,
            )
            .values(
                bake_status="pending",
                bake_started_at=None,
                bake_completed_at=None,
                bake_error=None,
            )
        )
        return bool(result.rowcount)

    @staticmethod
    async def get_textures_for_option(
        db: AsyncSession,
        option_id: uuid.UUID,
    ) -> list[PartOptionTexture]:
        result = await db.execute(
            select(PartOptionTexture)
            .where(PartOptionTexture.option_id == option_id)
            .order_by(PartOptionTexture.material_index.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_texture_for_material(
        db: AsyncSession,
        option_id: uuid.UUID,
        material_index: int,
    ) -> Optional[PartOptionTexture]:
        """The one texture for this option and material, if it exists.

        At most one row — uq_option_texture_material enforces it.
        """
        result = await db.execute(
            select(PartOptionTexture).where(
                PartOptionTexture.option_id == option_id,
                PartOptionTexture.material_index == material_index,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def count_current_textures(
        db: AsyncSession,
        option_id: uuid.UUID,
        recipe_hash: str,
    ) -> int:
        """Textures baked from the option's CURRENT recipe.

        A row whose hash differs was baked from a superseded recipe and must not
        be counted or served. Backs the `textures_done` half of bake progress.
        """
        result = await db.execute(
            select(func.count())
            .select_from(PartOptionTexture)
            .where(
                PartOptionTexture.option_id == option_id,
                PartOptionTexture.recipe_hash == recipe_hash,
            )
        )
        return int(result.scalar() or 0)

    # ------------------------------------------------------------------ #
    # Write — no commits; the service owns the transaction
    # ------------------------------------------------------------------ #
    @staticmethod
    def add(db: AsyncSession, instance: object) -> None:
        db.add(instance)

    @staticmethod
    def add_all(db: AsyncSession, instances: Sequence[object]) -> None:
        db.add_all(list(instances))

    @staticmethod
    async def delete(db: AsyncSession, instance: object) -> None:
        """Delete one row. Cascades to children via ON DELETE CASCADE.

        The database does not reach Azure Blob Storage: purging an option's or
        part's texture blobs is the service's job and must happen BEFORE the
        rows go, or the URLs needed to find them are gone.
        """
        await db.delete(instance)

    @staticmethod
    async def set_part_order(
        db: AsyncSession,
        part_id: uuid.UUID,
        order_index: int,
    ) -> None:
        await db.execute(
            update(ProductPart)
            .where(ProductPart.id == part_id)
            .values(order_index=order_index)
        )

    @staticmethod
    async def set_option_order(
        db: AsyncSession,
        option_id: uuid.UUID,
        order_index: int,
    ) -> None:
        await db.execute(
            update(PartOption)
            .where(PartOption.id == option_id)
            .values(order_index=order_index)
        )


configurator_repository = ConfiguratorRepository()
