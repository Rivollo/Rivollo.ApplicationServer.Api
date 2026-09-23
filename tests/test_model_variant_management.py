"""Model variant management and variant-scoped parts (ADR-014, step d).

Service rules against fakes — no database. Pinned here:

  * the original model is always first and always the default; it is not a
    row and cannot be renamed, reordered or deleted
  * every route resolves ownership to 404, never 403, including a variant of
    ANOTHER product of the same seller
  * parts are scoped to one model: overlap is checked only within that model,
    slugs stay unique per product, and product-level routes still mean the
    original model
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models.configurator import ProductPart
from app.schemas.configurator import ProductPartCreate
from app.services.configurator import model_variant_service as mv_module
from app.services.configurator import part_service as part_module
from app.services.configurator.material_service import MeshContext
from app.services.configurator.model_variant_service import ModelVariantService, UploadedFile
from app.services.configurator.part_service import PartService
from tests.test_model_variant_service import FakeSession, FakeStorage

OWNER = uuid.uuid4()
STRANGER = uuid.uuid4()
PRODUCT = uuid.uuid4()
OTHER_PRODUCT = uuid.uuid4()


def make_variant(**kw):
    defaults = dict(
        id=uuid.uuid4(),
        product_id=PRODUCT,
        name="Corner",
        glb_asset_id=uuid.uuid4(),
        usdz_asset_id=None,
        thumbnail_url=None,
        thumbnail_blob_url=None,
        order_index=1,
        isactive=True,
        compression_status="compressed",
        compression_error=None,
        original_size_bytes=100,
        compressed_size_bytes=40,
        width_m=2.0, depth_m=1.0, height_m=0.8,
        created_date=datetime.now(timezone.utc),
        updated_by=None,
        updated_date=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class Repo:
    """Enforces ownership the way the real WHERE clauses do."""

    def __init__(self):
        self.products = {PRODUCT: OWNER, OTHER_PRODUCT: OWNER}
        self.variants = {}
        self.parts = []
        self.added = []
        self.assets = {}
        self.mapped = {
            9: SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/model.glb"),
            11: SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/model.usdz"),
            1: SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/thumb.png"),
        }

    def add_variant(self, **kw):
        v = make_variant(**kw)
        self.variants[v.id] = v
        self.assets[v.glb_asset_id] = SimpleNamespace(id=v.glb_asset_id, image=f"https://cdn/{v.id}.glb")
        return v

    async def get_owned_product(self, db, product_id, user_id, *, for_update=False):
        if self.products.get(product_id) != user_id:
            return None
        return SimpleNamespace(id=product_id, created_by=user_id)

    async def get_owned_model_variant(self, db, variant_id, user_id):
        v = self.variants.get(variant_id)
        if v is None or not v.isactive or self.products.get(v.product_id) != user_id:
            return None
        return v

    async def get_model_variants(self, db, product_id):
        return sorted(
            (v for v in self.variants.values() if v.product_id == product_id and v.isactive),
            key=lambda v: v.order_index,
        )

    async def get_assets_by_ids(self, db, ids):
        return {i: self.assets[i] for i in ids if i in self.assets}

    async def get_product_mesh_asset(self, db, product_id):
        return self.mapped[9]

    async def get_product_asset(self, db, product_id, asset_id):
        return self.mapped.get(asset_id)

    async def get_model_mesh_asset(self, db, product_id, variant_id):
        if variant_id is None:
            return self.mapped[9]
        v = self.variants.get(variant_id)
        if v is None or not v.isactive or v.product_id != product_id:
            return None
        return self.assets[v.glb_asset_id]

    async def get_parts_for_product(self, db, product_id, *, variant_id=None, active_only=False):
        return [p for p in self.parts if p.product_id == product_id and p.variant_id == variant_id]

    async def get_sibling_parts(self, db, product_id, *, variant_id=None, exclude_id=None):
        return [
            p for p in self.parts
            if p.product_id == product_id and p.variant_id == variant_id
            and p.isactive and p.id != exclude_id
        ]

    async def part_slug_exists(self, db, product_id, slug, *, exclude_id=None):
        return any(p.product_id == product_id and p.slug == slug and p.id != exclude_id for p in self.parts)

    async def get_next_part_order_index(self, db, product_id):
        return len(self.parts)

    def add(self, db, instance):
        if getattr(instance, "id", None) is None:
            instance.id = uuid.uuid4()
        self.added.append(instance)
        if isinstance(instance, ProductPart):
            self.parts.append(instance)


@pytest.fixture
def env(monkeypatch):
    repo, storage = Repo(), FakeStorage()
    monkeypatch.setattr(mv_module, "repo", repo)
    monkeypatch.setattr(mv_module, "storage_service", storage)
    monkeypatch.setattr(part_module, "repo", repo)
    monkeypatch.setattr(mv_module.settings, "ENABLE_MODEL_VARIANTS", True)

    async def _mesh(_db, _pid, variant_id=None):
        # Each model has its own GLB, so its own glb_version.
        asset = await repo.get_model_mesh_asset(None, _pid, variant_id)
        if asset is None:
            raise HTTPException(status_code=400, detail="no model")
        return MeshContext(
            glb_version=f"asset:{asset.id}", model_url=asset.image,
            material_count=8, suggested_methods={i: "luminance" for i in range(8)},
        )

    monkeypatch.setattr(part_module.material_service, "get_mesh_context", _mesh)
    return SimpleNamespace(repo=repo, storage=storage, monkeypatch=monkeypatch)


async def expect(status_code, coro):
    with pytest.raises(HTTPException) as exc:
        await coro
    assert exc.value.status_code == status_code


# --------------------------------------------------------------------------- #
# List — the original model is always first and always the default
# --------------------------------------------------------------------------- #
async def test_list_puts_the_original_model_first(env):
    env.repo.add_variant(name="Corner", order_index=2)
    env.repo.add_variant(name="3 Seater", order_index=1)

    entries = await ModelVariantService.list_models(FakeSession(), PRODUCT, OWNER)

    assert [e.name for e in entries] == ["Default", "3 Seater", "Corner"]
    assert [e.is_original for e in entries] == [True, False, False]
    original = entries[0]
    assert original.glb_url == env.repo.mapped[9].image, "the original reads the mapped GLB"
    assert original.usdz_url == env.repo.mapped[11].image
    assert original.thumbnail_url == env.repo.mapped[1].image
    assert original.variant is None


async def test_list_with_no_variants_is_just_the_original(env):
    entries = await ModelVariantService.list_models(FakeSession(), PRODUCT, OWNER)
    assert len(entries) == 1 and entries[0].is_original


async def test_deleted_variants_are_not_listed(env):
    env.repo.add_variant(isactive=False)
    entries = await ModelVariantService.list_models(FakeSession(), PRODUCT, OWNER)
    assert len(entries) == 1


async def test_list_is_404_for_another_seller(env):
    await expect(404, ModelVariantService.list_models(FakeSession(), PRODUCT, STRANGER))


async def test_list_is_404_while_the_feature_is_off(env):
    env.monkeypatch.setattr(mv_module.settings, "ENABLE_MODEL_VARIANTS", False)
    await expect(404, ModelVariantService.list_models(FakeSession(), PRODUCT, OWNER))


# --------------------------------------------------------------------------- #
# Rename / delete / thumbnail — ownership and the original model
# --------------------------------------------------------------------------- #
async def test_rename(env):
    v = env.repo.add_variant(name="Corner")
    db = FakeSession()
    renamed = await ModelVariantService.rename(db, v.id, OWNER, "  L-Shape  ")
    assert renamed.name == "L-Shape"
    assert renamed.updated_by == OWNER
    assert db.committed == 1


async def test_rename_rejects_an_empty_name(env):
    v = env.repo.add_variant()
    await expect(400, ModelVariantService.rename(FakeSession(), v.id, OWNER, "   "))


async def test_soft_delete_keeps_the_row(env):
    v = env.repo.add_variant()
    await ModelVariantService.delete(FakeSession(), v.id, OWNER)
    assert v.isactive is False
    assert v.id in env.repo.variants, "soft delete: the row and its blobs stay"
    assert env.storage.deleted == []


async def test_deleting_the_last_variant_leaves_the_original(env):
    v = env.repo.add_variant()
    await ModelVariantService.delete(FakeSession(), v.id, OWNER)
    entries = await ModelVariantService.list_models(FakeSession(), PRODUCT, OWNER)
    assert [e.is_original for e in entries] == [True]


@pytest.mark.parametrize(
    "call",
    [
        lambda vid: ModelVariantService.rename(FakeSession(), vid, STRANGER, "X"),
        lambda vid: ModelVariantService.delete(FakeSession(), vid, STRANGER),
        lambda vid: ModelVariantService.set_thumbnail(
            FakeSession(), vid, STRANGER, UploadedFile("t.png", "image/png", b"png")
        ),
    ],
)
async def test_another_sellers_variant_is_404(env, call):
    v = env.repo.add_variant()
    await expect(404, call(v.id))


async def test_a_deleted_variant_cannot_be_edited(env):
    v = env.repo.add_variant(isactive=False)
    await expect(404, ModelVariantService.rename(FakeSession(), v.id, OWNER, "X"))


async def test_thumbnail_replaces_and_deletes_the_previous_file(env):
    v = env.repo.add_variant(thumbnail_url="https://cdn/old-thumb.png")
    updated = await ModelVariantService.set_thumbnail(
        FakeSession(), v.id, OWNER, UploadedFile("poster.png", "image/png", b"\x89PNG")
    )
    assert updated.thumbnail_url.endswith("thumbnail.png")
    assert env.storage.deleted == ["https://cdn/old-thumb.png"]
    path = env.storage.uploads[0][0]
    assert path.startswith(f"{OWNER}/{PRODUCT}/model-variants/{v.id}/")


async def test_thumbnail_must_be_an_image(env):
    v = env.repo.add_variant()
    await expect(
        400,
        ModelVariantService.set_thumbnail(
            FakeSession(), v.id, OWNER, UploadedFile("t.gif", "image/gif", b"GIF")
        ),
    )


# --------------------------------------------------------------------------- #
# Reorder
# --------------------------------------------------------------------------- #
async def test_reorder_sets_positions_after_the_original(env):
    a = env.repo.add_variant(name="A", order_index=1)
    b = env.repo.add_variant(name="B", order_index=2)
    await ModelVariantService.reorder(FakeSession(), PRODUCT, OWNER, [b.id, a.id])
    assert (b.order_index, a.order_index) == (1, 2)


@pytest.mark.parametrize("shape", ["missing", "duplicate", "foreign"])
async def test_reorder_must_name_every_variant_exactly_once(env, shape):
    a = env.repo.add_variant(order_index=1)
    b = env.repo.add_variant(order_index=2)
    other = env.repo.add_variant(product_id=OTHER_PRODUCT)
    ids = {"missing": [a.id], "duplicate": [a.id, a.id, b.id], "foreign": [a.id, b.id, other.id]}[shape]
    await expect(400, ModelVariantService.reorder(FakeSession(), PRODUCT, OWNER, ids))


async def test_reorder_is_404_for_another_seller(env):
    await expect(404, ModelVariantService.reorder(FakeSession(), PRODUCT, STRANGER, []))


# --------------------------------------------------------------------------- #
# The "original" / {id} path token
# --------------------------------------------------------------------------- #
async def test_original_token_is_the_original_model(env):
    assert await ModelVariantService.resolve_model(FakeSession(), PRODUCT, "original", OWNER) is None


async def test_variant_token_resolves(env):
    v = env.repo.add_variant()
    assert await ModelVariantService.resolve_model(FakeSession(), PRODUCT, str(v.id), OWNER) == v.id


@pytest.mark.parametrize("token", ["not-a-uuid", str(uuid.uuid4())])
async def test_unknown_token_is_404(env, token):
    await expect(404, ModelVariantService.resolve_model(FakeSession(), PRODUCT, token, OWNER))


async def test_a_variant_of_another_product_is_404_even_for_the_same_seller(env):
    other = env.repo.add_variant(product_id=OTHER_PRODUCT)
    await expect(404, ModelVariantService.resolve_model(FakeSession(), PRODUCT, str(other.id), OWNER))


async def test_token_is_404_for_another_seller(env):
    await expect(404, ModelVariantService.resolve_model(FakeSession(), PRODUCT, "original", STRANGER))


# --------------------------------------------------------------------------- #
# Parts scoped to one model
# --------------------------------------------------------------------------- #
async def create_part(name, indices, variant_id=None, user=OWNER, product=PRODUCT):
    return await PartService.create_part(
        FakeSession(), product, user,
        ProductPartCreate(name=name, material_indices=indices), variant_id=variant_id,
    )


async def test_part_on_a_variant_carries_its_id_and_glb_version(env):
    v = env.repo.add_variant()
    part = await create_part("Seat", [0], variant_id=v.id)
    assert part.variant_id == v.id
    assert part.glb_version == f"asset:{v.glb_asset_id}", "pinned to the VARIANT's GLB"


async def test_product_level_parts_still_mean_the_original_model(env):
    part = await create_part("Seat", [0])
    assert part.variant_id is None
    assert part.glb_version == f"asset:{env.repo.mapped[9].id}"


async def test_the_same_material_index_is_free_in_another_model(env):
    v = env.repo.add_variant()
    await create_part("Seat", [0, 3])
    part = await create_part("Seat", [0, 3], variant_id=v.id)
    assert part.material_indices == [0, 3]


async def test_overlap_is_still_rejected_within_one_model(env):
    v = env.repo.add_variant()
    await create_part("Seat", [0, 3], variant_id=v.id)
    await expect(400, create_part("Cushion", [3], variant_id=v.id))


async def test_slugs_stay_unique_per_product_across_models(env):
    """uq_parts_product_slug is unchanged, so a second 'Seat' is suffixed."""
    v = env.repo.add_variant()
    first = await create_part("Seat", [0])
    second = await create_part("Seat", [0], variant_id=v.id)
    assert (first.slug, second.slug) == ("seat", "seat-2")


async def test_listing_one_model_excludes_the_others(env):
    v = env.repo.add_variant()
    await create_part("Seat", [0])
    await create_part("Seat", [0], variant_id=v.id)

    original, _ = await PartService.list_parts(FakeSession(), PRODUCT, OWNER)
    variant, version = await PartService.list_parts(FakeSession(), PRODUCT, OWNER, variant_id=v.id)
    assert [p.variant_id for p in original] == [None]
    assert [p.variant_id for p in variant] == [v.id]
    assert version == f"asset:{v.glb_asset_id}"


async def test_part_on_another_products_variant_is_404(env):
    other = env.repo.add_variant(product_id=OTHER_PRODUCT)
    await expect(404, create_part("Seat", [0], variant_id=other.id))


async def test_part_on_a_deleted_variant_is_404(env):
    v = env.repo.add_variant(isactive=False)
    await expect(404, create_part("Seat", [0], variant_id=v.id))


async def test_part_on_another_sellers_variant_is_404(env):
    v = env.repo.add_variant()
    await expect(404, create_part("Seat", [0], variant_id=v.id, user=STRANGER))
