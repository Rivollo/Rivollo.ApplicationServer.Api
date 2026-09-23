"""The shopper payload with model variants (ADR-014, step e).

GET /public/products/{id}/configurator — the Python source of truth that
Rivollo.Viewer.Api mirrors. Pinned here:

  * no `variants` key at all unless the product has extra variants and the
    feature is on — a single-model product's payload is unchanged
  * the original model comes first and is the default; top-level model_url and
    parts stay the ORIGINAL's, for clients that predate variants
  * each variant's parts are filtered against THAT variant's GLB
  * the shopper contract carries no seller-only field
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.main import app
from app.services.configurator import shopper_service as module

PRODUCT = uuid.uuid4()
ORIGINAL_GLB = SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/model.glb")
ORIGINAL_USDZ = SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/model.usdz")
THUMB = SimpleNamespace(id=uuid.uuid4(), image="https://cdn/dev/u/p/thumb.png")


def texture():
    return SimpleNamespace(material_index=0, url="https://cdn/t.png", content_type="image/png", recipe_hash="h")


def option(**kw):
    defaults = dict(
        id=uuid.uuid4(), name="Charcoal", slug="charcoal", swatch_hex="#333333",
        order_index=0, is_default=False, isactive=True, bake_status="completed",
        recipe_hash="h", textures=[texture()],
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def part(glb_version, variant_id=None, **kw):
    defaults = dict(
        id=uuid.uuid4(), product_id=PRODUCT, variant_id=variant_id, name="Seat", slug="seat",
        material_indices=[0], order_index=0, isactive=True, shopper_selectable=True,
        glb_version=glb_version, options=[option()],
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class Repo:
    def __init__(self):
        self.variants = []
        self.assets = {}
        self.parts = [part(f"asset:{ORIGINAL_GLB.id}")]

    async def get_published_product(self, db, product_id):
        return SimpleNamespace(id=PRODUCT, name="Aria Sofa") if product_id == PRODUCT else None

    async def get_product_mesh_asset(self, db, product_id):
        return ORIGINAL_GLB

    async def get_product_asset(self, db, product_id, asset_id):
        return {11: ORIGINAL_USDZ, 1: THUMB}.get(asset_id)

    async def get_parts_for_product(self, db, product_id, *, variant_id=None, active_only=False):
        return [p for p in self.parts if p.variant_id == variant_id]

    async def get_model_variants(self, db, product_id):
        return sorted((v for v in self.variants if v.isactive), key=lambda v: v.order_index)

    async def get_assets_by_ids(self, db, ids):
        return {i: self.assets[i] for i in ids if i in self.assets}

    def add_variant(self, name="Corner", order_index=1, with_glb=True, **kw):
        glb_id = uuid.uuid4()
        variant = SimpleNamespace(
            id=uuid.uuid4(), product_id=PRODUCT, name=name, glb_asset_id=glb_id,
            usdz_asset_id=None, thumbnail_url="https://cdn/corner-thumb.png",
            order_index=order_index, isactive=True, width_m=2.9, depth_m=2.1, height_m=0.8,
            compression_status="compressed", original_glb_url="https://cdn/secret-original.glb",
        )
        for key, value in kw.items():
            setattr(variant, key, value)
        if with_glb:
            self.assets[glb_id] = SimpleNamespace(id=glb_id, image=f"https://cdn/{name}.glb")
        self.variants.append(variant)
        return variant


@pytest.fixture
def repo(monkeypatch):
    fake = Repo()
    monkeypatch.setattr(module, "repo", fake)
    monkeypatch.setattr(module.settings, "ENABLE_MODEL_VARIANTS", True)
    return fake


@pytest.fixture
def client(repo):
    async def _db():
        yield SimpleNamespace()

    app.dependency_overrides[get_db] = _db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def fetch(client):
    response = client.get(f"/public/products/{PRODUCT}/configurator")
    assert response.status_code == 200, response.text
    return response.json()["data"]


# --------------------------------------------------------------------------- #
# Unchanged when there is nothing to add
# --------------------------------------------------------------------------- #
def test_single_model_product_has_no_variants_key(client, repo):
    body = fetch(client)
    assert "variants" not in body
    assert body["model_url"] == ORIGINAL_GLB.image
    assert len(body["parts"]) == 1


def test_feature_off_has_no_variants_key_even_with_rows(client, repo):
    repo.add_variant()
    repo_settings = module.settings
    repo_settings.ENABLE_MODEL_VARIANTS = False
    try:
        assert "variants" not in fetch(client)
    finally:
        repo_settings.ENABLE_MODEL_VARIANTS = True


def test_only_deleted_variants_means_no_variants_key(client, repo):
    repo.add_variant(isactive=False)
    assert "variants" not in fetch(client)


def test_variant_without_a_glb_is_skipped(client, repo):
    repo.add_variant(with_glb=False)
    assert "variants" not in fetch(client), "nothing showable beyond the original"


# --------------------------------------------------------------------------- #
# With variants
# --------------------------------------------------------------------------- #
def test_original_first_and_default_then_variants_in_order(client, repo):
    repo.add_variant(name="Corner", order_index=2)
    repo.add_variant(name="3 Seater", order_index=1)
    body = fetch(client)

    variants = body["variants"]
    assert [v["name"] for v in variants] == ["Default", "3 Seater", "Corner"]
    assert [v["is_default"] for v in variants] == [True, False, False]
    original = variants[0]
    assert original["id"] == "original"
    assert original["glb_url"] == ORIGINAL_GLB.image
    assert original["usdz_url"] == ORIGINAL_USDZ.image
    assert original["thumbnail_url"] == THUMB.image
    assert original["parts"] == body["parts"]


def test_top_level_fields_stay_the_originals(client, repo):
    repo.add_variant()
    body = fetch(client)
    assert body["model_url"] == ORIGINAL_GLB.image
    assert body["ar_model_url"] == ORIGINAL_USDZ.image
    assert [p["name"] for p in body["parts"]] == ["Seat"]


def test_variant_carries_its_glb_dimensions_and_own_parts(client, repo):
    v = repo.add_variant(name="Corner")
    repo.parts.append(part(f"asset:{v.glb_asset_id}", variant_id=v.id, name="Chaise", slug="chaise"))
    corner = fetch(client)["variants"][1]

    assert corner["id"] == str(v.id)
    assert corner["glb_url"] == "https://cdn/Corner.glb"
    assert corner["usdz_url"] is None
    assert (corner["width_m"], corner["depth_m"], corner["height_m"]) == (2.9, 2.1, 0.8)
    assert [p["name"] for p in corner["parts"]] == ["Chaise"]


def test_variant_part_authored_against_another_glb_is_hidden(client, repo):
    v = repo.add_variant()
    repo.parts.append(part("asset:some-other-glb", variant_id=v.id, name="Stale"))
    assert fetch(client)["variants"][1]["parts"] == []


def test_variant_part_with_no_completed_option_is_hidden(client, repo):
    v = repo.add_variant()
    repo.parts.append(
        part(f"asset:{v.glb_asset_id}", variant_id=v.id, options=[option(bake_status="pending")])
    )
    assert fetch(client)["variants"][1]["parts"] == []


def test_shopper_variant_exposes_no_seller_fields(client, repo):
    repo.add_variant()
    corner = fetch(client)["variants"][1]
    assert set(corner) == {
        "id", "name", "glb_url", "usdz_url", "thumbnail_url", "is_default", "order_index",
        "width_m", "depth_m", "height_m", "parts",
    }
    assert "secret-original" not in str(corner)
