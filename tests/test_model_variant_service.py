"""ModelVariantService.create_variant — the upload pipeline (ADR-014).

Repository, blob storage and the Node compressor are faked; GLB inspection is
real, on GLBs built in memory. The properties pinned here are the ones that
keep the feature additive and safe:

  * no tbl_product_asset_mapping row, ever — so the product's model, list
    thumbnail, /assets and AR are untouched
  * ownership resolves to 404
  * compression that renames or reorders materials is refused
  * blobs never outlive a failed save
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models.configurator import ProductModelVariant
from app.models.models import ProductAsset, ProductAssetMapping
from app.services.configurator import model_variant_service as module
from app.services.configurator.model_variant_service import ModelVariantService, UploadedFile
from tests._glb_factory import make_glb

OWNER = uuid.uuid4()
STRANGER = uuid.uuid4()
PRODUCT = uuid.uuid4()


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeSession:
    def __init__(self, *, fail_commit=False):
        self.committed = 0
        self.rolled_back = 0
        self.fail_commit = fail_commit

    async def commit(self):
        if self.fail_commit:
            raise RuntimeError("database unavailable")
        self.committed += 1

    async def rollback(self):
        self.rolled_back += 1

    async def refresh(self, obj):
        # What the database's server_default=now() would have filled in.
        if getattr(obj, "created_date", None) is None:
            obj.created_date = datetime.now(timezone.utc)

    async def execute(self, *_a, **_kw):
        raise AssertionError("a service reached the database directly")


class FakeRepo:
    def __init__(self):
        self.product = SimpleNamespace(id=PRODUCT, created_by=OWNER)
        self.added = []
        self.flushed = []
        self.next_order = 1

    async def get_owned_product(self, db, product_id, user_id, *, for_update=False):
        if user_id != OWNER or product_id != PRODUCT:
            return None
        return self.product

    async def get_next_model_variant_order_index(self, db, product_id):
        return self.next_order

    def add(self, db, instance):
        self.added.append(instance)

    async def flush(self, db):
        # Records WHAT had been added when the flush happened: the asset row
        # must already be in, or its FK from the variant has nothing to point
        # at (the real session inserts in relationship order, not add order).
        self.flushed.append(tuple(type(i).__name__ for i in self.added))


class FakeStorage:
    def __init__(self, *, fail_on=None):
        self.uploads = []
        self.deleted = []
        self.fail_on = fail_on

    def upload_model_variant_file(self, *, user_id, product_id, variant_id, filename, content_type, stream):
        if self.fail_on and filename.startswith(self.fail_on):
            raise RuntimeError("blob storage unavailable")
        data = stream.read()
        path = f"{user_id}/{product_id}/model-variants/{variant_id}/{filename}"
        self.uploads.append((path, content_type, data))
        return f"https://cdn.example.net/dev/{path}", f"https://acct.blob.core.windows.net/dev/{path}"

    def delete_blob_by_cdn_url(self, url):
        self.deleted.append(url)
        return True


class FakeTrigger:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    async def trigger_conversion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error


class FakeCompressor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def compress(self, data):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def env(monkeypatch):
    repo, storage = FakeRepo(), FakeStorage()
    compressor = FakeCompressor(result=make_glb(draco_flag=True))
    monkeypatch.setattr(module, "repo", repo)
    monkeypatch.setattr(module, "storage_service", storage)
    monkeypatch.setattr(module, "glb_compression_service", compressor)
    trigger = FakeTrigger()
    monkeypatch.setattr(module, "usdz_trigger_service", trigger)
    monkeypatch.setattr(module.settings, "ENABLE_MODEL_VARIANTS", True)
    monkeypatch.setattr(module.settings, "ENABLE_DRACO_COMPRESSION", True)
    monkeypatch.setattr(module.settings, "ENABLE_VARIANT_USDZ", True)
    monkeypatch.setattr(module.settings, "MAX_VARIANT_GLB_BYTES", 10 * 1024 * 1024)
    return SimpleNamespace(
        repo=repo, storage=storage, compressor=compressor, trigger=trigger, monkeypatch=monkeypatch
    )


def glb_upload(data=None, filename="corner.glb"):
    return UploadedFile(filename=filename, content_type="model/gltf-binary", data=data if data is not None else make_glb())


async def create(db=None, user=OWNER, product=PRODUCT, name="Corner", glb=None, thumbnail=None):
    return await ModelVariantService.create_variant(
        db or FakeSession(), product, user, name=name, glb=glb or glb_upload(), thumbnail=thumbnail
    )


def added_of(repo, cls):
    return [o for o in repo.added if isinstance(o, cls)]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_creates_one_asset_and_one_variant_and_no_mapping(env):
    db = FakeSession()
    created = await create(db=db)

    assets = added_of(env.repo, ProductAsset)
    variants = added_of(env.repo, ProductModelVariant)
    assert len(assets) == 1 and len(variants) == 1
    assert added_of(env.repo, ProductAssetMapping) == [], "a mapping would replace the product's model"
    assert db.committed == 1

    asset, variant = assets[0], variants[0]
    assert asset.asset_id == 9
    assert asset.created_by == OWNER, "the purge finds unmapped assets only through created_by"
    assert asset.image == created.glb_url
    assert variant.glb_asset_id == asset.id
    assert variant.product_id == PRODUCT
    assert variant.name == "Corner"
    assert variant.order_index == 1
    assert variant.created_by == OWNER
    assert created.variant is variant


async def test_the_asset_row_is_flushed_before_the_variant_references_it(env):
    """Regression: the variant's FK failed in dev because the asset went second.

    glb_asset_id is a plain FK column with no relationship(), so SQLAlchemy's
    unit of work has nothing to order the two inserts by and can send the
    variant first - fk_model_variants_glb_asset then fails.
    """
    await create(db=FakeSession())

    assert env.repo.flushed == [("ProductAsset",)], (
        "the asset must be flushed on its own, before the variant is added"
    )
    assert [type(o).__name__ for o in env.repo.added] == ["ProductAsset", "ProductModelVariant"]


async def test_compressed_upload_records_sizes_and_keeps_the_original(env):
    original = make_glb()
    created = await create(glb=glb_upload(original))
    variant = created.variant

    assert variant.compression_status == "compressed"
    assert variant.compression_error is None
    assert variant.original_size_bytes == len(original)
    assert variant.compressed_size_bytes == len(env.compressor.result)
    assert variant.original_glb_url is not None and variant.original_glb_url.endswith("original.glb")

    stored = {path.rsplit("/", 1)[1]: data for path, _ct, data in env.storage.uploads}
    assert stored["model.glb"] == env.compressor.result
    assert stored["original.glb"] == original
    assert created.warnings == []


async def test_blobs_live_under_the_sellers_prefix(env):
    """{user_id}/{product_id}/model-variants/{variant_id}/ — swept by the purge job."""
    created = await create()
    for path, _ct, _data in env.storage.uploads:
        assert path.startswith(f"{OWNER}/{PRODUCT}/model-variants/{created.variant.id}/")


async def test_dimensions_are_stored_in_metres(env):
    created = await create(glb=glb_upload(make_glb(size=(2.4, 0.85, 1.0))))
    variant = created.variant
    assert variant.width_m == pytest.approx(2.4)
    assert variant.height_m == pytest.approx(0.85)
    assert variant.depth_m == pytest.approx(1.0)


async def test_name_is_trimmed(env):
    created = await create(name="  3 Seater  ")
    assert created.variant.name == "3 Seater"


async def test_thumbnail_is_stored_when_sent(env):
    thumb = UploadedFile(filename="poster.png", content_type="image/png", data=b"\x89PNG....")
    created = await create(thumbnail=thumb)
    assert created.variant.thumbnail_url.endswith("thumbnail.png")
    assert any(p.endswith("thumbnail.png") for p, _ct, _d in env.storage.uploads)


# --------------------------------------------------------------------------- #
# Compression fallbacks — the upload still succeeds, uncompressed
# --------------------------------------------------------------------------- #
async def test_compression_failure_falls_back_to_the_original(env):
    env.compressor.error = RuntimeError("gltf-transform exited 1")
    original = make_glb()
    created = await create(glb=glb_upload(original))
    variant = created.variant

    assert variant.compression_status == "fallback_original"
    assert "gltf-transform exited 1" in variant.compression_error
    assert variant.compressed_size_bytes is None
    assert variant.original_glb_url is None, "the served file IS the original; no second copy"
    assert [d for _p, _ct, d in env.storage.uploads] == [original]
    assert any("without Draco compression" in w for w in created.warnings)


async def test_compression_that_renames_materials_is_refused(env):
    env.compressor.result = make_glb(materials=("Material.001", "Material.002"), draco_flag=True)
    original = make_glb(materials=("Seat", "Legs"))
    created = await create(glb=glb_upload(original))

    assert created.variant.compression_status == "fallback_original"
    assert "names" in created.variant.compression_error
    assert env.storage.uploads[0][2] == original


async def test_compression_that_reorders_materials_is_refused(env):
    env.compressor.result = make_glb(materials=("Legs", "Seat"), draco_flag=True)
    created = await create(glb=glb_upload(make_glb(materials=("Seat", "Legs"))))
    assert created.variant.compression_status == "fallback_original"


async def test_output_that_is_not_draco_is_refused(env):
    env.compressor.result = make_glb(draco_flag=False)
    created = await create()
    assert created.variant.compression_status == "fallback_original"


async def test_already_draco_upload_is_not_recompressed(env):
    original = make_glb(draco_flag=True)
    created = await create(glb=glb_upload(original))

    assert env.compressor.calls == 0
    assert created.variant.compression_status == "compressed"
    assert created.variant.original_glb_url is None
    assert [d for _p, _ct, d in env.storage.uploads] == [original]


async def test_disabled_compression_falls_back(env):
    env.monkeypatch.setattr(module.settings, "ENABLE_DRACO_COMPRESSION", False)
    created = await create()
    assert env.compressor.calls == 0
    assert created.variant.compression_status == "fallback_original"


# --------------------------------------------------------------------------- #
# Dimension warnings — advisory, never a rejection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [(0.05, 0.03, 0.04), (25.0, 8.0, 9.0)])
async def test_implausible_size_warns_but_succeeds(env, size):
    created = await create(glb=glb_upload(make_glb(size=size)))
    assert any("unusual for furniture" in w for w in created.warnings)
    assert added_of(env.repo, ProductModelVariant)


async def test_unmeasurable_size_warns(env):
    created = await create(glb=glb_upload(make_glb(with_bounds=False)))
    assert created.variant.width_m is None
    assert any("could not be measured" in w for w in created.warnings)


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
async def assert_rejected(status_code, **kwargs):
    with pytest.raises(HTTPException) as exc:
        await create(**kwargs)
    assert exc.value.status_code == status_code
    return exc.value


async def test_feature_flag_off_looks_like_a_missing_route(env):
    env.monkeypatch.setattr(module.settings, "ENABLE_MODEL_VARIANTS", False)
    await assert_rejected(404)
    assert env.repo.added == [] and env.storage.uploads == []


async def test_another_sellers_product_is_404_not_403(env):
    await assert_rejected(404, user=STRANGER)
    assert env.storage.uploads == []


async def test_unknown_product_is_404(env):
    await assert_rejected(404, product=uuid.uuid4())


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
async def test_bad_name_is_400(env, name):
    await assert_rejected(400, name=name)


async def test_non_glb_extension_is_400(env):
    await assert_rejected(400, glb=glb_upload(filename="corner.gltf"))


async def test_empty_file_is_400(env):
    await assert_rejected(400, glb=glb_upload(data=b""))


async def test_oversized_file_is_413(env):
    env.monkeypatch.setattr(module.settings, "MAX_VARIANT_GLB_BYTES", 100)
    await assert_rejected(413)
    assert env.storage.uploads == []


async def test_corrupt_glb_is_400(env):
    await assert_rejected(400, glb=glb_upload(data=b"glTF" + b"\x02\x00\x00\x00" + b"garbage" * 10))


async def test_bad_thumbnail_type_is_400(env):
    thumb = UploadedFile(filename="poster.gif", content_type="image/gif", data=b"GIF89a")
    await assert_rejected(400, thumbnail=thumb)


# --------------------------------------------------------------------------- #
# Failure cleanup
# --------------------------------------------------------------------------- #
async def test_storage_failure_is_502_and_cleans_up(env):
    env.storage.fail_on = "original"  # model.glb succeeds, original.glb fails
    await assert_rejected(502)
    assert env.repo.added == []
    assert len(env.storage.deleted) == 1 and env.storage.deleted[0].endswith("model.glb")


async def test_database_failure_rolls_back_and_deletes_the_blobs(env):
    db = FakeSession(fail_commit=True)
    with pytest.raises(RuntimeError):
        await create(db=db)
    assert db.rolled_back == 1
    assert len(env.storage.deleted) == len(env.storage.uploads) == 2


# --------------------------------------------------------------------------- #
# Per-variant USDZ (step g)
# --------------------------------------------------------------------------- #
async def test_usdz_conversion_is_requested_for_the_variant(env):
    created = await create()
    (call,) = env.trigger.calls
    assert call["model_variant_id"] == str(created.variant.id)
    assert call["product_id"] == str(PRODUCT)
    assert call["user_id"] == str(OWNER)
    # Converted from the kept, uncompressed upload.
    assert call["glb_blob_url"].endswith("original.glb")


async def test_usdz_source_is_the_served_file_when_no_original_was_kept(env):
    env.compressor.error = RuntimeError("no node")
    await create()
    assert env.trigger.calls[0]["glb_blob_url"].endswith("model.glb")


async def test_a_failed_usdz_request_never_fails_the_upload(env):
    env.trigger.error = RuntimeError("ARM unavailable")
    created = await create()
    assert created.variant is not None


async def test_no_usdz_request_while_the_feature_is_off(env):
    """Default state: no converter run, and the variant is still created."""
    env.monkeypatch.setattr(module.settings, "ENABLE_VARIANT_USDZ", False)
    created = await create()
    assert created.variant is not None
    assert env.trigger.calls == []


async def test_no_usdz_request_when_the_upload_fails(env):
    env.storage.fail_on = "model"
    await assert_rejected(502)
    assert env.trigger.calls == []
