"""POST /products/{id}/configurator/model-variants — the HTTP contract (ADR-014).

Multipart parsing, status codes and the response shape. The pipeline rules are
covered in test_model_variant_service.py; this proves the route wires them up.
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user, get_db
from app.main import app
from app.services.configurator import model_variant_service as module
from tests._glb_factory import make_glb
from tests.test_model_variant_service import (
    FakeCompressor,
    FakeRepo,
    FakeSession,
    FakeStorage,
    FakeTrigger,
    OWNER,
    PRODUCT,
)

URL = f"/products/{PRODUCT}/configurator/model-variants"

RESPONSE_FIELDS = {
    "id", "product_id", "name", "glb_url", "usdz_url", "thumbnail_url", "order_index", "is_original",
    "isactive", "compression_status", "compression_error", "original_size_bytes",
    "compressed_size_bytes", "width_m", "depth_m", "height_m", "created_at", "warnings",
}


@pytest.fixture
def fakes(monkeypatch):
    repo, storage = FakeRepo(), FakeStorage()
    monkeypatch.setattr(module, "repo", repo)
    monkeypatch.setattr(module, "storage_service", storage)
    monkeypatch.setattr(module, "glb_compression_service", FakeCompressor(result=make_glb(draco_flag=True)))
    monkeypatch.setattr(module, "usdz_trigger_service", FakeTrigger())
    monkeypatch.setattr(module.settings, "ENABLE_MODEL_VARIANTS", True)
    monkeypatch.setattr(module.settings, "ENABLE_DRACO_COMPRESSION", True)
    return SimpleNamespace(repo=repo, storage=storage, monkeypatch=monkeypatch)


def _client(user_id):
    session = FakeSession()

    async def _db():
        yield session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    return TestClient(app)


@pytest.fixture
def client(fakes):
    with _client(OWNER) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def stranger_client(fakes):
    with _client(uuid.uuid4()) as c:
        yield c
    app.dependency_overrides.clear()


def _post(client, *, name="Corner", glb=None, filename="corner.glb", thumbnail=None):
    files = {"glb": (filename, glb if glb is not None else make_glb(size=(2.9, 0.8, 2.1)), "model/gltf-binary")}
    if thumbnail is not None:
        files["thumbnail"] = thumbnail
    return client.post(URL, data={"name": name}, files=files)


def test_created_variant_has_the_documented_shape(client, fakes):
    response = _post(client)
    assert response.status_code == 201, response.text
    body = response.json()["data"]
    assert set(body) == RESPONSE_FIELDS
    assert body["name"] == "Corner"
    assert body["is_original"] is False
    assert body["compression_status"] == "compressed"
    assert body["width_m"] == pytest.approx(2.9)
    assert body["depth_m"] == pytest.approx(2.1)
    assert body["glb_url"].endswith("/model.glb")
    assert body["warnings"] == []


def test_thumbnail_can_be_sent_with_the_model(client, fakes):
    response = _post(client, thumbnail=("poster.png", b"\x89PNG....", "image/png"))
    assert response.status_code == 201, response.text
    assert response.json()["data"]["thumbnail_url"].endswith("thumbnail.png")


def test_another_sellers_product_is_404(stranger_client, fakes):
    assert _post(stranger_client).status_code == 404
    assert fakes.storage.uploads == []


def test_feature_off_is_404(client, fakes):
    fakes.monkeypatch.setattr(module.settings, "ENABLE_MODEL_VARIANTS", False)
    assert _post(client).status_code == 404


def test_malformed_product_id_is_400(client, fakes):
    response = client.post(
        "/products/not-a-uuid/configurator/model-variants",
        data={"name": "Corner"},
        files={"glb": ("corner.glb", make_glb(), "model/gltf-binary")},
    )
    assert response.status_code == 400


def test_non_glb_is_400(client, fakes):
    assert _post(client, filename="corner.obj").status_code == 400


def test_oversized_upload_is_413_without_reading_it_all(client, fakes):
    fakes.monkeypatch.setattr(module.settings, "MAX_VARIANT_GLB_BYTES", 1024)
    assert _post(client, glb=make_glb() + b"\x00" * 4096).status_code == 413


def test_missing_name_is_422(client, fakes):
    response = client.post(URL, files={"glb": ("corner.glb", make_glb(), "model/gltf-binary")})
    assert response.status_code == 422
