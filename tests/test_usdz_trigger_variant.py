"""usdz_trigger_service — the converter job's arguments (ADR-014, step g).

Existing callers must produce exactly the arguments they always did; only a
model-variant conversion adds --model-variant-id.
"""

from types import SimpleNamespace

import pytest

from app.services import usdz_trigger_service as module


@pytest.fixture
def captured(monkeypatch):
    posted = []

    class _Credential:
        def get_token(self, _scope):
            return SimpleNamespace(token="t")

    def _post(url, json, headers, timeout):
        posted.append(json)
        return SimpleNamespace(ok=True, status_code=202, text="", raise_for_status=lambda: None)

    monkeypatch.setattr(module, "DefaultAzureCredential", _Credential, raising=False)
    monkeypatch.setattr(module.requests, "post", _post)
    return posted


def _args(posted):
    return posted[0]["containers"][0]["args"]


BASE = dict(
    glb_blob_url="https://blob/dev/u/p/model.glb",
    product_id="p",
    user_id="u",
    product_name="Sofa",
    output_blob_name="model.usdz",
    job_id="j",
)


def test_product_conversion_arguments_are_unchanged(captured):
    module.usdz_trigger_service._trigger_sync(**BASE)
    assert _args(captured) == [
        "--job-id=j",
        "--glb-blob-url=https://blob/dev/u/p/model.glb",
        "--output-blob-name=model.usdz",
        "--product-id=p",
        "--user-id=u",
        "--product-name=Sofa",
    ]


def test_variant_conversion_adds_only_the_variant_flag(captured):
    module.usdz_trigger_service._trigger_sync(**BASE, model_variant_id="v1")
    args = _args(captured)
    assert args[-1] == "--model-variant-id=v1"
    assert args[:-1] == [
        "--job-id=j",
        "--glb-blob-url=https://blob/dev/u/p/model.glb",
        "--output-blob-name=model.usdz",
        "--product-id=p",
        "--user-id=u",
        "--product-name=Sofa",
    ]
