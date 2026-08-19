"""Golden-parity regression test for the DB-driven fal model registry's
generic engine (app/integrations/fal/registry.py).

Before this registry became database-backed, each model had its own
hand-written build_body/extract_glb_url Python function. Those are gone now;
one generic engine (``_resolve_path`` / ``_first_matching_url`` /
``FalModelSpec.build_body``) serves every model, driven entirely by each
row's ``provider_config``. These tests pin that engine's behavior against the
exact request/response shapes the old per-model functions were verified
against (see the migration PR description), so a future change to the engine
that silently breaks one model's actual production shape fails a test
instead of a live generation.

Deliberately does NOT touch the database: FalModelSpec is a plain frozen
dataclass, and the engine functions are pure. No fixtures, no get_db
override needed.
"""

from app.integrations.fal.registry import FalModelSpec, _first_matching_url, _resolve_path

TEST_IMAGE_URL = "https://cdn.example.com/photo.jpg"


def _spec(key: str, image_url_field: str, template: dict, glb_paths, usdz_paths=()) -> FalModelSpec:
    return FalModelSpec(
        key=key,
        provider="fal_queue",
        label=key,
        description="",
        endpoint_id="fal-ai/test",
        credit_cost=1,
        baseline_estimate_seconds=1,
        free_plan_eligible=False,
        is_default=False,
        max_wait_seconds=600.0,
        image_url_field=image_url_field,
        request_body_template=template,
        glb_url_paths=tuple(glb_paths),
        usdz_url_paths=tuple(usdz_paths),
    )


# --------------------------------------------------------------------------- #
# build_body — each current model's request is a static template plus one
# substituted field.
# --------------------------------------------------------------------------- #
def test_build_body_substitutes_the_configured_image_field():
    spec = _spec("tripo", "image_url", {"face_limit": 50000}, ["model_urls.glb.url"])
    assert spec.build_body(TEST_IMAGE_URL) == {
        "face_limit": 50000,
        "image_url": TEST_IMAGE_URL,
    }


def test_build_body_uses_the_vendor_specific_field_name():
    # Hunyuan is the one model whose fal endpoint expects "input_image_url",
    # not "image_url" — this is exactly what image_url_field exists for.
    spec = _spec("hunyuan", "input_image_url", {"enable_pbr": True}, ["model_glb.url"])
    body = spec.build_body(TEST_IMAGE_URL)
    assert body["input_image_url"] == TEST_IMAGE_URL
    assert "image_url" not in body


def test_build_body_null_is_preserved_not_dropped():
    # SAM 3D nulls "prompt" explicitly so fal's own "car" default can't
    # compete with the no-mask direct-image request. None must survive.
    spec = _spec("sam3", "image_url", {"prompt": None, "export_textured_glb": True}, [])
    body = spec.build_body(TEST_IMAGE_URL)
    assert body["prompt"] is None


# --------------------------------------------------------------------------- #
# extract_glb_url — ordered "try this path, then that one" search.
# --------------------------------------------------------------------------- #
def test_extract_glb_url_tries_paths_in_order():
    spec = _spec("tripo", "image_url", {}, ["model_urls.glb.url", "model_mesh.url"])
    assert spec.extract_glb_url({"model_urls": {"glb": {"url": "A"}}}) == "A"
    assert spec.extract_glb_url({"model_mesh": {"url": "B"}}) == "B"
    assert spec.extract_glb_url({}) is None


def test_extract_glb_url_first_path_wins_when_both_present():
    spec = _spec("hunyuan", "input_image_url", {}, ["model_glb.url", "model_urls.glb.url"])
    result = {"model_glb": {"url": "PRIMARY"}, "model_urls": {"glb": {"url": "FALLBACK"}}}
    assert spec.extract_glb_url(result) == "PRIMARY"


def test_extract_glb_url_list_wildcard_returns_first_match():
    # SAM 3D falls back to the first entry of individual_glbs with a URL —
    # the one list construct the path grammar supports.
    spec = _spec("sam3", "image_url", {}, ["model_glb.url", "individual_glbs[].url"])
    result = {"individual_glbs": [{}, {"url": "SECOND"}, {"url": "THIRD"}]}
    assert spec.extract_glb_url(result) == "SECOND"


def test_extract_glb_url_combined_glb_wins_over_individual_list():
    spec = _spec("sam3", "image_url", {}, ["model_glb.url", "individual_glbs[].url"])
    result = {
        "model_glb": {"url": "COMBINED"},
        "individual_glbs": [{"url": "OBJ0"}],
    }
    assert spec.extract_glb_url(result) == "COMBINED"


def test_extract_glb_url_empty_string_is_not_a_match():
    spec = _spec("meshy", "image_url", {}, ["model_glb.url"])
    assert spec.extract_glb_url({"model_glb": {"url": ""}}) is None


# --------------------------------------------------------------------------- #
# extract_usdz_url / provides_usdz — only Meshy exports one today.
# --------------------------------------------------------------------------- #
def test_provides_usdz_false_when_no_paths_configured():
    spec = _spec("tripo", "image_url", {}, ["model_urls.glb.url"], usdz_paths=[])
    assert spec.provides_usdz is False
    assert spec.extract_usdz_url({"model_urls": {"usdz": {"url": "X"}}}) is None


def test_provides_usdz_true_and_extracts_when_configured():
    spec = _spec(
        "meshy", "image_url", {}, ["model_glb.url"], usdz_paths=["model_urls.usdz.url"]
    )
    assert spec.provides_usdz is True
    assert spec.extract_usdz_url({"model_urls": {"usdz": {"url": "X"}}}) == "X"


def test_extract_usdz_url_tolerates_null_entry():
    # Meshy can return {"usdz": null} even though the key is present — must
    # not raise, must resolve to None.
    spec = _spec("meshy", "image_url", {}, [], usdz_paths=["model_urls.usdz.url"])
    assert spec.extract_usdz_url({"model_urls": {"usdz": None}}) is None


# --------------------------------------------------------------------------- #
# _resolve_path — the primitive underneath both extractors.
# --------------------------------------------------------------------------- #
def test_resolve_path_returns_none_for_wrong_shape_rather_than_raising():
    assert _resolve_path("not a dict", ["a", "b"]) is None
    assert _resolve_path({"a": "not a dict either"}, ["a", "b"]) is None
    assert _resolve_path(None, ["a"]) is None


def test_resolve_path_list_wildcard_on_non_list_returns_none():
    assert _resolve_path({"items": "not a list"}, ["items[]", "url"]) is None


def test_first_matching_url_ignores_non_string_values():
    assert _first_matching_url({"model_glb": {"url": 12345}}, ("model_glb.url",)) is None
