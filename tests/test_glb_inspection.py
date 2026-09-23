"""glb_inspection — names, Draco detection and the bounding box (ADR-014)."""

import shutil
from pathlib import Path

import pytest

from app.services.configurator.glb_inspection import InvalidGlbError, inspect_glb
from tests._glb_factory import make_glb

COMPRESS_SCRIPT_DEPS = (
    Path(__file__).resolve().parent.parent / "scripts" / "glb_compress" / "node_modules"
)


def test_names_are_read_in_order():
    summary = inspect_glb(make_glb(materials=("Seat", "Legs", "Frame"), meshes=("Sofa", "Cushion")))
    assert summary.material_names == ("Seat", "Legs", "Frame")
    assert summary.mesh_names == ("Sofa", "Cushion")
    assert summary.draco_compressed is False


def test_bounding_box_in_metres_y_up():
    summary = inspect_glb(make_glb(size=(2.0, 0.8, 0.9)))
    assert summary.width_m == pytest.approx(2.0)
    assert summary.height_m == pytest.approx(0.8)
    assert summary.depth_m == pytest.approx(0.9)
    assert summary.largest_side_m == pytest.approx(2.0)


def test_node_scale_is_applied():
    """A model authored in centimetres and scaled 0.01 at the node is still metres."""
    summary = inspect_glb(make_glb(size=(200.0, 80.0, 90.0), node_scale=[0.01, 0.01, 0.01]))
    assert summary.width_m == pytest.approx(2.0)
    assert summary.height_m == pytest.approx(0.8)


def test_missing_bounds_yield_no_dimensions():
    summary = inspect_glb(make_glb(with_bounds=False))
    assert (summary.width_m, summary.height_m, summary.depth_m) == (None, None, None)
    assert summary.largest_side_m is None


def test_draco_extension_is_detected():
    assert inspect_glb(make_glb(draco_flag=True)).draco_compressed is True


def test_name_comparison_is_order_sensitive():
    a = inspect_glb(make_glb(materials=("Seat", "Legs")))
    b = inspect_glb(make_glb(materials=("Legs", "Seat")))
    assert a.same_names_as(a)
    assert not a.same_names_as(b)


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "too small"),
        (b"not a glb file at all", "not a binary glTF"),
        (b"glTF" + (1).to_bytes(4, "little") + (20).to_bytes(4, "little"), "glTF 2.0"),
    ],
)
def test_non_glb_input_is_rejected(data, message):
    with pytest.raises(InvalidGlbError, match=message):
        inspect_glb(data)


def test_glb_without_meshes_is_rejected():
    with pytest.raises(InvalidGlbError, match="no meshes"):
        inspect_glb(make_glb(meshes=()))


@pytest.mark.skipif(
    shutil.which("node") is None or not COMPRESS_SCRIPT_DEPS.is_dir(),
    reason="needs Node.js and `npm install` in scripts/glb_compress",
)
def test_real_draco_compression_preserves_names_and_size():
    """The actual gltf-transform pipeline, not a mock: names, order and bbox survive."""
    from app.services.glb_compression_service import glb_compression_service

    original_bytes = make_glb(materials=("Seat", "Legs", "Unused"), meshes=("Sofa", "Cushion"))
    compressed_bytes = glb_compression_service.compress(original_bytes)

    original = inspect_glb(original_bytes)
    compressed = inspect_glb(compressed_bytes)
    assert compressed.draco_compressed is True
    assert compressed.same_names_as(original)
    assert compressed.width_m == pytest.approx(original.width_m, abs=1e-3)
    assert compressed.height_m == pytest.approx(original.height_m, abs=1e-3)
    assert compressed.depth_m == pytest.approx(original.depth_m, abs=1e-3)
