-- =====================================================================
--  gltf_draco asset type
--
--  Adds ONE reference row to tbl_asset for the Draco-compressed glTF
--  PACKAGE (model.gltf + model.bin + texture files), stored as a single
--  .zip that the client unpacks in the browser before rendering.
--  Does not modify any existing row. Safe to re-run.
--
--  Why a new id instead of reusing 8 ("gltf") or 9 ("glb"):
--    * 9 (glb) keeps serving the Draco GLB. The USDZ converter job, the
--      colour configurator (app/database/color_variant_repo.MESH_ASSET_ID)
--      and every shipped client read it, so it must not change meaning.
--    * 8 (gltf) is the generic, UNCOMPRESSED glTF slot. Reusing it would
--      make "is this asset Draco-compressed?" unanswerable from the id
--      alone, and a viewer that cannot decode Draco has no way to tell.
--
--  Why id 17:
--    tbl_asset currently holds ids 1-11 and 13-16; 16 is the maximum and
--    12 is a gap. 17 is the next free id. The gap at 12 is deliberately
--    NOT reused — a missing id usually means a retired type, and reusing
--    it risks colliding with rows that still reference it elsewhere.
--
--    assetid is the KIND column, not a copy of id: 1 = image view,
--    2 = 3D model. A glTF package is a 3D model, so assetid = 2.
--
--  The application reads this id from settings.GLTF_DRACO_ASSET_ID
--  (default 17). If this script has to allocate a different id in some
--  environment, set GLTF_DRACO_ASSET_ID there to match.
-- =====================================================================

BEGIN;

-- Fail loudly if id 17 is already taken by a DIFFERENT type, rather than
-- silently skipping and leaving the app pointing at the wrong asset.
DO $$
DECLARE
    existing_name TEXT;
BEGIN
    SELECT name INTO existing_name FROM tbl_asset WHERE id = 17;
    IF existing_name IS NOT NULL AND existing_name <> 'gltf_draco' THEN
        RAISE EXCEPTION
            'tbl_asset id 17 is already used by %. Pick a free id, insert it, and set GLTF_DRACO_ASSET_ID to match.',
            existing_name;
    END IF;
END $$;

INSERT INTO tbl_asset (id, assetid, name, description, isactive)
VALUES (
    17,
    2,                              -- kind: 3D model (same as glb / gltf / usdz)
    'gltf_draco',
    'Draco-compressed glTF package (model.gltf + model.bin + textures) as a single .zip, flat at the archive root. URL points at the .zip; the client unpacks it and resolves the glTF relative uris against the unpacked entries.',
    TRUE
)
ON CONFLICT (id) DO NOTHING;

-- tbl_asset.id is autoincrement, so an explicit insert leaves the sequence
-- behind the data and the NEXT inserted row would collide. Only runs if the
-- column actually owns a sequence (a no-op for GENERATED ... AS IDENTITY).
DO $$
DECLARE
    seq_name TEXT := pg_get_serial_sequence('tbl_asset', 'id');
BEGIN
    IF seq_name IS NOT NULL THEN
        PERFORM setval(seq_name, GREATEST((SELECT MAX(id) FROM tbl_asset), 1));
    END IF;
END $$;

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT id, assetid, name, description, isactive
--   FROM tbl_asset
--  WHERE assetid = 2
--  ORDER BY id;
--
-- Expected: 8 gltf | 9 glb | 11 usdz | 14 glbanimation | 15 ply
--           16 voxel_ply | 17 gltf_draco


-- =====================================================================
--  ROLLBACK (development only)
--
--  Deletes the type only. Run the product-asset cleanup FIRST if any
--  gltf_draco rows were already written, or the delete will fail /
--  orphan them:
--    DELETE FROM tbl_product_asset_mapping WHERE product_asset_id IN
--      (SELECT id FROM tbl_product_assets WHERE asset_id = 17);
--    DELETE FROM tbl_product_assets WHERE asset_id = 17;
-- =====================================================================
-- DELETE FROM tbl_asset WHERE id = 17 AND name = 'gltf_draco';
