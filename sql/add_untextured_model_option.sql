-- =====================================================================
--  Solid-body (untextured) option for Tripo, Hunyuan and Meshy
--
--  Adds provider_config.untextured_overrides to the rows whose fal API can
--  return a mesh without texture. POST /createProductFal with
--  "textured": false merges these fields over request_body_template
--  (FalModelSpec.untextured in app/integrations/fal/registry.py). A row
--  without the key reports supports_untextured=false in GET /ai/3d-models
--  and rejects "textured": false with 400.
--
--  Each verified live 2026-11-24 on the same chair photo:
--
--    tripo    texture=false, pbr=false
--             grey material (0.4), no images, ~45k tris, 1.4 MB, 72s
--    hunyuan  generate_type=Geometry, enable_pbr=false, face_count=50000
--             no material/images, 50k tris, 0.6 MB, 86s. face_count is
--             lowered from the textured 500000: at 500k the solid mesh is
--             9 MB with no texture to carry detail. fal range 40k-1.5M.
--    meshy    should_texture=false, enable_pbr=false
--             no material/images, ~27k tris, 1.0 MB, 184s; Meshy's own
--             USDZ comes back untextured too.
--
--  sam3 is deliberately NOT given overrides: fal-ai/sam-3/3d-objects has
--  no geometry-only mode. export_textured_glb=false returns vertex colours
--  (~451k tris) instead of a texture — still coloured, not solid.
--  trellis has no texture-off parameter in its fal schema.
--
--  Only needed where tbl_mstr_3d_models already exists — the seed in
--  sql/create_3d_model_registry.sql carries these keys for new installs.
--  No DDL; touches no other key in provider_config. Safe to re-run.
--  Takes effect within MODEL_REGISTRY_CACHE_TTL_SECONDS (default 60s).
-- =====================================================================

BEGIN;

UPDATE tbl_mstr_3d_models
   SET provider_config = jsonb_set(provider_config, '{untextured_overrides}',
           '{"texture": false, "pbr": false}'::jsonb, true),
       updated_date = now()
 WHERE key = 'tripo';

UPDATE tbl_mstr_3d_models
   SET provider_config = jsonb_set(provider_config, '{untextured_overrides}',
           '{"generate_type": "Geometry", "enable_pbr": false, "face_count": 50000}'::jsonb, true),
       updated_date = now()
 WHERE key = 'hunyuan';

UPDATE tbl_mstr_3d_models
   SET provider_config = jsonb_set(provider_config, '{untextured_overrides}',
           '{"should_texture": false, "enable_pbr": false}'::jsonb, true),
       updated_date = now()
 WHERE key = 'meshy';

COMMIT;

-- VERIFY
-- SELECT key, provider_config->'untextured_overrides' FROM tbl_mstr_3d_models ORDER BY order_index;

-- ROLLBACK
-- UPDATE tbl_mstr_3d_models SET provider_config = provider_config - 'untextured_overrides'
--  WHERE key IN ('tripo', 'hunyuan', 'meshy');
