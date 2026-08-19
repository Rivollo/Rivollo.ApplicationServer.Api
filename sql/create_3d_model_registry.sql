-- =====================================================================
--  DB-driven 3D-generation model registry
--
--  Replaces the hardcoded FAL_MODELS dict in
--  app/integrations/fal/registry.py. From this point on, adding a model
--  on an existing provider (another fal.ai model), repricing one, changing
--  its default-plan estimate, reordering the picker, or disabling a model
--  is a data change in this table — no code change, no deploy.
--
--  `provider` is included even though every current row is 'fal_queue':
--  a genuinely different protocol (Tripo's own direct API — bearer auth,
--  a code==0 success envelope, real progress polling, a two-stage chained
--  pipeline) cannot be expressed as request-body JSON, so it will need its
--  own driver and its own `provider` value later. That is a code change
--  when it happens either way — this column just means it won't also need
--  a second migration or a table rename.
--
--  Creates ONE new table + one seed seller of 5 rows. Does not modify any
--  existing table. Safe to re-run (CREATE ... IF NOT EXISTS, INSERT ...
--  ON CONFLICT DO NOTHING) — re-running this file after rows have already
--  been hand-edited will NOT overwrite those edits.
-- =====================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS tbl_mstr_3d_models (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Stable identifier used by the API and stored against generations
    -- (ModelGenerationStat.model_key, product creation requests).
    key                         TEXT    NOT NULL UNIQUE,

    -- Which client/driver knows how to call this model. Always 'fal_queue'
    -- today — see the module docstring above.
    provider                    TEXT    NOT NULL DEFAULT 'fal_queue',

    -- Shown to sellers in the portal's model picker.
    label                       TEXT    NOT NULL,
    description                 TEXT    NOT NULL DEFAULT '',

    -- fal.ai queue endpoint, e.g. 'tripo3d/h3.1/image-to-3d'.
    endpoint_id                 TEXT    NOT NULL,

    -- AI credits charged per generation.
    credit_cost                 INTEGER NOT NULL,

    -- Seed ETA in seconds, used only until this model has enough real runs
    -- recorded (tbl_model_generation_stats) to compute a median.
    baseline_estimate_seconds   INTEGER NOT NULL,

    -- Upper bound on the fal poll loop. Most models finish inside the
    -- default; Meshy documents 5-10 minutes, which would race a 600s ceiling.
    max_wait_seconds            INTEGER NOT NULL DEFAULT 600,

    -- Whether Free-plan sellers may use this model on the direct-image path.
    free_plan_eligible          BOOLEAN NOT NULL DEFAULT false,

    -- Pre-selected model. Exactly one row should carry this — enforced by
    -- the partial unique index below, not just by convention.
    is_default                  BOOLEAN NOT NULL DEFAULT false,

    -- Display order in the picker.
    order_index                 INTEGER NOT NULL DEFAULT 0,

    -- Soft-disable without deleting: an inactive model drops out of the
    -- picker and can't be chosen for a NEW generation, but still resolves
    -- for an in-flight job or a historical stats/ETA lookup that already
    -- references its key.
    isactive                    BOOLEAN NOT NULL DEFAULT true,

    -- Provider-specific request/response shape — see the column comment
    -- applied below, and the module docstring in
    -- app/models/model_registry.py, for exactly what keys this holds.
    provider_config             JSONB   NOT NULL DEFAULT '{}'::jsonb,

    -- Hard-won operational knowledge that used to live only as Python
    -- comments — preserved here so the migration doesn't destroy it.
    notes                       TEXT,

    created_by                  UUID,
    created_date                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by                  UUID,
    updated_date                TIMESTAMPTZ
);

COMMENT ON COLUMN tbl_mstr_3d_models.provider_config IS
    'For provider=fal_queue: {image_url_field, request_body_template, glb_url_paths, usdz_url_paths}. '
    'glb_url_paths/usdz_url_paths are ordered dot-paths tried against the fal result JSON; a segment '
    'suffixed "[]" means "iterate this list, first match wins" (needed by sam3''s individual_glbs fallback).';

-- At most one default row, database-enforced.
CREATE UNIQUE INDEX IF NOT EXISTS ux_3d_models_single_default
    ON tbl_mstr_3d_models (is_default)
    WHERE is_default;

-- Picker order + "is this a real active model" filter both hit this shape.
CREATE INDEX IF NOT EXISTS ix_3d_models_active_order
    ON tbl_mstr_3d_models (order_index)
    WHERE isactive;


-- =====================================================================
--  Seed data — the 5 models live in app/integrations/fal/registry.py
--  today. Values transcribed 1:1 from that module (build_body /
--  extract_glb_url functions and their FalModelSpec entries) and verified
--  against it before this file was finalized. ON CONFLICT DO NOTHING so
--  re-running this file is always safe.
-- =====================================================================

INSERT INTO tbl_mstr_3d_models (
    key, provider, label, description, endpoint_id, credit_cost,
    baseline_estimate_seconds, max_wait_seconds, free_plan_eligible,
    is_default, order_index, isactive, provider_config, notes
) VALUES
(
    'sam3', 'fal_queue', 'SAM 3D', 'Whole-image reconstruction',
    'fal-ai/sam-3/3d-objects', 20, 180, 600, true, true, 0, true,
    '{
        "image_url_field": "image_url",
        "request_body_template": {
            "export_textured_glb": true,
            "prompt": null
        },
        "glb_url_paths": ["model_glb.url", "individual_glbs[].url"],
        "usdz_url_paths": []
    }'::jsonb,
    'The one direct-image model Free-plan sellers may use, and the '
    'registry default — listed first for both reasons. Priced the same '
    'on this direct path and the segmented /createProduct path (both hit '
    'fal-ai/sam-3/3d-objects) — see SAM3_PRODUCT_CREATION_AI_CREDIT_COST '
    'usage in app/api/routes/products.py, which now reads this same row '
    'instead of its own constant. '
    '"prompt" defaults to "car" upstream and drives auto-segmentation — '
    'wrong for an arbitrary product photo, so it is nulled explicitly '
    '(not omitted) so it cannot compete with the direct-image path''s '
    'lack of a mask. No independent duration measurement yet for this '
    'no-mask request shape; seeded from the segmented path''s history.'
),
(
    'tripo', 'fal_queue', 'Tripo', 'Fast and reliable',
    'tripo3d/h3.1/image-to-3d', 100, 180, 600, false, false, 1, true,
    '{
        "image_url_field": "image_url",
        "request_body_template": {
            "texture": true,
            "pbr": true,
            "texture_quality": "detailed",
            "geometry_quality": "detailed",
            "texture_alignment": "original_image",
            "orientation": "align_image",
            "face_limit": 50000
        },
        "glb_url_paths": ["model_urls.glb.url", "model_mesh.url"],
        "usdz_url_paths": []
    }'::jsonb,
    '"quad" topology is deliberately never set: Tripo''s docs warn quad '
    'topology makes it return FBX bytes instead of GLB, breaking every '
    'downstream step (viewer, colour configurator, USDZ conversion). '
    '"face_limit" is capped at 50000 — left unset, Tripo "adaptively '
    'determines the count" and on a real product photo that meant '
    '2,000,000 triangles / 1,035,799 vertices: a 60.8MB file whose '
    'geometry alone was ~56MB, and slow enough to drop the viewer''s '
    'frame rate. 50k keeps Tripo in line with Trellis (~36k verts) and '
    'Meshy (~29k), which render smoothly; surface detail comes from '
    'textures, which stay "detailed". Measured at 165s end-to-end on a '
    'real product photo with face_limit applied.'
),
(
    'hunyuan', 'fal_queue', 'Hunyuan', 'Full PBR textures',
    'fal-ai/hunyuan-3d/v3.1/pro/image-to-3d', 100, 240, 600, false, false, 2, true,
    '{
        "image_url_field": "input_image_url",
        "request_body_template": {
            "generate_type": "Normal",
            "enable_pbr": true,
            "face_count": 500000
        },
        "glb_url_paths": ["model_glb.url", "model_urls.glb.url"],
        "usdz_url_paths": []
    }'::jsonb,
    'Unlike every other model here, the image field is "input_image_url", '
    'not "image_url" — this is exactly what provider_config.image_url_field '
    'exists to handle per-row instead of assuming one universal key name. '
    '"generate_type": "Normal" produces a textured model; "Geometry" '
    'would return an untextured white mesh the configurator cannot '
    'recolour. PBR is always on: the colour configurator recolours the '
    'base-colour map while preserving normal/roughness/metallic detail, '
    'and without PBR maps a recoloured part looks like flat paint. '
    'face_count 500000 is fal''s own default (range 40,000-1,500,000). '
    'Measured end-to-end at ~202s (submit -> GLB downloaded) on a live run.'
),
(
    'trellis', 'fal_queue', 'Trellis', 'Sharpest detail',
    'fal-ai/trellis-2', 100, 180, 600, false, false, 3, true,
    '{
        "image_url_field": "image_url",
        "request_body_template": {
            "resolution": 1536,
            "ss_guidance_strength": 7.5,
            "ss_guidance_rescale": 0.7,
            "ss_guidance_interval_start": 0.6,
            "ss_guidance_interval_end": 1,
            "ss_sampling_steps": 12,
            "ss_rescale_t": 5,
            "shape_slat_guidance_strength": 7.5,
            "shape_slat_guidance_rescale": 0.5,
            "shape_slat_guidance_interval_start": 0.6,
            "shape_slat_guidance_interval_end": 1,
            "shape_slat_sampling_steps": 12,
            "shape_slat_rescale_t": 3,
            "tex_slat_guidance_strength": 1,
            "tex_slat_guidance_rescale": 0,
            "tex_slat_guidance_interval_start": 0.6,
            "tex_slat_guidance_interval_end": 0.9,
            "tex_slat_sampling_steps": 12,
            "tex_slat_rescale_t": 3,
            "decimation_target": 50000,
            "texture_size": 4096,
            "remesh": true,
            "remesh_band": 1,
            "remesh_project": 0,
            "uv_unwrap_angle_threshold_deg": 90,
            "uv_unwrap_refine_iterations": 0,
            "uv_unwrap_global_iterations": 1,
            "uv_unwrap_smooth_strength": 1
        },
        "glb_url_paths": ["model_glb.url"],
        "usdz_url_paths": []
    }'::jsonb,
    'The guidance/sampling parameters are pinned to fal''s documented '
    'defaults rather than omitted — Trellis exposes ~25 knobs, and '
    'silently changing one of their defaults would change every model '
    'generated; pinning keeps output reproducible. Two settings '
    'deliberately depart from the API defaults, both validated against '
    'a real product photo: "decimation_target": 50000, NOT the API '
    'default of 500000 — fal''s own docs say "500k is good for most '
    'uses, reduce to 20k-50k for web/mobile", and 500k does not merely '
    'produce a heavy file, it makes the remesh/UV-unwrap stage fail '
    'outright (a real product photo returned HTTP 500 after 404s with '
    'every sampling stage already complete; the same image at 50k '
    'succeeds). "texture_size": 4096, above the API default of 2048 — '
    'with geometry decimated for the web, fine surface detail has to '
    'come from the texture rather than triangle count. Measured at '
    '4.5MB total. Seed baseline from live runs on a real product photo: '
    '241s and 57s for the same input on different runners — fal''s '
    'queue variance is wide, so 180s sits between them.'
),
(
    'meshy', 'fal_queue', 'Meshy', 'Best overall quality',
    'fal-ai/meshy/v6/image-to-3d', 200, 360, 1200, false, false, 4, true,
    '{
        "image_url_field": "image_url",
        "request_body_template": {
            "model_type": "standard",
            "topology": "triangle",
            "target_polycount": 30000,
            "symmetry_mode": "auto",
            "should_remesh": true,
            "should_texture": true,
            "enable_pbr": true,
            "enable_rigging": false,
            "enable_animation": false,
            "enable_safety_checker": true
        },
        "glb_url_paths": ["model_glb.url", "model_urls.glb.url"],
        "usdz_url_paths": ["model_urls.usdz.url"]
    }'::jsonb,
    'Rigging and animation are deliberately OFF: fal''s own example '
    'enables them, but Meshy''s docs say rigging targets "humanoid '
    'characters with clearly defined limbs" — a chair or a shoe has '
    'none, so it adds minutes of work and returns nulls (fal''s own '
    'sample output shows rig_task_id: null). Turn them on only if '
    'Rivollo ever sells character models. "model_type": "standard", not '
    '"lowpoly", which would discard the detail being paid for. '
    'max_wait_seconds is 1200, not the 600 default — Meshy documents '
    '5-10 minutes, which would race a 600s poll ceiling; a live run on '
    'a real product photo took 204s, seeded between the two rather than '
    'trusting either alone. Meshy exports USDZ itself alongside the '
    'GLB (usdz_url_paths non-empty) — when present, that file is '
    'stored directly and the Azure GLB->USDZ conversion job is skipped '
    'entirely, since the vendor''s own export beats a converted one on '
    'both speed and fidelity.'
)
ON CONFLICT (key) DO NOTHING;

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT key, provider, label, credit_cost, is_default, free_plan_eligible,
--        isactive, order_index
--   FROM tbl_mstr_3d_models
--  ORDER BY order_index;
--
-- -- Confirm the partial unique index actually blocks a second default:
-- -- (run in a transaction you intend to ROLLBACK)
-- -- UPDATE tbl_mstr_3d_models SET is_default = true WHERE key = 'tripo';
-- -- should raise: duplicate key value violates unique constraint
-- -- "ux_3d_models_single_default"


-- =====================================================================
--  ROLLBACK (development only — this drops real pricing config)
-- =====================================================================
-- DROP TABLE IF EXISTS tbl_mstr_3d_models;
