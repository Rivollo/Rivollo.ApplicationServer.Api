-- =====================================================================
--  Color Configurator — schema
--  Run once per environment. Safe to re-run (IF NOT EXISTS everywhere).
--
--  Design
--    tbl_product_color_variants  the colourway RECIPE (source of truth).
--                                Tiny: a name, a swatch and a JSON list of
--                                per-material colour overrides.
--    tbl_variant_assets          the BAKED files derived from that recipe.
--                                Pure cache — every row can be deleted and
--                                regenerated from the recipe above.
--
--  The original product model (tbl_products.model_asset_id -> tbl_assets)
--  is never modified. Baked variants live in their own table so the viewer's
--  main-mesh lookup (tbl_product_assets, asset_id 9/11) stays deterministic.
-- =====================================================================

BEGIN;

-- ---------------------------------------------------------------------
-- 1. Colourways
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_product_color_variants (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id      UUID NOT NULL
                    REFERENCES tbl_products(id) ON DELETE CASCADE,

    -- Presentation
    name            TEXT    NOT NULL,                    -- "Forest Green"
    slug            TEXT    NOT NULL,                    -- "forest-green"
    swatch_hex      TEXT    NOT NULL,                    -- "#22C55E" (viewer dot)

    -- Flags
    is_default      BOOLEAN NOT NULL DEFAULT FALSE,      -- loads first in viewer
    is_original     BOOLEAN NOT NULL DEFAULT FALSE,      -- untouched model, undeletable
    isactive        BOOLEAN NOT NULL DEFAULT TRUE,       -- publish toggle
    order_index     INTEGER NOT NULL DEFAULT 0,

    -- THE RECIPE. Kept even though we bake, so a re-upload of the model or an
    -- improvement to the baker can regenerate every file without the seller
    -- recreating their colourways by hand.
    -- [{ "material_index": 0, "material_name": "Upper",
    --    "color": "#22C55E", "method": "luminance", "brightness": 1.0 }]
    overrides       JSONB   NOT NULL DEFAULT '[]'::jsonb,

    -- sha256(source_asset_id + canonical(overrides) + baker_version).
    -- Changes whenever the output would change -> invalidates stale bakes.
    config_hash     TEXT    NOT NULL,

    -- Bake lifecycle
    bake_status     TEXT    NOT NULL DEFAULT 'pending',  -- pending|baking|ready|failed
    bake_error      TEXT,
    baked_at        TIMESTAMPTZ,

    -- Audit (matches AuditMixin used across the codebase: UUID actor ids)
    created_by      UUID,
    created_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by      UUID,
    updated_date    TIMESTAMPTZ,

    CONSTRAINT ck_variant_bake_status
        CHECK (bake_status IN ('pending', 'baking', 'ready', 'failed')),
    CONSTRAINT ck_variant_swatch_hex
        CHECK (swatch_hex ~* '^#[0-9A-F]{6}$'),
    CONSTRAINT uq_variant_product_slug
        UNIQUE (product_id, slug)
);

-- Listing order in both portals.
CREATE INDEX IF NOT EXISTS ix_color_variants_product_order
    ON tbl_product_color_variants (product_id, order_index);

-- The viewer only ever reads active + ready rows.
CREATE INDEX IF NOT EXISTS ix_color_variants_product_active
    ON tbl_product_color_variants (product_id)
    WHERE isactive AND bake_status = 'ready';

-- At most ONE default per product, enforced by the database rather than code.
CREATE UNIQUE INDEX IF NOT EXISTS ux_color_variants_one_default
    ON tbl_product_color_variants (product_id)
    WHERE is_default;

-- At most ONE original per product.
CREATE UNIQUE INDEX IF NOT EXISTS ux_color_variants_one_original
    ON tbl_product_color_variants (product_id)
    WHERE is_original;


-- ---------------------------------------------------------------------
-- 2. Baked files (derived — safe to purge and rebuild)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_variant_assets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variant_id      UUID NOT NULL
                    REFERENCES tbl_product_color_variants(id) ON DELETE CASCADE,

    format          TEXT   NOT NULL,          -- 'glb' (usdz added by converter jobs)
    url             TEXT   NOT NULL,          -- CDN-fronted, served to clients
    blob_url        TEXT,                     -- raw Azure blob, for re-processing
    size_bytes      BIGINT,

    -- Must equal the parent's config_hash. If it doesn't, this file is stale
    -- and must be ignored until the re-bake completes.
    config_hash     TEXT   NOT NULL,

    created_by      UUID,
    created_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by      UUID,
    updated_date    TIMESTAMPTZ,

    CONSTRAINT ck_variant_asset_format CHECK (format IN ('glb', 'usdz')),
    CONSTRAINT uq_variant_asset_format UNIQUE (variant_id, format)
);

CREATE INDEX IF NOT EXISTS ix_variant_assets_variant
    ON tbl_variant_assets (variant_id);

COMMIT;


-- =====================================================================
--  BACKFILL — give every existing product with a model an "Original"
--  colourway, so the viewer always has at least one swatch to show.
--  Idempotent: skips products that already have one.
--  Original needs no bake — it points at the untouched model.
-- =====================================================================

BEGIN;

INSERT INTO tbl_product_color_variants (
    product_id, name, slug, swatch_hex,
    is_default, is_original, isactive, order_index,
    overrides, config_hash, bake_status, baked_at
)
SELECT
    p.id,
    'Original',
    'original',
    '#FFFFFF',
    TRUE,               -- default
    TRUE,               -- original
    TRUE,               -- active
    0,
    '[]'::jsonb,
    'original',         -- sentinel hash; the original is never baked
    'ready',
    now()
FROM tbl_products p
WHERE p.deleted_at IS NULL
  -- "Has a 3D model" is resolved through tbl_product_assets, the same way the
  -- rest of the app does it. asset_id 9 = GLB mesh.
  -- NOT via tbl_products.model_asset_id -> tbl_assets: that relation survives in
  -- the ORM models but the tbl_assets table does not exist in this database.
  AND EXISTS (
      SELECT 1
      FROM tbl_product_asset_mapping pam
      JOIN tbl_product_assets pa ON pa.id = pam.product_asset_id
      WHERE pam.productid = p.id
        AND pam.isactive = TRUE
        AND pa.asset_id = 9
  )
  AND NOT EXISTS (
      SELECT 1 FROM tbl_product_color_variants v
      WHERE v.product_id = p.id AND v.is_original
  );

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT bake_status, count(*) FROM tbl_product_color_variants GROUP BY 1;
-- SELECT v.name, v.swatch_hex, v.bake_status, a.format, a.url
--   FROM tbl_product_color_variants v
--   LEFT JOIN tbl_variant_assets a ON a.variant_id = v.id
--  WHERE v.product_id = '<product-uuid>'
--  ORDER BY v.order_index;


-- =====================================================================
--  ROLLBACK (development only)
-- =====================================================================
-- DROP TABLE IF EXISTS tbl_variant_assets;
-- DROP TABLE IF EXISTS tbl_product_color_variants;
