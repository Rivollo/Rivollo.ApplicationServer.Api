-- =====================================================================
--  Product Configurator — generated DDL reference
--
--  🔴 THIS FILE IS NOT THE DEPLOYMENT MECHANISM. DO NOT RUN IT BY HAND.
--
--  Alembic owns all Configurator DDL (ADR-009). This file exists only so
--  the schema can be READ and REVIEWED without running Alembic — for a
--  design review, a DBA read-through, or the Rivollo.AccountPurge.Job
--  contract update described below.
--
--  Apply the schema with Alembic, never with psql:
--
--      alembic upgrade head
--
--  Running this script by hand is exactly how tbl_product_color_variants
--  and tbl_variant_assets ended up outside the migration chain
--  (sql/create_color_variants.sql), so that a fresh environment brought
--  up with `alembic upgrade head` does not have them at all. Do not
--  repeat that.
--
--  ---------------------------------------------------------------------
--  Provenance — regenerate, do not edit
--  ---------------------------------------------------------------------
--  Generated from migrations/versions/c7a4e0d51b83_add_configurator_tables.py
--  with Alembic's offline mode. If the migration changes, regenerate:
--
--      alembic upgrade b8e2f4a10c73:c7a4e0d51b83 --sql
--
--  Hand-editing this file makes it disagree with the migration, which is
--  worse than not having it.
--
--  Revision:      c7a4e0d51b83
--  Down revision: b8e2f4a10c73  (add_login_otps)
--  Target:        PostgreSQL (pgcrypto already installed by 6317c2563d0b)
--
--  ---------------------------------------------------------------------
--  What this creates
--  ---------------------------------------------------------------------
--    tbl_product_parts         a seller-named region of a product, owning
--                              a set of glTF material indices (JSONB)
--    tbl_part_options          a shopper-selectable appearance; `recipe`
--                              covers both a generated recolour and an
--                              uploaded texture, discriminated by
--                              recipe.method (ADR-013)
--    tbl_part_option_textures  the baked texture per material index —
--                              pure cache, fully regenerable
--
--  Deliberately absent, each for a documented reason:
--    * no default_option_id column — the default is is_default on the
--      option, guarded by ux_part_options_one_default (ADR-011)
--    * no option_type / source_type column (ADR-013)
--    * no GIN index and no JSONB CHECK constraints (ADR-012)
--    * no FK from created_by / updated_by to tbl_users (ADR-010)
--
--  ---------------------------------------------------------------------
--  🔴 DEPLOYMENT BLOCKER — read before applying to production
--  ---------------------------------------------------------------------
--  tbl_product_parts.product_id -> tbl_products(id) is a NEW foreign key
--  referencing tbl_products. ACCOUNT_PURGE_JOB_HANDOFF.md section 25,
--  schema contract assertion 9, ABORTS EVERY PRODUCTION PURGE RUN on an
--  unrecognised FK to tbl_users or tbl_products.
--
--  Before this reaches production, Rivollo.AccountPurge.Job must:
--    1. allow-list this FK in assertion 9;
--    2. add all three tables to its section 4.2 inventory and section 7
--       cascade list (they cascade from DELETE FROM tbl_products);
--    3. add the product-scoped blob prefix
--       {container}/configurator/{product_id}/... to its section 15
--       deletion order.
--
--  The FK is kept, not removed: the purge deletes products before their
--  owner, and without the cascade that DELETE fails outright — the same
--  failure revision b3f8d21c4a76 exists to fix. See ADR-010.
-- =====================================================================


-- =====================================================================
--  UPGRADE  (b8e2f4a10c73 -> c7a4e0d51b83)
-- =====================================================================

BEGIN;

-- Running upgrade b8e2f4a10c73 -> c7a4e0d51b83

CREATE TABLE tbl_product_parts (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    product_id UUID NOT NULL, 
    name TEXT NOT NULL, 
    slug TEXT NOT NULL, 
    material_indices JSONB DEFAULT '[]'::jsonb NOT NULL, 
    material_type TEXT, 
    order_index INTEGER DEFAULT 0 NOT NULL, 
    shopper_selectable BOOLEAN DEFAULT true NOT NULL, 
    glb_version TEXT NOT NULL, 
    isactive BOOLEAN DEFAULT true NOT NULL, 
    created_by UUID, 
    created_date TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    updated_by UUID, 
    updated_date TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT fk_parts_product FOREIGN KEY(product_id) REFERENCES tbl_products (id) ON DELETE CASCADE, 
    CONSTRAINT uq_parts_product_slug UNIQUE (product_id, slug)
);

CREATE INDEX ix_parts_product_order ON tbl_product_parts (product_id, order_index);

CREATE INDEX ix_parts_product_active ON tbl_product_parts (product_id) WHERE isactive AND shopper_selectable;

CREATE TABLE tbl_part_options (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    part_id UUID NOT NULL, 
    name TEXT NOT NULL, 
    slug TEXT NOT NULL, 
    swatch_hex TEXT NOT NULL, 
    recipe JSONB DEFAULT '{}'::jsonb NOT NULL, 
    recipe_hash TEXT NOT NULL, 
    order_index INTEGER DEFAULT 0 NOT NULL, 
    is_default BOOLEAN DEFAULT false NOT NULL, 
    isactive BOOLEAN DEFAULT true NOT NULL, 
    bake_status TEXT DEFAULT 'pending' NOT NULL, 
    bake_error TEXT, 
    bake_started_at TIMESTAMP WITH TIME ZONE, 
    bake_completed_at TIMESTAMP WITH TIME ZONE, 
    bake_attempts INTEGER DEFAULT 0 NOT NULL, 
    created_by UUID, 
    created_date TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    updated_by UUID, 
    updated_date TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT fk_options_part FOREIGN KEY(part_id) REFERENCES tbl_product_parts (id) ON DELETE CASCADE, 
    CONSTRAINT uq_options_part_slug UNIQUE (part_id, slug), 
    CONSTRAINT ck_options_swatch_hex CHECK (swatch_hex ~* '^#[0-9A-F]{6}$'), 
    CONSTRAINT ck_options_bake_status CHECK (bake_status IN ('pending', 'baking', 'completed', 'failed'))
);

CREATE INDEX ix_options_part_order ON tbl_part_options (part_id, order_index);

CREATE UNIQUE INDEX ux_part_options_one_default ON tbl_part_options (part_id) WHERE is_default;

CREATE INDEX ix_options_stale ON tbl_part_options (bake_started_at) WHERE bake_status = 'baking';

CREATE TABLE tbl_part_option_textures (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    option_id UUID NOT NULL, 
    material_index INTEGER NOT NULL, 
    url TEXT NOT NULL, 
    blob_url TEXT, 
    content_type TEXT NOT NULL, 
    width INTEGER, 
    height INTEGER, 
    size_bytes BIGINT, 
    recipe_hash TEXT NOT NULL, 
    created_by UUID, 
    created_date TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    updated_by UUID, 
    updated_date TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT fk_option_textures_option FOREIGN KEY(option_id) REFERENCES tbl_part_options (id) ON DELETE CASCADE, 
    CONSTRAINT uq_option_texture_material UNIQUE (option_id, material_index), 
    CONSTRAINT ck_option_texture_index CHECK (material_index >= 0), 
    CONSTRAINT ck_option_texture_mime CHECK (content_type IN ('image/png', 'image/jpeg'))
);

CREATE INDEX ix_option_textures_option ON tbl_part_option_textures (option_id);

UPDATE alembic_version SET version_num='c7a4e0d51b83' WHERE alembic_version.version_num = 'b8e2f4a10c73';

COMMIT;


-- =====================================================================
--  DOWNGRADE  (c7a4e0d51b83 -> b8e2f4a10c73)
--
--  Drops in reverse dependency order: textures -> options -> parts.
--  Destroys every part, option and baked-texture record. The blobs under
--  {container}/configurator/{product_id}/... are NOT deleted by this —
--  the database has no reach into Azure Blob Storage. Purge those first
--  or they are orphaned and keep costing money.
-- =====================================================================

BEGIN;

-- Running downgrade c7a4e0d51b83 -> b8e2f4a10c73

DROP INDEX ix_option_textures_option;

DROP TABLE tbl_part_option_textures;

DROP INDEX ix_options_stale;

DROP INDEX ux_part_options_one_default;

DROP INDEX ix_options_part_order;

DROP TABLE tbl_part_options;

DROP INDEX ix_parts_product_active;

DROP INDEX ix_parts_product_order;

DROP TABLE tbl_product_parts;

UPDATE alembic_version SET version_num='b8e2f4a10c73' WHERE alembic_version.version_num = 'c7a4e0d51b83';

COMMIT;


-- =====================================================================
--  VERIFY  (after `alembic upgrade head`)
-- =====================================================================
-- Tables present:
--   SELECT table_name FROM information_schema.tables
--    WHERE table_name IN ('tbl_product_parts','tbl_part_options',
--                         'tbl_part_option_textures')
--    ORDER BY table_name;
--
-- The four CHECK constraints and three FKs:
--   SELECT c.conrelid::regclass AS table, c.conname, c.contype,
--          pg_get_constraintdef(c.oid) AS definition
--     FROM pg_constraint c
--    WHERE c.conrelid::regclass::text IN ('tbl_product_parts','tbl_part_options',
--                                         'tbl_part_option_textures')
--      AND c.contype IN ('c','f')
--    ORDER BY 1, 2;
--
-- Both partial indexes, with their WHERE clauses:
--   SELECT indexname, indexdef FROM pg_indexes
--    WHERE indexname IN ('ux_part_options_one_default','ix_options_stale',
--                        'ix_parts_product_active')
--    ORDER BY indexname;
--
-- ASSERTION 9 PRE-FLIGHT — the one FK the purge job must know about.
-- Expect exactly one row: tbl_product_parts.product_id -> tbl_products, 'c'.
-- ('c' = CASCADE, 'a' = NO ACTION, 'n' = SET NULL.)
--   SELECT c.conrelid::regclass AS table, a.attname AS column, c.confdeltype
--     FROM pg_constraint c
--     JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
--    WHERE c.contype = 'f'
--      AND c.confrelid = 'tbl_products'::regclass
--      AND c.conrelid::regclass::text LIKE 'tbl_p%part%';
--
-- No Configurator FK may reference tbl_users. Expect ZERO rows:
--   SELECT c.conrelid::regclass AS table, c.conname
--     FROM pg_constraint c
--    WHERE c.contype = 'f'
--      AND c.confrelid = 'tbl_users'::regclass
--      AND c.conrelid::regclass::text IN ('tbl_product_parts','tbl_part_options',
--                                         'tbl_part_option_textures');
-- =====================================================================
