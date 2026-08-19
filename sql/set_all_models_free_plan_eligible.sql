-- =====================================================================
--  Make every direct-image model free_plan_eligible, Tripo default/first
--
--  Supersedes sql/set_tripo_default_free_model.sql (which only touched
--  tripo + sam3) — that script was run against the dev database but only
--  partially applied there (is_default flipped, free_plan_eligible and
--  order_index did not), which is exactly the kind of gap this file's own
--  VERIFY step below exists to catch before you assume a change landed.
--
--  After this: Direct image has no Pro/Enterprise gate on any model — every
--  row is free_plan_eligible=true. Tripo stays the paid-plan default
--  (is_default) and is also order_index=0, so it's the first free-eligible
--  model a Free seller is pre-selected into (see the portal's
--  DirectImageEditor.tsx free-plan fallback, which picks the first
--  free_plan_eligible model in registry order — unchanged code, this is
--  purely what feeds it).
--
--  No application code change required for this.
--
--  Safe to re-run.
-- =====================================================================

BEGIN;

UPDATE tbl_mstr_3d_models
   SET free_plan_eligible = true
 WHERE key IN ('sam3', 'tripo', 'hunyuan', 'trellis', 'meshy');

-- Tripo first (order_index 0), so it's the free-plan fallback's first
-- match too, not just the paid-plan is_default. Unset the old default
-- before setting the new one, as two statements rather than one UPDATE
-- attempting both at once — tbl_mstr_3d_models has a partial unique index
-- allowing at most one is_default=true row, so this order guarantees the
-- table is never briefly in a two-defaults state that index would reject.
UPDATE tbl_mstr_3d_models SET is_default = false WHERE key != 'tripo';
UPDATE tbl_mstr_3d_models SET is_default = true  WHERE key = 'tripo';

UPDATE tbl_mstr_3d_models SET order_index = 0 WHERE key = 'tripo';
UPDATE tbl_mstr_3d_models SET order_index = 1 WHERE key = 'sam3';
UPDATE tbl_mstr_3d_models SET order_index = 2 WHERE key = 'hunyuan';
UPDATE tbl_mstr_3d_models SET order_index = 3 WHERE key = 'trellis';
UPDATE tbl_mstr_3d_models SET order_index = 4 WHERE key = 'meshy';

COMMIT;


-- =====================================================================
--  VERIFY — run this and actually read the output. Don't assume the
--  UPDATE above "took" just because it didn't error; a WHERE clause that
--  silently matches zero rows (wrong environment, wrong key spelling)
--  succeeds without complaint too.
-- =====================================================================
-- SELECT key, label, is_default, free_plan_eligible, order_index
--   FROM tbl_mstr_3d_models
--  ORDER BY order_index;
--
-- Expect all 5 rows free_plan_eligible = true, and exactly:
--   tripo    is_default=true   order_index=0
--   sam3     is_default=false  order_index=1
--   hunyuan  is_default=false  order_index=2
--   trellis  is_default=false  order_index=3
--   meshy    is_default=false  order_index=4


-- =====================================================================
--  ROLLBACK — restores Tripo+SAM 3D as the only free models
-- =====================================================================
-- BEGIN;
-- UPDATE tbl_mstr_3d_models SET free_plan_eligible = false WHERE key IN ('hunyuan', 'trellis', 'meshy');
-- COMMIT;
