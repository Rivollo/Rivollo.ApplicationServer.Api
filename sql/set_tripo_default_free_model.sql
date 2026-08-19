-- =====================================================================
--  Make Tripo the default direct-image model, and free-eligible
--
--  Only needed if tbl_mstr_3d_models was already created from an earlier
--  copy of sql/create_3d_model_registry.sql (whose seed had SAM 3D as the
--  default and the only free-eligible model). That file's seed INSERT
--  uses ON CONFLICT (key) DO NOTHING, so re-running it will NOT apply
--  this change to rows that already exist — hence this separate script.
--
--  If you have NOT created the table yet: don't run this file. Just run
--  the current sql/create_3d_model_registry.sql — its seed already has
--  Tripo as the default and free-eligible, and SAM 3D as free-eligible
--  but no longer default.
--
--  After this: Tripo is the pre-selected model in the Direct-image picker
--  for every plan, and Free-plan sellers may choose either Tripo or
--  SAM 3D (previously SAM 3D only). No other model's free_plan_eligible
--  changes. No application code change is required for this — the portal
--  and API both resolve the default/free-eligible model(s) from this
--  table, not from a hardcoded key.
--
--  Safe to re-run.
-- =====================================================================

BEGIN;

-- Unset the old default FIRST, then set the new one as a separate
-- statement — tbl_mstr_3d_models has a partial unique index allowing at
-- most one is_default=true row at a time, so doing this in two steps
-- (rather than one UPDATE trying to flip both at once) guarantees the
-- table is never briefly in a two-defaults state that index would reject.
UPDATE tbl_mstr_3d_models SET is_default = false WHERE key = 'sam3';
UPDATE tbl_mstr_3d_models SET is_default = true  WHERE key = 'tripo';

-- Free-plan sellers may now use Tripo too (previously SAM 3D only).
-- SAM 3D's own free_plan_eligible is untouched — still true.
UPDATE tbl_mstr_3d_models SET free_plan_eligible = true WHERE key = 'tripo';

-- Picker order: Tripo first (it's now the default), SAM 3D second.
UPDATE tbl_mstr_3d_models SET order_index = 0 WHERE key = 'tripo';
UPDATE tbl_mstr_3d_models SET order_index = 1 WHERE key = 'sam3';

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT key, label, credit_cost, is_default, free_plan_eligible, order_index
--   FROM tbl_mstr_3d_models
--  ORDER BY order_index;
--
-- Expect: tripo  is_default=true   free_plan_eligible=true   order_index=0
--         sam3   is_default=false  free_plan_eligible=true   order_index=1
--         (hunyuan/trellis/meshy unchanged)


-- =====================================================================
--  ROLLBACK — restores SAM 3D as the default and the only free model
-- =====================================================================
-- BEGIN;
-- UPDATE tbl_mstr_3d_models SET is_default = false WHERE key = 'tripo';
-- UPDATE tbl_mstr_3d_models SET is_default = true  WHERE key = 'sam3';
-- UPDATE tbl_mstr_3d_models SET free_plan_eligible = false WHERE key = 'tripo';
-- UPDATE tbl_mstr_3d_models SET order_index = 0 WHERE key = 'sam3';
-- UPDATE tbl_mstr_3d_models SET order_index = 1 WHERE key = 'tripo';
-- COMMIT;
