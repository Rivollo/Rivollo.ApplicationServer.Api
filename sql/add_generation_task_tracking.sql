-- =====================================================================
--  External generation-task tracking on tbl_jobs
--
--  The Tripo parts pipeline runs TWO chained tasks: geometry, then texture.
--  Geometry is the expensive half. If texturing fails, the geometry task has
--  already succeeded and already been paid for — regenerating it would burn
--  credits for work that is sitting on Tripo's servers, complete.
--
--  Storing the geometry task id lets a retry resume from stage 2. Same
--  principle as keeping `overrides` on tbl_product_color_variants: persist the
--  cheap intermediate so the expensive artifact never has to be rebuilt.
--
--  Adds 3 nullable columns to an EXISTING, EMPTY table (tbl_jobs, 0 rows).
--  Nullable + no defaults, so nothing that writes to tbl_jobs today changes
--  behaviour. Safe to re-run.
-- =====================================================================

BEGIN;

ALTER TABLE tbl_jobs
    -- Provider-side task id to resume from. For the Tripo parts pipeline this
    -- is the STAGE 1 (geometry) task, because that is the half worth saving.
    ADD COLUMN IF NOT EXISTS external_task_id TEXT,
    -- Which stage the job reached: 'geometry' | 'texture' | 'complete'.
    -- Distinguishes "geometry done, texturing failed" (resumable) from
    -- "geometry failed" (must restart).
    ADD COLUMN IF NOT EXISTS stage TEXT,
    -- Provider credits consumed, summed across stages. Tripo bills separately
    -- from our own AI credits, and the real cost is only knowable from the
    -- task responses.
    ADD COLUMN IF NOT EXISTS provider_credits INTEGER;

-- Resume lookup: "the most recent resumable job for this product".
CREATE INDEX IF NOT EXISTS ix_jobs_product_external_task
    ON tbl_jobs (product_id, created_date DESC)
    WHERE external_task_id IS NOT NULL;

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT column_name, data_type, is_nullable
--   FROM information_schema.columns
--  WHERE table_name = 'tbl_jobs'
--    AND column_name IN ('external_task_id', 'stage', 'provider_credits');


-- =====================================================================
--  ROLLBACK (development only)
-- =====================================================================
-- ALTER TABLE tbl_jobs
--     DROP COLUMN IF EXISTS external_task_id,
--     DROP COLUMN IF EXISTS stage,
--     DROP COLUMN IF EXISTS provider_credits;
