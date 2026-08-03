-- =====================================================================
--  3D generation timing stats
--
--  fal.ai does NOT return an estimated time (verified against the live
--  queue API: the status payload carries only status / queue_position /
--  logs / metrics, and `metrics` stays empty until the run completes).
--
--  So the estimate is derived from OUR OWN measured history instead: one
--  row per finished generation, and the ETA is the median of a model's
--  recent runs. That is more accurate than any fixed number because it
--  tracks the real conditions — fal's current queue depth, our network,
--  Draco compression and the Azure upload all included.
--
--  Creates ONE new table. Does not modify any existing table.
--  Safe to re-run.
-- =====================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS tbl_model_generation_stats (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Registry key: 'tripo' | 'hunyuan' | ...
    model_key         TEXT    NOT NULL,

    -- Wall-clock seconds from "generation started" to "GLB stored".
    -- Deliberately end-to-end rather than fal's inference_time: this is the
    -- number that predicts what the seller actually waits for.
    duration_seconds  INTEGER NOT NULL,

    -- Failures are recorded too, but excluded from the estimate — a run that
    -- died after 5s would drag the median down and understate the wait.
    succeeded         BOOLEAN NOT NULL DEFAULT TRUE,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT ck_generation_duration_positive CHECK (duration_seconds >= 0)
);

-- The estimate query: most recent N successful runs for one model.
CREATE INDEX IF NOT EXISTS ix_generation_stats_model_recent
    ON tbl_model_generation_stats (model_key, created_at DESC)
    WHERE succeeded;

COMMIT;


-- =====================================================================
--  VERIFY
-- =====================================================================
-- SELECT model_key,
--        count(*)                                        AS runs,
--        round(avg(duration_seconds))                    AS avg_s,
--        percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_seconds) AS median_s
--   FROM tbl_model_generation_stats
--  WHERE succeeded
--  GROUP BY model_key;


-- =====================================================================
--  ROLLBACK (development only)
-- =====================================================================
-- DROP TABLE IF EXISTS tbl_model_generation_stats;
