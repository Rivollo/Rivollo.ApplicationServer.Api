-- PHASE 4: Final Cleanup
-- Since the application now relies completely on the normalized 'tbl_plan_features'
-- table, the legacy 'quotas' JSON column is no longer needed.

ALTER TABLE public.tbl_mstr_plans 
DROP COLUMN quotas;
