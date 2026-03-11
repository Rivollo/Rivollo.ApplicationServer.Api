-- STEP 1: Add the exact integer columns for the limits you currently use
ALTER TABLE public.tbl_license_assignments
ADD COLUMN IF NOT EXISTS limit_max_products int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS limit_max_ai_credits int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS limit_max_public_views int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS limit_max_galleries int4 DEFAULT 0,

ADD COLUMN IF NOT EXISTS usage_products int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS usage_ai_credits int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS usage_public_views int4 DEFAULT 0,
ADD COLUMN IF NOT EXISTS usage_galleries int4 DEFAULT 0;

-- STEP 2: One-time migration of your existing JSON data into the columns
UPDATE public.tbl_license_assignments
SET 
    limit_max_products = COALESCE((limits::json->>'max_products')::int4, 0),
    limit_max_ai_credits = COALESCE((limits::json->>'max_ai_credits_month')::int4, 0),
    limit_max_public_views = COALESCE((limits::json->>'max_public_views')::int4, 0),
    limit_max_galleries = COALESCE((limits::json->>'max_galleries')::int4, 0),
    
    usage_products = COALESCE((usage_counters::json->>'max_products')::int4, 0),
    usage_ai_credits = COALESCE((usage_counters::json->>'ai_credits')::int4, 0),
    usage_public_views = COALESCE((usage_counters::json->>'public_views')::int4, 0),
    usage_galleries = COALESCE((usage_counters::json->>'galleries')::int4, 0);

-- STEP 3: Create the Synchronization Function
-- This function runs automatically on every insert or update
CREATE OR REPLACE FUNCTION sync_license_json_to_columns()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only sync if limits JSON is not null
    IF NEW.limits IS NOT NULL THEN
        NEW.limit_max_products = COALESCE((NEW.limits::json->>'max_products')::int4, 0);
        NEW.limit_max_ai_credits = COALESCE((NEW.limits::json->>'max_ai_credits_month')::int4, 0);
        NEW.limit_max_public_views = COALESCE((NEW.limits::json->>'max_public_views')::int4, 0);
        NEW.limit_max_galleries = COALESCE((NEW.limits::json->>'max_galleries')::int4, 0);
    END IF;

    -- Only sync if usage_counters JSON is not null
    IF NEW.usage_counters IS NOT NULL THEN
        NEW.usage_products = COALESCE((NEW.usage_counters::json->>'max_products')::int4, 0);
        NEW.usage_ai_credits = COALESCE((NEW.usage_counters::json->>'ai_credits')::int4, 0);
        NEW.usage_public_views = COALESCE((NEW.usage_counters::json->>'public_views')::int4, 0);
        NEW.usage_galleries = COALESCE((NEW.usage_counters::json->>'galleries')::int4, 0);
    END IF;

    RETURN NEW;
END;
$$;

-- STEP 4: Attach the Function to a DB Trigger
-- If you run this file twice, we drop the trigger first to prevent errors
DROP TRIGGER IF EXISTS trg_sync_license_json ON public.tbl_license_assignments;

CREATE TRIGGER trg_sync_license_json
BEFORE INSERT OR UPDATE OF limits, usage_counters
ON public.tbl_license_assignments
FOR EACH ROW
EXECUTE FUNCTION sync_license_json_to_columns();
