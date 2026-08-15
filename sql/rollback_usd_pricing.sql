-- Rollback for sql/add_usd_pricing.sql.
--
-- Run this only to undo the USD migration on a dev database. It is destructive:
-- dropping the columns discards the currency, promo and amount history of every
-- USD subscription that was created while they existed. There is no way to
-- recover that from the INR columns, because it was never stored there.
--
-- Before running this against anything with real subscriptions, deploy the
-- previous application image first. The current image reads these columns on
-- every subscription lookup and will fail once they are gone.
--
-- Every statement is idempotent, so a partially-applied rollback can be re-run.

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Safety check: refuse to run while USD subscriptions exist
-- ─────────────────────────────────────────────────────────────────────────────
-- A USD subscription that is still live in Razorpay will keep sending webhooks.
-- Once `currency` is gone, is_usd_subscription() can no longer recognise it, so
-- those events fall through to the INR handlers and would write INR amounts
-- against a dollar charge. Cancel the subscriptions in the Razorpay dashboard
-- first, then re-run.

DO $$
DECLARE
    live_usd INTEGER;
BEGIN
    SELECT count(*) INTO live_usd
    FROM information_schema.columns
    WHERE table_name = 'tbl_subscriptions' AND column_name = 'currency';

    IF live_usd > 0 THEN
        -- status is a non-native enum stored as text. Compared case-insensitively
        -- because SQLAlchemy persists the member name ('ACTIVE') while the enum's
        -- value is lowercase ('active') — this must not depend on which one is in
        -- the column.
        SELECT count(*) INTO live_usd
        FROM tbl_subscriptions
        WHERE currency = 'USD'
          AND lower(status::text) IN ('active', 'pending', 'trialing');

        IF live_usd > 0 THEN
            RAISE EXCEPTION
                'Refusing to roll back: % USD subscription(s) are still live. '
                'Cancel them in the Razorpay dashboard first — their webhooks '
                'would otherwise be processed as INR.', live_usd;
        END IF;
    END IF;
END $$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Drop the currency-aware subscription columns
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS currency;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS billing_country;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS promo_code;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS upfront_amount;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS full_amount;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS promo_period_active;
ALTER TABLE tbl_subscriptions DROP COLUMN IF EXISTS start_at;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Drop the USD tables
-- ─────────────────────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS tbl_promo_codes_usd;
DROP TABLE IF EXISTS tbl_plan_prices_usd;

-- tbl_payments.currency is deliberately NOT dropped. It predates this work —
-- the up-migration only adds it so an older snapshot converges, and the existing
-- INR payment path writes to it.

COMMIT;
