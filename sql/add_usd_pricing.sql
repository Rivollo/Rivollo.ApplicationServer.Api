-- USD billing for customers outside India -- STEP 1 of 2: schema only.
--
--     1. sql/add_usd_pricing.sql   (this file)  -- run FIRST, before deploying
--     2. sql/seed_usd_pricing.sql               -- run LAST, after deploying
--
-- The split is not tidiness. USD prices live in the existing tbl_plan_prices
-- alongside INR, separated by the `currency` column, and the INR lookup in
-- razorpay_subscription_service._get_plan_with_features calls
-- scalar_one_or_none() after filtering on (plan_id, billing_interval). A second
-- row for the same plan and interval makes that raise MultipleResultsFound.
--
-- So a USD price row is safe only once the deployed code filters on currency.
-- Insert one while the old image is still serving and every rupee checkout for
-- that plan fails -- an outage, not a mispricing. Seeding is therefore its own
-- file, to be run after the deploy rather than before it.
--
-- This file is safe to run at any time, against a running old or new image.
-- Everything in it is additive and idempotent: no existing column is altered or
-- dropped, every new column is nullable or defaulted so existing INR rows and
-- existing INR code read correctly, and the widened constraint is strictly
-- more permissive than the one it replaces.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Make tbl_plan_prices multi-currency
-- ─────────────────────────────────────────────────────────────────────────────

-- Already present on the model; here so an older snapshot converges.
ALTER TABLE tbl_plan_prices ADD COLUMN IF NOT EXISTS currency VARCHAR(3) NOT NULL DEFAULT 'INR';

UPDATE tbl_plan_prices SET currency = 'INR' WHERE currency IS NULL OR currency = '';

-- The old constraint permits only one row per (plan, interval), which is
-- exactly one currency. Widen it rather than drop it: without a uniqueness rule
-- a duplicate USD row would make the USD lookup ambiguous in the same way.
ALTER TABLE tbl_plan_prices
    DROP CONSTRAINT IF EXISTS tbl_plan_prices_plan_interval_key;

ALTER TABLE tbl_plan_prices
    DROP CONSTRAINT IF EXISTS tbl_plan_prices_plan_interval_currency_key;

ALTER TABLE tbl_plan_prices
    ADD CONSTRAINT tbl_plan_prices_plan_interval_currency_key
    UNIQUE (plan_id, billing_interval, currency);

COMMENT ON COLUMN tbl_plan_prices.price_inr IS
    'Price in WHOLE units of `currency` -- rupees for INR rows, dollars for USD '
    'rows. The column name predates multi-currency support and is kept because '
    'renaming it would touch every INR read path. Whole units only: 1999 is '
    'valid, 19.99 is not representable. A price ending in .99 would need a '
    'schema change, not just a different value.';

COMMENT ON COLUMN tbl_plan_prices.currency IS
    'ISO currency of this price row. Every read MUST filter on it -- two rows '
    'now exist per (plan, interval), and an unfiltered lookup matches both.';

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Make tbl_promo_codes multi-currency
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE tbl_promo_codes ADD COLUMN IF NOT EXISTS currency VARCHAR(3) NOT NULL DEFAULT 'INR';

-- Marks the promo advertised on the pricing page. It is auto-applied at
-- checkout when the customer submits no code, so the price shown and the price
-- charged cannot drift apart.
ALTER TABLE tbl_promo_codes ADD COLUMN IF NOT EXISTS is_public BOOLEAN NOT NULL DEFAULT false;

UPDATE tbl_promo_codes SET currency = 'INR' WHERE currency IS NULL OR currency = '';

COMMENT ON COLUMN tbl_promo_codes.currency IS
    'ISO currency this promo applies to. INR promos are applied through a '
    'Razorpay Offer (razorpay_offer_id); USD promos are computed server-side '
    'and charged as a subscription addon, because Offers are INR-locked on '
    'this account and fail silently against a USD plan.';

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Currency-aware subscription columns
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS currency            VARCHAR(3) NOT NULL DEFAULT 'INR';
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS billing_country     VARCHAR(2) NULL;
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS promo_code          VARCHAR(64) NULL;
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS upfront_amount      BIGINT NULL;
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS full_amount         BIGINT NULL;
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS promo_period_active BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE tbl_subscriptions ADD COLUMN IF NOT EXISTS start_at            TIMESTAMPTZ NULL;

COMMENT ON COLUMN tbl_subscriptions.currency IS
    'Currency this subscription bills in. Locked at first subscription and never '
    'changed -- a customer who subscribes in USD keeps paying USD even if they '
    'later browse from India, and vice versa.';

COMMENT ON COLUMN tbl_subscriptions.upfront_amount IS
    'Amount actually charged at the authentication transaction, in the smallest '
    'unit of `currency`. Stored rather than recomputed from the promo percentage '
    'so a future price change cannot make historical records lie.';

COMMENT ON COLUMN tbl_subscriptions.promo_period_active IS
    'True between subscription.authenticated and the first full-price charge -- '
    'i.e. while the customer is inside the discounted first period.';

-- Existing rows all predate USD support.
UPDATE tbl_subscriptions SET billing_country = 'IN' WHERE billing_country IS NULL;

-- The webhook resolves a subscription's currency by looking it up on
-- razorpay_subscription_id. No index is added for it here: the existing
-- idx_subscriptions_razorpay_sub_id already covers that column, and a second
-- index on the same column would be write overhead for no read benefit.

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Currency on payments
-- ─────────────────────────────────────────────────────────────────────────────
-- The column already exists on the model with a default; this is here only so a
-- database provisioned from an older snapshot converges.

ALTER TABLE tbl_payments ADD COLUMN IF NOT EXISTS currency VARCHAR(3) NOT NULL DEFAULT 'INR';
