-- USD billing for customers outside India -- SCHEMA ONLY.
--
-- This script contains no data. It adds columns and widens a constraint;
-- nothing is inserted, and no price or promo is created. The USD rows are
-- yours to write -- USD_ROLLOUT_TODO.md specifies exactly which rows must exist
-- and what each value means, so you can run them from your own client.
--
-- ORDER MATTERS, and this is the reason the data is not in here.
--
-- USD prices live in the existing tbl_plan_prices alongside INR, separated by
-- the `currency` column. The INR lookup in
-- razorpay_subscription_service._get_plan_with_features calls
-- scalar_one_or_none() after filtering on (plan_id, billing_interval), so a
-- second row for the same plan and interval makes it raise
-- MultipleResultsFound. On the image running today that means every rupee
-- checkout for that plan fails outright -- an outage, not a mispricing.
--
-- So: run this, deploy, confirm the new image is serving, and only then insert
-- USD rows. Inserting them first is the one sequence that breaks INR.
--
-- This file itself is safe to run at any time, against a running old or new
-- image. Everything is additive and idempotent: no existing column is altered
-- or dropped, every new column is nullable or defaulted so existing INR rows
-- and existing INR code read correctly, and the widened constraint is strictly
-- more permissive than the one it replaces.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Make tbl_plan_prices multi-currency
-- ─────────────────────────────────────────────────────────────────────────────

-- Already present on the model; here so an older snapshot converges.
ALTER TABLE tbl_plan_prices ADD COLUMN IF NOT EXISTS currency VARCHAR(3) NOT NULL DEFAULT 'INR';

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

-- billing_country is nullable with no default, so existing rows are left NULL
-- by the ALTER above. Backfilling them to 'IN' is a data change and therefore
-- yours to run -- see USD_ROLLOUT_TODO.md. Nothing breaks while they are NULL:
-- the column is only read for reporting, never for currency resolution.

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
