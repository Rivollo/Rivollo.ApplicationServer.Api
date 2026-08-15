-- USD billing for customers outside India.
-- Run this script manually against the database.
--
-- USD lives in the existing tbl_plan_prices and tbl_promo_codes, distinguished
-- by their `currency` column, rather than in parallel USD tables. Two things
-- had to change for that to be safe, and both are in this script:
--
--   1. tbl_plan_prices was unique on (plan_id, billing_interval), which a USD
--      row for an interval that already has an INR row would violate. The
--      constraint now includes currency.
--   2. The INR read paths did not filter on currency. They now do -- see
--      razorpay_subscription_service._get_plan_with_features and
--      PromoRepository.get_by_code. Without that filter the INR lookup would
--      match two rows and raise MultipleResultsFound, breaking every rupee
--      checkout for the plan. Deploy the application BEFORE seeding USD rows,
--      or seed them last, as this script does.
--
-- Every statement is additive and idempotent. No existing column is altered or
-- dropped, and every new column is nullable or carries a default chosen so
-- existing INR rows and existing INR code read correctly with no changes.

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

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Seed the Pro USD prices
-- ─────────────────────────────────────────────────────────────────────────────
-- $20/month, $200/year, stored as whole dollars (see the comment on price_inr).
-- Annual is 10x monthly -- that *is* the "2 months free", set permanently in the
-- list price. Do not set annual to 12x monthly and layer a 2-month promo on top:
-- that discount would silently expire and hit the customer's foreign card with a
-- ~20% increase a year later with no human in the loop.
--
-- ai_credit_limit and total_count are copied from the matching INR row so the
-- entitlement a USD customer receives is identical to an INR customer's.
-- razorpay_plan_id stays NULL until the plans exist in the Razorpay dashboard.
--
-- These run last on purpose. Until they exist, the widened constraint and the
-- new columns are inert, so this script is safe to run before the application
-- that filters on currency has been deployed.

INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, price_inr, currency, ai_credit_limit, total_count, description)
SELECT pp.plan_id, 'monthly', 20, 'USD', pp.ai_credit_limit, pp.total_count, 'Pro plan, billed monthly'
FROM tbl_plan_prices pp
JOIN tbl_mstr_plans p ON p.id = pp.plan_id
WHERE p.code = 'pro' AND pp.billing_interval = 'monthly' AND pp.currency = 'INR'
ON CONFLICT (plan_id, billing_interval, currency) DO NOTHING;

INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, price_inr, currency, ai_credit_limit, total_count, description)
SELECT pp.plan_id, 'yearly', 200, 'USD', pp.ai_credit_limit, pp.total_count, 'Pro plan, billed annually'
FROM tbl_plan_prices pp
JOIN tbl_mstr_plans p ON p.id = pp.plan_id
WHERE p.code = 'pro' AND pp.billing_interval = 'yearly' AND pp.currency = 'INR'
ON CONFLICT (plan_id, billing_interval, currency) DO NOTHING;

-- Zero-priced tiers (Free) are mirrored across so they still render on the USD
-- pricing page. razorpay_plan_id stays NULL for them, which is what marks them
-- as not purchasable through checkout -- a free tier never goes through the
-- payment gateway.

INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, price_inr, currency, ai_credit_limit, total_count, description)
SELECT pp.plan_id, pp.billing_interval, 0, 'USD', pp.ai_credit_limit, pp.total_count, p.name || ' plan'
FROM tbl_plan_prices pp
JOIN tbl_mstr_plans p ON p.id = pp.plan_id
WHERE pp.price_inr = 0
  AND pp.currency = 'INR'
  AND pp.isactive
  AND pp.billing_interval IN ('monthly', 'yearly')
ON CONFLICT (plan_id, billing_interval, currency) DO NOTHING;

-- Any paid tier other than Pro is deliberately absent in USD. A tier with no USD
-- row is simply not offered in USD, which is correct until it has a real USD
-- price and a Razorpay USD plan of its own.

-- After creating the plans in the Razorpay dashboard, run:
--
--   UPDATE tbl_plan_prices SET razorpay_plan_id = 'plan_XXXXXXXXXXXXXX'
--   WHERE currency = 'USD' AND billing_interval = 'monthly'
--     AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');
--
--   UPDATE tbl_plan_prices SET razorpay_plan_id = 'plan_YYYYYYYYYYYYYY'
--   WHERE currency = 'USD' AND billing_interval = 'yearly'
--     AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Seed the advertised first-month promo
-- ─────────────────────────────────────────────────────────────────────────────
-- 50% off the first month, monthly only, USD only. is_public = true, so the
-- pricing page advertises it and checkout auto-applies it when no code is
-- submitted.
--
-- discount_type uses the existing vocabulary of tbl_promo_codes ('percentage' /
-- 'fixed'), and max_usage is the existing redemption-cap column.

INSERT INTO tbl_promo_codes
    (id, code, discount_type, discount_value, billing_interval, plan_code,
     max_usage, valid_from, valid_to, is_active, is_public, currency, description)
VALUES
    (gen_random_uuid(), 'USDINTRO50', 'percentage', 50, 'monthly', 'pro',
     NULL, now(), now() + INTERVAL '5 years', true, true, 'USD',
     '50% off the first month for new USD customers')
ON CONFLICT (code) DO NOTHING;
