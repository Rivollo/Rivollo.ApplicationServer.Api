-- USD billing for customers outside India.
-- Run this script manually against the database.
--
-- Every statement is additive and idempotent. Nothing here alters or drops an
-- existing column, and every new column on tbl_subscriptions is either nullable
-- or carries a default chosen so existing INR rows and existing INR code read
-- correctly with no changes.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. USD price list
-- ─────────────────────────────────────────────────────────────────────────────
-- Separate table rather than rows on tbl_plan_prices: that table has a unique
-- constraint on (plan_id, billing_interval) and the INR lookup queries it by
-- that pair with no currency filter.

CREATE TABLE IF NOT EXISTS tbl_plan_prices_usd (
    id                   SERIAL PRIMARY KEY,
    plan_id              UUID NOT NULL REFERENCES tbl_mstr_plans(id) ON DELETE CASCADE,
    billing_interval     VARCHAR(20) NOT NULL,
    price_usd            INTEGER NOT NULL DEFAULT 0,     -- cents: $20.00 => 2000
    ai_credit_limit      INTEGER NOT NULL DEFAULT 0,
    razorpay_plan_id_usd VARCHAR(255) NULL,
    total_count          INTEGER NOT NULL DEFAULT 1200,
    description          VARCHAR(100) NULL,
    isactive             BOOLEAN NOT NULL DEFAULT true,
    created_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tbl_plan_prices_usd_plan_interval_key UNIQUE (plan_id, billing_interval)
);

ALTER TABLE tbl_plan_prices_usd
    DROP CONSTRAINT IF EXISTS tbl_plan_prices_usd_billing_interval_check;

ALTER TABLE tbl_plan_prices_usd
    ADD CONSTRAINT tbl_plan_prices_usd_billing_interval_check
    CHECK (billing_interval IN ('monthly', 'yearly'));

COMMENT ON COLUMN tbl_plan_prices_usd.price_usd IS
    'List price in cents. Always the full price — never a discounted price. '
    'Promotional pricing is applied as a subscription upfront amount, never by '
    'creating a cheaper plan, because Razorpay plan amounts cannot be edited '
    'after creation and a promotional plan is therefore a permanent price cut.';

COMMENT ON COLUMN tbl_plan_prices_usd.razorpay_plan_id_usd IS
    'Razorpay plan ID for the USD plan. NULL until the plans exist in the '
    'Razorpay dashboard; the USD checkout route rejects the interval while NULL.';

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. USD promo codes
-- ─────────────────────────────────────────────────────────────────────────────
-- Razorpay Offers are INR-locked on this account and fail silently in USD, so
-- USD discounts are computed server-side. No razorpay_offer_id column here.

CREATE TABLE IF NOT EXISTS tbl_promo_codes_usd (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code             VARCHAR(64) UNIQUE NOT NULL,
    discount_type    VARCHAR(20) NOT NULL,
    discount_value   INTEGER NOT NULL,
    billing_interval VARCHAR(20) NOT NULL DEFAULT 'monthly',
    plan_code        VARCHAR(50) NULL,
    max_redemptions  INTEGER NULL,
    used_count       INTEGER NOT NULL DEFAULT 0,
    valid_from       TIMESTAMPTZ NOT NULL,
    valid_to         TIMESTAMPTZ NOT NULL,
    is_active        BOOLEAN NOT NULL DEFAULT true,
    is_public        BOOLEAN NOT NULL DEFAULT false,
    description      VARCHAR(255) NULL,
    created_date     TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE tbl_promo_codes_usd
    DROP CONSTRAINT IF EXISTS tbl_promo_codes_usd_discount_type_check;

-- Same vocabulary as tbl_promo_codes.discount_type, so the two promo tables do
-- not disagree about what a discount type is called.
ALTER TABLE tbl_promo_codes_usd
    ADD CONSTRAINT tbl_promo_codes_usd_discount_type_check
    CHECK (discount_type IN ('percentage', 'fixed'));

-- Annual is never eligible: the two-months-free discount is permanent and
-- already inside the annual list price, so a promo on top would double-count it.
ALTER TABLE tbl_promo_codes_usd
    DROP CONSTRAINT IF EXISTS tbl_promo_codes_usd_billing_interval_check;

ALTER TABLE tbl_promo_codes_usd
    ADD CONSTRAINT tbl_promo_codes_usd_billing_interval_check
    CHECK (billing_interval = 'monthly');

COMMENT ON COLUMN tbl_promo_codes_usd.is_public IS
    'True for the promo advertised on the pricing page. The public promo is '
    'auto-applied at checkout when the customer submits no code, so the price '
    'shown is always the price charged.';

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
    'changed — a customer who subscribes in USD keeps paying USD even if they '
    'later browse from India, and vice versa.';

COMMENT ON COLUMN tbl_subscriptions.upfront_amount IS
    'Amount actually charged at the authentication transaction, in the smallest '
    'unit of `currency`. Stored rather than recomputed from the promo percentage '
    'so a future price change cannot make historical records lie.';

COMMENT ON COLUMN tbl_subscriptions.promo_period_active IS
    'True between subscription.authenticated and the first full-price charge — '
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
-- $20.00/month, $200.00/year. Annual is 10x monthly — that *is* the "2 months
-- free", set permanently in the list price. Do not set annual to 12x monthly and
-- layer a 2-month promo on top: that discount would silently expire and hit the
-- customer's foreign card with a ~20% increase a year later with no human in
-- the loop.
--
-- ai_credit_limit and total_count are copied from the matching INR row so the
-- entitlement a USD customer receives is identical to an INR customer's.
-- razorpay_plan_id_usd stays NULL until the two USD plans exist in the Razorpay
-- dashboard; fill it in with a follow-up UPDATE.

INSERT INTO tbl_plan_prices_usd (plan_id, billing_interval, price_usd, ai_credit_limit, total_count, description)
SELECT p.id, 'monthly', 2000, pp.ai_credit_limit, pp.total_count, 'Pro plan, billed monthly'
FROM tbl_mstr_plans p
JOIN tbl_plan_prices pp ON pp.plan_id = p.id AND pp.billing_interval = 'monthly'
WHERE p.code = 'pro'
ON CONFLICT (plan_id, billing_interval) DO NOTHING;

INSERT INTO tbl_plan_prices_usd (plan_id, billing_interval, price_usd, ai_credit_limit, total_count, description)
SELECT p.id, 'yearly', 20000, pp.ai_credit_limit, pp.total_count, 'Pro plan, billed annually'
FROM tbl_mstr_plans p
JOIN tbl_plan_prices pp ON pp.plan_id = p.id AND pp.billing_interval = 'yearly'
WHERE p.code = 'pro'
ON CONFLICT (plan_id, billing_interval) DO NOTHING;

-- Zero-priced tiers (Free) are mirrored across so they still render on the USD
-- pricing page. razorpay_plan_id_usd stays NULL for them, which is what marks
-- them as not purchasable through checkout — a free tier never goes through the
-- payment gateway.

INSERT INTO tbl_plan_prices_usd (plan_id, billing_interval, price_usd, ai_credit_limit, total_count, description)
SELECT pp.plan_id, pp.billing_interval, 0, pp.ai_credit_limit, pp.total_count, p.name || ' plan'
FROM tbl_plan_prices pp
JOIN tbl_mstr_plans p ON p.id = pp.plan_id
WHERE pp.price_inr = 0
  AND pp.isactive
  AND pp.billing_interval IN ('monthly', 'yearly')
ON CONFLICT (plan_id, billing_interval) DO NOTHING;

-- Any paid tier other than Pro is deliberately absent from this table. A tier
-- with no USD row is simply not offered in USD, which is correct until it has a
-- real USD price and a Razorpay USD plan of its own.

-- After creating the plans in the Razorpay dashboard, run:
--
--   UPDATE tbl_plan_prices_usd SET razorpay_plan_id_usd = 'plan_XXXXXXXXXXXXXX'
--   WHERE billing_interval = 'monthly'
--     AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');
--
--   UPDATE tbl_plan_prices_usd SET razorpay_plan_id_usd = 'plan_YYYYYYYYYYYYYY'
--   WHERE billing_interval = 'yearly'
--     AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Seed the advertised first-month promo
-- ─────────────────────────────────────────────────────────────────────────────
-- 50% off the first month, monthly only. is_public = true, so the pricing page
-- advertises it and checkout auto-applies it when no code is submitted — the
-- displayed price and the charged price cannot drift apart.

INSERT INTO tbl_promo_codes_usd
    (code, discount_type, discount_value, billing_interval, plan_code,
     max_redemptions, valid_from, valid_to, is_active, is_public, description)
VALUES
    ('USDINTRO50', 'percentage', 50, 'monthly', 'pro',
     NULL, now(), now() + INTERVAL '5 years', true, true,
     '50% off the first month for new USD customers')
ON CONFLICT (code) DO NOTHING;
