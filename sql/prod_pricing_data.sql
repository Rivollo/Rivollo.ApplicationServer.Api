-- PRODUCTION pricing data -- RUN LAST, AFTER THE NEW IMAGE IS SERVING.
--
-- ┌───────────────────────────────────────────────────────────────────────────┐
-- │ STOP. Fill in the four plan IDs below before running anything.            │
-- │                                                                           │
-- │ They must be LIVE-mode Razorpay plan IDs. The IDs used on dev were        │
-- │ created in whichever mode that dashboard session was in, and a Test-mode  │
-- │ plan id is invalid in Live -- checkout fails with a plan-not-found error   │
-- │ at the moment a real customer is trying to pay. Open the Razorpay         │
-- │ dashboard in LIVE mode and read the ids from there, even if they look     │
-- │ familiar.                                                                 │
-- └───────────────────────────────────────────────────────────────────────────┘
--
-- This file is DATA ONLY. All schema comes from, in this order:
--
--   1. sql/add_usd_pricing.sql            (currency columns, widened constraint)
--   2. sql/add_cancel_at_period_end.sql   (cancellation state)
--
-- Both are additive and idempotent, and both are safe to run against the OLD
-- image still serving production -- it never selects a column it does not know
-- about. Run them BEFORE deploying, not after: the new image asks for
-- cancel_at_period_end on every subscription query, and if the column is not
-- there yet that is an outage on live checkout, not a mispricing.
--
-- THIS file is the opposite: it must run AFTER the new image is live. The old
-- image's INR price lookup filters on (plan_id, billing_interval) with no
-- currency clause and calls scalar_one_or_none(). Insert a USD row for a plan
-- that already has an INR row and that call raises MultipleResultsFound --
-- every rupee checkout for that plan fails outright. Paying customers, live.
--
-- Full order:
--   1. run add_usd_pricing.sql
--   2. run add_cancel_at_period_end.sql
--   3. deploy the API
--   4. confirm GET https://api.rivollo.com/pricing returns 200 with "currency"
--      (it returns 404 on the old image -- that 404 is the gate)
--   5. run THIS file
--   6. deploy Portal, then Marketing

\set inr_yearly_plan  'REPLACE_WITH_LIVE_INR_YEARLY_PLAN_ID'
\set usd_monthly_plan 'REPLACE_WITH_LIVE_USD_MONTHLY_PLAN_ID'
\set usd_yearly_plan  'REPLACE_WITH_LIVE_USD_YEARLY_PLAN_ID'

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. INR annual -> Rs.19,999
-- ─────────────────────────────────────────────────────────────────────────────
-- Amounts here are MAJOR units: 19999 means Rs.19,999 and 19 means $19.
-- to_minor_units() multiplies by 100 at read time (app/utils/money.py), so
-- writing 1900 for $19 would quote $1,900.
--
-- INR monthly is deliberately absent. It stays at Rs.1,999 on its existing
-- plan, which is what keeps this migration-free: no live monthly subscription
-- changes price, and no mandate needs re-authorising.
UPDATE tbl_plan_prices pp
SET price_inr = 19999,
    razorpay_plan_id = :'inr_yearly_plan'
FROM tbl_mstr_plans p
WHERE pp.plan_id = p.id
  AND p.code = 'pro'
  AND pp.billing_interval = 'yearly'
  AND pp.currency = 'INR';

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. USD rows -- INSERT, not UPDATE. Production has never had any.
-- ─────────────────────────────────────────────────────────────────────────────
-- ai_credit_limit and total_count are copied from the matching INR row rather
-- than restated as literals, so the two currencies cannot drift apart on
-- anything except price.
INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, currency, price_inr, ai_credit_limit,
     razorpay_plan_id, total_count, isactive)
SELECT inr.plan_id, 'monthly', 'USD', 19, inr.ai_credit_limit,
       :'usd_monthly_plan', inr.total_count, true
FROM tbl_plan_prices inr
JOIN tbl_mstr_plans p ON p.id = inr.plan_id
WHERE p.code = 'pro' AND inr.billing_interval = 'monthly' AND inr.currency = 'INR'
ON CONFLICT (plan_id, billing_interval, currency) DO UPDATE
    SET price_inr = EXCLUDED.price_inr,
        razorpay_plan_id = EXCLUDED.razorpay_plan_id,
        isactive = true;

INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, currency, price_inr, ai_credit_limit,
     razorpay_plan_id, total_count, isactive)
SELECT inr.plan_id, 'yearly', 'USD', 199, inr.ai_credit_limit,
       :'usd_yearly_plan', inr.total_count, true
FROM tbl_plan_prices inr
JOIN tbl_mstr_plans p ON p.id = inr.plan_id
WHERE p.code = 'pro' AND inr.billing_interval = 'yearly' AND inr.currency = 'INR'
ON CONFLICT (plan_id, billing_interval, currency) DO UPDATE
    SET price_inr = EXCLUDED.price_inr,
        razorpay_plan_id = EXCLUDED.razorpay_plan_id,
        isactive = true;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. No promo may be active
-- ─────────────────────────────────────────────────────────────────────────────
-- Checkout now rejects promo codes outright, but /pricing still reads promo
-- rows to build its copy. An active row would advertise a discount checkout
-- refuses -- worse than no discount at all.
UPDATE tbl_promo_codes SET is_active = false WHERE is_active;

COMMIT;

-- ─────────────────────────────────────────────────────────────────────────────
-- Verification -- run all four. Do not deploy the frontends until they pass.
-- ─────────────────────────────────────────────────────────────────────────────
--
-- (a) Expect exactly these four rows, every razorpay_plan_id populated:
--
--   monthly | INR |  1999 | (pre-existing id, unchanged)
--   yearly  | INR | 19999 | <live inr yearly>
--   monthly | USD |    19 | <live usd monthly>
--   yearly  | USD |   199 | <live usd yearly>
--
--   SELECT pp.billing_interval, pp.currency, pp.price_inr,
--          pp.razorpay_plan_id, pp.isactive
--   FROM tbl_plan_prices pp
--   JOIN tbl_mstr_plans p ON p.id = pp.plan_id
--   WHERE p.code = 'pro'
--   ORDER BY pp.currency, pp.billing_interval;
--
-- (b) Expect 0 -- a paid row with no plan id 400s that interval at checkout:
--
--   SELECT count(*) FROM tbl_plan_prices
--   WHERE isactive AND price_inr > 0
--     AND (razorpay_plan_id IS NULL OR razorpay_plan_id = '');
--
-- (c) Expect 0 -- no promo may remain active:
--
--   SELECT count(*) FROM tbl_promo_codes WHERE is_active;
--
-- (d) Expect 0 -- no existing subscription should have been touched. Every
--     production subscriber must still be INR on their original plan:
--
--   SELECT count(*) FROM tbl_subscriptions
--   WHERE currency <> 'INR' OR start_at IS NOT NULL OR upfront_amount IS NOT NULL;
--
-- Then, from outside:
--
--   curl -s https://api.rivollo.com/pricing | head -c 200
--     -> "currency":"INR" from an Indian connection, "USD" from elsewhere
--
-- Rollback for this file: sql/rollback_usd_pricing.sql removes the USD rows.
-- The INR annual price and plan id would need restoring by hand -- note the
-- current values before you run step 1.
