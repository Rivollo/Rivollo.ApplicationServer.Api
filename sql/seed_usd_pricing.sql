-- USD billing for customers outside India -- STEP 2 of 2: the USD rows.
--
-- RUN THIS ONLY AFTER the application image that filters price and promo
-- lookups on currency has been deployed and is serving.
--
-- Why the order matters: razorpay_subscription_service._get_plan_with_features
-- selects tbl_plan_prices on (plan_id, billing_interval) and calls
-- scalar_one_or_none(). Once a USD row exists for a plan and interval that
-- already has an INR row, a lookup without a currency filter matches two rows
-- and raises MultipleResultsFound. On the old image that means every rupee
-- checkout for that plan fails outright.
--
-- To check the deployed image is new enough before running this:
--
--     GET /pricing  with header  cf-ipcountry: US
--
-- A response with "currency": "USD" means the currency-aware code is live. The
-- old image has no /pricing route at all.
--
-- Idempotent: ON CONFLICT DO NOTHING throughout, so re-running changes nothing.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Pro USD prices
-- ─────────────────────────────────────────────────────────────────────────────
-- $20/month, $200/year, stored as WHOLE DOLLARS -- price_inr holds whole units
-- of the row's currency, not minor units. See the column comment.
--
-- Annual is 10x monthly. That *is* the "2 months free", set permanently in the
-- list price. Do not set annual to 12x monthly and layer a 2-month promo on top:
-- that discount would silently expire and hit the customer's foreign card with a
-- ~20% increase a year later with no human in the loop.
--
-- ai_credit_limit and total_count are copied from the matching INR row so the
-- entitlement a USD customer receives is identical to an INR customer's.

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

-- The Free tier, so it still renders as a column on the USD pricing page.
--
-- This is an explicit insert rather than a "copy every zero-priced row" query,
-- because there is no zero-priced row to copy: tbl_plan_prices contains only
-- pro/monthly, pro/yearly and weekly/weekly. Free has never had a price row at
-- all -- the marketing page renders it from hardcoded copy. A mirror keyed on
-- price_inr = 0 therefore matched nothing, and a USD visitor would have seen a
-- pricing page with Pro and no free option, because the frontend drops any tier
-- the API does not return.
--
-- ai_credit_limit comes from the tier's own max_ai_credits_month feature limit
-- (100), which is where the free allowance is actually configured.
-- razorpay_plan_id stays NULL, which is what marks it as not purchasable -- a
-- free tier never goes through the payment gateway.

INSERT INTO tbl_plan_prices
    (plan_id, billing_interval, price_inr, currency, ai_credit_limit, total_count, description)
SELECT p.id, i.interval, 0, 'USD', 100, 0, 'Free plan'
FROM   tbl_mstr_plans p
CROSS  JOIN (VALUES ('monthly'), ('yearly')) AS i(interval)
WHERE  p.code = 'free'
ON CONFLICT (plan_id, billing_interval, currency) DO NOTHING;

-- Any paid tier other than Pro is deliberately absent in USD. A tier with no USD
-- row is simply not offered in USD, which is correct until it has a real USD
-- price and a Razorpay USD plan of its own.

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Point the USD rows at the Razorpay plans
-- ─────────────────────────────────────────────────────────────────────────────
-- Replace the IDs below with the plans from the Razorpay dashboard. They are
-- per-mode: a Test-mode plan ID only resolves against Test-mode API keys, so
-- these must come from the same mode as the RAZORPAY_KEY_ID the API is running.
--
-- Until they are set, USD checkout returns 400 rather than creating a
-- subscription it cannot fulfil, and /pricing reports available: false.
--
-- Dev (as supplied):  monthly plan_TQ2m22UBRutnZu
--                     annual  plan_TQ8SZe3nf6a0d3

UPDATE tbl_plan_prices SET razorpay_plan_id = 'plan_TQ2m22UBRutnZu'
WHERE currency = 'USD' AND billing_interval = 'monthly'
  AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');

UPDATE tbl_plan_prices SET razorpay_plan_id = 'plan_TQ8SZe3nf6a0d3'
WHERE currency = 'USD' AND billing_interval = 'yearly'
  AND plan_id = (SELECT id FROM tbl_mstr_plans WHERE code = 'pro');

-- total_count is how many billing cycles run before the subscription ends. It is
-- copied from the INR row, where 1200 months is a harmless 100 years -- but 1200
-- *years* on the annual row can make Razorpay reject the subscription outright
-- at checkout. Guarded, so it does nothing if the value is already sensible.

UPDATE tbl_plan_prices SET total_count = 10
WHERE currency = 'USD' AND billing_interval = 'yearly' AND total_count > 100;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. The advertised first-month promo
-- ─────────────────────────────────────────────────────────────────────────────
-- 50% off the first month, monthly only, USD only. is_public = true, so the
-- pricing page advertises it and checkout auto-applies it when no code is
-- submitted -- the displayed price and the charged price cannot drift apart.
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

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Verify
-- ─────────────────────────────────────────────────────────────────────────────

SELECT p.code,
       u.billing_interval,
       u.currency,
       u.price_inr,
       u.total_count,
       u.razorpay_plan_id
FROM   tbl_plan_prices u
JOIN   tbl_mstr_plans p ON p.id = u.plan_id
WHERE  p.code = 'pro'
ORDER  BY u.currency, u.billing_interval;
