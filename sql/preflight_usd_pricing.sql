-- Preflight for the USD migration. READ ONLY -- nothing here writes.
--
-- Run this before sql/add_usd_pricing.sql, on whichever database you are about
-- to migrate, and share the output. It answers the questions the migration
-- assumes the answers to:
--
--   * what the existing unique constraint on tbl_plan_prices is actually called
--   * which of the columns being added already exist
--   * what the INR prices, credit limits and cycle counts really are, since the
--     USD rows are seeded by copying them
--   * whether any promo code would collide
--
-- Output is safe to paste: the subscription and payment tables are reported as
-- counts only, so no customer data leaves the database.

\echo '=============== 1. Constraints on tbl_plan_prices ==============='
-- The migration drops a constraint BY NAME. If the live one is named
-- differently, that DROP silently does nothing, the old single-currency rule
-- survives, and seeding the USD rows fails on a uniqueness violation.
SELECT con.conname               AS constraint_name,
       pg_get_constraintdef(con.oid) AS definition
FROM   pg_constraint con
JOIN   pg_class rel ON rel.oid = con.conrelid
WHERE  rel.relname = 'tbl_plan_prices'
ORDER  BY con.contype, con.conname;

\echo ''
\echo '=============== 2. Columns that already exist ==============='
-- Everything the migration adds is IF NOT EXISTS, so this is about knowing what
-- will actually change rather than about safety.
SELECT table_name,
       column_name,
       data_type,
       is_nullable,
       column_default
FROM   information_schema.columns
WHERE  (table_name = 'tbl_plan_prices'  AND column_name IN ('currency'))
   OR  (table_name = 'tbl_promo_codes'  AND column_name IN ('currency', 'is_public', 'max_usage', 'billing_interval'))
   OR  (table_name = 'tbl_subscriptions' AND column_name IN
            ('currency', 'billing_country', 'promo_code', 'upfront_amount',
             'full_amount', 'promo_period_active', 'start_at'))
   OR  (table_name = 'tbl_payments'     AND column_name IN ('currency'))
ORDER  BY table_name, column_name;

\echo ''
\echo '=============== 3. Plans ==============='
SELECT id, code, name, isactive
FROM   tbl_mstr_plans
ORDER  BY code;

\echo ''
\echo '=============== 4. Plan prices (the rows USD is copied from) ==============='
-- price_inr, ai_credit_limit and total_count are read directly by the seed, so
-- these numbers decide what a USD customer gets. `currency` is selected
-- defensively -- if the column does not exist yet this query errors, which
-- itself answers question 2.
SELECT p.code             AS plan_code,
       pp.billing_interval,
       pp.currency,
       pp.price_inr,
       pp.ai_credit_limit,
       pp.total_count,
       pp.trial_period_days,
       pp.isactive,
       pp.razorpay_plan_id
FROM   tbl_plan_prices pp
JOIN   tbl_mstr_plans p ON p.id = pp.plan_id
ORDER  BY p.code, pp.currency, pp.billing_interval;

\echo ''
\echo '=============== 5. Anything already non-INR ==============='
-- Expected: zero rows. Anything here means a previous attempt left data behind.
SELECT count(*) AS non_inr_price_rows
FROM   tbl_plan_prices
WHERE  currency IS DISTINCT FROM 'INR';

\echo ''
\echo '=============== 6. Promo codes ==============='
-- discount_type vocabulary matters: the USD promo is seeded as 'percentage', and
-- a CHECK constraint or differing convention here would reject it.
SELECT code,
       discount_type,
       discount_value,
       billing_interval,
       plan_code,
       max_usage,
       used_count,
       is_active,
       valid_from,
       valid_to,
       razorpay_offer_id IS NOT NULL AS uses_razorpay_offer
FROM   tbl_promo_codes
ORDER  BY code;

\echo ''
\echo '=============== 7. Would USDINTRO50 collide? ==============='
-- `code` is unique across both currencies, so an existing row with this code
-- would make the seed a silent no-op.
SELECT count(*) AS existing_usdintro50
FROM   tbl_promo_codes
WHERE  upper(code) = 'USDINTRO50';

\echo ''
\echo '=============== 8. Feature codes ==============='
-- Entitlements resolve through these; the USD path reuses them unchanged.
SELECT f.code AS feature_code,
       p.code AS plan_code,
       pf.is_available,
       pf.limit_value
FROM   tbl_plan_features pf
JOIN   tbl_mstr_features f ON f.id = pf.feature_id
JOIN   tbl_mstr_plans   p ON p.id = pf.plan_id
ORDER  BY p.code, f.code;

\echo ''
\echo '=============== 9. Subscriptions -- counts only ==============='
-- Aggregated deliberately: this tells us how many rows the billing_country
-- backfill will touch without exporting anyone's subscription.
SELECT status,
       count(*) AS subscriptions
FROM   tbl_subscriptions
GROUP  BY status
ORDER  BY status;

\echo ''
\echo '=============== 10. Rows with a Razorpay subscription ID ==============='
-- The webhook resolves currency by this ID, so duplicates would matter.
SELECT count(*)                                  AS total_rows,
       count(razorpay_subscription_id)           AS with_rz_id,
       count(DISTINCT razorpay_subscription_id)  AS distinct_rz_ids
FROM   tbl_subscriptions;

\echo ''
\echo '=============== 11. Payments -- counts only ==============='
SELECT count(*) AS payments
FROM   tbl_payments;

\echo ''
\echo '=============== 12. Indexes on tbl_subscriptions ==============='
-- The migration deliberately adds none, on the grounds that
-- idx_subscriptions_razorpay_sub_id already covers the webhook lookup.
SELECT indexname, indexdef
FROM   pg_indexes
WHERE  tablename = 'tbl_subscriptions'
ORDER  BY indexname;
