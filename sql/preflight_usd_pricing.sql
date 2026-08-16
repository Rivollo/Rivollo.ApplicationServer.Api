-- Preflight for the USD migration. READ ONLY -- nothing here writes.
--
-- ONE statement, returning ONE result grid. Run it in DBeaver with Ctrl+Enter
-- (or psql, or anything else), then copy the whole grid and share it.
--
-- It answers the questions the migration assumes the answers to:
--
--   * what the existing unique constraint on tbl_plan_prices is really called.
--     The migration drops it BY NAME -- if the live name differs, that DROP
--     silently does nothing, the single-currency rule survives, and seeding the
--     USD rows fails on a uniqueness violation after the deploy.
--   * which of the columns being added already exist
--   * what the INR prices, credit limits and cycle counts actually are, since
--     the USD rows are seeded by copying them
--   * whether USDINTRO50 would collide with an existing code
--
-- The two config tables are dumped with to_jsonb() rather than named columns,
-- so this cannot fail because a column is missing -- it simply reports whatever
-- columns are there.
--
-- Safe to paste: subscriptions and payments are reported as counts only, so no
-- customer data leaves the database.

WITH report AS (

    -- 1. Constraints. The sharp one: what is the uniqueness rule actually named?
    SELECT 1 AS ord,
           'constraint' AS section,
           rel.relname || ' -> ' || con.conname AS item,
           pg_get_constraintdef(con.oid) AS detail
    FROM   pg_constraint con
    JOIN   pg_class rel ON rel.oid = con.conrelid
    WHERE  rel.relname IN ('tbl_plan_prices', 'tbl_promo_codes')

    UNION ALL

    -- 2. Which columns the migration adds already exist.
    SELECT 2,
           'column',
           table_name || '.' || column_name,
           data_type || '  nullable=' || is_nullable
                     || '  default=' || COALESCE(column_default, '(none)')
    FROM   information_schema.columns
    WHERE  (table_name = 'tbl_plan_prices'   AND column_name = 'currency')
       OR  (table_name = 'tbl_promo_codes'   AND column_name IN
                ('currency', 'is_public', 'max_usage', 'billing_interval', 'razorpay_offer_id'))
       OR  (table_name = 'tbl_subscriptions' AND column_name IN
                ('currency', 'billing_country', 'promo_code', 'upfront_amount',
                 'full_amount', 'promo_period_active', 'start_at'))
       OR  (table_name = 'tbl_payments'      AND column_name = 'currency')

    UNION ALL

    -- 3. Plans.
    SELECT 3,
           'plan',
           code,
           name || '  active=' || isactive::text || '  id=' || id::text
    FROM   tbl_mstr_plans

    UNION ALL

    -- 4. Plan prices -- the rows the USD seed copies ai_credit_limit and
    --    total_count from, so these numbers decide what a USD customer gets.
    SELECT 4,
           'plan_price',
           p.code || ' / ' || pp.billing_interval,
           to_jsonb(pp)::text
    FROM   tbl_plan_prices pp
    JOIN   tbl_mstr_plans p ON p.id = pp.plan_id

    UNION ALL

    -- 5. Promo codes -- the USD promo is seeded as discount_type 'percentage'
    --    and must match whatever vocabulary is already in use here.
    SELECT 5,
           'promo',
           pc.code,
           to_jsonb(pc)::text
    FROM   tbl_promo_codes pc

    UNION ALL

    -- 6. Would the seeded promo collide? `code` is unique across currencies, so
    --    an existing row would make the seed a silent no-op.
    SELECT 6,
           'promo_collision',
           'USDINTRO50 already present',
           count(*)::text
    FROM   tbl_promo_codes
    WHERE  upper(code) = 'USDINTRO50'

    UNION ALL

    -- 7. Plan features -- entitlements resolve through these, unchanged by USD.
    SELECT 7,
           'feature',
           p.code || ' / ' || f.code,
           'available=' || pf.is_available::text
                        || '  limit=' || COALESCE(pf.limit_value::text, '(none)')
    FROM   tbl_plan_features pf
    JOIN   tbl_mstr_features f ON f.id = pf.feature_id
    JOIN   tbl_mstr_plans   p ON p.id = pf.plan_id

    UNION ALL

    -- 8. How many rows the billing_country backfill will touch. Counts only.
    SELECT 8,
           'subscriptions',
           'status = ' || status::text,
           count(*)::text
    FROM   tbl_subscriptions
    GROUP  BY status

    UNION ALL

    -- 9. The webhook resolves currency by razorpay_subscription_id, so a gap
    --    between these two numbers means duplicates exist.
    SELECT 9,
           'subscriptions',
           'total / with rz_id / distinct rz_id',
           count(*)::text || ' / '
             || count(razorpay_subscription_id)::text || ' / '
             || count(DISTINCT razorpay_subscription_id)::text
    FROM   tbl_subscriptions

    UNION ALL

    SELECT 10, 'payments', 'row count', count(*)::text
    FROM   tbl_payments

    UNION ALL

    -- 11. Indexes, to confirm the migration is right that it needs to add none.
    SELECT 11,
           'index',
           tablename || ' -> ' || indexname,
           indexdef
    FROM   pg_indexes
    WHERE  tablename IN ('tbl_plan_prices', 'tbl_promo_codes', 'tbl_subscriptions')
)

SELECT ord AS "#", section, item, detail
FROM   report
ORDER  BY ord, item;
