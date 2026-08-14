-- Add billing_interval restriction to promo codes.
-- NULL = the promo applies to every billing interval of its plan.
-- Run this script manually against the database.

ALTER TABLE tbl_promo_codes
    ADD COLUMN IF NOT EXISTS billing_interval VARCHAR(20) NULL;

ALTER TABLE tbl_promo_codes
    DROP CONSTRAINT IF EXISTS tbl_promo_codes_billing_interval_check;

ALTER TABLE tbl_promo_codes
    ADD CONSTRAINT tbl_promo_codes_billing_interval_check
    CHECK (billing_interval IS NULL
           OR billing_interval IN ('monthly', 'yearly', 'weekly'));

COMMENT ON COLUMN tbl_promo_codes.billing_interval IS
    'Billing interval this promo is restricted to. NULL = any interval.';

-- PROMO50M is a first-month discount: monthly only.
-- Razorpay would refuse it on the yearly plan anyway (yearly price exceeds the
-- offer''s max-payment ceiling), but "On Offer Failure = Allow Payment" makes that
-- refusal silent — so the restriction has to be enforced here to avoid showing the
-- user a discount they will not receive.
UPDATE tbl_promo_codes
SET billing_interval = 'monthly'
WHERE code = 'PROMO50M';
