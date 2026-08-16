-- Self-serve subscription cancellation -- SCHEMA ONLY.
--
-- Adds one column so the API can tell "active and renewing" apart from
-- "active but already cancelled, running out the paid period".
--
-- Why it is needed: cancel_subscription() deliberately leaves status ACTIVE
-- when cancel_at_cycle_end is true, because the customer has paid through the
-- end of the period and Razorpay only confirms the cancellation via the
-- subscription.cancelled webhook, up to a month later. Without this column
-- that intermediate state is invisible -- /subscriptions/me reports a plain
-- ACTIVE subscription, so the UI keeps promising "Renews on the 14th" to
-- someone who has already cancelled, and keeps offering them a Cancel button
-- that would fire a second Razorpay call.
--
-- RUN THIS BEFORE DEPLOYING THE IMAGE THAT CONTAINS IT. Not after, not
-- alongside.
--
-- The column is mapped on the Subscription model, and the SQLAlchemy ORM names
-- every mapped column explicitly in its SELECT list. So the moment the new
-- image serves a request, every query against tbl_subscriptions asks for
-- cancel_at_period_end by name. If the column is not there yet Postgres raises
-- UndefinedColumn and the query fails outright -- which takes out
-- /subscriptions/me, the licensing checks, and checkout, for every customer in
-- both currencies. Not a mispriced page: an outage.
--
-- The reverse order is safe. This script against the image running today is a
-- no-op from that image's point of view: it never selects a column it does not
-- know about, so an old image and the new column coexist happily for as long
-- as you like. That asymmetry is what makes "migrate first, then deploy" the
-- only correct sequence, and it is the same reason add_usd_pricing.sql insists
-- on its own ordering.
--
-- Beyond the ordering it is additive and idempotent: the column is NOT NULL
-- with a false default, so every existing row reads as "not cancelled" --
-- which is what they all are -- and rerunning changes nothing.

ALTER TABLE tbl_subscriptions
    ADD COLUMN IF NOT EXISTS cancel_at_period_end BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN tbl_subscriptions.cancel_at_period_end IS
    'True once the customer has cancelled but paid access continues to '
    'current_period_end. Cleared only by a new subscription, never by resuming '
    '-- Razorpay cannot reverse a scheduled cancellation.';

-- ─────────────────────────────────────────────────────────────────────────────
-- Verification
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Expect one row: cancel_at_period_end | boolean | NO | false
--
--   SELECT column_name, data_type, is_nullable, column_default
--   FROM information_schema.columns
--   WHERE table_name = 'tbl_subscriptions'
--     AND column_name = 'cancel_at_period_end';
--
-- Expect 0 -- nothing is cancelled yet at the moment you run this:
--
--   SELECT count(*) FROM tbl_subscriptions WHERE cancel_at_period_end;
