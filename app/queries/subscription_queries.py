"""Subscription SQL query constants.

This module contains ONLY raw SQL strings — no logic, no imports, no classes.
All queries are parameterised to prevent SQL injection.

Naming convention:
    FIND_*    → SELECT queries
    DEACTIVATE_* / REVOKE_* → UPDATE queries

Usage:
    from app.queries.subscription_queries import DEACTIVATE_EXPIRED_SUBSCRIPTIONS
    await db.execute(text(DEACTIVATE_EXPIRED_SUBSCRIPTIONS), {"now": datetime.now(timezone.utc)})
"""

# ---------------------------------------------------------------------------
# SELECT: find active subscriptions whose billing period has ended
# ---------------------------------------------------------------------------
FIND_EXPIRED_SUBSCRIPTIONS = """
    SELECT id, user_id
    FROM   tbl_subscriptions
    WHERE  status = 'active'
      AND  current_period_end IS NOT NULL
      AND  current_period_end <= :now
"""

# ---------------------------------------------------------------------------
# UPDATE: mark expired subscriptions as 'canceled'
#         Returns the IDs of every row that was actually updated.
# ---------------------------------------------------------------------------
DEACTIVATE_EXPIRED_SUBSCRIPTIONS = """
    UPDATE tbl_subscriptions
    SET    status = 'canceled'
    WHERE  status = 'active'
      AND  current_period_end IS NOT NULL
      AND  current_period_end <= :now
    RETURNING id
"""

# ---------------------------------------------------------------------------
# UPDATE: revoke all active licenses that belong to a list of subscription IDs
#         :subscription_ids must be a Python list of UUIDs
# ---------------------------------------------------------------------------
REVOKE_LICENSES_FOR_SUBSCRIPTIONS = """
    UPDATE tbl_license_assignments
    SET    status = 'revoked'
    WHERE  subscription_id = ANY(:subscription_ids)
      AND  status = 'active'
"""
GET_USER_PLAN_CODE = """
    SELECT p.code
    FROM tbl_subscriptions s
    INNER JOIN tbl_mstr_plans p ON s.plan_id = p.id
    WHERE s.user_id = :user_id
      AND s.status = 'ACTIVE'
      ORDER BY s.created_date DESC
      LIMIT 1
"""