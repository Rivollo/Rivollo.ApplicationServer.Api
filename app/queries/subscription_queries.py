"""Subscription SQL query constants.

This module contains ONLY raw SQL strings — no logic, no imports, no classes.
All queries are parameterised to prevent SQL injection.

Naming convention:
    FIND_*    → SELECT queries
    GET_*     → SELECT queries
"""

# The expiry job's statements used to live here as raw SQL. They compared
# status against lowercase 'active' and wrote lowercase 'canceled' / 'revoked',
# but the column stores the enum NAMES in uppercase, so they never matched a
# row. The job now lives in app/services/subscription_deactivation_service.py
# and matches status without regard to case, as the query below does.

GET_USER_PLAN_CODE = """
    SELECT p.code
    FROM tbl_subscriptions s
    INNER JOIN tbl_mstr_plans p ON s.plan_id = p.id
    WHERE s.user_id = :user_id
      AND UPPER(s.status::text) = 'ACTIVE'
      ORDER BY s.created_date DESC
      LIMIT 1
"""