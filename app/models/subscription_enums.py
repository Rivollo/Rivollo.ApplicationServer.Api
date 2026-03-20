"""Subscription-related enums.

This module contains enums used by subscription models:
- SubscriptionStatus: Status of a subscription
- LicenseStatus: Status of a license assignment
"""

import enum


class SubscriptionStatus(str, enum.Enum):
    """Status of a subscription."""

    PENDING = "pending"
    TRIALING = "trialing"
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"


class LicenseStatus(str, enum.Enum):
    """Status of a license assignment."""

    INVITED = "invited"
    ACTIVE = "active"
    REVOKED = "revoked"

