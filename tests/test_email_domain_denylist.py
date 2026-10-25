"""Operator denylist for signup/login email domains.

Regression for koboywin.com / ehwit.com: absent from the upstream
`disposable-email-domains` package while publishing valid MX records, so the
staged check let them through. The denylist file + env var closes that gap,
and — unlike the upstream list — is enforced on the existing-user paths too.
"""

import pytest
from pydantic import ValidationError

from app.schemas.auth import LoginRequest, ForgotPasswordRequest, SendSignupOtpRequest
from app.utils.email_domain_check import is_denied_domain, is_denied_email, is_disposable


@pytest.mark.parametrize("domain", ["koboywin.com", "ehwit.com", "xyz.ehwit.com", "KOBOYWIN.COM"])
def test_bundled_denylist_blocks_known_abusers(domain):
    assert is_denied_domain(domain)
    assert is_disposable(domain)


@pytest.mark.parametrize("domain", ["gmail.com", "wingsbi.com", "com"])
def test_legit_domains_are_not_denied(domain):
    assert not is_denied_domain(domain)


def test_upstream_list_is_not_the_operator_denylist():
    # mailinator is disposable (upstream) but NOT operator-denied, so an
    # existing mailinator account could still log in. That asymmetry is the
    # point: only explicit operator decisions lock people out.
    assert is_disposable("mailinator.com")
    assert not is_denied_domain("mailinator.com")


def test_is_denied_email_handles_garbage():
    assert not is_denied_email("not-an-email")
    assert not is_denied_email("")
    assert is_denied_email("someone@ehwit.com")


def test_signup_otp_rejects_denied_domain():
    with pytest.raises(ValidationError):
        SendSignupOtpRequest(email="a@koboywin.com")


def test_login_rejects_denied_domain_but_not_upstream_disposable():
    with pytest.raises(ValidationError):
        LoginRequest(email="a@ehwit.com", password="password123")
    # Upstream-only disposable: login still allowed (existing-user path).
    LoginRequest(email="a@mailinator.com", password="password123")


def test_forgot_password_rejects_denied_domain():
    with pytest.raises(ValidationError):
        ForgotPasswordRequest(email="a@koboywin.com")
