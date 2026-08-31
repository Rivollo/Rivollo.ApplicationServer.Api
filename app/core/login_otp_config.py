"""Settings for the email OTP login flow.

Deliberately a SEPARATE ``BaseSettings`` class rather than new fields on
``app.core.config.Settings``. The OTP login feature is additive and isolated:
owning its own settings object is what keeps ``app/core/config.py`` — a file
every module in the project imports — out of this feature's change set.

pydantic-settings supports any number of independent settings classes reading
the same environment, so this costs nothing at runtime. The only divergence
from house style is that OTP code reads ``otp_settings`` instead of
``settings``; every other module is unaffected.

LOGIN_OTP_ENABLED defaults to FALSE on purpose. Two reasons:

  * The rollout sequence deploys the backend with the feature off, proves the
    deployment itself is safe, and only then enables it.
  * The pepper is required whenever the feature is on (see the validator). If
    the default were True, importing ``app.main`` anywhere without a pepper
    configured — every existing test does exactly that — would raise at import
    time. Defaulting to off keeps the requirement strict where it matters
    (a deployed, enabled environment) without making an unrelated test suite
    depend on a secret.
"""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoginOtpSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Kill switch. Off by default — see the module docstring.
    LOGIN_OTP_ENABLED: bool = Field(default=False)

    # Lifetime of a single code. Shorter than SIGNUP_OTP_EXPIRES_MINUTES (10)
    # because a login code is used immediately, whereas a signup code sits
    # inside a multi-step registration form.
    LOGIN_OTP_EXPIRES_MINUTES: int = Field(default=5)

    # Resends allowed per challenge series. 2 resends => 3 total sends.
    LOGIN_OTP_MAX_RESENDS: int = Field(default=2)

    # Minimum gap between two sends. A request refused by this cooldown does
    # NOT consume a resend — see LoginOtpService.request_otp.
    LOGIN_OTP_RESEND_COOLDOWN_SECONDS: int = Field(default=60)

    # Wrong codes allowed against ONE code. Reset to zero on every new send,
    # which is what keeps this counter independent of the resend budget.
    LOGIN_OTP_MAX_VERIFICATION_ATTEMPTS: int = Field(default=5)

    # How long the OTP flow is locked once the resend budget is spent.
    LOGIN_OTP_LOCKOUT_MINUTES: int = Field(default=30)

    # How long a series stays "current". Past this, the next request starts a
    # fresh series with a fresh budget. Equal to the lockout duration by
    # design: the rule is then expressible as "three codes per address per 30
    # minutes", and a user who simply gives up waits no longer than one who hit
    # the lock.
    LOGIN_OTP_SERIES_WINDOW_MINUTES: int = Field(default=30)

    # Per-IP budgets. In-process and therefore per-replica — a cheap first
    # line only. The authoritative controls are the database-backed resend
    # budget and attempt counter.
    LOGIN_OTP_REQUESTS_PER_IP_PER_MINUTE: int = Field(default=5)
    LOGIN_OTP_VERIFY_PER_IP_PER_MINUTE: int = Field(default=10)

    # Server-side secret mixed into the OTP hash before storage. This is the
    # control that makes SHA-256 acceptable for a 6-digit secret: without a
    # pepper, all 10^6 hashes are computable in milliseconds from a leaked
    # row, so a database-only compromise would expose every live code. Held in
    # configuration and never in the database.
    LOGIN_OTP_PEPPER: str = Field(default="")

    @model_validator(mode="after")
    def _require_pepper_when_enabled(self) -> "LoginOtpSettings":
        if self.LOGIN_OTP_ENABLED and not self.LOGIN_OTP_PEPPER:
            raise ValueError(
                "LOGIN_OTP_PEPPER must be set when LOGIN_OTP_ENABLED is true. "
                "Without it, stored OTP hashes are trivially reversible. "
                "Generate one with: python -c \"import secrets; "
                "print(secrets.token_urlsafe(32))\""
            )
        return self


otp_settings = LoginOtpSettings()
