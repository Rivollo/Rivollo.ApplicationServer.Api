"""Email service using Resend API.

Gracefully skips sending if RESEND_API_KEY is not configured.
"""

import logging
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"


async def _send(
    to_email: str,
    to_name: str,
    subject: str,
    html_body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
) -> None:
    """Send an email via Resend."""
    if not settings.RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not configured.")

    payload = {
        "from": f"{settings.RESEND_FROM_NAME} <{settings.RESEND_FROM_EMAIL}>",
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    }

    # Drop any CC/BCC that duplicates the recipient — Resend would deliver twice.
    cc_list = [addr for addr in (cc or []) if addr.lower() != to_email.lower()]
    if cc_list:
        payload["cc"] = cc_list

    bcc_list = [addr for addr in (bcc or []) if addr.lower() != to_email.lower()]
    if bcc_list:
        payload["bcc"] = bcc_list

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _RESEND_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
    except httpx.TimeoutException:
        logger.error("Resend timeout | to: %s | subject: %s", to_email, subject)
        raise RuntimeError("Failed to send email: request timed out.")
    except httpx.RequestError as exc:
        logger.error("Resend network error | to: %s | subject: %s | error: %s", to_email, subject, exc)
        raise RuntimeError(f"Failed to send email: network error — {exc}")

    if resp.status_code != 200:
        try:
            error_detail = resp.json()
        except Exception:
            error_detail = resp.text
        logger.error(
            "Resend error | to: %s | subject: %s | status: %s | detail: %s",
            to_email, subject, resp.status_code, error_detail,
        )
        raise RuntimeError(f"Resend error (status {resp.status_code}): {error_detail}")

    logger.info("Email sent | to: %s | subject: %s", to_email, subject)


_RESEND_AUDIENCES_URL = "https://api.resend.com/audiences"


async def _add_contact(email: str, name: Optional[str] = None) -> None:
    """Add or update a contact in the configured Resend audience.

    Resend upserts by email — calling this for an address already in the
    audience just updates it, never duplicates or errors. Requires
    RESEND_API_KEY to have "Full access" permission; a "Sending access" key
    is rejected here with a 401, even though it can send emails fine.
    """
    if not settings.RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not configured.")
    if not settings.RESEND_AUDIENCE_ID:
        raise RuntimeError("RESEND_AUDIENCE_ID is not configured.")

    first_name, _, last_name = (name or "").strip().partition(" ")

    payload: dict = {"email": email, "unsubscribed": False}
    if first_name:
        payload["first_name"] = first_name
    if last_name:
        payload["last_name"] = last_name

    url = f"{_RESEND_AUDIENCES_URL}/{settings.RESEND_AUDIENCE_ID}/contacts"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
    except httpx.TimeoutException:
        logger.error("Resend timeout adding contact | email: %s", email)
        raise RuntimeError("Failed to add contact: request timed out.")
    except httpx.RequestError as exc:
        logger.error("Resend network error adding contact | email: %s | error: %s", email, exc)
        raise RuntimeError(f"Failed to add contact: network error — {exc}")

    if resp.status_code not in (200, 201):
        try:
            error_detail = resp.json()
        except Exception:
            error_detail = resp.text
        logger.error(
            "Resend error adding contact | email: %s | status: %s | detail: %s",
            email, resp.status_code, error_detail,
        )
        raise RuntimeError(f"Resend error (status {resp.status_code}): {error_detail}")

    logger.info("Contact added to Resend audience | email: %s", email)


class EmailService:

    @staticmethod
    async def send_otp_email(to_email: str, name: str, otp: str, expires_minutes: int) -> None:
        """Send the password reset OTP email."""
        subject = f"{settings.RESEND_FROM_NAME} — Your Password Reset OTP"
        html_body = _otp_template(name=name, otp=otp, expires_minutes=expires_minutes)
        await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)

    @staticmethod
    async def send_password_reset_success_email(to_email: str, name: str) -> None:
        """Send a confirmation email after a successful password reset."""
        subject = f"{settings.RESEND_FROM_NAME} — Password Reset Successful"
        html_body = _reset_success_template(name=name, frontend_url=settings.FRONTEND_URL)
        await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)

    @staticmethod
    async def send_support_contact_email(fullname: str, comment: Optional[str], user_email: str) -> None:
        """Send a support contact notification to the support team."""
        if not settings.SUPPORT_EMAIL:
            raise RuntimeError("SUPPORT_EMAIL is not configured.")
        subject = f"New Support Request from {fullname}"
        html_body = _support_contact_template(fullname=fullname, comment=comment, user_email=user_email)
        await _send(to_email=settings.SUPPORT_EMAIL, to_name="Support Team", subject=subject, html_body=html_body)

    @staticmethod
    async def send_welcome_email(to_email: str, name: str) -> None:
        """Send a welcome email after successful account creation.

        BCC'd to WELCOME_EMAIL_CC so the team is notified of every new signup,
        without that address being visible to the recipient or other BCCs.
        """
        subject = f"Welcome to {settings.RESEND_FROM_NAME}!"
        html_body = _welcome_template(name=name, frontend_url=settings.FRONTEND_URL)
        await _send(
            to_email=to_email,
            to_name=name,
            subject=subject,
            html_body=html_body,
            bcc=settings.get_welcome_email_cc(),
        )

    @staticmethod
    async def add_user_to_audience(email: str, name: Optional[str] = None) -> None:
        """Add a newly created user's email to the configured Resend audience.

        Call this only for a genuinely NEW user, right alongside the welcome
        email — never on login, so a returning user is never re-added or
        otherwise touched. Skips silently (no exception) if RESEND_AUDIENCE_ID
        isn't configured, so environments without an audience set up (e.g. a
        fresh local .env) don't fail signup over an optional feature.
        """
        if not settings.RESEND_AUDIENCE_ID:
            logger.info("RESEND_AUDIENCE_ID not configured — skipping audience contact add for %s", email)
            return
        await _add_contact(email=email, name=name)

    @staticmethod
    async def send_signup_verification_otp(to_email: str, otp: str, expires_minutes: int) -> None:
        """Send the signup email verification OTP."""
        subject = f"{settings.RESEND_FROM_NAME} — Verify Your Email"
        html_body = _signup_otp_template(otp=otp, expires_minutes=expires_minutes)
        await _send(to_email=to_email, to_name=to_email, subject=subject, html_body=html_body)

# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------

def _banner_header(from_name: str) -> str:
    """Shared branded banner used at the top of every email."""
    return f"""
          <!-- ── Banner Header ── -->
          <tr>
            <td style="background:linear-gradient(135deg,#3a5bd9 0%,#1a1a4e 100%);padding:28px 40px 24px;">
              <table cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td style="vertical-align:middle;">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <!-- Grid icon -->
                        <td style="padding-right:12px;vertical-align:middle;">
                          <table cellpadding="0" cellspacing="0" style="width:32px;height:32px;background-color:rgba(255,255,255,0.15);border-radius:6px;">
                            <tr>
                              <td style="padding:6px 6px 3px 6px;">
                                <table cellpadding="0" cellspacing="0">
                                  <tr>
                                    <td style="width:8px;height:8px;background-color:#ffffff;border-radius:2px;font-size:0;line-height:0;">&nbsp;</td>
                                    <td style="width:3px;font-size:0;">&nbsp;</td>
                                    <td style="width:8px;height:8px;background-color:#ffffff;border-radius:2px;font-size:0;line-height:0;">&nbsp;</td>
                                  </tr>
                                  <tr><td colspan="3" style="height:3px;font-size:0;">&nbsp;</td></tr>
                                  <tr>
                                    <td style="width:8px;height:8px;background-color:#ffffff;border-radius:2px;font-size:0;line-height:0;">&nbsp;</td>
                                    <td style="width:3px;font-size:0;">&nbsp;</td>
                                    <td style="width:8px;height:8px;background-color:#ffffff;border-radius:2px;font-size:0;line-height:0;">&nbsp;</td>
                                  </tr>
                                </table>
                              </td>
                            </tr>
                          </table>
                        </td>
                        <!-- Wordmark -->
                        <td style="vertical-align:middle;">
                          <span style="color:#ffffff;font-size:22px;font-weight:700;letter-spacing:0.5px;font-family:Arial,sans-serif;">
                            {from_name}
                          </span>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>"""


def _footer(from_name: str) -> str:
    """Shared footer row for every email."""
    current_year = date.today().year
    support_email = settings.SUPPORT_EMAIL
    return f"""
          <!-- ── Footer ── -->
          <tr>
            <td style="background-color:#f8f9ff;padding:20px 40px;text-align:center;border-top:1px solid #e8eaf4;">
              <p style="margin:0 0 6px;font-size:12px;color:#aaaaaa;line-height:1.6;">
                Need help? Reach us at
                <a href="mailto:{support_email}" style="color:#3a5bd9;text-decoration:none;">{support_email}</a>
              </p>
              <p style="margin:0;font-size:11px;color:#cccccc;">
                &copy; {current_year} {from_name}. All rights reserved.
              </p>
            </td>
          </tr>"""


def _otp_template(name: str, otp: str, expires_minutes: int) -> str:
    banner = _banner_header(settings.RESEND_FROM_NAME)
    footer = _footer(settings.RESEND_FROM_NAME)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Password Reset OTP</title>
</head>
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

          {banner}

          <!-- ── Body ── -->
          <tr>
            <td style="padding:40px 40px 32px;">

              <p style="margin:0 0 8px;font-size:11px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;font-weight:600;">
                Password Reset
              </p>

              <p style="margin:0 0 16px;font-size:22px;color:#1a1a4e;font-weight:700;line-height:1.3;">
                Reset your password
              </p>

              <p style="margin:0 0 8px;font-size:15px;color:#333333;line-height:1.5;">
                Hi <strong>{name}</strong>,
              </p>
              <p style="margin:0 0 28px;font-size:14px;color:#666666;line-height:1.8;">
                We received a request to reset the password for your {settings.RESEND_FROM_NAME} account.
                Use the one-time code below to continue. For your security, this code expires in
                <strong style="color:#3a5bd9;">{expires_minutes} minutes</strong>.
              </p>

              <!-- OTP Box -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 28px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="background-color:#eef1fc;border:2px solid #3a5bd9;border-radius:12px;padding:20px 52px;text-align:center;">
                          <span style="font-size:42px;font-weight:700;letter-spacing:16px;color:#1a1a4e;font-family:'Courier New',Courier,monospace;">
                            {otp}
                          </span>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <!-- Tip box -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;">
                <tr>
                  <td style="background-color:#fff8e6;border-left:3px solid #f59e0b;border-radius:4px;padding:12px 16px;">
                    <p style="margin:0;font-size:13px;color:#92400e;line-height:1.6;">
                      <strong>Security tip:</strong> Never share this code with anyone.
                      {settings.RESEND_FROM_NAME} will never ask for your OTP via phone or chat.
                    </p>
                  </td>
                </tr>
              </table>

              <!-- Divider -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 16px;">
                <tr>
                  <td style="border-top:1px solid #eeeeee;font-size:0;line-height:0;">&nbsp;</td>
                </tr>
              </table>

              <p style="margin:0;font-size:12px;color:#999999;line-height:1.7;">
                If you did not request a password reset, you can safely ignore this email —
                your account remains secure and no changes have been made.
              </p>

            </td>
          </tr>

          {footer}

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""


def _reset_success_template(name: str, frontend_url: str) -> str:
    banner = _banner_header(settings.RESEND_FROM_NAME)
    footer = _footer(settings.RESEND_FROM_NAME)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Password Reset Successful</title>
</head>
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

          {banner}

          <!-- ── Body ── -->
          <tr>
            <td style="padding:40px 40px 32px;">

              <p style="margin:0 0 8px;font-size:11px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;font-weight:600;">
                Account Security
              </p>

              <p style="margin:0 0 16px;font-size:22px;color:#1a1a4e;font-weight:700;line-height:1.3;">
                Password updated successfully
              </p>

              <p style="margin:0 0 24px;font-size:15px;color:#333333;line-height:1.5;">
                Hi <strong>{name}</strong>,
              </p>

              <!-- Success icon -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="width:64px;height:64px;background:linear-gradient(135deg,#e8edfb,#d0d9f7);border-radius:50%;text-align:center;vertical-align:middle;">
                          <span style="font-size:28px;color:#3a5bd9;line-height:64px;display:block;">&#10003;</span>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <p style="margin:0 0 8px;font-size:14px;color:#555555;line-height:1.8;text-align:center;">
                Your {settings.RESEND_FROM_NAME} password has been reset successfully.
              </p>
              <p style="margin:0 0 28px;font-size:14px;color:#555555;line-height:1.8;text-align:center;">
                You can now sign in with your new credentials.
              </p>

              <!-- Login CTA -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 28px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="background:linear-gradient(135deg,#3a5bd9,#1a1a4e);border-radius:30px;">
                          <a href="{frontend_url}/login"
                             style="display:inline-block;color:#ffffff;text-decoration:none;font-size:14px;font-weight:700;padding:13px 40px;border-radius:30px;font-family:Arial,sans-serif;letter-spacing:0.3px;">
                            Sign In to {settings.RESEND_FROM_NAME} &rarr;
                          </a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <!-- Alert box -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 16px;">
                <tr>
                  <td style="background-color:#fff1f2;border-left:3px solid #f43f5e;border-radius:4px;padding:12px 16px;">
                    <p style="margin:0;font-size:13px;color:#9f1239;line-height:1.6;">
                      <strong>Wasn't you?</strong> If you did not make this change, please
                      <a href="mailto:{settings.SUPPORT_EMAIL}" style="color:#9f1239;">contact our support team</a>
                      immediately to secure your account.
                    </p>
                  </td>
                </tr>
              </table>

            </td>
          </tr>

          {footer}

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""


@dataclass(frozen=True)
class RivolloEmailSettings:
    """Values the new signup-OTP/welcome templates below are parametrized by.

    Sourced from the real app settings (see EMAIL_SETTINGS just below) —
    this dataclass exists only so the template functions stay easy to unit
    test / reuse standalone, not as a second place these values live.
    """

    resend_from_name: str = "Rivollo"
    support_email: str = "contact@rivollo.com"
    # Must be a real public HTTPS URL — email clients cannot load a local
    # file path or a data: URI. Empty falls back to a text wordmark.
    resend_logo_url: str = ""
    discord_invite_url: str = "https://discord.gg/cHwWTSFN5"


EMAIL_SETTINGS = RivolloEmailSettings(
    resend_from_name=settings.RESEND_FROM_NAME,
    support_email=settings.SUPPORT_EMAIL,
    resend_logo_url=settings.RESEND_LOGO_URL,
    discord_invite_url=settings.DISCORD_INVITE_URL,
)


def _safe_name(name: str | None) -> str:
    """Return an HTML-safe display name."""
    value = (name or "").strip()
    return escape(value if value else "there")


def _brand_banner(email_settings: RivolloEmailSettings = EMAIL_SETTINGS) -> str:
    """Centered logo/wordmark header for the signup-OTP and welcome emails.

    Distinct from _banner_header above (which the other three templates use
    unchanged) — this one renders an actual <img> when RESEND_LOGO_URL is
    configured, falling back to a styled text wordmark when it isn't.
    """

    brand_name = escape(email_settings.resend_from_name or "Rivollo")
    logo_url = escape(email_settings.resend_logo_url or "", quote=True)

    if logo_url:
        brand_html = f"""
          <a href="https://www.rivollo.com" target="_blank" style="display:inline-block;text-decoration:none;">
            <img src="{logo_url}" alt="{brand_name}" width="190" border="0"
              style="display:block;width:190px;max-width:190px;height:auto;margin:0 auto;border:0;outline:none;text-decoration:none;" />
          </a>
        """
    else:
        brand_html = f"""
          <a href="https://www.rivollo.com" target="_blank"
            style="display:inline-block;font-family:Arial,Helvetica,sans-serif;font-size:28px;line-height:34px;font-weight:800;letter-spacing:1px;color:#2364c7;text-decoration:none;">
            {brand_name}
          </a>
        """

    return f"""
          <tr>
            <td align="center" bgcolor="#ffffff" style="padding:28px 24px;background-color:#ffffff;border-bottom:1px solid #eaecf0;text-align:center;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" style="margin:0 auto;">
                <tr><td align="center" style="text-align:center;">{brand_html}</td></tr>
              </table>
            </td>
          </tr>"""


def _brand_footer(email_settings: RivolloEmailSettings = EMAIL_SETTINGS) -> str:
    """Shared transactional footer for the signup-OTP and welcome emails."""

    current_year = date.today().year
    brand_name = escape(email_settings.resend_from_name or "Rivollo")
    support_email = escape(email_settings.support_email or "contact@rivollo.com", quote=True)

    return f"""
          <tr>
            <td align="center" bgcolor="#f8f9fc" style="padding:24px 36px 26px;background-color:#f8f9fc;border-top:1px solid #eaecf0;text-align:center;">
              <p style="margin:0 0 8px;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:21px;color:#98a2b3;">
                Need help? Reach us at
                <a href="mailto:{support_email}" style="color:#315BD6;text-decoration:none;">{support_email}</a>
              </p>
              <p style="margin:0 0 7px;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:20px;color:#b0b7c3;">
                &copy; {current_year} {brand_name}. All rights reserved.
              </p>
              <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:20px;">
                <a href="https://www.rivollo.com" target="_blank" style="color:#98a2b3;text-decoration:none;">www.rivollo.com</a>
              </p>
            </td>
          </tr>"""


def _signup_otp_template(
    otp: str,
    expires_minutes: int,
    name: str = "there",
    email_settings: RivolloEmailSettings = EMAIL_SETTINGS,
) -> str:
    banner = _brand_banner(email_settings)
    footer = _brand_footer(email_settings)

    display_name = _safe_name(name)
    otp_value = escape(str(otp))
    expiry = max(1, int(expires_minutes))
    brand_name = escape(email_settings.resend_from_name or "Rivollo")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta name="x-apple-disable-message-reformatting" />
  <meta name="color-scheme" content="light" />
  <meta name="supported-color-schemes" content="light" />
  <title>Verify your {brand_name} account</title>
  <!--[if mso]>
  <style>
    table {{ border-collapse:collapse; }}
    td, p, a, h1 {{ font-family:Arial,Helvetica,sans-serif !important; }}
  </style>
  <![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#f4f6fb;font-family:Arial,Helvetica,sans-serif;color:#101828;">

  <div style="display:none;font-size:1px;color:#f4f6fb;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;mso-hide:all;">
    Your {brand_name} verification code is {otp_value}. It expires in {expiry} minutes.
  </div>

  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#f4f6fb" style="width:100%;background-color:#f4f6fb;">
    <tr>
      <td align="center" style="padding:32px 16px;">
        <table role="presentation" width="620" cellpadding="0" cellspacing="0" border="0" bgcolor="#ffffff"
          style="width:100%;max-width:620px;background-color:#ffffff;border-radius:16px;overflow:hidden;">

          {banner}

          <tr>
            <td align="center" bgcolor="#ffffff" style="padding:46px 42px 22px;background-color:#ffffff;text-align:center;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" style="margin:0 auto 20px;">
                <tr>
                  <td align="center" bgcolor="#eef4ff" style="padding:9px 18px;background-color:#eef4ff;border-radius:999px;text-align:center;">
                    <span style="font-size:12px;line-height:16px;font-weight:700;letter-spacing:1px;color:#2459c4;text-transform:uppercase;">
                      Account Verification
                    </span>
                  </td>
                </tr>
              </table>
              <h1 style="margin:0 0 14px;font-size:30px;line-height:39px;font-weight:800;color:#10145f;">Verify your email</h1>
              <p style="margin:0 auto 8px;max-width:500px;font-size:16px;line-height:27px;color:#344054;">Hi <strong>{display_name}</strong>,</p>
              <p style="margin:0 auto;max-width:500px;font-size:15px;line-height:26px;color:#667085;">
                Thanks for signing up for {brand_name}. Use the verification code below to complete your account setup.
              </p>
            </td>
          </tr>

          <tr>
            <td align="center" bgcolor="#ffffff" style="padding:18px 42px 22px;background-color:#ffffff;text-align:center;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" style="margin:0 auto;">
                <tr>
                  <td align="center" bgcolor="#f5f7ff" style="padding:20px 32px;background-color:#f5f7ff;border:1px solid #dfe4ff;border-radius:14px;text-align:center;">
                    <span style="font-family:'Courier New',Courier,monospace;font-size:34px;line-height:42px;font-weight:800;letter-spacing:8px;color:#315BD6;">
                      {otp_value}
                    </span>
                  </td>
                </tr>
              </table>
              <p style="margin:18px 0 0;font-size:14px;line-height:22px;color:#667085;">
                This code will expire in <strong style="color:#315BD6;">{expiry} minutes</strong>.
              </p>
            </td>
          </tr>

          <tr>
            <td bgcolor="#ffffff" style="padding:10px 42px 42px;background-color:#ffffff;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#f8f9fc"
                style="width:100%;background-color:#f8f9fc;border-radius:12px;">
                <tr>
                  <td style="padding:18px 20px;text-align:left;">
                    <p style="margin:0 0 6px;font-size:14px;line-height:22px;font-weight:700;color:#344054;">
                      Keep your verification code private
                    </p>
                    <p style="margin:0;font-size:13px;line-height:21px;color:#667085;">
                      {brand_name} will never ask you to share this code with anyone.
                      If you did not attempt to create an account, you can safely ignore this email.
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          {footer}

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""


def _welcome_template(
    name: str,
    frontend_url: str,
    email_settings: RivolloEmailSettings = EMAIL_SETTINGS,
) -> str:
    banner = _brand_banner(email_settings)
    footer = _brand_footer(email_settings)

    display_name = _safe_name(name)
    brand_name = escape(email_settings.resend_from_name or "Rivollo")
    dashboard_url = escape((frontend_url or "https://app.rivollo.com").rstrip("/"), quote=True)
    discord_url = escape(email_settings.discord_invite_url or "https://discord.gg/cHwWTSFN5", quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta name="x-apple-disable-message-reformatting" />
  <meta name="color-scheme" content="light" />
  <meta name="supported-color-schemes" content="light" />
  <title>Welcome to {brand_name}</title>
  <!--[if mso]>
  <style>
    table {{ border-collapse:collapse; }}
    td, p, a, h1, h2 {{ font-family:Arial,Helvetica,sans-serif !important; }}
  </style>
  <![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#f4f6fb;font-family:Arial,Helvetica,sans-serif;color:#101828;">

  <div style="display:none;font-size:1px;color:#f4f6fb;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;mso-hide:all;">
    Welcome to {brand_name}, {display_name}. Your account is ready to start creating your first interactive 3D experience.
  </div>

  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#f4f6fb" style="width:100%;background-color:#f4f6fb;">
    <tr>
      <td align="center" style="padding:32px 16px;">
        <table role="presentation" width="680" cellpadding="0" cellspacing="0" border="0" bgcolor="#ffffff"
          style="width:100%;max-width:680px;background-color:#ffffff;border-radius:16px;overflow:hidden;">

          {banner}

          <tr>
            <td align="center" bgcolor="#ffffff" style="padding:48px 48px 30px;background-color:#ffffff;text-align:center;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" align="center" style="margin:0 auto 20px;">
                <tr>
                  <td align="center" bgcolor="#eef4ff" style="padding:9px 18px;background-color:#eef4ff;border-radius:999px;">
                    <span style="font-size:12px;line-height:16px;font-weight:700;letter-spacing:1px;color:#2459c4;text-transform:uppercase;">
                      Welcome to {brand_name}
                    </span>
                  </td>
                </tr>
              </table>
              <h1 style="margin:0 0 12px;font-size:32px;line-height:41px;font-weight:800;color:#10145f;">Welcome, {display_name}!</h1>
              <p style="margin:0 auto;max-width:560px;font-size:16px;line-height:28px;color:#667085;">
                Your {brand_name} account is ready. Turn product images into interactive 3D experiences,
                customize how they are presented, and share them with customers from one place.
              </p>
            </td>
          </tr>

          <tr>
            <td align="center" bgcolor="#ffffff" style="padding:38px 48px 34px;background-color:#ffffff;text-align:center;">
              <h2 style="margin:0 0 10px;font-size:23px;line-height:31px;font-weight:800;color:#10145f;">
                Ready to create your first 3D model?
              </h2>
              <p style="margin:0 0 24px;font-size:15px;line-height:25px;color:#667085;">
                Open your {brand_name} dashboard and start creating.
              </p>
              <table role="presentation" width="270" cellpadding="0" cellspacing="0" border="0" align="center" style="width:270px;margin:0 auto;">
                <tr>
                  <td align="center" bgcolor="#315BD6" style="background-color:#315BD6;border-radius:28px;text-align:center;">
                    <a href="{dashboard_url}" target="_blank"
                      style="display:block;width:268px;padding:16px 0;border:1px solid #315BD6;border-radius:28px;background-color:#315BD6;color:#ffffff !important;font-family:Arial,Helvetica,sans-serif;font-size:16px;line-height:20px;font-weight:700;text-decoration:none;text-align:center;">
                      Go to My Dashboard &rarr;
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <tr>
            <td align="center" style="padding:0 48px 44px;text-align:center;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#f8f9fc"
                style="width:100%;background-color:#f8f9fc;border-radius:12px;">
                <tr>
                  <td align="center" style="padding:24px;text-align:center;">
                    <p style="margin:0 0 7px;font-size:15px;line-height:23px;font-weight:700;color:#344054;">
                      Join the {brand_name} community
                    </p>
                    <p style="margin:0 0 20px;font-size:13px;line-height:21px;color:#667085;">
                      Get product updates, share your 3D creations, and connect with our team on Discord.
                    </p>
                    <table role="presentation" width="270" cellpadding="0" cellspacing="0" border="0" align="center" style="width:270px;margin:0 auto;">
                      <tr>
                        <td align="center" bgcolor="#5865F2" style="background-color:#5865F2;border-radius:28px;text-align:center;">
                          <a href="{discord_url}" target="_blank"
                            style="display:block;width:268px;padding:16px 0;border:1px solid #5865F2;border-radius:28px;background-color:#5865F2;color:#ffffff !important;font-family:Arial,Helvetica,sans-serif;font-size:16px;line-height:20px;font-weight:700;text-decoration:none;text-align:center;">
                            Join {brand_name} on Discord &rarr;
                          </a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          {footer}

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""


def _support_contact_template(fullname: str, comment: Optional[str], user_email: str) -> str:
    banner = _banner_header(settings.RESEND_FROM_NAME)
    footer = _footer(settings.RESEND_FROM_NAME)
    comment_html = comment.replace("\n", "<br/>") if comment else "<em style='color:#999999;'>No message provided.</em>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>New Support Request</title>
</head>
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

          {banner}

          <!-- Body -->
          <tr>
            <td style="padding:40px 40px 32px;">

              <p style="margin:0 0 8px;font-size:11px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;font-weight:600;">
                Support Request
              </p>

              <p style="margin:0 0 24px;font-size:22px;color:#1a1a4e;font-weight:700;line-height:1.3;">
                New message from {fullname}
              </p>

              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;border:1px solid #e8eaf4;border-radius:8px;overflow:hidden;">
                <tr>
                  <td style="background-color:#f8f9ff;padding:10px 16px;font-size:12px;color:#888888;font-weight:600;text-transform:uppercase;letter-spacing:1px;width:120px;">Name</td>
                  <td style="padding:10px 16px;font-size:14px;color:#1a1a4e;">{fullname}</td>
                </tr>
                <tr>
                  <td style="background-color:#f8f9ff;padding:10px 16px;font-size:12px;color:#888888;font-weight:600;text-transform:uppercase;letter-spacing:1px;border-top:1px solid #e8eaf4;">Email</td>
                  <td style="padding:10px 16px;font-size:14px;color:#1a1a4e;border-top:1px solid #e8eaf4;">
                    <a href="mailto:{user_email}" style="color:#3a5bd9;text-decoration:none;">{user_email}</a>
                  </td>
                </tr>
                <tr>
                  <td style="background-color:#f8f9ff;padding:10px 16px;font-size:12px;color:#888888;font-weight:600;text-transform:uppercase;letter-spacing:1px;border-top:1px solid #e8eaf4;vertical-align:top;">Message</td>
                  <td style="padding:10px 16px;font-size:14px;color:#333333;line-height:1.7;border-top:1px solid #e8eaf4;">{comment_html}</td>
                </tr>
              </table>

            </td>
          </tr>

          {footer}

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""

