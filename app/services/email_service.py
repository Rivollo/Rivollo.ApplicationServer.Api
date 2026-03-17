"""Email service using SendGrid API v3.

Gracefully skips sending if SENDGRID_API_KEY is not configured.
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


async def _send(to_email: str, to_name: str, subject: str, html_body: str) -> None:
    """Send an email via SendGrid. No-op if API key is not configured."""
    if not settings.SENDGRID_API_KEY:
        logger.warning(
            "SENDGRID_API_KEY not configured — skipping email to %s | subject: %s",
            to_email,
            subject,
        )
        return

    payload = {
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": settings.SENDGRID_FROM_EMAIL, "name": settings.SENDGRID_FROM_NAME},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _SENDGRID_URL,
                json=payload,
                headers={"Authorization": f"Bearer {settings.SENDGRID_API_KEY}"},
            )
        if resp.status_code not in (200, 202):
            logger.error(
                "SendGrid returned %s for email to %s: %s",
                resp.status_code,
                to_email,
                resp.text,
            )
    except httpx.RequestError as exc:
        logger.exception("Failed to send email to %s: %s", to_email, exc)


class EmailService:

    @staticmethod
    async def send_otp_email(to_email: str, name: str, otp: str, expires_minutes: int) -> None:
        """Send the password reset OTP email."""
        subject = f"{settings.SENDGRID_FROM_NAME} — Your Password Reset OTP"
        html_body = _otp_template(name=name, otp=otp, expires_minutes=expires_minutes)
        await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)

    @staticmethod
    async def send_password_reset_success_email(to_email: str, name: str) -> None:
        """Send a confirmation email after a successful password reset."""
        subject = f"{settings.SENDGRID_FROM_NAME} — Password Reset Successful"
        html_body = _reset_success_template(name=name, frontend_url=settings.FRONTEND_URL)
        await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------

def _otp_template(name: str, otp: str, expires_minutes: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Password Reset OTP</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f4f7;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f7;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

          <!-- Header -->
          <tr>
            <td style="background-color:#1a1a2e;padding:32px 40px;text-align:center;">
              <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;letter-spacing:1px;">
                {settings.SENDGRID_FROM_NAME}
              </h1>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="margin:0 0 16px;font-size:16px;color:#333333;">Hi <strong>{name}</strong>,</p>
              <p style="margin:0 0 24px;font-size:15px;color:#555555;line-height:1.6;">
                We received a request to reset your password. Use the OTP below to continue.
                This code is valid for <strong>{expires_minutes} minutes</strong>.
              </p>

              <!-- OTP Box -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;">
                <tr>
                  <td align="center">
                    <div style="display:inline-block;background-color:#f0f0ff;border:2px dashed #1a1a2e;border-radius:8px;padding:20px 40px;">
                      <span style="font-size:40px;font-weight:700;letter-spacing:12px;color:#1a1a2e;">
                        {otp}
                      </span>
                    </div>
                  </td>
                </tr>
              </table>

              <p style="margin:0 0 8px;font-size:13px;color:#888888;">
                If you did not request a password reset, you can safely ignore this email.
              </p>
              <p style="margin:0;font-size:13px;color:#888888;">
                Do not share this OTP with anyone.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#f4f4f7;padding:20px 40px;text-align:center;">
              <p style="margin:0;font-size:12px;color:#aaaaaa;">
                &copy; 2025 {settings.SENDGRID_FROM_NAME}. All rights reserved.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _reset_success_template(name: str, frontend_url: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Password Reset Successful</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f4f7;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f7;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

          <!-- Header -->
          <tr>
            <td style="background-color:#1a1a2e;padding:32px 40px;text-align:center;">
              <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;letter-spacing:1px;">
                {settings.SENDGRID_FROM_NAME}
              </h1>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="margin:0 0 16px;font-size:16px;color:#333333;">Hi <strong>{name}</strong>,</p>

              <!-- Success icon -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;">
                <tr>
                  <td align="center">
                    <div style="width:64px;height:64px;background-color:#e6f9f0;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto;">
                      <span style="font-size:32px;">&#10003;</span>
                    </div>
                  </td>
                </tr>
              </table>

              <p style="margin:0 0 24px;font-size:15px;color:#555555;line-height:1.6;text-align:center;">
                Your password has been reset successfully.<br/>
                You can now log in with your new password.
              </p>

              <!-- Login button -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 32px;">
                <tr>
                  <td align="center">
                    <a href="{frontend_url}/login"
                       style="display:inline-block;background-color:#1a1a2e;color:#ffffff;text-decoration:none;font-size:15px;font-weight:600;padding:14px 40px;border-radius:6px;">
                      Go to Login
                    </a>
                  </td>
                </tr>
              </table>

              <p style="margin:0;font-size:13px;color:#888888;text-align:center;">
                If you did not make this change, please contact our support team immediately.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#f4f4f7;padding:20px 40px;text-align:center;">
              <p style="margin:0;font-size:12px;color:#aaaaaa;">
                &copy; 2025 {settings.SENDGRID_FROM_NAME}. All rights reserved.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
