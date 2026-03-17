"""Email service using SendGrid API v3.

Gracefully skips sending if SENDGRID_API_KEY is not configured.
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_SENDGRID_URL = settings.SENDGRID_URL


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
            try:
                error_detail = resp.json()
            except Exception:
                error_detail = resp.text

            logger.error(
                "SendGrid failed to send email to %s | subject: %s | status: %s | error: %s",
                to_email,
                subject,
                resp.status_code,
                error_detail,
            )
            raise RuntimeError(
                f"Failed to send email via SendGrid (status {resp.status_code}): {error_detail}"
            )
        logger.info("Email sent successfully to %s | subject: %s", to_email, subject)
    except httpx.TimeoutException:
        logger.error("SendGrid request timed out for email to %s | subject: %s", to_email, subject)
        raise RuntimeError("Failed to send email: request timed out. Please try again later.")
    except httpx.RequestError as exc:
        logger.error(
            "SendGrid network error for email to %s | subject: %s | error: %s",
            to_email,
            subject,
            exc,
        )
        raise RuntimeError("Failed to send email: a network error occurred. Please try again later.")


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
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;">

          <!-- ── Header ── -->
          <tr>
            <td style="background:linear-gradient(135deg,#3a5bd9 0%,#1a1a4e 100%);padding:24px 40px;">
              <table cellpadding="0" cellspacing="0">
                <tr>
                  <!-- Logo icon -->
                  <td style="padding-right:10px;vertical-align:middle;">
                    <table cellpadding="0" cellspacing="0" style="width:28px;height:28px;background-color:rgba(255,255,255,0.2);border-radius:5px;">
                      <tr>
                        <td style="padding:5px 5px 2px 5px;">
                          <table cellpadding="0" cellspacing="0">
                            <tr>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                              <td style="width:2px;font-size:0;">&nbsp;</td>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                            </tr>
                            <tr><td colspan="3" style="height:2px;font-size:0;">&nbsp;</td></tr>
                            <tr>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                              <td style="width:2px;font-size:0;">&nbsp;</td>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                            </tr>
                          </table>
                        </td>
                      </tr>
                    </table>
                  </td>
                  <!-- Wordmark -->
                  <td style="vertical-align:middle;">
                    <span style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:0.5px;">
                      {settings.SENDGRID_FROM_NAME}
                    </span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- ── Body ── -->
          <tr>
            <td style="padding:36px 40px 28px;">

              <!-- Eyebrow label -->
              <p style="margin:0 0 10px;font-size:10px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;">
                Password Reset
              </p>

              <p style="margin:0 0 14px;font-size:15px;color:#1a1a4e;font-weight:500;line-height:1.5;">
                Hi <strong>{name}</strong>,
              </p>
              <p style="margin:0 0 26px;font-size:14px;color:#555555;line-height:1.7;">
                We received a request to reset your password. Use the OTP below to
                continue. This code is valid for
                <strong style="color:#3a5bd9;">{expires_minutes} minutes</strong>.
              </p>

              <!-- OTP Box -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 26px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="background-color:#eef1fc;border:2px solid #3a5bd9;border-radius:10px;padding:18px 44px;text-align:center;">
                          <span style="font-size:40px;font-weight:700;letter-spacing:14px;color:#1a1a4e;font-family:Courier,monospace;">
                            {otp}
                          </span>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <!-- Divider -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 16px;">
                <tr>
                  <td style="border-top:1px solid #eeeeee;font-size:0;line-height:0;">&nbsp;</td>
                </tr>
              </table>

              <p style="margin:0 0 6px;font-size:12px;color:#999999;line-height:1.6;">
                If you did not request a password reset, you can safely ignore this email.
              </p>
              <p style="margin:0;font-size:12px;color:#999999;line-height:1.6;">
                Do not share this OTP with anyone.
              </p>

            </td>
          </tr>

          <!-- ── Footer ── -->
          <tr>
            <td style="background-color:#f8f9ff;padding:16px 40px;text-align:center;border-top:1px solid #e8eaf4;">
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
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;">

          <!-- ── Header ── -->
          <tr>
            <td style="background:linear-gradient(135deg,#3a5bd9 0%,#1a1a4e 100%);padding:24px 40px;">
              <table cellpadding="0" cellspacing="0">
                <tr>
                  <!-- Logo icon -->
                  <td style="padding-right:10px;vertical-align:middle;">
                    <table cellpadding="0" cellspacing="0" style="width:28px;height:28px;background-color:rgba(255,255,255,0.2);border-radius:5px;">
                      <tr>
                        <td style="padding:5px 5px 2px 5px;">
                          <table cellpadding="0" cellspacing="0">
                            <tr>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                              <td style="width:2px;font-size:0;">&nbsp;</td>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                            </tr>
                            <tr><td colspan="3" style="height:2px;font-size:0;">&nbsp;</td></tr>
                            <tr>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                              <td style="width:2px;font-size:0;">&nbsp;</td>
                              <td style="width:7px;height:7px;background-color:#ffffff;border-radius:1px;font-size:0;line-height:0;">&nbsp;</td>
                            </tr>
                          </table>
                        </td>
                      </tr>
                    </table>
                  </td>
                  <!-- Wordmark -->
                  <td style="vertical-align:middle;">
                    <span style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:0.5px;">
                      {settings.SENDGRID_FROM_NAME}
                    </span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- ── Body ── -->
          <tr>
            <td style="padding:36px 40px 28px;">

              <!-- Eyebrow label -->
              <p style="margin:0 0 10px;font-size:10px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;">
                Account Security
              </p>

              <p style="margin:0 0 24px;font-size:15px;color:#1a1a4e;font-weight:500;line-height:1.5;">
                Hi <strong>{name}</strong>,
              </p>

              <!-- Success icon -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="width:60px;height:60px;background-color:#e8edfb;border-radius:50%;text-align:center;vertical-align:middle;">
                          <span style="font-size:26px;color:#3a5bd9;line-height:60px;display:block;">&#10003;</span>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <p style="margin:0 0 26px;font-size:14px;color:#555555;line-height:1.7;text-align:center;">
                Your password has been reset successfully.<br/>
                You can now log in with your new password.
              </p>

              <!-- Login CTA button (pill shape) -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 26px;">
                <tr>
                  <td align="center">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td style="background:linear-gradient(135deg,#3a5bd9,#1a1a4e);border-radius:30px;">
                          <a href="{frontend_url}/login"
                             style="display:inline-block;color:#ffffff;text-decoration:none;font-size:14px;font-weight:700;padding:12px 36px;border-radius:30px;font-family:Arial,sans-serif;">
                            Go to Login &rarr;
                          </a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <!-- Divider -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 16px;">
                <tr>
                  <td style="border-top:1px solid #eeeeee;font-size:0;line-height:0;">&nbsp;</td>
                </tr>
              </table>

              <p style="margin:0;font-size:12px;color:#999999;text-align:center;line-height:1.6;">
                If you did not make this change, please contact our support team immediately.
              </p>

            </td>
          </tr>

          <!-- ── Footer ── -->
          <tr>
            <td style="background-color:#f8f9ff;padding:16px 40px;text-align:center;border-top:1px solid #e8eaf4;">
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