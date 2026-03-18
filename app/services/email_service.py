"""Email service using SendGrid API v3.

Gracefully skips sending if SENDGRID_API_KEY is not configured.
"""

import logging
from datetime import date

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

    @staticmethod
    async def send_welcome_email(to_email: str, name: str) -> None:
        """Send a welcome email after successful account creation."""
        subject = f"Welcome to {settings.SENDGRID_FROM_NAME}!"
        html_body = _welcome_template(name=name, frontend_url=settings.FRONTEND_URL)
        await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)

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


def _footer(from_name: str, from_email: str) -> str:
    """Shared footer row for every email."""
    current_year = date.today().year
    return f"""
          <!-- ── Footer ── -->
          <tr>
            <td style="background-color:#f8f9ff;padding:20px 40px;text-align:center;border-top:1px solid #e8eaf4;">
              <p style="margin:0 0 6px;font-size:12px;color:#aaaaaa;line-height:1.6;">
                Need help? Reach us at
                <a href="mailto:{from_email}" style="color:#3a5bd9;text-decoration:none;">{from_email}</a>
              </p>
              <p style="margin:0;font-size:11px;color:#cccccc;">
                &copy; {current_year} {from_name}. All rights reserved.
              </p>
            </td>
          </tr>"""


def _otp_template(name: str, otp: str, expires_minutes: int) -> str:
    banner = _banner_header(settings.SENDGRID_FROM_NAME)
    footer = _footer(settings.SENDGRID_FROM_NAME, settings.SENDGRID_FROM_EMAIL)
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
                We received a request to reset the password for your {settings.SENDGRID_FROM_NAME} account.
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
                      {settings.SENDGRID_FROM_NAME} will never ask for your OTP via phone or chat.
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
    banner = _banner_header(settings.SENDGRID_FROM_NAME)
    footer = _footer(settings.SENDGRID_FROM_NAME, settings.SENDGRID_FROM_EMAIL)
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
                Your {settings.SENDGRID_FROM_NAME} password has been reset successfully.
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
                            Sign In to {settings.SENDGRID_FROM_NAME} &rarr;
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
                      <a href="mailto:{settings.SENDGRID_FROM_EMAIL}" style="color:#9f1239;">contact our support team</a>
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


def _welcome_template(name: str, frontend_url: str) -> str:
    banner = _banner_header(settings.SENDGRID_FROM_NAME)
    footer = _footer(settings.SENDGRID_FROM_NAME, settings.SENDGRID_FROM_EMAIL)
    font_stack = "-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Welcome to {settings.SENDGRID_FROM_NAME}</title>
</head>
<body style="margin:0;padding:0;background-color:#f0f2f8;font-family:{font_stack};">

  <!-- Preheader text (hidden, shows in inbox preview) -->
  <div style="display:none;max-height:0;overflow:hidden;mso-hide:all;font-size:1px;color:#f0f2f8;line-height:1px;">
    Your {settings.SENDGRID_FROM_NAME} account is ready — start building your product catalogue today.&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;&nbsp;&#847;
  </div>

  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0f2f8;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

          {banner}

          <!-- ── Hero ── -->
          <tr>
            <td style="background:linear-gradient(180deg,#eef1fc 0%,#ffffff 100%);padding:40px 40px 30px;text-align:center;">
              <p style="margin:0 0 8px;font-size:11px;letter-spacing:2.5px;color:#3a5bd9;text-transform:uppercase;font-weight:600;">
                Welcome to {settings.SENDGRID_FROM_NAME}
              </p>
              <p style="margin:0 0 14px;font-size:26px;color:#1a1a4e;font-weight:700;line-height:1.3;">
                Hello, {name} — glad to have you.
              </p>
              <p style="margin:0;font-size:15px;color:#555555;line-height:1.8;max-width:460px;display:inline-block;">
                Your account is ready. Turn any product into an immersive 3D experience, instantly.
              </p>
            </td>
          </tr>

          <!-- ── Divider ── -->
          <tr>
            <td style="padding:0 40px;">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr><td style="border-top:1px solid #eeeeee;font-size:0;line-height:0;">&nbsp;</td></tr>
              </table>
            </td>
          </tr>

          <!-- ── Features ── -->
          <tr>
            <td style="padding:28px 40px 8px;">
              <p style="margin:0 0 20px;font-size:13px;color:#1a1a4e;font-weight:700;text-transform:uppercase;letter-spacing:1px;">
                What you can do with {settings.SENDGRID_FROM_NAME}
              </p>

              <!-- Feature 1 -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
                <tr>
                  <td style="width:40px;vertical-align:top;padding-top:2px;">
                    <table cellpadding="0" cellspacing="0"><tr>
                      <td style="width:34px;height:34px;background-color:#eef1fc;border-radius:8px;text-align:center;vertical-align:middle;">
                        <span style="font-size:16px;line-height:34px;display:block;">&#127919;</span>
                      </td>
                    </tr></table>
                  </td>
                  <td style="padding-left:16px;vertical-align:top;">
                    <p style="margin:0 0 3px;font-size:14px;color:#1a1a4e;font-weight:600;">2D → interactive 3D conversion</p>
                    <p style="margin:0;font-size:13px;color:#777777;line-height:1.7;">Turn product photos into web-ready 3D models with hotspots, custom backgrounds, and purchase links.</p>
                  </td>
                </tr>
              </table>

              <!-- Feature 2 -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 18px;">
                <tr>
                  <td style="width:40px;vertical-align:top;padding-top:2px;">
                    <table cellpadding="0" cellspacing="0"><tr>
                      <td style="width:34px;height:34px;background-color:#eef1fc;border-radius:8px;text-align:center;vertical-align:middle;">
                        <span style="font-size:16px;line-height:34px;display:block;">&#128279;</span>
                      </td>
                    </tr></table>
                  </td>
                  <td style="padding-left:16px;vertical-align:top;">
                    <p style="margin:0 0 3px;font-size:14px;color:#1a1a4e;font-weight:600;">Share your storefront instantly</p>
                    <p style="margin:0;font-size:13px;color:#777777;line-height:1.7;">Generate a shareable link and send it to clients, buyers, or partners.</p>
                  </td>
                </tr>
              </table>

              <!-- Feature 4 -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 32px;">
                <tr>
                  <td style="width:40px;vertical-align:top;padding-top:2px;">
                    <table cellpadding="0" cellspacing="0"><tr>
                      <td style="width:34px;height:34px;background-color:#eef1fc;border-radius:8px;text-align:center;vertical-align:middle;">
                        <span style="font-size:16px;line-height:34px;display:block;">&#128202;</span>
                      </td>
                    </tr></table>
                  </td>
                  <td style="padding-left:16px;vertical-align:top;">
                    <p style="margin:0 0 3px;font-size:14px;color:#1a1a4e;font-weight:600;">Product analytics</p>
                    <p style="margin:0;font-size:13px;color:#777777;line-height:1.7;">Track views, engagement, and buyer interest per product — so you know exactly what's working.</p>
                  </td>
                </tr>
              </table>

            </td>
          </tr>

          <!-- ── CTA ── -->
          <tr>
            <td style="padding:0 40px 40px;text-align:center;">
              <table cellpadding="0" cellspacing="0" style="margin:0 auto;">
                <tr>
                  <td style="background:linear-gradient(135deg,#3a5bd9,#1a1a4e);border-radius:30px;">
                    <a href="{frontend_url}"
                       style="display:inline-block;color:#ffffff;text-decoration:none;font-size:15px;font-weight:700;padding:14px 48px;border-radius:30px;font-family:{font_stack};letter-spacing:0.3px;">
                      Go to my dashboard &rarr;
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- ── Divider ── -->
          <tr>
            <td style="padding:0 40px;">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr><td style="border-top:1px solid #eeeeee;font-size:0;line-height:0;">&nbsp;</td></tr>
              </table>
            </td>
          </tr>

          <!-- ── Footer ── -->
          <tr>
            <td style="padding:20px 40px 28px;text-align:center;">
              <p style="margin:0 0 6px;font-size:12px;color:#aaaaaa;line-height:1.8;">
                You received this email because you signed up for {settings.SENDGRID_FROM_NAME}.
                <a href="{frontend_url}/unsubscribe" style="color:#aaaaaa;text-decoration:underline;">Unsubscribe</a>
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

