"""Delivery of the login OTP email.

Reuses the existing Resend transport WITHOUT modifying it. The module-level
``_send`` in app/services/email_service.py already owns the HTTPS call, the
10-second timeout, the error handling and the cc/bcc de-duplication, and it is
exactly the interface this flow needs.

Importing an underscore-named function across modules is a convention
violation, and it is confined to this file on purpose so it is visible in
review rather than scattered. The alternative — adding a send_login_otp method
to EmailService — would modify the existing email service, which this feature
must not do.

The existing public method EmailService.send_otp_email is deliberately NOT
reused: its subject is "Your Password Reset OTP" and its body is password-reset
copy. Sending that for a sign-in is not a cosmetic problem — people are trained
to treat an unexpected password-reset email as a security alarm, so reusing it
would manufacture false alarms and teach them to ignore real ones.

The template below is therefore self-contained, reading only public settings so
that no further private symbols are borrowed.
"""

import logging
from html import escape

from app.core.config import settings
from app.services.email_service import _send

logger = logging.getLogger(__name__)


async def send_login_otp(to_email: str, name: str, otp: str, expires_minutes: int) -> None:
    """Email a sign-in code.

    Raises whatever _send raises (RuntimeError on timeout, network error or a
    non-200 from Resend). Callers run this through FastAPI BackgroundTasks, so
    a failure is logged by _send and never reaches the HTTP response — which is
    what keeps the timing of a request for a registered address
    indistinguishable from one for an unregistered address.
    """
    subject = f"{settings.RESEND_FROM_NAME} — Your sign-in code"
    html_body = _login_otp_template(
        name=name, otp=otp, expires_minutes=expires_minutes
    )
    await _send(to_email=to_email, to_name=name, subject=subject, html_body=html_body)


def _login_otp_template(name: str, otp: str, expires_minutes: int) -> str:
    """Build the sign-in code email.

    Every interpolated value is escaped: ``name`` comes from the database and
    is user-controlled. The OTP itself is digits only, but is escaped anyway so
    no future change to the code alphabet can open an injection path.
    """
    brand = escape(settings.RESEND_FROM_NAME or "Rivollo")
    safe_name = escape((name or "").strip()) or "there"
    safe_otp = escape(otp)
    logo_url = settings.RESEND_LOGO_URL

    if logo_url:
        header = (
            f'<img src="{escape(logo_url)}" alt="{brand}" height="32" '
            'style="display:block;border:0;outline:none;text-decoration:none;height:32px;" />'
        )
    else:
        header = (
            f'<span style="font-size:20px;font-weight:700;color:#111827;'
            f'letter-spacing:-0.02em;">{brand}</span>'
        )

    return f"""\
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background-color:#f4f5f7;
               font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           style="background-color:#f4f5f7;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                 style="max-width:520px;background-color:#ffffff;border-radius:12px;
                        border:1px solid #e5e7eb;overflow:hidden;">
            <tr>
              <td style="padding:28px 32px 8px 32px;">{header}</td>
            </tr>
            <tr>
              <td style="padding:8px 32px 0 32px;">
                <h1 style="margin:0 0 12px 0;font-size:20px;line-height:1.3;
                           color:#111827;font-weight:600;">
                  Your sign-in code
                </h1>
                <p style="margin:0 0 20px 0;font-size:15px;line-height:1.6;color:#4b5563;">
                  Hi {safe_name}, use this code to sign in to your {brand} account.
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:0 32px;">
                <div style="background-color:#f9fafb;border:1px solid #e5e7eb;
                            border-radius:10px;padding:20px;text-align:center;">
                  <div style="font-size:32px;font-weight:700;letter-spacing:8px;
                              color:#111827;font-family:'SF Mono',Menlo,Consolas,monospace;">
                    {safe_otp}
                  </div>
                </div>
                <p style="margin:16px 0 0 0;font-size:14px;line-height:1.6;color:#6b7280;">
                  This code expires in {expires_minutes} minutes and can be used once.
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:20px 32px 28px 32px;">
                <div style="border-top:1px solid #e5e7eb;padding-top:16px;">
                  <p style="margin:0;font-size:13px;line-height:1.6;color:#6b7280;">
                    If you didn't try to sign in, you can ignore this email —
                    your account is safe and nothing has changed.
                    {brand} will never ask you for this code by phone, chat or email.
                  </p>
                </div>
              </td>
            </tr>
          </table>
          <p style="margin:16px 0 0 0;font-size:12px;color:#9ca3af;">
            &copy; {brand}
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
"""
