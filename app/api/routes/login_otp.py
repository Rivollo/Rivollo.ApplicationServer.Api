"""Email OTP login routes.

A router of its own, deliberately: app/api/routes/auth.py holds signup, the
signup OTP pair, password login, Google login, password reset, account restore
and the app token, and this feature must leave every one of them untouched. A
separate module means that file never opens, and it gives operations a clean
kill switch — one commented-out include_router line removes the feature with no
risk of having disturbed login on the way out.

Both endpoints carry the same AppTokenVerified gate as every other /auth/*
route, so a client application must present a valid app token before any OTP
work happens.

Path shape: /auth/otp/request and /auth/otp/verify are nested because
/auth/verify-otp is ALREADY TAKEN by password-reset verification. Nesting is
precedented here by /auth/account/restore.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from app.api.deps import DB, AppTokenVerified
from app.core.login_otp_config import otp_settings
from app.schemas.login_otp import (
    OtpLoginRequest,
    OtpLoginResponse,
    OtpLoginVerifyRequest,
)
from app.services.activity_service import ActivityService
from app.services.login_otp_email import send_login_otp
from app.services.login_otp_rate_limit import (
    client_ip,
    request_limiter,
    verify_limiter,
)
from app.services.login_otp_service import (
    GENERIC_REQUEST_MESSAGE,
    LoginOtpService,
)
from app.utils.envelopes import api_success

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth-otp"])


def _require_enabled() -> None:
    """Refuse when the feature is switched off.

    This is rollback level 1: setting LOGIN_OTP_ENABLED=false takes both
    endpoints out of service with a restart and no deploy, and cannot affect
    password login, Google login, signup or password reset.
    """
    if not otp_settings.LOGIN_OTP_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sign-in by email code is not available.",
        )


@router.post("/auth/otp/request", response_model=dict)
async def request_login_otp(
    payload: OtpLoginRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: DB,
    _: AppTokenVerified,
):
    """Send a sign-in code to an existing user, or resend the current one.

    This endpoint IS the resend endpoint: calling it again either reports the
    remaining cooldown or issues the next code. Three sends are allowed per
    challenge (one initial plus two resends); a fourth request locks the OTP
    flow for this address for 30 minutes.

    Answers 200 with an identical body whether or not the address belongs to an
    account, so it cannot be used to discover who has one. It never creates a
    user — signup remains the only way an account is created.

    A code IS sent to an account pending deletion: signing in inside the
    recovery window is how such an account comes back, and for an account with
    no password and no Google identity this is the only way back.
    """
    _require_enabled()
    ip = client_ip(request)
    request_limiter.check(ip)

    result = await LoginOtpService.request_otp(
        db=db, email=payload.email, request_ip=ip
    )

    if result.should_send:
        # Queued rather than awaited, for two reasons: Resend's latency (up to
        # a 10s timeout) stays out of the response, so a registered and an
        # unregistered address cannot be told apart by timing; and a Resend
        # outage cannot turn a code request into a 500.
        #
        # The trade is that delivery failure is invisible to the caller. _send
        # logs it, and the user's recovery is to request again after the
        # cooldown.
        background_tasks.add_task(
            send_login_otp,
            to_email=result.deliver_to_email,
            name=result.deliver_to_name,
            otp=result.deliver_otp,
            expires_minutes=result.expires_in_minutes,
        )

    # No email and no code in the audit row. The purge job's handoff already
    # records that the address must not be logged, and the code obviously must
    # not be. IP and user agent come from the request object.
    await ActivityService.log_activity(
        db=db,
        action="auth.otp.requested",
        target_type="login_otp",
        request=request,
    )

    return api_success(
        OtpLoginResponse(
            message=GENERIC_REQUEST_MESSAGE,
            expires_in_minutes=result.expires_in_minutes,
            resends_remaining=result.resends_remaining,
            resend_available_in_seconds=result.resend_available_in_seconds,
        ).model_dump()
    )


@router.post("/auth/otp/verify", response_model=dict)
async def verify_login_otp(
    payload: OtpLoginVerifyRequest,
    request: Request,
    db: DB,
    _: AppTokenVerified,
):
    """Verify a sign-in code and return an authenticated session.

    On success returns exactly the body /auth/login returns — the same
    AuthResponse, carrying a token minted by the same AuthService.generate_token
    — so a client that can handle a password login can handle this with no new
    response handling.

    A correct code inside the recovery window also RESTORES an account that is
    pending deletion, reporting it as account_restored / products_restored in
    the same body — the same thing a password sign-in does on /auth/login.

    Every failure mode answers 400 with one identical message. 403 if the
    account is deactivated, or if it was deleted and its recovery period has
    already ended; 429 if the address is locked or the caller is rate-limited.
    """
    _require_enabled()
    verify_limiter.check(client_ip(request))

    auth_response = await LoginOtpService.verify_otp(
        db=db,
        email=payload.email,
        otp=payload.otp,
        remember_me=payload.remember_me,
        request=request,
    )

    return api_success(auth_response.model_dump())
