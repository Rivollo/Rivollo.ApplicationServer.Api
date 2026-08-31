"""Request/response schemas for the email OTP login endpoints.

Separate from app/schemas/auth.py so that file — which defines the request and
response models of signup, password login, Google login and password reset —
stays out of this feature's change set. AuthResponse is imported from it for
the verify response, so an OTP login and a password login return exactly the
same body.
"""

from pydantic import BaseModel, EmailStr, Field

from app.schemas.auth import AuthResponse  # noqa: F401  (re-exported for routes)


class OtpLoginRequest(BaseModel):
    """Ask for a sign-in code, or resend the current one.

    Deliberately NOT gated on the disposable-domain check that
    SendSignupOtpRequest applies. This is an existing-user path: the address is
    already registered, and rejecting it here would lock a real user out of
    their own account if their domain later landed on the blocklist. The same
    reasoning app/schemas/auth.py records for LoginRequest.
    """

    email: EmailStr


class OtpLoginResponse(BaseModel):
    """Answer to a code request.

    Identical in shape and content for a registered address, an unregistered
    one, a deactivated account and an account pending deletion — the caller
    cannot tell them apart.
    """

    message: str
    expires_in_minutes: int
    # How many resends are left in this series. Lets the client render
    # "2 resends left" without a second endpoint.
    resends_remaining: int
    # Seconds until another request will actually send. Drives the client's
    # resend countdown.
    resend_available_in_seconds: int


class OtpLoginVerifyRequest(BaseModel):
    """Submit a code and receive a session.

    remember_me lives here rather than on the request endpoint because it is a
    property of the session being created, and the session is created here —
    matching LoginRequest and GoogleAuthRequest, where the flag travels with
    the call that mints the token.
    """

    email: EmailStr
    # Same constraints as the existing VerifyOTPRequest, so a client that can
    # drive the password-reset OTP field can drive this one.
    otp: str = Field(..., min_length=6, max_length=6)
    remember_me: bool = False
