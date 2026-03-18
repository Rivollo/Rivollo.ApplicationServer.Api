"""Authentication schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, model_validator, field_validator

# Blocked disposable / temporary email domains (blacklist approach).
# All other domains — including business/company emails — are allowed.
BLOCKED_EMAIL_DOMAINS: frozenset[str] = frozenset({
    # Common disposable / temp-mail services
    "mailinator.com",
    "guerrillamail.com",
    "guerrillamail.net",
    "guerrillamail.org",
    "guerrillamail.biz",
    "guerrillamail.de",
    "guerrillamail.info",
    "trashmail.com",
    "trashmail.me",
    "trashmail.net",
    "trashmail.at",
    "trashmail.io",
    "trashmail.xyz",
    "tempmail.com",
    "temp-mail.org",
    "temp-mail.io",
    "throwam.com",
    "throwam.net",
    "yopmail.com",
    "yopmail.fr",
    "cool.fr.nf",
    "jetable.fr.nf",
    "nospam.ze.tc",
    "nomail.xl.cx",
    "mega.zik.dj",
    "speed.1s.fr",
    "courriel.fr.nf",
    "moncourrier.fr.nf",
    "monemail.fr.nf",
    "monmail.fr.nf",
    "mailnull.com",
    "maildrop.cc",
    "sharklasers.com",
    "guerrillamailblock.com",
    "grr.la",
    "spam4.me",
    "dispostable.com",
    "discard.email",
    "spamgourmet.com",
    "spamgourmet.net",
    "spamgourmet.org",
    "spamhereplease.com",
    "spamhereplease.net",
    "spamhereplease.org",
    "fakeinbox.com",
    "mailnesia.com",
    "mailnesia.net",
    "mailnesia.org",
    "mailscrap.com",
    "disposableemailaddresses.com",
    "throwam.com",
    "throwam.net",
    "throwam.org",
    "spamobox.com",
    "filzmail.com",
    "gowikicampus.com",
    "gowikicars.com",
    "gowikifilms.com",
    "gowikigames.com",
    "gowikimusic.com",
    "gowikinetwork.com",
    "gowikitravel.com",
    "gowikitv.com",
    "mt2009.com",
    "mt2014.com",
    "mt2015.com",
    "spamfree24.org",
    "objectmail.com",
    "obobbo.com",
    "oneoffmail.com",
    "owlpic.com",
    "pjjkp.com",
    "plexolan.de",
    "poczta.onet.pl",
    "put2.net",
    "rcpt.at",
    "receivemail.org",
    "recursor.net",
    "regbypass.com",
    "regbypass.comsafe-mail.net",
    "rhyta.com",
    "rmqkr.net",
    "royal.net",
    "rtrtr.com",
    "s0ny.net",
    "safe-mail.net",
    "safersignup.de",
    "safetymail.info",
    "safetypost.de",
    "sandelf.de",
    "schafmail.de",
    "shiftmail.com",
    "sibmail.com",
    "sneakemail.com",
    "sofimail.com",
    "sogetthis.com",
    "soodonims.com",
    "spam.la",
    "spamavert.com",
    "spambox.info",
    "spambox.irishspringrealty.com",
    "spambox.us",
    "spamcannon.com",
    "spamcannon.net",
    "spamcon.org",
    "spamcorpse.com",
    "spamday.com",
    "spamex.com",
    "spamfree.eu",
    "spamgob.com",
    "spamhole.com",
    "spamify.com",
    "spaminator.de",
    "spamkill.info",
    "spaml.com",
    "spaml.de",
    "spammotel.com",
    "spamoff.de",
    "spamslicer.com",
    "spamspot.com",
    "spamthis.co.uk",
    "spamtroll.net",
    "speed.1s.fr",
    "supergreatmail.com",
    "supermailer.jp",
    "superrito.com",
    "superstachel.de",
    "suremail.info",
    "svkmail.com",
    "sweetxxx.de",
    "tafmail.com",
    "tagyourself.com",
    "talkinator.com",
    "teewars.org",
    "teleworm.com",
    "teleworm.us",
    "tempalias.com",
    "tempe-mail.com",
    "tempemail.biz",
    "tempemail.com",
    "tempemail.net",
    "tempinbox.co.uk",
    "tempinbox.com",
    "tempmail.eu",
    "tempmailer.com",
    "tempmailer.de",
    "temporarily.de",
    "temporaryemail.net",
    "temporaryforwarding.com",
    "temporaryinbox.com",
    "temporarymailaddress.com",
    "tempthe.net",
    "thanksnospam.info",
    "thisisnotmyrealemail.com",
    "throam.com",
    "throwam.com",
    "throwam.net",
    "throwam.us",
    "throwaway.email",
    "tilien.com",
    "tittbit.in",
    "tizi.com",
    "tm.in.ua",
    "tmailinator.com",
    "toiea.com",
    "tradermail.info",
    "trash-mail.at",
    "trash-mail.com",
    "trash-mail.de",
    "trash-mail.ga",
    "trash-mail.io",
    "trash-mail.net",
    "trash-mail.tk",
    "trash2009.com",
    "trashdevil.com",
    "trashdevil.de",
    "trashemail.de",
    "trashmail.at",
    "trashmail.com",
    "trashmail.io",
    "trashmail.me",
    "trashmail.net",
    "trashmail.org",
    "trashmail.xyz",
    "trashmailer.com",
    "trashspam.com",
    "trbvm.com",
    "twinmail.de",
    "tyldd.com",
    "uggsrock.com",
    "umail.net",
    "upliftnow.com",
    "uplipht.com",
    "uroid.com",
    "us.af",
    "venompen.com",
    "veryrealemail.com",
    "viditag.com",
    "viewcastmedia.com",
    "viewcastmedia.net",
    "viewcastmedia.org",
    "vomoto.com",
    "vubby.com",
    "walala.org",
    "walkmail.net",
    "walkmail.ru",
    "wh4f.org",
    "whyspam.me",
    "wilemail.com",
    "willhackforfood.biz",
    "willselfdestruct.com",
    "wmail.cf",
    "wolfsmail.tk",
    "wrestlingpages.com",
    "wudet.men",
    "wuzupmail.net",
    "wwwnew.eu",
    "xagloo.co",
    "xagloo.com",
    "xemaps.com",
    "xents.com",
    "xmaily.com",
    "xoxy.net",
    "xyzfree.net",
    "yapped.net",
    "yeah.net",
    "yep.it",
    "yogamaven.com",
    "yopmail.com",
    "yopmail.fr",
    "yopmail.pp.ua",
    "youmail.ga",
    "yourdomain.com",
    "yuurok.com",
    "z1p.biz",
    "za.com",
    "zebins.com",
    "zebins.eu",
    "zehnminuten.de",
    "zehnminutenmail.de",
    "zetmail.com",
    "zippymail.info",
    "zoemail.net",
    "zoemail.org",
    "zomg.info",
    "zxcv.com",
    "zxcvbnm.com",
    "zzrgg.com",
})


def is_valid_email_domain(email: str) -> bool:
    """
    Returns True if the email domain is NOT on the blacklist.
    Blocks known disposable / temporary email providers; allows all others,
    including business / company domains.
    """
    try:
        domain = email.strip().lower().split("@")[1]
        return domain not in BLOCKED_EMAIL_DOMAINS
    except IndexError:
        return False


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False


class SignupRequest(BaseModel):
    """Signup request with email, password, and optional name."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    remember_me: bool = False

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


class GoogleAuthRequest(BaseModel):
    """Google OAuth authentication request."""

    credential: str = Field(..., min_length=1, description="Google OAuth credential token")
    remember_me: bool = False


class UserResponse(BaseModel):
    """User response model."""

    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """Authentication response with user and token."""

    user: UserResponse
    token: str
    expires_at: Optional[datetime] = None


class UserUpdateRequest(BaseModel):
    """Request to update user profile."""

    name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    """Request to initiate a password reset."""

    email: EmailStr


class VerifyOTPRequest(BaseModel):
    """Request to verify the OTP sent to the user's email."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class ResetPasswordRequest(BaseModel):
    """Request to complete a password reset using a verified token."""

    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    @model_validator(mode="after")
    def passwords_match(self) -> "ResetPasswordRequest":
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

