"""Country and currency resolution for billing.

Cloudflare sits in front of every deployable and sets ``CF-IPCountry`` on each
request. Cloudflare overwrites any client-supplied value for that header, so it
cannot be spoofed by a browser — but only on the hop that actually passes
through Cloudflare.

The marketing site renders its pricing page server-side, so its call to this API
is server-to-server and carries the container's own egress IP. It forwards the
visitor's real country as ``x-rvl-country`` instead. That forwarded header is
attacker-controllable, so the two resolvers here are deliberately different:

    resolve_display_country  — accepts the forwarded header. Worst case of a
                               forged value is a visitor seeing the wrong price
                               on a page; no money moves.
    resolve_checkout_country — CF-IPCountry only. This is the one that decides
                               whether a customer is allowed onto the USD rails,
                               and RBI requires domestic transactions in INR, so
                               it never trusts a value a caller could set.
"""

from typing import Optional

from fastapi import Request

INR = "INR"
USD = "USD"

INDIA = "IN"

# Cloudflare uses these for requests it cannot geolocate (XX) and for Tor
# traffic (T1). Both fall through to the USD path along with every other
# non-India country code.
_UNKNOWN_COUNTRY_CODES = {"XX", "T1"}


def _normalise(raw: Optional[str]) -> Optional[str]:
    """Upper-case a header value, treating blanks and unknown markers as absent."""
    if not raw:
        return None
    code = raw.strip().upper()
    if len(code) != 2 or code in _UNKNOWN_COUNTRY_CODES:
        return None
    return code


def resolve_display_country(request: Request) -> Optional[str]:
    """Resolve the country for *displaying* prices.

    Prefers Cloudflare's own header; falls back to the marketing site's
    forwarded ``x-rvl-country``. Returns None when neither yields a usable code.
    """
    return _normalise(request.headers.get("cf-ipcountry")) or _normalise(
        request.headers.get("x-rvl-country")
    )


def resolve_checkout_country(request: Request) -> Optional[str]:
    """Resolve the country for *charging* a customer.

    Cloudflare's header only — a forwarded header is never accepted here.
    """
    return _normalise(request.headers.get("cf-ipcountry"))


def currency_for_country(country: Optional[str]) -> str:
    """Map a country code to the currency it is billed in.

    India bills INR. Everything else — including an absent or unresolvable
    country — bills USD.
    """
    return INR if country == INDIA else USD


def is_india(country: Optional[str]) -> bool:
    return country == INDIA
