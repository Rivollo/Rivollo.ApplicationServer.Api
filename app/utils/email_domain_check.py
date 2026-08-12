"""Staged signup email verification: syntax -> allowlist -> disposable -> MX.

The standard input-validation order — syntactic validation, then semantic
policy, then network. Each stage after the first is only reached if the one
before it had nothing to say, so the common cases never touch DNS:

  Stage 1  SYNTAX      email-validator, pure CPU. Always first: every later
                       stage keys off the domain, and the domain is a
                       substructure of the address, so it cannot be trusted
                       until the address as a whole is known to parse.
                       Yields the ascii/punycode domain DNS can look up.
  Stage 2  ALLOWLIST   trusted domains (EMAIL_DOMAIN_ALLOWLIST) -> OK, no DNS.
  Stage 3  DISPOSABLE  set lookup against 8k known temp-mail domains, free.
  Stage 4  MX / DNS    the only stage that touches the network.

Stage 1 is deliberately the module's only parser. An allowlist probe ahead
of it would need a second, cheaper one, and two parsers obliged to agree
about RFC 5322 will drift — the earlier hand-rolled version already accepted
a 250-character local part that email-validator rejects.

Replaces the old substring keyword blacklist, which both let through any
temp-mail domain not in its 47-keyword list (97.6% of the real blocklist,
including wildcards like xyz.mailinator.com) and falsely rejected real
mail-accepting domains that merely *contained* a keyword (discard.io).

Rules, in order of importance:

  1. Cache by domain, not by address — one DNS round-trip serves every
     signup from that domain.
  2. Hard timeout on every DNS call. `timeout` is the per-nameserver
     attempt; `lifetime` is the total budget across retries and is the one
     that actually caps request latency.
  3. Fail OPEN on infrastructure errors (timeout, SERVFAIL, no reachable
     nameserver). A bad DNS minute must not reject real signups.
  4. Fail CLOSED only on definitive answers (bad syntax, NXDOMAIN, null MX,
     no mail route at all, known disposable).

We call email-validator with check_deliverability=False and run our own MX
lookup instead of letting it do the DNS. Its own deliverability check raises
the same EmailNotValidError for a resolver timeout as for a nonexistent
domain, which would collapse stages 1 and 4 into a single fail-CLOSED
answer — exactly the bad-DNS-minute-rejects-real-users case.

Async only, deliberately. dnspython's sync resolver blocks the event loop
for *every* concurrent request, not just the one doing the lookup, so a
sync variant in an all-async codebase is a footgun waiting to be called.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import dns.asyncresolver
import dns.exception
import dns.name
import dns.resolver
from disposable_email_domains import blocklist
from email_validator import EmailNotValidError, validate_email

from app.core.config import settings

logger = logging.getLogger(__name__)

# Built at import, but NEVER allowed to raise at import. Resolver() reads
# /etc/resolv.conf and raises NoResolverConfiguration if the file is missing,
# unreadable, or lists no nameserver — and because app.main imports the auth
# router, which imports this module, that exception would stop the whole API
# from starting rather than just disabling the email check. Degrade instead:
# _resolver = None makes stage 4 return UNKNOWN, which fails open per rule 3.
try:
    _resolver: Optional[dns.asyncresolver.Resolver] = dns.asyncresolver.Resolver()
    _resolver.timeout = settings.EMAIL_DNS_TIMEOUT_SECONDS
    _resolver.lifetime = settings.EMAIL_DNS_LIFETIME_SECONDS
except Exception:
    logger.error(
        "No usable DNS resolver configuration — signup MX checks are DISABLED "
        "for this process. Syntax, allowlist and disposable checks still apply.",
        exc_info=True,
    )
    _resolver = None

# Trusted domains — stage 2, short-circuited to allowed before the blocklist
# and before any DNS. Configured via EMAIL_DOMAIN_ALLOWLIST so enterprise or
# customer domains can be added without a deploy. Guards against a bad
# upstream blocklist release taking out signups from a major provider.
ALLOWLIST: frozenset[str] = settings.get_email_domain_allowlist()

# Providers that treat +tags as routing to the same mailbox.
PLUS_TAG_PROVIDERS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com",
    "live.com", "proton.me", "protonmail.com", "fastmail.com",
})


class Verdict(str, Enum):
    OK = "ok"
    INVALID_SYNTAX = "invalid_syntax"  # not a well-formed address at all
    NO_MX = "no_mx"          # domain resolves but takes no mail
    NULL_MX = "null_mx"      # domain explicitly refuses mail (RFC 7505)
    NX_DOMAIN = "nx_domain"  # domain does not exist
    DISPOSABLE = "disposable"
    UNKNOWN = "unknown"      # lookup failed on our side — let them through


BLOCKING = frozenset({
    Verdict.INVALID_SYNTAX, Verdict.NO_MX, Verdict.NULL_MX,
    Verdict.NX_DOMAIN, Verdict.DISPOSABLE,
})

MESSAGES = {
    # Distinct from NX_DOMAIN on purpose: a bad local part is not a domain
    # typo, and telling the user to check after the @ sends them hunting in
    # the wrong half of their address.
    Verdict.INVALID_SYNTAX: "That does not look like a valid email address.",
    Verdict.NX_DOMAIN: "That domain does not exist. Check the spelling after the @.",
    Verdict.NO_MX: "That domain cannot receive email. Use an address you can open.",
    Verdict.NULL_MX: "That domain does not accept email. Use an address you can open.",
    Verdict.DISPOSABLE: (
        "Temporary email addresses are not accepted. Use a permanent address."
    ),
}


@dataclass(frozen=True)
class CheckResult:
    domain: Optional[str]
    verdict: Verdict
    allowed: bool
    message: Optional[str] = None


def normalize_email(email: str) -> Optional[str]:
    """Collapse provider-equivalent aliases to a single canonical address.

    email-validator's own `.normalized` does NOT do this — it lowercases the
    domain but leaves Gmail dots and +tags intact, so one mailbox still
    yields unlimited distinct signups.

    Returned value is only meaningful for dedupe/lookup. Never send mail to
    it — always use the address the user actually typed.
    """
    email = (email or "").strip().lower()
    if email.count("@") != 1:
        return None
    local, domain = email.split("@")
    if domain in PLUS_TAG_PROVIDERS:
        local = local.split("+")[0]
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}" if local else None


def is_disposable(domain: str) -> bool:
    """True if the domain (or any parent domain) is a known temp-mail service.

    Walks up the subdomain chain because many services hand out wildcards
    like xyz.mailinator.com, which a flat set lookup would miss.
    """
    domain = domain.strip().lower()
    if domain in ALLOWLIST:
        return False
    parts = domain.split(".")
    # Stop before the bare TLD — ".com" must never match.
    for i in range(len(parts) - 1):
        if ".".join(parts[i:]) in blocklist:
            return True
    return False


def _interpret_mx(answers) -> Verdict:
    records = list(answers)
    if not records:
        return Verdict.NO_MX
    # RFC 7505 null MX: exactly one record whose exchange is the DNS root.
    if len(records) == 1 and records[0].exchange == dns.name.root:
        return Verdict.NULL_MX
    return Verdict.OK


async def _check_mx(domain: str) -> Verdict:
    if _resolver is None:
        # No resolver configuration in this process; see the import above.
        return Verdict.UNKNOWN

    try:
        return _interpret_mx(await _resolver.resolve(domain, "MX"))

    except dns.resolver.NXDOMAIN:
        return Verdict.NX_DOMAIN

    except dns.resolver.NoAnswer:
        # No MX record. RFC 5321 falls back to the A record, and plenty of
        # small business domains rely on that — blocking them costs real
        # customers.
        try:
            await _resolver.resolve(domain, "A")
            return Verdict.OK
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            return Verdict.NO_MX
        except dns.exception.DNSException:
            # A-record lookup broke on OUR side (timeout / no nameserver).
            # Returning NO_MX here would fail CLOSED on an infrastructure
            # error, which violates rule 3 — so fail open instead.
            return Verdict.UNKNOWN

    except dns.exception.DNSException:
        # Covers Timeout, NoNameservers (SERVFAIL), and everything else
        # that is our problem rather than the user's.
        return Verdict.UNKNOWN


# In-process, per-worker. A 4-worker gunicorn therefore does up to 4x the
# DNS traffic of a shared cache — acceptable while there is no Redis, and
# the TTLs below keep steady-state lookups near zero either way.
_cache: dict[str, tuple[Verdict, float]] = {}
TTL_OK = 30 * 24 * 3600
TTL_FAIL = 24 * 3600


def _cache_get(domain: str) -> Optional[Verdict]:
    hit = _cache.get(domain)
    if not hit:
        return None
    verdict, stored_at = hit
    ttl = TTL_OK if verdict is Verdict.OK else TTL_FAIL
    if time.time() - stored_at > ttl:
        _cache.pop(domain, None)
        return None
    return verdict


def _result(domain: Optional[str], verdict: Verdict) -> CheckResult:
    return CheckResult(
        domain=domain,
        verdict=verdict,
        allowed=verdict not in BLOCKING,
        message=MESSAGES.get(verdict),
    )


def _parse_domain(email: str) -> Optional[str]:
    """Validate address syntax and return its ascii (punycode) domain.

    check_deliverability=False keeps this pure CPU — no DNS, no event-loop
    blocking, and no conflating a resolver timeout with a bad address. The
    MX question is stage 4's job.

    Returns the ascii domain so IDNs survive: user@münchen.de yields
    xn--mnchen-3ya.de, which is what a resolver can actually look up. The
    old regex rejected such addresses outright.
    """
    try:
        info = validate_email(email, check_deliverability=False)
    except EmailNotValidError:
        return None
    return info.ascii_domain.lower() if info.ascii_domain else None


async def check_email(email: str) -> CheckResult:
    """Verify an address is well-formed, non-disposable, and takes mail.

    Only for the SIGNUP path. On login the address is already in the
    database, so a DNS query there is pure latency.
    """
    email = (email or "").strip().lower()

    # --- Stage 1: syntax (email-validator, pure CPU, no network) ---------
    # Always first. The later stages key off the domain, and the domain is a
    # substructure of the address — reading it out to make a trust decision
    # before confirming the address parses means trusting an unvalidated
    # field. This is also the only parser in the module: an allowlist probe
    # ahead of it would need its own, and two parsers that must agree drift.
    domain = _parse_domain(email)
    if domain is None:
        return _result(None, Verdict.INVALID_SYNTAX)

    # --- Stage 2: allowlist (no lookup at all) --------------------------
    # Trusted domains exit here, before the blocklist and before any DNS.
    # Runs on the ascii/punycode domain from stage 1, so unicode IDNs match
    # their allowlist entry too.
    if domain in ALLOWLIST:
        return _result(domain, Verdict.OK)

    cached = _cache_get(domain)
    if cached is not None:
        return _result(domain, cached)

    # --- Stage 3: disposable blocklist (free set lookup) ----------------
    if is_disposable(domain):
        _cache[domain] = (Verdict.DISPOSABLE, time.time())
        return _result(domain, Verdict.DISPOSABLE)

    # --- Stage 4: MX / DNS (the only stage that hits the network) -------
    verdict = await _check_mx(domain)
    if verdict is Verdict.UNKNOWN:
        # Don't cache — retry on the next attempt rather than pinning a
        # transient failure in place for TTL_FAIL.
        logger.warning("Email domain check inconclusive for %s — allowing", domain)
    else:
        _cache[domain] = (verdict, time.time())
    return _result(domain, verdict)
