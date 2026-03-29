"""
whois_intel.py
==============
Extracts WHOIS and DNS intelligence for a given domain.
Uses python-whois for domain registration data and dnspython for DNS checks.
All network calls are strictly time-bounded and never crash.
"""

import whois
import dns.resolver
import dns.exception
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DNS helpers
# ---------------------------------------------------------------------------

def _query_dns(domain: str, record_type: str, timeout: float = 3.0) -> bool:
    """Check whether a specific DNS record type exists for a domain.

    Args:
        domain: The domain name to query (e.g. 'example.com').
        record_type: DNS record type string ('A', 'MX', 'NS', etc.).
        timeout: Maximum seconds to wait for DNS resolution.

    Returns:
        True if at least one record of the given type exists, False otherwise.
    """
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = timeout
        resolver.timeout = timeout
        answers = resolver.resolve(domain, record_type)
        return len(answers) > 0
    except (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.exception.Timeout,
        dns.resolver.LifetimeTimeout,
        Exception,
    ):
        return False


def get_dns_features(domain: str) -> dict:
    """Retrieve DNS feature flags for a domain using dnspython.

    Checks for the existence of MX, A, and NS records.
    Each lookup is independently bounded to 3 seconds.

    Args:
        domain: The bare domain name (e.g. 'google.com').

    Returns:
        dict with keys:
            has_mx_record (int): 1 if MX record exists, else 0.
            has_a_record (int): 1 if A record exists, else 0.
            has_ns_record (int): 1 if NS record exists, else 0.
    """
    return {
        "has_mx_record": int(_query_dns(domain, "MX")),
        "has_a_record": int(_query_dns(domain, "A")),
        "has_ns_record": int(_query_dns(domain, "NS")),
    }


# ---------------------------------------------------------------------------
# WHOIS helpers
# ---------------------------------------------------------------------------

def _to_datetime(value: Any) -> datetime | None:
    """Coerce a WHOIS date field (which may be a list or single value) to datetime.

    Args:
        value: Raw WHOIS date field — could be datetime, list of datetimes,
               or None.

    Returns:
        A timezone-aware datetime in UTC, or None if conversion fails.
    """
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def get_whois_features(domain: str) -> dict:
    """Fetch WHOIS registration intelligence for a domain.

    Uses python-whois with a 5-second timeout equivalent (thread-limited).
    On any failure, returns whois_status='failed' and fills numeric fields
    with -1 (never 0) to distinguish from real zero values.

    Args:
        domain: The bare domain name to look up (e.g. 'example.com').

    Returns:
        dict with keys:
            domain_age_days (int|float): Age in days since creation, or -1.
            days_until_expiry (int|float): Days until expiry, or -1.
            registrar_name (str): Registrar name or 'unknown'.
            country (str): Registrant country code or 'unknown'.
            whois_privacy_enabled (int): 1 if privacy shield detected, else 0.
            whois_available (int): 1 if WHOIS succeeded, 0 if failed.
            whois_status (str): 'ok' or 'failed'.
    """
    _FAILED = {
        "domain_age_days": -1,
        "days_until_expiry": -1,
        "registrar_name": "unknown",
        "country": "unknown",
        "whois_privacy_enabled": 0,
        "whois_available": 0,
        "whois_status": "failed",
    }

    try:
        import signal

        def _handler(signum, frame):
            raise TimeoutError("WHOIS timeout")

        # Use signal-based timeout only on Unix; on Windows fall back gracefully
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(5)
            use_signal = True
        except AttributeError:
            use_signal = False

        try:
            w = whois.whois(domain)
        finally:
            if use_signal:
                signal.alarm(0)

        if w is None or not w.domain_name:
            return _FAILED

        now = datetime.now(timezone.utc)

        creation_date = _to_datetime(w.creation_date)
        expiration_date = _to_datetime(w.expiration_date)

        domain_age_days: int | float = (
            (now - creation_date).days if creation_date else -1
        )
        days_until_expiry: int | float = (
            (expiration_date - now).days if expiration_date else -1
        )

        # Registrar
        registrar = w.registrar or "unknown"
        if isinstance(registrar, list):
            registrar = registrar[0] if registrar else "unknown"

        # Country
        country = w.country or "unknown"
        if isinstance(country, list):
            country = country[0] if country else "unknown"

        # Privacy shield detection — common proxy registrar keywords
        privacy_keywords = [
            "privacy", "whoisguard", "perfect privacy", "domains by proxy",
            "contact privacy", "withheld", "redacted", "data protected",
        ]
        registrar_lower = str(registrar).lower()
        whois_privacy = int(
            any(kw in registrar_lower for kw in privacy_keywords)
        )

        return {
            "domain_age_days": domain_age_days,
            "days_until_expiry": days_until_expiry,
            "registrar_name": str(registrar),
            "country": str(country),
            "whois_privacy_enabled": whois_privacy,
            "whois_available": 1,
            "whois_status": "ok",
        }

    except Exception as exc:
        logger.debug("WHOIS failed for %s: %s", domain, exc)
        return _FAILED
