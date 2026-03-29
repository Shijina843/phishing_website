"""
rdap_intel.py
=============
Fetches domain intelligence using the RDAP protocol via direct HTTP calls
through httpx. Falls back to WHOIS if RDAP is unavailable.
No third-party paid APIs are used.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from whois_intel import get_whois_features

logger = logging.getLogger(__name__)

_RDAP_BASE = "https://rdap.org/domain/{domain}"
_TIMEOUT = 5.0  # seconds, strict upper bound for RDAP calls

# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_rdap_date(value: Any) -> datetime | None:
    """Parse an RDAP date string into a timezone-aware UTC datetime.

    RDAP dates are typically ISO-8601 strings such as '2023-04-01T00:00:00Z'.

    Args:
        value: Raw value from RDAP JSON — expected to be a string or None.

    Returns:
        timezone-aware datetime in UTC, or None on failure.
    """
    if not value or not isinstance(value, str):
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# RDAP extraction helpers
# ---------------------------------------------------------------------------

def _extract_event_date(events: list, event_action: str) -> datetime | None:
    """Look up a specific event date from an RDAP events list.

    Args:
        events: List of RDAP event dicts, each having 'eventAction' and
                'eventDate' keys.
        event_action: The event action label to search for
                      (e.g. 'registration', 'expiration').

    Returns:
        Parsed datetime or None if not found.
    """
    for event in events or []:
        if isinstance(event, dict) and event.get("eventAction") == event_action:
            return _parse_rdap_date(event.get("eventDate"))
    return None


def _extract_registrar(rdap_data: dict) -> str:
    """Extract the registrar name from RDAP JSON data.

    Args:
        rdap_data: Parsed RDAP response dict.

    Returns:
        Registrar name string or 'unknown'.
    """
    entities = rdap_data.get("entities", [])
    for entity in entities:
        roles = entity.get("roles", [])
        if "registrar" in roles:
            vcard = entity.get("vcardArray", [])
            if isinstance(vcard, list) and len(vcard) > 1:
                for prop in vcard[1]:
                    if isinstance(prop, list) and prop[0] == "fn":
                        name = prop[-1]
                        if name:
                            return str(name)
            name = entity.get("handle") or entity.get("ldhName")
            if name:
                return str(name)
    return "unknown"


def _extract_country(rdap_data: dict) -> str:
    """Extract the registrant country from RDAP JSON data.

    Args:
        rdap_data: Parsed RDAP response dict.

    Returns:
        Two-letter country code or 'unknown'.
    """
    entities = rdap_data.get("entities", [])
    for entity in entities:
        roles = entity.get("roles", [])
        if "registrar" in roles or "registrant" in roles:
            vcard = entity.get("vcardArray", [])
            if isinstance(vcard, list) and len(vcard) > 1:
                for prop in vcard[1]:
                    if isinstance(prop, list) and prop[0] == "adr":
                        # adr value is a list; country is typically the last element
                        adr_val = prop[-1]
                        if isinstance(adr_val, list) and len(adr_val) >= 7:
                            country = adr_val[6]
                            if country:
                                return str(country)
    return "unknown"


# ---------------------------------------------------------------------------
# Main RDAP fetch
# ---------------------------------------------------------------------------

def _fetch_rdap(domain: str) -> dict | None:
    """Fetch raw RDAP JSON for a domain.

    Args:
        domain: Bare domain name (e.g. 'google.com').

    Returns:
        Parsed JSON dict on success, None on any error.
    """
    url = _RDAP_BASE.format(domain=domain)
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
    except Exception as exc:
        logger.debug("RDAP fetch failed for %s: %s", domain, exc)
    return None


def _parse_rdap_data(rdap_data: dict) -> dict:
    """Parse a successful RDAP response into a normalized feature dict.

    Args:
        rdap_data: Parsed RDAP JSON response.

    Returns:
        Feature dict with domain_age_days, days_until_expiry, registrar_name,
        country, whois_privacy_enabled, whois_available, whois_status,
        and data_source='rdap'.
    """
    now = datetime.now(timezone.utc)
    events = rdap_data.get("events", [])

    creation_date = _extract_event_date(events, "registration")
    expiration_date = _extract_event_date(events, "expiration")

    domain_age_days: int | float = (
        (now - creation_date).days if creation_date else -1
    )
    days_until_expiry: int | float = (
        (expiration_date - now).days if expiration_date else -1
    )

    registrar = _extract_registrar(rdap_data)
    country = _extract_country(rdap_data)

    # Privacy detection via remarks or redacted notices
    remarks = rdap_data.get("remarks", [])
    notices = rdap_data.get("notices", [])
    privacy_keywords = ["redacted", "privacy", "withheld", "protected"]
    privacy_text = " ".join(
        str(r.get("description", "")).lower()
        for r in (remarks + notices)
        if isinstance(r, dict)
    )
    whois_privacy = int(any(kw in privacy_text for kw in privacy_keywords))

    return {
        "domain_age_days": domain_age_days,
        "days_until_expiry": days_until_expiry,
        "registrar_name": registrar,
        "country": country,
        "whois_privacy_enabled": whois_privacy,
        "whois_available": 1,
        "whois_status": "ok",
        "data_source": "rdap",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_domain_intelligence(domain: str) -> dict:
    """Fetch domain intelligence with RDAP → WHOIS fallback chain.

    Attempts:
      1. RDAP via direct HTTP (httpx) — 5 second timeout.
      2. WHOIS via python-whois on RDAP failure.
      3. Returns all -1 / unavailable if both fail.

    Args:
        domain: Bare domain name to inspect (e.g. 'example.com').

    Returns:
        dict containing:
            domain_age_days (int|float): Days since creation, or -1.
            days_until_expiry (int|float): Days until expiry, or -1.
            registrar_name (str): Registrar name or 'unknown'.
            country (str): Country code or 'unknown'.
            whois_privacy_enabled (int): 1 if privacy detected, else 0.
            whois_available (int): 1 on success, 0 on failure.
            whois_status (str): 'ok' or 'failed'.
            data_source (str): 'rdap', 'whois', or 'unavailable'.
    """
    _UNAVAILABLE = {
        "domain_age_days": -1,
        "days_until_expiry": -1,
        "registrar_name": "unknown",
        "country": "unknown",
        "whois_privacy_enabled": 0,
        "whois_available": 0,
        "whois_status": "failed",
        "data_source": "unavailable",
    }

    # --- Try RDAP first ---
    try:
        rdap_data = _fetch_rdap(domain)
        if rdap_data:
            result = _parse_rdap_data(rdap_data)
            logger.debug("RDAP succeeded for %s", domain)
            return result
    except Exception as exc:
        logger.debug("RDAP parse error for %s: %s", domain, exc)

    # --- Fallback to WHOIS ---
    try:
        whois_result = get_whois_features(domain)
        if whois_result.get("whois_available") == 1:
            whois_result["data_source"] = "whois"
            logger.debug("WHOIS fallback succeeded for %s", domain)
            return whois_result
    except Exception as exc:
        logger.debug("WHOIS fallback error for %s: %s", domain, exc)

    # --- Both failed ---
    logger.debug("All intelligence sources failed for %s", domain)
    return _UNAVAILABLE
