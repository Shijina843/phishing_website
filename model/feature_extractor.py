"""
feature_extractor.py
====================
Assembles the full feature vector for a URL by combining:
  - URL-based structural features (Layer 1)
  - Tranco whitelist check (Layer 2)
  - WHOIS / RDAP domain intelligence (Layers 3 & 4)
  - DNS record checks (Layer 5)

The resulting dict is passed to the XGBoost model for classification.
"""

import math
import os
import re
import urllib.parse
import logging
from pathlib import Path
from typing import Any

import tldextract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "update", "bank",
    "confirm", "account", "signin", "password", "webscr",
    "paypal", "ebay", "amazon", "microsoft", "apple",
]

_SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".online", ".site", ".info", ".biz", ".link", ".click", ".pw",
    ".win", ".loan", ".download", ".gdn",
}

# Tranco whitelist: cached in memory after first load
_WHITELIST: set[str] | None = None
_WHITELIST_LOADED = False

# Path is relative to the project root — can be overridden by env var
_TRANCO_CSV_PATH = os.environ.get(
    "TRANCO_CSV_PATH",
    str(Path(__file__).resolve().parent.parent / "dataset" / "tranco_top1m.csv"),
)


# ---------------------------------------------------------------------------
# Whitelist loading
# ---------------------------------------------------------------------------

def _load_whitelist() -> set[str]:
    """Load the Tranco Top-1M domain whitelist from a local CSV file.

    The CSV is expected to have the domain in its second column (index 1),
    following the standard Tranco format: rank,domain.
    If the file is absent or unreadable the function returns an empty set
    and logs a notice rather than raising an exception.

    Returns:
        A set of lowercase domain strings, possibly empty.
    """
    global _WHITELIST, _WHITELIST_LOADED
    if _WHITELIST_LOADED:
        return _WHITELIST or set()

    _WHITELIST_LOADED = True
    path = Path(_TRANCO_CSV_PATH)
    if not path.exists():
        logger.info(
            "Tranco whitelist not found at %s — whitelist check disabled.", path
        )
        _WHITELIST = set()
        return _WHITELIST

    try:
        domains: set[str] = set()
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    domains.add(parts[1].strip().lower())
        _WHITELIST = domains
        logger.info("Loaded %d domains into whitelist from %s", len(domains), path)
    except Exception as exc:
        logger.warning("Failed to load whitelist from %s: %s", path, exc)
        _WHITELIST = set()

    return _WHITELIST


def is_whitelisted(domain: str) -> bool:
    """Check whether a domain is in the Tranco whitelist.

    Args:
        domain: The registered domain to check (e.g. 'google.com').

    Returns:
        True if the domain is on the Tranco list, False otherwise.
    """
    whitelist = _load_whitelist()
    return domain.lower() in whitelist


# ---------------------------------------------------------------------------
# URL normalization & parsing
# ---------------------------------------------------------------------------

def normalize_url(raw_url: str) -> str:
    """Normalize a raw URL to a consistent form for feature extraction.

    Prepends 'http://' if no scheme is present, and lowercases the domain
    portion while preserving the path case.

    Args:
        raw_url: The URL string as entered by the user.

    Returns:
        A normalized URL string.
    """
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "http://" + raw_url
    parsed = urllib.parse.urlparse(raw_url)
    normalized = parsed._replace(netloc=parsed.netloc.lower())
    return urllib.parse.urlunparse(normalized)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy of a string.

    Higher entropy indicates more randomness, which is typical of
    algorithmically generated domain names used in phishing.

    Args:
        text: Input string to compute entropy for.

    Returns:
        Shannon entropy as a float (bits per character), or 0.0 for empty string.
    """
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# IP detection
# ---------------------------------------------------------------------------

_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)


def _is_ip_address(hostname: str) -> bool:
    """Determine whether a hostname is an IPv4 address.

    Args:
        hostname: The hostname portion of the URL (without port).

    Returns:
        True if the hostname looks like an IPv4 address.
    """
    return bool(_IPV4_RE.match(hostname.split(":")[0]))


# ---------------------------------------------------------------------------
# URL feature extraction — Layer 1
# ---------------------------------------------------------------------------

def extract_url_features(url: str) -> dict:
    """Extract structural and lexical features from a URL string.

    Parses the URL and computes a comprehensive set of numeric and boolean
    features used by the ML model.

    Args:
        url: A normalized URL string (must include scheme).

    Returns:
        dict of URL-derived features:
            url_length, domain_length, num_dots, num_hyphens, num_digits,
            has_ip, has_at_symbol, has_double_slash, subdomain_depth,
            suspicious_keywords, digit_ratio, special_char_count,
            tld_suspicious, https_used, url_entropy.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full_url = url

    # tldextract breakdown
    ext = tldextract.extract(url)
    subdomain = ext.subdomain or ""
    domain = ext.domain or hostname
    tld_part = ("." + ext.suffix) if ext.suffix else ""

    # Basic counts
    url_length = len(full_url)
    domain_length = len(domain)
    num_dots = full_url.count(".")
    num_hyphens = full_url.count("-")
    num_digits = sum(ch.isdigit() for ch in full_url)
    digit_ratio = num_digits / url_length if url_length > 0 else 0.0

    # Boolean flags
    has_ip = int(_is_ip_address(hostname))
    has_at_symbol = int("@" in full_url)
    has_double_slash = int("//" in parsed.path)
    https_used = int(parsed.scheme == "https")

    # Subdomain depth: count components separated by '.'
    subdomain_depth = len(subdomain.split(".")) if subdomain else 0

    # Suspicious keyword presence in URL
    url_lower = full_url.lower()
    suspicious_keywords = int(
        any(kw in url_lower for kw in _SUSPICIOUS_KEYWORDS)
    )

    # Special character count in full URL (non-alphanumeric, non-URL-structural)
    special_chars = re.sub(r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]", "", full_url)
    special_char_count = len(special_chars)

    # Suspicious TLD
    tld_suspicious = int(tld_part.lower() in _SUSPICIOUS_TLDS)

    # Shannon entropy of the full URL
    url_entropy = _shannon_entropy(full_url)

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "has_double_slash": has_double_slash,
        "subdomain_depth": subdomain_depth,
        "suspicious_keywords": suspicious_keywords,
        "digit_ratio": digit_ratio,
        "special_char_count": special_char_count,
        "tld_suspicious": tld_suspicious,
        "https_used": https_used,
        "url_entropy": url_entropy,
    }


# ---------------------------------------------------------------------------
# Full pipeline — Layer 6
# ---------------------------------------------------------------------------

def extract_features(
    raw_url: str,
    enable_network: bool = True,
) -> dict:
    """Run the full feature extraction pipeline for a single URL.

    Steps:
      1. Normalize URL.
      2. Check Tranco whitelist — short-circuit if whitelisted.
      3. Extract URL structural features.
      4. Fetch RDAP/WHOIS domain intelligence (with WHOIS fallback).
      5. Fetch DNS record flags.
      6. Merge everything into one flat feature dict.

    Args:
        raw_url: Raw URL string from the user or dataset.
        enable_network: If False, skips RDAP/WHOIS/DNS calls (used in testing).

    Returns:
        dict with keys:
            * All URL features from extract_url_features().
            * domain_age_days, days_until_expiry, registrar_name, country,
              whois_privacy_enabled, whois_available, data_source.
            * has_mx_record, has_a_record, has_ns_record.
            * _whitelisted (bool): True if this URL was whitelisted.
            * _domain (str): The registered domain used for lookups.
    """
    url = normalize_url(raw_url)
    ext = tldextract.extract(url)
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    # ---- Layer 2: Whitelist check ----
    if is_whitelisted(registered_domain) or is_whitelisted(ext.domain):
        url_feats = extract_url_features(url)
        # Return a clean feature dict with whitelist flag set
        features = {
            **url_feats,
            "domain_age_days": -1,
            "days_until_expiry": -1,
            "registrar_name": "unknown",
            "country": "unknown",
            "whois_privacy_enabled": 0,
            "whois_available": 0,
            "data_source": "whitelist",
            "has_mx_record": 1,
            "has_a_record": 1,
            "has_ns_record": 1,
            "_whitelisted": True,
            "_domain": registered_domain,
        }
        return features

    # ---- Layer 1: URL features ----
    url_feats = extract_url_features(url)

    # ---- Layers 3/4: RDAP + WHOIS fallback ----
    if enable_network:
        from rdap_intel import get_domain_intelligence
        from whois_intel import get_dns_features

        domain_intel = get_domain_intelligence(registered_domain)
        dns_feats = get_dns_features(registered_domain)
    else:
        domain_intel = {
            "domain_age_days": -1,
            "days_until_expiry": -1,
            "registrar_name": "unknown",
            "country": "unknown",
            "whois_privacy_enabled": 0,
            "whois_available": 0,
            "whois_status": "failed",
            "data_source": "unavailable",
        }
        dns_feats = {
            "has_mx_record": 0,
            "has_a_record": 0,
            "has_ns_record": 0,
        }

    features: dict[str, Any] = {
        **url_feats,
        **domain_intel,
        **dns_feats,
        "_whitelisted": False,
        "_domain": registered_domain,
    }
    return features


# ---------------------------------------------------------------------------
# Feature columns used by the trained model (in order)
# ---------------------------------------------------------------------------

MODEL_FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "has_ip",
    "has_at_symbol",
    "has_double_slash",
    "subdomain_depth",
    "suspicious_keywords",
    "digit_ratio",
    "special_char_count",
    "tld_suspicious",
    "https_used",
    "url_entropy",
    "domain_age_days",
    "days_until_expiry",
    "whois_privacy_enabled",
    "whois_available",
    "has_mx_record",
    "has_a_record",
    "has_ns_record",
]

# Boolean / categorical columns that must NOT be scaled
BOOLEAN_COLUMNS = {
    "has_ip", "has_at_symbol", "has_double_slash", "suspicious_keywords",
    "tld_suspicious", "https_used", "whois_privacy_enabled", "whois_available",
    "has_mx_record", "has_a_record", "has_ns_record",
}

# Continuous columns that ARE scaled by StandardScaler
CONTINUOUS_COLUMNS = [c for c in MODEL_FEATURE_COLUMNS if c not in BOOLEAN_COLUMNS]
