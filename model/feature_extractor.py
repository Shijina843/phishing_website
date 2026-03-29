"""
feature_extractor.py
====================
Assembles the 15 fast URL-lexical features corresponding exactly to
the pre-extracted columns in dataset/phishing_data.csv.
Bypasses slow network operations like WHOIS/DNS to accelerate everything.
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

# Constants
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

# Tranco whitelist
_WHITELIST: set[str] | None = None
_WHITELIST_LOADED = False

_TRANCO_CSV_PATH = os.environ.get(
    "TRANCO_CSV_PATH",
    str(Path(__file__).resolve().parent.parent / "dataset" / "tranco_top1m.csv"),
)

def _load_whitelist() -> set[str]:
    global _WHITELIST, _WHITELIST_LOADED
    if _WHITELIST_LOADED:
        return _WHITELIST or set()

    _WHITELIST_LOADED = True
    path = Path(_TRANCO_CSV_PATH)
    if not path.exists():
        logger.info("Tranco whitelist not found at %s", path)
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
        logger.info("Loaded %d domains into whitelist", len(domains))
    except Exception as exc:
        logger.warning("Failed to load whitelist: %s", exc)
        _WHITELIST = set()

    return _WHITELIST


def is_whitelisted(domain: str) -> bool:
    whitelist = _load_whitelist()
    return domain.lower() in whitelist


def normalize_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "http://" + raw_url
    parsed = urllib.parse.urlparse(raw_url)
    normalized = parsed._replace(netloc=parsed.netloc.lower())
    return urllib.parse.urlunparse(normalized)


_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

def _is_ip_address(hostname: str) -> bool:
    return bool(_IPV4_RE.match(hostname.split(":")[0]))


def extract_features(raw_url: str, enable_network: bool = False) -> dict:
    """Extract 15 core URL features matching phishing_data.csv exactly."""
    url = normalize_url(raw_url)
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    
    ext = tldextract.extract(url)
    subdomain = ext.subdomain or ""
    domain = ext.domain or hostname
    tld_part = ("." + ext.suffix) if ext.suffix else ""
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    is_safe = is_whitelisted(registered_domain) or is_whitelisted(ext.domain)

    # 15 Fast Lexical Features matching CSV columns
    length_url = len(url)
    length_hostname = len(hostname)
    ip_feature = int(_is_ip_address(hostname))
    nb_dots = url.count(".")
    nb_hyphens = url.count("-")
    nb_at = url.count("@")
    nb_qm = url.count("?")
    nb_and = url.count("&")
    nb_eq = url.count("=")
    nb_slash = url.count("/")
    nb_www = url.count("www")
    
    num_digits_url = sum(ch.isdigit() for ch in url)
    ratio_digits_url = num_digits_url / length_url if length_url > 0 else 0.0
    
    num_digits_host = sum(ch.isdigit() for ch in hostname)
    ratio_digits_host = num_digits_host / length_hostname if length_hostname > 0 else 0.0
    
    url_lower = url.lower()
    phish_hints = sum(url_lower.count(kw) for kw in _SUSPICIOUS_KEYWORDS)
    suspecious_tld = int(tld_part.lower() in _SUSPICIOUS_TLDS)
    
    return {
        "length_url": length_url,
        "length_hostname": length_hostname,
        "ip": ip_feature,
        "nb_dots": nb_dots,
        "nb_hyphens": nb_hyphens,
        "nb_at": nb_at,
        "nb_qm": nb_qm,
        "nb_and": nb_and,
        "nb_eq": nb_eq,
        "nb_slash": nb_slash,
        "nb_www": nb_www,
        "ratio_digits_url": ratio_digits_url,
        "ratio_digits_host": ratio_digits_host,
        "phish_hints": phish_hints,
        "suspecious_tld": suspecious_tld,
        
        # Meta flags expected by app.py
        "_whitelisted": is_safe,
        "_domain": registered_domain,
    }


MODEL_FEATURE_COLUMNS = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_eq",
    "nb_slash",
    "nb_www",
    "ratio_digits_url",
    "ratio_digits_host",
    "phish_hints",
    "suspecious_tld",
]

BOOLEAN_COLUMNS = {
    "ip", "suspecious_tld"
}

CONTINUOUS_COLUMNS = [c for c in MODEL_FEATURE_COLUMNS if c not in BOOLEAN_COLUMNS]
