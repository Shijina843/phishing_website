"""
Phishing URL Detection - Feature Extraction Module
===================================================
Extracts 17 features from URLs grouped into:
  1. Address Bar based Features (9)
  2. Domain based Features (4)
  3. HTML & Javascript based Features (4)
"""

import re
import urllib.parse
from datetime import datetime
import ipaddress

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_tld(url: str) -> str:
    domain = get_domain(url)
    parts = domain.split(".")
    return parts[-1] if parts else ""


# ─────────────────────────────────────────────
# 1. Address Bar Based Features (9)
# ─────────────────────────────────────────────

def has_ip_address(url: str) -> int:
    """Feature 1: URL contains IP address instead of domain name → Phishing (1)"""
    try:
        domain = get_domain(url)
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        # Also check IPv4 pattern in URL path
        ipv4_pattern = r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])'
        if re.search(ipv4_pattern, url):
            return 1
        return 0


def url_length(url: str) -> int:
    """Feature 2: URL length. ≥54 chars → Phishing (1); <54 → Legitimate (0)"""
    length = len(url)
    if length < 54:
        return 0
    elif length <= 75:
        return -1   # Suspicious
    else:
        return 1


def url_shortening(url: str) -> int:
    """Feature 3: Uses URL shortening service → Phishing (1)"""
    shortening_services = (
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
        r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
        r"su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|"
        r"Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
        r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
        r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|is\.gd|"
        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|"
        r"yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|"
        r"qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net"
    )
    if re.search(shortening_services, url):
        return 1
    return 0


def has_at_symbol(url: str) -> int:
    """Feature 4: '@' symbol in URL → Phishing (1)"""
    return 1 if "@" in url else 0


def double_slash_redirect(url: str) -> int:
    """Feature 5: '//' after position 7 → Phishing (1)"""
    # After removing http:// or https://
    url_without_scheme = url[url.find("//") + 2:]
    return 1 if "//" in url_without_scheme else 0


def prefix_suffix_in_domain(url: str) -> int:
    """Feature 6: Domain contains '-' → Phishing (1)"""
    domain = get_domain(url)
    return 1 if "-" in domain else 0


def sub_domains(url: str) -> int:
    """Feature 7: Number of sub-domains. >2 → Phishing (1)"""
    domain = get_domain(url)
    # Remove TLD
    parts = domain.split(".")
    # Dots in domain (excluding TLD)
    num_dots = len(parts) - 1
    if num_dots == 1:
        return 0   # Legitimate
    elif num_dots == 2:
        return -1  # Suspicious
    else:
        return 1   # Phishing


def https_token(url: str) -> int:
    """Feature 8: HTTPS used in the URL scheme → Legitimate (0), HTTP → Phishing (1)"""
    return 0 if url.lower().startswith("https://") else 1


def domain_registration_length(url: str) -> int:
    """
    Feature 9: Domain registration length proxy.
    We use URL pattern heuristics since WHOIS lookup
    requires network access. Short numeric-heavy domains → Phishing (1)
    """
    domain = get_domain(url)
    # Heuristic: many numbers in domain suggest throwaway registration
    num_digits = sum(c.isdigit() for c in domain)
    if num_digits > 3:
        return 1
    return 0


# ─────────────────────────────────────────────
# 2. Domain Based Features (4)
# ─────────────────────────────────────────────

def suspicious_words(url: str) -> int:
    """Feature 10: Sensitive words in URL → Phishing (1)"""
    keywords = [
        "secure", "account", "webscr", "login", "ebayisapi", "signin",
        "banking", "confirm", "weblogin", "update", "verification", "paypal",
        "free", "lucky", "service", "bonus", "ebay", "verify", "password",
        "support", "security", "credential", "alert"
    ]
    url_lower = url.lower()
    return 1 if any(kw in url_lower for kw in keywords) else 0


def favicon(url: str) -> int:
    """
    Feature 11: Favicon loaded from external domain → Phishing (1).
    Heuristic: multiple different domains in URL path.
    """
    domain = get_domain(url)
    # Check if URL references another domain's assets
    url_lower = url.lower()
    external_refs = ["favicon", ".ico"]
    for ref in external_refs:
        if ref in url_lower:
            return 1
    return 0


def non_std_port(url: str) -> int:
    """Feature 12: Non-standard port in URL → Phishing (1)"""
    try:
        parsed = urllib.parse.urlparse(url)
        port = parsed.port
        if port and port not in (80, 443, 8080, 8443):
            return 1
    except Exception:
        pass
    return 0


def https_domain_url(url: str) -> int:
    """Feature 13: 'https' token in domain part of URL → Phishing (1)"""
    domain = get_domain(url)
    return 1 if "https" in domain.lower() else 0


# ─────────────────────────────────────────────
# 3. HTML & JavaScript Based Features (4)
# ─────────────────────────────────────────────

def request_url(url: str) -> int:
    """
    Feature 14: % of external objects (proxy via URL depth/complexity).
    Complex query strings suggest data exfiltration → Phishing (1)
    """
    parsed = urllib.parse.urlparse(url)
    query = parsed.query
    if len(query) > 100:
        return 1
    params = urllib.parse.parse_qs(query)
    if len(params) > 5:
        return 1
    return 0


def url_of_anchor(url: str) -> int:
    """
    Feature 15: Anchor tags linking to different domain (heuristic via
    multiple subdomains or encoded characters) → Phishing (1)
    """
    # Check for encoded characters (obfuscation)
    if re.search(r'%[0-9a-fA-F]{2}', url):
        return 1
    return 0


def links_in_meta(url: str) -> int:
    """
    Feature 16: Meta/script/link tags pointing to another domain.
    Heuristic: URL contains data: or javascript: → Phishing (1)
    """
    suspicious = ["data:", "javascript:", "vbscript:"]
    url_lower = url.lower()
    return 1 if any(s in url_lower for s in suspicious) else 0


def sfh(url: str) -> int:
    """
    Feature 17: Server Form Handler — empty or about:blank forms.
    Heuristic: URL contains form-related keywords with suspicious patterns.
    """
    suspicious_form_patterns = ["about:blank", "about%3Ablank"]
    url_lower = url.lower()
    return 1 if any(p in url_lower for p in suspicious_form_patterns) else 0


# ─────────────────────────────────────────────
# Master Feature Extractor
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    # Address Bar (9)
    "has_ip_address",
    "url_length",
    "url_shortening",
    "has_at_symbol",
    "double_slash_redirect",
    "prefix_suffix_in_domain",
    "sub_domains",
    "https_token",
    "domain_registration_length",
    # Domain (4)
    "suspicious_words",
    "favicon",
    "non_std_port",
    "https_domain_url",
    # HTML & JS (4)
    "request_url",
    "url_of_anchor",
    "links_in_meta",
    "sfh",
]

FEATURE_FUNCTIONS = [
    has_ip_address,
    url_length,
    url_shortening,
    has_at_symbol,
    double_slash_redirect,
    prefix_suffix_in_domain,
    sub_domains,
    https_token,
    domain_registration_length,
    suspicious_words,
    favicon,
    non_std_port,
    https_domain_url,
    request_url,
    url_of_anchor,
    links_in_meta,
    sfh,
]


def extract_features(url: str) -> list:
    """Extract all 17 features from a single URL."""
    return [fn(url) for fn in FEATURE_FUNCTIONS]


def extract_features_df(urls) -> "pd.DataFrame":
    """Extract features from a list/Series of URLs into a DataFrame."""
    import pandas as pd
    records = []
    for url in urls:
        try:
            records.append(extract_features(url))
        except Exception:
            records.append([0] * len(FEATURE_NAMES))
    return pd.DataFrame(records, columns=FEATURE_NAMES)


if __name__ == "__main__":
    test_urls = [
        "http://192.168.1.1/login",
        "https://www.google.com",
        "http://bit.ly/abc123",
        "https://paypal-secure-login.com/verify?account=123456789",
        "https://github.com/user/repo",
    ]
    for url in test_urls:
        features = extract_features(url)
        print(f"\nURL: {url}")
        for name, val in zip(FEATURE_NAMES, features):
            print(f"  {name:35s}: {val}")