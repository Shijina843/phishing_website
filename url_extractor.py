"""
URL Feature Extraction Module
============================
Extract the 15 features from raw URLs for phishing detection.
"""

import re
from urllib.parse import urlparse
import socket
import requests
from datetime import datetime


class URLFeatureExtractor:
    """Extract 15 engineered features from a raw URL"""
    
    TINY_URL_SERVICES = [
        'bit.ly', 'tinyurl.com', 'short.link', 'ow.ly', 'goo.gl',
        'buff.ly', 't.co', 'adf.ly', 'tiny.cc', 'is.gd'
    ]
    
    SUSPICIOUS_TLDs = ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.click', '.digital', '.loan']
    
    @staticmethod
    def extract_features(url: str) -> dict:
        """
        Extract 15 features from a URL.
        
        Returns
        -------
        dict with all 15 feature values
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            features = {}
            
            # 1. Have_IP: URL contains IP address
            features['Have_IP'] = URLFeatureExtractor._has_ip(domain)
            
            # 2. Have_At: URL contains @ symbol
            features['Have_At'] = 1 if '@' in url else 0
            
            # 3. URL_Length: Length of URL (normalized to 0/1)
            features['URL_Length'] = 1 if len(url) > 54 else 0
            
            # 4. URL_Depth: Depth of URL path (number of /)
            features['URL_Depth'] = 1 if path.count('/') > 1 else 0
            
            # 5. Redirection: Double slash // in path
            features['Redirection'] = 1 if '//' in path else 0
            
            # 6. https_Domain: Uses HTTPS
            features['https_Domain'] = 1 if parsed.scheme == 'https' else 0
            
            # 7. Web_Forwards: Check for forwards (similar to Tiny_URL concept)
            # Uses shortened URL service or unusual forwards
            features['Web_Forwards'] = URLFeatureExtractor._has_web_forwards(url, domain)
            
            # 8. Prefix/Suffix: Contains - in domain
            domain_only = domain.split(':')[0]  # Remove port if exists
            features['Prefix/Suffix'] = 1 if '-' in domain_only else 0
            
            # 9. DNS_Record: Check if domain has valid DNS (heuristic: if popular domain = 1)
            features['DNS_Record'] = URLFeatureExtractor._check_dns(domain_only)
            
            # 10. Web_Traffic: Estimate traffic (heuristic: based on domain characteristics)
            features['Web_Traffic'] = URLFeatureExtractor._estimate_traffic(domain)
            
            # 11. Domain_Age: Estimate domain age (heuristic)
            features['Domain_Age'] = URLFeatureExtractor._estimate_domain_age(domain)
            
            # 12. Domain_End: Check if TLD is suspicious
            features['Domain_End'] = URLFeatureExtractor._check_domain_extension(domain)
            
            # 13. iFrame: Check for iframe markers in URL (usually rare in legitimate)
            features['iFrame'] = 1 if 'iframe' in url.lower() else 0
            
            # 14. Mouse_Over: Usually not in URL, default to 0
            features['Mouse_Over'] = 0
            
            # 15. Right_Click: Usually not in URL, default to 0
            features['Right_Click'] = 0
            
            return features
            
        except Exception as e:
            print(f"[!] Error extracting features from {url}: {e}")
            # Return safe defaults on error
            return URLFeatureExtractor._get_default_features()
    
    @staticmethod
    def _has_ip(domain: str) -> int:
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return 1 if re.match(ip_pattern, domain) else 0
    
    @staticmethod
    def _is_tiny_url(domain: str) -> int:
        """Check if domain is a shortened URL service or has web forwards"""
        for service in URLFeatureExtractor.TINY_URL_SERVICES:
            if service in domain.lower():
                return 1
        return 0
    
    @staticmethod
    def _has_web_forwards(url: str, domain: str) -> int:
        """Check for shortcuts/forwarding services (same as Tiny_URL in dataset)"""
        return URLFeatureExtractor._is_tiny_url(domain)
    
    @staticmethod
    def _check_dns(domain: str) -> int:
        """
        Try to resolve domain (simple check).
        Returns 1 if domain exists (can be resolved).
        """
        try:
            # Remove port if exists
            domain_only = domain.split(':')[0]
            socket.gethostbyname(domain_only)
            return 1  # Domain exists
        except:
            return 0  # Domain doesn't exist or resolution failed
    
    @staticmethod
    def _estimate_traffic(domain: str) -> int:
        """
        Estimate if domain likely has web traffic.
        Heuristic: known popular domains = 1, others = 0
        """
        popular_domains = [
            'google', 'facebook', 'amazon', 'apple', 'microsoft', 'github',
            'stackoverflow', 'wikipedia', 'linkedin', 'twitter', 'reddit',
            'youtube', 'instagram', 'netflix', 'ebay', 'walmart',
            'alibaba', 'sbi', 'hdfc', 'icici', 'axis',  # Indian banks
        ]
        
        domain_lower = domain.lower()
        for pop_domain in popular_domains:
            if pop_domain in domain_lower:
                return 1
        
        # Check for common TLDs (more likely to have traffic)
        if any(domain_lower.endswith(tld) for tld in ['.com', '.org', '.edu', '.gov', '.co.in']):
            return 1
        
        return 0
    
    @staticmethod
    def _estimate_domain_age(domain: str) -> int:
        """
        Estimate domain age (heuristic).
        Known old/established domains = 1, unknown = 0
        """
        established_domains = [
            'google', 'facebook', 'amazon', 'apple', 'microsoft', 'github',
            'stackoverflow', 'wikipedia', 'linkedin', 'twitter', 'reddit',
            'youtube', 'instagram', 'netflix', 'ebay', 'walmart',
            'bank', 'gov', 'edu', 'org', 'sbi', 'hdfc', 'icici', 'axis',
        ]
        
        domain_lower = domain.lower()
        for est_domain in established_domains:
            if est_domain in domain_lower:
                return 1
        
        return 0
    
    @staticmethod
    def _check_domain_extension(domain: str) -> int:
        """
        Check if domain extension is suspicious.
        Returns 0 if legitimate/common TLD, 1 if suspicious
        """
        domain_lower = domain.lower()
        
        # Common/legitimate TLDs
        legitimate_tlds = [
            '.com', '.org', '.edu', '.gov', '.net', '.co.uk', '.co.in',
            '.in', '.us', '.ca', '.au', '.de', '.fr', '.it', '.es',
            '.co.jp', '.co.kr', '.ua', '.ru', '.cn', '.br', '.mx'
        ]
        
        for tld in legitimate_tlds:
            if domain_lower.endswith(tld):
                return 0
        
        # Suspicious TLDs
        for tld in URLFeatureExtractor.SUSPICIOUS_TLDs:
            if domain_lower.endswith(tld):
                return 1
        
        # Other TLDs - return 0 (not marked as suspicious)
        return 0
    
    @staticmethod
    def _get_default_features() -> dict:
        """Return safe default feature values"""
        return {
            'Have_IP': 0,
            'Have_At': 0,
            'URL_Length': 0,
            'URL_Depth': 0,
            'Redirection': 0,
            'https_Domain': 0,
            'Web_Forwards': 0,
            'Prefix/Suffix': 0,
            'DNS_Record': 0,
            'Web_Traffic': 0,
            'Domain_Age': 0,
            'Domain_End': 0,
            'iFrame': 0,
            'Mouse_Over': 0,
            'Right_Click': 0,
        }


# Test the extractor
if __name__ == '__main__':
    test_urls = [
        'https://sbi.bank.in',
        'https://sbi.bank.co.uk',
        'https://appleid.apple.com-sa.pm',
        'http://192.168.1.1:8080',
        'https://graphicriver.net',
    ]
    
    for url in test_urls:
        features = URLFeatureExtractor.extract_features(url)
        print(f"\nURL: {url}")
        for key, val in features.items():
            print(f"  {key}: {val}")
