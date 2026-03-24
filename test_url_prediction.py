from url_extractor import URLFeatureExtractor
from predict import PhishingDetector

detector = PhishingDetector()

test_urls = [
    'https://sbi.bank.in',
    'https://appleid.apple.com-sa.pm',
    'https://graphicriver.net',
    'https://www.google.com',
    'https://icicibank.com',
]

print("\n" + "="*70)
print("URL PREDICTION TEST RESULTS")
print("="*70)

for url in test_urls:
    result = detector.predict_from_url(url, 'Random Forest')
    print(f'\nURL: {url}')
    print(f'  Prediction: {result["prediction"]}')
    print(f'  Confidence: {result["confidence"]}%')
