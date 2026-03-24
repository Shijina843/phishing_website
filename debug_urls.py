import pandas as pd
from pathlib import Path

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / "DataFiles"

phishing_df = pd.read_csv(DATA_DIR / 'phishing_urls.csv')
legit_df = pd.read_csv(DATA_DIR / 'legitimate_urls.csv')

# Test URL
test_url = 'graphicriver.net'

print(f"Total phishing URLs in CSV: {len(phishing_df)}")
print(f"Total legitimate URLs in CSV: {len(legit_df)}")
print()

print(f"Testing URL: {test_url}")
print(f"In phishing CSV: {any(phishing_df['Domain'].str.lower().str.strip() == test_url)}")
print(f"In legit CSV: {any(legit_df['Domain'].str.lower().str.strip() == test_url)}")
print()

# Check what's actually in the dicts as app loads them
from predict import PhishingDetector

detector = PhishingDetector()
feature_names = detector.feature_names

print(f"Feature names: {feature_names}")
print()

# Load like the app does
phishing_urls = {}
legit_urls = {}

for idx, row in phishing_df.iterrows():
    domain = str(row['Domain']).lower().strip()
    features = {col: row[col] for col in feature_names if col in phishing_df.columns}
    phishing_urls[domain] = features

for idx, row in legit_df.iterrows():
    domain = str(row['Domain']).lower().strip()
    features = {col: row[col] for col in feature_names if col in legit_df.columns}
    legit_urls[domain] = features

print(f"Dict sizes: phishing={len(phishing_urls)}, legit={len(legit_urls)}")
print()

print(f"Is '{test_url}' in phishing_urls? {test_url in phishing_urls}")
print(f"Is '{test_url}' in legit_urls? {test_url in legit_urls}")
print()

if test_url in phishing_urls:
    print(f"\nFound in phishing_urls:")
    for k, v in phishing_urls[test_url].items():
        print(f"  {k}: {v}")

if test_url in legit_urls:
    print(f"\nFound in legit_urls:")
    for k, v in legit_urls[test_url].items():
        print(f"  {k}: {v}")
