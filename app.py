"""
Phishing URL Detection - Flask Web Application
==============================================
Web interface for users to check if URLs are phishing or legitimate.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from predict import PhishingDetector
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize detector once on startup
detector = PhishingDetector()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DataFiles"

# Load dataset URLs on startup
def load_dataset_urls():
    """Load URLs from dataset for lookup"""
    phishing_df = pd.read_csv(DATA_DIR / 'phishing_urls.csv')
    legit_df = pd.read_csv(DATA_DIR / 'legitimate_urls.csv')
    
    phishing_urls = {}
    legit_urls = {}
    
    # Create lookup dictionaries (normalized keys)
    for idx, row in phishing_df.iterrows():
        domain = str(row['Domain']).lower().strip()
        features = {col: row[col] for col in detector.feature_names if col in phishing_df.columns}
        phishing_urls[domain] = features
    
    for idx, row in legit_df.iterrows():
        domain = str(row['Domain']).lower().strip()
        features = {col: row[col] for col in detector.feature_names if col in legit_df.columns}
        legit_urls[domain] = features
    
    return phishing_urls, legit_urls

phishing_urls, legit_urls = load_dataset_urls()
print(f"[+] Loaded {len(phishing_urls)} phishing URLs and {len(legit_urls)} legitimate URLs from dataset")


@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    Accepts either:
    - URL from dataset: {"url": "example.com"}
    - Features: {"features": {...}}
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Option 1: URL input (lookup from dataset)
        if 'url' in data:
            url = data['url'].strip()
            if not url:
                return jsonify({"error": "URL cannot be empty"}), 400
            
            # Normalize URL for lookup
            url_lower = url.lower().strip()
            
            # Check if URL is in our dataset
            if url_lower in legit_urls:
                features = legit_urls[url_lower]
                result = detector.predict_from_features(features, 'Random Forest')
                return jsonify({
                    "success": True,
                    "url": url,
                    "prediction": result['prediction'],
                    "confidence": result['confidence'],
                    "primary_model": "Random Forest",
                    "is_phishing": result['prediction'] == "Phishing",
                    "source": "Dataset (Legitimate)",
                    "extracted_features": features
                })
            
            elif url_lower in phishing_urls:
                features = phishing_urls[url_lower]
                result = detector.predict_from_features(features, 'Random Forest')
                return jsonify({
                    "success": True,
                    "url": url,
                    "prediction": result['prediction'],
                    "confidence": result['confidence'],
                    "primary_model": "Random Forest",
                    "is_phishing": result['prediction'] == "Phishing",
                    "source": "Dataset (Phishing)",
                    "extracted_features": features
                })
            
            else:
                return jsonify({
                    "error": f"URL '{url}' not found in dataset. Please use manual feature entry or try a URL from the examples.",
                    "hint": "Use the 'Enter Features' tab to provide feature values for custom URLs."
                }), 404
        
        # Option 2: Direct feature input
        elif 'features' in data:
            features = data['features']
            # Validate all required features are present
            missing = [f for f in detector.feature_names if f not in features]
            if missing:
                return jsonify({
                    "error": f"Missing features: {missing}",
                    "required_features": detector.feature_names
                }), 400
            
            result = detector.predict_from_features(features, 'Random Forest')
            
            return jsonify({
                "success": True,
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "primary_model": "Random Forest",
                "is_phishing": result['prediction'] == "Phishing"
            })
        
        else:
            return jsonify({
                "error": "Invalid request format. Provide either 'url' or 'features'",
                "example_url": {"url": "example.com"},
                "example_features": {"features": {f: 0 for f in detector.feature_names}}
            }), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models and feature names"""
    return jsonify({
        "models": detector.available_models(),
        "feature_names": detector.feature_names,
        "metrics": detector.get_metrics()
    })


@app.route('/api/example', methods=['GET'])
def get_example():
    """Get example phishing and legitimate URLs from dataset"""
    try:
        phish_df = pd.read_csv(DATA_DIR / 'phishing_urls.csv')
        legit_df = pd.read_csv(DATA_DIR / 'legitimate_urls.csv')
        
        return jsonify({
            "phishing_examples": phish_df['Domain'].head(3).tolist(),
            "legitimate_examples": legit_df['Domain'].head(3).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Page not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("  Phishing URL Detection - Web Application")
    print("=" * 70)
    print(f"[+] Models loaded: {', '.join(detector.available_models())}")
    print(f"[+] Features used: {len(detector.feature_names)}")
    print("\n[*] Starting Flask app...")
    print("[*] Open http://localhost:5000 in your browser")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
