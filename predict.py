"""
Phishing URL Detection - Prediction Module
==========================================
Load trained models and predict from CSV data or manual features.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from url_extractor import URLFeatureExtractor

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "DataFiles"


class PhishingDetector:
    """
    Unified interface for all 5 trained classifiers.
    Uses the same 15 features as the training CSV data.

    Usage
    -----
    detector = PhishingDetector()
    result   = detector.predict_from_features(feature_dict)
    print(result)
    """

    MODEL_META = {
        "Decision Tree":    {"file": "decision_tree.pkl",   "type": "sklearn"},
        "Random Forest":    {"file": "random_forest.pkl",   "type": "sklearn"},
        "XGBoost":          {"file": "xgboost.pkl",         "type": "xgboost"},
        "Autoencoder NN":   {"file": "autoencoder_nn.keras","type": "keras"},
        "SVM":              {"file": "svm.pkl",             "type": "sklearn", "scaled": True},
    }

    def __init__(self):
        self.models  = {}
        self.scaler  = None
        self.metrics = {}
        self.feature_names = None
        self._load_all()

    def _load_all(self):
        # Get feature names from training data
        features_path = DATA_DIR / "features.csv"
        if features_path.exists():
            df = pd.read_csv(features_path)
            self.feature_names = [col for col in df.columns if col != 'label']
            print(f"[+] Found {len(self.feature_names)} features")
        
        # Scaler
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Metrics
        metrics_path = MODEL_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                for m in json.load(f):
                    self.metrics[m["model"]] = m

        # Models
        for name, meta in self.MODEL_META.items():
            path = MODEL_DIR / meta["file"]
            if not path.exists():
                continue
            try:
                if meta["type"] == "sklearn":
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
                elif meta["type"] == "xgboost":
                    from xgboost import XGBClassifier
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
                elif meta["type"] == "keras":
                    import tensorflow as tf
                    self.models[name] = tf.keras.models.load_model(str(path))
                print(f"[+] Loaded {name}")
            except Exception as e:
                print(f"[!] Could not load {name}: {e}")

    def predict_from_features(self, features: dict, model_name: str = "Random Forest") -> dict:
        """
        Predict from a feature dictionary.
        
        Parameters
        ----------
        features : dict
            Dictionary with feature names as keys and values (0 or 1 for binary features)
        model_name : str
            Which model to use for primary prediction
            
        Returns
        -------
        dict with prediction results
        """
        if not self.feature_names:
            return {"error": "Feature names not initialized"}
        
        # Build feature array in correct order
        feat_arr = np.array([features.get(f, 0) for f in self.feature_names]).reshape(1, -1)
        
        all_predictions = {}
        for mname, model in self.models.items():
            meta = self.MODEL_META[mname]
            try:
                if meta.get("scaled") and self.scaler:
                    X = self.scaler.transform(feat_arr)
                else:
                    X = feat_arr
                
                if meta["type"] == "keras":
                    X_sc = self.scaler.transform(feat_arr) if self.scaler else feat_arr
                    _, prob = model.predict(X_sc, verbose=0)
                    prob = float(prob[0][0])
                    pred = int(prob >= 0.5)
                else:
                    prob = float(model.predict_proba(X)[0][1])
                    pred = int(model.predict(X)[0])
                
                all_predictions[mname] = {
                    "label": pred, 
                    "confidence": round(prob * 100, 2),
                    "class": "Phishing" if pred == 1 else "Legitimate"
                }
            except Exception as e:
                all_predictions[mname] = {"error": str(e)}

        # Primary prediction
        primary = all_predictions.get(model_name, list(all_predictions.values())[0])
        
        return {
            "primary_model": model_name,
            "prediction": primary.get("class", "Unknown"),
            "confidence": primary.get("confidence", 0),
            "all_predictions": all_predictions,
            "features": features,
        }
    
    def predict_from_csv(self, csv_path: str, num_samples: int = 5):
        """
        Predict from a CSV file with features.
        """
        df = pd.read_csv(csv_path)
        results = []
        
        for idx, row in df.head(num_samples).iterrows():
            features = {col: row[col] for col in self.feature_names if col in df.columns}
            result = self.predict_from_features(features)
            results.append(result)
        
        return results

    def predict_from_url(self, url: str, model_name: str = "Random Forest") -> dict:
        """
        Predict directly from a raw URL by extracting features.
        
        Parameters
        ----------
        url : str
            The URL to check (e.g., 'https://example.com')
        model_name : str
            Which model to use for prediction
            
        Returns
        -------
        dict with prediction results including extracted features
        """
        try:
            # Extract features from URL
            features = URLFeatureExtractor.extract_features(url)
            
            # Predict using extracted features
            result = self.predict_from_features(features, model_name)
            result['url'] = url
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to process URL: {str(e)}",
                "url": url
            }

    def available_models(self):
        return list(self.models.keys())

    def get_metrics(self):
        return self.metrics


if __name__ == "__main__":
    detector = PhishingDetector()
    if not detector.models:
        print("[!] No models found. Please run train_models.py first.")
    else:
        print("\n" + "="*60)
        print("  Phishing URL Detector - Prediction Examples")
        print("="*60)
        
        # Example: Predict from test CSV data
        test_csv = DATA_DIR / "phishing_urls.csv"
        if test_csv.exists():
            print("\n[+] Predictions from Phishing URLs CSV:")
            results = detector.predict_from_csv(str(test_csv), num_samples=3)
            for i, r in enumerate(results, 1):
                print(f"\n  Sample {i}:")
                print(f"    Prediction: {r['prediction']}")
                print(f"    Confidence: {r['confidence']}%")
                for model, pred in r['all_predictions'].items():
                    if 'error' not in pred:
                        print(f"      {model}: {pred['class']} ({pred['confidence']}%)")
        
        # Example: Manual feature prediction
        print("\n" + "-"*60)
        print("[+] Manual Feature Prediction Example:")
        example_features = {
            'Have_IP': 1,
            'Have_At': 0,
            'URL_Length': 1,
            'URL_Depth': 0,
            'Redirection': 1,
            'https_Domain': 1,
            'Tiny_URL': 1,
            'Prefix/Suffix': 1,
            'DNS_Record': 0,
            'Web_Traffic': 0,
            'Domain_Age': 0,
            'Domain_End': 0,
            'iFrame': 1,
            'Mouse_Over': 0,
            'Right_Click': 0,
        }
        r = detector.predict_from_features(example_features)
        print(f"\nManual Features: {example_features}")
        print(f"Prediction: {r['prediction']}")
        print(f"Confidence: {r['confidence']}%")