"""
app.py
======
Flask backend for Phishing URL Detection System.
Handles both API endpoint for prediction and the frontend application.
"""

import os
import sqlite3
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template

import joblib
import pandas as pd
import numpy as np

# Adjust python path to import our custom feature extractor module
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))

from feature_extractor import extract_features, MODEL_FEATURE_COLUMNS, CONTINUOUS_COLUMNS
from whois_intel import get_whois_features

app = Flask(__name__)

# --- SQLite Database Initialization ---
DB_NAME = "history.db"

def init_db():
    """Initializes the SQLite database used for storing scan history."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            verdict TEXT,
            confidence REAL,
            timestamp TEXT,
            risk_flags TEXT,
            domain_age INTEGER,
            registrar TEXT,
            country TEXT,
            data_source TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Model Loading at Startup ---
MODEL_DIR = Path(__file__).resolve().parent / 'model'
try:
    with open(MODEL_DIR / 'model.pkl', 'rb') as f:
        model = joblib.load(f)
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    print("✅ Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("⚠️ WARNING: model.pkl or scaler.pkl missing. Predictions won't work unless trained.")
    model = None
    scaler = None

# --- Risk Flag Engine (Layer 9) ---
def compute_risk_flags(features: dict) -> list[str]:
    """Calculate rule-based flags from the extracted features."""
    flags = []
    
    # Layer 9 Rules
    if features.get('_whitelisted', False):
        # Whitelist hit -> clear all flags
        return flags
        
    if features.get('suspecious_tld', 0) == 1:
        flags.append("Suspicious TLD")
        
    if features.get('ip', 0) == 1:
        flags.append("IP used instead of domain")
        
    if features.get('phish_hints', 0) >= 2:
        flags.append("High number of suspicious keywords")

    if features.get('ratio_digits_url', 0) > 0.2:
        flags.append("High ratio of digits in URL")
        
    return flags

# --- Routes ---

@app.route("/", methods=["GET"])
def index():
    """Render the single-page interface."""
    return render_template("index.html")

@app.route("/history", methods=["GET"])
def get_history():
    """Return the last 10 scans from the local DB."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM scans ORDER BY id DESC LIMIT 10')
        rows = c.fetchall()
        history = [dict(ix) for ix in rows]
        conn.close()
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a URL via JSON POST, runs feature extraction, prediction,
    and returns a summarized JSON record.
    """
    if not model or not scaler:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
        
    data = request.json
    if not data or not data.get("url"):
        return jsonify({"error": "No URL provided"}), 400
        
    url = data["url"]
    
    try:
        # 1. Feature Extraction
        feats = extract_features(url, enable_network=True)
        
        flags = compute_risk_flags(feats)
        
        # Format features into DataFrame to feed the model
        df_new = pd.DataFrame([feats])
        
        # Make a copy to scale continuous vars
        X_infer = df_new[MODEL_FEATURE_COLUMNS].copy()
        
        X_infer[CONTINUOUS_COLUMNS] = scaler.transform(X_infer[CONTINUOUS_COLUMNS])
        
        # 2. Prediction Model
        # Predict probability of class 1 (Phishing)
        y_prob = model.predict_proba(X_infer)[0, 1]
        
        confidence = float(y_prob) * 100
        
        # 3.5 Re-enable WHOIS just for UI cosmetics (Does not affect the ML Model)
        domain_for_whois = feats.get('_domain', "")
        whois_data = get_whois_features(domain_for_whois) if domain_for_whois else {}
        
        # --- NEW OVERRIDE: WHOIS Failure Penalty ---
        # If WHOIS fails and it's NOT a mathematically whitelisted domain
        if whois_data.get('whois_status', 'failed') == 'failed' and not feats.get('_whitelisted', False):
            confidence += 48.0
            confidence = min(confidence, 100.0) # Cap at 100%
            flags.append("WHOIS Data Hidden/Blocked")
            
        # 3. Verdict threshold (Layer 9 score zones)
        if feats.get('_whitelisted', False):
            verdict = "LEGITIMATE"
            confidence = 0.0 # Threat score is 0 if mathematically whitelisted
        else:
            if confidence < 40.0:
                verdict = "LEGITIMATE"
            elif confidence <= 65.0:
                verdict = "PARTIALLY VULNERABLE"
            else:
                verdict = "PHISHING"
        
        # Prepare response
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        response_data = {
            "url": url,
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "timestamp": timestamp,
            "risk_flags": flags,
            "parameters": {k: v for k, v in feats.items() if not k.startswith('_')},
            "whois_summary": {
                "domain_age": whois_data.get('domain_age_days', -1) if whois_data.get('domain_age_days', -1) != -1 else 'Unknown',
                "registrar": whois_data.get('registrar_name', 'Unknown'),
                "country": whois_data.get('country', 'Unknown'),
                "data_source": 'Live WHOIS Lookup'
            }
        }
        
        # 4. Save to Database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO scans 
            (url, verdict, confidence, timestamp, risk_flags, domain_age, registrar, country, data_source) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            response_data["url"],
            response_data["verdict"],
            response_data["confidence"],
            response_data["timestamp"],
            ", ".join(flags),
            str(response_data["whois_summary"]["domain_age"]),
            response_data["whois_summary"]["registrar"],
            response_data["whois_summary"]["country"],
            response_data["whois_summary"]["data_source"]
        ))
        conn.commit()
        conn.close()
        
        return jsonify(response_data), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
