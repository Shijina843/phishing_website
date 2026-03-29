"""
train_model.py
==============
Trains the XGBoost model for Phishing URL Detection.
Uses 15 URL lexical features taken exactly from phishing_data.csv,
eliminating SLOW network queries to make training blazingly fast.
"""

import os
import logging
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sys

# Ensure feature_extractor can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import MODEL_FEATURE_COLUMNS, CONTINUOUS_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train():
    """Main training pipeline taking features directly from dataset."""
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "dataset" / "phishing_data.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")
        
    logger.info(f"Loading pre-extracted raw features from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check that all features exist
    missing_cols = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in CSV: {missing_cols}")
        
    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in CSV")
        
    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df['label']
    
    logger.info("Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info("Scaling continuous features...")
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if CONTINUOUS_COLUMNS:
        X_train_scaled[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train[CONTINUOUS_COLUMNS])
        X_test_scaled[CONTINUOUS_COLUMNS] = scaler.transform(X_test[CONTINUOUS_COLUMNS])
    
    logger.info(f"Training XGBoost Classifier on {len(MODEL_FEATURE_COLUMNS)} features...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    logger.info("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    X_test_scaled['label'] = y_test.values
    X_test_scaled.to_csv('test_data_for_eval.csv', index=False)
    
    logger.info("Training complete! Model bypasses WHOIS and perfectly matches the 15 features.")

if __name__ == "__main__":
    train()
