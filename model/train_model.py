"""
train_model.py
==============
Trains the XGBoost model for Phishing URL Detection.
Reverts to Path 1: Uses `feature_extractor.py` to extract our custom 22 features
from raw URLs so the model exactly matches the realtime `app.py` pipeline.
"""

import os
import glob
import logging
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sys

# Ensure feature_extractor can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, MODEL_FEATURE_COLUMNS, CONTINUOUS_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_raw_url_data() -> pd.DataFrame:
    """Load a dataset containing raw URLs and their respective labels."""
    # We will use the large phishing_site_urls.csv present in the root folder
    root_dir = Path(__file__).resolve().parent.parent.parent
    large_csv = root_dir / "phishing_site_urls.csv"
    
    if large_csv.exists():
        logger.info(f"Found large raw URL dataset: {large_csv}")
        df = pd.read_csv(large_csv)
        
        # Format the columns to match expected 'url' and 'label'
        if 'URL' in df.columns and 'Label' in df.columns:
            df = df.rename(columns={'URL': 'url'})
            df['label'] = df['Label'].apply(lambda x: 1 if str(x).strip().lower() == 'bad' else 0)
            
            # Since online extraction is slow (DNS/WHOIS/RDAP requests),
            # we sample a well-balanced subset of 1000 URLs to train quickly.
            # In a true production environment, you would pre-extract the whole 30MB file.
            logger.info("Sampling 1000 URLs (500 Phishing, 500 Legitimate) for fast training...")
            df_phish = df[df['label'] == 1].sample(500, random_state=42)
            df_legit = df[df['label'] == 0].sample(500, random_state=42)
            df_sampled = pd.concat([df_phish, df_legit], ignore_index=True)
            
            return df_sampled
            
    # Fallback to any local dataset/ CSVs
    dataset_dir = Path(__file__).resolve().parent.parent / 'dataset'
    csv_files = glob.glob(str(dataset_dir / "*.csv"))
    
    dfs = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        if 'url' in temp_df.columns and 'label' in temp_df.columns:
            dfs.append(temp_df)
            
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} raw URLs from local dataset dir.")
        return combined

    raise FileNotFoundError("Could not find a valid raw URL dataset (phishing_site_urls.csv)!")


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature extraction on every URL using our custom 22-feature pipeline."""
    logger.info("Extracting our 22 Custom Features from URLs using feature_extractor.py...")
    
    feature_list = []
    
    for url in tqdm(df['url'], desc="Extracting features"):
        try:
            # We enable network to grab WHOIS/RDAP. This takes ~1-3s per URL.
            feats = extract_features(url, enable_network=True)
            feature_list.append(feats)
        except Exception as e:
            logger.debug(f"Failed to extract features for {url}: {e}")
            fallback = {col: -1 for col in MODEL_FEATURE_COLUMNS}
            fallback['_whitelisted'] = False
            feature_list.append(fallback)
            
    features_df = pd.DataFrame(feature_list)
    features_df['label'] = df['label'].values
    
    return features_df


def train():
    """Main training pipeline."""
    # 1. Load Raw Data
    df = load_raw_url_data()
    
    # 2. Extract exactly the 22 features the Flask app expects
    features_df = extract_all_features(df)
    
    # 3. Pull the specific columns designated in feature_extractor.py
    X = features_df[MODEL_FEATURE_COLUMNS].copy()
    y = features_df['label']
    
    logger.info("Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info("Scaling continuous features...")
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # We only scale the continuous columns (boolean columns stay 0/1)
    X_train_scaled[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train[CONTINUOUS_COLUMNS])
    X_test_scaled[CONTINUOUS_COLUMNS] = scaler.transform(X_test[CONTINUOUS_COLUMNS])
    
    logger.info("Training XGBoost Classifier on exactly 22 features...")
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
    
    logger.info("Training complete! model.pkl now strictly expects 22 features.")
    logger.info("You can safely run `python app.py` and the backend will match perfectly.")

if __name__ == "__main__":
    train()
