"""
Phishing URL Detection - Dataset Preparation & Model Training
=============================================================
Trains 5 classifiers on 10 000-URL dataset (5000 phishing + 5000 legitimate)
and saves trained models + performance metrics.

Data Sources
------------
Phishing : PhishTank   → DataFiles/phishing_urls.csv
Legitimate: UNB CIC    → DataFiles/legitimate_urls.csv
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "DataFiles"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

from feature_extraction import extract_features_df, FEATURE_NAMES


# ─────────────────────────────────────────────
# 1. Load / Generate Dataset
# ─────────────────────────────────────────────

def load_data(n_phishing: int = 5000, n_legit: int = 5000):
    """
    Load pre-extracted features from CSV files.
    The CSVs already contain engineered features from website analysis.
    """
    phishing_path = DATA_DIR / "phishing_urls.csv"
    legit_path    = DATA_DIR / "legitimate_urls.csv"

    if phishing_path.exists() and legit_path.exists():
        print("[+] Loading dataset with engineered features …")
        phish_df = pd.read_csv(phishing_path)
        legit_df = pd.read_csv(legit_path)
        
        # Find common feature columns (handling name mismatches)
        phish_cols = set(phish_df.columns) - {'Domain', 'Label', 'label'}
        legit_cols = set(legit_df.columns) - {'Domain', 'Label', 'label'}
        common_cols = sorted(phish_cols & legit_cols)
        
        if not common_cols:
            # Alternative: use all columns except the meta ones
            common_cols = [col for col in phish_df.columns 
                          if col not in ['Domain', 'Label', 'label']]
        
        print(f"    Using {len(common_cols)} features: {common_cols[:5]}...")
        
        # Sample data
        phish_features = phish_df[common_cols].sample(
            min(n_phishing, len(phish_df)), random_state=42
        )
        legit_features = legit_df[common_cols].sample(
            min(n_legit, len(legit_df)), random_state=42
        ).reset_index(drop=True)
        
        phish_features = phish_features.reset_index(drop=True)
        
        X = pd.concat([phish_features, legit_features], ignore_index=True)
        y = [1] * len(phish_features) + [0] * len(legit_features)
        
        return X, pd.Series(y, name="label")
    else:
        print("[!] Dataset CSVs not found — using legacy feature extraction.")
        urls, labels = load_data_legacy(n_phishing, n_legit)
        return prepare_features(urls, labels)


def load_data_legacy(n_phishing: int = 5000, n_legit: int = 5000):
    """Legacy method using feature extraction on URLs."""


def _generate_synthetic_urls(n_phishing, n_legit):
    """Generate representative synthetic URLs for demo / CI purposes."""
    rng = np.random.default_rng(42)

    phishing_templates = [
        "http://192.168.{}.{}/login/secure",
        "http://paypal-secure-login{}.com/verify?account={}",
        "http://bit.ly/{}{}",
        "http://www.free-prize-{}.tk/claim?user={}",
        "http://banking-update{}.info/confirm-password?id={}",
        "http://ebay-login-secure{}.com/signin/verify",
        "http://amazon-account-verify{}.net/login?redirect={}",
        "http://secure-paypal{}.phish/webscr?cmd=login",
        "http://192.0.2.{}/account/login",
        "http://update-your-info{}.biz/banking/verify?token={}",
    ]
    legit_templates = [
        "https://www.google.com/search?q={}",
        "https://github.com/{}/{}",
        "https://stackoverflow.com/questions/{}",
        "https://www.wikipedia.org/wiki/{}",
        "https://www.youtube.com/watch?v={}",
        "https://www.amazon.com/dp/{}",
        "https://docs.python.org/3/library/{}.html",
        "https://www.bbc.com/news/technology-{}",
        "https://www.nytimes.com/2024/{}/article",
        "https://arxiv.org/abs/{}.{}",
    ]

    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    phish_urls, legit_urls = [], []

    for i in range(n_phishing):
        tmpl = phishing_templates[i % len(phishing_templates)]
        url  = tmpl.format(rng.integers(1, 255), rng.integers(1, 255))
        phish_urls.append(url)

    for i in range(n_legit):
        tmpl = legit_templates[i % len(legit_templates)]
        w1, w2 = words[i % len(words)], words[(i + 3) % len(words)]
        url  = tmpl.format(w1, w2)
        legit_urls.append(url)

    return phish_urls, legit_urls


# ─────────────────────────────────────────────
# 2. Feature Extraction
# ─────────────────────────────────────────────

def prepare_features(urls, labels):
    print("[+] Extracting features …")
    X = extract_features_df(urls)
    y = pd.Series(labels, name="label")
    print(f"    Dataset shape: {X.shape}  |  Phishing: {sum(labels)}  Legitimate: {len(labels)-sum(labels)}")
    return X, y


# ─────────────────────────────────────────────
# 3. Metrics Helper
# ─────────────────────────────────────────────

def compute_metrics(name, y_true, y_pred, y_prob=None):
    metrics = {
        "model": name,
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob) * 100, 2)
    return metrics


# ─────────────────────────────────────────────
# 4. Train Classifiers
# ─────────────────────────────────────────────

def train_decision_tree(X_train, y_train, X_test, y_test):
    print("\n[+] Training Decision Tree …")
    dt = DecisionTreeClassifier(
        max_depth=15, 
        min_samples_split=5, 
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]
    m = compute_metrics("Decision Tree", y_test, y_pred, y_prob)
    print(f"    Accuracy: {m['accuracy']}%  |  F1: {m['f1_score']}%")
    with open(MODEL_DIR / "decision_tree.pkl", "wb") as f:
        pickle.dump(dt, f)
    return m


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n[+] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=15,
        min_samples_split=5, 
        class_weight='balanced',  # Handle class imbalance
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    m = compute_metrics("Random Forest", y_test, y_pred, y_prob)
    print(f"    Accuracy: {m['accuracy']}%  |  F1: {m['f1_score']}%")
    with open(MODEL_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    return m


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[+] Training XGBoost …")
    # Calculate scale_pos_weight to handle class imbalance
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    scale_pos_weight = neg_samples / pos_samples
    
    xgb = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        subsample=0.8, 
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        use_label_encoder=False, 
        eval_metric="logloss",
        random_state=42, 
        n_jobs=-1
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    m = compute_metrics("XGBoost", y_test, y_pred, y_prob)
    print(f"    Accuracy: {m['accuracy']}%  |  F1: {m['f1_score']}%")
    with open(MODEL_DIR / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb, f)
    return m


def train_autoencoder(X_train, y_train, X_test, y_test, scaler):
    """
    Autoencoder Neural Network used as a classifier:
    - Encoder → bottleneck → decoder trained on ALL data (unsupervised pre-training)
    - Fine-tuned classification head attached to bottleneck
    """
    print("\n[+] Training Autoencoder Neural Network …")
    n_features = X_train.shape[1]

    # ── Build autoencoder ──────────────────────────────────────────────────
    inputs  = Input(shape=(n_features,), name="input")
    x = Dense(64, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    encoded = Dense(32, activation="relu", name="bottleneck")(x)
    x = Dense(64, activation="relu")(encoded)
    x = BatchNormalization()(x)
    decoded = Dense(n_features, activation="linear", name="reconstruction")(x)

    # Classification head
    cls = Dense(16, activation="relu")(encoded)
    cls = Dropout(0.2)(cls)
    output = Dense(1, activation="sigmoid", name="classifier")(cls)

    model = Model(inputs=inputs, outputs=[decoded, output])
    model.compile(
        optimizer="adam",
        loss={"reconstruction": "mse", "classifier": "binary_crossentropy"},
        loss_weights={"reconstruction": 0.3, "classifier": 0.7},
        metrics={"classifier": "accuracy"}
    )

    # ── Train ──────────────────────────────────────────────────────────────
    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_classifier_accuracy", mode="max"),
        ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss", mode="min")
    ]
    model.fit(
        X_tr_sc,
        {"reconstruction": X_tr_sc, "classifier": y_train.values},
        validation_data=(
            X_te_sc,
            {"reconstruction": X_te_sc, "classifier": y_test.values}
        ),
        epochs=100, batch_size=64,
        callbacks=callbacks, verbose=0
    )

    _, y_prob = model.predict(X_te_sc, verbose=0)
    y_pred    = (y_prob.flatten() >= 0.5).astype(int)
    y_prob_1d = y_prob.flatten()

    m = compute_metrics("Autoencoder NN", y_test, y_pred, y_prob_1d)
    print(f"    Accuracy: {m['accuracy']}%  |  F1: {m['f1_score']}%")

    model.save(str(MODEL_DIR / "autoencoder_nn.keras"))
    return m


def train_svm(X_train, y_train, X_test, y_test, scaler):
    print("\n[+] Training SVM …")
    X_tr_sc = scaler.transform(X_train)
    X_te_sc = scaler.transform(X_test)
    svm = SVC(
        kernel="rbf", 
        C=10, 
        gamma="scale", 
        probability=True, 
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    svm.fit(X_tr_sc, y_train)
    y_pred = svm.predict(X_te_sc)
    y_prob = svm.predict_proba(X_te_sc)[:, 1]
    m = compute_metrics("SVM", y_test, y_pred, y_prob)
    print(f"    Accuracy: {m['accuracy']}%  |  F1: {m['f1_score']}%")
    with open(MODEL_DIR / "svm.pkl", "wb") as f:
        pickle.dump(svm, f)
    return m


# ─────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   Phishing URL Detection — Model Training Pipeline")
    print("=" * 60)

    # Load features
    X, y = load_data(n_phishing=5000, n_legit=5000)
    print(f"\n[+] Dataset shape: {X.shape}  |  Phishing: {sum(y)}  Legitimate: {len(y)-sum(y)}")

    # Persist feature matrix for notebook use
    feature_df = X.copy()
    feature_df["label"] = y.values
    feature_df.to_csv(DATA_DIR / "features.csv", index=False)
    print(f"[+] Feature matrix saved → DataFiles/features.csv")

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Scaler (for SVM & AE)
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Train all models
    results = []
    results.append(train_decision_tree(X_train, y_train, X_test, y_test))
    results.append(train_random_forest(X_train, y_train, X_test, y_test))
    results.append(train_xgboost(X_train, y_train, X_test, y_test))
    results.append(train_autoencoder(X_train, y_train, X_test, y_test, scaler))
    results.append(train_svm(X_train, y_train, X_test, y_test, scaler))

    # Save metrics
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("   RESULTS SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results)[["model","accuracy","precision","recall","f1_score","roc_auc"]]
    print(summary.to_string(index=False))
    print("\n[✓] All models saved to /models/")


if __name__ == "__main__":
    main()