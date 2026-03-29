"""
evaluate_model.py
=================
Evaluates the trained XGBoost model using our custom 22-feature pipeline.

Loads the saved `model.pkl` and `scaler.pkl`.
Prints a full classification report (Accuracy, Precision, Recall, F1).
Generates `confusion_matrix.png` and `feature_importance.png`.
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# We use the feature columns locally defined by our Python URL Extractor
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import MODEL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_test_data(file_path: str = 'test_data_for_eval.csv') -> pd.DataFrame:
    """Load test dataset saved during training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Run train_model.py first.")
    df = pd.read_csv(file_path)
    return df

def evaluate():
    """Evaluate the model using full classification report, CM, and feature importance."""
    logger.info("Loading model and test dataset...")
    
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        logger.error("model.pkl or scaler.pkl missing. Please run train_model.py first.")
        return
        
    model = joblib.load('model.pkl')
    test_df = load_test_data()
    
    X_test = test_df[MODEL_FEATURE_COLUMNS].copy()
    y_test = test_df['label']
    
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Evaluation (22 Features Pipeline) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'], 
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    logger.info("Saved confusion_matrix.png")
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = min(10, len(MODEL_FEATURE_COLUMNS))
    sorted_features = [MODEL_FEATURE_COLUMNS[i] for i in indices]
    sorted_importances = importances[indices]
    
    print("\nTop 10 Most Important Features:")
    for i in range(top_n):
        print(f"{i+1}. {sorted_features[i]}: {sorted_importances[i]:.4f}")
        
    sns.barplot(x=sorted_importances[:top_n], y=sorted_features[:top_n], palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    logger.info("Saved feature_importance.png")
    
if __name__ == "__main__":
    evaluate()
