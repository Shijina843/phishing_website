# Phishing URL Detection System

A complete Machine Learning based phishing detection pipeline utilizing Structural URL extraction, WHOIS intelligence, direct RDAP queries, and DNS configuration checks.

Includes a fast Flask backend with an animated TailwindCSS frontend user interface. The model is an XGBoost Classifier trained on a composite feature vector.

## Features Built

1. **Layer 1: URL Preprocessing** - URL normalization, shannon entropy, special chars, depth parsing (tldextract).
2. **Layer 2: Whitelist Handling** - Local Top 1M domains short-circuit.
3. **Layer 3: WHOIS Intel** - Registration dates, privacy shielding.
4. **Layer 4: RDAP Direct Querying** - Replaces paid APIs. Fallbacks to WHOIS.
5. **Layer 5: DNS Verification** - A, MX, NS records.
6. **Layer 6: Feature Unification** - Scaling applied only to continuous columns.
7. **Layer 7: Training Pipeline** - `train_model.py` that processes local multi-file datasets and exports `joblib` artifacts.
8. **Layer 8: Evaluation Script** - Confusion matrices, metric scores, and feature importance output.
9. **Layer 9: Risk Engine** - Custom rule engine modifying final system outputs.
10. **Layer 10 & 11: Application** - Fast endpoint with SQLite SQLite historical logging and modern Tailwind styling.

## How to Run

1. Change directory to the newly created project:
   ```bash
   cd phishing-detector
   ```
2. Install the rigid package requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script on the sample dataset (this will create `model.pkl` and `scaler.pkl` in `model/`):
   ```bash
   python model/train_model.py
   ```
4. Run the evaluation to generate graph artifacts (`confusion_matrix.png`, `feature_importance.png` inside the root):
   ```bash
   python model/evaluate_model.py
   ```
5. Launch the Flask App:
   ```bash
   python app.py
   ```
6. Open your browser and navigate to exactly: `http://localhost:5000` or `http://127.0.0.1:5000`
