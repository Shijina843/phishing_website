# Phishing URL Detection System

A high-performance, Machine Learning based phishing detection pipeline utilizing Structural URL extraction, DNS configuration checks, direct RDAP queries, and WHOIS intelligence. 

This project incorporates a fast Flask backend with an animated TailwindCSS frontend user interface. The core detection engine relies on an XGBoost Classifier trained entirely on a composite vector of blazingly fast lexical and structural features, while network operations (WHOIS/RDAP/DNS) are creatively utilized for dynamic rule-based risk penalization and UI intelligence overlays.

## Features Built

The system architecture is designed across multiple modular layers to ensure reliability, performance, and accuracy:

1. **Layer 1: URL Preprocessing** - URL normalization, shannon entropy, special characters scanning, and structural depth parsing using `tldextract`.
2. **Layer 2: Whitelisting Engine** - Short-circuits the prediction by checking the Tranco Top 1 Million list, bypassing deep scans for mathematically proven benign domains.
3. **Layer 3: WHOIS Intelligence** - Time-bounded domain registration lookups (days since creation, expiration dates, country, registrar, privacy shielding).
4. **Layer 4: RDAP Direct Querying** - Replaces expensive paid APIs with pure `httpx` based direct HTTP queries to `rdap.org`, defaulting back to python-whois if RDAP is unresponsive.
5. **Layer 5: DNS Verification** - Active resolution of `A`, `MX`, and `NS` records using `dnspython` to verify domain authenticity.
6. **Layer 6: Feature Unification** - Continuous features are rigorously scaled, while categorical boolean flags (like `ip` or `suspicious_tld`) are maintained raw.
7. **Layer 7: Training Pipeline** - `train_model.py` maps features directly from `dataset/phishing_data.csv` to generate optimized `joblib` artifacts (`model.pkl`, `scaler.pkl`). This ensures the model runs efficiently without slow network dependency during inference.
8. **Layer 8: Evaluation Scripts** - Automatically produces evaluation metrics such as log-loss, `confusion_matrix.png`, and `feature_importance.png`.
9. **Layer 9: Dynamic Risk Engine** - A secondary heuristic engine defined in `app.py` that applies penalty factors. If WHOIS data is intentionally hidden/blocked or records are missing (e.g., domain age unknown), the engine dynamically adjusts the ultimate confidence score and flags the risk.
10. **Layer 10 & 11: Application Tier** - A rapid, single-page Flask endpoint connected to an SQLite database (`history.db`) for logging scans, wrapped in a polished Tailwind CSS UI.

## Detailed Folder & File Structure

* **`app.py`**
  The main entry point for the Flask backend. It defines API endpoints (`/predict`, `/history`) and the main UI route. It manages the risk flag engine, calculates penalty scores based on missing WHOIS values, sets the verdict (LEGITIMATE, PARTIALLY VULNERABLE, PHISHING), and logs all activity into the `history.db` SQLite database.

* **`model/`**
  Houses the ML intelligence, scripts, and pre-trained `.pkl` binaries.
  * **`feature_extractor.py`**: Blazing fast script to assemble 15 core URL features extracted purely top-down (e.g., URL length, hyphens, digits ratio, suspicious keywords). 
  * **`train_model.py`**: ML pipeline configured to ingest `dataset/phishing_data.csv` features and output an `XGBClassifier` with tuned hyperparameters (depth 6, 300 estimators).
  * **`evaluate_model.py`**: Evaluation module to export diagnostic plots from the model.
  * **`whois_intel.py`**: Safely sandboxed (5-second timeout) WHOIS extractor relying on `python-whois` and DNS checks using `dnspython`.
  * **`rdap_intel.py`**: Cutting-edge JSON extraction of domain data directly from Internet Registries' RDAP servers, offering speedier & cleaner metadata.
  * **`model.pkl` & `scaler.pkl`**: Serialized `scikit-learn` Standard Scaler and `XGBoost` model artifacts.

* **`dataset/`**
  Contains required datasets.
  * **`phishing_data.csv`**: Target feature dataset used for training the XGBoost model.
  * **`tranco_top1m.csv`**: Massive whitelist of the most visited internet domains for quick-pass optimization.

* **`templates/` & `static/`**
  Contains the application's Jinja2 HTML templates and static assets (CSS, JS). Specifically includes an interactive Tailwind UI for domain analysis.

## Machine Learning Methodology

The model relies exclusively on 15 core network-independent features to achieve near-instant inference capabilities:
`length_url`, `length_hostname`, `ip`, `nb_dots`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_and`, `nb_eq`, `nb_slash`, `nb_www`, `ratio_digits_url`, `ratio_digits_host`, `phish_hints`, `suspecious_tld`

Network queries (WHOIS & RDAP) act purely as a protective secondary layer modifying final confidence scores dynamically instead of lagging the AI algorithms.

## Prerequisites

- Python 3.9+ 
- Minimum memory: 1GB RAM (for ML arrays)

## Installation & Running Locally

1. **Clone and Enter the Directory**
   ```bash
   # Make sure you are inside the project folder
   cd phishing_website
   ```

2. **Install the Rigid Dependencies**
   Ensure you have a clean virtual environment optionally, then install exactly matched versions of libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Initial Model**
   Run the training pipeline (takes only seconds since it avoids network calls on individual rows). This will generate `model.pkl` and `scaler.pkl` inside the `model/` folder.
   ```bash
   python model/train_model.py
   ```

4. **Evaluate the Model (Optional)**
   Verify model success and generate artifact PNGs locally:
   ```bash
   python model/evaluate_model.py
   ```

5. **Start the Flask Server**
   ```bash
   python app.py
   ```

6. **Access UI**
   Open your browser and navigate to:
   * `http://localhost:5000`
   * or `http://127.0.0.1:5000`
