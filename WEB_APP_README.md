# 🔍 Phishing URL Detection Web Application

## Quick Start

### 1. Install Flask (if not already installed)
```bash
pip install flask
```

### 2. Run the Web Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

---

## Features

✅ **Web Interface** - Beautiful, responsive design  
✅ **Real-time Predictions** - Check URLs instantly  
✅ **Confidence Display** - See prediction confidence percentage  
✅ **Example URLs** - Try phishing and legitimate examples  
✅ **85.8% Accuracy** - Trained on 10,000 URLs (5,000 phishing + 5,000 legitimate)  

---

## How to Use

### Method 1: Enter Features Manually
1. Copy the features in JSON format
2. Paste into the input field
3. Click "Check"

**Example:**
```json
{
  "Have_IP": 1,
  "Have_At": 0,
  "URL_Length": 1,
  "URL_Depth": 0,
  "Redirection": 1,
  "https_Domain": 1,
  "Tiny_URL": 1,
  "Prefix/Suffix": 1,
  "DNS_Record": 0,
  "Web_Traffic": 0,
  "Domain_Age": 0,
  "Domain_End": 0,
  "iFrame": 1,
  "Mouse_Over": 0,
  "Right_Click": 0
}
```

### Method 2: Try Examples
Click on any example URL at the bottom - it will be marked as **Phishing** or **Legitimate**

---

## API Endpoints

### `/api/predict` (POST)
Predict if a URL is phishing

**Request:**
```json
{
  "features": {
    "Have_IP": 1,
    "Have_At": 0,
    ...
  }
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "Phishing",
  "confidence": 99.45,
  "is_phishing": true,
  "primary_model": "Random Forest"
}
```

---

### `/api/models` (GET)
Get available models and feature names

**Response:**
```json
{
  "models": ["Decision Tree", "Random Forest", "XGBoost", "Autoencoder NN", "SVM"],
  "feature_names": ["Have_IP", "Have_At", ...],
  "metrics": [...]
}
```

---

### `/api/example` (GET)
Get example URLs from dataset

**Response:**
```json
{
  "phishing_examples": ["eevee.tv", "appleid.apple.com-sa.pm", ...],
  "legitimate_examples": ["graphicriver.net", "ecnavi.jp", ...]
}
```

---

## Project Structure

```
phishingwebsite/
├── app.py                    # Flask app
├── predict.py               # Prediction module
├── train_models.py          # Training script
├── test_predictions.py      # Test script
├── requirements.txt         # Dependencies
├── models/                  # Trained models
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── autoencoder_nn.keras
│   └── svm.pkl
├── DataFiles/               # Dataset
│   ├── phishing_urls.csv
│   ├── legitimate_urls.csv
│   └── features.csv
└── templates/               # HTML templates
    └── index.html
```

---

## Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | **85.80%** | **84.67%** | **91.16%** |
| Decision Tree | 85.75% | 84.60% | 91.09% |
| XGBoost | 85.60% | 84.38% | 91.03% |
| Autoencoder NN | 83.70% | 81.44% | 89.92% |
| SVM | 82.35% | 79.05% | 87.37% |

---

## Features Used (15 total)

1. **Have_IP** - URL contains IP address
2. **Have_At** - URL contains @ symbol
3. **URL_Length** - Length of URL
4. **URL_Depth** - Depth of URL
5. **Redirection** - URL contains redirection
6. **https_Domain** - Uses HTTPS
7. **Tiny_URL** - Uses URL shortening service
8. **Prefix/Suffix** - Domain contains hyphens
9. **DNS_Record** - DNS record exists
10. **Web_Traffic** - Website has web traffic
11. **Domain_Age** - Domain age
12. **Domain_End** - Domain expiration
13. **iFrame** - Contains iFrame
14. **Mouse_Over** - Uses mouse-over event
15. **Right_Click** - Disables right-click

---

## Troubleshooting

### Issue: "Port 5000 already in use"
Change port in app.py:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: "Module not found"
Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Models not found
Run training first:
```bash
python train_models.py
```

---

## Future Enhancements

- [ ] Extract features from actual URLs automatically
- [ ] Upload CSV file with multiple URLs
- [ ] Show all 5 models' predictions
- [ ] Batch prediction capability
- [ ] API key authentication
- [ ] Result history/statistics
- [ ] Deploy to cloud (Heroku, AWS, etc.)

---

**Created:** March 23, 2026  
**Model Accuracy:** 85.8%  
**Training Dataset:** 10,000 URLs
