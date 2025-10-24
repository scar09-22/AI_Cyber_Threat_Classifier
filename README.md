# AI-Driven Cyber Threat Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.99%25-brightgreen.svg)](.)

An **enterprise-grade machine learning system** for **real-time network intrusion detection** using the **CICIDS2017 dataset**.  
Includes **Explainable AI (SHAP)** and an **interactive Streamlit dashboard** for threat visualization and analysis.

---

## Overview

This project implements a **production-ready cyber threat detection system** that achieves **99.99% accuracy** with **zero false positives** on the CICIDS2017 benchmark dataset.  
It can detect various types of attacks such as **DDoS**, **Port Scans**, **Brute Force**, and **Web Exploits** in real-time.

### Key Features

- **High Performance** — 99.99% accuracy, 100% precision, 99.94% recall  
- **Zero False Positives** — No false alarms, full reliability  
- **Explainable AI** — SHAP integration for feature-level model interpretation  
- **Real-Time Detection** — Live and batch prediction modes  
- **Interactive Dashboard** — Streamlit app for monitoring and analytics  
- **Multiple Models** — RandomForest, XGBoost, and LightGBM comparison  

---

## Performance Metrics

| Metric | Score | Details |
|--------|-------|---------|
| **Accuracy** | 99.99% | 77,229 correct / 77,233 total |
| **Precision** | 100.00% | Zero false positives |
| **Recall** | 99.94% | Only 4 attacks missed out of 6,351 |
| **F1-Score** | 99.97% | Excellent balance |

### Confusion Matrix

|  | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | 70,882 | 0 |
| **Actual Attack** | 4 | 6,347 |

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset

- **Source**: University of New Brunswick  
- **Samples**: 400,000+ network flows  
- **Features**: 68 network flow characteristics  
- **Attack Types**: DDoS, PortScan, Brute Force, Web Attacks, DoS Slowloris, Bot, Infiltration  
- **Classes**: Binary classification (Benign vs Attack)

---

## Installation

### Prerequisites
- Python **3.10+**
- Minimum **8GB RAM**
- Kaggle account with API key

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/scar09-22/AI_Cyber_Threat_Classifier.git
cd AI_Cyber_Threat_Classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage

### Complete ML Pipeline

```bash
# 1. Download dataset
python3 scripts/data_acquisition.py

# 2. Preprocess data
python3 scripts/data_preprocessing.py

# 3. Train models
python3 scripts/model_training.py

# 4. Generate SHAP explanations
python3 scripts/explainability.py

# 5. Launch the interactive dashboard
streamlit run scripts/dashboard_app.py
```

---

## Technical Stack

| Layer | Technologies |
|-------|--------------|
| **ML / AI** | scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly |
| **Data Handling** | pandas, numpy |
| **Environment** | Python 3.10+ |

---

## Model Architecture

**RandomForest Classifier**
- n_estimators = 100  
- max_depth = 20  
- Training samples = 320,000+  
- Test samples = 77,233  

---

## Attack Detection Results

| Attack Type | Detection Rate |
|--------------|----------------|
| DDoS | 99.98% |
| PortScan | 100% |
| DoS Slowloris | 99.92% |
| FTP-Patator | 99.96% |
| Web Attacks | 100% |

---

## Author

**Shiva** — [@scar09-22](https://github.com/scar09-22)

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Canadian Institute for Cybersecurity (CICIDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)  
- [SHAP](https://github.com/slundberg/shap) for explainable AI  
- [Streamlit](https://streamlit.io) for dashboard visualization
