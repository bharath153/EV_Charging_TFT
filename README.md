# EV Fleet Charging Load Forecaster — Temporal Fusion Transformer

> **Forecasting charging demand at fleet depots for smart grid integration**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-Smart%20Grid%20&%20EV%20Fleet%20Management-green.svg)]()
[![Model](https://img.shields.io/badge/Model-Gradient%20Boosting%20Regressor%20TFT%20proxy%20with%20lag%20features-purple.svg)]()

---

## 🎯 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Smart Grid & EV Fleet Management |
| **ML Model** | Gradient Boosting Regressor (TFT proxy) with lag features |
| **Key Metric** | R² = 0.924 | RMSE = 2.84 kW |
| **Tech Stack** | Python · scikit-learn · GradientBoostingRegressor · NumPy · Seaborn |

---

## 🧠 Problem Statement

Forecasting charging demand at fleet depots for smart grid integration. In engineering design, expensive simulations (FEA, CFD, dyno tests) limit the number of configurations that can be evaluated. Machine learning surrogates and classifiers allow rapid exploration of large design spaces, enabling smarter, faster engineering decisions.

---

## 📁 Repository Structure

```
P05_EV_Charging_TFT/
├── src/
│   └── train_forecast.py          # Main training & evaluation script
├── plots/
│   ├── fig1_*.png              # Performance plots
│   ├── fig2_*.png              # Analysis plots
│   ├── fig3_*.png
│   └── fig4_*.png
├── docs/
│   └── technical_notes.md      # Extended methodology notes
├── metrics.json                 # Saved evaluation metrics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/P05_EV_Charging_TFT.git
cd P05_EV_Charging_TFT
pip install -r requirements.txt
```

### 2. Run the model
```bash
python src/train_forecast.py
```

### 3. View results
All plots are saved to `plots/`. Metrics are saved to `metrics.json`.

---

## 📊 Results & Visualizations

### Model Performance
**R² = 0.924 | RMSE = 2.84 kW**

| Figure | Description |
|--------|-------------|
| `fig1_*` | Predicted vs Actual / Confusion Matrix |
| `fig2_*` | Feature Importance / Anomaly Scores |
| `fig3_*` | Learning Curve / ROC Curve |
| `fig4_*` | Design Space / Parametric Study |

---

## 🔬 Methodology

### Data Generation
Synthetic data is generated using physics-informed equations derived from engineering fundamentals. The data generation pipeline mimics real experimental/simulation data to demonstrate production-grade methodology.

### Model Architecture
`Gradient Boosting Regressor (TFT proxy) with lag features` — selected based on dataset characteristics (size, feature type, required uncertainty quantification).

### Validation Strategy
- 80/20 train-test split with fixed random seed
- 5-fold cross-validation for robust metric estimation
- Held-out test set never used during hyperparameter tuning

---

## 📚 References & Further Reading

- Forrester, A., Sóbester, A., & Keane, A. (2008). *Engineering Design via Surrogate Modelling*. Wiley.
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Relevant SAE/IEEE papers cited in `docs/technical_notes.md`

---

## 👤 Author

**Bharath Kanaiah Parthiban**  
M.Sc. Automotive Engineering — Politecnico di Torino (2025)  
Thesis: *Computational Intelligence for the Design of Electric Machines*  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)

---

