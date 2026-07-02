# Interview Q&A — EV Fleet Charging Load Forecaster — Temporal Fusion Transformer
## Gradient Boosting Regressor (TFT proxy) with lag features | Smart Grid & EV Fleet Management

### Q1: What is the Temporal Fusion Transformer (TFT)?

TFT (Lim et al., 2021, International Journal of Forecasting) is a transformer-based architecture specifically designed for multi-horizon time-series forecasting. Its key innovations are: (1) a gating mechanism to suppress irrelevant inputs, (2) separate handling of static, known-future, and unknown-future covariates, and (3) an attention mechanism for interpretable long-range dependency capture. In this project, a Gradient Boosting Regressor with engineered lag features approximates TFT-like behavior for a simpler implementation.

---

### Q2: What is valley filling and why does it matter?

Valley filling is a demand-side management strategy where EV charging is shifted to off-peak hours (typically 11 PM–6 AM) to reduce peak grid load and take advantage of cheaper, often cleaner overnight electricity. Load forecasting is essential for valley filling algorithms because you need to predict how much demand you will need to shift.

---

### Q3: How would you handle missing CAN/telemetry data?

Missing sensor data is handled via: (1) Forward-fill for short gaps (<5 min), (2) Interpolation for medium gaps, (3) Masking tokens in transformer architectures (native support), and (4) Indicator features flagging missing observations so the model learns to discount imputed values. Feature engineering for robustness to missing data is a core skill in production ML.

---

### Q4: What external datasets exist for EV charging load?

Public datasets include: ACN-Data (Caltech EV charging dataset, 1.4M sessions), BTC (Boulder Transportation Charging), and the EU Open Data Portal EV consumption logs. The UK National Grid ESO provides public half-hourly demand data. The PJM Interconnection in the US provides hourly load data covering 13 states.

---

### Q5: How does this relate to V2G (Vehicle-to-Grid)?

V2G allows EVs to return energy to the grid during peak demand. Accurate load forecasting is the prerequisite for V2G dispatch: you need to know which vehicles will be connected, when, and at what SoC, to decide how much energy they can safely return without compromising the next trip's range. Nissan (Leaf V2G pilot in the UK) and Volkswagen (WeCharge) are active in this space.

---

