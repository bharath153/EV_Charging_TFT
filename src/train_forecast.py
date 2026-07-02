"""
EV Charging Load Forecaster — Temporal Fusion Transformer (simulated)
======================================================================
Forecasts EV charging demand at a depot/fleet charging station
using time-series features and autoregressive gradient boosting
as a proxy for the Temporal Fusion Transformer architecture.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, json
from datetime import datetime, timedelta

np.random.seed(55)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Generate 2 years of hourly charging data ──
hours_total = 24 * 730   # 2 years
t = np.arange(hours_total)
hour_of_day   = t % 24
day_of_week   = (t // 24) % 7
week_of_year  = (t // (24*7)) % 52

# Charging pattern: commuter fleet (peaks 7-9 AM, 5-7 PM)
morning_peak = np.exp(-0.5*((hour_of_day - 7.5)/1.2)**2)
evening_peak = np.exp(-0.5*((hour_of_day - 18.0)/1.5)**2)
weekend_factor = np.where((day_of_week >= 5), 0.35, 1.0)
seasonal = 1 + 0.12 * np.sin(2*np.pi * week_of_year / 52)

# kW demand
demand = (
    25 * morning_peak
    + 40 * evening_peak
    + 8                              # baseline (overnight slow charge)
    + 5 * np.sin(2*np.pi*t/8760)   # annual seasonality
) * weekend_factor * seasonal + np.random.normal(0, 3, hours_total)
demand = np.clip(demand, 2, 120)

# ── Feature Engineering (TFT-inspired) ──
def make_features(demand, t, lookback=24):
    X, y = [], []
    for i in range(lookback, len(demand)):
        feats = list(demand[i-lookback:i])          # past 24h lags
        feats += [hour_of_day[i], day_of_week[i]]   # time covariates
        feats += [np.mean(demand[i-24:i]),           # 24h rolling mean
                  np.std(demand[i-24:i]),            # 24h rolling std
                  demand[i-168] if i >= 168 else demand[i]]  # same hour last week
        X.append(feats); y.append(demand[i])
    return np.array(X), np.array(y)

X, y = make_features(demand, t)
split = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=5, subsample=0.8, random_state=42)
gbr.fit(X_tr, y_tr)
y_pred = gbr.predict(X_te)

r2   = r2_score(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
mae  = mean_absolute_error(y_te, y_pred)
mape = np.mean(np.abs((y_te - y_pred)/y_te)) * 100
print(f"R²={r2:.4f}  RMSE={rmse:.3f} kW  MAE={mae:.3f} kW  MAPE={mape:.2f}%")
json.dump({"R2": round(r2,4), "RMSE_kW": round(rmse,3), "MAPE_%": round(mape,2)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — 7-day forecast vs actual
n_show = 24*7
x_show = np.arange(n_show)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x_show, y_te[:n_show], color='#1976D2', lw=1.5, label='Actual Demand')
ax.plot(x_show, y_pred[:n_show], color='#FF5722', lw=1.5, ls='--', label='TFT Forecast')
ax.fill_between(x_show, y_pred[:n_show]-rmse*1.96, y_pred[:n_show]+rmse*1.96,
                alpha=0.15, color='#FF5722', label='95% Prediction Interval')
ax.set_xlabel('Hour of Test Period', fontsize=11)
ax.set_ylabel('Charging Load (kW)', fontsize=11)
ax.set_title('EV Fleet Charging Load Forecast — 7-Day Window', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_forecast_7day.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Hourly average demand heatmap (hour × day of week)
avg_matrix = np.zeros((24, 7))
for h in range(24):
    for d in range(7):
        mask = (hour_of_day == h) & (day_of_week == d)
        avg_matrix[h, d] = demand[mask].mean() if mask.sum() > 0 else 0

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(avg_matrix, ax=ax, cmap='YlOrRd',
            xticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
            yticklabels=[f'{h:02d}:00' for h in range(24)],
            cbar_kws={'label':'Avg Load (kW)'})
ax.set_xlabel('Day of Week', fontsize=11); ax.set_ylabel('Hour of Day', fontsize=11)
ax.set_title('EV Charging Load Heatmap — Hour × Day', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_load_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Scatter + residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('EV Load Forecaster — Model Diagnostics', fontsize=14, fontweight='bold')
ax = axes[0]
ax.scatter(y_te[:2000], y_pred[:2000], alpha=0.25, s=12, color='#009688', edgecolors='none')
lims = [y_te.min()-2, y_te.max()+2]
ax.plot(lims, lims, 'r--', lw=2); ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual Load (kW)', fontsize=11); ax.set_ylabel('Predicted Load (kW)', fontsize=11)
ax.set_title(f'Predicted vs Actual  |  R² = {r2:.4f}', fontsize=11)
ax = axes[1]
resid = y_te - y_pred
ax.hist(resid, bins=50, color='#009688', alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', lw=2, ls='--')
ax.axvline(resid.mean(), color='red', lw=1.5, label=f'Mean = {resid.mean():.2f} kW')
ax.set_xlabel('Residual (kW)', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Residual Distribution  |  MAPE = {mape:.2f}%', fontsize=11); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Peak demand by hour
avg_by_hour = np.array([demand[hour_of_day==h].mean() for h in range(24)])
std_by_hour = np.array([demand[hour_of_day==h].std()  for h in range(24)])
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(24), avg_by_hour, color='#1565C0', alpha=0.8, edgecolor='white', label='Mean Load')
ax.errorbar(range(24), avg_by_hour, yerr=std_by_hour, fmt='none', color='black', capsize=3)
ax.set_xlabel('Hour of Day', fontsize=11); ax.set_ylabel('Avg Charging Load (kW)', fontsize=11)
ax.set_title('Diurnal Load Profile — Fleet Charging Station', fontsize=13, fontweight='bold')
ax.set_xticks(range(24)); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_diurnal_profile.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")
