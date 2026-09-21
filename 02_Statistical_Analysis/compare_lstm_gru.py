# ==========================================================
# Combine LSTM and GRU results and compare model performance.
#
# Run this script after both model-training scripts have finished. The script
# aligns the independent test predictions by date, calculates model-to-model
# comparisons, creates combined tables, and generates publication figures.
# ==========================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication figure resolution (PNG). PDF is vector, resolution-independent.
DPI = 1200
import seaborn as sns
from scipy.stats import norm

# ----------------------------------------------------------
# 0. Portable paths
# ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
parser = argparse.ArgumentParser(description="Compare LSTM and GRU results.")
parser.add_argument("--lstm-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_LSTM_80_20_SINGLE_RUN_FINAL"))
parser.add_argument("--gru-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_GRU_80_20_SINGLE_RUN_FINAL"))
parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_Comparison_SINGLE_RUN_FINAL"))
args = parser.parse_args()

LSTM_RESULTS_DIR = os.path.abspath(args.lstm_dir)
GRU_RESULTS_DIR  = os.path.abspath(args.gru_dir)
OUT_DIR          = os.path.abspath(args.output_dir)
PLOTS_DIR        = os.path.join(OUT_DIR, "Figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. Load summaries
# ----------------------------------------------------------
LSTM_CSV = os.path.join(LSTM_RESULTS_DIR, "LSTM_results_summary_SINGLE_RUN.csv")
GRU_CSV  = os.path.join(GRU_RESULTS_DIR,  "GRU_results_summary_SINGLE_RUN.csv")

if not os.path.isfile(LSTM_CSV):
    raise FileNotFoundError(f"LSTM summary not found: {LSTM_CSV}")
if not os.path.isfile(GRU_CSV):
    raise FileNotFoundError(f"GRU summary not found: {GRU_CSV}")

lstm_df = pd.read_csv(LSTM_CSV)
gru_df  = pd.read_csv(GRU_CSV)

print(f"LSTM rows: {len(lstm_df)}")
print(f"GRU rows:  {len(gru_df)}")

# ----------------------------------------------------------
# 2. Diebold–Mariano: LSTM vs GRU (need prediction CSVs)
# ----------------------------------------------------------
def diebold_mariano_test(y_true, pred1, pred2, power=2, h=1):
    y_true = np.asarray(y_true, dtype=float)
    pred1  = np.asarray(pred1,  dtype=float)
    pred2  = np.asarray(pred2,  dtype=float)
    e1 = y_true - pred1
    e2 = y_true - pred2
    d = (np.abs(e1) - np.abs(e2)) if power == 1 else (e1 ** 2 - e2 ** 2)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    d_mean = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    max_lag = max(0, int(h) - 1)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        weight = 1.0 - (lag / (max_lag + 1))
        long_run_var += 2.0 * weight * cov
    if long_run_var <= 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(long_run_var / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def significance_label(p):
    if np.isnan(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


STATIONS = ["Tan_Chau", "Vam_Nao", "Can_Tho"]
HORIZONS = [1, 24, 48, 72, 96, 120, 144, 168]

dm_records = []

for station in STATIONS:
    for horizon in HORIZONS:
        # Load prediction CSVs
        lstm_pred_file = os.path.join(
            LSTM_RESULTS_DIR, station,
            f"LSTM_{station}_horizon_{horizon}h_predictions.csv"
        )
        gru_pred_file = os.path.join(
            GRU_RESULTS_DIR, station,
            f"GRU_{station}_horizon_{horizon}h_predictions.csv"
        )

        if not (os.path.isfile(lstm_pred_file) and os.path.isfile(gru_pred_file)):
            print(f"  Skipping {station} H={horizon}h (prediction files missing)")
            continue

        lstm_preds = pd.read_csv(lstm_pred_file)
        gru_preds  = pd.read_csv(gru_pred_file)

        # Align by Date
        merged = pd.merge(
            lstm_preds[["Date", "Actual", "LSTM_Predicted"]],
            gru_preds[["Date", "GRU_Predicted"]],
            on="Date",
            how="inner"
        )
        if len(merged) == 0:
            continue

        y_true = merged["Actual"].values
        y_lstm = merged["LSTM_Predicted"].values
        y_gru  = merged["GRU_Predicted"].values

        dm_stat, p_val = diebold_mariano_test(y_true, y_lstm, y_gru, power=2, h=horizon)

        dm_records.append({
            "Station": station,
            "Horizon_h": horizon,
            "N_Samples": len(merged),
            "DM_LSTM_vs_GRU": round(dm_stat, 4) if np.isfinite(dm_stat) else np.nan,
            "p_LSTM_vs_GRU":  round(p_val, 4)  if np.isfinite(p_val)  else np.nan,
            "Signif_LSTM_vs_GRU": significance_label(p_val),
        })
        print(f"  {station} H={horizon:3d}h | DM={dm_stat:.4f} | p={p_val:.4f} [{significance_label(p_val)}]")

dm_df = pd.DataFrame(dm_records)

# ----------------------------------------------------------
# 3. Build combined table
# ----------------------------------------------------------
# Merge LSTM and GRU summaries on Station + Horizon
lstm_sub = lstm_df[["Station", "Horizon_h", "Lag", "Trainable_Params", "Training_Time_s",
                    "LSTM_R2", "LSTM_NSE", "LSTM_KGE", "LSTM_RMSE", "LSTM_MAE",
                    "DM_LSTM_vs_Naive", "p_LSTM_vs_Naive", "Signif_LSTM_vs_Naive",
                    "LSTM_Flood_RMSE", "LSTM_Flood_MAE", "LSTM_Flood_Bias",
                    "LSTM_Peak_Error",
                    "LSTM_Resid_Bias", "LSTM_Resid_SD", "LSTM_LjungBox_p", "LSTM_Shapiro_p"]]

gru_sub = gru_df[["Station", "Horizon_h",
                  "GRU_R2", "GRU_NSE", "GRU_KGE", "GRU_RMSE", "GRU_MAE",
                  "DM_GRU_vs_Naive", "p_GRU_vs_Naive", "Signif_GRU_vs_Naive",
                  "GRU_Flood_RMSE", "GRU_Flood_MAE", "GRU_Flood_Bias",
                  "GRU_Peak_Error",
                  "GRU_Resid_Bias", "GRU_Resid_SD", "GRU_LjungBox_p", "GRU_Shapiro_p"]]

combined = pd.merge(lstm_sub, gru_sub, on=["Station", "Horizon_h"], how="outer")
combined = pd.merge(combined, dm_df, on=["Station", "Horizon_h"], how="outer")

# Also bring Naïve metrics (same in both files, take from LSTM)
naive_cols = ["Naive_R2", "Naive_NSE", "Naive_KGE", "Naive_RMSE", "Naive_MAE",
              "Naive_Flood_RMSE", "Naive_Flood_MAE", "Naive_Flood_Bias",
              "Naive_Peak_Error"]
naive_sub = lstm_df[["Station", "Horizon_h"] + naive_cols]
combined = pd.merge(combined, naive_sub, on=["Station", "Horizon_h"], how="outer")

# Reorder columns logically
col_order = [
    "Station", "Horizon_h", "Lag", "Trainable_Params", "Training_Time_s",
    "LSTM_R2", "GRU_R2", "Naive_R2",
    "LSTM_NSE", "GRU_NSE", "Naive_NSE",
    "LSTM_KGE", "GRU_KGE", "Naive_KGE",
    "LSTM_RMSE", "GRU_RMSE", "Naive_RMSE",
    "LSTM_MAE", "GRU_MAE", "Naive_MAE",
    "DM_LSTM_vs_GRU", "p_LSTM_vs_GRU", "Signif_LSTM_vs_GRU",
    "DM_LSTM_vs_Naive", "p_LSTM_vs_Naive", "Signif_LSTM_vs_Naive",
    "DM_GRU_vs_Naive", "p_GRU_vs_Naive", "Signif_GRU_vs_Naive",
    "LSTM_Flood_RMSE", "GRU_Flood_RMSE", "Naive_Flood_RMSE",
    "LSTM_Flood_MAE", "GRU_Flood_MAE", "Naive_Flood_MAE",
    "LSTM_Flood_Bias", "GRU_Flood_Bias", "Naive_Flood_Bias",
    "LSTM_Peak_Error", "GRU_Peak_Error", "Naive_Peak_Error",
    "LSTM_Resid_Bias", "GRU_Resid_Bias",
    "LSTM_Resid_SD", "GRU_Resid_SD",
    "LSTM_LjungBox_p", "GRU_LjungBox_p",
    "LSTM_Shapiro_p", "GRU_Shapiro_p",
]
combined = combined[[c for c in col_order if c in combined.columns]]

# Save
combined.to_csv(os.path.join(OUT_DIR, "Combined_LSTM_GRU_Naive_Comparison.csv"), index=False)
combined.to_excel(os.path.join(OUT_DIR, "Combined_LSTM_GRU_Naive_Comparison.xlsx"), index=False)
print(f"\nSaved combined table: {os.path.join(OUT_DIR, 'Combined_LSTM_GRU_Naive_Comparison.csv')}")

# ----------------------------------------------------------
# 4. Compact comparison tables for manuscript
# ----------------------------------------------------------
# Table A: Overall metrics (NSE, RMSE, MAE, R², KGE)
metric_cols = ["Station", "Horizon_h", "Lag"]
for m in ["R2", "NSE", "KGE", "RMSE", "MAE"]:
    metric_cols += [f"LSTM_{m}", f"GRU_{m}", f"Naive_{m}"]

compact_metrics = combined[metric_cols].copy()
compact_metrics.to_csv(os.path.join(OUT_DIR, "Table_Overall_Metrics.csv"), index=False)
compact_metrics.to_excel(os.path.join(OUT_DIR, "Table_Overall_Metrics.xlsx"), index=False)

# Table B: Statistical tests
dm_cols = ["Station", "Horizon_h", "Lag",
           "DM_LSTM_vs_GRU", "p_LSTM_vs_GRU", "Signif_LSTM_vs_GRU",
           "DM_LSTM_vs_Naive", "p_LSTM_vs_Naive", "Signif_LSTM_vs_Naive",
           "DM_GRU_vs_Naive", "p_GRU_vs_Naive", "Signif_GRU_vs_Naive"]
compact_dm = combined[[c for c in dm_cols if c in combined.columns]].copy()
compact_dm.to_csv(os.path.join(OUT_DIR, "Table_Statistical_Tests.csv"), index=False)
compact_dm.to_excel(os.path.join(OUT_DIR, "Table_Statistical_Tests.xlsx"), index=False)

# Table C: Flood metrics
flood_cols = ["Station", "Horizon_h", "Lag"]
for m in ["Flood_RMSE", "Flood_MAE", "Flood_Bias", "Peak_Error"]:
    flood_cols += [f"LSTM_{m}", f"GRU_{m}", f"Naive_{m}"]
compact_flood = combined[[c for c in flood_cols if c in combined.columns]].copy()
compact_flood.to_csv(os.path.join(OUT_DIR, "Table_Flood_Metrics.csv"), index=False)
compact_flood.to_excel(os.path.join(OUT_DIR, "Table_Flood_Metrics.xlsx"), index=False)

# Table D: Residual diagnostics
resid_cols = ["Station", "Horizon_h", "Lag",
              "LSTM_Resid_Bias", "GRU_Resid_Bias",
              "LSTM_Resid_SD", "GRU_Resid_SD",
              "LSTM_LjungBox_p", "GRU_LjungBox_p",
              "LSTM_Shapiro_p", "GRU_Shapiro_p"]
compact_resid = combined[[c for c in resid_cols if c in combined.columns]].copy()
compact_resid.to_csv(os.path.join(OUT_DIR, "Table_Residual_Diagnostics.csv"), index=False)
compact_resid.to_excel(os.path.join(OUT_DIR, "Table_Residual_Diagnostics.xlsx"), index=False)

print("Saved manuscript tables:")
print("  • Table_Overall_Metrics")
print("  • Table_Statistical_Tests")
print("  • Table_Flood_Metrics")
print("  • Table_Residual_Diagnostics")

# ----------------------------------------------------------
# 5. Comparison plots
# ----------------------------------------------------------
# Plot 1: NSE vs Horizon for all stations (line plot)
plt.figure(figsize=(12, 6))
for station in STATIONS:
    sub = combined[combined["Station"] == station].sort_values("Horizon_h")
    plt.plot(sub["Horizon_h"], sub["LSTM_NSE"], marker="o", label=f"{station} LSTM")
    plt.plot(sub["Horizon_h"], sub["GRU_NSE"],  marker="s", linestyle="--", label=f"{station} GRU")
    plt.plot(sub["Horizon_h"], sub["Naive_NSE"], marker="^", linestyle=":", alpha=0.6, label=f"{station} Naïve")

plt.xlabel("Forecast Horizon (h)", fontsize=12)
plt.ylabel("NSE", fontsize=12)
plt.title("Nash–Sutcliffe Efficiency vs. Forecast Horizon", fontsize=13, fontweight="bold")
plt.legend(ncol=3, fontsize=8, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "NSE_vs_Horizon_comparison.png"), dpi=DPI)
plt.savefig(os.path.join(PLOTS_DIR, "NSE_vs_Horizon_comparison.pdf"))
plt.close()

# Plot 2: RMSE vs Horizon
plt.figure(figsize=(12, 6))
for station in STATIONS:
    sub = combined[combined["Station"] == station].sort_values("Horizon_h")
    plt.plot(sub["Horizon_h"], sub["LSTM_RMSE"], marker="o", label=f"{station} LSTM")
    plt.plot(sub["Horizon_h"], sub["GRU_RMSE"],  marker="s", linestyle="--", label=f"{station} GRU")
    plt.plot(sub["Horizon_h"], sub["Naive_RMSE"], marker="^", linestyle=":", alpha=0.6, label=f"{station} Naïve")

plt.xlabel("Forecast Horizon (h)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE vs. Forecast Horizon", fontsize=13, fontweight="bold")
plt.legend(ncol=3, fontsize=8, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "RMSE_vs_Horizon_comparison.png"), dpi=DPI)
plt.savefig(os.path.join(PLOTS_DIR, "RMSE_vs_Horizon_comparison.pdf"))
plt.close()

# Plot 3: Heatmap of NSE (Station × Horizon) for LSTM
pivot_lstm = combined.pivot_table(index="Station", columns="Horizon_h", values="LSTM_NSE", aggfunc="first")
pivot_gru  = combined.pivot_table(index="Station", columns="Horizon_h", values="GRU_NSE",  aggfunc="first")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.heatmap(pivot_lstm, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=axes[0], cbar_kws={"label": "NSE"})
axes[0].set_title("LSTM NSE", fontweight="bold")

sns.heatmap(pivot_gru, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=axes[1], cbar_kws={"label": "NSE"})
axes[1].set_title("GRU NSE", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "NSE_Heatmap_LSTM_vs_GRU.png"), dpi=DPI)
plt.savefig(os.path.join(PLOTS_DIR, "NSE_Heatmap_LSTM_vs_GRU.pdf"))
plt.close()

# Plot 4: Flood Peak Error comparison
flood_err = combined[["Station", "Horizon_h", "LSTM_Peak_Error", "GRU_Peak_Error", "Naive_Peak_Error"]].copy()
flood_err = flood_err.melt(id_vars=["Station", "Horizon_h"], var_name="Model", value_name="Peak_Error")
flood_err["Model"] = flood_err["Model"].str.replace("_Peak_Error", "")

plt.figure(figsize=(12, 6))
sns.barplot(data=flood_err, x="Horizon_h", y="Peak_Error", hue="Model", palette="muted")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Flood Peak Error by Horizon", fontsize=13, fontweight="bold")
plt.xlabel("Forecast Horizon (h)", fontsize=12)
plt.ylabel("Peak Error (Predicted − Observed)", fontsize=12)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "Flood_Peak_Error_comparison.png"), dpi=DPI)
plt.savefig(os.path.join(PLOTS_DIR, "Flood_Peak_Error_comparison.pdf"))
plt.close()

print(f"\nSaved comparison figures to: {PLOTS_DIR}")
print("\n=== COMBINATION & STATISTICAL COMPARISON COMPLETE ===")
