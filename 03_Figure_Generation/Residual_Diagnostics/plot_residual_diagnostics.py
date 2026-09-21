# ==========================================================
#  RESIDUAL DIAGNOSTICS - FULL VERSION (Figure 14 fix)
#  Addresses additional methodological requirements (analysis 2 & analysis 3):
#   1) Extended from a single input forecast file (1 station/1 horizon)
#      to ALL input forecast files at once (all stations x all horizons)
#   2) Added Bias (mean residual) - was missing
#   3) Activated Ljung-Box test (import was already present but
#      unused) - gives a quantitative p-value for autocorrelation
#      instead of relying on the ACF/PACF plots visually only
#   4) Added a 4th panel: Heteroscedasticity (residuals vs
#      predicted values + Spearman rho/p-value) - was completely
#      missing before
#   5) All four diagnostics saved to one summary CSV table across
#      every station/horizon, in addition to the per-file figures
#
#  HOW TO USE IN COLAB:
#  1. Run this script from the repository root or from a configured Python environment.
#  2. Upload all forecast files at once - every file that
#     has "Date", "Actual", "Predicted" columns, for every
#     station and every horizon (e.g. the files produced by your
#     LSTM_three_stations.py / GRU_three_stations.py scripts).
#  3. Each filename must contain the model (LSTM/GRU), the
#     station name, and the horizon, e.g.
#     "LSTM_forecast_Vam_Nao_horizon_24h.xlsx" - parsed
#     automatically, same convention as your other scripts.
#  4. For each file, a 4-panel Elsevier-style figure (1200 dpi)
#     is generated and saved; a summary CSV table across all
#     files is also produced.
#  5. All figures + the summary CSV are written to the configured output directory and archive.
#     automatically at the end.
# ==========================================================

import io
import os
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import argparse
from pathlib import Path

print("=== RESIDUAL DIAGNOSTICS - ALL STATIONS x ALL HORIZONS ===")

# ==========================================================
# ELSEVIER PUBLICATION STYLE (unchanged from your original)
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.linewidth": 2.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True
})

# -------------------------------
# FILENAME PARSING (Station / Model / Horizon)
# -------------------------------
STATION_ALIASES = {
    "vamnao": "Vam_Nao",
    "tanchau": "Tan_Chau",
    "cantho": "Can_Tho",
}

def parse_filename(fname):
    name_lower = fname.lower()
    if "lstm" in name_lower:
        model = "LSTM"
    elif "gru" in name_lower:
        model = "GRU"
    else:
        model = "Model"

    clean = name_lower.replace(" ", "").replace("_", "").replace("-", "")
    station = "UnknownStation"
    for alias, canonical in STATION_ALIASES.items():
        if alias in clean:
            station = canonical
            break

    horizon_match = re.search(r'(\d+)\s*h(?:our)?', name_lower)
    horizon = int(horizon_match.group(1)) if horizon_match else None

    return model, station, horizon

# ==========================================================
# PORTABLE INPUT / OUTPUT CONFIGURATION
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate residual diagnostics for all forecast files.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"))
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "Residual_Diagnostics"))
args = parser.parse_args()
INPUT_FILES = [p for p in Path(args.input_dir).resolve().rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
if not INPUT_FILES:
    raise ValueError(f"No forecast files found under {args.input_dir}")
OUTPUT_DIR = str(Path(args.output_dir).resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nFound {len(INPUT_FILES)} forecast files.\n")

summary_records = []

# ==========================================================
# MAIN LOOP: ONE 4-PANEL FIGURE PER FILE
# ==========================================================
for filename_path in INPUT_FILES:
    filename = str(filename_path)

    model, station, horizon = parse_filename(filename)
    print(f"Processing: {filename}  ->  Station={station}, Model={model}, Horizon={horizon}h")

    # -------------------------------
    # LOAD FILE
    # -------------------------------
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(filename)
        elif filename.lower().endswith(".xlsx"):
            df = pd.read_excel(filename)
        else:
            print(f"  [skip] Unsupported file format: {filename}")
            continue
    except Exception as e:
        print(f"  [skip] Could not read {filename}: {e}")
        continue

    # Determine the expected prediction column name based on the model
    expected_pred_col = None
    if model == "LSTM":
        expected_pred_col = "LSTM_Predicted"
    elif model == "GRU":
        expected_pred_col = "GRU_Predicted"
    # Fallback for generic 'Predicted' column if model-specific not found
    if expected_pred_col not in df.columns and "Predicted" in df.columns:
        expected_pred_col = "Predicted"

    if not {"Date", "Actual"}.issubset(df.columns) or expected_pred_col is None or expected_pred_col not in df.columns:
        print(f"  [skip] {filename} missing Date/Actual or a valid Predicted column "
              f"(expected: {expected_pred_col}, found: {list(df.columns)})")
        continue

    # -------------------------------
    # PREPROCESS
    # -------------------------------
    df['Date'] = pd.to_datetime(df['Date'])
    df['Residuals'] = df['Actual'] - df[expected_pred_col] # Use the identified prediction_col
    residuals = df['Residuals'].dropna()

    if len(residuals) < 20:
        print(f"  [skip] Not enough data points in {filename} ({len(residuals)} rows)")
        continue

    # -------------------------------
    # DIAGNOSTIC 1: BIAS (mean residual) - NEW, addresses additional methodological requirement
    # -------------------------------
    bias = residuals.mean()
    bias_std = residuals.std()

    # -------------------------------
    # DIAGNOSTIC 2: NORMALITY (Shapiro-Wilk) - already existed, kept as-is
    # -------------------------------
    shapiro_sample = residuals.sample(min(len(residuals), 5000), random_state=42)
    _, shapiro_p = stats.shapiro(shapiro_sample)

    # -------------------------------
    # DIAGNOSTIC 3: AUTOCORRELATION - Ljung-Box p-value, NEW
    # (the import already existed in your original code but was
    #  never actually called - this activates it)
    # -------------------------------
    lb_lags = min(24, len(residuals) // 5)
    lb_lags = max(lb_lags, 1)
    try:
        lb_result = acorr_ljungbox(residuals, lags=[lb_lags], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].values[0]
    except Exception as e:
        lb_pvalue = np.nan
        print(f"  [warn] Ljung-Box test failed for {filename}: {e}")

    # -------------------------------
    # DIAGNOSTIC 4: HETEROSCEDASTICITY - NEW, was completely missing
    # -------------------------------
    try:
        het_rho, het_p = spearmanr(np.abs(residuals), df.loc[residuals.index, expected_pred_col]) # Use the identified prediction_col
    except Exception as e:
        het_rho, het_p = np.nan, np.nan
        print(f"  [warn] Heteroscedasticity test failed for {filename}: {e}")

    summary_records.append({
        "Station": station, "Model": model, "Horizon": horizon,
        "N": len(residuals),
        "Bias_MeanResidual": bias, "Residual_Std": bias_std,
        "Shapiro_p": shapiro_p, "Normal_at_0.05": "Yes" if shapiro_p > 0.05 else "No",
        "LjungBox_p": lb_pvalue, "Autocorrelated_at_0.05": "Yes" if lb_pvalue < 0.05 else "No",
        "Heterosced_rho": het_rho, "Heterosced_p": het_p,
        "Heteroscedastic_at_0.05": "Yes" if het_p < 0.05 else "No",
    })

    # ==========================================================
    # FIGURE LAYOUT - 4 PANELS (added Panel 4: Heteroscedasticity)
    # ==========================================================
    fig = plt.figure(figsize=(11.2, 18.0), dpi=200)  # dpi lowered for speed; raise to 1200 for final export
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.05, 1], hspace=0.38)

    # -------------------- PANEL 1 — Residual Time Series --------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['Date'], df['Residuals'], color='#002147', lw=0.8, label='Residuals')
    ax1.axhline(0, color='#B22222', linestyle='--', lw=1.5, label='Zero Line')
    ax1.set_title(f'Residual Time Series - {station} - {model} - {horizon}h',
                  fontsize=16, fontweight='bold', pad=8)
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Residual (Observed - Predicted)', fontweight='bold')
    ax1.legend(frameon=True, edgecolor='black', fontsize=12)
    ax1.grid(False)
    ax1.text(0.02, 0.90, f'Bias (mean residual) = {bias:.3f}',
              transform=ax1.transAxes, fontsize=11, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # -------------------- PANEL 2 — Histogram + Gaussian Fit --------------------
    ax2 = fig.add_subplot(gs[1])
    sns.histplot(residuals, kde=False, stat="density", color='#4C72B0',
                 edgecolor='black', linewidth=0.6, label='Residual Distribution', ax=ax2)
    mu, std = stats.norm.fit(residuals)
    x = np.linspace(*ax2.get_xlim(), 400)
    ax2.plot(x, stats.norm.pdf(x, mu, std), color='#B22222', lw=2.5, label='Gaussian Fit')
    ax2.set_title('Residual Distribution with Gaussian Fit',
                  fontsize=16, fontweight='bold', pad=8)
    ax2.set_xlabel('Residual Magnitude', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.legend(frameon=True, edgecolor='black', fontsize=12)
    ax2.text(0.05, 0.88, f'Shapiro-Wilk p = {shapiro_p:.2e}',
              transform=ax2.transAxes, fontsize=12, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))
    ax2.grid(False)

    # -------------------- PANEL 3 — ACF & PACF (+ Ljung-Box) --------------------
    gs_sub = gs[2].subgridspec(1, 2, wspace=0.30)
    ax3a = fig.add_subplot(gs_sub[0])
    ax3b = fig.add_subplot(gs_sub[1])

    plot_acf(residuals, lags=40, ax=ax3a, color='#002147',
             vlines_kwargs={"colors": '#002147'}, alpha=0.05)
    ax3a.set_title('Autocorrelation Function (ACF)', fontsize=15, fontweight='bold', pad=6)
    ax3a.set_xlabel('Lag (hours)', fontweight='bold')
    ax3a.set_ylabel('Correlation', fontweight='bold')
    ax3a.grid(False)
    ax3a.text(0.05, 0.90, f'Ljung-Box p = {lb_pvalue:.2e}',
              transform=ax3a.transAxes, fontsize=10, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plot_pacf(residuals, lags=40, ax=ax3b, color='#002147',
              vlines_kwargs={"colors": '#002147'}, method='ywm', alpha=0.05)
    ax3b.set_title('Partial Autocorrelation Function (PACF)', fontsize=15, fontweight='bold', pad=6)
    ax3b.set_xlabel('Lag (hours)', fontweight='bold')
    ax3b.set_ylabel('Correlation', fontweight='bold')
    ax3b.grid(False)

    # -------------------- PANEL 4 — Heteroscedasticity (NEW) --------------------
    ax4 = fig.add_subplot(gs[3])
    ax4.scatter(df.loc[residuals.index, expected_pred_col], residuals, # Use the identified prediction_col
                s=8, alpha=0.4, color='#4C72B0', edgecolor='none')
    ax4.axhline(0, color='#B22222', linestyle='--', lw=1.5)
    ax4.set_title('Heteroscedasticity: Residuals vs Predicted Values',
                  fontsize=15, fontweight='bold', pad=6)
    ax4.set_xlabel('Predicted Water Level', fontweight='bold')
    ax4.set_ylabel('Residual', fontweight='bold')
    ax4.grid(False)
    ax4.text(0.02, 0.90, f'Spearman rho = {het_rho:.3f}, p = {het_p:.2e}',
              transform=ax4.transAxes, fontsize=11, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # -------------------- PANEL LABEL --------------------
    fig.text(0.01, 0.99, "b)", fontsize=28, fontweight='bold', va='top')

    # -------------------- SAVE --------------------
    plt.tight_layout(rect=[0.05, 0.03, 0.96, 0.97])
    out_name = f"Residual_Diagnostic_{station}_{model}_{horizon}h.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)  # raise dpi to 1200 for final Elsevier submission
    plt.close(fig)
    print(f"  Saved figure: {out_name}")

# ==========================================================
# SAVE SUMMARY TABLE (all diagnostics, all files, one CSV)
# ==========================================================
summary_df = pd.DataFrame(summary_records)
summary_df = summary_df.sort_values(["Station", "Model", "Horizon"]).reset_index(drop=True)
summary_csv = os.path.join(OUTPUT_DIR, "Residual_Diagnostics_Summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary table: {summary_csv}")

# ==========================================================
# ZIP AND DOWNLOAD EVERYTHING
# ==========================================================
zip_path = str(Path(OUTPUT_DIR).with_suffix(".zip"))
with zipfile.ZipFile(zip_path, "w") as zipf:
    for root, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            full_path = os.path.join(root, fname)
            arcname = os.path.relpath(full_path, start=OUTPUT_DIR)
            zipf.write(full_path, arcname=arcname)

print(f"\nDownloading results zip: {zip_path}")
print(f"Results archive: {zip_path}")

print("\n=== DONE: residual diagnostics generated for all input forecast files ===")