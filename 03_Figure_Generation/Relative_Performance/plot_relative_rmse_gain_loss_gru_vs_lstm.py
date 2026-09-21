# ==========================================================
#  GOOGLE COLAB - Figure 13 directly from LSTM/GRU forecast files
#  (the per-horizon "Actual vs Predicted" files produced by your
#   LSTM_three_stations.py / GRU_three_stations.py scripts)
#
#  HOW TO USE IN COLAB:
#  1. Run this script from the repository root or from a configured Python environment.
#  2. Provide the directory containing the forecast files with --input-dir.
#     every LSTM_forecast_*.xlsx and GRU_forecast_*.xlsx file,
#     for every station and every horizon (up to 48 files:
#     3 stations x 8 horizons x 2 models).
#  3. Each filename must contain the model name (LSTM/GRU), the
#     station name, and the horizon (e.g.
#     "LSTM_forecast_Vam_Nao_horizon_24h.xlsx"). The script parses
#     these automatically from the filename - no manual mapping.
#  4. Each file must have "Actual" and "Predicted" columns.
#  5. The script computes RMSE per file, then the relative
#     performance of GRU vs LSTM per station/horizon:
#        Relative Gain of GRU (%) = (RMSE_LSTM - RMSE_GRU) / RMSE_LSTM * 100
#     (this matches the manuscript's original Figure 13 formula)
#  6. It plots the figure (same colors/style as the original) and
#     saves the PNG and underlying data CSV to the configured output directory.
# ==========================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate relative GRU-vs-LSTM RMSE gain/loss figure.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"))
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "Relative_Performance"))
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir).resolve(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
forecast_paths = [p for p in Path(args.input_dir).resolve().rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
if not forecast_paths:
    raise ValueError(f"No forecast files found under {args.input_dir}")
print("=== Figure 13 from LSTM/GRU forecast files ===")
print(f"Found {len(forecast_paths)} candidate files.")

# -------------------------------
# 2. STATION NAME ALIASES (edit here if your naming differs)
# -------------------------------
STATION_ALIASES = {
    "vamnao": "Vam_Nao",
    "tanchau": "Tan_Chau",
    "cantho": "Can_Tho",
}
STATION_ORDER = ["Vam_Nao", "Tan_Chau", "Can_Tho"]
STATION_LABELS = {"Vam_Nao": "Station 1", "Tan_Chau": "Station 2", "Can_Tho": "Station 3"}
STATION_COLORS = {"Vam_Nao": "#1f77b4", "Tan_Chau": "#ff7f0e", "Can_Tho": "#2ca02c"}
HORIZONS = [1, 24, 48, 72, 96, 120, 144, 168]

def parse_filename(fname):
    """Extract (model, station, horizon) from a forecast filename."""
    name_lower = fname.lower()

    # model
    if "lstm" in name_lower:
        model = "LSTM"
    elif "gru" in name_lower:
        model = "GRU"
    else:
        return None, None, None

    # station
    clean = name_lower.replace(" ", "").replace("_", "").replace("-", "")
    station = None
    for alias, canonical in STATION_ALIASES.items():
        if alias in clean:
            station = canonical
            break

    # horizon: look for a number immediately followed by 'h'
    horizon_match = re.search(r'(\d+)\s*h(?:our)?', name_lower)
    horizon = int(horizon_match.group(1)) if horizon_match else None

    return model, station, horizon

# -------------------------------
# 3. READ EACH FILE, COMPUTE RMSE
# -------------------------------
records = []
skipped = []

for path in forecast_paths:
    fname = str(path)
    model, station, horizon = parse_filename(fname)

    if model is None or station is None or horizon is None:
        skipped.append((fname, f"model={model}, station={station}, horizon={horizon}"))
        continue

    try:
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(fname)
        else:
            df = pd.read_excel(fname)
    except Exception as e:
        skipped.append((fname, f"read error: {e}"))
        continue

    # Determine the predicted column name based on the model
    predicted_col_name = f"{model}_Predicted"

    if not {"Actual", predicted_col_name}.issubset(df.columns):
        skipped.append((fname, f"missing Actual/{predicted_col_name} columns (found: {list(df.columns)})"))
        continue

    actual = df["Actual"].to_numpy()
    predicted = df[predicted_col_name].to_numpy()
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    records.append({
        "Station": station, "Model": model, "Horizon": horizon,
        "RMSE": rmse, "N": len(actual), "SourceFile": fname
    })
    print(f"  Parsed {fname}: Station={station}, Model={model}, Horizon={horizon}h, RMSE={rmse:.3f}")

if skipped:
    print(f"\n{len(skipped)} file(s) skipped:")
    for fname, reason in skipped:
        print(f"  - {fname}: {reason}")

metrics_df = pd.DataFrame(records)
if metrics_df.empty:
    raise ValueError("No valid forecast files could be parsed. Check filenames and columns.")

metrics_df.to_csv(OUTPUT_DIR / "RMSE_summary_from_uploaded_files.csv", index=False)
print("\nSaved RMSE summary: RMSE_summary_from_uploaded_files.csv")

# -------------------------------
# 4. COMPUTE RELATIVE PERFORMANCE (RMSE-based, matches original Fig. 13)
# -------------------------------
rel_records = []
for station in STATION_ORDER:
    for horizon in HORIZONS:
        lstm_row = metrics_df[(metrics_df.Station == station) & (metrics_df.Model == "LSTM") & (metrics_df.Horizon == horizon)]
        gru_row = metrics_df[(metrics_df.Station == station) & (metrics_df.Model == "GRU") & (metrics_df.Horizon == horizon)]

        if lstm_row.empty or gru_row.empty:
            continue

        rmse_lstm = lstm_row["RMSE"].values[0]
        rmse_gru = gru_row["RMSE"].values[0]
        rel_gain_pct = (rmse_lstm - rmse_gru) / rmse_lstm * 100.0

        rel_records.append({
            "Station": station, "Horizon": horizon,
            "RMSE_LSTM": rmse_lstm, "RMSE_GRU": rmse_gru,
            "Relative_Gain_GRU_pct": rel_gain_pct
        })

result_df = pd.DataFrame(rel_records)
if result_df.empty:
    raise ValueError("Could not compute relative performance - need matching LSTM and GRU "
                      "files for at least one station/horizon.")

result_df.to_csv(OUTPUT_DIR / "Figure13_underlying_data.csv", index=False)
print("Saved Figure 13 data: Figure13_underlying_data.csv")

# -------------------------------
# 5. PLOT - SAME STYLE/COLORS AS ORIGINAL FIGURE, HIGH QUALITY
# -------------------------------
plt.rcParams["font.size"] = 11

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
n_stations = len(STATION_ORDER)
bar_width = 0.8 / n_stations
x = np.arange(len(HORIZONS))

for i, station in enumerate(STATION_ORDER):
    station_data = result_df[result_df.Station == station].set_index("Horizon")
    values = [station_data.loc[h, "Relative_Gain_GRU_pct"] if h in station_data.index else np.nan
              for h in HORIZONS]
    offset = (i - (n_stations - 1) / 2) * bar_width
    ax.bar(x + offset, values, width=bar_width,
           label=STATION_LABELS[station], color=STATION_COLORS[station],
           edgecolor="black", linewidth=0.6)

ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(HORIZONS)
ax.set_xlabel("Horizon (h)", fontsize=12)
ax.set_ylabel("Relative Performance Gain/Loss of GRU (%)", fontsize=12)
ax.set_title("Relative Performance Gain/Loss of GRU Model compared to LSTM", fontsize=13)
ax.legend(frameon=True, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

OUTPUT_PNG = str(OUTPUT_DIR / "Figure13_relative_performance_HQ.png")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved high-resolution figure: {OUTPUT_PNG}")

# -------------------------------
# 6. DOWNLOAD RESULTS
# -------------------------------




print("\n=== DONE ===")
