# ==========================================================
#  GOOGLE COLAB - Extended Figure 17 (ALL stations x ALL horizons)
#  Observed vs Predicted hydrographs during each station's peak
#  flood event window, for LSTM and GRU overlaid together.
#
#  HOW TO USE IN COLAB:
#  1. Run this script from the repository root or from a configured Python environment.
#  2. Upload all forecast files at once - every
#     LSTM_forecast_*.xlsx and GRU_forecast_*.xlsx file, for
#     every station and every horizon (up to 48 files: 3
#     stations x 8 horizons x 2 models). Each file must have a
#     "Date" column, an "Actual" column, and a predicted-value
#     column named either "Predicted", "LSTM_Predicted", or
#     "GRU_Predicted" (auto-detected per file, no manual
#     renaming needed). Its filename must contain the model,
#     station, and horizon (e.g.
#     "LSTM_forecast_Vam_Nao_horizon_24h.xlsx") - same
#     convention as your other scripts.
#  3. For each station, the script automatically finds the peak
#     flood event within that station's test period (using the
#     1-hour horizon file as reference) and defines a +/- window
#     around it (default: 3 days before, 4 days after the peak).
#  4. It then plots ONE figure per station: a grid of subplots
#     (one per horizon), each showing Observed (black), LSTM
#     (blue), and GRU (red) during that event window - the same
#     style as the original Figure 17, extended to all horizons.
#  5. All 3 figures (one per station) are saved at high
#     resolution and downloaded automatically.
# ==========================================================

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

print("=== Extended Figure 17: ALL stations x ALL horizons ===")

# -------------------------------
# 1. CONFIG
# -------------------------------
STATION_ALIASES = {"vamnao": "Vam_Nao", "tanchau": "Tan_Chau", "cantho": "Can_Tho"}
STATION_ORDER = ["Vam_Nao", "Tan_Chau", "Can_Tho"]
HORIZONS = [1, 24, 48, 72, 96, 120, 144, 168]

DAYS_BEFORE_PEAK = 3   # window start = peak_date - N days
DAYS_AFTER_PEAK = 4    # window end   = peak_date + N days

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate peak-flood event hydrographs for LSTM and GRU.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"))
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "Peak_Flood_Event"))
args = parser.parse_args()
OUTPUT_DIR = str(Path(args.output_dir).resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 2. FILENAME PARSER (same convention as your other scripts)
# -------------------------------
def parse_filename(fname):
    name_lower = fname.lower()
    if "lstm" in name_lower:
        model = "LSTM"
    elif "gru" in name_lower:
        model = "GRU"
    else:
        return None, None, None

    clean = name_lower.replace(" ", "").replace("_", "").replace("-", "")
    station = None
    for alias, canonical in STATION_ALIASES.items():
        if alias in clean:
            station = canonical
            break

    horizon_match = re.search(r'(\d+)\s*h(?:our)?', name_lower)
    horizon = int(horizon_match.group(1)) if horizon_match else None

    return model, station, horizon

# -------------------------------
# 3. DISCOVER FORECAST FILES
# -------------------------------
forecast_paths = [p for p in Path(args.input_dir).resolve().rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
if not forecast_paths:
    raise ValueError(f"No forecast files found under {args.input_dir}")
print(f"\n{len(forecast_paths)} forecast files discovered.\n")

# -------------------------------
# 4. LOAD AND ORGANIZE ALL FILES
#    data[station][horizon][model] = DataFrame(Date, Actual, Predicted)
# -------------------------------
data = {s: {h: {} for h in HORIZONS} for s in STATION_ORDER}
skipped = []

for path in forecast_paths:
    fname = str(path)
    model, station, horizon = parse_filename(fname)
    if model is None or station is None or horizon is None:
        skipped.append((fname, "could not parse model/station/horizon from filename"))
        continue
    if station not in STATION_ORDER or horizon not in HORIZONS:
        skipped.append((fname, f"parsed station={station}, horizon={horizon} not in expected sets"))
        continue

    try:
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(fname)
        else:
            df = pd.read_excel(fname)
    except Exception as e:
        skipped.append((fname, f"read error: {e}"))
        continue

    if not {"Date", "Actual"}.issubset(df.columns):
        skipped.append((fname, f"missing Date/Actual columns (found: {list(df.columns)})"))
        continue

    # Accept either a generic "Predicted" column, or a model-specific
    # column such as "LSTM_Predicted" / "GRU_Predicted" - standardize
    # to "Predicted" internally either way.
    if "Predicted" in df.columns:
        pass  # already in the expected format
    elif f"{model}_Predicted" in df.columns:
        df = df.rename(columns={f"{model}_Predicted": "Predicted"})
    else:
        predicted_candidates = [c for c in df.columns if "predicted" in c.lower()]
        if predicted_candidates:
            df = df.rename(columns={predicted_candidates[0]: "Predicted"})
        else:
            skipped.append((fname, f"no Predicted / {model}_Predicted column found "
                                     f"(found: {list(df.columns)})"))
            continue

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    data[station][horizon][model] = df
    print(f"  Loaded {fname}: {station} / {model} / {horizon}h ({len(df)} rows)")

if skipped:
    print(f"\n{len(skipped)} file(s) skipped:")
    for fname, reason in skipped:
        print(f"  - {fname}: {reason}")

# -------------------------------
# 5. DETERMINE PEAK EVENT WINDOW PER STATION
#    (using the 1-hour horizon file as reference - highest
#     temporal resolution / earliest test-period coverage)
# -------------------------------
event_windows = {}
for station in STATION_ORDER:
    ref_df = None
    for model in ["LSTM", "GRU"]:
        candidate = data[station].get(1, {}).get(model)
        if candidate is not None:
            ref_df = candidate
            break
    if ref_df is None:
        print(f"  [warn] No 1-hour file found for {station} - cannot determine event window.")
        continue

    peak_idx = ref_df["Actual"].idxmax()
    peak_date = ref_df.loc[peak_idx, "Date"]
    peak_value = ref_df.loc[peak_idx, "Actual"]

    window_start = peak_date - pd.Timedelta(days=DAYS_BEFORE_PEAK)
    window_end = peak_date + pd.Timedelta(days=DAYS_AFTER_PEAK)
    event_windows[station] = (window_start, window_end, peak_date, peak_value)

    print(f"\n{station}: peak Actual = {peak_value:.1f} at {peak_date} "
          f"-> event window [{window_start.date()} to {window_end.date()}]")

# -------------------------------
# 6. PLOT - ONE FIGURE PER STATION, ONE SUBPLOT PER HORIZON
# -------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
})

for station in STATION_ORDER:
    if station not in event_windows:
        continue
    window_start, window_end, peak_date, peak_value = event_windows[station]

    n_horizons = len(HORIZONS)
    n_cols = 4
    n_rows = int(np.ceil(n_horizons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), dpi=150)
    axes = np.array(axes).flatten()

    for i, horizon in enumerate(HORIZONS):
        ax = axes[i]
        horizon_data = data[station].get(horizon, {})

        if "LSTM" not in horizon_data and "GRU" not in horizon_data:
            ax.set_visible(False)
            continue

        # Actual series (same across models for a given station/horizon)
        ref_df = horizon_data.get("LSTM", horizon_data.get("GRU"))
        mask = (ref_df["Date"] >= window_start) & (ref_df["Date"] <= window_end)
        window_df = ref_df.loc[mask]

        ax.plot(window_df["Date"], window_df["Actual"], color="black", lw=1.6, label="Observed")

        if "LSTM" in horizon_data:
            lstm_df = horizon_data["LSTM"]
            lstm_mask = (lstm_df["Date"] >= window_start) & (lstm_df["Date"] <= window_end)
            ax.plot(lstm_df.loc[lstm_mask, "Date"], lstm_df.loc[lstm_mask, "Predicted"],
                     color="#1f77b4", lw=1.3, label="LSTM")

        if "GRU" in horizon_data:
            gru_df = horizon_data["GRU"]
            gru_mask = (gru_df["Date"] >= window_start) & (gru_df["Date"] <= window_end)
            ax.plot(gru_df.loc[gru_mask, "Date"], gru_df.loc[gru_mask, "Predicted"],
                     color="#d62728", lw=1.3, label="GRU")

        ax.set_title(f"{horizon}-h horizon", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Water level (cm)")
        ax.tick_params(axis='x', rotation=30)
        if i == 0:
            ax.legend(fontsize=9, frameon=True)

    # hide any unused subplot axes
    for j in range(len(HORIZONS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Observed vs Predicted Water Level During Peak Flood Event - {station.replace('_',' ')}\n"
                 f"(Event window: {window_start.date()} to {window_end.date()}, peak = {peak_value:.0f} cm)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(OUTPUT_DIR, f"Figure17_extended_{station}.png")
    fig.savefig(out_path, dpi=1200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

# -------------------------------
# 7. DOWNLOAD ALL FIGURES
# -------------------------------
print("\n=== DONE ===")