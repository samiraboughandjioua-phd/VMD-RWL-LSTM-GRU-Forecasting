# ==============================================================================
# STEP 1: UPLOAD EXCEL FILES
# ==============================================================================

import argparse
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate monthly/yearly NSE heatmaps.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"))
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "NSE_Heatmaps"))
parser.add_argument("--model", default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument("--station", default=None)
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir).resolve(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_paths = [p for p in Path(args.input_dir).resolve().rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
selected_paths = [p for p in all_paths if args.model.lower() in p.name.lower() and (args.station is None or args.station.lower().replace(" ", "_") in p.name.lower().replace(" ", "_"))]
print("Discovered forecast files:")
for filename in selected_paths: print("  ", filename)


# ==============================================================================
# STEP 2: SETTINGS
# ==============================================================================

horizons = [
    '1h', '24h', '48h', '72h',
    '96h', '120h', '144h', '168h'
]

months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

years = list(range(2016, 2023))


# ==============================================================================
# STEP 3: NSE FUNCTION
# ==============================================================================

def calculate_nse(obs, pred):

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # Remove NaN and infinite values
    valid = np.isfinite(obs) & np.isfinite(pred)

    obs = obs[valid]
    pred = pred[valid]

    # Not enough valid data
    if len(obs) == 0:
        return np.nan

    # NSE denominator
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    # Cannot calculate NSE if observations have zero variance
    if denominator == 0:
        return np.nan

    # NSE numerator
    numerator = np.sum((obs - pred) ** 2)

    return 1 - (numerator / denominator)


# ==============================================================================
# STEP 4: FIND HORIZON FILES
# ==============================================================================

# Modified to include .csv files
all_files = [str(p) for p in selected_paths]

nse_results = {}

print("\n" + "=" * 70)
print("FILE DETECTION")
print("=" * 70)

for h in horizons:

    # Example:
    # 1h  -> 1
    # 24h -> 24
    hour = h.replace("h", "")

    matched = []

    for filename in all_files:

        fname = filename.lower()

        # Matches:
        # horizon_1h.xlsx
        # GRU_Can_Tho_horizon_1h.xlsx
        # GRU_Can_Tho_horizon_24h.xlsx
        # horizon-24h.xlsx
        pattern = rf"horizon[_-]?{hour}h(?:[_\.-]|$)"

        if re.search(pattern, fname):
            matched.append(filename)

    if not matched:
        print(f"❌ {h}: NO FILE FOUND")
        continue

    filepath = matched[0]

    print(f"✓ {h}: {filepath}")


    # ==========================================================================
    # READ EXCEL FILE
    # ==========================================================================

    try:
        # Updated to read CSV or Excel based on file extension
        if filepath.lower().endswith(('.csv')):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

    except Exception as e:
        print(f"   ❌ ERROR reading file: {e}")
        continue


    # ==========================================================================
    # CHECK REQUIRED COLUMNS (Changed 'Predicted' to 'LSTM_Predicted')
    # ==========================================================================

    required_columns = [
        'Date',
        'Actual',
        'LSTM_Predicted' # Changed from 'Predicted' to 'LSTM_Predicted'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print(
            f"   ❌ Missing columns: {missing_columns}"
        )

        print(
            f"   Available columns: {list(df.columns)}"
        )

        continue


    # ==========================================================================
    # CONVERT DATA TYPES
    # ==========================================================================

    df['Date'] = pd.to_datetime(
        df['Date'],
        errors='coerce'
    )

    df['Actual'] = pd.to_numeric(
        df['Actual'],
        errors='coerce'
    )

    # Using 'LSTM_Predicted' column for the 'Predicted' values
    df['Predicted'] = pd.to_numeric(
        df['LSTM_Predicted'], # Changed from 'Predicted' to 'LSTM_Predicted'
        errors='coerce'
    )


    # Remove invalid dates
    df = df.dropna(subset=['Date'])


    # ==========================================================================
    # FILTER 2016–2022
    # ==========================================================================

    df = df[
        (df['Date'].dt.year >= 2016) &
        (df['Date'].dt.year <= 2022)
    ].copy()


    if df.empty:

        print(
            f"   ❌ No data found between 2016 and 2022"
        )

        continue


    # ==========================================================================
    # CREATE YEAR AND MONTH
    # ==========================================================================

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month


    # ==========================================================================
    # CALCULATE MONTHLY NSE
    # ==========================================================================

    results = []

    for (year, month), group in df.groupby(
        ['Year', 'Month'],
        sort=True
    ):

        nse = calculate_nse(
            group['Actual'].values,
            group['Predicted'].values
        )

        results.append({
            'Year': year,
            'Month': month,
            'NSE': nse
        })


    monthly_nse = pd.DataFrame(results)


    # ==========================================================================
    # CREATE COMPLETE 2016–2022 × 12 MONTH MATRIX
    # ==========================================================================

    grid = monthly_nse.pivot(
        index='Year',
        columns='Month',
        values='NSE'
    )

    grid = grid.reindex(
        index=years,
        columns=range(1, 13)
    )

    grid.columns = months

    nse_results[h] = grid


    print(
        f"   ✓ Data points: {len(df):,}"
    )

    print(
        f"   ✓ Monthly NSE values: "
        f"{monthly_nse['NSE'].notna().sum()}"
    )


# ==============================================================================
# STEP 5: CHECK HORIZONS
# ==============================================================================

print("\n" + "=" * 70)
print("HORIZONS AVAILABLE")
print("=" * 70)

for h in horizons:

    if h in nse_results:
        print(f"✓ {h}")
    else:
        print(f"❌ {h}")


if len(nse_results) == 0:

    raise ValueError(
        "No valid horizon files were detected."
    )


# ==============================================================================
# STEP 6: FIGURE SETTINGS
# ==============================================================================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10


# ==============================================================================
# STEP 7: CREATE 4 × 2 FIGURE
# ==============================================================================

fig, axes = plt.subplots(
    4,
    2,
    figsize=(14, 16),
    dpi=1200
)

axes = axes.flatten()


# ==============================================================================
# STEP 8: PLOT EACH HORIZON
# ==============================================================================

for idx, h in enumerate(horizons):

    ax = axes[idx]

    if h not in nse_results:

        ax.text(
            0.5,
            0.5,
            f"No data available\n{h}",
            ha='center',
            va='center',
            fontsize=14,
            transform=ax.transAxes
        )

        ax.set_title(
            f"Horizon: {h}",
            fontweight='bold'
        )

        ax.set_xticks([])
        ax.set_yticks([])

        continue


    grid = nse_results[h]


    # ==========================================================================
    # HEATMAP
    # ==========================================================================

    sns.heatmap(
        grid,
        ax=ax,
        annot=True,
        fmt=".2f",

        # Diverging map is appropriate because NSE can be negative
        cmap="RdBu_r",

        # Display range
        vmin=-1,
        vmax=1,

        center=0,

        # Grid lines
        linewidths=0.5,
        linecolor='lightgrey',

        # Colorbar
        cbar=True,
        cbar_kws={
            'label': 'NSE',
            'shrink': 0.8
        },

        # Annotation
        annot_kws={
            'size': 8,
            'weight': 'bold'
        },

        # Missing values
        mask=grid.isna()
    )


    ax.set_title(
        f"Horizon: {h}",
        fontsize=14,
        fontweight='bold',
        pad=8
    )

    ax.set_xlabel(
        "Month",
        fontsize=14
    )

    ax.set_ylabel(
        "Year",
        fontsize=14
    )

    ax.tick_params(
        axis='x',
        rotation=0,
        labelsize=14
    )

    ax.tick_params(
        axis='y',
        rotation=0,
        labelsize=14
    )


# ==============================================================================
# STEP 9: OVERALL TITLE
# ==============================================================================

fig.suptitle(
    "Monthly NSE Heatmap Overview (2016–2022)",
    fontsize=16,
    fontweight='bold',
    y=0.995
)


# ==============================================================================
# STEP 10: LAYOUT
# ==============================================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)


# ==============================================================================
# STEP 11: SAVE
# ==============================================================================

output_file = "Monthly_NSE_Heatmap_Overview.png"

plt.savefig(
    str(OUTPUT_DIR / output_file),
    dpi=1200,
    bbox_inches='tight'
)
fig.text( 0.02, 0.99, "a)", fontsize=18, fontweight='bold', va='top', ha='left' )

# ==============================================================================
# STEP 12: DISPLAY
# ==============================================================================

plt.show()


print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(
    f"Saved as: {output_file}"
)


# ==============================================================================
# STEP 13: DOWNLOAD
# ==============================================================================

