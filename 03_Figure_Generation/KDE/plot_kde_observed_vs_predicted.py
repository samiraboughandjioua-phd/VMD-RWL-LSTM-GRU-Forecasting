# ==============================================================================
# STEP 1: IMPORT LIBRARIES
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
import argparse
from pathlib import Path


# ==============================================================================
# STEP 2: UPLOAD FILES FIRST
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate KDE plots for observed and predicted water levels.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"))
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "KDE"))
parser.add_argument("--model", default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument("--station", default="Can_Tho")
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir).resolve(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DISCOVERING CSV / EXCEL FORECAST FILES")
print("=" * 70)
all_candidates = [p for p in Path(args.input_dir).resolve().rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
filtered = [p for p in all_candidates if args.model.lower() in p.name.lower() and args.station.lower().replace(" ", "_") in p.name.lower().replace(" ", "_")]

print("\nCandidate files:")
for filename in filtered:
    print(f"  ✓ {filename}")


# ==============================================================================
# STEP 3: KDE PIPELINE FUNCTION
# ==============================================================================

def run_kde_pipeline(
    file_path,
    station_name="Can Tho",
    horizon="1h",
    actual_col="Actual",
    pred_col="LSTM_Predicted",
):
    """
    Read uploaded CSV/Excel file,
    calculate KDE,
    calculate Q1 and Q3,
    plot observed and predicted distributions,
    and save the figure at 1200 DPI.
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING: {file_path}")
    print("=" * 70)


    # ==========================================================================
    # 1. CHECK FILE
    # ==========================================================================

    if not os.path.exists(file_path):

        print(f"❌ File not found: {file_path}")

        return


    # ==========================================================================
    # 2. READ FILE AUTOMATICALLY
    # ==========================================================================

    try:

        if file_path.lower().endswith(".csv"):

            df = pd.read_csv(file_path)

        elif file_path.lower().endswith((".xls", ".xlsx")):

            df = pd.read_excel(file_path)

        else:

            print(
                f"❌ Unsupported file format: {file_path}"
            )

            return

    except Exception as e:

        print(
            f"❌ Error reading file: {e}"
        )

        return


    print("✓ File loaded successfully")
    print(f"✓ Number of rows: {len(df):,}")


    # ==========================================================================
    # 3. CHECK REQUIRED COLUMNS
    # ==========================================================================

    if actual_col not in df.columns:

        print(
            f"❌ Column '{actual_col}' not found."
        )

        print(
            f"Available columns: {list(df.columns)}"
        )

        return


    if pred_col not in df.columns:

        print(
            f"❌ Column '{pred_col}' not found."
        )

        print(
            f"Available columns: {list(df.columns)}"
        )

        return


    # ==========================================================================
    # 4. EXTRACT OBSERVED AND PREDICTED DATA
    # ==========================================================================

    observed = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    ).dropna().values

    predicted = pd.to_numeric(
        df[pred_col],
        errors="coerce"
    ).dropna().values


    if len(observed) < 2:

        print(
            "❌ Not enough observed data for KDE."
        )

        return


    if len(predicted) < 2:

        print(
            "❌ Not enough predicted data for KDE."
        )

        return


    print(
        f"✓ Observed values: {len(observed):,}"
    )

    print(
        f"✓ Predicted values: {len(predicted):,}"
    )


    # ==========================================================================
    # 5. CALCULATE Q1 AND Q3
    # ==========================================================================

    obs_q1, obs_q3 = np.percentile(
        observed,
        [25, 75]
    )

    pred_q1, pred_q3 = np.percentile(
        predicted,
        [25, 75]
    )


    print("\nQuartiles:")

    print(
        f"Observed Q1 = {obs_q1:.2f}"
    )

    print(
        f"Observed Q3 = {obs_q3:.2f}"
    )

    print(
        f"Predicted Q1 = {pred_q1:.2f}"
    )

    print(
        f"Predicted Q3 = {pred_q3:.2f}"
    )


    # ==========================================================================
    # 6. CREATE FIGURE
    # ==========================================================================

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        dpi=1200
    )


    # ==========================================================================
    # 7. OBSERVED KDE
    # ==========================================================================

    sns.kdeplot(
        observed,
        ax=ax,
        color="tab:blue",
        label="Observed",
        fill=True,
        alpha=0.4,
        linewidth=2
    )


    # ==========================================================================
    # 8. PREDICTED KDE
    # ==========================================================================

    sns.kdeplot(
        predicted,
        ax=ax,
        color="tab:orange",
        label="Predicted",
        fill=True,
        alpha=0.4,
        linewidth=2
    )


    # ==========================================================================
    # 9. KDE DENSITY FUNCTIONS
    # ==========================================================================

    obs_kde = gaussian_kde(
        observed
    )

    pred_kde = gaussian_kde(
        predicted
    )


    # ==========================================================================
    # 10. GET DENSITY AT Q1 AND Q3
    # ==========================================================================

    obs_q1_density = obs_kde(
        [obs_q1]
    )[0]

    obs_q3_density = obs_kde(
        [obs_q3]
    )[0]

    pred_q1_density = pred_kde(
        [pred_q1]
    )[0]

    pred_q3_density = pred_kde(
        [pred_q3]
    )[0]


    # ==========================================================================
    # 11. OBSERVED Q1 AND Q3 POINTS
    # ==========================================================================

    ax.scatter(
        [obs_q1, obs_q3],
        [
            obs_q1_density,
            obs_q3_density
        ],
        color="tab:blue",
        s=35,
        zorder=5
    )


    # ==========================================================================
    # 12. OBSERVED Q1 LABEL
    # ==========================================================================

    ax.text(
        obs_q1 - 12,
        obs_q1_density + 0.0003,
        f"Q1: {obs_q1:.2f}",
        color="tab:blue",
        fontweight="bold",
        fontsize=14
    )


    # ==========================================================================
    # 13. OBSERVED Q3 LABEL
    # ==========================================================================

    ax.text(
        obs_q3 + 3,
        obs_q3_density + 0.0003,
        f"Q3: {obs_q3:.2f}",
        color="tab:blue",
        fontweight="bold",
        fontsize=14
    )


    # ==========================================================================
    # 14. PREDICTED Q1 AND Q3 POINTS
    # ==========================================================================

    ax.scatter(
        [pred_q1, pred_q3],
        [
            pred_q1_density,
            pred_q3_density
        ],
        color="tab:orange",
        s=35,
        zorder=5
    )


    # ==========================================================================
    # 15. PREDICTED Q1 LABEL
    # ==========================================================================

    ax.text(
        pred_q1 - 12,
        pred_q1_density - 0.0006,
        f"Q1: {pred_q1:.2f}",
        color="tab:orange",
        bbox=dict(
            boxstyle="square,pad=0.1",
            fc="white",
            ec="none",
            alpha=0.7
        ),
        fontweight="bold",
        fontsize=14
    )


    # ==========================================================================
    # 16. PREDICTED Q3 LABEL
    # ==========================================================================

    ax.text(
        pred_q3 + 3,
        pred_q3_density - 0.0006,
        f"Q3: {pred_q3:.2f}",
        color="tab:orange",
        bbox=dict(
            boxstyle="square,pad=0.1",
            fc="white",
            ec="none",
            alpha=0.7
        ),
        fontweight="bold",
        fontsize=14
    )


    # ==========================================================================
    # 17. HEADING
    # ONLY STATION NAME + HORIZON
    # ==========================================================================

    ax.set_title(
        f"{station_name} — Horizon: {horizon}",
        loc="left",
        fontweight="bold",
        fontsize=14,
        pad=8
    )


    # ==========================================================================
    # 18. AXIS LABELS
    # ==========================================================================

    ax.set_xlabel(
        "Water Level (cm)",
        fontsize=14
    )

    ax.set_ylabel(
        "Density",
        fontsize=14
    )


    # ==========================================================================
    # 19. REMOVE TOP AND RIGHT SPINES
    # ==========================================================================

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


    # ==========================================================================
    # 20. LEGEND
    # ==========================================================================

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=12
    )


    # ==========================================================================
    # 21. LAYOUT
    # ==========================================================================

    plt.tight_layout()


    # ==========================================================================
    # 22. OUTPUT FILE NAME
    # ==========================================================================

    output_filename = (
        f"LSTM_"
        f"{station_name.replace(' ', '')}_"
        f"{horizon}_KDE_1200DPI.png"
    )


    # ==========================================================================
    # 23. SAVE FIGURE
    # ==========================================================================

    plt.savefig(
        output_filename,
        dpi=1200,
        bbox_inches="tight"
    )


    print(
        f"\n✅ Figure successfully saved:"
    )

    print(
        f"   {output_filename}"
    )


    # ==========================================================================
    # 24. DISPLAY FIGURE
    # ==========================================================================

    plt.show()


    # ==========================================================================
    # 25. CLOSE FIGURE
    # ==========================================================================

    plt.close(fig)


# ==============================================================================
# STEP 4: FIND UPLOADED FILES
# ==============================================================================

print("\n" + "=" * 70)
print("SEARCHING FOR HORIZON FILES")
print("=" * 70)


uploaded_files = [str(p) for p in filtered]


# ==============================================================================
# FIND 1-HOUR FILE
# ==============================================================================

file_1h = None

for filename in uploaded_files:

    filename_lower = filename.lower()

    if (
        "horizon_1h" in filename_lower
        and filename_lower.endswith(
            (".csv", ".xls", ".xlsx")
        )
    ):

        file_1h = filename
        break


# ==============================================================================
# FIND 168-HOUR FILE
# ==============================================================================

file_168h = None

for filename in uploaded_files:

    filename_lower = filename.lower()

    if (
        "horizon_168h" in filename_lower
        and filename_lower.endswith(
            (".csv", ".xls", ".xlsx")
        )
    ):

        file_168h = filename
        break


# ==============================================================================
# STEP 5: DISPLAY FILE DETECTION
# ==============================================================================

if file_1h is not None:

    print(
        f"✓ 1-hour file found: {file_1h}"
    )

else:

    print(
        "❌ 1-hour file was not found."
    )


if file_168h is not None:

    print(
        f"✓ 168-hour file found: {file_168h}"
    )

else:

    print(
        "❌ 168-hour file was not found."
    )


# ==============================================================================
# STEP 6: RUN 1-HOUR KDE
# ==============================================================================

if file_1h is not None:

    run_kde_pipeline(
        file_path=file_1h,
        station_name="Can Tho",
        horizon="1h",
        actual_col="Actual",
        pred_col="LSTM_Predicted"
    )


# ==============================================================================
# STEP 7: RUN 168-HOUR KDE
# ==============================================================================

if file_168h is not None:

    run_kde_pipeline(
        file_path=file_168h,
        station_name="Can Tho",
        horizon="168h",
        actual_col="Actual",
        pred_col="LSTM_Predicted"
    )


# ==============================================================================
# STEP 8: FINISHED
# ==============================================================================

print("\n" + "=" * 70)
print("ALL PROCESSING FINISHED")
print("=" * 70)