# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Portable Python script derived from the original analysis workflow.

Original file is located at
    https://colab.research.google.com/drive/1gNDC_gv6-8GkrOLROdZ1XAx8Vyi7v95j
"""

# =========================================================
# MONTHLY VIOLIN PLOTS - ELSEVIER STYLE
# =========================================================
#
# Columns in BOTH files:
#     Date
#     Actual
#     LSTM_Predicted
#
# Output:
#     (a) 1-hour Horizon PNG - 1200 DPI
#     (b) 168-hour Horizon PNG - 1200 DPI
#
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import argparse
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================

actual_col = "Actual"
pred_col = "LSTM_Predicted"

month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------

palette = {
    "Actual": "blue",
    "LSTM Predicted": "red"
}

# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate monthly violin plots.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"), help="Directory containing forecast files.")
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "Violin"))
parser.add_argument("--model", default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument("--station", default=None)
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir).resolve(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("white")


# =========================================================
# ELSEVIER-STYLE FONT SETTINGS
# =========================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "DejaVu Serif"
]

plt.rcParams["axes.linewidth"] = 1.3
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


# =========================================================
# FUNCTION TO READ CSV / EXCEL
# =========================================================

def read_file(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":

        df = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:

        df = pd.read_excel(file_path)

    else:

        raise ValueError(
            "Only CSV, XLSX, or XLS files are supported."
        )

    return df


# =========================================================
# FUNCTION TO CREATE VIOLIN PLOT
# =========================================================

def monthly_violin_plot(
        file_path,
        horizon_label,
        panel_label):

    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    df = read_file(file_path)

    print("\n========================================")
    print("File:", file_path)
    print("Columns found:")
    print(df.columns.tolist())
    print("========================================")

    # -----------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    required_columns = [
        "Date",
        "Actual",
        "LSTM_Predicted"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing columns: {missing_columns}\n\n"
            "Your file MUST contain exactly these required "
            "columns:\n"
            "Date\n"
            "Actual\n"
            "LSTM_Predicted"
        )

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    df["Datetime"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["Datetime"]
    ).copy()

    # -----------------------------------------------------
    # NUMERIC VALUES
    # -----------------------------------------------------

    df[actual_col] = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    )

    df[pred_col] = pd.to_numeric(
        df[pred_col],
        errors="coerce"
    )

    # -----------------------------------------------------
    # YEAR
    # -----------------------------------------------------

    df["Year"] = df["Datetime"].dt.year

    # -----------------------------------------------------
    # MONTH
    # -----------------------------------------------------

    df["Month_Name"] = (
        df["Datetime"]
        .dt.strftime("%b")
    )

    # -----------------------------------------------------
    # FILTER 2016–2022
    # -----------------------------------------------------

    df = df[
        (df["Year"] >= 2016) &
        (df["Year"] <= 2022)
    ].copy()

    # -----------------------------------------------------
    # MONTH ORDER
    # -----------------------------------------------------

    df["Month_Name"] = pd.Categorical(
        df["Month_Name"],
        categories=month_order,
        ordered=True
    )

    # -----------------------------------------------------
    # MELT DATA
    # -----------------------------------------------------

    df_melt = df.melt(
        id_vars=["Month_Name"],
        value_vars=[
            actual_col,
            pred_col
        ],
        var_name="Type",
        value_name="Water Level"
    )

    # -----------------------------------------------------
    # Rename legend names
    # -----------------------------------------------------

    df_melt["Type"] = df_melt["Type"].replace({
        "Actual": "Actual",
        "LSTM_Predicted": "LSTM Predicted"
    })

    # Remove missing values
    df_melt = df_melt.dropna(
        subset=["Water Level"]
    )

    # =====================================================
    # CREATE FIGURE
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )

    # =====================================================
    # VIOLIN PLOT
    # =====================================================

    sns.violinplot(
        data=df_melt,
        x="Month_Name",
        y="Water Level",
        hue="Type",

        split=True,

        palette=palette,

        inner="quartile",

        cut=0,

        linewidth=1.2,

        saturation=0.85,

        ax=ax
    )

    # =====================================================

    # =====================================================
    # X LABEL
    # =====================================================

    ax.set_xlabel(
        "Month",
        fontsize=18,
        fontweight="bold",
        labelpad=20
    )

    # =====================================================
    # Y LABEL
    # =====================================================

    ax.set_ylabel(
        "Water Level",
        fontsize=20,
        fontweight="bold",
        labelpad=20
    )

    # =====================================================
    # TICK LABELS
    # =====================================================

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=20,
        width=1.3,
        length=6
    )

    # Bold month labels
    for label in ax.get_xticklabels():

        label.set_fontweight("bold")

    for label in ax.get_yticklabels():

        label.set_fontweight("bold")

    # =====================================================
    # PANEL LABEL
    # =====================================================
    #
    # (a) = 1 hour
    # (b) = 168 hours
    #
    # Upper-left corner
    # =====================================================

    ax.text(
        -0.075,
        1.055,
        panel_label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # =====================================================
    # LEGEND
    # =====================================================

    ax.legend(
        title=None,
        loc="upper right",
        fontsize=18,
        frameon=True,
        borderpad=0.04,
        handlelength=1.5
    )

    # =====================================================
    # REMOVE TOP AND RIGHT SPINES
    # =====================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    # =====================================================
    # NO GRID
    # =====================================================

    ax.grid(False)

    # =====================================================
    # LAYOUT
    # =====================================================

    plt.tight_layout(
        pad=1.5
    )

    # =====================================================
    # OUTPUT FILE NAME
    # =====================================================

    output_name = (
        f"Figures/"
        f"Violin_Monthly_"
        f"Actual_vs_LSTM_Predicted_"
        f"{horizon_label}.png"
    )

    # =====================================================
    # SAVE PNG AT 1200 DPI
    # =====================================================

    fig.savefig(
        str(OUTPUT_DIR / output_name),
        dpi=1200,
        format="png",
        bbox_inches="tight",
        pad_inches=0.08
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    plt.show()

    # Close figure after displaying
    plt.close(fig)

    # =====================================================
    # DOWNLOAD PNG
    # =====================================================


    print("\n✓ Figure generated successfully")
    print("✓ Panel:", panel_label)
    print("✓ Horizon:", horizon_label)
    print("✓ Format: PNG")
    print("✓ Resolution: 1200 DPI")
    print("✓ Output:", output_name)


# =========================================================
# SELECT 1-HOUR AND 168-HOUR FILES
# =========================================================
files_found = []
for path in Path(args.input_dir).resolve().rglob("*"):
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        continue
    name = path.name.lower()
    if args.model.lower() not in name:
        continue
    if args.station and args.station.lower().replace(" ", "_") not in name.replace(" ", "_"):
        continue
    if "horizon_1h" in name or "horizon-1h" in name or "horizon_168h" in name or "horizon-168h" in name:
        files_found.append(path)
file_names = sorted(files_found, key=lambda p: 1 if "1h" in p.name.lower() else 168)
if len(file_names) != 2:
    raise ValueError(f"Expected 1h and 168h {args.model} files; found {len(file_names)}. Use --station if needed.")


# =========================================================
# FIGURE (a)
# 1-HOUR HORIZON
# =========================================================

monthly_violin_plot(
    file_names[0],
    "1h Horizon",
    "(a)"
)


# =========================================================
# FIGURE (b)
# 168-HOUR HORIZON
# =========================================================

monthly_violin_plot(
    file_names[1],
    "168h Horizon",
    "(b)"
)


# =========================================================
# FINISHED
# =========================================================

print(
    "\n"
    "====================================================\n"
    "🎉 BOTH FIGURES COMPLETED\n"
    "====================================================\n"
    "✓ (a) 1-hour Horizon\n"
    "✓ (b) 168-hour Horizon\n"
    "✓ Actual = Blue\n"
    "✓ LSTM Predicted = Red\n"
    "✓ 2016–2022\n"
    "✓ PNG format\n"
    "✓ 1200 DPI\n"
    "✓ Elsevier-style formatting\n"
    "✓ Larger labels and text\n"
    "===================================================="
)





