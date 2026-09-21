import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# ==========================================================
# ELSEVIER CLEAN COMPACT STYLE
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.linewidth": 1.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True
})

horizon_titles = [
    "1-h Horizon", "24-h Horizon",
    "48-h Horizon", "72-h Horizon",
    "96-h Horizon", "120-h Horizon",
    "144-h Horizon", "168-h Horizon"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Generate observed-vs-predicted scatter panels.")
parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "outputs"), help="Directory containing forecast CSV/XLSX files.")
parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "Figures" / "Scatter"), help="Directory for figures.")
parser.add_argument("--model", default="LSTM", choices=["LSTM", "GRU"], help="Model to plot.")
parser.add_argument("--station", default=None, help="Optional station filter, e.g. Can_Tho.")
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
def load_file(file_path):
    if str(file_path).lower().endswith('.csv'):
        return pd.read_csv(file_path)
    if str(file_path).lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file type")

# ----------------------------------------------------------
def preprocess_df(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2022-12-31')]
    df = df.dropna(subset=['Actual', 'Predicted'])
    return df

# ----------------------------------------------------------
def calculate_metrics(actual, predicted):
    r = np.corrcoef(actual, predicted)[0, 1]
    r2 = r**2
    return r, r2

# ----------------------------------------------------------
def main():

    files_found = []
    for path in Path(args.input_dir).resolve().rglob("*"):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        name = path.name.lower()
        if args.model.lower() not in name:
            continue
        if args.station and args.station.lower().replace(" ", "_") not in name.replace(" ", "_"):
            continue
        if "horizon_" in name or "horizon-" in name:
            files_found.append(path)
    files_found = sorted(files_found, key=lambda p: int(__import__('re').search(r'(\d+)\s*h', p.name.lower()).group(1)) if __import__('re').search(r'(\d+)\s*h', p.name.lower()) else 9999)
    if len(files_found) != 8:
        raise ValueError(f"Expected exactly 8 {args.model} forecast files; found {len(files_found)}. Use --input-dir/--station to select one station.")
    dfs = [preprocess_df(load_file(path)) for path in files_found]

    # ======================================================
    # CLEAN PANEL LAYOUT
    # ======================================================
    fig, axes = plt.subplots(
        4, 2,
        figsize=(11.2, 12.8),
        dpi=1200
    )

    axes = axes.flatten()

    for i, df in enumerate(dfs):

        ax = axes[i]

        actual = df['Actual'].values
        predicted = df['Predicted'].values

        # ==========================================
        # Scatter plot (NEW COLOR)
        # ==========================================
        ax.scatter(
            actual,
            predicted,
            s=8,
            alpha=0.85,
            color="#133353"
        )

        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())

        # ==========================================
        # 1:1 reference line
        # ==========================================
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="#444444",
            linestyle="--",
            linewidth=1.2
        )

        r, r2 = calculate_metrics(actual, predicted)

        # ==========================================
        # Statistics box
        # ==========================================
        ax.text(
            0.05, 0.93,
            f"R = {r:.3f}\nR² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.35',
                linewidth=1,
                alpha=0.95
            )
        )

        # ==========================================
        # Title
        # ==========================================
        ax.set_title(
            f"{horizon_titles[i]} (2016–2022)",
            fontsize=13,
            fontweight='bold',
            pad=4
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # Axis labels layout
        if i % 2 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Predicted", fontweight='bold')

        if i < 6:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Actual", fontweight='bold')

    # ======================================================
    # PANEL SPACING
    # ======================================================
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.07,
        hspace=0.20,
        wspace=0.12
    )

    # Panel label
    fig.text(
        0.012,
        0.985,
        "a)",
        fontsize=24,
        fontweight="bold",
        va="top"
    )

    # ======================================================
    # SAVE FIGURE
    # ======================================================
    png_name = "Figures/Elsevier_Clean_Compact_1200DPI.png"
    pdf_name = "Figures/Elsevier_Clean_Compact_1200DPI.pdf"

    plt.savefig(str(OUTPUT_DIR / png_name), dpi=1200, bbox_inches='tight')
    plt.savefig(str(OUTPUT_DIR / pdf_name), dpi=1200, bbox_inches='tight')

    plt.show()


# ----------------------------------------------------------
if __name__ == "__main__":
    main()
