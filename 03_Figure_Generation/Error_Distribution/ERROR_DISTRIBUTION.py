import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update({
    "font.size": 16,
    "font.family": "sans-serif",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 300,
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the Can Tho GRU error-distribution box plot from prediction CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs",
        help="Directory containing GRU_Can_Tho_horizon_*h_predictions.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "Figures"),
        help="Directory in which the figure will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "GRU_Can_Tho_horizon_*h_predictions.csv")
    filenames = glob.glob(pattern)
    data_list = []

    for filepath in filenames:
        filename = os.path.basename(filepath)
        match = re.search(
            r"GRU_Can_Tho_horizon_(\d+)h_predictions\.csv$",
            filename,
            re.IGNORECASE,
        )
        if not match:
            continue

        h_num = int(match.group(1))
        df = pd.read_csv(filepath)

        if "Actual" not in df.columns or "GRU_Predicted" not in df.columns:
            print(
                f"Skipping {filename}: missing 'Actual' or 'GRU_Predicted' column."
            )
            continue

        df = df[["Actual", "GRU_Predicted"]].copy()
        df["Error"] = df["GRU_Predicted"] - df["Actual"]

        low_q = df["Actual"].quantile(1 / 3)
        high_q = df["Actual"].quantile(2 / 3)

        df["Water Level"] = np.select(
            [df["Actual"] <= low_q, df["Actual"] <= high_q],
            ["Low Water Level", "Medium Water Level"],
            default="High Water Level",
        )
        df["Horizon"] = f"{h_num}h"

        data_list.append(
            {
                "horizon_num": h_num,
                "df": df[["Horizon", "Error", "Water Level"]],
            }
        )

    if not data_list:
        raise FileNotFoundError(
            "No valid Can Tho GRU prediction files were found in "
            f"'{args.input_dir}'. Expected files matching "
            "GRU_Can_Tho_horizon_*h_predictions.csv."
        )

    data_list.sort(key=lambda item: item["horizon_num"])
    combined_df = pd.concat(
        [item["df"] for item in data_list], ignore_index=True
    )
    horizon_order = [f"{item['horizon_num']}h" for item in data_list]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    palette = {
        "Low Water Level": "#1f77b4",
        "Medium Water Level": "#d62728",
        "High Water Level": "#2ca02c",
    }
    hue_order = ["Low Water Level", "Medium Water Level", "High Water Level"]

    sns.boxplot(
        data=combined_df,
        x="Horizon",
        y="Error",
        hue="Water Level",
        order=horizon_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        fliersize=1.5,
        linewidth=1,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Forecast Horizon (hours)", fontweight="bold", fontsize=16)
    ax.set_ylabel("Error Distribution (cm)", fontweight="bold", fontsize=16)
    ax.set_ylim(-110, 110)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)

    ax.text(
        0.02, 0.93, "b)", transform=ax.transAxes, fontsize=18, fontweight="bold"
    )

    sns.despine(top=False, right=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        title="",
        frameon=False,
        fontsize=13,
        loc="upper right",
    )

    plt.tight_layout()
    output_file = os.path.join(
        args.output_dir, "GRU_Error_Distribution_Can_Tho_Font16.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_file}")


if __name__ == "__main__":
    main()
