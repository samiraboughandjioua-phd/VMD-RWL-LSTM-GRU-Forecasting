
# ==============================================================
# Additional statistical and event-based analyses for LSTM and GRU.
#
# This script does not retrain either model. It reads the final independent
# test-set prediction files and performs post-processing analyses, including
# block-bootstrap confidence intervals, residual variance diagnostics, a
# 2018 maximum-water-level comparison, percentile sensitivity analysis, and
# event-based high-water detection metrics.
#
# All thresholds are derived from the training observations before being
# applied to the independent test predictions. The model architecture, lag
# windows, and training hyperparameters are not modified here.
# ==============================================================

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 0. PATHS — MUST MATCH THE EXISTING LSTM/GRU SCRIPTS
# --------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
parser = argparse.ArgumentParser(description="Run additional statistical analyses for LSTM/GRU RWL forecasts.")
parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
parser.add_argument("--lstm-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_LSTM_80_20_SINGLE_RUN_FINAL"))
parser.add_argument("--gru-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_GRU_80_20_SINGLE_RUN_FINAL"))
parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Additional_Statistical_Analysis"))
args = parser.parse_args()
BASE_DIR = os.path.abspath(args.data_dir)
LSTM_RESULTS_DIR = os.path.abspath(args.lstm_dir)
GRU_RESULTS_DIR = os.path.abspath(args.gru_dir)
OUT_DIR = os.path.abspath(args.output_dir)
os.makedirs(OUT_DIR, exist_ok=True)

STATIONS = {
    "Tan_Chau": os.path.join(
        BASE_DIR, "Hourly_Water_Level_1984_2022_Tan_Chau_station.xlsx"
    ),
    "Vam_Nao": os.path.join(
        BASE_DIR, "Hourly_Water_Level_1984_2022_Vam_Nao_station.xlsx"
    ),
    "Can_Tho": os.path.join(
        BASE_DIR, "Hourly_Water_Level_1984_2022_Can_Tho_station.xlsx"
    ),
}

FIG_DIR = os.path.join(OUT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = [1, 24, 48, 72, 96, 120, 144, 168]
FORECAST_CONFIG = {
    1: 94,
    24: 138,
    48: 148,
    72: 152,
    96: 220,
    120: 262,
    144: 288,
    168: 302,
}

TRAIN_SPLIT = 0.80
BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 24       # 24-hour blocks preserve short-term temporal dependence
CI_LEVEL = 0.95
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------------------
# 1. METRICS
# --------------------------------------------------------------
def nse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / den


def kge(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_obs = np.mean(y_true)
    std_obs = np.std(y_true, ddof=1)

    if std_obs == 0 or mean_obs == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(r):
        return np.nan

    alpha = np.std(y_pred, ddof=1) / std_obs
    beta = np.mean(y_pred) / mean_obs

    return 1.0 - np.sqrt(
        (r - 1.0) ** 2 +
        (alpha - 1.0) ** 2 +
        (beta - 1.0) ** 2
    )


def metric_vector(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


# --------------------------------------------------------------
# 2. LOAD STATION DATA
# --------------------------------------------------------------
def load_station(file_path):
    df = pd.read_excel(file_path, engine="openpyxl").dropna()

    datetime_col = None
    for col in df.columns:
        name = str(col).lower()
        if "datetime" in name or "date" in name or "time" in name:
            datetime_col = col
            break

    if datetime_col is None:
        raise ValueError(f"No datetime column found: {file_path}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"No numeric water-level column found: {file_path}")

    water_col = numeric_cols[0]
    series = df[water_col].astype(float).dropna()

    return series


# --------------------------------------------------------------
# 3. BLOCK BOOTSTRAP
# --------------------------------------------------------------
def block_bootstrap_indices(n, block_length, rng):
    """
    Circular moving-block bootstrap.
    Produces n indices from contiguous blocks.
    """
    if n <= 1:
        return np.arange(n)

    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))

    starts = rng.integers(0, n, size=n_blocks)

    indices = []
    for s in starts:
        block = (s + np.arange(block_length)) % n
        indices.extend(block.tolist())

    return np.asarray(indices[:n], dtype=int)


def bootstrap_metric_ci(
    y_true,
    y_pred,
    reps=1000,
    block_length=24,
    ci_level=0.95,
    rng=None,
):
    """
    Block-bootstrap confidence intervals for forecast metrics.
    The original paired y_true/y_pred observations are resampled together.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    point = metric_vector(y_true, y_pred)
    boot = {m: [] for m in point.keys()}

    if rng is None:
        rng = np.random.default_rng(42)

    for _ in range(reps):
        idx = block_bootstrap_indices(len(y_true), block_length, rng)

        yt = y_true[idx]
        yp = y_pred[idx]

        try:
            vals = metric_vector(yt, yp)
            for m, v in vals.items():
                if np.isfinite(v):
                    boot[m].append(v)
        except Exception:
            continue

    alpha = 1.0 - ci_level

    row = {}
    for m in point.keys():
        values = np.asarray(boot[m], dtype=float)
        if len(values) == 0:
            lo = hi = np.nan
        else:
            lo = np.quantile(values, alpha / 2)
            hi = np.quantile(values, 1 - alpha / 2)

        row[f"{m}_Point"] = point[m]
        row[f"{m}_CI_Lower"] = lo
        row[f"{m}_CI_Upper"] = hi
        row[f"{m}_Bootstrap_N"] = len(values)

    return row


# --------------------------------------------------------------
# 4. HETEROSCEDASTICITY
# --------------------------------------------------------------
def heteroscedasticity_test(y_true, y_pred):
    """
    Breusch-Pagan test using absolute/squared residual behavior
    against fitted values.

    H0: homoscedastic residual variance.
    p < 0.05 -> evidence against homoscedasticity.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(residuals)
    )

    residuals = residuals[valid]
    fitted = y_pred[valid]

    if len(residuals) < 20:
        return {
            "BP_LM": np.nan,
            "BP_LM_p": np.nan,
            "BP_F": np.nan,
            "BP_F_p": np.nan,
            "Heteroscedasticity": "Insufficient data",
        }

    # BP requires exogenous variables; use fitted value and intercept.
    exog = sm.add_constant(fitted)

    try:
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(residuals, exog)
    except Exception:
        return {
            "BP_LM": np.nan,
            "BP_LM_p": np.nan,
            "BP_F": np.nan,
            "BP_F_p": np.nan,
            "Heteroscedasticity": "Test failed",
        }

    conclusion = "Evidence of heteroscedasticity" if f_p < 0.05 else "No significant evidence"

    return {
        "BP_LM": lm_stat,
        "BP_LM_p": lm_p,
        "BP_F": f_stat,
        "BP_F_p": f_p,
        "Heteroscedasticity": conclusion,
    }


def save_heteroscedasticity_plot(
    y_true,
    y_pred,
    station,
    horizon,
    model_name,
):
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        y_pred,
        residuals,
        s=7,
        alpha=0.25,
        edgecolors="none",
    )
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.xlabel("Fitted / Predicted Water Level")
    plt.ylabel("Residual (Observed − Predicted)")
    plt.title(
        f"Residual vs Fitted | {model_name} | {station} | {horizon} h"
    )
    plt.tight_layout()

    base = f"{model_name}_{station}_H{horizon}_heteroscedasticity"
    plt.savefig(os.path.join(FIG_DIR, base + ".png"), dpi=600)
    plt.savefig(os.path.join(FIG_DIR, base + ".pdf"))
    plt.close()


# --------------------------------------------------------------
# 5. EVENT METRICS FOR 90/95/99 PERCENTILES
# --------------------------------------------------------------
def event_metrics(y_true, y_pred, threshold):
    observed_event = np.asarray(y_true) >= threshold
    predicted_event = np.asarray(y_pred) >= threshold

    tp = int(np.sum(observed_event & predicted_event))
    tn = int(np.sum(~observed_event & ~predicted_event))
    fp = int(np.sum(~observed_event & predicted_event))
    fn = int(np.sum(observed_event & ~predicted_event))

    n_obs_events = tp + fn
    n_non_events = tn + fp

    hit_rate = tp / n_obs_events if n_obs_events > 0 else np.nan
    false_alarm_rate = fp / n_non_events if n_non_events > 0 else np.nan

    precision = precision_score(
        observed_event,
        predicted_event,
        zero_division=0,
    )
    recall = recall_score(
        observed_event,
        predicted_event,
        zero_division=0,
    )
    f1 = f1_score(
        observed_event,
        predicted_event,
        zero_division=0,
    )

    return {
        "Threshold": threshold,
        "Observed_Event_Count": int(np.sum(observed_event)),
        "Predicted_Event_Count": int(np.sum(predicted_event)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Hit_Rate": hit_rate,
        "False_Alarm_Rate": false_alarm_rate,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# --------------------------------------------------------------
# 6. 2018 MAXIMUM COMPARISON
# --------------------------------------------------------------
def maximum_2018_analysis(
    station_series,
    train_end_timestamp,
    dates,
    y_true,
    y_lstm,
    y_gru,
    horizon,
):
    """
    Compares:
      - maximum water level in training period
      - maximum observed level in 2018 test period
      - LSTM predicted maximum in 2018
      - GRU predicted maximum in 2018

    The training maximum is computed from the actual chronological
    training portion used by the original scripts.
    """
    station_series = station_series.sort_index()

    train_series = station_series.loc[
        station_series.index < train_end_timestamp
    ]

    year_2018 = station_series[
        station_series.index.year == 2018
    ]

    test_dates = pd.to_datetime(dates)
    mask_2018 = test_dates.year == 2018

    if len(train_series) == 0 or len(year_2018) == 0 or not np.any(mask_2018):
        return {
            "Train_Start": train_series.index.min() if len(train_series) else pd.NaT,
            "Train_End": train_series.index.max() if len(train_series) else pd.NaT,
            "Train_Max": np.nan,
            "2018_Observed_Max": np.nan,
            "2018_Observed_vs_Train_Max_Difference": np.nan,
            "2018_Observed_vs_Train_Max_Percent": np.nan,
            "2018_LSTM_Max": np.nan,
            "2018_GRU_Max": np.nan,
            "LSTM_Peak_Magnitude_Error_2018": np.nan,
            "GRU_Peak_Magnitude_Error_2018": np.nan,
        }

    train_max = float(train_series.max())
    observed_2018_max = float(year_2018.max())

    difference = observed_2018_max - train_max
    percent = (
        100.0 * difference / abs(train_max)
        if train_max != 0
        else np.nan
    )

    y_true_2018 = np.asarray(y_true)[mask_2018]
    y_lstm_2018 = np.asarray(y_lstm)[mask_2018]
    y_gru_2018 = np.asarray(y_gru)[mask_2018]

    if len(y_true_2018) == 0:
        return {
            "Train_Start": train_series.index.min(),
            "Train_End": train_series.index.max(),
            "Train_Max": train_max,
            "2018_Observed_Max": observed_2018_max,
            "2018_Observed_vs_Train_Max_Difference": difference,
            "2018_Observed_vs_Train_Max_Percent": percent,
            "2018_LSTM_Max": np.nan,
            "2018_GRU_Max": np.nan,
            "LSTM_Peak_Magnitude_Error_2018": np.nan,
            "GRU_Peak_Magnitude_Error_2018": np.nan,
        }

    lstm_max = float(np.max(y_lstm_2018))
    gru_max = float(np.max(y_gru_2018))

    return {
        "Train_Start": train_series.index.min(),
        "Train_End": train_series.index.max(),
        "Train_Max": train_max,
        "2018_Observed_Max": observed_2018_max,
        "2018_Observed_vs_Train_Max_Difference": difference,
        "2018_Observed_vs_Train_Max_Percent": percent,
        "2018_LSTM_Max": lstm_max,
        "2018_GRU_Max": gru_max,
        "LSTM_Peak_Magnitude_Error_2018": lstm_max - observed_2018_max,
        "GRU_Peak_Magnitude_Error_2018": gru_max - observed_2018_max,
    }


# --------------------------------------------------------------
# 7. MAIN ANALYSIS
# --------------------------------------------------------------
bootstrap_rows = []
hetero_rows = []
sensitivity_rows = []
max2018_rows = []

for station, station_file in STATIONS.items():

    print("\n" + "=" * 80)
    print(f"STATION: {station}")
    print("=" * 80)

    station_series = load_station(station_file)
    n_total = len(station_series)
    train_end_idx = int(n_total * TRAIN_SPLIT)

    train_series = station_series.iloc[:train_end_idx]
    train_end_timestamp = station_series.index[train_end_idx]

    # TRAIN-ONLY thresholds
    thresholds = {
        90: float(np.percentile(train_series.values, 90)),
        95: float(np.percentile(train_series.values, 95)),
        99: float(np.percentile(train_series.values, 99)),
    }

    print(f"Total observations: {n_total}")
    print(f"Training observations: {len(train_series)}")
    print(f"Training end timestamp: {train_end_timestamp}")
    print(
        "Thresholds (TRAIN ONLY): "
        f"P90={thresholds[90]:.4f}, "
        f"P95={thresholds[95]:.4f}, "
        f"P99={thresholds[99]:.4f}"
    )

    for horizon in HORIZONS:

        lstm_file = os.path.join(
            LSTM_RESULTS_DIR,
            station,
            f"LSTM_{station}_horizon_{horizon}h_predictions.csv",
        )
        gru_file = os.path.join(
            GRU_RESULTS_DIR,
            station,
            f"GRU_{station}_horizon_{horizon}h_predictions.csv",
        )

        if not os.path.isfile(lstm_file):
            print(f"Missing LSTM predictions: {lstm_file}")
            continue

        if not os.path.isfile(gru_file):
            print(f"Missing GRU predictions: {gru_file}")
            continue

        lstm_df = pd.read_csv(lstm_file)
        gru_df = pd.read_csv(gru_file)

        # Align both models by Date.
        lstm_keep = lstm_df[
            ["Date", "Actual", "LSTM_Predicted"]
        ].copy()

        gru_keep = gru_df[
            ["Date", "GRU_Predicted"]
        ].copy()

        merged = pd.merge(
            lstm_keep,
            gru_keep,
            on="Date",
            how="inner",
        )

        merged["Date"] = pd.to_datetime(merged["Date"])
        merged = merged.sort_values("Date").dropna()

        if len(merged) < 20:
            print(f"Too few aligned test samples: {station}, H={horizon}")
            continue

        dates = merged["Date"].values
        y_true = merged["Actual"].values.astype(float)
        y_lstm = merged["LSTM_Predicted"].values.astype(float)
        y_gru = merged["GRU_Predicted"].values.astype(float)

        # ------------------------------------------------------
        # A. Bootstrap CI
        # ------------------------------------------------------
        lstm_boot = bootstrap_metric_ci(
            y_true,
            y_lstm,
            reps=BOOTSTRAP_REPS,
            block_length=BLOCK_LENGTH,
            ci_level=CI_LEVEL,
            rng=rng,
        )

        gru_boot = bootstrap_metric_ci(
            y_true,
            y_gru,
            reps=BOOTSTRAP_REPS,
            block_length=BLOCK_LENGTH,
            ci_level=CI_LEVEL,
            rng=rng,
        )

        bootstrap_rows.append({
            "Station": station,
            "Horizon_h": horizon,
            "Lag": FORECAST_CONFIG[horizon],
            "Model": "LSTM",
            **lstm_boot,
        })

        bootstrap_rows.append({
            "Station": station,
            "Horizon_h": horizon,
            "Lag": FORECAST_CONFIG[horizon],
            "Model": "GRU",
            **gru_boot,
        })

        # ------------------------------------------------------
        # B. Heteroscedasticity
        # ------------------------------------------------------
        lstm_hetero = heteroscedasticity_test(y_true, y_lstm)
        gru_hetero = heteroscedasticity_test(y_true, y_gru)

        hetero_rows.append({
            "Station": station,
            "Horizon_h": horizon,
            "Lag": FORECAST_CONFIG[horizon],
            "Model": "LSTM",
            **lstm_hetero,
        })

        hetero_rows.append({
            "Station": station,
            "Horizon_h": horizon,
            "Lag": FORECAST_CONFIG[horizon],
            "Model": "GRU",
            **gru_hetero,
        })

        save_heteroscedasticity_plot(
            y_true, y_lstm, station, horizon, "LSTM"
        )
        save_heteroscedasticity_plot(
            y_true, y_gru, station, horizon, "GRU"
        )

        # ------------------------------------------------------
        # C. 90/95/99 percentile sensitivity
        # ------------------------------------------------------
        for percentile, threshold in thresholds.items():

            lstm_event = event_metrics(
                y_true, y_lstm, threshold
            )
            gru_event = event_metrics(
                y_true, y_gru, threshold
            )

            sensitivity_rows.append({
                "Station": station,
                "Horizon_h": horizon,
                "Lag": FORECAST_CONFIG[horizon],
                "Percentile": percentile,
                "Model": "LSTM",
                **lstm_event,
            })

            sensitivity_rows.append({
                "Station": station,
                "Horizon_h": horizon,
                "Lag": FORECAST_CONFIG[horizon],
                "Percentile": percentile,
                "Model": "GRU",
                **gru_event,
            })

        # ------------------------------------------------------
        # D. 2018 maximum comparison
        # ------------------------------------------------------
        max_result = maximum_2018_analysis(
            station_series=station_series,
            train_end_timestamp=train_end_timestamp,
            dates=dates,
            y_true=y_true,
            y_lstm=y_lstm,
            y_gru=y_gru,
            horizon=horizon,
        )

        max2018_rows.append({
            "Station": station,
            "Horizon_h": horizon,
            "Lag": FORECAST_CONFIG[horizon],
            **max_result,
        })

        print(
            f"{station} | H={horizon:3d} h | "
            f"Bootstrap complete | "
            f"BP LSTM p={lstm_hetero['BP_F_p']:.4g} | "
            f"BP GRU p={gru_hetero['BP_F_p']:.4g}"
        )

# --------------------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------------------
bootstrap_df = pd.DataFrame(bootstrap_rows)
hetero_df = pd.DataFrame(hetero_rows)
sensitivity_df = pd.DataFrame(sensitivity_rows)
max2018_df = pd.DataFrame(max2018_rows)

bootstrap_csv = os.path.join(
    OUT_DIR, "Table_Bootstrap_95CI_LSTM_GRU.csv"
)
bootstrap_xlsx = os.path.join(
    OUT_DIR, "Table_Bootstrap_95CI_LSTM_GRU.xlsx"
)

hetero_csv = os.path.join(
    OUT_DIR, "Table_Heteroscedasticity_LSTM_GRU.csv"
)
hetero_xlsx = os.path.join(
    OUT_DIR, "Table_Heteroscedasticity_LSTM_GRU.xlsx"
)

sensitivity_csv = os.path.join(
    OUT_DIR, "Table_Percentile_Sensitivity_90_95_99.csv"
)
sensitivity_xlsx = os.path.join(
    OUT_DIR, "Table_Percentile_Sensitivity_90_95_99.xlsx"
)

max2018_csv = os.path.join(
    OUT_DIR, "Table_2018_Maximum_Comparison.csv"
)
max2018_xlsx = os.path.join(
    OUT_DIR, "Table_2018_Maximum_Comparison.xlsx"
)

bootstrap_df.to_csv(bootstrap_csv, index=False)
bootstrap_df.to_excel(bootstrap_xlsx, index=False)

hetero_df.to_csv(hetero_csv, index=False)
hetero_df.to_excel(hetero_xlsx, index=False)

sensitivity_df.to_csv(sensitivity_csv, index=False)
sensitivity_df.to_excel(sensitivity_xlsx, index=False)

max2018_df.to_csv(max2018_csv, index=False)
max2018_df.to_excel(max2018_xlsx, index=False)

# --------------------------------------------------------------
# 9. MANUSCRIPT-FRIENDLY SUMMARY TABLES
# --------------------------------------------------------------

# A. Bootstrap compact table
bootstrap_compact = bootstrap_df[
    [
        "Station", "Horizon_h", "Lag", "Model",
        "R2_Point", "R2_CI_Lower", "R2_CI_Upper",
        "NSE_Point", "NSE_CI_Lower", "NSE_CI_Upper",
        "KGE_Point", "KGE_CI_Lower", "KGE_CI_Upper",
        "RMSE_Point", "RMSE_CI_Lower", "RMSE_CI_Upper",
        "MAE_Point", "MAE_CI_Lower", "MAE_CI_Upper",
    ]
].copy()

bootstrap_compact.to_csv(
    os.path.join(OUT_DIR, "Table_Bootstrap_Compact_For_Manuscript.csv"),
    index=False,
)

# B. Event sensitivity compact
sensitivity_compact = sensitivity_df[
    [
        "Station", "Horizon_h", "Lag", "Percentile", "Model",
        "Threshold",
        "Observed_Event_Count",
        "Predicted_Event_Count",
        "TP", "TN", "FP", "FN",
        "Hit_Rate",
        "False_Alarm_Rate",
        "Precision",
        "Recall",
        "F1",
    ]
].copy()

sensitivity_compact.to_csv(
    os.path.join(OUT_DIR, "Table_Event_Sensitivity_Compact.csv"),
    index=False,
)

# --------------------------------------------------------------
# 10. SUMMARY BY MODEL / PERCENTILE
# --------------------------------------------------------------
summary_sensitivity = (
    sensitivity_df
    .groupby(["Model", "Percentile"], as_index=False)
    [
        [
            "Hit_Rate",
            "False_Alarm_Rate",
            "Precision",
            "Recall",
            "F1",
        ]
    ]
    .mean()
)

summary_sensitivity.to_csv(
    os.path.join(OUT_DIR, "Summary_Event_Sensitivity_Mean.csv"),
    index=False,
)
summary_sensitivity.to_excel(
    os.path.join(OUT_DIR, "Summary_Event_Sensitivity_Mean.xlsx"),
    index=False,
)

# --------------------------------------------------------------
# 11. Extreme-event sensitivity plots
# --------------------------------------------------------------

# Plot: mean hit rate by percentile
plt.figure(figsize=(8, 5))

for model in ["LSTM", "GRU"]:
    sub = summary_sensitivity[
        summary_sensitivity["Model"] == model
    ].sort_values("Percentile")

    plt.plot(
        sub["Percentile"],
        sub["Hit_Rate"],
        marker="o",
        linewidth=2,
        label=model,
    )

plt.xlabel("Extreme-event threshold percentile")
plt.ylabel("Hit Rate")
plt.xticks([90, 95, 99])
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "Sensitivity_Hit_Rate_90_95_99.png"),
    dpi=600,
)
plt.savefig(
    os.path.join(FIG_DIR, "Sensitivity_Hit_Rate_90_95_99.pdf")
)
plt.close()

# Plot: mean false alarm rate by percentile
plt.figure(figsize=(8, 5))

for model in ["LSTM", "GRU"]:
    sub = summary_sensitivity[
        summary_sensitivity["Model"] == model
    ].sort_values("Percentile")

    plt.plot(
        sub["Percentile"],
        sub["False_Alarm_Rate"],
        marker="o",
        linewidth=2,
        label=model,
    )

plt.xlabel("Extreme-event threshold percentile")
plt.ylabel("False Alarm Rate")
plt.xticks([90, 95, 99])
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "Sensitivity_False_Alarm_90_95_99.png"),
    dpi=600,
)
plt.savefig(
    os.path.join(FIG_DIR, "Sensitivity_False_Alarm_90_95_99.pdf")
)
plt.close()

# --------------------------------------------------------------
# 12. README / interpretation guide
# --------------------------------------------------------------
readme = r"""
analysis ADDITIONAL ANALYSES
============================

This folder contains post-processing analyses based on the final independent
TEST predictions generated by the existing LSTM and GRU scripts.

No model retraining is performed here.

1. BOOTSTRAP 95% CI
-------------------
Use:
    Table_Bootstrap_95CI_LSTM_GRU.xlsx
    Table_Bootstrap_Compact_For_Manuscript.csv

Interpretation:
The confidence intervals quantify uncertainty in the test-set performance
metrics. The bootstrap is block-based (24-hour blocks) to reduce the problem
of treating temporally correlated hourly observations as independent.

2. HETEROSCEDASTICITY
---------------------
Use:
    Table_Heteroscedasticity_LSTM_GRU.xlsx
    Figures/*_heteroscedasticity.*

Breusch-Pagan:
    p < 0.05  -> evidence of non-constant residual variance
    p >= 0.05 -> no significant evidence of heteroscedasticity

The residual-vs-fitted plots should be inspected together with the test.

3. 2018 MAXIMUM COMPARISON
--------------------------
Use:
    Table_2018_Maximum_Comparison.xlsx

The table compares:
    - training-period maximum observed water level
    - observed maximum in 2018
    - LSTM maximum predicted level in 2018
    - GRU maximum predicted level in 2018
    - peak magnitude errors

This directly addresses the analysis request to quantify the 2018 anomaly.

4. 90/95/99 PERCENTILE SENSITIVITY
----------------------------------
Use:
    Table_Percentile_Sensitivity_90_95_99.xlsx
    Summary_Event_Sensitivity_Mean.xlsx
    Figures/Sensitivity_*.*

The 90th, 95th and 99th percentile thresholds are calculated ONLY from
training-period observations. The thresholds are then applied to the
independent test predictions.

5. EVENT-BASED METRICS
----------------------
The sensitivity table reports:
    Hit Rate
    False Alarm Rate
    Precision
    Recall
    F1
    TP / TN / FP / FN

These metrics complement RMSE/MAE/NSE and provide an operational view of
extreme-event detection.

IMPORTANT FOR THE MANUSCRIPT
----------------------------
These analyses do NOT solve the separate analysis request for repeated
multi-seed model training. If multiple-run mean ± SD is required, LSTM/GRU
must be retrained with several independent random seeds.

They also do not replace the manuscript revisions concerning:
    - direct vs recursive forecasting wording
    - literature/references
    - language editing
    - novelty/research gap
    - operational recommendation wording
    - limitations of the univariate input design
"""
with open(
    os.path.join(OUT_DIR, "README_Reviewer_Additional_Analyses.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(readme)

print("\n" + "=" * 80)
print("analysis ADDITIONAL ANALYSES COMPLETE")
print("=" * 80)
print(f"Output directory: {OUT_DIR}")
print(f"Bootstrap CI:     {bootstrap_xlsx}")
print(f"Heterosced.:      {hetero_xlsx}")
print(f"Percentiles:      {sensitivity_xlsx}")
print(f"2018 comparison:  {max2018_xlsx}")
print(f"Figures:          {FIG_DIR}")
print("\nNo LSTM/GRU retraining was performed.")
