# ==========================================================
# Multi-Horizon Hourly Water-Level Forecasting - LSTM
# Single-run implementation for hourly river water-level forecasting.
#
# The script trains one LSTM configuration for three stations and eight
# forecast horizons. It writes prediction files and summary metrics used by
# the downstream comparison and statistical-analysis scripts.
#
# The experimental design uses the same input lags, preprocessing, training
# settings, and random seed as the GRU implementation so that the recurrent
# cell type is the main architectural difference between the two models.
# ==========================================================

import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro, norm
from statsmodels.stats.diagnostic import acorr_ljungbox

# ----------------------------------------------------------
# 1. Reproducibility
# ----------------------------------------------------------
# A fixed seed makes the single training run reproducible as far as the
# underlying libraries and hardware permit.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------------
# 2. GPU check
# ----------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected:", gpus[0].name)
    except RuntimeError as e:
        print("GPU configuration warning:", e)
else:
    print("No GPU detected. Using CPU.")

# ----------------------------------------------------------
# 3. Portable project directories
# ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
parser = argparse.ArgumentParser(description="Train the LSTM model for hourly RWL forecasting.")
parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"),
                    help="Directory containing the three station Excel files.")
parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "Results_LSTM_80_20_SINGLE_RUN_FINAL"),
                    help="Directory for model predictions and summary results.")
args = parser.parse_args()
BASE_DIR = os.path.abspath(args.data_dir)
RESULTS_DIR = os.path.abspath(args.output_dir)
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.isdir(BASE_DIR):
    raise FileNotFoundError(
        f"Data directory not found: {BASE_DIR}. "
        "Place the three station files in the repository data/ directory or pass --data-dir."
    )

# ----------------------------------------------------------
# 4. Station files
# ----------------------------------------------------------
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

for station, path in STATIONS.items():
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{station} file not found: {path}")

# ----------------------------------------------------------
# 5. Forecast horizons and input lags
# ----------------------------------------------------------
# Each horizon uses its predefined historical input window. The same mapping
# is used by the LSTM and GRU scripts.
FORECAST_CONFIG = {
    1: 94, 24: 138, 48: 148, 72: 152,
    96: 220, 120: 262, 144: 288, 168: 302,
}

# ----------------------------------------------------------
# 6. Training configuration
# ----------------------------------------------------------
TRAIN_SPLIT = 0.80
EPOCHS = 50
BATCH_SIZE = 64
UNITS_1 = 64
UNITS_2 = 32
DROPOUT = 0.20
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
FLOOD_PERCENTILE = 95

# ----------------------------------------------------------
# 7. Performance metrics
# ----------------------------------------------------------
# The functions below calculate the metrics reported for each station and
# forecast horizon, including R2, NSE, KGE, RMSE, and MAE.
def calculate_nse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / den


def calculate_kge(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_true = np.mean(y_true)
    std_true = np.std(y_true, ddof=1)
    if mean_true == 0 or std_true == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred, ddof=1) / std_true
    beta = np.mean(y_pred) / mean_true
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calculate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "NSE": calculate_nse(y_true, y_pred),
        "KGE": calculate_kge(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def diebold_mariano_test(y_true, pred1, pred2, power=2, h=1):
    y_true = np.asarray(y_true, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
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


def flood_metrics(dates, y_true, y_pred, threshold):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    flood_mask = y_true >= threshold
    if np.sum(flood_mask) == 0:
        return {
            "Flood_RMSE": np.nan, "Flood_MAE": np.nan, "Flood_Bias": np.nan,
            "Peak_Error": np.nan,
        }

    yt = y_true[flood_mask]
    yp = y_pred[flood_mask]

    flood_rmse = np.sqrt(np.mean((yt - yp) ** 2))
    flood_mae = np.mean(np.abs(yt - yp))
    flood_bias = np.mean(yp - yt)

    obs_peak_idx = int(np.argmax(y_true))
    pred_peak_idx = int(np.argmax(y_pred))

    peak_error = y_pred[pred_peak_idx] - y_true[obs_peak_idx]
    return {
        "Flood_RMSE": flood_rmse, "Flood_MAE": flood_mae, "Flood_Bias": flood_bias,
        "Peak_Error": peak_error,
    }


def residual_diagnostics(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    bias = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1)

    try:
        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    try:
        sample = residuals
        if len(sample) > 5000:
            rng = np.random.default_rng(SEED)
            sample = rng.choice(sample, 5000, replace=False)
        _, shapiro_p = shapiro(sample)
    except Exception:
        shapiro_p = np.nan

    return {
        "Resid_Bias": bias, "Resid_SD": std_residual,
        "LjungBox_p": lb_p, "Shapiro_p": shapiro_p,
    }


def persistence_forecast(X_test_scaled, scaler):
    """Naive baseline: predicted value = last observed value in the
    input window, inverse-transformed. NOT trained -- no leakage."""
    last_scaled = X_test_scaled[:, -1, 0].reshape(-1, 1)
    return scaler.inverse_transform(last_scaled).flatten()

# ----------------------------------------------------------
# 8. Data loading
# ----------------------------------------------------------
def load_station(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    df = df.dropna()

    datetime_col = None
    for col in df.columns:
        name = str(col).lower()
        if "datetime" in name or "date" in name or "time" in name:
            datetime_col = col
            break
    if datetime_col is None:
        raise ValueError(f"No datetime column found in {file_path}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric water-level column found in {file_path}")
    if len(numeric_cols) > 1:
        print(
            f"WARNING: {file_path} contains {len(numeric_cols)} numeric columns. "
            f"Using the first numeric column only: {numeric_cols[0]}"
        )

    water_col = numeric_cols[0]
    series = df[[water_col]].dropna().astype(np.float32)
    if len(series) < 1000:
        raise ValueError(f"Too few observations after cleaning: {file_path}")

    print(f"Water-level column used: {water_col}")
    print(f"Observations: {len(series)}")
    return series


def create_dataset(data, lag, horizon):
    X, y = [], []
    last_start = len(data) - lag - horizon + 1
    for i in range(last_start):
        X.append(data[i:i + lag])
        y.append(data[i + lag + horizon - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def sample_boundary(raw_boundary, lag, horizon):
    return max(0, raw_boundary - lag - horizon + 1)

# ----------------------------------------------------------
# 9. LSTM architecture: 64 -> 32 (IDENTICAL to GRU script)
# ----------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(UNITS_1, activation="tanh", return_sequences=True),
        Dropout(DROPOUT),
        LSTM(UNITS_2, activation="tanh", return_sequences=False),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=CLIPNORM,
        ),
        loss="mse"
    )
    return model

# ----------------------------------------------------------
# 10. Results directory (EXACT name expected downstream)
# ----------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

summary_rows = []

# ----------------------------------------------------------
# 11. Main loop: 3 stations x 8 horizons, single run
# ----------------------------------------------------------
for station_name, file_path in STATIONS.items():

    print("\n" + "=" * 72)
    print(f"STATION: {station_name}")
    print("=" * 72)

    df = load_station(file_path)
    raw_values = df.values.astype(np.float32)
    n_total = len(raw_values)
    train_end = int(n_total * TRAIN_SPLIT)

    print(f"Total observations: {n_total}")
    print(f"Training observations: {train_end}")
    print(f"Testing observations:  {n_total - train_end}")

    raw_train = raw_values[:train_end]
    flood_threshold = float(np.percentile(raw_train, FLOOD_PERCENTILE))
    print(f"Flood threshold (P{FLOOD_PERCENTILE}, train-only): {flood_threshold:.3f}")

    scaler = MinMaxScaler()
    scaler.fit(raw_train)
    scaled_all = scaler.transform(raw_values).astype(np.float32)

    station_dir = os.path.join(RESULTS_DIR, station_name)
    os.makedirs(station_dir, exist_ok=True)

    for horizon, lag in FORECAST_CONFIG.items():

        print("\n" + "-" * 72)
        print(f"LSTM | {station_name} | Horizon={horizon} h | Lag={lag}")
        print("-" * 72)

        X, y = create_dataset(scaled_all, lag, horizon)
        train_size = sample_boundary(train_end, lag, horizon)

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        if len(X_train) == 0 or len(X_test) == 0:
            print("Not enough data. Skipping this horizon.")
            continue

        print("Training shape:", X_train.shape)
        print("Testing shape: ", X_test.shape)

        test_start = train_end
        test_end = test_start + len(y_test)
        timestamps = df.index[test_start:test_end]

        if len(timestamps) != len(y_test):
            raise RuntimeError(
                f"Timestamp mismatch: {len(timestamps)} dates vs {len(y_test)} targets"
            )

        model = build_model((X_train.shape[1], X_train.shape[2]))
        trainable_params = int(
            np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        )

        train_start = time.time()
        model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()]
        )
        training_time_s = time.time() - train_start

        y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_test_original = scaler.inverse_transform(y_test)
        y_pred_original = scaler.inverse_transform(y_pred)

        y_true = y_test_original[:, 0]
        y_fore = y_pred_original[:, 0]

        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_fore))):
            print(f"WARNING: NaN/Inf for {station_name} | H={horizon}h. Skipping.")
            continue

        y_naive = persistence_forecast(X_test, scaler)

        lstm_metrics = calculate_metrics(y_true, y_fore)
        naive_metrics = calculate_metrics(y_true, y_naive)

        dm_stat, p_val = diebold_mariano_test(y_true, y_fore, y_naive, power=2, h=horizon)

        lstm_flood = flood_metrics(timestamps, y_true, y_fore, flood_threshold)
        naive_flood = flood_metrics(timestamps, y_true, y_naive, flood_threshold)

        resid = residual_diagnostics(y_true, y_fore)

        summary_rows.append({
            "Station": station_name, "Horizon_h": horizon, "Lag": lag,
            "Trainable_Params": trainable_params, "Training_Time_s": training_time_s,
            "LSTM_R2": lstm_metrics["R2"], "LSTM_NSE": lstm_metrics["NSE"],
            "LSTM_KGE": lstm_metrics["KGE"], "LSTM_RMSE": lstm_metrics["RMSE"],
            "LSTM_MAE": lstm_metrics["MAE"],
            "Naive_R2": naive_metrics["R2"], "Naive_NSE": naive_metrics["NSE"],
            "Naive_KGE": naive_metrics["KGE"], "Naive_RMSE": naive_metrics["RMSE"],
            "Naive_MAE": naive_metrics["MAE"],
            "DM_LSTM_vs_Naive": dm_stat, "p_LSTM_vs_Naive": p_val,
            "Signif_LSTM_vs_Naive": significance_label(p_val),
            "LSTM_Flood_RMSE": lstm_flood["Flood_RMSE"],
            "LSTM_Flood_MAE": lstm_flood["Flood_MAE"],
            "LSTM_Flood_Bias": lstm_flood["Flood_Bias"],
            "LSTM_Peak_Error": lstm_flood["Peak_Error"],
                        "Naive_Flood_RMSE": naive_flood["Flood_RMSE"],
            "Naive_Flood_MAE": naive_flood["Flood_MAE"],
            "Naive_Flood_Bias": naive_flood["Flood_Bias"],
            "Naive_Peak_Error": naive_flood["Peak_Error"],
                        "LSTM_Resid_Bias": resid["Resid_Bias"], "LSTM_Resid_SD": resid["Resid_SD"],
            "LSTM_LjungBox_p": resid["LjungBox_p"], "LSTM_Shapiro_p": resid["Shapiro_p"],
        })

        # Prediction CSV -- EXACT filename/columns expected downstream
        pred_df = pd.DataFrame({
            "Date": timestamps, "Actual": y_true,
            "LSTM_Predicted": y_fore, "Naive_Predicted": y_naive,
        })
        pred_file = os.path.join(
            station_dir, f"LSTM_{station_name}_horizon_{horizon}h_predictions.csv"
        )
        pred_df.to_csv(pred_file, index=False)

        print(f"Saved: {pred_file}")
        print(
            f"[LSTM]  R2={lstm_metrics['R2']:.4f} | NSE={lstm_metrics['NSE']:.4f} | "
            f"KGE={lstm_metrics['KGE']:.4f} | RMSE={lstm_metrics['RMSE']:.4f} | "
            f"MAE={lstm_metrics['MAE']:.4f}"
        )
        print(
            f"[Naive] R2={naive_metrics['R2']:.4f} | NSE={naive_metrics['NSE']:.4f} | "
            f"RMSE={naive_metrics['RMSE']:.4f} | MAE={naive_metrics['MAE']:.4f}"
        )
        print(f"DM LSTM vs Naive: stat={dm_stat:.4f} | p={p_val:.4f}")
        print(f"Trainable params={trainable_params:,} | Training time={training_time_s:.2f} s")

        # Save running summary after every horizon (crash-safe)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, "LSTM_results_summary_SINGLE_RUN.csv"), index=False
        )

print(f"\nSaved summary: "
      f"{os.path.join(RESULTS_DIR, 'LSTM_results_summary_SINGLE_RUN.csv')}")
print(f"\n=== LSTM (SINGLE-RUN COMPLETE) FINISHED SUCCESSFULLY ===")
