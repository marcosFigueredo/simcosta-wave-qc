from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

import importlib.util

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc04", BASE_DIR / "codigos" / "04_lstm_peak_qc_ba1.py")
qc04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc04)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "ablacao_lstm"

MODEL_SEEDS = [42, 7, 123]
ANOMALY_SEEDS = [42, 43, 44, 45, 46]

# Each config changes exactly one factor relative to the baseline (multivariate,
# 3-step averaged prediction, LOOKBACK=24h - identical to 04_lstm_peak_qc_ba1.py),
# so the ablation isolates the marginal effect of that single factor.
CONFIGS = [
    {"name": "Multivariada 3 passos 24h (baseline)", "lookback": 24, "multi_step": True, "univariate": False},
    {"name": "Univariada (sem auxiliares) 24h", "lookback": 24, "multi_step": True, "univariate": True},
    {"name": "Passo unico 24h", "lookback": 24, "multi_step": False, "univariate": False},
    {"name": "Janela 12h", "lookback": 12, "multi_step": True, "univariate": False},
    {"name": "Janela 30h", "lookback": 30, "multi_step": True, "univariate": False},
]


def make_sequences_variant(values: np.ndarray, target_idx: int, lookback: int, multi_step: bool):
    x_seq, y_seq, center_idx = [], [], []
    if multi_step:
        for end in range(lookback + 1, len(values) - 1):
            x_seq.append(values[end - 1 - lookback : end - 1, :])
            y_seq.append(values[end - 1 : end + 2, target_idx])
            center_idx.append(end)
        y_seq = np.asarray(y_seq)
        y_center = y_seq[:, 1] if len(y_seq) else y_seq
    else:
        for end in range(lookback, len(values)):
            x_seq.append(values[end - lookback : end, :])
            y_seq.append(values[end, target_idx])
            center_idx.append(end)
        y_center = np.asarray(y_seq)
    return np.asarray(x_seq), np.asarray(y_seq), y_center, np.asarray(center_idx)


def build_model_variant(n_features: int, lookback: int, multi_step: bool) -> Sequential:
    output_units = 3 if multi_step else 1
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(lookback, n_features)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(0.2),
            Dense(output_units, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def run_config(config: dict, data_multivariate: pd.DataFrame, data_univariate: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    lookback = config["lookback"]
    multi_step = config["multi_step"]
    data = data_univariate if config["univariate"] else data_multivariate
    feature_cols = list(data.columns)
    target_idx = feature_cols.index(qc04.TARGET)

    split = int(len(data) * (1 - qc04.TEST_FRACTION))
    train_df = data.iloc[:split]
    test_df = data.iloc[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    target_min = scaler.data_min_[target_idx]
    target_range = scaler.data_range_[target_idx]

    x_train, y_train_full, _, _ = make_sequences_variant(train_scaled, target_idx, lookback, multi_step)
    x_test, _, observed_scaled, _ = make_sequences_variant(test_scaled, target_idx, lookback, multi_step)

    observed = observed_scaled * target_range + target_min

    prediction_rows = []
    qc_rows = []

    for model_seed in MODEL_SEEDS:
        qc04.set_reproducibility(model_seed)
        model = build_model_variant(len(feature_cols), lookback, multi_step)
        model.fit(
            x_train,
            y_train_full,
            epochs=12,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        pred_scaled_raw = model.predict(x_test, verbose=0)
        pred_scaled = pred_scaled_raw.mean(axis=1) if multi_step else pred_scaled_raw.ravel()
        predicted = pred_scaled * target_range + target_min

        mape = np.nanmean(np.abs((observed - predicted) / np.where(np.abs(observed) < 1e-6, np.nan, observed))) * 100
        prediction_rows.append(
            {
                "config": config["name"],
                "model_seed": model_seed,
                "mae": mean_absolute_error(observed, predicted),
                "mse": mean_squared_error(observed, predicted),
                "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
                "mape": float(mape),
            }
        )

        diff_clean = qc04.diffmean(observed, predicted)
        robust_scale = qc04.fit_robust_scale(diff_clean)

        for anomaly_seed in ANOMALY_SEEDS:
            observed_with_anom, labels = qc04.inject_point_anomalies(observed, anomaly_seed)
            diff = qc04.diffmean(observed_with_anom, predicted)
            pvalues = qc04.robust_t_pvalues(diff, robust_scale)
            flags = qc04.detect_probabilistic(pvalues)
            values = qc04.metrics(labels, flags)
            qc_rows.append({"config": config["name"], "model_seed": model_seed, "anomaly_seed": anomaly_seed, **values})

    return prediction_rows, qc_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qc04.set_reproducibility(MODEL_SEEDS[0])
    df = qc04.read_data()
    hourly = qc04.build_hourly_dataset(df)
    selected_features, _ = qc04.select_features(hourly)
    data_multivariate = hourly[selected_features].dropna().tail(qc04.MAX_HOURLY_RECORDS)
    data_univariate = hourly.loc[data_multivariate.index, [qc04.TARGET]]

    all_prediction_rows = []
    all_qc_rows = []
    for config in CONFIGS:
        pred_rows, qc_rows = run_config(config, data_multivariate, data_univariate)
        all_prediction_rows.extend(pred_rows)
        all_qc_rows.extend(qc_rows)
        print("Concluido:", config["name"])

    prediction_df = pd.DataFrame(all_prediction_rows)
    qc_df = pd.DataFrame(all_qc_rows)

    prediction_agg = prediction_df.groupby("config")[["mae", "mse", "rmse", "mape"]].agg(["mean", "std"])
    qc_agg = qc_df.groupby("config")[["precision", "recall", "f1"]].agg(["mean", "std"])

    prediction_df.to_csv(OUT_DIR / "ablacao_predicao_replicacoes.csv", index=False)
    qc_df.to_csv(OUT_DIR / "ablacao_qc_replicacoes.csv", index=False)
    prediction_agg.to_csv(OUT_DIR / "ablacao_predicao_agregada.csv")
    qc_agg.to_csv(OUT_DIR / "ablacao_qc_agregada.csv")

    print("Resultados em:", OUT_DIR)


if __name__ == "__main__":
    main()
