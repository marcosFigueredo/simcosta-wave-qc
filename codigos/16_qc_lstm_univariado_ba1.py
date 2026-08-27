from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc13", BASE_DIR / "codigos" / "13_qc_lstm_causal_ordinal_ba1.py")
qc13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc13)
qc10 = qc13.qc10
qc04 = qc13.qc04

TARGET = qc13.TARGET
LOOKBACK = qc13.LOOKBACK
CLASS_NAMES = qc13.CLASS_NAMES
CLASS_TO_INT = qc13.CLASS_TO_INT

TRAIN_FRACTION, VAL_FRACTION = qc10.TRAIN_FRACTION, qc10.VAL_FRACTION
MODEL_SEED = 42
TRAIN_ANOMALY_SEED, VAL_ANOMALY_SEED, TEST_ANOMALY_SEED = 301, 302, 303
TRAIN_EVENTS_PER_FAMILY, VAL_EVENTS_PER_FAMILY, TEST_EVENTS_PER_FAMILY = 14, 10, 4

CONFIG = "univariado"


# ---------------------------------------------------------------------------
# Leitura generica de qualquer arquivo OCEAN da SiMCosta, so a variavel-alvo
# (Hsig). Usada tanto para treinar na BA-1 quanto para validar em outras
# boias sem depender de nenhuma variavel auxiliar especifica de uma boia.
# ---------------------------------------------------------------------------

def read_buoy_target_hourly(path: Path, target: str = TARGET) -> pd.DataFrame:
    df = pd.read_csv(path, comment="/", na_values=["NULL", "null", "", "NaN"])
    df["datetime_utc"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]], errors="coerce", utc=True
    )
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc").set_index("datetime_utc")
    df[target] = pd.to_numeric(df[target], errors="coerce")

    raw_hourly = df[[target]].resample("1h").median()
    hourly = raw_hourly.copy()
    hourly[target] = hourly[target].interpolate(limit=3, limit_direction="both")
    missing_mask = raw_hourly[target].isna().reindex(hourly.index).fillna(True).astype(int)
    hourly = hourly.dropna(subset=[target])
    missing_mask = missing_mask.reindex(hourly.index).fillna(0).astype(int)
    return hourly, missing_mask


def build_predictor_univariate(lookback: int) -> Model:
    inputs = Input(shape=(lookback, 1))
    l1 = LSTM(128, return_sequences=True)(inputs)
    l2 = LSTM(64, return_sequences=True)(l1)
    h = LSTM(32)(l2)
    dropped = Dropout(0.2)(h)
    pred = Dense(1, activation="linear")(dropped)
    predictor = Model(inputs, pred, name="predictor_univariado")
    import tensorflow as tf
    predictor.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return predictor


def process_split_univariate(hourly: pd.DataFrame, missing_mask: pd.Series, predictor: Model, scaler: MinMaxScaler,
                              events_per_family: int, anomaly_seed: int, ref_mean: float, ref_std: float) -> dict:
    scaled = scaler.transform(hourly)
    x_seq, y_seq_scaled, center_idx = qc13.qc10.make_sequences_single(scaled, 0, LOOKBACK)
    target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]

    pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
    predicted = pred_scaled * target_range + target_min
    observed = y_seq_scaled * target_range + target_min
    timestamps = hourly.index[center_idx]
    mask_at_center = missing_mask.to_numpy()[center_idx]

    y_aug, class_int, family = qc13.inject_anomalies_causal(observed, events_per_family, anomaly_seed)
    residual = y_aug - predicted

    stats = qc13.causal_stat_vector(y_aug, residual)
    accum = qc13.accumulation_features(residual)
    class_int, confidence = qc13.weak_label_decoupled(y_aug, class_int, mask_at_center, ref_mean, ref_std)
    sentinel_flag = pd.Series(family).str.startswith("G_").to_numpy()

    return {
        "timestamps": timestamps, "observed": observed, "y_aug": y_aug, "predicted": predicted,
        "residual": residual, "abs_residual": np.abs(residual), "mask": mask_at_center,
        "class_int": class_int, "confidence": confidence, "family": family, "stats": stats, "accum": accum,
        "sentinel_flag": sentinel_flag,
    }


def assemble_features_univariate(d: dict) -> np.ndarray:
    stats, accum = d["stats"], d["accum"]
    s_block = np.column_stack([stats["acc"], stats["level"], stats["rate"], stats["range"], stats["mad_score"], stats["gpd_score"]])
    accum_block = np.column_stack(
        [accum[f"ewma_{lam}"] for lam in qc13.EWMA_LAMBDAS]
        + [accum["cusum_pos"], accum["cusum_neg"]]
        + [accum[f"slope_{w}"] for w in qc13.SLOPE_WINDOWS]
        + [accum["persistence"]]
    )
    y_aug_feat = np.nan_to_num(d["y_aug"], nan=float(np.nanmedian(d["y_aug"])))
    residual_feat = np.nan_to_num(d["residual"], nan=0.0)
    abs_residual_feat = np.nan_to_num(d["abs_residual"], nan=0.0)
    residual_block = np.column_stack([y_aug_feat, d["predicted"], residual_feat, abs_residual_feat])
    return np.column_stack([residual_block, s_block, accum_block, d["mask"].reshape(-1, 1)])


def train_pipeline_univariate(model_seed: int = MODEL_SEED) -> dict:
    qc04.set_reproducibility(model_seed)
    path = qc04.DATA_PATH
    hourly, missing_mask = read_buoy_target_hourly(path)
    hourly = hourly.tail(qc10.MAX_HOURLY_RECORDS)
    missing_mask = missing_mask.reindex(hourly.index)

    n = len(hourly)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    train_hourly, val_hourly, test_hourly = hourly.iloc[:n_train], hourly.iloc[n_train:n_val], hourly.iloc[n_val:]
    train_mask, val_mask, test_mask = missing_mask.iloc[:n_train], missing_mask.iloc[n_train:n_val], missing_mask.iloc[n_val:]

    scaler = MinMaxScaler()
    scaler.fit(train_hourly)

    predictor = build_predictor_univariate(LOOKBACK)
    train_scaled = scaler.transform(train_hourly)
    val_scaled = scaler.transform(val_hourly)
    x_train, y_train, _ = qc10.make_sequences_single(train_scaled, 0, LOOKBACK)
    x_val, y_val, _ = qc10.make_sequences_single(val_scaled, 0, LOOKBACK)
    predictor.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=32, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    train_target = train_hourly[TARGET].to_numpy()
    qc13.set_physical_range(train_target)
    ref_mean, ref_std = float(np.nanmean(train_target)), float(np.nanstd(train_target))

    d_train = process_split_univariate(train_hourly, train_mask, predictor, scaler, TRAIN_EVENTS_PER_FAMILY, TRAIN_ANOMALY_SEED, ref_mean, ref_std)
    d_val = process_split_univariate(val_hourly, val_mask, predictor, scaler, VAL_EVENTS_PER_FAMILY, VAL_ANOMALY_SEED, ref_mean, ref_std)
    d_test = process_split_univariate(test_hourly, test_mask, predictor, scaler, TEST_EVENTS_PER_FAMILY, TEST_ANOMALY_SEED, ref_mean, ref_std)

    feats_train, feats_val, feats_test = assemble_features_univariate(d_train), assemble_features_univariate(d_val), assemble_features_univariate(d_test)

    model, feat_scaler = qc13.train_ordinal(
        feats_train, d_train["class_int"], d_train["confidence"],
        feats_val, d_val["class_int"], d_val["confidence"], seed=model_seed,
    )
    x_val_s = feat_scaler.transform(feats_val)
    probs_val = qc13.ordinal_probs(model, x_val_s)
    tau_b = qc13.tune_tau_b(d_val["class_int"], probs_val, d_val["sentinel_flag"])

    x_test_s = feat_scaler.transform(feats_test)
    probs_test = qc13.ordinal_probs(model, x_test_s)
    ev = qc13.evaluate_ordinal(d_test["class_int"], probs_test, tau_b, d_test["sentinel_flag"])

    return {
        "predictor": predictor, "classifier": model, "feat_scaler": feat_scaler, "scaler": scaler,
        "tau_b": tau_b, "ref_mean": ref_mean, "ref_std": ref_std,
        "range_min": qc13.RANGE_MIN, "range_max": qc13.RANGE_MAX,
        "d_test": d_test, "eval": ev, "probs_test": probs_test,
    }


def main() -> None:
    print("Treinando modelo univariado (so Hsig) na BA-1...")
    artifacts = train_pipeline_univariate()
    ev = artifacts["eval"]
    print(f"BA-1 (univariado) -> macro_f1={ev['macro_f1']:.3f} f1_bad={ev['f1_bad']:.3f} "
          f"binary_f1={ev['binary_f1']:.3f} tau_b={ev['tau_b']:.2f}")
    print("Matriz de confusao:\n", ev["confusion_matrix"])
    dist = pd.Series(artifacts["d_test"]["class_int"]).map({v: k for k, v in CLASS_TO_INT.items()}).value_counts()
    print(dist)


if __name__ == "__main__":
    main()
