from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc04", BASE_DIR / "codigos" / "04_lstm_peak_qc_ba1.py")
qc04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc04)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_3classes"
FIG_DIR = OUT_DIR / "figures"

TARGET = qc04.TARGET
LOOKBACK = qc04.LOOKBACK
MAX_HOURLY_RECORDS = qc04.MAX_HOURLY_RECORDS

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
# remaining 0.20 is test

MODEL_SEED = 42
TRAIN_ANOMALY_SEED = 101
VAL_ANOMALY_SEED = 102
TEST_ANOMALY_SEED = 103

# Higher density in training so the classifier head sees enough SUSPECT/BAD
# examples of every family; validation/test kept close to an operationally
# realistic prevalence (same order of magnitude as qc04.ANOMALY_RATIO ~1/130).
TRAIN_INJECT_RATE = 0.04
VAL_INJECT_RATE = 0.015
TEST_INJECT_RATE = 0.015

# Physically implausible range for Hsig at BA-1 (Baia de Todos os Santos is
# sheltered; storm waves well above this would already be a known event).
RANGE_MIN, RANGE_MAX = 0.0, 8.0

CLASS_NAMES = ["GOOD", "SUSPECT", "BAD"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_NAMES)}

ABLATION_MODELS = ["B_residual", "C_residual_stats", "D_hidden_residual", "E_full", "F_no_lstm"]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    df = qc04.read_data()
    hourly = qc04.build_hourly_dataset(df)
    selected_features, corr = qc04.select_features(hourly)
    data = hourly[selected_features].dropna().tail(MAX_HOURLY_RECORDS)

    available = [c for c in qc04.WAVE_FEATURES + qc04.WATER_FEATURES if c in df.columns]
    raw_hourly = df[available].resample("1h").median()
    missing_before_interp = raw_hourly[TARGET].isna().reindex(data.index).fillna(True).astype(int)

    return data, selected_features, corr, missing_before_interp


def chronological_split(data: pd.DataFrame, mask: pd.Series) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    n = len(data)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    splits = {
        "train": (data.iloc[:n_train], mask.iloc[:n_train]),
        "val": (data.iloc[n_train:n_val], mask.iloc[n_train:n_val]),
        "test": (data.iloc[n_val:], mask.iloc[n_val:]),
    }
    return splits


def make_sequences_single(values: np.ndarray, target_idx: int, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seq, y_seq, center_idx = [], [], []
    for end in range(lookback, len(values)):
        x_seq.append(values[end - lookback:end, :])
        y_seq.append(values[end, target_idx])
        center_idx.append(end)
    return np.asarray(x_seq), np.asarray(y_seq), np.asarray(center_idx)


# ---------------------------------------------------------------------------
# Phase A: predictor LSTM (direct single-step prediction) + hidden state
# ---------------------------------------------------------------------------

def build_predictor(n_features: int, lookback: int) -> tuple[Model, Model]:
    inputs = Input(shape=(lookback, n_features))
    l1 = LSTM(128, return_sequences=True)(inputs)
    l2 = LSTM(64, return_sequences=True)(l1)
    h = LSTM(32)(l2)
    dropped = Dropout(0.2)(h)
    pred = Dense(1, activation="linear")(dropped)
    predictor = Model(inputs, pred, name="predictor")
    predictor.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    # Shares the same LSTM layer instances as `predictor`; predicting with
    # this model after predictor.fit() uses the trained weights.
    encoder = Model(inputs, h, name="encoder")
    return predictor, encoder


# ---------------------------------------------------------------------------
# Phase B: anomaly injection families (protocol section 7.1) + weak/strong
# label generation (protocol section 6.3)
# ---------------------------------------------------------------------------

def _pick_slots(n: int, rng: np.random.Generator, n_events: int, min_gap: int, max_duration: int) -> list[int]:
    used = np.zeros(n, dtype=bool)
    slots = []
    candidates = list(rng.permutation(np.arange(min_gap, n - max_duration - min_gap)))
    for idx in candidates:
        if len(slots) >= n_events:
            break
        lo, hi = idx - min_gap, idx + max_duration + min_gap
        if used[lo:hi].any():
            continue
        used[lo:hi] = True
        slots.append(idx)
    return slots


def inject_anomalies(y: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Injects the 7 perturbation families from the protocol.

    Returns (y_aug, class_int, family_name). class_int uses CLASS_TO_INT;
    untouched points get class_int = -1 (to be weak-labeled separately).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    y_aug = y.copy()
    class_int = np.full(n, -1, dtype=int)
    family = np.array([""] * n, dtype=object)
    std = float(np.nanstd(y))

    n_total_events = max(1, int(round(n * rate)))
    # split the event budget across the 7 families as evenly as possible
    n_families = 7
    per_family = max(1, n_total_events // n_families)

    slots = _pick_slots(n, rng, per_family * n_families, min_gap=6, max_duration=24)
    rng.shuffle(slots)
    chunks = [slots[i * per_family:(i + 1) * per_family] for i in range(n_families)]

    # A. spike aditivo
    for idx in chunks[0]:
        s = rng.choice([-1, 1])
        k = rng.choice([0.5, 1, 2, 3, 4, 6])
        y_aug[idx] = max(0.0, y[idx] + s * k * std)
        class_int[idx] = CLASS_TO_INT["BAD"] if k >= 3 else CLASS_TO_INT["SUSPECT"]
        family[idx] = f"A_spike_k{k}"

    # B. perturbacao multiplicativa (reproduz o artigo de referencia)
    for idx in chunks[1]:
        factor = rng.choice([5.0, 10.0, 1 / 5, 1 / 10])
        y_aug[idx] = y[idx] * factor
        class_int[idx] = CLASS_TO_INT["BAD"]
        family[idx] = f"B_mult_x{factor:.2f}"

    # C. mudanca de nivel
    for idx in chunks[2]:
        d = int(rng.choice([3, 6, 12, 24]))
        delta = rng.choice([-1, 1]) * rng.uniform(1.0, 4.0) * std
        end = min(n, idx + d)
        y_aug[idx:end] = np.maximum(0.0, y[idx:end] + delta)
        cls = CLASS_TO_INT["BAD"] if abs(delta) >= 3 * std else CLASS_TO_INT["SUSPECT"]
        class_int[idx:end] = cls
        family[idx:end] = f"C_level_shift_d{d}"

    # D. drift
    for idx in chunks[3]:
        d = int(rng.choice([3, 6, 12, 24]))
        delta = rng.choice([-1, 1]) * rng.uniform(0.1, 0.4) * std
        end = min(n, idx + d)
        j = np.arange(end - idx)
        y_aug[idx:end] = np.maximum(0.0, y[idx:end] + j * delta)
        total_shift = abs(delta) * (end - idx)
        cls = CLASS_TO_INT["BAD"] if total_shift >= 3 * std else CLASS_TO_INT["SUSPECT"]
        class_int[idx:end] = cls
        family[idx:end] = f"D_drift_d{d}"

    # E. sensor travado
    for idx in chunks[4]:
        d = int(rng.choice([3, 6, 12, 24]))
        end = min(n, idx + d)
        y_aug[idx:end] = y[idx]
        class_int[idx:end] = CLASS_TO_INT["BAD"]
        family[idx:end] = f"E_stuck_d{d}"

    # F. ruido em rajada
    for idx in chunks[5]:
        d = int(rng.choice([3, 6, 12, 24]))
        sigma_a = rng.uniform(0.5, 2.5) * std
        end = min(n, idx + d)
        noise = rng.normal(0, sigma_a, size=end - idx)
        y_aug[idx:end] = np.maximum(0.0, y[idx:end] + noise)
        cls = CLASS_TO_INT["BAD"] if sigma_a >= 1.5 * std else CLASS_TO_INT["SUSPECT"]
        class_int[idx:end] = cls
        family[idx:end] = f"F_burst_noise_d{d}"

    # G. falha de ausencia ou codigo sentinela (regra deterministica -> BAD)
    for idx in chunks[6]:
        choice = rng.choice(["nan", "sentinel", "negative"])
        if choice == "nan":
            y_aug[idx] = np.nan
        elif choice == "sentinel":
            y_aug[idx] = -999.0
        else:
            y_aug[idx] = -abs(rng.uniform(0.1, 1.0))
        class_int[idx] = CLASS_TO_INT["BAD"]
        family[idx] = f"G_missing_sentinel_{choice}"

    return y_aug, class_int, family


def weak_label_real_points(
    y_aug: np.ndarray,
    class_int: np.ndarray,
    residual: np.ndarray,
    spike: np.ndarray,
    robust_p: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weak GOOD/SUSPECT labels for points untouched by synthetic injection
    (protocol section 6.3). Returns (class_int, confidence)."""
    n = len(y_aug)
    confidence = np.where(class_int == CLASS_TO_INT["BAD"], 1.0, np.where(class_int == CLASS_TO_INT["SUSPECT"], 0.5, 0.0))
    untouched = class_int == -1

    near_gap = pd.Series(missing_mask).rolling(3, center=True, min_periods=1).max().to_numpy().astype(bool)

    good_strong = (
        untouched
        & (y_aug >= RANGE_MIN) & (y_aug <= RANGE_MAX)
        & (spike < np.nanpercentile(spike, 75))
        & (robust_p > 0.10)
        & (~near_gap)
    )
    suspect_weak = (
        untouched
        & ~good_strong
        & ((robust_p < 0.05) | (spike > np.nanpercentile(spike, 90)) | near_gap)
    )
    good_weak = untouched & ~good_strong & ~suspect_weak

    class_int = class_int.copy()
    class_int[good_strong] = CLASS_TO_INT["GOOD"]
    confidence[good_strong] = 1.0
    class_int[suspect_weak] = CLASS_TO_INT["SUSPECT"]
    confidence[suspect_weak] = 0.5
    class_int[good_weak] = CLASS_TO_INT["GOOD"]
    confidence[good_weak] = 0.5

    return class_int, confidence


# ---------------------------------------------------------------------------
# Statistical indicator vector s_t (protocol section 5.3)
# ---------------------------------------------------------------------------

def spike_score(y: np.ndarray) -> np.ndarray:
    s = pd.Series(y)
    return (s - (s.shift(1) + s.shift(-1)) / 2).abs().to_numpy()


def rate_score(y: np.ndarray) -> np.ndarray:
    s = pd.Series(y)
    return (s - s.shift(1)).abs().to_numpy()


def range_indicator(y: np.ndarray) -> np.ndarray:
    return ((y < RANGE_MIN) | (y > RANGE_MAX) | np.isnan(y)).astype(float)


def build_stat_vector(y_aug: np.ndarray, residual: np.ndarray, robust_scale: float, gpd_params: dict) -> dict[str, np.ndarray]:
    y_clean_for_stats = np.nan_to_num(y_aug, nan=np.nanmedian(y_aug))
    spike = spike_score(y_clean_for_stats)
    spike = np.nan_to_num(spike, nan=0.0)
    rate = rate_score(y_clean_for_stats)
    rate = np.nan_to_num(rate, nan=0.0)
    rng_ind = range_indicator(y_aug)

    residual_clean = np.nan_to_num(residual, nan=0.0)
    mad = qc04.robust_t_pvalues(residual_clean, robust_scale)
    mad_score = -np.log(mad + 1e-12)
    gpd_p = qc04.gpd_tail_pvalues(residual_clean, gpd_params)
    gpd_score = -np.log(gpd_p + 1e-12)

    return {
        "spike": spike,
        "rate": rate,
        "range": rng_ind,
        "mad_score": mad_score,
        "mad_pvalue": mad,
        "gpd_score": gpd_score,
        "gpd_pvalue": gpd_p,
    }


# ---------------------------------------------------------------------------
# Full pipeline per split: predict, inject, label, build features
# ---------------------------------------------------------------------------

def process_split(
    name: str,
    split_df: pd.DataFrame,
    mask_series: pd.Series,
    predictor: Model,
    encoder: Model,
    scaler: MinMaxScaler,
    selected_features: list[str],
    target_idx: int,
    inject_rate: float,
    anomaly_seed: int,
    robust_scale: float | None,
    gpd_params: dict | None,
) -> dict:
    scaled = scaler.transform(split_df)
    x_seq, y_seq_scaled, center_idx = make_sequences_single(scaled, target_idx, LOOKBACK)
    target_min = scaler.data_min_[target_idx]
    target_range = scaler.data_range_[target_idx]

    pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
    predicted = pred_scaled * target_range + target_min
    h = encoder.predict(x_seq, verbose=0)

    observed = y_seq_scaled * target_range + target_min
    timestamps = split_df.index[center_idx]
    mask_at_center = mask_series.to_numpy()[center_idx]

    y_aug, class_int, family = inject_anomalies(observed, inject_rate, anomaly_seed)
    residual = y_aug - predicted

    if robust_scale is None:
        diff_clean = residual
        robust_scale = qc04.fit_robust_scale(diff_clean)
        gpd_params = qc04.fit_gpd_tail(diff_clean)

    stats = build_stat_vector(y_aug, residual, robust_scale, gpd_params)
    class_int, confidence = weak_label_real_points(y_aug, class_int, residual, stats["spike"], stats["mad_pvalue"], mask_at_center)

    aux_features = [c for c in selected_features if c != TARGET]
    aux_idx = [selected_features.index(c) for c in aux_features]
    z_t = scaled[center_idx][:, aux_idx]

    delta_hsig = np.r_[np.nan, np.diff(np.nan_to_num(y_aug, nan=np.nanmedian(y_aug)))]
    delta_hsig = np.nan_to_num(delta_hsig, nan=0.0)

    return {
        "timestamps": timestamps,
        "observed": observed,
        "y_aug": y_aug,
        "predicted": predicted,
        "residual": residual,
        "abs_residual": np.abs(residual),
        "delta_hsig": delta_hsig,
        "h": h,
        "z_t": z_t,
        "mask": mask_at_center,
        "class_int": class_int,
        "confidence": confidence,
        "family": family,
        "stats": stats,
        "robust_scale": robust_scale,
        "gpd_params": gpd_params,
        "aux_features": aux_features,
    }


def assemble_feature_sets(d: dict) -> dict[str, np.ndarray]:
    stats = d["stats"]
    s_block = np.column_stack([stats["spike"], stats["mad_score"], stats["gpd_score"], stats["rate"], stats["range"]])
    residual_block = np.column_stack([d["y_aug"], d["predicted"], d["residual"], d["abs_residual"]])

    persistence_pred = d["y_aug"] - d["delta_hsig"]
    persistence_residual = d["y_aug"] - persistence_pred
    persistence_stats = np.column_stack(
        [
            spike_score(np.nan_to_num(d["y_aug"], nan=np.nanmedian(d["y_aug"]))),
            np.abs(persistence_residual),
            d["stats"]["range"],
        ]
    )
    persistence_stats = np.nan_to_num(persistence_stats, nan=0.0)

    return {
        "B_residual": residual_block,
        "C_residual_stats": np.column_stack([residual_block, s_block]),
        "D_hidden_residual": np.column_stack([d["h"], residual_block]),
        "E_full": np.column_stack([d["h"], residual_block, s_block, d["z_t"], d["mask"].reshape(-1, 1)]),
        "F_no_lstm": np.column_stack([d["y_aug"], d["z_t"], persistence_stats]),
    }


# ---------------------------------------------------------------------------
# Classifier head (protocol section 5.4)
# ---------------------------------------------------------------------------

def build_classifier(n_inputs: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            Dense(64, activation="relu", input_shape=(n_inputs,)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def train_classifier(
    x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    seed: int,
) -> tuple[tf.keras.Model, StandardScaler]:
    tf.random.set_seed(seed)
    feat_scaler = StandardScaler()
    x_train_s = feat_scaler.fit_transform(x_train)
    x_val_s = feat_scaler.transform(x_val)

    class_weights_arr = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
    class_weight_map = {i: w for i, w in enumerate(class_weights_arr)}
    sample_weight = np.array([class_weight_map[c] for c in y_train]) * w_train

    model = build_classifier(x_train.shape[1])
    model.fit(
        x_train_s, y_train,
        sample_weight=sample_weight,
        validation_data=(x_val_s, y_val, w_val),
        epochs=60, batch_size=64, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    )
    return model, feat_scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(axis=1)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    result = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1_good": f1_per_class[0],
        "f1_suspect": f1_per_class[1],
        "f1_bad": f1_per_class[2],
    }
    for c, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            result[f"auroc_{name.lower()}"] = roc_auc_score(y_bin, probs[:, c])
            result[f"auprc_{name.lower()}"] = average_precision_score(y_bin, probs[:, c])
        else:
            result[f"auroc_{name.lower()}"] = np.nan
            result[f"auprc_{name.lower()}"] = np.nan
        brier = np.mean((probs[:, c] - y_bin) ** 2)
        result[f"brier_{name.lower()}"] = brier
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return result


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidences)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Lightweight baseline comparison (BAD vs not-BAD), reusing qc04 detectors
# ---------------------------------------------------------------------------

def run_baselines(test_data: dict) -> dict[str, dict]:
    y_bin_true = (test_data["class_int"] == CLASS_TO_INT["BAD"]).astype(int)
    residual = test_data["residual"]
    y_aug = test_data["y_aug"]

    diff = qc04.diffmean(np.nan_to_num(y_aug, nan=np.nanmedian(y_aug)), test_data["predicted"])
    lstm_peak_flags = qc04.detect_lstm_peak(diff)

    robust_p = test_data["stats"]["mad_pvalue"]
    robust_flags = qc04.detect_probabilistic(robust_p)

    gpd_p = test_data["stats"]["gpd_pvalue"]
    gpd_flags = qc04.detect_probabilistic(gpd_p)

    traditional_flags = qc04.traditional_baseline(np.nan_to_num(y_aug, nan=np.nanmedian(y_aug)))

    iso_features = np.column_stack([np.nan_to_num(y_aug, nan=np.nanmedian(y_aug)), test_data["z_t"]])
    iso_model = IsolationForest(n_estimators=200, contamination=min(0.3, max(0.001, y_bin_true.mean())), random_state=MODEL_SEED)
    iso_model.fit(iso_features)
    iso_flags = (iso_model.predict(iso_features) == -1).astype(int)

    def bmetrics(y_pred):
        return {
            "precision": float((y_pred & y_bin_true).sum() / max(1, y_pred.sum())),
            "recall": float((y_pred & y_bin_true).sum() / max(1, y_bin_true.sum())),
            "f1": f1_score(y_bin_true, y_pred, zero_division=0),
        }

    return {
        "LSTM-Peak (H/T/d fixos)": bmetrics(lstm_peak_flags),
        "Robusto mediana/MAD": bmetrics(robust_flags),
        "GPD-POT": bmetrics(gpd_flags),
        "Spike tradicional": bmetrics(traditional_flags),
        "Isolation Forest": bmetrics(iso_flags),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_confusion_matrix_figure(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_probability_timeline_figure(test_data: dict, probs: np.ndarray, path: Path) -> None:
    n = min(1500, len(probs))
    ts = test_data["timestamps"][-n:]
    p = probs[-n:]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.stackplot(ts, p[:, 0], p[:, 1], p[:, 2], labels=CLASS_NAMES, colors=["#16875d", "#d17a22", "#b3261e"], alpha=0.85)
    ax.set_ylabel("Probabilidade")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(summary: dict, ablation_table: pd.DataFrame, baseline_table: pd.DataFrame, class_dist: pd.DataFrame) -> None:
    lines = [
        "# QC-LSTM de tres classes para Hsig (BA-1)",
        "",
        "Reproducao do protocolo `codigos/protocolo_teste_qc_lstm_3classes.md`. A LSTM deixa de ser apenas",
        "um preditor auxiliar de um detector de picos com regra fixa; sua previsao, seu estado oculto e",
        "indicadores estatisticos alimentam uma cabeca classificadora treinada que decide diretamente",
        "Q_t in {GOOD, SUSPECT, BAD}.",
        "",
        "## Divisao dos dados",
        "",
        f"- Treino {TRAIN_FRACTION:.0%}, validacao {VAL_FRACTION:.0%}, teste {1 - TRAIN_FRACTION - VAL_FRACTION:.0%}, estritamente cronologica.",
        f"- Taxa de injecao sintetica, treino {TRAIN_INJECT_RATE:.1%}, validacao/teste {TEST_INJECT_RATE:.1%}.",
        "",
        "## Desempenho preditivo da LSTM (fase A, previsao direta de passo unico)",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
        f"| MAE | {summary['mae']:.4f} |",
        f"| RMSE | {summary['rmse']:.4f} |",
        f"| MAPE (%) | {summary['mape']:.2f} |",
        "",
        "## Distribuicao das classes no teste",
        "",
        class_dist.to_markdown(index=False),
        "",
        "## Modelo completo (E_full) no conjunto de teste",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
        f"| Macro-F1 | {summary['full']['macro_f1']:.3f} |",
        f"| Weighted-F1 | {summary['full']['weighted_f1']:.3f} |",
        f"| Balanced accuracy | {summary['full']['balanced_accuracy']:.3f} |",
        f"| MCC | {summary['full']['mcc']:.3f} |",
        f"| F1 GOOD | {summary['full']['f1_good']:.3f} |",
        f"| F1 SUSPECT | {summary['full']['f1_suspect']:.3f} |",
        f"| F1 BAD | {summary['full']['f1_bad']:.3f} |",
        f"| AUPRC BAD | {summary['full']['auprc_bad']:.3f} |",
        f"| ECE | {summary['full']['ece']:.3f} |",
        "",
        "Matriz de confusao (linhas = real, colunas = predito, ordem GOOD/SUSPECT/BAD),",
        "",
        "```",
        str(summary["full"]["confusion_matrix"]),
        "```",
        "",
        "## Ablacao (Tabela 4 do protocolo, modelos B a F)",
        "",
        ablation_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Modelo B usa so residual, C acrescenta o vetor estatistico s_t, D troca s_t pelo estado oculto",
        "h_t da LSTM, E_full e o modelo completo (h_t + residual + s_t + variaveis auxiliares z_t + mascara",
        "m_t), F_no_lstm remove a LSTM inteiramente e usa apenas um residual de persistencia e as variaveis",
        "auxiliares observadas, para medir se a representacao temporal aprendida agrega informacao.",
        "",
        "## Comparacao com baselines binarios (deteccao de BAD)",
        "",
        "Os seis detectores originais (LSTM-Peak, robusto mediana/MAD, GPD-POT, spike tradicional,",
        "Isolation Forest) so produzem uma decisao binaria; aqui sao comparados apenas na deteccao da",
        "classe BAD do teste (VAE-LSTM omitido nesta rodada por custo computacional).",
        "",
        baseline_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Limitacoes desta rodada",
        "",
        "Esta primeira execucao cobre as fases A a C do protocolo (preditor, geracao de rotulos,",
        "cabeca classificadora com LSTM congelada) com uma unica semente de modelo e uma unica semente",
        "de injecao por particao. Ainda faltam a bateria de testes comportamentais (secao 8, monotonicidade,",
        "coerencia fisica contextual, causalidade, recuperacao, falha persistente), o teste de estabilidade",
        "multi-semente (secao 10.5) e o fine-tuning conjunto opcional (fase D).",
        "",
        "## Figuras geradas",
        "",
        "- `fig_matriz_confusao.png`: matriz de confusao do modelo completo no teste.",
        "- `fig_probabilidades_tempo.png`: probabilidades GOOD/SUSPECT/BAD ao longo do tempo (ultimos pontos do teste).",
    ]
    (OUT_DIR / "qc_lstm_3classes_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reusable pipeline (trains predictor + all ablation classifiers, returns
# every artifact needed both by main() and by downstream scripts such as the
# behavioral-test and multi-seed-stability suites, which import this module
# and call run_pipeline() instead of retraining from scratch by hand).
# ---------------------------------------------------------------------------

def run_pipeline(
    model_seed: int = MODEL_SEED,
    train_anomaly_seed: int = TRAIN_ANOMALY_SEED,
    val_anomaly_seed: int = VAL_ANOMALY_SEED,
    test_anomaly_seed: int = TEST_ANOMALY_SEED,
    train_inject_rate: float = TRAIN_INJECT_RATE,
    val_inject_rate: float = VAL_INJECT_RATE,
    test_inject_rate: float = TEST_INJECT_RATE,
    ablation_models: list[str] | None = None,
) -> dict:
    ablation_models = ablation_models if ablation_models is not None else ABLATION_MODELS

    qc04.set_reproducibility(model_seed)
    data, selected_features, corr, missing_mask = load_data()
    splits = chronological_split(data, missing_mask)
    train_df, train_mask = splits["train"]
    val_df, val_mask = splits["val"]
    test_df, test_mask = splits["test"]

    target_idx = selected_features.index(TARGET)
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    predictor, encoder = build_predictor(len(selected_features), LOOKBACK)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    x_train_pred, y_train_pred, _ = make_sequences_single(train_scaled, target_idx, LOOKBACK)
    x_val_pred, y_val_pred, _ = make_sequences_single(val_scaled, target_idx, LOOKBACK)
    predictor.fit(
        x_train_pred, y_train_pred,
        validation_data=(x_val_pred, y_val_pred),
        epochs=25, batch_size=32, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    d_train = process_split("train", train_df, train_mask, predictor, encoder, scaler, selected_features, target_idx,
                             train_inject_rate, train_anomaly_seed, None, None)
    d_val = process_split("val", val_df, val_mask, predictor, encoder, scaler, selected_features, target_idx,
                           val_inject_rate, val_anomaly_seed, d_train["robust_scale"], d_train["gpd_params"])
    d_test = process_split("test", test_df, test_mask, predictor, encoder, scaler, selected_features, target_idx,
                            test_inject_rate, test_anomaly_seed, d_train["robust_scale"], d_train["gpd_params"])

    mae = float(np.mean(np.abs(d_test["observed"] - d_test["predicted"])))
    rmse = float(np.sqrt(np.mean((d_test["observed"] - d_test["predicted"]) ** 2)))
    mape = float(np.nanmean(np.abs((d_test["observed"] - d_test["predicted"]) / np.where(np.abs(d_test["observed"]) < 1e-6, np.nan, d_test["observed"]))) * 100)

    feats_train = assemble_feature_sets(d_train)
    feats_val = assemble_feature_sets(d_val)
    feats_test = assemble_feature_sets(d_test)

    ablation_rows = []
    classifiers = {}
    full_eval = None
    full_probs = None
    for name in ablation_models:
        model, feat_scaler = train_classifier(
            feats_train[name], d_train["class_int"], d_train["confidence"],
            feats_val[name], d_val["class_int"], d_val["confidence"],
            seed=model_seed,
        )
        classifiers[name] = (model, feat_scaler)
        x_test_s = feat_scaler.transform(feats_test[name])
        probs = model.predict(x_test_s, verbose=0)
        ev = evaluate_classifier(d_test["class_int"], probs)
        ev["ece"] = expected_calibration_error(d_test["class_int"], probs)
        ablation_rows.append(
            {
                "config": name,
                "macro_f1": ev["macro_f1"],
                "f1_bad": ev["f1_bad"],
                "auprc_bad": ev["auprc_bad"],
                "ece": ev["ece"],
                "balanced_accuracy": ev["balanced_accuracy"],
            }
        )
        if name == "E_full":
            full_eval = ev
            full_probs = probs

    ablation_table = pd.DataFrame(ablation_rows)
    class_dist = pd.Series(d_test["class_int"]).map({v: k for k, v in CLASS_TO_INT.items()}).value_counts().rename_axis("classe").reset_index(name="n")

    baseline_metrics = run_baselines(d_test)
    baseline_table = pd.DataFrame(baseline_metrics).T.reset_index().rename(columns={"index": "metodo"})
    e_full_row = ablation_table[ablation_table["config"] == "E_full"].iloc[0]
    baseline_table = pd.concat(
        [
            baseline_table,
            pd.DataFrame([{"metodo": "QC-LSTM 3 classes (E_full, so BAD)", "precision": np.nan, "recall": np.nan, "f1": e_full_row["f1_bad"]}]),
        ],
        ignore_index=True,
    )

    return {
        "data": data,
        "selected_features": selected_features,
        "aux_features": [c for c in selected_features if c != TARGET],
        "corr": corr,
        "target_idx": target_idx,
        "scaler": scaler,
        "predictor": predictor,
        "encoder": encoder,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "d_train": d_train,
        "d_val": d_val,
        "d_test": d_test,
        "feats_train": feats_train,
        "feats_val": feats_val,
        "feats_test": feats_test,
        "classifiers": classifiers,
        "ablation_table": ablation_table,
        "class_dist": class_dist,
        "baseline_table": baseline_table,
        "full_eval": full_eval,
        "full_probs": full_probs,
        "summary": {"mae": mae, "rmse": rmse, "mape": mape, "full": full_eval},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = run_pipeline()
    d_test = artifacts["d_test"]
    full_eval = artifacts["full_eval"]
    full_probs = artifacts["full_probs"]
    ablation_table = artifacts["ablation_table"]
    baseline_table = artifacts["baseline_table"]
    class_dist = artifacts["class_dist"]
    summary = artifacts["summary"]

    save_confusion_matrix_figure(full_eval["confusion_matrix"], FIG_DIR / "fig_matriz_confusao.png")
    save_probability_timeline_figure(d_test, full_probs, FIG_DIR / "fig_probabilidades_tempo.png")

    per_obs = pd.DataFrame(
        {
            "timestamp": d_test["timestamps"],
            "Hsig_observed": d_test["y_aug"],
            "Hsig_predicted": d_test["predicted"],
            "prediction_residual": d_test["residual"],
            "absolute_residual": d_test["abs_residual"],
            "p_good": full_probs[:, 0],
            "p_suspect": full_probs[:, 1],
            "p_bad": full_probs[:, 2],
            "qc_label": [CLASS_NAMES[i] for i in full_probs.argmax(axis=1)],
            "qc_confidence": full_probs.max(axis=1),
            "true_label": [CLASS_NAMES[i] for i in d_test["class_int"]],
            "label_confidence": d_test["confidence"],
            "family": d_test["family"],
            "stat_spike": d_test["stats"]["spike"],
            "stat_mad": d_test["stats"]["mad_score"],
            "stat_gpd": d_test["stats"]["gpd_score"],
            "missing_mask": d_test["mask"],
            "model_version": "qc_lstm_3classes_v1",
        }
    )
    per_obs.to_csv(OUT_DIR / "qc_lstm_3classes_resultados_teste.csv", index=False)
    ablation_table.to_csv(OUT_DIR / "ablacao_modelos_b_a_f.csv", index=False)
    baseline_table.to_csv(OUT_DIR / "comparacao_baselines_bad.csv", index=False)
    class_dist.to_csv(OUT_DIR / "distribuicao_classes_teste.csv", index=False)
    pd.DataFrame([{"mae": summary["mae"], "rmse": summary["rmse"], "mape": summary["mape"]}]).to_csv(OUT_DIR / "metricas_predicao_lstm.csv", index=False)

    write_report(summary, ablation_table, baseline_table, class_dist)

    print("Resultados em:", OUT_DIR)
    print(ablation_table.to_string(index=False))
    print(class_dist.to_string(index=False))


if __name__ == "__main__":
    main()
