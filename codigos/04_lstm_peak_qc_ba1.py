from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import genpareto
from scipy.stats import t as student_t
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Layer, LSTM, Reshape
from tensorflow.keras.models import Model, Sequential


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
WIND_DATA_PATH = BASE_DIR / "dadosSimcosta" / "ERA5_BA-1_WIND_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "lstm_peak_qc"
FIG_DIR = OUT_DIR / "figures"

TARGET = "Hsig"
# HM0/Hmax/Havg/H10 are other order-statistics of the same wave burst as Hsig
# (r > 0.98 by construction) - excluded from auxiliary-feature selection to
# avoid a circular "predict Hsig from near-duplicates of Hsig" regression.
HSIG_DUPLICATE_FEATURES = {"HM0", "Hmax", "Havg", "H10"}
WAVE_FEATURES = [
    "Hsig", "HM0", "Hmax", "Havg", "H10", "Tp", "Tp5", "Tsig", "Tavg", "T10",
    "Avg_Wv_Dir_N", "Avg_Wv_Spread_N",
]
WATER_FEATURES = ["Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2", "Avg_Turb", "Avg_Chl", "Avg_CDOM"]
# ERA5 reanalysis proxy (codigos/00_fetch_wind_era5_ba1.py) - not in-situ buoy
# wind, but the physically independent covariate the reference paper uses
# (wind speed) that the BA-1 OCEAN file itself does not report.
WIND_FEATURES = ["wind_speed_10m", "wind_gusts_10m"]
LOOKBACK = 24
TEST_FRACTION = 0.30
ANOMALY_RATIO = 1 / 130
MAX_HOURLY_RECORDS = 8760

# The paper's peak-detection thresholds (H=1, T=1, d=50) are fixed constants,
# not calibrated to any particular site's residual distribution. ALPHA and
# GPD_TAIL_QUANTILE instead calibrate two probabilistic alternatives on BA-1's
# own DiffMean distribution, fit on the *clean* (pre-injection) diff so the
# "normal" reference isn't contaminated by the synthetic anomalies it is later
# asked to detect.
ALPHA = 0.01
GPD_TAIL_QUANTILE = 0.90
ROBUST_T_DOF = 4

# The paper averages 5 independent anomaly-injection replications per site
# (Section 3.2/4.3) instead of reporting a single noisy draw. Mirrored here:
# for each trained model, every anomaly seed draws its own independent set of
# ~20 synthetic anomalies and every QC method is scored against it.
ANOMALY_SEEDS = [42, 43, 44, 45, 46]

# The anomaly-seed loop above only tests robustness to *which points* get
# corrupted; it says nothing about whether the "robust median/MAD wins" result
# depends on this one particular trained LSTM. MODEL_SEEDS retrains the model
# from scratch for each seed (fresh weight init + fresh EarlyStopping split),
# recalibrates the probabilistic tests on that model's own residuals, and
# reruns the full ANOMALY_SEEDS loop - the real test of whether the
# conclusion survives a second source of randomness.
MODEL_SEEDS = [42, 7, 123]

# ALPHA=0.01 was picked, not derived. ALPHA_GRID re-thresholds the already
# -computed p-values (no retraining/recalibration needed) to check how
# precision/recall/F1 move as alpha varies, instead of reporting one
# arbitrary operating point.
ALPHA_GRID = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]

# The two baselines the paper itself compares LSTM-Peak against (Section 3.3)
# but that were missing from this reproduction until now: Isolation Forest
# (Liu et al. 2009) and VAE-LSTM (Lin et al. 2020, cited by the paper).
# Isolation Forest is fit only on the clean training period, contamination
# set to the paper's own class-imbalance ratio (Section 3.2, ~1:130) - the
# same domain estimate ANOMALY_RATIO already encodes for injection.
ISO_FOREST_N_ESTIMATORS = 200

# VAE-LSTM: a VAE reconstructs short windows (local features), an LSTM
# models the temporal correlation between the VAE's latent codes (long-term
# structure); anomaly signal = reconstruction error + latent-prediction
# error, calibrated the same way as the robust-t/GPD tests (clean period
# only) so all three probabilistic-style methods share one decision rule.
VAE_WINDOW = LOOKBACK
VAE_LATENT_DIM = 8
VAE_KL_WEIGHT = 1e-3
VAE_EPOCHS = 40
LATENT_LOOKBACK = 5
LATENT_LSTM_UNITS = 32
LATENT_LSTM_EPOCHS = 40


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, comment="/", na_values=["NULL", "null", "", "NaN"])
    df["datetime_utc"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc").set_index("datetime_utc")
    for col in WAVE_FEATURES + WATER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_wind_data() -> pd.DataFrame:
    wind = pd.read_csv(WIND_DATA_PATH, comment="/", na_values=["NULL", "null", "", "NaN"])
    wind["datetime_utc"] = pd.to_datetime(wind["datetime_utc"], utc=True)
    wind = wind.set_index("datetime_utc")
    for col in WIND_FEATURES:
        wind[col] = pd.to_numeric(wind[col], errors="coerce")
    return wind[WIND_FEATURES]


def build_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in WAVE_FEATURES + WATER_FEATURES if c in df.columns]
    hourly = df[available].resample("1h").median()
    hourly = hourly.join(read_wind_data(), how="left")
    hourly = hourly.interpolate(limit=3, limit_direction="both")
    hourly = hourly.dropna(subset=[TARGET])
    return hourly


def select_features(hourly: pd.DataFrame) -> tuple[list[str], pd.Series]:
    candidates = [c for c in hourly.columns if c != TARGET]
    corr = hourly[[TARGET] + candidates].corr(method="pearson")[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    eligible = corr.drop(index=[c for c in HSIG_DUPLICATE_FEATURES if c in corr.index])
    selected = [TARGET] + eligible.head(5).index.tolist()
    return selected, corr


def make_sequences(values: np.ndarray, target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Input window must stop at t-2 (last included index end-2): the 3-step
    # target {t-1, t, t+1} has to stay fully unseen by the input, otherwise
    # the model gets t-1 "for free" and part of the averaged prediction is
    # not a real forecast.
    x_seq, y_seq, center_idx = [], [], []
    for end in range(LOOKBACK + 1, len(values) - 1):
        x_seq.append(values[end - 1 - LOOKBACK:end - 1, :])
        y_seq.append(values[end - 1:end + 2, target_idx])
        center_idx.append(end)
    return np.asarray(x_seq), np.asarray(y_seq), np.asarray(center_idx)


def build_model(n_features: int) -> Sequential:
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(0.2),
            Dense(3, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def inject_point_anomalies(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Uses its own local Generator (not the global numpy RNG seeded for model
    # training) so each replication seed is independently reproducible
    # regardless of call order.
    rng = np.random.default_rng(seed)
    y_bad = y.copy()
    labels = np.zeros(len(y_bad), dtype=int)
    n_anom = max(1, int(round(len(y_bad) * ANOMALY_RATIO)))
    possible = np.arange(10, len(y_bad) - 10)
    chosen = rng.choice(possible, size=n_anom, replace=False)
    normal_std = np.nanstd(y_bad)
    for idx in chosen:
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(3.0, 6.0) * normal_std
        y_bad[idx] = max(0.0, y_bad[idx] + direction * magnitude)
        labels[idx] = 1
    return y_bad, labels


def diffmean(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    prev_x = np.r_[np.nan, observed[:-1]]
    next_x = np.r_[observed[1:], np.nan]
    local_mean = np.nanmean(np.vstack([prev_x, observed, next_x]), axis=0)
    local_mean = np.where(np.abs(local_mean) < 1e-6, np.nan, local_mean)
    return np.abs(observed - predicted) / local_mean


def detect_lstm_peak(diff: np.ndarray) -> np.ndarray:
    clean_diff = pd.Series(diff).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    peaks_up, _ = find_peaks(clean_diff, height=1, distance=50, prominence=1)
    peaks_down, _ = find_peaks(-clean_diff, height=1, distance=50, prominence=1)
    flags = np.zeros(len(clean_diff), dtype=int)
    flags[np.r_[peaks_up, peaks_down]] = 1
    return flags


def _clean_series(diff: np.ndarray) -> np.ndarray:
    return pd.Series(diff).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()


def fit_robust_scale(diff_clean: np.ndarray, dof: int = ROBUST_T_DOF) -> float:
    # DiffMean is already |e|/local_mean (non-negative); treat it as a folded
    # (half) Student-t and back out the scale from the robust median, so the
    # "typical error" reference comes from BA-1's own residuals instead of a
    # constant copied from the paper's 4 Chinese stations.
    clean = _clean_series(diff_clean)
    median = np.median(clean)
    return float(median / student_t.ppf(0.75, dof))


def robust_t_pvalues(diff: np.ndarray, scale: float, dof: int = ROBUST_T_DOF) -> np.ndarray:
    clean = _clean_series(diff)
    z = clean / scale
    return 2 * student_t.sf(z, dof)


def fit_gpd_tail(diff_clean: np.ndarray, quantile: float = GPD_TAIL_QUANTILE) -> dict:
    # Peaks-over-threshold: model the upper tail of DiffMean with a
    # Generalized Pareto Distribution instead of assuming a light-tailed
    # (Gaussian/Student-t) shape everywhere - the standard extreme-value-theory
    # way to get calibrated tail probabilities without guessing the bulk shape.
    clean = _clean_series(diff_clean)
    threshold = float(np.quantile(clean, quantile))
    exceedances = clean[clean > threshold] - threshold
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    return {
        "threshold": threshold,
        "shape": float(shape),
        "scale": float(scale),
        "p_exceed_threshold": float(np.mean(clean > threshold)),
        "clean_sorted": np.sort(clean),
    }


def gpd_tail_pvalues(diff: np.ndarray, gpd_params: dict) -> np.ndarray:
    clean = _clean_series(diff)
    threshold = gpd_params["threshold"]
    sorted_clean = gpd_params["clean_sorted"]
    pvals = np.empty_like(clean, dtype=float)

    above = clean > threshold
    pvals[above] = gpd_params["p_exceed_threshold"] * genpareto.sf(
        clean[above] - threshold, gpd_params["shape"], loc=0, scale=gpd_params["scale"]
    )
    below = ~above
    ranks = np.searchsorted(sorted_clean, clean[below], side="left")
    pvals[below] = 1.0 - ranks / len(sorted_clean)
    return pvals


def detect_probabilistic(pvalues: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return (pvalues < alpha).astype(int)


def fit_isolation_forest(train_features_scaled: np.ndarray, seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=ISO_FOREST_N_ESTIMATORS,
        contamination=ANOMALY_RATIO,
        random_state=seed,
    )
    model.fit(train_features_scaled)
    return model


def detect_isolation_forest(model: IsolationForest, features_scaled: np.ndarray) -> np.ndarray:
    return (model.predict(features_scaled) == -1).astype(int)


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(Layer):
    # Keras 3's functional API forbids raw tf.* ops directly on symbolic
    # KerasTensors outside a Layer.call - the ELBO (reconstruction + KL) has
    # to be computed inside a Layer for add_loss() to be legal here.
    def call(self, tensors):
        inputs, outputs, z_mean, z_log_var = tensors
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=[1, 2]))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(recon_loss + VAE_KL_WEIGHT * kl_loss)
        return outputs


def build_vae(window: int, n_features: int, latent_dim: int = VAE_LATENT_DIM) -> tuple[Model, Model]:
    inputs = Input(shape=(window, n_features))
    x = Flatten()(inputs)
    x = Dense(64, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="vae_encoder")

    latent_inputs = Input(shape=(latent_dim,))
    y = Dense(64, activation="relu")(latent_inputs)
    y = Dense(window * n_features, activation="linear")(y)
    outputs = Reshape((window, n_features))(y)
    decoder = Model(latent_inputs, outputs, name="vae_decoder")

    z_mean_out, z_log_var_out, z_out = encoder(inputs)
    reconstruction = decoder(z_out)
    vae_outputs = VAELossLayer()([inputs, reconstruction, z_mean_out, z_log_var_out])
    vae = Model(inputs, vae_outputs, name="vae")
    vae.compile(optimizer="adam")
    return vae, encoder


def build_latent_lstm(latent_dim: int) -> Sequential:
    model = Sequential(
        [
            LSTM(LATENT_LSTM_UNITS, input_shape=(LATENT_LOOKBACK, latent_dim)),
            Dense(latent_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def make_windows_ending_at(features: np.ndarray, end_indices: np.ndarray, window: int) -> np.ndarray:
    return np.stack([features[e - window + 1 : e + 1] for e in end_indices])


def make_latent_sequences(z_seq: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(z_seq)):
        x.append(z_seq[i - lookback : i])
        y.append(z_seq[i])
    return np.asarray(x), np.asarray(y)


def vae_lstm_reconstruction_error(vae: Model, windows: np.ndarray, target_idx: int) -> np.ndarray:
    # Scored on the target feature at the window's last timestep only, so the
    # score lines up point-for-point with every other method's timestamp t.
    recon = vae.predict(windows, verbose=0)
    return (windows[:, -1, target_idx] - recon[:, -1, target_idx]) ** 2


def vae_lstm_latent_error(latent_lstm: Sequential, z_context: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
    # z_context supplies the LATENT_LOOKBACK previous latent codes needed to
    # score the first point of z_seq without looking into the future.
    z_full = np.vstack([z_context, z_seq])
    x_lat, y_lat = make_latent_sequences(z_full, LATENT_LOOKBACK)
    z_pred = latent_lstm.predict(x_lat, verbose=0)
    return np.mean((y_lat - z_pred) ** 2, axis=1)


def calibrate_vae_lstm_score(error_recon_clean: np.ndarray, error_latent_clean: np.ndarray) -> dict:
    recon_scale = float(np.median(error_recon_clean)) or 1e-8
    latent_scale = float(np.median(error_latent_clean)) or 1e-8
    combined_clean = error_recon_clean / recon_scale + error_latent_clean / latent_scale
    robust_scale = fit_robust_scale(combined_clean)
    return {"recon_scale": recon_scale, "latent_scale": latent_scale, "robust_scale": robust_scale}


def vae_lstm_pvalues(error_recon: np.ndarray, error_latent: np.ndarray, calibration: dict) -> np.ndarray:
    combined = error_recon / calibration["recon_scale"] + error_latent / calibration["latent_scale"]
    return robust_t_pvalues(combined, calibration["robust_scale"])


def evaluate_replication(
    observed: np.ndarray,
    predicted: np.ndarray,
    robust_scale: float,
    gpd_params: dict,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    observed_with_anom, labels = inject_point_anomalies(observed, seed)
    diff = diffmean(observed_with_anom, predicted)

    lstm_peak_flags = detect_lstm_peak(diff)
    traditional_flags = traditional_baseline(observed_with_anom)
    robust_pvalues = robust_t_pvalues(diff, robust_scale)
    robust_flags = detect_probabilistic(robust_pvalues)
    gpd_pvalues = gpd_tail_pvalues(diff, gpd_params)
    gpd_flags = detect_probabilistic(gpd_pvalues)

    qc_metrics = {
        "LSTM-Peak (artigo, H/T/d fixos)": metrics(labels, lstm_peak_flags),
        "Robusto mediana/MAD (Student-t)": metrics(labels, robust_flags),
        "GPD-POT (cauda)": metrics(labels, gpd_flags),
        "Spike tradicional robusto": metrics(labels, traditional_flags),
    }
    detail = {
        "observed_with_anom": observed_with_anom,
        "labels": labels,
        "diff": diff,
        "lstm_peak_flags": lstm_peak_flags,
        "traditional_flags": traditional_flags,
        "robust_pvalues": robust_pvalues,
        "robust_flags": robust_flags,
        "gpd_pvalues": gpd_pvalues,
        "gpd_flags": gpd_flags,
    }
    return qc_metrics, detail


def alpha_sweep_rows(labels: np.ndarray, pvalues: np.ndarray, method: str, model_seed: int, anomaly_seed: int) -> list[dict]:
    # Re-thresholds already-computed p-values at each candidate alpha - cheap,
    # no need to refit or retrain anything.
    rows = []
    for alpha in ALPHA_GRID:
        flags = detect_probabilistic(pvalues, alpha)
        values = metrics(labels, flags)
        rows.append({"model_seed": model_seed, "anomaly_seed": anomaly_seed, "method": method, "alpha": alpha, **values})
    return rows


def traditional_baseline(observed: np.ndarray) -> np.ndarray:
    s = pd.Series(observed)
    classic = (s - (s.shift(1) + s.shift(-1)) / 2).abs()
    threshold = classic.median() + 6 * 1.4826 * (classic - classic.median()).abs().median()
    flags = (classic > threshold).fillna(False).astype(int).to_numpy()
    return flags


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "detected": int(y_pred.sum()),
        "true_anomalies": int(y_true.sum()),
    }


def save_figures(results: pd.DataFrame, corr: pd.Series, qc_agg: pd.DataFrame, alpha_agg: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    corr.sort_values().plot(kind="barh", color="#3f6f8f")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("r")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_correlacao_hsig_variaveis.png", dpi=180)
    plt.close()

    sample = results.tail(min(1200, len(results)))
    plt.figure(figsize=(13, 5))
    plt.plot(sample["timestamp"], sample["observed"], label="Observed Hsig", color="#25364a", linewidth=1)
    plt.plot(sample["timestamp"], sample["predicted"], label="LSTM predicted Hsig", color="#d17a22", linewidth=1)
    bad = sample[sample["is_injected_anomaly"] == 1]
    plt.scatter(bad["timestamp"], bad["observed"], color="#b3261e", s=18, label="Injected anomaly", zorder=3)
    flagged = sample[sample["lstm_peak_flag"] == 1]
    plt.scatter(flagged["timestamp"], flagged["observed"], facecolors="none", edgecolors="#16875d", s=45, label="LSTM-Peak flag", zorder=4)
    plt.ylabel("Hsig")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_hsig_observado_previsto_flags.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 4))
    plt.plot(sample["timestamp"], sample["diffmean"], color="#5b4b8a", linewidth=1, zorder=1)
    lstm_flagged = sample[sample["lstm_peak_flag"] == 1]
    robust_flagged = sample[sample["robust_t_flag"] == 1]
    gpd_flagged = sample[sample["gpd_pot_flag"] == 1]
    plt.scatter(lstm_flagged["timestamp"], lstm_flagged["diffmean"], facecolors="none", edgecolors="#b3261e", s=70, linewidths=1.5, label="LSTM-Peak (artigo)", zorder=3)
    plt.scatter(robust_flagged["timestamp"], robust_flagged["diffmean"], marker="x", color="#16875d", s=45, label="Robusto mediana/MAD", zorder=4)
    plt.scatter(gpd_flagged["timestamp"], gpd_flagged["diffmean"], marker="+", color="#d17a22", s=60, label="GPD-POT", zorder=5)
    plt.axhline(1, color="black", linewidth=0.8, linestyle="--", label="H=1 (limiar fixo do artigo)")
    plt.ylabel("DiffMean")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_diffmean_peak_detection.png", dpi=180)
    plt.close()

    methods = qc_agg.index.tolist()
    x = np.arange(len(methods))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, qc_agg[("precision", "mean")], width, yerr=qc_agg[("precision", "std")], capsize=3, label="Precision", color="#3f6f8f")
    plt.bar(x, qc_agg[("recall", "mean")], width, yerr=qc_agg[("recall", "std")], capsize=3, label="Recall", color="#d17a22")
    plt.bar(x + width, qc_agg[("f1", "mean")], width, yerr=qc_agg[("f1", "std")], capsize=3, label="F1", color="#16875d")
    plt.xticks(x, methods, rotation=20, ha="right", fontsize=8)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_comparacao_metodos_qc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    colors = {"Robusto mediana/MAD (Student-t)": "#16875d", "GPD-POT (cauda)": "#d17a22", "VAE-LSTM": "#7b4fa6"}
    for method, color in colors.items():
        sub = alpha_agg.loc[method].sort_index()
        plt.plot(sub.index, sub[("precision", "mean")], marker="o", color=color, linestyle="-", label=f"{method} - Precision")
        plt.plot(sub.index, sub[("recall", "mean")], marker="s", color=color, linestyle="--", label=f"{method} - Recall")
        plt.plot(sub.index, sub[("f1", "mean")], marker="^", color=color, linestyle=":", label=f"{method} - F1")
    plt.axvline(ALPHA, color="black", linewidth=0.8, linestyle="--", label=f"α usado ({ALPHA})")
    plt.xscale("log")
    plt.xlabel("α (nível de significância)")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_sensibilidade_alpha.png", dpi=180)
    plt.close()


def write_report(summary: dict[str, object], selected_features: list[str], corr: pd.Series) -> None:
    lines = [
        "# Teste LSTM-Peak inspirado em Xie et al. (2023) para BA-1",
        "",
        "## Objetivo",
        "",
        "Testar, de forma controlada, a proposta do artigo `QC.pdf`: usar uma LSTM multivariada para prever a altura significativa de onda e aplicar detecção estatística de picos sobre a diferença entre valor observado e valor previsto.",
        "",
        "## Adaptação para a boia BA-1",
        "",
        "O artigo usa altura significativa de onda e velocidade do vento. O arquivo OCEAN da BA-1 não contém vento medido pela boia. Para aproximar a variável física usada no artigo, este teste incorpora vento horário de reanálise ERA5 (via API Open-Meteo) no ponto de grade mais próximo das coordenadas da boia — ver `codigos/00_fetch_wind_era5_ba1.py` e `dadosSimcosta/ERA5_BA-1_WIND_2019-08-02_2026-07-25.csv`. Isso NÃO é vento medido in situ: é uma estimativa de modelo em grade de ~0.25°, a 10 m de altura (o anemômetro da SiMCosta, quando presente, mede a 3 m), portanto tratada como uma fragilidade metodológica documentada, não como equivalente ao vento do artigo original. Para evitar um segundo problema — selecionar automaticamente `HM0`, `Hmax`, `Havg` ou `H10` como variável auxiliar, que são apenas outras estatísticas de ordem do mesmo registro de onda que gera `Hsig` (r>0.98) e tornariam a previsão quase tautológica — essas variáveis são excluídas do conjunto candidato; a seleção das 5 variáveis auxiliares ocorre entre variáveis fisicamente distintas (vento ERA5, período, direção/espalhamento de onda e variáveis da coluna d'água).",
        "",
        f"- Variável-alvo: `{TARGET}`",
        f"- Variáveis usadas no modelo: {', '.join(f'`{c}`' for c in selected_features)}",
        f"- Janela temporal de entrada: {LOOKBACK} horas",
        f"- Fração de teste: {TEST_FRACTION:.0%}",
        f"- Taxa de anomalias artificiais: aproximadamente 1:{int(round(1 / ANOMALY_RATIO))}",
        "",
        "## Correlações com Hsig",
        "",
        "| Variável | r de Pearson |",
        "|---|---:|",
    ]
    for var, value in corr.items():
        lines.append(f"| {var} | {value:.3f} |")

    lines.extend(
        [
            "",
            f"## Desempenho preditivo da LSTM (média ± desvio-padrão, {summary['n_model_seeds']} modelos retreinados)",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| MAE | {summary['mae']:.4f} ± {summary['mae_std']:.4f} |",
            f"| MSE | {summary['mse']:.4f} |",
            f"| RMSE | {summary['rmse']:.4f} |",
            f"| MAPE (%) | {summary['mape']:.2f} ± {summary['mape_std']:.2f} |",
            "",
            f"## Desempenho do QC com anomalias artificiais (média ± desvio-padrão, {summary['n_model_seeds']} modelos × {summary['n_anomaly_seeds']} réplicas = {summary['n_model_seeds'] * summary['n_anomaly_seeds']} execuções)",
            "",
            "Vai além do protocolo do artigo (Seção 3.2/4.3, que só varia as anomalias sintéticas): aqui a LSTM também é retreinada do zero em cada modelo (init de pesos e split de EarlyStopping diferentes), para checar se a vantagem de um método sobre outro depende de qual modelo treinado calhou de ser usado, e não só de qual conjunto de anomalias foi sorteado.",
            "",
            "| Método | Precision | Recall | F1 | Flags detectadas |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    qc_agg = summary["qc_metrics_agg"]
    for method in qc_agg.index:
        p_m, p_s = qc_agg.loc[method, ("precision", "mean")], qc_agg.loc[method, ("precision", "std")]
        r_m, r_s = qc_agg.loc[method, ("recall", "mean")], qc_agg.loc[method, ("recall", "std")]
        f_m, f_s = qc_agg.loc[method, ("f1", "mean")], qc_agg.loc[method, ("f1", "std")]
        d_m, d_s = qc_agg.loc[method, ("detected", "mean")], qc_agg.loc[method, ("detected", "std")]
        lines.append(
            f"| {method} | {p_m:.3f} ± {p_s:.3f} | {r_m:.3f} ± {r_s:.3f} | "
            f"{f_m:.3f} ± {f_s:.3f} | {d_m:.1f} ± {d_s:.1f} |"
        )

    lines.extend(
        [
            "",
            "### Consistência por modelo retreinado (F1 médio das 5 réplicas de anomalia, por modelo)",
            "",
            "| Método | " + " | ".join(f"Modelo seed={s}" for s in MODEL_SEEDS) + " |",
            "|---|" + "---:|" * len(MODEL_SEEDS),
        ]
    )
    qc_by_model = summary["qc_metrics_by_model"]
    for method in qc_agg.index:
        row_vals = []
        for s in MODEL_SEEDS:
            f_m = qc_by_model.loc[(method, s), ("f1", "mean")]
            row_vals.append(f"{f_m:.3f}")
        lines.append(f"| {method} | " + " | ".join(row_vals) + " |")

    calib = summary["calibration"]
    lines.extend(
        [
            "",
            "## Teste de erro probabilístico (extensão além do artigo)",
            "",
            "O artigo decide `Q_t=0`/`Q_t=1` com constantes fixas (`H=1, T=1, d=50`), iguais para as 4 estações chinesas, sem calibração por site nem controle formal de taxa de falso positivo. Aqui isso é substituído/comparado por dois testes estatísticos calibrados na própria distribuição de DiffMean da BA-1, ajustados apenas no período *limpo* (antes da injeção de anomalias artificiais) para não contaminar a referência do que é 'erro normal' com as próprias anomalias que o teste precisa detectar:",
            "",
            "1. **Robusto mediana/MAD (Student-t dobrada)**: como o DiffMean já é `|erro|/média local` (não-negativo), ele é tratado como uma Student-t dobrada; a escala é obtida a partir da mediana do DiffMean limpo (robusta a outliers) e `dof` fixo (mais cauda pesada que Gaussiana). Cada ponto recebe uma probabilidade de cauda; sinaliza-se `Q_t=0` se p < α.",
            "2. **GPD-POT (peaks-over-threshold)**: a cauda superior do DiffMean limpo (acima do percentil 90) é ajustada com uma Distribuição de Pareto Generalizada — a abordagem padrão da teoria de valores extremos para estimar probabilidade de cauda sem supor a forma da distribuição inteira. Mais apropriado se o erro tiver cauda mais pesada do que a Student-t consegue capturar.",
            "",
            f"- α (nível de significância): {calib['alpha']}",
            f"- Robusto mediana/MAD: dof={calib['robust_t_dof']}, escala ajustada={calib['robust_scale']:.4f}",
            f"- GPD-POT: limiar (percentil {calib['gpd_threshold_quantile']:.0%})={calib['gpd_threshold']:.4f}, forma (ξ)={calib['gpd_shape']:.4f}, escala={calib['gpd_scale']:.4f}, P(DiffMean>limiar)={calib['gpd_p_exceed_threshold']:.4f}",
            "",
            "(Parâmetros de calibração acima são do modelo de referência, seed={}; cada um dos {} modelos retreinados recalibra os dois testes nos seus próprios resíduos.)".format(MODEL_SEEDS[0], len(MODEL_SEEDS)),
            "",
            "## Baselines nativos do artigo: Isolation Forest e VAE-LSTM",
            "",
            "O artigo (Seção 3.3) compara o LSTM-Peak com dois outros modelos, que estavam ausentes desta reprodução até agora e foram adicionados:",
            "",
            "3. **Isolation Forest** (Liu et al., 2009): ajustado apenas no período de treino limpo (nunca vê as anomalias injetadas), com `contamination` igual à mesma proporção 1:130 que o próprio artigo usa para a regra de injeção (Seção 3.2). Aplicado sobre o vetor de features do timestamp de teste (alvo + variáveis auxiliares), reajustado a cada um dos modelos retreinados.",
            "4. **VAE-LSTM** (Lin et al., 2020, citado pelo artigo): um VAE reconstrói janelas curtas (features locais) e uma LSTM modela a correlação de longo prazo entre os códigos latentes do VAE; o sinal de anomalia combina o erro de reconstrução do alvo com o erro de predição no espaço latente. Em vez de um limiar arbitrário, esse escore combinado é calibrado com o mesmo teste Student-t robusto (mediana/MAD) usado acima, ajustado só no período limpo — o mesmo tratamento estatístico aplicado de forma uniforme onde há um escore contínuo de anomalia.",
            "",
            f"- VAE-LSTM: escala de reconstrução={calib.get('vae_recon_scale', float('nan')):.6f}, escala do preditor latente={calib.get('vae_latent_scale', float('nan')):.6f}, escala robusta do escore combinado={calib.get('vae_robust_scale', float('nan')):.4f}",
            "",
            "## Sensibilidade a α",
            "",
            f"α={calib['alpha']} foi escolhido, não derivado. `fig05_sensibilidade_alpha.png` e `sensibilidade_alpha.csv` mostram como Precision/Recall/F1 dos três testes calibrados por α (robusto mediana/MAD, GPD-POT, VAE-LSTM) respondem a α em {ALPHA_GRID}, para checar se a conclusão é sensível à escolha específica de α ou se é estável num intervalo razoável.",
            "",
            "## Interpretação geral",
            "",
            "**Ressalva de leitura importante**: as anomalias sintéticas injetadas são grandes (3-6× o desvio-padrão de Hsig), então quase todo método consegue recall alto - o que de fato diferencia os métodos aqui é a taxa de falso positivo em cima de dados reais (precision), não a capacidade de detectar um desvio grande. Uma comparação com anomalias sutis (mais próximas do limiar de detecção) tende a separar melhor os métodos pela capacidade de detecção, não só pela robustez a ruído normal.",
            "",
            "Este teste não prova ainda um método final para a BA-1; ele avalia se a lógica do artigo é transferível para os dados disponíveis. A principal diferença metodológica é a ausência de vento no arquivo OCEAN. Como vento é a variável física usada no artigo para auxiliar a previsão de altura de onda, a BA-1 depende aqui de relações internas entre variáveis de onda e da coluna d'água. Isso ainda pode reduzir a capacidade de distinguir evento físico real de falha instrumental, mesmo com as estatísticas de ordem redundantes com `Hsig` (`HM0`, `Hmax`, `Havg`, `H10`) já excluídas da seleção de variáveis auxiliares.",
            "",
            "Se a LSTM apresentar erro preditivo baixo, mas o LSTM-Peak tiver baixo recall, isso indica que o modelo reproduz a série, mas o critério de pico no resíduo não está capturando bem as anomalias artificiais. Se o recall for alto e a precision baixa, o método é sensível, mas gera muitos falsos positivos. Se ambos forem baixos, a proposta do artigo não se transfere diretamente para a BA-1 sem vento, calibração regional ou mudança de modelo.",
            "",
            "## Figuras geradas",
            "",
            "- `fig01_correlacao_hsig_variaveis.png`: seleção das variáveis auxiliares.",
            "- `fig02_hsig_observado_previsto_flags.png`: comparação observado, previsto, anomalias artificiais e flags do LSTM-Peak.",
            "- `fig03_diffmean_peak_detection.png`: razão de diferença local com flags dos 3 métodos baseados em DiffMean (LSTM-Peak, Robusto mediana/MAD, GPD-POT).",
            "- `fig04_comparacao_metodos_qc.png`: Precision/Recall/F1 dos 6 métodos lado a lado (média ± desvio-padrão sobre modelos × réplicas).",
            "- `fig05_sensibilidade_alpha.png`: Precision/Recall/F1 dos 3 testes calibrados por α (Robusto mediana/MAD, GPD-POT, VAE-LSTM).",
        ]
    )
    (OUT_DIR / "lstm_peak_qc_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_reproducibility(MODEL_SEEDS[0])
    df = read_data()
    hourly = build_hourly_dataset(df)
    selected_features, corr = select_features(hourly)
    data = hourly[selected_features].dropna().tail(MAX_HOURLY_RECORDS)

    split = int(len(data) * (1 - TEST_FRACTION))
    train_df = data.iloc[:split]
    test_df = data.iloc[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    target_idx = selected_features.index(TARGET)

    x_train, y_train, _ = make_sequences(train_scaled, target_idx)
    x_test, y_test_scaled, center_idx = make_sequences(test_scaled, target_idx)
    observed_scaled = y_test_scaled[:, 1]
    target_min = scaler.data_min_[target_idx]
    target_range = scaler.data_range_[target_idx]
    observed = observed_scaled * target_range + target_min

    # Clean (pre-injection) feature snapshot at every test timestamp used by
    # Isolation Forest - only the target column gets swapped for the
    # anomaly-injected values inside the replication loop below.
    aux_matrix_test_clean_scaled = test_scaled[center_idx, :]

    replication_rows = []
    alpha_rows = []
    prediction_rows = []
    reference_detail = None
    reference_predicted = None
    reference_calibration = None

    for model_seed in MODEL_SEEDS:
        # Retrains from scratch (fresh weight init, fresh EarlyStopping split)
        # so the QC comparison isn't resting on a single lucky trained model -
        # this is the second, independent source of randomness beyond which
        # points get corrupted (ANOMALY_SEEDS).
        set_reproducibility(model_seed)
        model = build_model(len(selected_features))
        model.fit(
            x_train,
            y_train,
            epochs=12,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        pred_scaled_3 = model.predict(x_test, verbose=0)
        pred_scaled = pred_scaled_3.mean(axis=1)
        predicted = pred_scaled * target_range + target_min

        mape = np.nanmean(np.abs((observed - predicted) / np.where(np.abs(observed) < 1e-6, np.nan, observed))) * 100
        prediction_rows.append(
            {
                "model_seed": model_seed,
                "mae": mean_absolute_error(observed, predicted),
                "mse": mean_squared_error(observed, predicted),
                "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
                "mape": float(mape),
            }
        )

        # Calibrate the two probabilistic tests on this model's own clean
        # (pre-injection) diff, then reuse across all anomaly replications for
        # this model - otherwise the "normal" reference would itself be built
        # from the same synthetic anomalies it is later asked to detect.
        diff_clean = diffmean(observed, predicted)
        robust_scale = fit_robust_scale(diff_clean)
        gpd_params = fit_gpd_tail(diff_clean)

        # Isolation Forest (paper's Section 3.3 baseline): fit only on the
        # clean training period, contamination = the paper's own 1:130 class
        # ratio - retrained per model_seed so its own randomness (tree
        # bootstrap) is covered by the same robustness check as the LSTM.
        iso_model = fit_isolation_forest(train_scaled, model_seed)

        # VAE-LSTM (paper's other Section 3.3 baseline): VAE on clean windows
        # first (local features + reconstruction), then an LSTM over the
        # VAE's own latent codes (long-term correlation) - both retrained
        # from scratch per model_seed, same as the main LSTM above.
        vae_train_windows = make_windows_ending_at(
            train_scaled, np.arange(VAE_WINDOW - 1, len(train_scaled)), VAE_WINDOW
        )
        vae, vae_encoder = build_vae(VAE_WINDOW, len(selected_features))
        vae.fit(
            vae_train_windows,
            epochs=VAE_EPOCHS,
            batch_size=64,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        z_train_mean, _, _ = vae_encoder.predict(vae_train_windows, verbose=0)

        x_lat_train, y_lat_train = make_latent_sequences(z_train_mean, LATENT_LOOKBACK)
        latent_lstm = build_latent_lstm(VAE_LATENT_DIM)
        latent_lstm.fit(
            x_lat_train,
            y_lat_train,
            epochs=LATENT_LSTM_EPOCHS,
            batch_size=64,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        z_train_mean_tail = z_train_mean[-LATENT_LOOKBACK:]

        error_recon_train = vae_lstm_reconstruction_error(vae, vae_train_windows, target_idx)
        z_pred_train = latent_lstm.predict(x_lat_train, verbose=0)
        error_latent_train = np.mean((y_lat_train - z_pred_train) ** 2, axis=1)
        # The first LATENT_LOOKBACK windows have no latent-history yet, so
        # trim the reconstruction series to line up with the latent-error one
        # before calibrating the two jointly.
        vae_calibration = calibrate_vae_lstm_score(error_recon_train[LATENT_LOOKBACK:], error_latent_train)

        for anomaly_seed in ANOMALY_SEEDS:
            qc_metrics_i, detail_i = evaluate_replication(observed, predicted, robust_scale, gpd_params, anomaly_seed)

            observed_with_anom_scaled = (detail_i["observed_with_anom"] - target_min) / target_range

            iso_features = aux_matrix_test_clean_scaled.copy()
            iso_features[:, target_idx] = observed_with_anom_scaled
            iso_flags = detect_isolation_forest(iso_model, iso_features)
            qc_metrics_i["Isolation Forest"] = metrics(detail_i["labels"], iso_flags)
            detail_i["iso_forest_flags"] = iso_flags

            test_scaled_corrupted = test_scaled.copy()
            test_scaled_corrupted[center_idx, target_idx] = observed_with_anom_scaled
            vae_windows = make_windows_ending_at(test_scaled_corrupted, center_idx, VAE_WINDOW)
            error_recon = vae_lstm_reconstruction_error(vae, vae_windows, target_idx)
            z_mean_test, _, _ = vae_encoder.predict(vae_windows, verbose=0)
            error_latent = vae_lstm_latent_error(latent_lstm, z_train_mean_tail, z_mean_test)
            vae_pvalues = vae_lstm_pvalues(error_recon, error_latent, vae_calibration)
            vae_flags = detect_probabilistic(vae_pvalues)
            qc_metrics_i["VAE-LSTM"] = metrics(detail_i["labels"], vae_flags)
            detail_i["vae_lstm_pvalue"] = vae_pvalues
            detail_i["vae_lstm_flag"] = vae_flags

            for method, values in qc_metrics_i.items():
                replication_rows.append({"model_seed": model_seed, "anomaly_seed": anomaly_seed, "method": method, **values})

            alpha_rows.extend(alpha_sweep_rows(detail_i["labels"], detail_i["robust_pvalues"], "Robusto mediana/MAD (Student-t)", model_seed, anomaly_seed))
            alpha_rows.extend(alpha_sweep_rows(detail_i["labels"], detail_i["gpd_pvalues"], "GPD-POT (cauda)", model_seed, anomaly_seed))
            alpha_rows.extend(alpha_sweep_rows(detail_i["labels"], vae_pvalues, "VAE-LSTM", model_seed, anomaly_seed))

            if model_seed == MODEL_SEEDS[0] and anomaly_seed == ANOMALY_SEEDS[0]:
                reference_detail = detail_i
                reference_predicted = predicted
                reference_calibration = {
                    "robust_scale": robust_scale,
                    "gpd_threshold": gpd_params["threshold"],
                    "gpd_shape": gpd_params["shape"],
                    "gpd_scale": gpd_params["scale"],
                    "gpd_p_exceed_threshold": gpd_params["p_exceed_threshold"],
                    "vae_recon_scale": vae_calibration["recon_scale"],
                    "vae_latent_scale": vae_calibration["latent_scale"],
                    "vae_robust_scale": vae_calibration["robust_scale"],
                }

    replication_df = pd.DataFrame(replication_rows)
    qc_agg = replication_df.groupby("method")[["precision", "recall", "f1", "detected"]].agg(["mean", "std"])
    qc_agg_by_model = replication_df.groupby(["method", "model_seed"])[["precision", "recall", "f1", "detected"]].agg(["mean", "std"])

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_agg = alpha_df.groupby(["method", "alpha"])[["precision", "recall", "f1"]].agg(["mean", "std"])

    prediction_df = pd.DataFrame(prediction_rows)

    timestamps = test_df.index[center_idx]
    results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "observed_original": observed,
            "observed": reference_detail["observed_with_anom"],
            "predicted": reference_predicted,
            "residual": reference_detail["observed_with_anom"] - reference_predicted,
            "diffmean": reference_detail["diff"],
            "is_injected_anomaly": reference_detail["labels"],
            "lstm_peak_flag": reference_detail["lstm_peak_flags"],
            "traditional_spike_flag": reference_detail["traditional_flags"],
            "robust_t_pvalue": reference_detail["robust_pvalues"],
            "robust_t_flag": reference_detail["robust_flags"],
            "gpd_pot_pvalue": reference_detail["gpd_pvalues"],
            "gpd_pot_flag": reference_detail["gpd_flags"],
            "iso_forest_flag": reference_detail["iso_forest_flags"],
            "vae_lstm_pvalue": reference_detail["vae_lstm_pvalue"],
            "vae_lstm_flag": reference_detail["vae_lstm_flag"],
        }
    )

    summary = {
        "mae": prediction_df["mae"].mean(),
        "mse": prediction_df["mse"].mean(),
        "rmse": prediction_df["rmse"].mean(),
        "mape": prediction_df["mape"].mean(),
        "mae_std": prediction_df["mae"].std(),
        "mape_std": prediction_df["mape"].std(),
        "n_anomaly_seeds": len(ANOMALY_SEEDS),
        "n_model_seeds": len(MODEL_SEEDS),
        "qc_metrics_agg": qc_agg,
        "qc_metrics_by_model": qc_agg_by_model,
        "calibration": {
            "alpha": ALPHA,
            "robust_t_dof": ROBUST_T_DOF,
            "gpd_threshold_quantile": GPD_TAIL_QUANTILE,
            **reference_calibration,
        },
    }

    corr.to_csv(OUT_DIR / "correlacao_hsig_variaveis.csv", header=["pearson_r"])
    replication_df.to_csv(OUT_DIR / "metricas_qc_replicacoes.csv", index=False)
    qc_agg.to_csv(OUT_DIR / "metricas_qc_multiseed.csv")
    qc_agg_by_model.to_csv(OUT_DIR / "metricas_qc_por_modelo.csv")
    alpha_df.to_csv(OUT_DIR / "sensibilidade_alpha_replicacoes.csv", index=False)
    alpha_agg.to_csv(OUT_DIR / "sensibilidade_alpha.csv")
    prediction_df.to_csv(OUT_DIR / "metricas_predicao_lstm_por_modelo.csv", index=False)
    pd.DataFrame([summary["calibration"]]).T.rename(columns={0: "value"}).to_csv(OUT_DIR / "calibracao_testes_probabilisticos.csv")
    pd.DataFrame(
        [
            {"metric": "MAE", "value": summary["mae"]},
            {"metric": "MSE", "value": summary["mse"]},
            {"metric": "RMSE", "value": summary["rmse"]},
            {"metric": "MAPE_percent", "value": summary["mape"]},
        ]
    ).to_csv(OUT_DIR / "metricas_predicao_lstm.csv", index=False)
    results.to_csv(OUT_DIR / "lstm_peak_qc_resultados.csv", index=False)
    save_figures(results, corr, qc_agg, alpha_agg)
    write_report(summary, selected_features, corr)

    print("Resultados em:", OUT_DIR)


if __name__ == "__main__":
    main()
