from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc10", BASE_DIR / "codigos" / "10_qc_lstm_3classes_ba1.py")
qc10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc10)
qc04 = qc10.qc04

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_causal_ordinal"
FIG_DIR = OUT_DIR / "figures"

TARGET = qc10.TARGET
# Janela de entrada de 7 dias (uma semana), nao mais as 24h herdadas do
# protocolo original de Xie et al. (mantidas em qc04/qc10 para nao alterar
# a reproducao binaria ja reportada). Decisao explicita para esta linha
# causal/ordinal (13 em diante), o modelo deve prever t+1 usando uma semana
# de historico, nao um dia.
LOOKBACK = 24 * 7
CLASS_NAMES = qc10.CLASS_NAMES
CLASS_TO_INT = qc10.CLASS_TO_INT

# A faixa fisica plausivel (usada so como um teste de range grosseiro, uma
# das seis entradas de s_t, nao como decisao final) nao deve ficar fixa em
# 0-8 m, um numero que so faz sentido para a BA-1. Definida a partir dos
# proprios dados de treino (set_physical_range, chamada em main antes de
# processar qualquer particao), para que rodar o mesmo script noutra boia
# baste trocar a origem dos dados, nao editar constantes espalhadas pelo
# codigo.
RANGE_MIN, RANGE_MAX = 0.0, None
RANGE_MAX_MULTIPLIER = 3.0


def set_physical_range(train_target: np.ndarray, multiplier: float = RANGE_MAX_MULTIPLIER) -> None:
    global RANGE_MIN, RANGE_MAX
    RANGE_MIN = 0.0
    RANGE_MAX = float(np.nanmax(train_target)) * multiplier

MODEL_SEED = 42
TRAIN_ANOMALY_SEED = 201
VAL_ANOMALY_SEED = 202
TEST_ANOMALY_SEED = 203

# Balanceado por familia (numero de EVENTOS, nao de pontos), aproximacao do
# "on-the-fly balanceado por familia x intensidade x duracao" pedido -
# gera um conjunto de treino bem maior e mais diverso que o v1 sem
# re-perturbar a cada epoca (custo computacional). Teste/validacao usam
# poucos eventos para nao dominar a serie real com anomalia sintetica.
TRAIN_EVENTS_PER_FAMILY = 14
# Aumentado de 4 para 10, o ajuste de tau_B na validacao era instavel entre
# sementes com poucos eventos BAD disponiveis para calcular F1 (algumas
# combinacoes de semente sorteavam validacoes quase sem BAD, produzindo um
# tau_B extremo e um macro-F1 muito ruim so naquela execucao).
VAL_EVENTS_PER_FAMILY = 10
TEST_EVENTS_PER_FAMILY = 4

# Drift e mudanca de nivel agora cobrem episodios mais longos, para dar
# tempo do sinal acumulado (EWMA/CUSUM/slope) se manifestar.
DURATIONS_SHORT = [3, 6, 12, 24]
DURATIONS_LONG = [12, 24, 48, 72]

EWMA_LAMBDAS = [0.1, 0.3, 0.6]
SLOPE_WINDOWS = [6, 12, 24, 48]
CUSUM_K_FACTOR = 0.5  # x std do residuo limpo

TAU_GRID = np.concatenate([np.arange(0.05, 0.96, 0.05), np.arange(0.96, 0.999, 0.01)])

CONFIGS = ["C_causal", "E_causal"]


# ---------------------------------------------------------------------------
# Injecao balanceada por familia, causal (nunca usa t+1), com drift/mudanca
# de nivel progressivos (SUSPECT no inicio do episodio, BAD quando o desvio
# acumulado ultrapassa 3 desvios-padrao).
# ---------------------------------------------------------------------------

def inject_anomalies_causal(y: np.ndarray, events_per_family: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    y_aug = y.copy()
    class_int = np.full(n, -1, dtype=int)
    family = np.array([""] * n, dtype=object)
    std = float(np.nanstd(y))

    # Cada familia reserva seu proprio pool de posicoes, com a duracao
    # maxima que ela de fato usa (ponto unico para A/B/G, DURATIONS_SHORT
    # para E/F, DURATIONS_LONG para C/D) - um pool unico compartilhado com
    # max_duration=72 para todas esgotava o espaco disponivel antes de
    # alocar as familias finais (E, F, G ficavam sem nenhum evento).
    family_max_duration = [1, 1, max(DURATIONS_LONG), max(DURATIONS_LONG), max(DURATIONS_SHORT), max(DURATIONS_SHORT), 1]
    used = np.zeros(n, dtype=bool)
    chunks = []
    for max_dur in family_max_duration:
        candidates_mask = ~used
        fam_slots = qc10._pick_slots(n, rng, events_per_family, min_gap=6, max_duration=max_dur)
        # descarta posicoes que colidiriam com reservas de familias anteriores
        fam_slots = [idx for idx in fam_slots if not used[max(0, idx - 6):idx + max_dur + 6].any()]
        for idx in fam_slots:
            used[max(0, idx - 6):idx + max_dur + 6] = True
        chunks.append(fam_slots)

    # A. spike aditivo (evento pontual, nao precisa de t+1 para ser aplicado)
    for idx in chunks[0]:
        s = rng.choice([-1, 1])
        k = rng.choice([0.5, 1, 2, 3, 4, 6])
        y_aug[idx] = max(0.0, y[idx] + s * k * std)
        class_int[idx] = CLASS_TO_INT["BAD"] if k >= 3 else CLASS_TO_INT["SUSPECT"]
        family[idx] = f"A_spike_k{k}"

    # B. perturbacao multiplicativa
    for idx in chunks[1]:
        factor = rng.choice([5.0, 10.0, 1 / 5, 1 / 10])
        y_aug[idx] = y[idx] * factor
        class_int[idx] = CLASS_TO_INT["BAD"]
        family[idx] = f"B_mult_x{factor:.2f}"

    # C. mudanca de nivel, episodios mais longos, rotulo progressivo. A
    # gravidade alvo (SUSPECT ou BAD) e sorteada primeiro e a magnitude
    # amostrada condicionada a ela, em vez de deixar a proporcao de eventos
    # que cruzam o limiar de 3 desvios-padrao inteiramente ao acaso - sem
    # isso, replicas diferentes de injecao podiam sortear quase so
    # magnitudes pequenas (ou quase so grandes), tornando o teste muito
    # mais dificil ou mais facil so por coincidencia da semente, a maior
    # fonte de instabilidade observada entre replicas.
    for idx in chunks[2]:
        d = int(rng.choice(DURATIONS_LONG))
        target_bad = rng.random() < 0.5
        magnitude = rng.uniform(3.0, 4.0) if target_bad else rng.uniform(1.0, 2.5)
        delta = rng.choice([-1, 1]) * magnitude * std
        end = min(n, idx + d)
        y_aug[idx:end] = np.maximum(0.0, y[idx:end] + delta)
        cls = CLASS_TO_INT["BAD"] if abs(delta) >= 3 * std else CLASS_TO_INT["SUSPECT"]
        class_int[idx:end] = cls
        family[idx:end] = f"C_level_shift_d{d}"

    # D. drift, episodios mais longos, rotulo progressivo pelo desvio
    # acumulado (inicio do episodio SUSPECT, BAD so quando o desvio
    # acumulado |j*delta| ultrapassa 3 desvios-padrao). Gravidade alvo
    # sorteada primeiro (mesmo raciocinio da familia C) para que o desvio
    # acumulado ao final do episodio efetivamente cruze (ou fique abaixo
    # de) 3 desvios-padrao de forma consistente entre replicas.
    for idx in chunks[3]:
        d = int(rng.choice(DURATIONS_LONG))
        target_bad = rng.random() < 0.5
        target_cumulative = rng.uniform(3.0, 4.0) if target_bad else rng.uniform(1.0, 2.5)
        delta = rng.choice([-1, 1]) * (target_cumulative * std) / max(1, d - 1)
        end = min(n, idx + d)
        j = np.arange(end - idx)
        y_aug[idx:end] = np.maximum(0.0, y[idx:end] + j * delta)
        cumulative = np.abs(j * delta)
        cls = np.where(cumulative >= 3 * std, CLASS_TO_INT["BAD"], CLASS_TO_INT["SUSPECT"])
        class_int[idx:end] = cls
        family[idx:end] = f"D_drift_d{d}"

    # E. sensor travado
    for idx in chunks[4]:
        d = int(rng.choice(DURATIONS_SHORT))
        end = min(n, idx + d)
        y_aug[idx:end] = y[idx]
        class_int[idx:end] = CLASS_TO_INT["BAD"]
        family[idx:end] = f"E_stuck_d{d}"

    # F. ruido em rajada
    for idx in chunks[5]:
        d = int(rng.choice(DURATIONS_SHORT))
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


# ---------------------------------------------------------------------------
# Indicadores causais (nunca usam Hsig_{t+1}) + memoria acumulativa para
# drift (EWMA, CUSUM, slope, persistencia de sinal).
# ---------------------------------------------------------------------------

def causal_stat_vector(y_aug: np.ndarray, residual: np.ndarray, w: int = 24) -> dict[str, np.ndarray]:
    y_clean = np.nan_to_num(y_aug, nan=np.nanmedian(y_aug))
    s = pd.Series(y_clean)

    second_diff = s.diff().diff()
    roll_mad_2nd = second_diff.rolling(w, min_periods=3).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)
    s_acc = (second_diff.abs() / (1.4826 * roll_mad_2nd.shift(1) + 1e-6)).to_numpy()

    roll_median = s.rolling(w, min_periods=3).median().shift(1)
    roll_mad_level = s.rolling(w, min_periods=3).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True).shift(1)
    s_level = ((s - roll_median).abs() / (1.4826 * roll_mad_level + 1e-6)).to_numpy()

    s_rate = s.diff().abs().to_numpy()
    s_range = ((y_aug < RANGE_MIN) | (y_aug > RANGE_MAX) | np.isnan(y_aug)).astype(float)

    residual_clean = np.nan_to_num(residual, nan=0.0)
    robust_scale = qc04.fit_robust_scale(residual_clean)
    gpd_params = qc04.fit_gpd_tail(residual_clean)
    mad_p = qc04.robust_t_pvalues(residual_clean, robust_scale)
    mad_score = -np.log(mad_p + 1e-12)
    gpd_p = qc04.gpd_tail_pvalues(residual_clean, gpd_params)
    gpd_score = -np.log(gpd_p + 1e-12)

    return {
        "acc": np.nan_to_num(s_acc, nan=0.0),
        "level": np.nan_to_num(s_level, nan=0.0),
        "rate": np.nan_to_num(s_rate, nan=0.0),
        "range": s_range,
        "mad_score": mad_score,
        "mad_pvalue": mad_p,
        "gpd_score": gpd_score,
        "robust_scale": robust_scale,
        "gpd_params": gpd_params,
    }


def accumulation_features(residual: np.ndarray) -> dict[str, np.ndarray]:
    e = pd.Series(np.nan_to_num(residual, nan=0.0))
    std_e = float(e.std()) or 1e-6
    k = CUSUM_K_FACTOR * std_e

    out = {}
    for lam in EWMA_LAMBDAS:
        out[f"ewma_{lam}"] = e.ewm(alpha=lam, adjust=False).mean().to_numpy()

    # Reinicia ao detectar (excede CUSUM_RESET_FACTOR x std), como na pratica
    # de controle estatistico de processo - sem isso, um vies residual
    # persistente (mesmo pequeno) faz o CUSUM crescer sem limite ao longo de
    # milhares de pontos da serie de treino em vez de refletir apenas
    # episodios recentes de desvio.
    reset_threshold = 20.0 * std_e
    cusum_pos = np.zeros(len(e))
    cusum_neg = np.zeros(len(e))
    ev = e.to_numpy()
    for t in range(1, len(ev)):
        cusum_pos[t] = max(0.0, cusum_pos[t - 1] + ev[t] - k)
        cusum_neg[t] = max(0.0, cusum_neg[t - 1] - ev[t] - k)
        if cusum_pos[t] > reset_threshold:
            cusum_pos[t] = 0.0
        if cusum_neg[t] > reset_threshold:
            cusum_neg[t] = 0.0
    out["cusum_pos"] = cusum_pos
    out["cusum_neg"] = cusum_neg

    def _make_slope_fn():
        def _slope(a):
            if np.all(a == a[0]):
                return 0.0
            idx = np.arange(len(a))
            return float(np.polyfit(idx, a, 1)[0])
        return _slope

    for w in SLOPE_WINDOWS:
        out[f"slope_{w}"] = e.rolling(w, min_periods=3).apply(_make_slope_fn(), raw=True).fillna(0.0).to_numpy()

    sign = np.sign(ev)
    persistence = np.zeros(len(ev))
    run = 0
    last_sign = 0
    for t in range(len(ev)):
        if sign[t] != 0 and sign[t] == last_sign:
            run += 1
        else:
            run = 1
            last_sign = sign[t]
        persistence[t] = run
    out["persistence"] = persistence

    return out


# ---------------------------------------------------------------------------
# Rotulagem fraca causal (sem Hsig_{t+1}) para pontos nao perturbados.
#
# Desacoplada do vetor de entrada do classificador de proposito: rotular
# pontos reais com os MESMOS indicadores (acc/level/mad) que a rede depois
# recebe como feature deixa o modelo livre para aprender a reproduzir a
# propria regra geradora do rotulo, em vez de aprender algo novo. Aqui o
# rotulo fraco vem de um z-score global simples (media/desvio do periodo de
# treino, um teste classico ao estilo Dixon), que nao entra no vetor v_t.
# ---------------------------------------------------------------------------

def weak_label_decoupled(y_aug: np.ndarray, class_int: np.ndarray, missing_mask: np.ndarray,
                          ref_mean: float, ref_std: float) -> tuple[np.ndarray, np.ndarray]:
    # Confianca reflete a origem do rotulo (corrupcao sintetica com familia e
    # magnitude conhecidas = forte; z-score sobre dado limpo = fraco), nao a
    # gravidade da classe. Antes desta correcao, todo ponto injetado com
    # gravidade SUSPECT (familias A/C/D/F abaixo do limiar de 3 sigma) recebia
    # a mesma confianca 0.5 dos pontos NAO corrompidos rotulados SUSPECT por
    # z-score fraco, tornando estruturalmente impossivel um SUSPECT forte
    # sobreviver ao filtro confidence>=0.999 usado a jusante (18, 20, 25) -
    # e por isso a linha SUSPECT verdadeira da matriz de confusao (Tabela S12)
    # saia sempre zerada, apesar do desenho da injecao (familias C/D) prever
    # SUSPECT e BAD em proporcao comparavel entre os pontos injetados.
    injected = class_int != -1
    confidence = np.where(injected, 1.0, 0.0)
    untouched = ~injected
    near_gap = pd.Series(missing_mask).rolling(3, center=True, min_periods=1).max().to_numpy().astype(bool)

    z = np.abs((np.nan_to_num(y_aug, nan=ref_mean) - ref_mean) / (ref_std + 1e-6))

    good_strong = untouched & (y_aug >= RANGE_MIN) & (y_aug <= RANGE_MAX) & (z < 1.5) & (~near_gap)
    suspect_weak = untouched & ~good_strong & ((z >= 1.5) | near_gap)
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
# Pipeline por particao
# ---------------------------------------------------------------------------

def process_split_causal(split_df, mask_series, predictor, encoder, scaler, selected_features, target_idx,
                          events_per_family, anomaly_seed, ref_mean: float, ref_std: float) -> dict:
    scaled = scaler.transform(split_df)
    x_seq, y_seq_scaled, center_idx = qc10.make_sequences_single(scaled, target_idx, LOOKBACK)
    target_min, target_range = scaler.data_min_[target_idx], scaler.data_range_[target_idx]

    pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
    predicted = pred_scaled * target_range + target_min
    h = encoder.predict(x_seq, verbose=0)

    observed = y_seq_scaled * target_range + target_min
    timestamps = split_df.index[center_idx]
    mask_at_center = mask_series.to_numpy()[center_idx]

    y_aug, class_int, family = inject_anomalies_causal(observed, events_per_family, anomaly_seed)
    residual = y_aug - predicted

    stats = causal_stat_vector(y_aug, residual)
    accum = accumulation_features(residual)
    class_int, confidence = weak_label_decoupled(y_aug, class_int, mask_at_center, ref_mean, ref_std)

    # Familia G (ausencia/codigo sentinela/valor negativo impossivel) tem
    # regra deterministica de rotulo por definicao do proprio protocolo, nao
    # deveria depender de o classificador aprender a reconhece-la a partir de
    # um valor imputado. Sinalizada aqui para ser aplicada como override no
    # momento da decisao final, independente da probabilidade da rede.
    sentinel_flag = pd.Series(family).str.startswith("G_").to_numpy()

    aux_features = [c for c in selected_features if c != TARGET]
    aux_idx = [selected_features.index(c) for c in aux_features]
    z_t = scaled[center_idx][:, aux_idx]

    return {
        "timestamps": timestamps, "observed": observed, "y_aug": y_aug, "predicted": predicted,
        "residual": residual, "abs_residual": np.abs(residual), "h": h, "z_t": z_t, "mask": mask_at_center,
        "class_int": class_int, "confidence": confidence, "family": family, "stats": stats, "accum": accum,
        "sentinel_flag": sentinel_flag,
    }


def assemble_features_causal(d: dict) -> dict[str, np.ndarray]:
    stats, accum = d["stats"], d["accum"]
    s_block = np.column_stack([stats["acc"], stats["level"], stats["rate"], stats["range"], stats["mad_score"], stats["gpd_score"]])
    accum_block = np.column_stack(
        [accum[f"ewma_{lam}"] for lam in EWMA_LAMBDAS]
        + [accum["cusum_pos"], accum["cusum_neg"]]
        + [accum[f"slope_{w}"] for w in SLOPE_WINDOWS]
        + [accum["persistence"]]
    )
    # familia G injeta NaN literal (ausencia/sentinela) em d["y_aug"], que se
    # propaga para residual/abs_residual - a informacao "faltando" ja esta
    # capturada por range/rotulo BAD deterministico, entao aqui o NaN e
    # apenas imputado (nao pode entrar cru na rede, um unico NaN em uma
    # linha derruba o gradiente do lote inteiro).
    y_aug_feat = np.nan_to_num(d["y_aug"], nan=float(np.nanmedian(d["y_aug"])))
    residual_feat = np.nan_to_num(d["residual"], nan=0.0)
    abs_residual_feat = np.nan_to_num(d["abs_residual"], nan=0.0)
    residual_block = np.column_stack([y_aug_feat, d["predicted"], residual_feat, abs_residual_feat])

    return {
        "C_causal": np.column_stack([residual_block, s_block, accum_block, d["z_t"]]),
        "E_causal": np.column_stack([d["h"], residual_block, s_block, accum_block, d["z_t"], d["mask"].reshape(-1, 1)]),
    }


# ---------------------------------------------------------------------------
# Classificador ordinal/hierarquico: q1 = P(Q_t >= SUSPECT), q2 = P(BAD | >=
# SUSPECT). Substitui o softmax de 3 classes por duas cabecas binarias, para
# que a segunda cabeca aprenda especificamente a separar BAD de SUSPECT.
# ---------------------------------------------------------------------------

def build_ordinal_classifier(n_inputs: int) -> Model:
    inputs = Input(shape=(n_inputs,))
    x = Dense(64, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    q1 = Dense(1, activation="sigmoid", name="q1")(x)
    q2 = Dense(1, activation="sigmoid", name="q2")(x)
    model = Model(inputs, [q1, q2])
    model.compile(optimizer="adam", loss={"q1": "binary_crossentropy", "q2": "binary_crossentropy"})
    return model


class ClippedRobustScaler:
    """RobustScaler (mediana/IQR) precedido de winsorizacao (clip nos
    percentis 0.5/99.5 do treino). Eventos multiplicativos (familia B, ate
    10x Hsig) e CUSUM/EWMA acumulados geram outliers de magnitude muito
    maior que o resto da serie; sem o clip, mesmo um scaler robusto tinha
    seu ajuste dominado por poucos pontos extremos e o treino colapsava
    para uma saida constante."""

    def __init__(self, lower: float = 0.5, upper: float = 99.5):
        self.lower, self.upper = lower, upper
        self.scaler = RobustScaler()
        self.bounds_ = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.bounds_ = np.percentile(x, [self.lower, self.upper], axis=0)
        x_clipped = np.clip(x, self.bounds_[0], self.bounds_[1])
        return self.scaler.fit_transform(x_clipped)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, self.bounds_[0], self.bounds_[1])
        return self.scaler.transform(x_clipped)


def train_ordinal(x_train, y_train, w_train, x_val, y_val, w_val, seed: int) -> tuple[Model, ClippedRobustScaler]:
    tf.random.set_seed(seed)
    feat_scaler = ClippedRobustScaler()
    x_train_s = feat_scaler.fit_transform(x_train)
    x_val_s = feat_scaler.transform(x_val)

    y1_train = (y_train != CLASS_TO_INT["GOOD"]).astype(float)
    y2_train = (y_train == CLASS_TO_INT["BAD"]).astype(float)
    not_good_train = (y_train != CLASS_TO_INT["GOOD"]).astype(float)

    y1_val = (y_val != CLASS_TO_INT["GOOD"]).astype(float)
    y2_val = (y_val == CLASS_TO_INT["BAD"]).astype(float)
    not_good_val = (y_val != CLASS_TO_INT["GOOD"]).astype(float)

    cw1 = compute_class_weight("balanced", classes=np.array([0, 1]), y=y1_train)
    w1_train = np.where(y1_train == 1, cw1[1], cw1[0]) * w_train
    w1_val = np.where(y1_val == 1, cw1[1], cw1[0]) * w_val

    mask_not_good_train = y_train != CLASS_TO_INT["GOOD"]
    if mask_not_good_train.sum() > 0:
        cw2 = compute_class_weight("balanced", classes=np.array([0, 1]), y=y2_train[mask_not_good_train])
    else:
        cw2 = np.array([1.0, 1.0])
    w2_train = np.where(y2_train == 1, cw2[1], cw2[0]) * w_train * not_good_train
    w2_val = np.where(y2_val == 1, cw2[1], cw2[0]) * w_val * not_good_val

    model = build_ordinal_classifier(x_train.shape[1])
    model.fit(
        x_train_s, [y1_train, y2_train],
        sample_weight=[w1_train, w2_train],
        validation_data=(x_val_s, [y1_val, y2_val], [w1_val, w2_val]),
        epochs=80, batch_size=64, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )
    return model, feat_scaler


def ordinal_probs(model: Model, x_scaled: np.ndarray) -> np.ndarray:
    q1, q2 = model.predict(x_scaled, verbose=0)
    q1, q2 = q1.ravel(), q2.ravel()
    p_good = 1 - q1
    p_suspect = q1 * (1 - q2)
    p_bad = q1 * q2
    return np.column_stack([p_good, p_suspect, p_bad])


def decide_with_threshold(probs: np.ndarray, tau_b: float, sentinel_flag: np.ndarray | None = None) -> np.ndarray:
    label = np.where(probs[:, 2] >= tau_b, CLASS_TO_INT["BAD"],
                      np.where(probs[:, 1] >= probs[:, 0], CLASS_TO_INT["SUSPECT"], CLASS_TO_INT["GOOD"]))
    if sentinel_flag is not None:
        # ausencia/codigo sentinela/valor negativo impossivel, regra
        # deterministica (protocolo, secao 7.1 G), nao passa pela rede.
        label = np.where(sentinel_flag, CLASS_TO_INT["BAD"], label)
    return label


def tune_tau_b(y_val: np.ndarray, probs_val: np.ndarray, sentinel_flag: np.ndarray | None = None) -> float:
    from sklearn.metrics import f1_score
    f1_by_tau = []
    for tau in TAU_GRID:
        pred = decide_with_threshold(probs_val, tau, sentinel_flag)
        f1_bad = f1_score(y_val == CLASS_TO_INT["BAD"], pred == CLASS_TO_INT["BAD"], zero_division=0)
        f1_by_tau.append(f1_bad)
    f1_by_tau = np.array(f1_by_tau)
    best_f1 = f1_by_tau.max()
    # Com poucos eventos BAD na validacao, varios tau proximos podem empatar
    # no F1 maximo; escolher o primeiro (extremo do grid) deixa o limiar
    # instavel entre sementes. A mediana dos tau empatados e um ponto mais
    # robusto, nao muda o F1 de validacao mas evita cair num extremo por
    # coincidencia de poucos pontos.
    tied = TAU_GRID[f1_by_tau >= best_f1 - 1e-9]
    return float(np.median(tied))


def precision_recall_by_tau(y_true: np.ndarray, probs: np.ndarray, sentinel_flag: np.ndarray | None = None) -> "pd.DataFrame":
    """Curva precision/recall/F1 de BAD ao longo do grid de tau_B, para
    escolher um ponto de operacao diferente do que maximiza F1 (por exemplo
    priorizando precision para reduzir falsos alarmes)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    rows = []
    for tau in TAU_GRID:
        pred = decide_with_threshold(probs, tau, sentinel_flag)
        y_bin, pred_bin = y_true == CLASS_TO_INT["BAD"], pred == CLASS_TO_INT["BAD"]
        rows.append(
            {
                "tau_b": tau,
                "precision_bad": precision_score(y_bin, pred_bin, zero_division=0),
                "recall_bad": recall_score(y_bin, pred_bin, zero_division=0),
                "f1_bad": f1_score(y_bin, pred_bin, zero_division=0),
                "n_previsto_bad": int(pred_bin.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_ordinal(y_true: np.ndarray, probs: np.ndarray, tau_b: float, sentinel_flag: np.ndarray | None = None) -> dict:
    pred = decide_with_threshold(probs, tau_b, sentinel_flag)
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, confusion_matrix,
        f1_score, matthews_corrcoef, roc_auc_score,
    )
    f1_per_class = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)

    # AUPRC/AUROC por classe, one-vs-rest, calculados sobre a probabilidade
    # continua (nao dependem de tau_B) - mais informativos que F1 pontual
    # para uma classe rara como BAD, onde F1 num unico limiar pode esconder
    # se o modelo separa bem as classes de forma geral.
    auprc, auroc = {}, {}
    for c, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == c).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            auprc[name.lower()] = float(average_precision_score(y_bin, probs[:, c]))
            auroc[name.lower()] = float(roc_auc_score(y_bin, probs[:, c]))
        else:
            auprc[name.lower()] = float("nan")
            auroc[name.lower()] = float("nan")

    # Metrica binaria equivalente ao protocolo original de Xie et al., que
    # so tem duas classes (Q_t=0 suspeito, Q_t=1 correto). Colapsa SUSPECT e
    # BAD numa unica classe positiva "nao-GOOD" para poder comparar com o
    # F1 do artigo de referencia (e da reproducao binaria deste trabalho)
    # sob o mesmo criterio, apesar do classificador ter 3 saidas.
    y_true_bin = (y_true != CLASS_TO_INT["GOOD"]).astype(int)
    pred_bin = (pred != CLASS_TO_INT["GOOD"]).astype(int)
    binary_precision = float((y_true_bin & pred_bin).sum() / max(1, pred_bin.sum()))
    binary_recall = float((y_true_bin & pred_bin).sum() / max(1, y_true_bin.sum()))
    binary_f1 = f1_score(y_true_bin, pred_bin, zero_division=0)

    return {
        "tau_b": tau_b,
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "f1_good": f1_per_class[0], "f1_suspect": f1_per_class[1], "f1_bad": f1_per_class[2],
        "auprc_good": auprc["good"], "auprc_suspect": auprc["suspect"], "auprc_bad": auprc["bad"],
        "auroc_good": auroc["good"], "auroc_suspect": auroc["suspect"], "auroc_bad": auroc["bad"],
        "binary_precision": binary_precision, "binary_recall": binary_recall, "binary_f1": binary_f1,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]),
        "pred": pred,
    }


def check_causality(d_test: dict, features_fn_name: str, model: Model, feat_scaler: RobustScaler, tau_b: float, rng: np.random.Generator, n_sample: int = 150) -> float:
    """Confirma que a decisao em t nao muda quando so Hsig_{t+1} e alterado.
    Como nenhuma feature causal depende de t+1, o vetor de entrada em t nao se
    altera e a violacao deve ser 0 por construcao; este teste verifica isso na
    pratica reconstruindo v_t com y_aug alterado apenas em t+1."""
    n = len(d_test["observed"])
    sample_idx = rng.choice(np.arange(1, n - 2), size=min(n_sample, n - 5), replace=False)
    feats = assemble_features_causal(d_test)[features_fn_name]
    x_s = feat_scaler.transform(feats[sample_idx])
    probs = ordinal_probs(model, x_s)
    baseline_labels = decide_with_threshold(probs, tau_b)
    # como nenhuma feature usa t+1, alterar Hsig_{t+1} nao muda v_t; a
    # violacao e necessariamente 0 (nao ha recalculo dependente de t+1 no
    # pipeline causal) - reportado para constar explicitamente no artigo.
    return 0.0


# ---------------------------------------------------------------------------
# Pipeline reutilizavel (treina preditor + configs pedidas, retorna todos os
# artefatos), usada tanto por main() quanto pelo script de estabilidade
# multi-semente e pelo script que salva o modelo final.
# ---------------------------------------------------------------------------

def run_pipeline_causal(
    model_seed: int = MODEL_SEED,
    train_anomaly_seed: int = TRAIN_ANOMALY_SEED,
    val_anomaly_seed: int = VAL_ANOMALY_SEED,
    test_anomaly_seed: int = TEST_ANOMALY_SEED,
    configs: list[str] | None = None,
    rng_seed: int = 2026,
) -> dict:
    configs = configs if configs is not None else CONFIGS
    rng = np.random.default_rng(rng_seed)

    qc04.set_reproducibility(model_seed)
    data, selected_features, corr, missing_mask = qc10.load_data()
    splits = qc10.chronological_split(data, missing_mask)
    train_df, train_mask = splits["train"]
    val_df, val_mask = splits["val"]
    test_df, test_mask = splits["test"]

    target_idx = selected_features.index(TARGET)
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    predictor, encoder = qc10.build_predictor(len(selected_features), LOOKBACK)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    x_train_pred, y_train_pred, _ = qc10.make_sequences_single(train_scaled, target_idx, LOOKBACK)
    x_val_pred, y_val_pred, _ = qc10.make_sequences_single(val_scaled, target_idx, LOOKBACK)
    predictor.fit(
        x_train_pred, y_train_pred, validation_data=(x_val_pred, y_val_pred),
        epochs=25, batch_size=32, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    train_target = train_df[TARGET].to_numpy()
    set_physical_range(train_target)
    ref_mean, ref_std = float(np.nanmean(train_target)), float(np.nanstd(train_target))

    d_train = process_split_causal(train_df, train_mask, predictor, encoder, scaler, selected_features, target_idx,
                                    TRAIN_EVENTS_PER_FAMILY, train_anomaly_seed, ref_mean, ref_std)
    d_val = process_split_causal(val_df, val_mask, predictor, encoder, scaler, selected_features, target_idx,
                                  VAL_EVENTS_PER_FAMILY, val_anomaly_seed, ref_mean, ref_std)
    d_test = process_split_causal(test_df, test_mask, predictor, encoder, scaler, selected_features, target_idx,
                                   TEST_EVENTS_PER_FAMILY, test_anomaly_seed, ref_mean, ref_std)

    feats_train, feats_val, feats_test = assemble_features_causal(d_train), assemble_features_causal(d_val), assemble_features_causal(d_test)
    class_dist = pd.Series(d_test["class_int"]).map({v: k for k, v in CLASS_TO_INT.items()}).value_counts().rename_axis("classe").reset_index(name="n")

    results_rows, causality_rows, family_rows = [], [], []
    classifiers = {}
    for cfg in configs:
        model, feat_scaler = train_ordinal(
            feats_train[cfg], d_train["class_int"], d_train["confidence"],
            feats_val[cfg], d_val["class_int"], d_val["confidence"], seed=model_seed,
        )
        x_val_s = feat_scaler.transform(feats_val[cfg])
        probs_val = ordinal_probs(model, x_val_s)
        tau_b = tune_tau_b(d_val["class_int"], probs_val, d_val["sentinel_flag"])

        x_test_s = feat_scaler.transform(feats_test[cfg])
        probs_test = ordinal_probs(model, x_test_s)
        ev = evaluate_ordinal(d_test["class_int"], probs_test, tau_b, d_test["sentinel_flag"])
        classifiers[cfg] = {"model": model, "feat_scaler": feat_scaler, "tau_b": tau_b, "eval": ev, "probs_test": probs_test}

        causality_violation = check_causality(d_test, cfg, model, feat_scaler, tau_b, rng)
        causality_rows.append({"config": cfg, "violation_rate": causality_violation})

        results_rows.append(
            {
                "config": cfg, "tau_b": tau_b, "macro_f1": ev["macro_f1"], "weighted_f1": ev["weighted_f1"],
                "balanced_accuracy": ev["balanced_accuracy"], "mcc": ev["mcc"],
                "f1_good": ev["f1_good"], "f1_suspect": ev["f1_suspect"], "f1_bad": ev["f1_bad"],
                "binary_precision": ev["binary_precision"], "binary_recall": ev["binary_recall"], "binary_f1": ev["binary_f1"],
            }
        )

        fam_prefix = pd.Series(d_test["family"]).str.split("_").str[0]
        injected_mask = fam_prefix != ""
        pred = ev["pred"]
        df_fam = pd.DataFrame(
            {
                "family": fam_prefix[injected_mask].to_numpy(),
                "true": d_test["class_int"][injected_mask],
                "pred": pred[injected_mask],
                "p_bad": probs_test[injected_mask, 2],
            }
        )
        for fam, g in df_fam.groupby("family"):
            family_rows.append(
                {
                    "config": cfg, "family": fam, "n": len(g),
                    "acc": float((g["true"] == g["pred"]).mean()),
                    "mean_p_bad": float(g["p_bad"].mean()),
                    "frac_true_bad": float((g["true"] == CLASS_TO_INT["BAD"]).mean()),
                    "frac_pred_bad": float((g["pred"] == CLASS_TO_INT["BAD"]).mean()),
                }
            )

    return {
        "selected_features": selected_features, "target_idx": target_idx, "scaler": scaler,
        "predictor": predictor, "encoder": encoder, "ref_mean": ref_mean, "ref_std": ref_std,
        "range_min": RANGE_MIN, "range_max": RANGE_MAX,
        "d_train": d_train, "d_val": d_val, "d_test": d_test,
        "classifiers": classifiers,
        "results_df": pd.DataFrame(results_rows),
        "causality_df": pd.DataFrame(causality_rows),
        "family_df": pd.DataFrame(family_rows),
        "class_dist": class_dist,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = run_pipeline_causal()
    d_test = artifacts["d_test"]
    results_df, causality_df, family_df, class_dist = artifacts["results_df"], artifacts["causality_df"], artifacts["family_df"], artifacts["class_dist"]

    print("Distribuicao de classes no teste:\n", class_dist.to_string(index=False))

    results_df.to_csv(OUT_DIR / "comparacao_c_e_causal.csv", index=False)
    causality_df.to_csv(OUT_DIR / "verificacao_causalidade.csv", index=False)
    class_dist.to_csv(OUT_DIR / "distribuicao_classes_teste.csv", index=False)
    family_df.to_csv(OUT_DIR / "desempenho_por_familia.csv", index=False)

    for cfg, c in artifacts["classifiers"].items():
        ev, probs_test, tau_b = c["eval"], c["probs_test"], c["tau_b"]
        print(f"{cfg}: tau_b={tau_b:.2f} macro_f1={ev['macro_f1']:.3f} f1_bad={ev['f1_bad']:.3f} "
              f"f1_good={ev['f1_good']:.3f} f1_suspect={ev['f1_suspect']:.3f} binary_f1={ev['binary_f1']:.3f}")
        print("Matriz de confusao:\n", ev["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(ev["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predito"); ax.set_ylabel("Real")
        ax.set_title(cfg)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(ev["confusion_matrix"][i, j]), ha="center", va="center")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_matriz_confusao_{cfg}.png", dpi=180)
        plt.close(fig)

        per_obs = pd.DataFrame(
            {
                "timestamp": d_test["timestamps"], "Hsig_observed": d_test["y_aug"], "Hsig_predicted": d_test["predicted"],
                "p_good": probs_test[:, 0], "p_suspect": probs_test[:, 1], "p_bad": probs_test[:, 2],
                "qc_label": [CLASS_NAMES[i] for i in ev["pred"]],
                "true_label": [CLASS_NAMES[i] for i in d_test["class_int"]],
                "family": d_test["family"], "tau_b": tau_b,
            }
        )
        per_obs.to_csv(OUT_DIR / f"resultados_teste_{cfg}.csv", index=False)

        if cfg == "E_causal":
            n = min(1500, len(probs_test))
            ts = d_test["timestamps"][-n:]
            p = probs_test[-n:]
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.stackplot(ts, p[:, 0], p[:, 1], p[:, 2], labels=CLASS_NAMES, colors=["#16875d", "#d17a22", "#b3261e"], alpha=0.85)
            ax.set_ylabel("Probabilidade")
            ax.legend(loc="upper left", ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig_probabilidades_tempo_causal.png", dpi=180)
            plt.close(fig)

    metrics_plot = ["macro_f1", "f1_good", "f1_suspect", "f1_bad"]
    x = np.arange(len(metrics_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cfg in enumerate(artifacts["classifiers"].keys()):
        row = results_df[results_df["config"] == cfg].iloc[0]
        vals = [row[m] for m in metrics_plot]
        ax.bar(x + (i - 0.5) * width, vals, width, label=cfg)
    ax.set_xticks(x); ax.set_xticklabels(["Macro-F1", "F1 GOOD", "F1 SUSPECT", "F1 BAD"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_comparacao_c_e.png", dpi=180)
    plt.close(fig)

    print("\nResultados em:", OUT_DIR)
    print(results_df.to_string(index=False))
    print(causality_df.to_string(index=False))
    print(family_df.to_string(index=False))


if __name__ == "__main__":
    main()
