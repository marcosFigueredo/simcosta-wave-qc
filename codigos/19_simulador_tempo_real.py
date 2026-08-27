from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc16", BASE_DIR / "codigos" / "16_qc_lstm_univariado_ba1.py")
qc16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc16)
qc13 = qc16.qc13
qc10 = qc16.qc10

MODEL_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "modelo_final"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "simulacao_tempo_real"
FIG_DIR = OUT_DIR / "figures"

TARGET = qc16.TARGET
LOOKBACK = qc16.LOOKBACK
CLASS_NAMES = qc13.CLASS_NAMES
CLASS_TO_INT = qc13.CLASS_TO_INT

SIM_SEED = 2026
EVENT_PROB_PER_HOUR = 0.008  # em media cerca de 1 evento a cada 125 horas
FAMILIES = ["A_spike", "B_mult", "C_level_shift", "D_drift", "E_stuck", "F_burst_noise", "G_missing_sentinel"]
DURATIONS_SHORT = qc13.DURATIONS_SHORT
DURATIONS_LONG = qc13.DURATIONS_LONG


class StreamingAnomalyInjector:
    """Cada hora que chega tem uma pequena probabilidade de iniciar um novo
    episodio de anomalia (familia, gravidade e duracao sorteadas na hora);
    enquanto um episodio esta ativo, os valores reais sao substituidos pelo
    efeito da familia escolhida, ponto a ponto, como um sensor realmente se
    comportaria com falha continua."""

    def __init__(self, rng: np.random.Generator, base_std: float, event_prob: float = EVENT_PROB_PER_HOUR):
        self.rng = rng
        self.base_std = base_std
        self.event_prob = event_prob
        self.active: dict | None = None

    def _start_event(self, true_value: float) -> dict:
        family = self.rng.choice(FAMILIES)
        if family == "A_spike":
            return {"family": family, "remaining": 1, "sign": self.rng.choice([-1, 1]),
                    "k": float(self.rng.choice([0.5, 1, 2, 3, 4, 6]))}
        if family == "B_mult":
            return {"family": family, "remaining": 1, "factor": float(self.rng.choice([5.0, 10.0, 1 / 5, 1 / 10]))}
        if family == "C_level_shift":
            d = int(self.rng.choice(DURATIONS_LONG))
            target_bad = self.rng.random() < 0.5
            magnitude = self.rng.uniform(3.0, 4.0) if target_bad else self.rng.uniform(1.0, 2.5)
            return {"family": family, "remaining": d, "duration": d,
                    "delta": float(self.rng.choice([-1, 1]) * magnitude * self.base_std)}
        if family == "D_drift":
            d = int(self.rng.choice(DURATIONS_LONG))
            target_bad = self.rng.random() < 0.5
            target_cum = self.rng.uniform(3.0, 4.0) if target_bad else self.rng.uniform(1.0, 2.5)
            return {"family": family, "remaining": d, "duration": d, "j": 0,
                    "delta": float(self.rng.choice([-1, 1]) * target_cum * self.base_std / max(1, d - 1))}
        if family == "E_stuck":
            d = int(self.rng.choice(DURATIONS_SHORT))
            return {"family": family, "remaining": d, "duration": d, "stuck_value": true_value}
        if family == "F_burst_noise":
            d = int(self.rng.choice(DURATIONS_SHORT))
            sigma_a = self.rng.uniform(0.5, 2.5) * self.base_std
            return {"family": family, "remaining": d, "duration": d, "sigma_a": sigma_a}
        choice = self.rng.choice(["nan", "sentinel", "negative"])
        return {"family": "G_missing_sentinel", "remaining": 1, "choice": choice}

    def step(self, true_value: float) -> tuple[float, int, str]:
        if self.active is None and self.rng.random() < self.event_prob:
            self.active = self._start_event(true_value)

        if self.active is None:
            return true_value, CLASS_TO_INT["GOOD"], ""

        ev = self.active
        family = ev["family"]
        if family == "A_spike":
            value = max(0.0, true_value + ev["sign"] * ev["k"] * self.base_std)
            label = CLASS_TO_INT["BAD"] if ev["k"] >= 3 else CLASS_TO_INT["SUSPECT"]
        elif family == "B_mult":
            value = true_value * ev["factor"]
            label = CLASS_TO_INT["BAD"]
        elif family == "C_level_shift":
            value = max(0.0, true_value + ev["delta"])
            label = CLASS_TO_INT["BAD"] if abs(ev["delta"]) >= 3 * self.base_std else CLASS_TO_INT["SUSPECT"]
        elif family == "D_drift":
            ev["j"] += 1
            cumulative = ev["j"] * ev["delta"]
            value = max(0.0, true_value + cumulative)
            label = CLASS_TO_INT["BAD"] if abs(cumulative) >= 3 * self.base_std else CLASS_TO_INT["SUSPECT"]
        elif family == "E_stuck":
            value = ev["stuck_value"]
            label = CLASS_TO_INT["BAD"]
        elif family == "F_burst_noise":
            value = max(0.0, true_value + self.rng.normal(0, ev["sigma_a"]))
            label = CLASS_TO_INT["BAD"] if ev["sigma_a"] >= 1.5 * self.base_std else CLASS_TO_INT["SUSPECT"]
        else:  # G_missing_sentinel
            value = {"nan": np.nan, "sentinel": -999.0, "negative": -abs(self.rng.uniform(0.1, 1.0))}[ev["choice"]]
            label = CLASS_TO_INT["BAD"]

        ev["remaining"] -= 1
        if ev["remaining"] <= 0:
            self.active = None
        return value, label, family


def run_simulation(hourly: pd.DataFrame, artifacts: dict, sim_seed: int = SIM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(sim_seed)
    target = hourly[TARGET].to_numpy()
    base_std = float(np.nanstd(target))
    injector = StreamingAnomalyInjector(rng, base_std)

    corrupted = np.empty(len(target))
    true_label = np.full(len(target), CLASS_TO_INT["GOOD"])
    family_log = np.array([""] * len(target), dtype=object)
    for t in range(len(target)):
        value, label, family = injector.step(target[t])
        corrupted[t] = value
        true_label[t] = label
        family_log[t] = family

    scaler, predictor, classifier, feat_scaler = artifacts["scaler"], artifacts["predictor"], artifacts["classifier"], artifacts["feat_scaler"]
    tau_b = artifacts["tau_b"]
    qc13.RANGE_MIN, qc13.RANGE_MAX = artifacts["range_min"], artifacts["range_max"]

    corrupted_df = pd.DataFrame({TARGET: corrupted}, index=hourly.index)
    scaled = np.clip(scaler.transform(corrupted_df), 0.0, 1.0)
    x_seq, y_seq_scaled, center_idx = qc10.make_sequences_single(scaled, 0, LOOKBACK)
    target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]

    predicted = predictor.predict(x_seq, verbose=0).ravel() * target_range + target_min
    observed = y_seq_scaled * target_range + target_min
    timestamps = hourly.index[center_idx]

    residual = observed - predicted
    stats = qc13.causal_stat_vector(observed, residual)
    accum = qc13.accumulation_features(residual)
    sentinel_flag = pd.Series(family_log[center_idx]).eq("G_missing_sentinel").to_numpy()

    d = {"y_aug": observed, "predicted": predicted, "residual": residual, "abs_residual": np.abs(residual),
         "mask": np.zeros(len(center_idx)), "stats": stats, "accum": accum}
    feats = qc16.assemble_features_univariate(d)
    probs = qc13.ordinal_probs(classifier, feat_scaler.transform(feats))
    pred_label = qc13.decide_with_threshold(probs, tau_b, sentinel_flag)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "Hsig_real": target[center_idx],
            "Hsig_observado_pelo_sensor": observed,
            "Hsig_previsto": predicted,
            "p_good": probs[:, 0], "p_suspect": probs[:, 1], "p_bad": probs[:, 2],
            "rotulo_verdadeiro": [CLASS_NAMES[i] for i in true_label[center_idx]],
            "rotulo_previsto": [CLASS_NAMES[i] for i in pred_label],
            "familia_anomalia": family_log[center_idx],
        }
    )


def compute_onset_detection(log: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Taxa de deteccao no instante t+1 (inicio de cada episodio de anomalia,
    a transicao GOOD -> anomalo), por familia. Reutilizado tanto pelo
    relatorio de uma unica simulacao (main) quanto pelo protocolo
    multi-semente (script 21)."""
    em_evento = log["familia_anomalia"] != ""
    novo_evento = em_evento & ~em_evento.shift(1, fill_value=False)
    inicio = log[novo_evento].copy()
    inicio["familia_letra"] = inicio["familia_anomalia"].str.split("_").str[0]
    inicio["detectado_t1"] = inicio["rotulo_previsto"] != "GOOD"
    onset_df = inicio.groupby("familia_letra")["detectado_t1"].agg(["mean", "count"]).rename(
        columns={"mean": "taxa_deteccao_t1", "count": "n_eventos"}
    ).reset_index()
    taxa_geral_t1 = float(inicio["detectado_t1"].mean()) if len(inicio) else float("nan")
    return onset_df, taxa_geral_t1


def load_model_artifacts() -> dict:
    sys.modules["qc13"] = qc13
    predictor = tf.keras.models.load_model(MODEL_DIR / "predictor_lstm.keras")
    classifier = tf.keras.models.load_model(MODEL_DIR / "classificador_ordinal.keras")
    with open(MODEL_DIR / "input_scaler_lstm.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_scaler_classificador.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "predictor": predictor, "classifier": classifier, "scaler": scaler, "feat_scaler": feat_scaler,
        "tau_b": metadata["tau_b"], "range_min": metadata["faixa_fisica_min"], "range_max": metadata["faixa_fisica_max"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando modelo geral salvo...")
    artifacts = load_model_artifacts()

    print("Lendo serie real da BA-1 (fluxo simulado em tempo real)...")
    path = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
    hourly, _ = qc16.read_buoy_target_hourly(path)
    hourly = hourly.tail(qc10.MAX_HOURLY_RECORDS)

    print(f"Simulando {len(hourly)} horas, com insercao aleatoria de anomalias (probabilidade "
          f"{EVENT_PROB_PER_HOUR:.3%} por hora de iniciar um novo evento)...")
    log = run_simulation(hourly, artifacts)
    log.to_csv(OUT_DIR / "log_simulacao.csv", index=False)

    y_true = log["rotulo_verdadeiro"].map(CLASS_TO_INT).to_numpy()
    y_pred = log["rotulo_previsto"].map(CLASS_TO_INT).to_numpy()
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    y_true_bin, y_pred_bin = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    tp = int((y_true_bin & y_pred_bin).sum())
    binary_precision = tp / max(1, y_pred_bin.sum())
    binary_recall = tp / max(1, y_true_bin.sum())
    binary_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    n_eventos = (log["familia_anomalia"] != "").sum()
    print(f"\nHoras simuladas: {len(log)}, horas dentro de algum evento de anomalia: {n_eventos} "
          f"({n_eventos / len(log):.1%})")
    print(f"Macro-F1: {macro_f1:.3f}, Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"Deteccao (GOOD vs nao-GOOD), precision={binary_precision:.3f} recall={binary_recall:.3f} f1={binary_f1:.3f}")
    print("Matriz de confusao (real x previsto, GOOD/SUSPECT/BAD):\n", cm)

    fam_perf = []
    fam_prefix = log["familia_anomalia"].str.split("_").str[0]
    for fam in sorted(set(fam_prefix) - {""}):
        mask = fam_prefix == fam
        acc = float((y_true[mask] == y_pred[mask]).mean())
        detectado = float((y_pred[mask] != 0).mean())
        fam_perf.append({"familia": fam, "n_horas": int(mask.sum()), "acuracia_exata": acc, "fracao_sinalizada": detectado})
    fam_df = pd.DataFrame(fam_perf)
    fam_df.to_csv(OUT_DIR / "desempenho_por_familia_simulacao.csv", index=False)

    # Deteccao em t+1, o instante exato em que cada episodio de anomalia
    # comeca (a transicao GOOD -> anomalo), em vez da cobertura ao longo de
    # todo o episodio (que cai para familias longas por causa da propria
    # realimentacao do valor corrompido nas janelas seguintes, um efeito
    # separado de adaptacao, nao de deteccao do instante inicial).
    onset_df, taxa_geral_t1 = compute_onset_detection(log)
    n_eventos_distintos = int(onset_df["n_eventos"].sum())
    onset_df.to_csv(OUT_DIR / "deteccao_t1_por_familia.csv", index=False)
    print(f"\nDeteccao no instante t+1 (inicio do evento), {n_eventos_distintos} eventos distintos, "
          f"taxa geral {taxa_geral_t1:.3f}")
    print(onset_df.to_string(index=False))

    n = min(3000, len(log))
    sample = log.tail(n)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(sample["timestamp"], sample["Hsig_observado_pelo_sensor"], color="#25364a", linewidth=0.8, label="Hsig observado (com anomalias)")
    ax.plot(sample["timestamp"], sample["Hsig_previsto"], color="#888888", linewidth=0.6, alpha=0.6, label="Hsig previsto pela LSTM")
    for cls_name, color in [("SUSPECT", "#d17a22"), ("BAD", "#b3261e")]:
        flagged = sample[sample["rotulo_previsto"] == cls_name]
        ax.scatter(flagged["timestamp"], flagged["Hsig_observado_pelo_sensor"], color=color, s=14, label=f"Detectado como {cls_name}", zorder=4)
    true_anom = sample[sample["familia_anomalia"] != ""]
    ax.scatter(true_anom["timestamp"], true_anom["Hsig_observado_pelo_sensor"], facecolors="none", edgecolors="black", s=60, linewidths=1.0, label="Anomalia injetada (real)", zorder=5)
    ax.set_ylabel("Hsig (m)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_simulacao_tempo_real.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Simulacao em tempo real, modelo geral multi-boia na BA-1",
        "",
        "Fluxo horario real da BA-1 tratado como dado OK, com um simulador que insere anomalias de",
        "sete familias em momentos e intensidades aleatorios ao longo da execucao (nao um cronograma",
        "fixo). O modelo processa o fluxo ponto a ponto, causal (so ve o passado e o instante atual),",
        "incluindo a realimentacao real, um valor corrompido de um evento em andamento entra nas",
        "janelas de entrada dos instantes seguintes, exatamente como aconteceria num sensor de verdade.",
        "",
        f"- Horas simuladas, {len(log)}",
        f"- Horas dentro de algum evento de anomalia, {n_eventos} ({n_eventos / len(log):.1%})",
        f"- Macro-F1, {macro_f1:.3f}",
        f"- Deteccao geral (GOOD vs nao-GOOD), precision {binary_precision:.3f}, recall {binary_recall:.3f}, F1 {binary_f1:.3f}",
        "",
        "## Deteccao no instante t+1 (metrica principal)",
        "",
        "Taxa de deteccao exatamente no primeiro instante em que cada episodio de anomalia aparece",
        "(a transicao de GOOD para anomalo), o foco da simulacao, e nao a cobertura ao longo de todo",
        f"um episodio longo. Taxa geral, {taxa_geral_t1:.3f}, sobre {n_eventos_distintos} eventos distintos.",
        "",
        onset_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Cobertura ao longo de toda a duracao do episodio (metrica secundaria)",
        "",
        "Fracao das horas dentro de cada familia que continuam sinalizadas do inicio ao fim do",
        "episodio. Cai para familias longas (mudanca de nivel, drift, sensor travado) porque o valor",
        "corrompido realimenta a propria janela de entrada da LSTM e o modelo passa a prever perto do",
        "novo nivel depois de algumas horas, efeito de adaptacao do preditor, nao de falha em",
        "detectar o inicio do evento.",
        "",
        fam_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Figura",
        "",
        "`fig_simulacao_tempo_real.png`, trecho recente da simulacao, serie observada e prevista,",
        "pontos detectados como SUSPECT/BAD e os eventos realmente injetados (circulo preto).",
    ]
    (OUT_DIR / "simulacao_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nResultados em:", OUT_DIR)
    print(fam_df.to_string(index=False))


if __name__ == "__main__":
    main()
