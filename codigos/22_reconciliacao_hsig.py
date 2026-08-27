from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc19", BASE_DIR / "codigos" / "19_simulador_tempo_real.py")
qc19 = importlib.util.module_from_spec(_spec)
sys.modules["qc19"] = qc19
_spec.loader.exec_module(qc19)
qc16 = qc19.qc16
qc10 = qc19.qc10
qc13 = qc19.qc13

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "reconciliacao_hsig"

# Ordem de severidade comum aos dois ramos, use/GOOD=0 (menos severo) ate
# remove/BAD=2 (mais severo), a mesma logica de "o mais severo prevalece"
# ja usada para consolidar os sete testes classicos em R (Secao 2.2).
R_RANK = {"use": 0, "review": 1, "remove": 2}
Q_RANK = {"GOOD": 0, "SUSPECT": 1, "BAD": 2}
RANK_TO_LABEL = {0: "GOOD", 1: "SUSPECT", 2: "BAD"}
RANK_TO_R_LABEL = {0: "use", 1: "review", 2: "remove"}


def load_classic_r_hsig_hourly() -> pd.Series:
    """Rotulo classico R do ramo estatistico, so a variavel Hsig, agregado
    a hora pelo pior (mais severo) valor dentro de cada hora, coerente com
    a resolucao horaria usada pelo ramo com IA."""
    path = BASE_DIR / "resultados_qc_ba1" / "base_qc_ready" / "base_qc_ready_long.csv"
    df = pd.read_csv(path, usecols=["Timestamp", "variable", "final_decision"])
    hsig = df[df["variable"] == "Hsig"].copy()
    hsig["Timestamp"] = pd.to_datetime(hsig["Timestamp"], utc=True)
    hsig["rank"] = hsig["final_decision"].map(R_RANK)
    hourly_rank = hsig.set_index("Timestamp")["rank"].resample("1h").max()
    return hourly_rank


def infer_q_hsig_hourly(hourly: pd.DataFrame, artifacts: dict) -> pd.Series:
    """Rotulo Q_t do classificador causal e ordinal (modelo geral), aplicado
    a serie real da BA-1 SEM nenhuma anomalia sintetica injetada."""
    target = hourly[qc19.TARGET].to_numpy()
    scaler, predictor, classifier, feat_scaler = artifacts["scaler"], artifacts["predictor"], artifacts["classifier"], artifacts["feat_scaler"]
    tau_b = artifacts["tau_b"]
    qc13.RANGE_MIN, qc13.RANGE_MAX = artifacts["range_min"], artifacts["range_max"]

    scaled = np.clip(scaler.transform(hourly), 0.0, 1.0)
    x_seq, y_seq_scaled, center_idx = qc10.make_sequences_single(scaled, 0, qc19.LOOKBACK)
    target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]

    predicted = predictor.predict(x_seq, verbose=0).ravel() * target_range + target_min
    observed = y_seq_scaled * target_range + target_min
    timestamps = hourly.index[center_idx]

    residual = observed - predicted
    stats = qc13.causal_stat_vector(observed, residual)
    accum = qc13.accumulation_features(residual)
    sentinel_flag = np.zeros(len(center_idx), dtype=bool)

    d = {"y_aug": observed, "predicted": predicted, "residual": residual, "abs_residual": np.abs(residual),
         "mask": np.zeros(len(center_idx)), "stats": stats, "accum": accum}
    feats = qc16.assemble_features_univariate(d)
    probs = qc13.ordinal_probs(classifier, feat_scaler.transform(feats))
    pred_label = qc13.decide_with_threshold(probs, tau_b, sentinel_flag)
    label_names = [qc13.CLASS_NAMES[i] for i in pred_label]
    return pd.Series(label_names, index=timestamps)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando modelo geral salvo...")
    artifacts = qc19.load_model_artifacts()

    print("Lendo serie real da BA-1 (sem anomalia sintetica)...")
    path = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
    hourly, _ = qc16.read_buoy_target_hourly(path)
    hourly = hourly.tail(qc10.MAX_HOURLY_RECORDS)

    q_series = infer_q_hsig_hourly(hourly, artifacts)
    r_rank_hourly = load_classic_r_hsig_hourly()

    df = pd.DataFrame({"Q_t": q_series})
    df["Q_rank"] = df["Q_t"].map(Q_RANK)
    df["R_rank"] = r_rank_hourly.reindex(df.index)
    df = df.dropna(subset=["R_rank"])
    df["R_rank"] = df["R_rank"].astype(int)
    df["R_t"] = df["R_rank"].map(RANK_TO_R_LABEL)

    df["reconciliado_rank"] = df[["Q_rank", "R_rank"]].max(axis=1)
    df["Q_reconciliado"] = df["reconciliado_rank"].map(RANK_TO_LABEL)
    df.to_csv(OUT_DIR / "reconciliacao_hsig_serie.csv")

    n = len(df)
    concordancia = float((df["Q_rank"] == df["R_rank"]).mean())
    escalado_pelo_classico = int(((df["R_rank"] > df["Q_rank"])).sum())
    escalado_pela_ia = int(((df["Q_rank"] > df["R_rank"])).sum())

    confusion = pd.crosstab(df["R_t"], df["Q_t"], rownames=["R (classico)"], colnames=["Q_t (IA)"])
    confusion = confusion.reindex(index=["use", "review", "remove"], columns=["GOOD", "SUSPECT", "BAD"], fill_value=0)
    confusion.to_csv(OUT_DIR / "matriz_concordancia_r_q.csv")

    dist_final = df["Q_reconciliado"].value_counts().reindex(["GOOD", "SUSPECT", "BAD"], fill_value=0)
    dist_final_pct = (100 * dist_final / n).round(2)

    print(f"\nHoras com os dois rotulos disponiveis: {n}")
    print(f"Concordancia exata R vs Q_t: {concordancia:.3f}")
    print(f"Horas escaladas pelo ramo classico (R mais severo que Q_t): {escalado_pelo_classico} ({100*escalado_pelo_classico/n:.1f}%)")
    print(f"Horas escaladas pelo ramo IA (Q_t mais severo que R): {escalado_pela_ia} ({100*escalado_pela_ia/n:.1f}%)")
    print("\nMatriz de concordancia (linhas R, colunas Q_t):")
    print(confusion.to_string())
    print("\nDistribuicao final apos reconciliacao (Q_t^final = mais severo):")
    print(dist_final_pct.to_string())

    lines = [
        "# Reconciliacao do rotulo de Hsig entre o ramo estatistico classico e o ramo com IA",
        "",
        "Regra, para cada hora, mapeia R (ramo classico, use/review/remove) e Q_t (ramo IA,",
        "GOOD/SUSPECT/BAD) para o mesmo espaco ordinal de severidade (0, 1, 2) e adota o mais",
        "severo dos dois como rotulo final, a mesma logica ja usada para consolidar os sete testes",
        "classicos dentro do proprio R.",
        "",
        f"- Horas com os dois rotulos disponiveis, {n}",
        f"- Concordancia exata R vs Q_t, {concordancia:.3f}",
        f"- Horas escaladas pelo ramo classico (R mais severo), {escalado_pelo_classico} ({100*escalado_pelo_classico/n:.1f}%)",
        f"- Horas escaladas pelo ramo IA (Q_t mais severo), {escalado_pela_ia} ({100*escalado_pela_ia/n:.1f}%)",
        "",
        "## Matriz de concordancia (linhas R, colunas Q_t)",
        "",
        confusion.to_markdown(),
        "",
        "## Distribuicao final apos reconciliacao (%)",
        "",
        dist_final_pct.to_markdown(),
    ]
    (OUT_DIR / "reconciliacao_hsig_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nResultados em:", OUT_DIR)


if __name__ == "__main__":
    main()
