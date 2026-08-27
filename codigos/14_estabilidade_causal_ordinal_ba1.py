from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc13", BASE_DIR / "codigos" / "13_qc_lstm_causal_ordinal_ba1.py")
qc13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc13)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_causal_ordinal" / "estabilidade"

# Protocolo completo, 3 sementes de modelo x 5 replicas de injecao = 15
# execucoes. So a configuracao C (residuo + estatisticas causais +
# acumulacao, sem estado oculto) e avaliada aqui, ja praticamente empatada
# com E no desempenho principal e mais simples/portavel para outra boia.
MODEL_SEEDS = [42, 7, 123]
ANOMALY_REPLICAS = [0, 1, 2, 3, 4]
CONFIG = "C_causal"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_seed in MODEL_SEEDS:
        for r in ANOMALY_REPLICAS:
            train_seed = 201 + r * 10
            val_seed = 202 + r * 10
            test_seed = 203 + r * 10
            artifacts = qc13.run_pipeline_causal(
                model_seed=model_seed, train_anomaly_seed=train_seed, val_anomaly_seed=val_seed,
                test_anomaly_seed=test_seed, configs=[CONFIG],
            )
            ev = artifacts["classifiers"][CONFIG]["eval"]
            tau_b = artifacts["classifiers"][CONFIG]["tau_b"]
            causality_violation = artifacts["causality_df"].loc[artifacts["causality_df"]["config"] == CONFIG, "violation_rate"].iloc[0]
            rows.append(
                {
                    "model_seed": model_seed, "anomaly_replica": r, "tau_b": tau_b,
                    "macro_f1": ev["macro_f1"], "weighted_f1": ev["weighted_f1"],
                    "balanced_accuracy": ev["balanced_accuracy"], "mcc": ev["mcc"],
                    "f1_good": ev["f1_good"], "f1_suspect": ev["f1_suspect"], "f1_bad": ev["f1_bad"],
                    "binary_precision": ev["binary_precision"], "binary_recall": ev["binary_recall"],
                    "binary_f1": ev["binary_f1"], "causality_violation": causality_violation,
                }
            )
            print(f"model_seed={model_seed} anomaly_replica={r} -> "
                  f"macro_f1={ev['macro_f1']:.3f} binary_f1={ev['binary_f1']:.3f} f1_bad={ev['f1_bad']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "estabilidade_execucoes.csv", index=False)

    metrics = ["macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "f1_good", "f1_suspect", "f1_bad", "binary_f1"]
    overall = df[metrics].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metrica"})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in ["macro_f1", "binary_f1"]:
        vals = df[metric].to_numpy()
        boot_means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(N_BOOTSTRAP)])
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        overall.loc[len(overall)] = [f"{metric}_bootstrap_ci95_low", ci_low, np.nan]
        overall.loc[len(overall)] = [f"{metric}_bootstrap_ci95_high", ci_high, np.nan]
    overall.to_csv(OUT_DIR / "estabilidade_agregada.csv", index=False)

    ranking = df.groupby("model_seed")["macro_f1"].agg(["mean", "std"]).sort_values("mean", ascending=False).reset_index()
    ranking.to_csv(OUT_DIR / "ranking_por_semente_modelo.csv", index=False)

    macro_f1_std = df["macro_f1"].std()
    binary_f1_mean = df["binary_f1"].mean()
    meets_stability = bool(macro_f1_std < 0.05)
    beats_reference_ba1 = bool(binary_f1_mean > 0.750)
    beats_reference_original = bool(binary_f1_mean > 0.8140)

    print("\nResultados em:", OUT_DIR)
    print(overall.to_string(index=False))
    print(f"Desvio-padrao macro-F1 = {macro_f1_std:.4f} (meta <0.05: {meets_stability})")
    print(f"Binary-F1 medio = {binary_f1_mean:.4f} "
          f"(supera reproducao BA-1 do LSTM-Peak, 0.750: {beats_reference_ba1}; "
          f"supera artigo original, 0.8140: {beats_reference_original})")

    lines = [
        "# Estabilidade multi-semente, classificador causal e ordinal (config C)",
        "",
        f"{len(MODEL_SEEDS)} sementes de modelo x {len(ANOMALY_REPLICAS)} replicas de injecao = "
        f"{len(rows)} execucoes completas (protocolo completo 3x5).",
        "",
        "## Resultado agregado",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Desvio-padrao do macro-F1 entre execucoes, {macro_f1_std:.4f} (meta, < 0.05, atingida, {meets_stability}).",
        "",
        f"Binary-F1 medio (GOOD vs nao-GOOD, comparavel ao protocolo original de Xie et al.), "
        f"{binary_f1_mean:.4f}, supera a reproducao binaria deste trabalho para a BA-1 (0.750, {beats_reference_ba1}) "
        f"e o artigo original nas quatro estacoes chinesas (0.8140, {beats_reference_original}).",
        "",
        "## Ranking dos modelos por semente (media do macro-F1 sobre as replicas de injecao)",
        "",
        ranking.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Execucoes individuais",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUT_DIR / "estabilidade_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
