from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc10", BASE_DIR / "codigos" / "10_qc_lstm_3classes_ba1.py")
qc10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc10)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_3classes" / "estabilidade"

# O protocolo (secao 10.5) sugere 3 sementes de LSTM x 5 sementes de injecao =
# 15 execucoes. Reduzido aqui para 3 x 3 = 9 execucoes (cada uma retreina a
# LSTM preditora e a cabeca classificadora E_full do zero) para manter o
# tempo de execucao administravel nesta rodada; documentado como reducao
# deliberada, nao omissao.
MODEL_SEEDS = [42, 7, 123]
ANOMALY_REPLICAS = [0, 1, 2]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_seed in MODEL_SEEDS:
        for r in ANOMALY_REPLICAS:
            train_seed = 101 + r * 10
            val_seed = 102 + r * 10
            test_seed = 103 + r * 10
            artifacts = qc10.run_pipeline(
                model_seed=model_seed,
                train_anomaly_seed=train_seed,
                val_anomaly_seed=val_seed,
                test_anomaly_seed=test_seed,
                ablation_models=["E_full"],
            )
            ev = artifacts["full_eval"]
            rows.append(
                {
                    "model_seed": model_seed,
                    "anomaly_replica": r,
                    "macro_f1": ev["macro_f1"],
                    "weighted_f1": ev["weighted_f1"],
                    "balanced_accuracy": ev["balanced_accuracy"],
                    "mcc": ev["mcc"],
                    "f1_good": ev["f1_good"],
                    "f1_suspect": ev["f1_suspect"],
                    "f1_bad": ev["f1_bad"],
                    "auprc_bad": ev["auprc_bad"],
                }
            )
            print(f"model_seed={model_seed} anomaly_replica={r} -> macro_f1={ev['macro_f1']:.3f} f1_bad={ev['f1_bad']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "estabilidade_execucoes.csv", index=False)

    metrics = ["macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "f1_good", "f1_suspect", "f1_bad", "auprc_bad"]
    overall = df[metrics].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metrica"})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro_f1_vals = df["macro_f1"].to_numpy()
    boot_means = np.array([rng.choice(macro_f1_vals, size=len(macro_f1_vals), replace=True).mean() for _ in range(N_BOOTSTRAP)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    overall.loc[len(overall)] = ["macro_f1_bootstrap_ci95_low", ci_low, np.nan]
    overall.loc[len(overall)] = ["macro_f1_bootstrap_ci95_high", ci_high, np.nan]
    overall.to_csv(OUT_DIR / "estabilidade_agregada.csv", index=False)

    ranking = df.groupby("model_seed")["macro_f1"].agg(["mean", "std"]).sort_values("mean", ascending=False).reset_index()
    ranking.to_csv(OUT_DIR / "ranking_por_semente_modelo.csv", index=False)

    macro_f1_std = df["macro_f1"].std()
    meets_target = bool(macro_f1_std < 0.05)

    lines = [
        "# Estabilidade multi-semente - QC-LSTM 3 classes (secao 10.5 do protocolo)",
        "",
        f"{len(MODEL_SEEDS)} sementes de modelo (LSTM preditora + cabeca classificadora retreinadas do zero)",
        f"x {len(ANOMALY_REPLICAS)} replicas de injecao sintetica = {len(rows)} execucoes completas "
        "(reduzido do 3x5=15 sugerido pelo protocolo por custo computacional nesta rodada).",
        "",
        "## Resultado agregado",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Desvio-padrao do macro-F1 entre execucoes, {macro_f1_std:.4f} "
        f"(meta do protocolo, < 0.05, atingida: {meets_target}).",
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

    print("Resultados em:", OUT_DIR)
    print(overall.to_string(index=False))
    print(f"Desvio-padrao macro-F1 = {macro_f1_std:.4f} (meta <0.05: {meets_target})")


if __name__ == "__main__":
    main()
