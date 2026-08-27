from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc18", BASE_DIR / "codigos" / "18_qc_lstm_geral_multiboia.py")
qc18 = importlib.util.module_from_spec(_spec)
sys.modules["qc18"] = qc18
_spec.loader.exec_module(qc18)
qc13 = qc18.qc13

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "estabilidade"

# semente de modelo: inicializacao dos pesos e ordem dos lotes de treino da
# LSTM preditora e da cabeca classificadora (retreino completo a cada valor).
MODEL_SEEDS = [42, 7, 123]
# semente de anomalia: desloca as sementes de injecao sintetica usadas em
# cada boia/split (train/val/test), independente da semente de modelo.
ANOMALY_SEED_BASES = [600, 610, 620, 630, 640]

METRIC_COLS = ["macro_f1", "f1_good", "f1_suspect", "f1_bad", "binary_f1",
               "binary_precision", "binary_recall", "auprc_bad", "auroc_bad"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando e dividindo as series das boias (uma unica vez, reaproveitado em todas as replicas)...")
    per_buoy = qc18.load_buoy_splits()
    if len(per_buoy) < 2:
        raise RuntimeError("Poucas boias disponiveis para treino geral.")

    rows = []
    total = len(MODEL_SEEDS) * len(ANOMALY_SEED_BASES)
    i = 0
    for model_seed, anomaly_seed_base in product(MODEL_SEEDS, ANOMALY_SEED_BASES):
        i += 1
        print(f"\n=== Replica {i}/{total}: model_seed={model_seed} anomaly_seed_base={anomaly_seed_base} ===")
        art = qc18.train_general_pipeline(model_seed=model_seed, anomaly_seed_base=anomaly_seed_base, per_buoy=per_buoy)
        ev = art["ev_global"]
        row = {"model_seed": model_seed, "anomaly_seed_base": anomaly_seed_base, "tau_b": art["tau_b"],
               "n_teste": len(art["d_test"]["class_int"])}
        row.update({k: ev[k] for k in METRIC_COLS})
        rows.append(row)
        print(f"  -> macro_f1={ev['macro_f1']:.3f} f1_bad={ev['f1_bad']:.3f} binary_f1={ev['binary_f1']:.3f} "
              f"auprc_bad={ev['auprc_bad']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "estabilidade_geral_execucoes.csv", index=False)

    agg_rows = []
    for col in METRIC_COLS:
        vals = df[col].to_numpy()
        # IC 95% via bootstrap simples sobre as 15 replicas
        rng = np.random.default_rng(2026)
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        agg_rows.append({"metrica": col, "media": vals.mean(), "desvio_padrao": vals.std(ddof=1),
                          "min": vals.min(), "max": vals.max(), "ic95_inferior": lo, "ic95_superior": hi})
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUT_DIR / "estabilidade_geral_agregada.csv", index=False)

    por_model_seed = df.groupby("model_seed")[METRIC_COLS].agg(["mean", "std"])
    por_model_seed.to_csv(OUT_DIR / "estabilidade_geral_por_semente_modelo.csv")

    lines = [
        "# Estabilidade multi-semente do modelo geral multi-boia (janela de 7 dias)",
        "",
        f"{len(MODEL_SEEDS)} sementes de modelo (retreino completo da LSTM preditora e do classificador",
        "ordinal a cada uma) x " + f"{len(ANOMALY_SEED_BASES)} sementes de injecao de anomalia (deslocam as",
        "sementes de todas as boias/splits), totalizando " + f"{total} replicas independentes, mesmas",
        f"{len(per_buoy)} boias e mesma divisao 60/20/20 cronologica em todas as replicas ({', '.join(per_buoy.keys())}).",
        "",
        "## Metricas agregadas (15 replicas), media, desvio padrao e IC 95% via bootstrap",
        "",
        df_agg.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Todas as execucoes",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUT_DIR / "estabilidade_geral_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nResultados em:", OUT_DIR)
    print(df_agg.to_string(index=False))


if __name__ == "__main__":
    main()
