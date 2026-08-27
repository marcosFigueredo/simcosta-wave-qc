"""Re-evaluate the saved general model on strong labels without retraining."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qc18 = load_module("qc18", BASE_DIR / "codigos" / "18_qc_lstm_geral_multiboia.py")
qc19 = load_module("qc19", BASE_DIR / "codigos" / "19_simulador_tempo_real.py")


def main() -> None:
    per_buoy = qc18.load_buoy_splits()
    art = qc19.load_model_artifacts()
    train_all = pd.concat([per_buoy[b]["train"][0] for b in per_buoy])
    qc18.qc13.set_physical_range(train_all[qc18.TARGET].to_numpy())
    ref_mean = float(train_all[qc18.TARGET].mean())
    ref_std = float(train_all[qc18.TARGET].std())

    rows = []
    confusion_rows = []
    pooled_y, pooled_probs, pooled_sentinel = [], [], []
    for i, (name, splits) in enumerate(per_buoy.items()):
        test_h, test_m = splits["test"]
        d = qc18.process_buoy_segment(
            name, test_h, test_m, art["predictor"], art["scaler"],
            qc18.TEST_EVENTS_PER_FAMILY, 801 + i, ref_mean, ref_std,
        )
        features = qc18.qc16.assemble_features_univariate(d)
        probs = qc18.qc13.ordinal_probs(art["classifier"], art["feat_scaler"].transform(features))
        full = qc18.qc13.evaluate_ordinal(d["class_int"], probs, art["tau_b"], d["sentinel_flag"])
        strong = d["confidence"] >= 0.999
        pooled_y.append(d["class_int"][strong])
        pooled_probs.append(probs[strong])
        pooled_sentinel.append(d["sentinel_flag"][strong])
        ev = qc18.qc13.evaluate_ordinal(
            d["class_int"][strong], probs[strong], art["tau_b"], d["sentinel_flag"][strong]
        )
        # Usa a mesma decisao por cascata de tau_b que gera todas as outras
        # metricas (evaluate_ordinal ja devolve essa matriz pronta), nao um
        # argmax bruto sobre as probabilidades, que discordaria do F1/precisao
        # relatados na mesma linha e da Figura S5.
        cm = ev["confusion_matrix"]
        for true_class in range(3):
            confusion_rows.append({"boia": name, "true_class": true_class,
                                   "pred_good": int(cm[true_class, 0]),
                                   "pred_suspect": int(cm[true_class, 1]),
                                   "pred_bad": int(cm[true_class, 2])})
        rows.append({
            "boia": name, "n": int(strong.sum()), "n_weak": int((~strong).sum()),
            "tau_b": art["tau_b"], "binary_f1_all": full["binary_f1"],
            **{k: ev[k] for k in ["macro_f1", "f1_good", "f1_suspect", "f1_bad",
                                   "binary_f1", "binary_precision", "binary_recall",
                                   "auprc_bad", "auroc_bad"]},
        })

    df = pd.DataFrame(rows)
    y_pool = np.concatenate(pooled_y)
    p_pool = np.concatenate(pooled_probs)
    s_pool = np.concatenate(pooled_sentinel)
    ev_pool = qc18.qc13.evaluate_ordinal(y_pool, p_pool, art["tau_b"], s_pool)
    cm_pool = ev_pool["confusion_matrix"]
    for true_class in range(3):
        confusion_rows.append({"boia": "TODAS (rótulos fortes)", "true_class": true_class,
                               "pred_good": int(cm_pool[true_class, 0]),
                               "pred_suspect": int(cm_pool[true_class, 1]),
                               "pred_bad": int(cm_pool[true_class, 2])})
    strong_rows = {"boia": "TODAS (rótulos fortes)", "n": int(df["n"].sum()),
                   "n_weak": int(df["n_weak"].sum()), "tau_b": art["tau_b"],
                   "binary_f1_all": np.nan,
                   **{k: ev_pool[k] for k in ["macro_f1", "f1_good", "f1_suspect", "f1_bad",
                                               "binary_f1", "binary_precision", "binary_recall",
                                               "auprc_bad", "auroc_bad"]}}
    df_out = pd.concat([pd.DataFrame([strong_rows]), df], ignore_index=True)
    out = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "desempenho_geral_por_boia_strong.csv"
    df_out.to_csv(out, index=False)
    pd.DataFrame(confusion_rows).to_csv(out.parent / "matrizes_confusao_strong.csv", index=False)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
