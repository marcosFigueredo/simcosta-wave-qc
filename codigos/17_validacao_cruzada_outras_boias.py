from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc16", BASE_DIR / "codigos" / "16_qc_lstm_univariado_ba1.py")
qc16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc16)
qc13 = qc16.qc13
qc10 = qc16.qc10

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_causal_ordinal" / "validacao_cruzada_boias"
DATA_DIR = BASE_DIR / "dadosSimcosta"

# Boias candidatas, so a variavel-alvo Hsig e usada, nenhuma variavel
# auxiliar - a mesma logica que permitiu treinar o modelo univariado na
# BA-1 e o que permite aplica-lo, sem retreinar, a qualquer boia SiMCosta
# que reporte Hsig no mesmo formato de arquivo OCEAN.
OTHER_BUOYS = {
    "ES-1": "SIMCOSTA_ES-1_OCEAN_2023-05-30_2026-07-25.csv",
    "PR-1": "SIMCOSTA_PR-1_OCEAN_2013-11-20_2025-05-23.csv",
    "RJ-1": "SIMCOSTA_RJ-1_OCEAN_2015-07-29_2016-10-13.csv",
    "RJ-2": "SIMCOSTA_RJ-2_OCEAN_2015-07-29_2016-12-20.csv",
    "RJ-4": "SIMCOSTA_RJ-4_OCEAN_2017-08-28_2026-07-25.csv",
}

MAX_HOURLY_RECORDS = qc10.MAX_HOURLY_RECORDS
EVENTS_PER_FAMILY = 6
ANOMALY_SEED = 501


def evaluate_on_buoy(buoy_name: str, path: Path, ba1_artifacts: dict) -> dict | None:
    hourly, missing_mask = qc16.read_buoy_target_hourly(path)
    if len(hourly) < qc16.LOOKBACK + 200:
        print(f"{buoy_name}: serie horaria curta demais ({len(hourly)} pontos), pulando.")
        return None

    hourly = hourly.tail(MAX_HOURLY_RECORDS)
    missing_mask = missing_mask.reindex(hourly.index).fillna(0).astype(int)

    scaler = ba1_artifacts["scaler"]
    predictor = ba1_artifacts["predictor"]
    classifier = ba1_artifacts["classifier"]
    feat_scaler = ba1_artifacts["feat_scaler"]
    tau_b = ba1_artifacts["tau_b"]

    # Simulador de implantacao, o modelo e usado exatamente como foi
    # treinado na BA-1 (mesma calibracao de faixa fisica e de media/desvio
    # de referencia), sem nenhum ajuste por boia - o objetivo e ver como o
    # modelo ja treinado se comporta recebendo o fluxo real de outra boia,
    # nao adaptar a regua a cada boia.
    qc13.RANGE_MIN, qc13.RANGE_MAX = ba1_artifacts["range_min"], ba1_artifacts["range_max"]

    scaled = scaler.transform(hourly)
    scaled = np.clip(scaled, 0.0, 1.0)  # a serie de outra boia pode exceder o min/max de treino da BA-1
    x_seq, y_seq_scaled, center_idx = qc10.make_sequences_single(scaled, 0, qc16.LOOKBACK)
    target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]

    pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
    predicted = pred_scaled * target_range + target_min
    observed = y_seq_scaled * target_range + target_min
    mask_at_center = missing_mask.to_numpy()[center_idx]

    y_aug, class_int, family = qc13.inject_anomalies_causal(observed, EVENTS_PER_FAMILY, ANOMALY_SEED)
    residual = y_aug - predicted
    stats = qc13.causal_stat_vector(y_aug, residual)
    accum = qc13.accumulation_features(residual)
    # Simulador, o dado real (nao injetado) de outra boia e considerado OK
    # por definicao, sem rotulagem fraca por z-score, so os pontos
    # sinteticamente injetados tem rotulo SUSPECT/BAD.
    class_int = np.where(class_int == -1, qc13.CLASS_TO_INT["GOOD"], class_int)
    sentinel_flag = pd.Series(family).str.startswith("G_").to_numpy()

    d = {
        "y_aug": y_aug, "predicted": predicted, "residual": residual, "abs_residual": np.abs(residual),
        "mask": mask_at_center, "stats": stats, "accum": accum,
    }
    feats = qc16.assemble_features_univariate(d)
    x_s = feat_scaler.transform(feats)
    probs = qc13.ordinal_probs(classifier, x_s)
    ev = qc13.evaluate_ordinal(class_int, probs, tau_b, sentinel_flag)

    class_dist = pd.Series(class_int).map({v: k for k, v in qc13.CLASS_TO_INT.items()}).value_counts()
    print(f"{buoy_name}: n_teste={len(class_int)} macro_f1={ev['macro_f1']:.3f} f1_bad={ev['f1_bad']:.3f} "
          f"binary_f1={ev['binary_f1']:.3f} auprc_bad={ev['auprc_bad']:.3f}")
    print(f"  distribuicao classes: {dict(class_dist)}")

    return {
        "boia": buoy_name, "n_pontos": len(class_int),
        "macro_f1": ev["macro_f1"], "weighted_f1": ev["weighted_f1"], "balanced_accuracy": ev["balanced_accuracy"],
        "f1_good": ev["f1_good"], "f1_suspect": ev["f1_suspect"], "f1_bad": ev["f1_bad"],
        "binary_f1": ev["binary_f1"], "binary_precision": ev["binary_precision"], "binary_recall": ev["binary_recall"],
        "auprc_bad": ev["auprc_bad"], "auroc_bad": ev["auroc_bad"],
        "n_bad_real": int((class_int == qc13.CLASS_TO_INT["BAD"]).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Treinando o modelo de referencia (univariado, so Hsig) na BA-1...")
    ba1_artifacts = qc16.train_pipeline_univariate()
    ev_ba1 = ba1_artifacts["eval"]
    print(f"BA-1 (origem) -> macro_f1={ev_ba1['macro_f1']:.3f} f1_bad={ev_ba1['f1_bad']:.3f} binary_f1={ev_ba1['binary_f1']:.3f}\n")

    rows = [
        {
            "boia": "BA-1 (origem, treino)", "n_pontos": len(ba1_artifacts["d_test"]["class_int"]),
            "macro_f1": ev_ba1["macro_f1"], "weighted_f1": ev_ba1["weighted_f1"], "balanced_accuracy": ev_ba1["balanced_accuracy"],
            "f1_good": ev_ba1["f1_good"], "f1_suspect": ev_ba1["f1_suspect"], "f1_bad": ev_ba1["f1_bad"],
            "binary_f1": ev_ba1["binary_f1"], "binary_precision": ev_ba1["binary_precision"], "binary_recall": ev_ba1["binary_recall"],
            "auprc_bad": ev_ba1["auprc_bad"], "auroc_bad": ev_ba1["auroc_bad"],
            "n_bad_real": int((ba1_artifacts["d_test"]["class_int"] == qc13.CLASS_TO_INT["BAD"]).sum()),
        }
    ]

    for buoy_name, filename in OTHER_BUOYS.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"{buoy_name}: arquivo nao encontrado ({filename}), pulando.")
            continue
        result = evaluate_on_buoy(buoy_name, path, ba1_artifacts)
        if result is not None:
            rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "validacao_cruzada_resultados.csv", index=False)

    lines = [
        "# Validacao cruzada entre boias, modelo univariado treinado na BA-1",
        "",
        "Modelo treinado uma unica vez na BA-1 (so a variavel Hsig, sem nenhuma variavel auxiliar) e",
        "aplicado sem retreino nem recalibracao de limiar as demais boias SiMCosta que relatam Hsig,",
        "com anomalias sinteticas injetadas na propria serie real de cada boia para permitir avaliacao",
        "quantitativa (nao ha rotulo manual de qualidade em nenhuma delas).",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "BA-1 e a boia de origem (onde o modelo foi treinado), as demais linhas sao transferencia pura,",
        "sem nenhum ajuste especifico aplicado aos dados dessa boia.",
    ]
    (OUT_DIR / "validacao_cruzada_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nResultados em:", OUT_DIR)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
