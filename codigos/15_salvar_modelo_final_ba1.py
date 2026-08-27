from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc13", BASE_DIR / "codigos" / "13_qc_lstm_causal_ordinal_ba1.py")
qc13 = importlib.util.module_from_spec(_spec)
sys.modules["qc13"] = qc13  # necessario para o pickle do ClippedRobustScaler localizar a classe
_spec.loader.exec_module(qc13)

MODEL_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_causal_ordinal" / "modelo_final"
CONFIG = "C_causal"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Treinando o modelo final (config", CONFIG, ") com as sementes canonicas do pipeline...")
    artifacts = qc13.run_pipeline_causal(configs=[CONFIG])

    predictor = artifacts["predictor"]
    clf_info = artifacts["classifiers"][CONFIG]
    classifier = clf_info["model"]
    feat_scaler = clf_info["feat_scaler"]
    tau_b = clf_info["tau_b"]
    ev = clf_info["eval"]

    predictor.save(MODEL_DIR / "predictor_lstm.keras")
    classifier.save(MODEL_DIR / "classificador_ordinal.keras")

    with open(MODEL_DIR / "input_scaler_lstm.pkl", "wb") as f:
        pickle.dump(artifacts["scaler"], f)
    with open(MODEL_DIR / "feature_scaler_classificador.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)

    aux_features = [c for c in artifacts["selected_features"] if c != qc13.TARGET]
    feature_order_c_causal = (
        ["Hsig_observado", "Hsig_previsto", "residuo", "residuo_absoluto"]
        + ["S_acc", "S_level", "S_rate", "S_range", "S_MAD_score", "S_GPD_score"]
        + [f"EWMA_lambda_{lam}" for lam in qc13.EWMA_LAMBDAS]
        + ["CUSUM_pos", "CUSUM_neg"]
        + [f"slope_{w}h" for w in qc13.SLOPE_WINDOWS]
        + ["persistencia_sinal"]
        + [f"aux__{c}" for c in aux_features]
    )

    metadata = {
        "modelo_versao": "qc_lstm_causal_ordinal_v1",
        "config": CONFIG,
        "variavel_alvo": qc13.TARGET,
        "janela_entrada_horas": qc13.LOOKBACK,
        "variaveis_selecionadas_lstm": artifacts["selected_features"],
        "variaveis_auxiliares_ordem": aux_features,
        "vetor_features_classificador_ordem": feature_order_c_causal,
        "n_features_classificador": len(feature_order_c_causal),
        "tau_b": tau_b,
        "faixa_fisica_min": artifacts["range_min"],
        "faixa_fisica_max": artifacts["range_max"],
        "faixa_fisica_max_multiplicador": qc13.RANGE_MAX_MULTIPLIER,
        "referencia_z_score_media": artifacts["ref_mean"],
        "referencia_z_score_desvio": artifacts["ref_std"],
        "ewma_lambdas": qc13.EWMA_LAMBDAS,
        "slope_janelas_horas": qc13.SLOPE_WINDOWS,
        "cusum_k_fator": qc13.CUSUM_K_FACTOR,
        "cusum_reset_fator": 20.0,
        "classes": qc13.CLASS_NAMES,
        "classe_para_inteiro": qc13.CLASS_TO_INT,
        "metricas_teste_referencia": {
            "macro_f1": ev["macro_f1"], "f1_good": ev["f1_good"], "f1_suspect": ev["f1_suspect"],
            "f1_bad": ev["f1_bad"], "binary_f1": ev["binary_f1"], "balanced_accuracy": ev["balanced_accuracy"],
            "mcc": ev["mcc"],
            "auprc_good": ev["auprc_good"], "auprc_suspect": ev["auprc_suspect"], "auprc_bad": ev["auprc_bad"],
            "auroc_good": ev["auroc_good"], "auroc_suspect": ev["auroc_suspect"], "auroc_bad": ev["auroc_bad"],
        },
        "nota_auprc_bad": (
            "AUPRC de BAD deve ser lido contra o baseline aleatorio, que e a propria prevalencia da "
            "classe no teste (tipicamente 2 a 3 por cento aqui), nao contra 1.0. Um AUPRC de ~0.12 com "
            "prevalencia de ~2.3 por cento e cerca de 5x o baseline aleatorio, sinal real mas modesto; a "
            "varredura completa do grid de tau_B (ver precision_recall_by_tau em "
            "codigos/13_qc_lstm_causal_ordinal_ba1.py) mostra que nenhum limiar leva a precisao de BAD "
            "muito acima de 0.18, um teto estrutural dado o quanto BAD se sobrepoe a SUSPECT no espaco "
            "de features, nao um efeito de escolha de limiar."
        ),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Modelo salvo em", MODEL_DIR)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
