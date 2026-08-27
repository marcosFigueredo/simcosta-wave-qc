from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc16", BASE_DIR / "codigos" / "16_qc_lstm_univariado_ba1.py")
qc16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc16)
qc13 = qc16.qc13
sys.modules["qc13"] = qc13  # necessario para pickle do ClippedRobustScaler localizar a classe
qc10 = qc16.qc10

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = BASE_DIR / "dadosSimcosta"

TARGET = qc16.TARGET
LOOKBACK = qc16.LOOKBACK
CLASS_NAMES = qc13.CLASS_NAMES
CLASS_TO_INT = qc13.CLASS_TO_INT

# Todas as boias entram no treinamento geral, so a variavel-alvo Hsig,
# nenhuma variavel auxiliar (nem todas as boias tem as mesmas disponiveis).
BUOYS = {
    "BA-1": "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv",
    "ES-1": "SIMCOSTA_ES-1_OCEAN_2023-05-30_2026-07-25.csv",
    "PR-1": "SIMCOSTA_PR-1_OCEAN_2013-11-20_2025-05-23.csv",
    "RJ-1": "SIMCOSTA_RJ-1_OCEAN_2015-07-29_2016-10-13.csv",
    "RJ-2": "SIMCOSTA_RJ-2_OCEAN_2015-07-29_2016-12-20.csv",
    "RJ-4": "SIMCOSTA_RJ-4_OCEAN_2017-08-28_2026-07-25.csv",
}
MAX_HOURLY_RECORDS_PER_BUOY = qc10.MAX_HOURLY_RECORDS

MODEL_SEED = 42
TRAIN_EVENTS_PER_FAMILY, VAL_EVENTS_PER_FAMILY, TEST_EVENTS_PER_FAMILY = 10, 6, 4
TRAIN_FRACTION, VAL_FRACTION = qc10.TRAIN_FRACTION, qc10.VAL_FRACTION


def load_buoy_splits() -> dict[str, dict]:
    """Le e separa cada boia (60/20/20 cronologico, dentro da propria serie
    da boia, nunca misturando tempo entre boias)."""
    per_buoy = {}
    for name, filename in BUOYS.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"{name}: arquivo nao encontrado, pulando.")
            continue
        hourly, missing_mask = qc16.read_buoy_target_hourly(path)
        if len(hourly) < LOOKBACK + 300:
            print(f"{name}: serie curta demais ({len(hourly)} pontos), pulando.")
            continue
        hourly = hourly.tail(MAX_HOURLY_RECORDS_PER_BUOY)
        missing_mask = missing_mask.reindex(hourly.index).fillna(0).astype(int)

        n = len(hourly)
        n_train = int(n * TRAIN_FRACTION)
        n_val = int(n * (TRAIN_FRACTION + VAL_FRACTION))
        per_buoy[name] = {
            "train": (hourly.iloc[:n_train], missing_mask.iloc[:n_train]),
            "val": (hourly.iloc[n_train:n_val], missing_mask.iloc[n_train:n_val]),
            "test": (hourly.iloc[n_val:], missing_mask.iloc[n_val:]),
        }
        print(f"{name}: {n} horas ({hourly.index.min().date()} a {hourly.index.max().date()}), "
              f"treino {n_train}, val {n_val - n_train}, teste {n - n_val}")
    return per_buoy


def process_buoy_segment(buoy_name: str, hourly: pd.DataFrame, mask: pd.Series, predictor, scaler,
                          events_per_family: int, anomaly_seed: int, ref_mean: float, ref_std: float) -> dict:
    scaled = scaler.transform(hourly)
    scaled = np.clip(scaled, 0.0, 1.0)
    x_seq, y_seq_scaled, center_idx = qc10.make_sequences_single(scaled, 0, LOOKBACK)
    target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]

    pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
    predicted = pred_scaled * target_range + target_min
    observed = y_seq_scaled * target_range + target_min
    timestamps = hourly.index[center_idx]
    mask_at_center = mask.to_numpy()[center_idx]

    y_aug, class_int, family = qc13.inject_anomalies_causal(observed, events_per_family, anomaly_seed)
    residual = y_aug - predicted
    stats = qc13.causal_stat_vector(y_aug, residual)
    accum = qc13.accumulation_features(residual)
    class_int, confidence = qc13.weak_label_decoupled(y_aug, class_int, mask_at_center, ref_mean, ref_std)
    sentinel_flag = pd.Series(family).str.startswith("G_").to_numpy()

    return {
        "buoy": np.array([buoy_name] * len(center_idx)), "timestamps": timestamps,
        "observed": observed, "y_aug": y_aug, "predicted": predicted,
        "residual": residual, "abs_residual": np.abs(residual), "mask": mask_at_center,
        "class_int": class_int, "confidence": confidence, "family": family,
        "stats": stats, "accum": accum, "sentinel_flag": sentinel_flag,
    }


def pool(dicts: list[dict]) -> dict:
    """Concatena os dicionarios de varias boias campo a campo. stats/accum
    (dicts internos) sao concatenados chave a chave."""
    out = {}
    array_keys = ["buoy", "timestamps", "observed", "y_aug", "predicted", "residual", "abs_residual",
                  "mask", "class_int", "confidence", "family", "sentinel_flag"]
    for k in array_keys:
        out[k] = np.concatenate([d[k] for d in dicts])
    for sub in ["stats", "accum"]:
        out[sub] = {}
        keys = [k for k in dicts[0][sub].keys() if isinstance(dicts[0][sub][k], np.ndarray)]
        for k in keys:
            out[sub][k] = np.concatenate([d[sub][k] for d in dicts])
        for k in dicts[0][sub].keys():
            if k not in keys:
                out[sub][k] = dicts[0][sub][k]
    return out


def train_general_pipeline(model_seed: int = MODEL_SEED, anomaly_seed_base: int = 600,
                            per_buoy: dict | None = None) -> dict:
    """Roda o pipeline completo do modelo geral multi-boia e devolve os
    artefatos em memoria (sem salvar nada em disco, sem plotar). Reutilizavel
    tanto pelo main() (que salva o modelo final) quanto pelo protocolo de
    estabilidade multi-semente (script 20), que so precisa das metricas.

    model_seed controla a inicializacao/ordem de treino da rede (LSTM +
    classificador). anomaly_seed_base desloca as sementes de injecao de
    anomalia por boia e por split (train/val/test), mantendo-as
    independentes entre si e do model_seed.
    """
    qc13.qc04.set_reproducibility(model_seed)
    if per_buoy is None:
        per_buoy = load_buoy_splits()
    if len(per_buoy) < 2:
        raise RuntimeError("Poucas boias disponiveis para treino geral.")

    train_all = pd.concat([per_buoy[b]["train"][0] for b in per_buoy])
    scaler = MinMaxScaler()
    scaler.fit(train_all)

    predictor = qc16.build_predictor_univariate(LOOKBACK)
    x_train_list, y_train_list, x_val_list, y_val_list = [], [], [], []
    for name, splits in per_buoy.items():
        train_h, _ = splits["train"]
        val_h, _ = splits["val"]
        train_scaled = np.clip(scaler.transform(train_h), 0.0, 1.0)
        val_scaled = np.clip(scaler.transform(val_h), 0.0, 1.0)
        x_tr, y_tr, _ = qc10.make_sequences_single(train_scaled, 0, LOOKBACK)
        x_va, y_va, _ = qc10.make_sequences_single(val_scaled, 0, LOOKBACK)
        x_train_list.append(x_tr); y_train_list.append(y_tr)
        x_val_list.append(x_va); y_val_list.append(y_va)
    x_train, y_train = np.concatenate(x_train_list), np.concatenate(y_train_list)
    x_val, y_val = np.concatenate(x_val_list), np.concatenate(y_val_list)

    from tensorflow.keras.callbacks import EarlyStopping
    print(f"\nTreinando LSTM preditora geral com {len(x_train)} janelas de treino (todas as boias combinadas)...")
    predictor.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=64, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    )

    train_target_all = train_all[TARGET].to_numpy()
    qc13.set_physical_range(train_target_all)
    ref_mean, ref_std = float(np.nanmean(train_target_all)), float(np.nanstd(train_target_all))

    d_train_list, d_val_list, d_test_list = [], [], []
    for i, (name, splits) in enumerate(per_buoy.items()):
        train_h, train_m = splits["train"]
        val_h, val_m = splits["val"]
        test_h, test_m = splits["test"]
        d_train_list.append(process_buoy_segment(name, train_h, train_m, predictor, scaler, TRAIN_EVENTS_PER_FAMILY, anomaly_seed_base + 1 + i, ref_mean, ref_std))
        d_val_list.append(process_buoy_segment(name, val_h, val_m, predictor, scaler, VAL_EVENTS_PER_FAMILY, anomaly_seed_base + 101 + i, ref_mean, ref_std))
        d_test_list.append(process_buoy_segment(name, test_h, test_m, predictor, scaler, TEST_EVENTS_PER_FAMILY, anomaly_seed_base + 201 + i, ref_mean, ref_std))

    d_train, d_val, d_test = pool(d_train_list), pool(d_val_list), pool(d_test_list)

    feats_train = qc16.assemble_features_univariate(d_train)
    feats_val = qc16.assemble_features_univariate(d_val)
    feats_test = qc16.assemble_features_univariate(d_test)

    print(f"Treinando cabeca classificadora geral com {len(feats_train)} pontos combinados...")
    model, feat_scaler = qc13.train_ordinal(
        feats_train, d_train["class_int"], d_train["confidence"],
        feats_val, d_val["class_int"], d_val["confidence"], seed=model_seed,
    )
    probs_val = qc13.ordinal_probs(model, feat_scaler.transform(feats_val))
    tau_b = qc13.tune_tau_b(d_val["class_int"], probs_val, d_val["sentinel_flag"])

    probs_test = qc13.ordinal_probs(model, feat_scaler.transform(feats_test))
    ev_all = qc13.evaluate_ordinal(d_test["class_int"], probs_test, tau_b, d_test["sentinel_flag"])
    strong_mask = d_test["confidence"] >= 0.999
    ev_global = qc13.evaluate_ordinal(
        d_test["class_int"][strong_mask], probs_test[strong_mask], tau_b,
        d_test["sentinel_flag"][strong_mask],
    )
    print(f"\nGeral (todas as boias combinadas) -> macro_f1={ev_global['macro_f1']:.3f} "
          f"f1_bad={ev_global['f1_bad']:.3f} binary_f1={ev_global['binary_f1']:.3f} "
          f"auprc_bad={ev_global['auprc_bad']:.3f} tau_b={tau_b:.2f}")

    rows = [{"boia": "TODAS (rótulos fortes)", "n": int(strong_mask.sum()),
             "n_weak": int((~strong_mask).sum()), "tau_b": tau_b,
             "binary_f1_all": ev_all["binary_f1"], **{
        k: ev_global[k] for k in ["macro_f1", "f1_good", "f1_suspect", "f1_bad", "binary_f1",
                                   "binary_precision", "binary_recall", "auprc_bad", "auroc_bad"]
    }}]

    for name in per_buoy:
        mask_b = d_test["buoy"] == name
        if mask_b.sum() == 0:
            continue
        strong_b = mask_b & strong_mask
        ev_b_all = qc13.evaluate_ordinal(d_test["class_int"][mask_b], probs_test[mask_b], tau_b, d_test["sentinel_flag"][mask_b])
        ev_b = qc13.evaluate_ordinal(d_test["class_int"][strong_b], probs_test[strong_b], tau_b, d_test["sentinel_flag"][strong_b])
        rows.append({"boia": name, "n": int(strong_b.sum()), "n_weak": int((mask_b & ~strong_mask).sum()),
                     "tau_b": tau_b, "binary_f1_all": ev_b_all["binary_f1"], **{
            k: ev_b[k] for k in ["macro_f1", "f1_good", "f1_suspect", "f1_bad", "binary_f1",
                                  "binary_precision", "binary_recall", "auprc_bad", "auroc_bad"]
        }})
        print(f"  {name}: macro_f1={ev_b['macro_f1']:.3f} binary_f1={ev_b['binary_f1']:.3f} "
              f"auprc_bad={ev_b['auprc_bad']:.3f} n={mask_b.sum()}")

    df = pd.DataFrame(rows)

    return {
        "predictor": predictor, "classifier": model, "feat_scaler": feat_scaler, "scaler": scaler,
        "tau_b": tau_b, "ref_mean": ref_mean, "ref_std": ref_std,
        "range_min": qc13.RANGE_MIN, "range_max": qc13.RANGE_MAX,
        "per_buoy_names": list(per_buoy.keys()), "df_por_boia": df, "ev_global": ev_global,
        "d_test": d_test,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    per_buoy = load_buoy_splits()
    art = train_general_pipeline(MODEL_SEED, anomaly_seed_base=600, per_buoy=per_buoy)
    predictor, model, feat_scaler, scaler = art["predictor"], art["classifier"], art["feat_scaler"], art["scaler"]
    tau_b, ref_mean, ref_std = art["tau_b"], art["ref_mean"], art["ref_std"]
    df, ev_global = art["df_por_boia"], art["ev_global"]
    df.to_csv(OUT_DIR / "desempenho_geral_por_boia.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(ev_global["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(ev_global["confusion_matrix"][i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_matriz_confusao_geral.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Modelo geral multi-boia (so Hsig, sem variaveis auxiliares)",
        "",
        f"Treinado com dados combinados de {len(per_buoy)} boias SiMCosta ({', '.join(per_buoy.keys())}),",
        "divisao 60/20/20 cronologica dentro de cada boia (nunca misturando tempo entre boias),",
        "depois agrupadas num unico conjunto de treino/validacao/teste.",
        "",
        df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "As métricas principais usam somente pontos com rótulo forte; n_weak registra os pontos",
        "com rótulo fraco, e binary_f1_all conserva a análise de sensibilidade no conjunto completo.",
    ]
    (OUT_DIR / "geral_multiboia_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")

    # -----------------------------------------------------------------
    # Salvar o modelo geral (mesmo padrao do modelo BA-1 especialista em
    # resultados_qc_ba1/qc_lstm_causal_ordinal/modelo_final/), incluindo
    # documentacao de entrada/saida, para reuso no simulador em tempo real.
    # -----------------------------------------------------------------
    model_dir = OUT_DIR / "modelo_final"
    model_dir.mkdir(parents=True, exist_ok=True)

    predictor.save(model_dir / "predictor_lstm.keras")
    model.save(model_dir / "classificador_ordinal.keras")
    with open(model_dir / "input_scaler_lstm.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_dir / "feature_scaler_classificador.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)

    feature_order = (
        ["Hsig_observado", "Hsig_previsto", "residuo", "residuo_absoluto"]
        + ["S_acc", "S_level", "S_rate", "S_range", "S_MAD_score", "S_GPD_score"]
        + [f"EWMA_lambda_{lam}" for lam in qc13.EWMA_LAMBDAS]
        + ["CUSUM_pos", "CUSUM_neg"]
        + [f"slope_{w}h" for w in qc13.SLOPE_WINDOWS]
        + ["persistencia_sinal", "mascara_ausencia"]
    )
    metadata = {
        "modelo_versao": "qc_lstm_geral_multiboia_v1",
        "boias_de_treino": list(per_buoy.keys()),
        "variavel_alvo": TARGET,
        "janela_entrada_horas": LOOKBACK,
        "vetor_features_classificador_ordem": feature_order,
        "n_features_classificador": len(feature_order),
        "tau_b": tau_b,
        "faixa_fisica_min": qc13.RANGE_MIN,
        "faixa_fisica_max": qc13.RANGE_MAX,
        "referencia_z_score_media": ref_mean,
        "referencia_z_score_desvio": ref_std,
        "ewma_lambdas": qc13.EWMA_LAMBDAS,
        "slope_janelas_horas": qc13.SLOPE_WINDOWS,
        "cusum_k_fator": qc13.CUSUM_K_FACTOR,
        "cusum_reset_fator": 20.0,
        "classes": CLASS_NAMES,
        "classe_para_inteiro": CLASS_TO_INT,
        "metricas_teste_por_boia": df.set_index("boia").to_dict(orient="index"),
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    readme = [
        "# Modelo geral multi-boia, especificacao de entrada e saida",
        "",
        f"Treinado com dados combinados de {', '.join(per_buoy.keys())} (so Hsig, sem variaveis",
        "auxiliares). Mesma logica de entrada/saida do modelo especialista da BA-1",
        "(`resultados_qc_ba1/qc_lstm_causal_ordinal/modelo_final/README_entrada_saida.md`), a",
        "diferenca e so o vetor de features aqui tem 21 posicoes (sem as 5 variaveis auxiliares que",
        "so a BA-1 tinha disponiveis) em vez de 25, e os parametros de calibracao (`tau_b`, faixa",
        "fisica, media/desvio de referencia) vem do treino combinado, nao so da BA-1.",
        "",
        "Arquivos, `predictor_lstm.keras` (preve Hsig um passo a frente, entrada (24h, 1)),",
        "`classificador_ordinal.keras` (duas saidas sigmoid q1/q2, entrada com 21 features na ordem",
        "de `metadata.json -> vetor_features_classificador_ordem`), `input_scaler_lstm.pkl`,",
        "`feature_scaler_classificador.pkl`, `metadata.json`.",
    ]
    (model_dir / "README_entrada_saida.md").write_text("\n".join(readme), encoding="utf-8")

    print("\nModelo geral salvo em", model_dir)
    print("\nResultados em:", OUT_DIR)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
