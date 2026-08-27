from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parents[1]
_spec16 = importlib.util.spec_from_file_location("qc16", BASE_DIR / "codigos" / "16_qc_lstm_univariado_ba1.py")
qc16 = importlib.util.module_from_spec(_spec16)
_spec16.loader.exec_module(qc16)
qc13 = qc16.qc13
qc10 = qc16.qc10

_spec18 = importlib.util.spec_from_file_location("qc18", BASE_DIR / "codigos" / "18_qc_lstm_geral_multiboia.py")
qc18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(qc18)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "leave_one_buoy_out_e3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "dadosSimcosta"

LOOKBACK = qc18.LOOKBACK
TARGET = qc18.TARGET
MODEL_SEED = 42
ANOMALY_SEED_BASE = 700


def run_leave_one_buoy_out() -> pd.DataFrame:
    print("=" * 80)
    print("INICIANDO EXPERIMENTO E3: LEAVE-ONE-BUOY-OUT (6 DOBRAS)")
    print("=" * 80)
    
    per_buoy = qc18.load_buoy_splits()
    buoy_names = list(per_buoy.keys())
    
    # Carrega também os resultados do modelo geral (onde a boia estava no treino)
    csv_pooled = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "desempenho_geral_por_boia.csv"
    df_pooled = pd.read_csv(csv_pooled).set_index("boia") if csv_pooled.exists() else None

    results = []

    for fold_idx, held_out in enumerate(buoy_names, 1):
        print(f"\n>>> DOBRA {fold_idx}/6: Boia mantida de fora (held-out): {held_out}")
        train_buoys = {k: v for k, v in per_buoy.items() if k != held_out}
        print(f"    Treinando em 5 boias: {list(train_buoys.keys())}")
        
        qc13.qc04.set_reproducibility(MODEL_SEED)

        train_all = pd.concat([train_buoys[b]["train"][0] for b in train_buoys])
        scaler = MinMaxScaler()
        scaler.fit(train_all)

        predictor = qc16.build_predictor_univariate(LOOKBACK)
        x_train_list, y_train_list, x_val_list, y_val_list = [], [], [], []
        
        for name, splits in train_buoys.items():
            train_h, _ = splits["train"]
            val_h, _ = splits["val"]
            train_scaled = np.clip(scaler.transform(train_h), 0.0, 1.0)
            val_scaled = np.clip(scaler.transform(val_h), 0.0, 1.0)
            x_tr, y_tr, _ = qc10.make_sequences_single(train_scaled, 0, LOOKBACK)
            x_va, y_va, _ = qc10.make_sequences_single(val_scaled, 0, LOOKBACK)
            x_train_list.append(x_tr)
            y_train_list.append(y_tr)
            x_val_list.append(x_va)
            y_val_list.append(y_va)
            
        x_train, y_train = np.concatenate(x_train_list), np.concatenate(y_train_list)
        x_val, y_val = np.concatenate(x_val_list), np.concatenate(y_val_list)

        predictor.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=64, verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        )

        train_target_all = train_all[TARGET].to_numpy()
        # qc16 and qc18 load qc13 through separate importlib module objects;
        # configure both references before process_buoy_segment uses the range.
        qc13.set_physical_range(train_target_all)
        qc18.qc13.set_physical_range(train_target_all)
        ref_mean, ref_std = float(np.nanmean(train_target_all)), float(np.nanstd(train_target_all))

        d_train_list, d_val_list = [], []
        for i, (name, splits) in enumerate(train_buoys.items()):
            train_h, train_m = splits["train"]
            val_h, val_m = splits["val"]
            d_train_list.append(qc18.process_buoy_segment(name, train_h, train_m, predictor, scaler, qc18.TRAIN_EVENTS_PER_FAMILY, ANOMALY_SEED_BASE + 1 + i, ref_mean, ref_std))
            d_val_list.append(qc18.process_buoy_segment(name, val_h, val_m, predictor, scaler, qc18.VAL_EVENTS_PER_FAMILY, ANOMALY_SEED_BASE + 101 + i, ref_mean, ref_std))

        d_train, d_val = qc18.pool(d_train_list), qc18.pool(d_val_list)

        feats_train = qc16.assemble_features_univariate(d_train)
        feats_val = qc16.assemble_features_univariate(d_val)

        model, feat_scaler = qc13.train_ordinal(
            feats_train, d_train["class_int"], d_train["confidence"],
            feats_val, d_val["class_int"], d_val["confidence"], seed=MODEL_SEED,
        )
        probs_val = qc13.ordinal_probs(model, feat_scaler.transform(feats_val))
        tau_b = qc13.tune_tau_b(d_val["class_int"], probs_val, d_val["sentinel_flag"])

        # Avaliação na boia deixada de fora (held-out)
        test_h, test_m = per_buoy[held_out]["test"]
        d_test = qc18.process_buoy_segment(held_out, test_h, test_m, predictor, scaler, qc18.TEST_EVENTS_PER_FAMILY, ANOMALY_SEED_BASE + 201 + fold_idx, ref_mean, ref_std)
        feats_test = qc16.assemble_features_univariate(d_test)
        
        probs_test = qc13.ordinal_probs(model, feat_scaler.transform(feats_test))
        strong_mask = d_test["confidence"] >= 0.999
        ev_all = qc13.evaluate_ordinal(d_test["class_int"], probs_test, tau_b, d_test["sentinel_flag"])
        ev = qc13.evaluate_ordinal(
            d_test["class_int"][strong_mask], probs_test[strong_mask], tau_b,
            d_test["sentinel_flag"][strong_mask],
        )

        pooled_bin_f1 = df_pooled.loc[held_out, "binary_f1"] if (df_pooled is not None and held_out in df_pooled.index) else np.nan
        pooled_macro_f1 = df_pooled.loc[held_out, "macro_f1"] if (df_pooled is not None and held_out in df_pooled.index) else np.nan
        
        diff_bin_f1 = ev["binary_f1"] - pooled_bin_f1

        row = {
            "held_out_buoy": held_out,
            "train_buoys": ",".join(train_buoys.keys()),
            "n_test": int(strong_mask.sum()),
            "n_weak": int((~strong_mask).sum()),
            "binary_f1_all": ev_all["binary_f1"],
            "tau_b": tau_b,
            "macro_f1": ev["macro_f1"],
            "f1_good": ev["f1_good"],
            "f1_suspect": ev["f1_suspect"],
            "f1_bad": ev["f1_bad"],
            "binary_f1": ev["binary_f1"],
            "binary_precision": ev["binary_precision"],
            "binary_recall": ev["binary_recall"],
            "auprc_bad": ev["auprc_bad"],
            "auroc_bad": ev["auroc_bad"],
            "pooled_binary_f1": pooled_bin_f1,
            "delta_binary_f1": diff_bin_f1,
        }
        results.append(row)
        print(f"    --> Resultado em {held_out}: Binary-F1 = {ev['binary_f1']:.3f} (Pooled = {pooled_bin_f1:.3f}, Delta = {diff_bin_f1:+.3f}) | Macro-F1 = {ev['macro_f1']:.3f} | F1_BAD = {ev['f1_bad']:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "leave_one_buoy_out_resultados.csv", index=False)
    
    # Médias agregadas
    mean_row = {
        "held_out_buoy": "Média LOBO",
        "train_buoys": "-",
        "n_test": int(df_res["n_test"].sum()),
        "tau_b": float(df_res["tau_b"].mean()),
        "macro_f1": float(df_res["macro_f1"].mean()),
        "f1_good": float(df_res["f1_good"].mean()),
        "f1_suspect": float(df_res["f1_suspect"].mean()),
        "f1_bad": float(df_res["f1_bad"].mean()),
        "binary_f1": float(df_res["binary_f1"].mean()),
        "binary_precision": float(df_res["binary_precision"].mean()),
        "binary_recall": float(df_res["binary_recall"].mean()),
        "auprc_bad": float(df_res["auprc_bad"].mean()),
        "auroc_bad": float(df_res["auroc_bad"].mean()),
        "pooled_binary_f1": float(df_res["pooled_binary_f1"].mean()),
        "delta_binary_f1": float(df_res["delta_binary_f1"].mean()),
    }
    df_full = pd.concat([df_res, pd.DataFrame([mean_row])], ignore_index=True)
    df_full.to_csv(OUT_DIR / "leave_one_buoy_out_tabela_completa.csv", index=False)
    
    print("\n" + "=" * 80)
    print("RESUMO LEAVE-ONE-BUOY-OUT (E3):")
    print(df_full[["held_out_buoy", "n_test", "macro_f1", "f1_bad", "binary_f1", "pooled_binary_f1", "delta_binary_f1"]])
    print("=" * 80)
    return df_full


if __name__ == "__main__":
    run_leave_one_buoy_out()
