from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ajustes_metricas import descriptive_variance_allocation, streaming_operational_metrics

BASE_DIR = Path(__file__).resolve().parents[1]
_spec16 = importlib.util.spec_from_file_location("qc16", BASE_DIR / "codigos" / "16_qc_lstm_univariado_ba1.py")
qc16 = importlib.util.module_from_spec(_spec16)
_spec16.loader.exec_module(qc16)
qc13 = qc16.qc13
qc10 = qc16.qc10

_spec18 = importlib.util.spec_from_file_location("qc18", BASE_DIR / "codigos" / "18_qc_lstm_geral_multiboia.py")
qc18 = importlib.util.module_from_spec(_spec18)
_spec18.loader.exec_module(qc18)

_spec01 = importlib.util.spec_from_file_location("qc01", BASE_DIR / "codigos" / "01_spike_test_simcosta_ba1.py")
qc01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(qc01)

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "ajustes_vagner"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "dadosSimcosta"


# ---------------------------------------------------------------------------
# 1. Erro Preditivo da LSTM Univariada 168h na Partição de Teste Combinada (Tabela 2)
# ---------------------------------------------------------------------------
def compute_prediction_error_e2() -> dict[str, float]:
    print("--> 1. Calculando erro de previsão da LSTM univariada (E2)...")
    model_dir = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "modelo_final"
    import pickle
    import tensorflow as tf
    predictor = tf.keras.models.load_model(model_dir / "predictor_lstm.keras")
    with open(model_dir / "input_scaler_lstm.pkl", "rb") as f:
        scaler = pickle.load(f)

    per_buoy = qc18.load_buoy_splits()
    
    y_true_all = []
    y_pred_all = []
    
    for name, splits in per_buoy.items():
        test_hourly, _ = splits["test"]
        scaled = scaler.transform(test_hourly)
        scaled = np.clip(scaled, 0.0, 1.0)
        x_seq, y_seq_scaled, _ = qc10.make_sequences_single(scaled, 0, qc16.LOOKBACK)
        target_min, target_range = scaler.data_min_[0], scaler.data_range_[0]
        
        pred_scaled = predictor.predict(x_seq, verbose=0).ravel()
        pred_m = pred_scaled * target_range + target_min
        obs_m = y_seq_scaled * target_range + target_min
        
        y_true_all.append(obs_m)
        y_pred_all.append(pred_m)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE com proteção contra divisão por zero (< 0.01m)
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    res = {"mae": mae, "rmse": rmse, "mape": mape, "n_test_points": len(y_true)}
    print(f"   E2 (Six buoys pooled): MAE = {mae:.4f} m, RMSE = {rmse:.4f} m, MAPE = {mape:.2f} % (n={len(y_true)})")
    
    pd.DataFrame([res]).to_csv(OUT_DIR / "erro_predicao_e2.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# 2. Censo Detalhado das 6 Boias SiMCosta (Tabela S4)
# ---------------------------------------------------------------------------
def compute_buoy_census() -> pd.DataFrame:
    print("--> 2. Gerando Censo das Boias SiMCosta (Tabela S4)...")
    rows = []
    buoys = qc18.BUOYS
    
    for name, filename in buoys.items():
        path = DATA_DIR / filename
        df = pd.read_csv(path, comment="/", na_values=["NULL", "null", "", "NaN"])
        df["datetime_utc"] = pd.to_datetime(
            df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]], errors="coerce", utc=True
        )
        df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
        df = df.drop_duplicates("datetime_utc").set_index("datetime_utc")
        
        raw_samples = len(df)
        dt_min = df.index.min()
        dt_max = df.index.max()
        
        # Horas totais no período
        total_hours_period = int((dt_max - dt_min).total_seconds() / 3600) + 1
        
        # Hsig
        hsig_raw = pd.to_numeric(df["Hsig"], errors="coerce")
        hsig_valid_raw = int(hsig_raw.notna().sum())
        
        # Resampling horário e cobertura
        raw_hourly = df[["Hsig"]].resample("1h").median()
        hourly_valid = int(raw_hourly["Hsig"].notna().sum())
        hourly_coverage_pct = (hourly_valid / total_hours_period) * 100.0
        
        # Truncamento / pipeline
        hourly_interp, _ = qc16.read_buoy_target_hourly(path)
        n_interp = len(hourly_interp)
        hourly_proc = hourly_interp.tail(qc18.MAX_HOURLY_RECORDS_PER_BUOY)
        n_proc = len(hourly_proc)
        
        n_train = int(n_proc * qc18.TRAIN_FRACTION)
        n_val = int(n_proc * (qc18.TRAIN_FRACTION + qc18.VAL_FRACTION)) - n_train
        n_test = n_proc - (n_train + n_val)
        
        # Sequências de teste (Lookback=168h)
        n_test_seq = max(0, n_test - qc16.LOOKBACK)
        
        rows.append({
            "buoy": name,
            "filename": filename,
            "start_date": dt_min.strftime("%Y-%m-%d"),
            "end_date": dt_max.strftime("%Y-%m-%d"),
            "raw_samples": raw_samples,
            "span_hours": total_hours_period,
            "hsig_valid_raw": hsig_valid_raw,
            "hourly_coverage_pct": round(hourly_coverage_pct, 2),
            "hourly_proc": n_proc,
            "train_hours": n_train,
            "val_hours": n_val,
            "test_hours": n_test,
            "test_sequences": n_test_seq,
        })

    df_census = pd.DataFrame(rows)
    df_census.to_csv(OUT_DIR / "censo_boias_tabela_s4.csv", index=False)
    print(df_census[["buoy", "start_date", "end_date", "hourly_coverage_pct", "hourly_proc", "train_hours", "test_hours", "test_sequences"]])
    return df_census


# ---------------------------------------------------------------------------
# 3. Disparo dos 7 Testes do Quality Index e Matriz de Concordância (Tabela S5)
# ---------------------------------------------------------------------------
def compute_qi_firing_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("--> 3. Calculando Disparos do QI e Concordância Par a Par (Tabela S5)...")
    df = qc01.read_data()
    
    # Processa todas as variáveis
    var_results = {}
    all_flags_list = []
    
    for var in qc01.MAIN_VARIABLES:
        if var in df.columns and df[var].notna().sum() > 100:
            res = qc01.qc_variable(df, var)
            var_results[var] = res
            
            # 7 flags booleanas
            f = res[["flag_classic_spike", "flag_mad", "flag_second_difference", "flag_qartod_spike", "flag_roc", "flag_local_std", "flag_range"]].copy()
            f.columns = ["Spike (Classic)", "Median/MAD", "Second Diff.", "QARTOD", "Rate of Change", "Local Std.", "Physical Range"]
            f["variable"] = var
            all_flags_list.append(f)

    all_flags = pd.concat(all_flags_list, ignore_index=True)
    test_names = ["Spike (Classic)", "Median/MAD", "Second Diff.", "QARTOD", "Rate of Change", "Local Std.", "Physical Range"]
    
    # Firing rates individuais
    total_obs = len(all_flags)
    firing_counts = all_flags[test_names].sum()
    firing_pcts = (firing_counts / total_obs) * 100.0
    
    df_firing = pd.DataFrame({
        "Test": test_names,
        "Firing_Count": firing_counts.values,
        "Firing_Pct": firing_pcts.values,
    })
    
    # Também especificamente para Hsig
    hsig_res = var_results["Hsig"]
    hsig_flags = hsig_res[["flag_classic_spike", "flag_mad", "flag_second_difference", "flag_qartod_spike", "flag_roc", "flag_local_std", "flag_range"]].copy()
    hsig_flags.columns = test_names
    hsig_obs = len(hsig_flags.dropna())
    df_firing["Hsig_Count"] = hsig_flags.sum().values
    df_firing["Hsig_Pct"] = (hsig_flags.sum().values / hsig_obs) * 100.0

    # Matriz de concordância (Jaccard / Co-ocorrência) par a par
    matrix_overlap = pd.DataFrame(index=test_names, columns=test_names, dtype=float)
    for t1 in test_names:
        for t2 in test_names:
            c1 = all_flags[t1].astype(bool)
            c2 = all_flags[t2].astype(bool)
            # Concordância percentual quando t1 dispara: P(t2 | t1)
            p_t2_given_t1 = (c1 & c2).sum() / max(1, c1.sum()) * 100.0
            matrix_overlap.loc[t1, t2] = round(p_t2_given_t1, 1)

    df_firing.to_csv(OUT_DIR / "disparos_qi_tabela_s5.csv", index=False)
    matrix_overlap.to_csv(OUT_DIR / "concordancia_qi_tabela_s5.csv")
    print(df_firing)
    print("Matriz de Co-ocorrência P(Coluna | Linha) %:")
    print(matrix_overlap)
    return df_firing, matrix_overlap


# ---------------------------------------------------------------------------
# 4. Decomposição da Variância ANOVA Bidirecional (3x5) da Estabilidade
# ---------------------------------------------------------------------------
def compute_variance_decomposition() -> pd.DataFrame:
    print("--> 4. Executando Decomposição de Variância ANOVA (3x5)...")
    csv_path = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "estabilidade" / "estabilidade_geral_execucoes.csv"
    df = pd.read_csv(csv_path)
    
    metrics = ["macro_f1", "f1_good", "f1_suspect", "f1_bad", "binary_f1", "binary_precision", "binary_recall", "auprc_bad", "auroc_bad"]
    
    results = []
    
    n_models = df["model_seed"].nunique()  # 3
    n_anom = df["anomaly_seed_base"].nunique()  # 5
    N = len(df)  # 15
    
    for metric in metrics:
        vals = df.pivot(index="model_seed", columns="anomaly_seed_base", values=metric).to_numpy()
        grand_mean = np.mean(vals)
        
        # Somas dos quadrados
        ss_total = np.sum((vals - grand_mean) ** 2)
        row_means = np.mean(vals, axis=1)  # medias por model_seed
        col_means = np.mean(vals, axis=0)  # medias por anomaly_seed
        
        ss_model = n_anom * np.sum((row_means - grand_mean) ** 2)
        ss_anom = n_models * np.sum((col_means - grand_mean) ** 2)
        ss_resid = ss_total - ss_model - ss_anom
        ss_resid = max(0.0, ss_resid)
        
        # Decomposição descritiva. Com uma observação por célula 3x5,
        # interação e erro residual não são separáveis inferencialmente.
        allocation = descriptive_variance_allocation(df, metric)
        
        mean_val = float(df[metric].mean())
        sd_val = float(df[metric].std())
        
        results.append({
            "metric": metric,
            "mean": round(mean_val, 4),
            "sd_total": round(sd_val, 4),
            "pct_var_model_seed": round(allocation["pct_var_model_seed"], 1),
            "pct_var_anomaly_seed": round(allocation["pct_var_anomaly_seed"], 1),
            "pct_var_remaining_interaction_or_error": round(allocation["pct_var_remaining_interaction_or_error"], 1),
        })

    df_anova = pd.DataFrame(results)
    df_anova.to_csv(OUT_DIR / "decomposicao_variancia_anova.csv", index=False)
    print(df_anova)
    return df_anova


# ---------------------------------------------------------------------------
# 5. Métricas Complementares do Simulador de Streaming (Falso Alarme e Cobertura)
# ---------------------------------------------------------------------------
def compute_streaming_metrics() -> pd.DataFrame:
    print("--> 5. Extraindo Métricas do Simulador de Streaming...")
    sim_dir = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "simulacao_tempo_real"
    
    # Avalia log_simulacao.csv
    log_file = sim_dir / "log_simulacao.csv"
    if log_file.exists():
        df_log = pd.read_csv(log_file)
        
        metrics = streaming_operational_metrics(df_log)
        
        print(f"   Simulador (Seed Referência):")
        print(f"   - Horas limpas: {metrics['n_clean_hours']}, Falsos Alarmes: {metrics['false_alarms']} ({metrics['false_alarm_rate_pct']:.2f} %)")
        print(f"   - Episódios: {metrics['n_episodes']}, Cobertura Média ao longo do episódio: {metrics['mean_episode_coverage_pct']:.2f} %")

        res = {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()}
        pd.DataFrame([res]).to_csv(OUT_DIR / "metricas_streaming_complementares.csv", index=False)
        return pd.DataFrame([res])
    return None


if __name__ == "__main__":
    compute_prediction_error_e2()
    compute_buoy_census()
    compute_qi_firing_rates()
    compute_variance_decomposition()
    compute_streaming_metrics()
