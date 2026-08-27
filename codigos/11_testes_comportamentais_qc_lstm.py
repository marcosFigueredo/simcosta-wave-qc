from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc10", BASE_DIR / "codigos" / "10_qc_lstm_3classes_ba1.py")
qc10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc10)
qc04 = qc10.qc04

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_3classes" / "testes_comportamentais"

RNG_SEED = 2026
K_GRID = [0.5, 1, 2, 3, 4, 6]
N_SAMPLE = 200
N_SAMPLE_CAUSALITY = 150
N_SAMPLE_AUX = 100
N_EVENTS_RECOVERY = 12
N_EVENTS_PERSISTENT = 6
RECOVERY_HORIZON = 24
TAU_GOOD = 0.5
PERSISTENT_DURATION = 12
PERSISTENT_BUFFER = 12


# ---------------------------------------------------------------------------
# Helpers shared by the "static" tests (monotonicity, context consistency,
# causality, auxiliary sensitivity): the LSTM prediction and hidden state at
# position t only depend on the input window x_{t-L..t-1}, never on the
# target value AT t or t+1, so h_t/predicted_t can be reused unchanged while
# only the downstream residual/stat/z_t block is recomputed per scenario.
# ---------------------------------------------------------------------------

def build_feature_vector(
    d_test: dict, idx: int, y_val: float, robust_scale: float, gpd_params: dict,
    z_override: np.ndarray | None = None, next_y_override: float | None = None,
    prev_y_override: float | None = None,
) -> np.ndarray:
    h = d_test["h"][idx]
    predicted = d_test["predicted"][idx]
    y_clean = d_test["observed"]
    prev_y = prev_y_override if prev_y_override is not None else (y_clean[idx - 1] if idx > 0 else y_clean[idx])
    next_y = next_y_override if next_y_override is not None else (y_clean[idx + 1] if idx < len(y_clean) - 1 else y_clean[idx])

    residual = y_val - predicted
    abs_residual = abs(residual)
    spike = abs(y_val - (prev_y + next_y) / 2)
    rate = abs(y_val - prev_y)
    range_ind = 1.0 if (y_val < qc10.RANGE_MIN or y_val > qc10.RANGE_MAX or np.isnan(y_val)) else 0.0
    mad_p = qc04.robust_t_pvalues(np.array([residual]), robust_scale)[0]
    mad_score = -np.log(mad_p + 1e-12)
    gpd_p = qc04.gpd_tail_pvalues(np.array([residual]), gpd_params)[0]
    gpd_score = -np.log(gpd_p + 1e-12)
    z_t = z_override if z_override is not None else d_test["z_t"][idx]
    mask = d_test["mask"][idx]

    return np.concatenate([h, [y_val, predicted, residual, abs_residual, spike, mad_score, gpd_score, rate, range_ind], z_t, [mask]])


def score_vectors(vectors: list[np.ndarray], clf_model, feat_scaler) -> np.ndarray:
    x = np.stack(vectors)
    x_s = feat_scaler.transform(x)
    return clf_model.predict(x_s, verbose=0)


# ---------------------------------------------------------------------------
# 8.3 Monotonicity test
# ---------------------------------------------------------------------------

def test_monotonicity(d_test: dict, clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    std = float(np.nanstd(d_test["observed"]))
    n = len(d_test["observed"])
    sample_idx = rng.choice(np.arange(5, n - 5), size=min(N_SAMPLE, n - 10), replace=False)

    vectors, meta = [], []
    for idx in sample_idx:
        for k in K_GRID:
            for s in (1, -1):
                y_hyp = max(0.0, d_test["observed"][idx] + s * k * std)
                vectors.append(build_feature_vector(d_test, idx, y_hyp, d_test["robust_scale"], d_test["gpd_params"]))
                meta.append((idx, s, k))

    probs = score_vectors(vectors, clf_model, feat_scaler)
    p_bad = probs[:, qc10.CLASS_TO_INT["BAD"]]
    df = pd.DataFrame(meta, columns=["idx", "sign", "k"])
    df["p_bad"] = p_bad

    monotonic_ok, total_pairs = 0, 0
    for (idx, s), group in df.groupby(["idx", "sign"]):
        group = group.sort_values("k")
        vals = group["p_bad"].to_numpy()
        for a, b in zip(vals[:-1], vals[1:]):
            total_pairs += 1
            if b >= a - 1e-6:
                monotonic_ok += 1
    rate = monotonic_ok / total_pairs if total_pairs else float("nan")

    return {
        "monotonicity_rate": rate,
        "n_windows": len(sample_idx),
        "n_pairs": total_pairs,
        "meets_target_0_90": bool(rate >= 0.90),
        "detail": df,
    }


# ---------------------------------------------------------------------------
# 8.4 Context consistency test
# ---------------------------------------------------------------------------

def test_context_consistency(d_test: dict, aux_features: list[str], corr: pd.Series, clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    std = float(np.nanstd(d_test["observed"]))
    n = len(d_test["observed"])
    sample_idx = rng.choice(np.arange(5, n - 5), size=min(N_SAMPLE, n - 10), replace=False)
    delta = 3.0 * std
    signs = np.array([np.sign(corr.get(f, 0.0)) or 1.0 for f in aux_features])

    vectors, meta = [], []
    for idx in sample_idx:
        y_hyp = max(0.0, d_test["observed"][idx] + delta)
        # scenario A: Hsig aumenta sem qualquer suporte das variaveis auxiliares
        vectors.append(build_feature_vector(d_test, idx, y_hyp, d_test["robust_scale"], d_test["gpd_params"]))
        meta.append((idx, "A"))
        # scenario B: variaveis auxiliares se movem na direcao coerente com Hsig (sinal da correlacao)
        z_b = np.clip(d_test["z_t"][idx] + signs * 0.15, 0.0, 1.0)
        vectors.append(build_feature_vector(d_test, idx, y_hyp, d_test["robust_scale"], d_test["gpd_params"], z_override=z_b))
        meta.append((idx, "B"))

    probs = score_vectors(vectors, clf_model, feat_scaler)
    df = pd.DataFrame(meta, columns=["idx", "scenario"])
    df["p_good"] = probs[:, qc10.CLASS_TO_INT["GOOD"]]
    df["p_suspect"] = probs[:, qc10.CLASS_TO_INT["SUSPECT"]]
    df["p_bad"] = probs[:, qc10.CLASS_TO_INT["BAD"]]

    wide = df.pivot(index="idx", columns="scenario", values=["p_bad", "p_suspect"])
    ok = (wide[("p_bad", "A")] > wide[("p_bad", "B")]) | (wide[("p_suspect", "B")] > wide[("p_bad", "B")])
    rate = float(ok.mean())

    return {
        "coherence_rate": rate,
        "n_windows": len(sample_idx),
        "mean_p_bad_A": float(wide[("p_bad", "A")].mean()),
        "mean_p_bad_B": float(wide[("p_bad", "B")].mean()),
        "detail": df,
    }


# ---------------------------------------------------------------------------
# 8.5 Causality test
# ---------------------------------------------------------------------------

def test_causality(d_test: dict, clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    std = float(np.nanstd(d_test["observed"]))
    n = len(d_test["observed"])
    sample_idx = rng.choice(np.arange(1, n - 2), size=min(N_SAMPLE_CAUSALITY, n - 5), replace=False)

    vectors, meta = [], []
    for idx in sample_idx:
        y_val = d_test["observed"][idx]
        vectors.append(build_feature_vector(d_test, idx, y_val, d_test["robust_scale"], d_test["gpd_params"]))
        meta.append((idx, "baseline"))
        shocked_next = d_test["observed"][idx] + 6 * std
        vectors.append(build_feature_vector(d_test, idx, y_val, d_test["robust_scale"], d_test["gpd_params"], next_y_override=shocked_next))
        meta.append((idx, "future_shocked"))

    probs = score_vectors(vectors, clf_model, feat_scaler)
    df = pd.DataFrame(meta, columns=["idx", "variant"])
    df["label"] = probs.argmax(axis=1)

    wide = df.pivot(index="idx", columns="variant", values="label")
    violated = wide["baseline"] != wide["future_shocked"]
    rate = float(violated.mean())

    return {
        "causality_violation_rate": rate,
        "n_windows": len(sample_idx),
        "note": "Violacao esperada: o indicador stat_spike usa Hsig_{t+1} por definicao (secao 5.3 do protocolo, documentado como adequado para QC atrasado, nao para operacao causal em tempo real). O estado oculto e a previsao da LSTM em si permanecem causais (dependem so de x_{t-L..t-1}).",
        "detail": df,
    }


# ---------------------------------------------------------------------------
# 8.6 Auxiliary sensitivity test
# ---------------------------------------------------------------------------

def test_auxiliary_sensitivity(d_test: dict, aux_features: list[str], clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    n = len(d_test["observed"])
    sample_idx = rng.choice(np.arange(5, n - 5), size=min(N_SAMPLE_AUX, n - 10), replace=False)

    vectors, meta = [], []
    for idx in sample_idx:
        y_val = d_test["observed"][idx]
        vectors.append(build_feature_vector(d_test, idx, y_val, d_test["robust_scale"], d_test["gpd_params"]))
        meta.append((idx, "baseline", None))
        for j, feat in enumerate(aux_features):
            z_pert = d_test["z_t"][idx].copy()
            z_pert[j] = np.clip(z_pert[j] + 0.05, 0.0, 1.0)
            vectors.append(build_feature_vector(d_test, idx, y_val, d_test["robust_scale"], d_test["gpd_params"], z_override=z_pert))
            meta.append((idx, feat, j))

    probs = score_vectors(vectors, clf_model, feat_scaler)
    df = pd.DataFrame(meta, columns=["idx", "feature", "feature_idx"])
    df["p_good"] = probs[:, qc10.CLASS_TO_INT["GOOD"]]
    df["label"] = probs.argmax(axis=1)

    baseline = df[df["feature"] == "baseline"].set_index("idx")[["p_good", "label"]]
    perturbed = df[df["feature"] != "baseline"].copy()
    perturbed = perturbed.join(baseline, on="idx", rsuffix="_base")

    flips_to_bad = ((perturbed["label_base"] != qc10.CLASS_TO_INT["BAD"]) & (perturbed["label"] == qc10.CLASS_TO_INT["BAD"])).mean()
    mean_delta_p_good = (perturbed["p_good"] - perturbed["p_good_base"]).abs().mean()

    return {
        "flip_to_bad_rate": float(flips_to_bad),
        "mean_abs_delta_p_good": float(mean_delta_p_good),
        "n_windows": len(sample_idx),
        "n_features": len(aux_features),
        "detail": perturbed,
    }


# ---------------------------------------------------------------------------
# 8.7 / 8.8 Feedback simulations: recovery and persistent-failure tests need
# the contaminated value to actually re-enter subsequent input windows, so
# these two work on a modified copy of the scaled array (fed through the
# real predictor/encoder again), unlike the static tests above.
# ---------------------------------------------------------------------------

def score_positions(scaled_array: np.ndarray, end_positions: np.ndarray, artifacts: dict, robust_scale: float, gpd_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookback = qc10.LOOKBACK
    target_idx = artifacts["target_idx"]
    scaler = artifacts["scaler"]
    target_min, target_range = scaler.data_min_[target_idx], scaler.data_range_[target_idx]
    aux_idx = [i for i in range(scaled_array.shape[1]) if i != target_idx]

    x_batch = np.stack([scaled_array[e - lookback:e] for e in end_positions])
    pred_scaled = artifacts["predictor"].predict(x_batch, verbose=0).ravel()
    h_batch = artifacts["encoder"].predict(x_batch, verbose=0)
    predicted = pred_scaled * target_range + target_min
    y_vals = scaled_array[end_positions, target_idx] * target_range + target_min

    prev_pos = np.clip(end_positions - 1, 0, len(scaled_array) - 1)
    next_pos = np.clip(end_positions + 1, 0, len(scaled_array) - 1)
    prev_y = scaled_array[prev_pos, target_idx] * target_range + target_min
    next_y = scaled_array[next_pos, target_idx] * target_range + target_min

    residual = y_vals - predicted
    abs_residual = np.abs(residual)
    spike = np.abs(y_vals - (prev_y + next_y) / 2)
    rate = np.abs(y_vals - prev_y)
    range_ind = ((y_vals < qc10.RANGE_MIN) | (y_vals > qc10.RANGE_MAX)).astype(float)
    mad_p = qc04.robust_t_pvalues(residual, robust_scale)
    mad_score = -np.log(mad_p + 1e-12)
    gpd_p = qc04.gpd_tail_pvalues(residual, gpd_params)
    gpd_score = -np.log(gpd_p + 1e-12)
    z_t = scaled_array[end_positions][:, aux_idx]
    mask = np.zeros(len(end_positions))

    v_full = np.column_stack([h_batch, y_vals, predicted, residual, abs_residual, spike, mad_score, gpd_score, rate, range_ind, z_t, mask])
    return v_full, predicted, y_vals


def test_recovery(artifacts: dict, clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    lookback = qc10.LOOKBACK
    test_scaled = artifacts["scaler"].transform(artifacts["test_df"])
    target_idx = artifacts["target_idx"]
    target_min, target_range = artifacts["scaler"].data_min_[target_idx], artifacts["scaler"].data_range_[target_idx]
    std = float(np.nanstd(artifacts["d_test"]["observed"]))
    robust_scale, gpd_params = artifacts["d_train"]["robust_scale"], artifacts["d_train"]["gpd_params"]

    n_scaled = len(test_scaled)
    candidates = np.arange(lookback + 5, n_scaled - RECOVERY_HORIZON - 5)
    events = rng.choice(candidates, size=min(N_EVENTS_RECOVERY, len(candidates)), replace=False)

    rows = []
    for event_pos in events:
        clean_val_scaled = test_scaled[event_pos, target_idx]
        anomaly_val = (test_scaled[event_pos, target_idx] * target_range + target_min) + 4 * std
        anomaly_val_scaled = (anomaly_val - target_min) / target_range

        # modo 1: realimentacao do proprio valor observado (corrompido)
        arr_observed = test_scaled.copy()
        arr_observed[event_pos, target_idx] = anomaly_val_scaled

        # modo 2: substituicao por previsao/imputacao robusta no ponto contaminado
        window = test_scaled[event_pos - lookback:event_pos][None, :, :]
        pred_scaled_point = artifacts["predictor"].predict(window, verbose=0).ravel()[0]
        arr_imputed = test_scaled.copy()
        arr_imputed[event_pos, target_idx] = pred_scaled_point

        positions = np.arange(event_pos, event_pos + RECOVERY_HORIZON + 1)
        for mode_name, arr in (("realimentacao_observado", arr_observed), ("substituicao_predicao", arr_imputed)):
            v_full, _, _ = score_positions(arr, positions, artifacts, robust_scale, gpd_params)
            x_s = feat_scaler.transform(v_full)
            probs = clf_model.predict(x_s, verbose=0)
            p_good = probs[:, qc10.CLASS_TO_INT["GOOD"]]
            recovered = np.where(p_good >= TAU_GOOD)[0]
            t_rec = int(recovered[0]) if len(recovered) else None
            rows.append({"event_pos": int(event_pos), "mode": mode_name, "t_rec": t_rec, "censored": t_rec is None})

    df = pd.DataFrame(rows)
    summary = df.groupby("mode").apply(
        lambda g: pd.Series(
            {
                "t_rec_mean": g["t_rec"].dropna().mean(),
                "t_rec_median": g["t_rec"].dropna().median(),
                "censor_rate": g["censored"].mean(),
                "n_events": len(g),
            }
        ),
        include_groups=False,
    ).reset_index()

    return {"summary": summary, "detail": df}


def test_persistent_failure(artifacts: dict, clf_model, feat_scaler, rng: np.random.Generator) -> dict:
    lookback = qc10.LOOKBACK
    test_scaled = artifacts["scaler"].transform(artifacts["test_df"])
    target_idx = artifacts["target_idx"]
    target_min, target_range = artifacts["scaler"].data_min_[target_idx], artifacts["scaler"].data_range_[target_idx]
    std = float(np.nanstd(artifacts["d_test"]["observed"]))
    robust_scale, gpd_params = artifacts["d_train"]["robust_scale"], artifacts["d_train"]["gpd_params"]

    n_scaled = len(test_scaled)
    d = PERSISTENT_DURATION
    candidates = np.arange(lookback + 5, n_scaled - d - PERSISTENT_BUFFER - 5)
    events = rng.choice(candidates, size=min(N_EVENTS_PERSISTENT, len(candidates)), replace=False)

    families = ["stuck", "drift", "level_shift"]
    rows = []
    for event_pos in events:
        for family in families:
            arr = test_scaled.copy()
            base_val = arr[event_pos, target_idx] * target_range + target_min
            if family == "stuck":
                new_vals = np.full(d, base_val)
            elif family == "drift":
                delta = 0.25 * std
                new_vals = base_val + np.arange(d) * delta
            else:  # level_shift
                delta = 3.0 * std
                new_vals = np.full(d, base_val + delta)
            new_vals_scaled = (new_vals - target_min) / target_range
            arr[event_pos:event_pos + d, target_idx] = new_vals_scaled

            positions = np.arange(event_pos - 2, event_pos + d + PERSISTENT_BUFFER)
            v_full, _, _ = score_positions(arr, positions, artifacts, robust_scale, gpd_params)
            x_s = feat_scaler.transform(v_full)
            probs = clf_model.predict(x_s, verbose=0)
            labels = probs.argmax(axis=1)
            offsets = positions - event_pos

            episode_mask = (offsets >= 0) & (offsets < d)
            episode_labels = labels[episode_mask]
            suspect_or_bad = np.where(labels != qc10.CLASS_TO_INT["GOOD"])[0]
            bad_only = np.where(labels == qc10.CLASS_TO_INT["BAD"])[0]
            first_suspect = int(offsets[suspect_or_bad[0]]) if len(suspect_or_bad) and offsets[suspect_or_bad[0]] >= 0 else None
            first_bad = int(offsets[bad_only[0]]) if len(bad_only) and offsets[bad_only[0]] >= 0 else None
            detection_prop = float((episode_labels != qc10.CLASS_TO_INT["GOOD"]).mean())

            rows.append(
                {
                    "event_pos": int(event_pos),
                    "family": family,
                    "first_suspect_offset": first_suspect,
                    "first_bad_offset": first_bad,
                    "detection_proportion": detection_prop,
                }
            )

    df = pd.DataFrame(rows)
    summary = df.groupby("family").agg(
        first_suspect_mean=("first_suspect_offset", "mean"),
        first_bad_mean=("first_bad_offset", "mean"),
        detection_proportion_mean=("detection_proportion", "mean"),
        n_events=("family", "count"),
    ).reset_index()

    return {"summary": summary, "detail": df}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results: dict) -> None:
    mono, ctx, caus, aux, rec, pers = (
        results["monotonicity"], results["context"], results["causality"],
        results["aux_sensitivity"], results["recovery"], results["persistent"],
    )
    lines = [
        "# Bateria de testes comportamentais - QC-LSTM 3 classes (secao 8 do protocolo)",
        "",
        "Todos os testes usam o modelo `E_full` treinado por `codigos/10_qc_lstm_3classes_ba1.py`",
        "(uma unica semente de modelo/injecao) aplicado sobre o conjunto de teste real da BA-1.",
        "",
        "## 8.3 Teste de monotonicidade",
        "",
        f"- Taxa de monotonicidade, {mono['monotonicity_rate']:.3f} (meta do protocolo, >= 0.90, "
        f"atingida: {mono['meets_target_0_90']}).",
        f"- {mono['n_windows']} janelas reais, spikes aditivos k in {K_GRID}, ambas as direcoes.",
        "",
        "## 8.4 Teste de coerencia fisica contextual",
        "",
        f"- Proporcao de janelas em que o mesmo aumento de Hsig produz p(BAD) menor (ou p(SUSPECT) maior "
        f"que p(BAD)) quando ha suporte coerente das variaveis auxiliares, {ctx['coherence_rate']:.3f}.",
        f"- p(BAD) medio sem suporte fisico (cenario A), {ctx['mean_p_bad_A']:.3f}; com suporte fisico "
        f"(cenario B), {ctx['mean_p_bad_B']:.3f}.",
        "",
        "## 8.5 Teste de causalidade",
        "",
        f"- Taxa de violacao de causalidade (rotulo em t muda quando Hsig_t+1 e alterado), "
        f"{caus['causality_violation_rate']:.3f}.",
        f"- {caus['note']}",
        "",
        "## 8.6 Teste de sensibilidade a entrada auxiliar",
        "",
        f"- Taxa de mudanca de rotulo para BAD por perturbacao pequena e isolada em uma unica variavel "
        f"auxiliar, {aux['flip_to_bad_rate']:.4f} (meta implicita, proximo de 0).",
        f"- Variacao media absoluta em p(GOOD) por essas perturbacoes, {aux['mean_abs_delta_p_good']:.4f}.",
        "",
        "## 8.7 Teste de recuperacao",
        "",
        "Anomalia isolada (spike +4 desvios-padrao) realimentada nas janelas seguintes; dois modos de",
        "recuperacao comparados.",
        "",
        rec["summary"].to_markdown(index=False, floatfmt=".2f"),
        "",
        "## 8.8 Teste de falha persistente",
        "",
        f"Episodios de {PERSISTENT_DURATION} horas por familia (sensor travado, drift, mudanca de nivel),",
        "tempo ate a primeira sinalizacao SUSPECT/BAD e proporcao do episodio detectada.",
        "",
        pers["summary"].to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Leitura geral",
        "",
        "O teste de causalidade confirma, de forma isolada e mensuravel, a ressalva ja documentada no",
        "protocolo (secao 5.3) de que o indicador estatistico de spike usa informacao futura (Hsig_t+1) e",
        "portanto so e apropriado para QC atrasado, nao para operacao estritamente causal em tempo real;",
        "a previsao e o estado oculto da LSTM continuam causais. Os demais testes comportamentais indicam",
        "se o classificador reage de forma qualitativamente correta (monotonicidade a magnitude do desvio,",
        "reducao de falso alarme quando ha suporte fisico coerente, estabilidade a ruido irrelevante nas",
        "variaveis auxiliares, recuperacao apos anomalia isolada) e nao apenas um bom desempenho agregado.",
    ]
    (OUT_DIR / "testes_comportamentais_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    artifacts = qc10.run_pipeline(ablation_models=["E_full"])
    d_test = artifacts["d_test"]
    clf_model, feat_scaler = artifacts["classifiers"]["E_full"]
    aux_features = artifacts["aux_features"]
    corr = artifacts["corr"]

    mono = test_monotonicity(d_test, clf_model, feat_scaler, rng)
    ctx = test_context_consistency(d_test, aux_features, corr, clf_model, feat_scaler, rng)
    caus = test_causality(d_test, clf_model, feat_scaler, rng)
    aux = test_auxiliary_sensitivity(d_test, aux_features, clf_model, feat_scaler, rng)
    rec = test_recovery(artifacts, clf_model, feat_scaler, rng)
    pers = test_persistent_failure(artifacts, clf_model, feat_scaler, rng)

    mono["detail"].to_csv(OUT_DIR / "detalhe_monotonicidade.csv", index=False)
    ctx["detail"].to_csv(OUT_DIR / "detalhe_coerencia_fisica.csv", index=False)
    caus["detail"].to_csv(OUT_DIR / "detalhe_causalidade.csv", index=False)
    aux["detail"].to_csv(OUT_DIR / "detalhe_sensibilidade_auxiliar.csv", index=False)
    rec["detail"].to_csv(OUT_DIR / "detalhe_recuperacao.csv", index=False)
    rec["summary"].to_csv(OUT_DIR / "resumo_recuperacao.csv", index=False)
    pers["detail"].to_csv(OUT_DIR / "detalhe_falha_persistente.csv", index=False)
    pers["summary"].to_csv(OUT_DIR / "resumo_falha_persistente.csv", index=False)

    results = {"monotonicity": mono, "context": ctx, "causality": caus, "aux_sensitivity": aux, "recovery": rec, "persistent": pers}
    write_report(results)

    print("Resultados em:", OUT_DIR)
    print(f"Monotonicidade: {mono['monotonicity_rate']:.3f} (meta >=0.90: {mono['meets_target_0_90']})")
    print(f"Coerencia fisica: {ctx['coherence_rate']:.3f}")
    print(f"Violacao de causalidade: {caus['causality_violation_rate']:.3f}")
    print(f"Flip para BAD por ruido auxiliar: {aux['flip_to_bad_rate']:.4f}")
    print(rec["summary"].to_string(index=False))
    print(pers["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
