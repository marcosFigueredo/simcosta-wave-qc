from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ajustes_metricas import streaming_operational_metrics

BASE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qc19", BASE_DIR / "codigos" / "19_simulador_tempo_real.py")
qc19 = importlib.util.module_from_spec(_spec)
sys.modules["qc19"] = qc19
_spec.loader.exec_module(qc19)
qc16 = qc19.qc16
qc10 = qc19.qc10

OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "qc_lstm_geral_multiboia" / "simulacao_tempo_real" / "estabilidade"

# sementes do simulador: cada uma sorteia uma sequencia diferente de
# episodios de anomalia (quando comecam, de qual familia, gravidade e
# duracao) sobre a MESMA serie real da BA-1 e o MESMO modelo ja treinado e
# salvo em disco (nao ha retreino aqui, so variacao do fluxo simulado).
SIM_SEEDS = [2026, 7, 123, 501, 909, 3141, 8, 55, 2024, 77]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando modelo geral salvo (mesmo modelo em todas as replicas)...")
    artifacts = qc19.load_model_artifacts()

    print("Lendo serie real da BA-1...")
    path = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
    hourly, _ = qc16.read_buoy_target_hourly(path)
    hourly = hourly.tail(qc10.MAX_HOURLY_RECORDS)

    all_onset_rows = []
    geral_rows = []
    operational_rows = []
    for i, sim_seed in enumerate(SIM_SEEDS, start=1):
        print(f"\n=== Simulacao {i}/{len(SIM_SEEDS)}: sim_seed={sim_seed} ===")
        log = qc19.run_simulation(hourly, artifacts, sim_seed=sim_seed)
        log.to_csv(OUT_DIR / f"log_simulacao_seed_{sim_seed}.csv", index=False)
        operational_rows.append({"sim_seed": sim_seed, **streaming_operational_metrics(log)})
        onset_df, taxa_geral_t1 = qc19.compute_onset_detection(log)
        onset_df["sim_seed"] = sim_seed
        all_onset_rows.append(onset_df)
        n_eventos_distintos = int(onset_df["n_eventos"].sum())
        geral_rows.append({"sim_seed": sim_seed, "taxa_deteccao_t1_geral": taxa_geral_t1, "n_eventos_distintos": n_eventos_distintos})
        print(f"  -> taxa geral t+1 = {taxa_geral_t1:.3f} ({n_eventos_distintos} eventos distintos)")
        print(onset_df.to_string(index=False))

    df_onset_all = pd.concat(all_onset_rows, ignore_index=True)
    df_onset_all.to_csv(OUT_DIR / "deteccao_t1_todas_sementes.csv", index=False)

    df_geral = pd.DataFrame(geral_rows)
    df_geral.to_csv(OUT_DIR / "taxa_geral_por_semente.csv", index=False)
    pd.DataFrame(operational_rows).to_csv(OUT_DIR / "metricas_operacionais_por_semente.csv", index=False)

    # Agregacao ponderada por eventos: cada familia pode ter poucos eventos
    # numa dada semente, entao juntamos os eventos de TODAS as sementes por
    # familia antes de calcular a taxa (equivale a rodar uma simulacao ~10x
    # mais longa), alem de reportar a media/desvio das taxas por-semente.
    rows_agg = []
    for fam, g in df_onset_all.groupby("familia_letra"):
        n_total = int(g["n_eventos"].sum())
        taxa_pond = float((g["taxa_deteccao_t1"] * g["n_eventos"]).sum() / n_total) if n_total else float("nan")
        taxas_por_semente = g["taxa_deteccao_t1"].to_numpy()
        rng = np.random.default_rng(2026)
        boot = np.array([rng.choice(taxas_por_semente, size=len(taxas_por_semente), replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5]) if len(taxas_por_semente) > 1 else (float("nan"), float("nan"))
        rows_agg.append({
            "familia": fam, "n_sementes": len(g), "n_eventos_total": n_total,
            "taxa_deteccao_t1_ponderada": taxa_pond,
            "media_das_taxas_por_semente": taxas_por_semente.mean(),
            "desvio_padrao_entre_sementes": taxas_por_semente.std(ddof=1) if len(taxas_por_semente) > 1 else 0.0,
            "ic95_inferior": lo, "ic95_superior": hi,
        })
    df_agg = pd.DataFrame(rows_agg).sort_values("familia")
    df_agg.to_csv(OUT_DIR / "deteccao_t1_agregada_por_familia.csv", index=False)

    taxa_geral_media = df_geral["taxa_deteccao_t1_geral"].mean()
    taxa_geral_std = df_geral["taxa_deteccao_t1_geral"].std(ddof=1)

    lines = [
        "# Estabilidade do simulador em tempo real, multiplas sementes",
        "",
        f"Mesmo modelo geral ja treinado e salvo (sem retreino), mesma serie real da BA-1, "
        f"{len(SIM_SEEDS)} sementes de simulador diferentes (cada uma sorteia uma sequencia distinta",
        "de episodios de anomalia, quando comecam, familia, gravidade e duracao). Metrica principal,",
        "deteccao no instante t+1 (inicio de cada episodio).",
        "",
        f"Taxa geral t+1, media {taxa_geral_media:.3f}, desvio padrao entre sementes {taxa_geral_std:.3f} "
        f"(n={len(SIM_SEEDS)} sementes).",
        "",
        "## Por familia (eventos de todas as sementes agrupados, IC 95% via bootstrap sobre as taxas por-semente)",
        "",
        df_agg.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Taxa geral por semente",
        "",
        df_geral.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUT_DIR / "estabilidade_simulador_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nResultados em:", OUT_DIR)
    print(df_agg.to_string(index=False))


if __name__ == "__main__":
    main()
