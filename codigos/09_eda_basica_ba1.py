from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "eda_basica"
FIG_DIR = OUT_DIR / "figures"

VARIABLES = [
    "Hsig", "Tp", "Hmax", "HM0",
    "Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2",
    "Avg_Turb", "Avg_Chl", "Avg_CDOM",
]

UNITS = {
    "Hsig": "m", "Tp": "s", "Hmax": "m", "HM0": "m",
    "Avg_Sal": "psu", "Avg_DO": "ml/L", "Avg_W_Tmp1": "C", "Avg_W_Tmp2": "C",
    "Avg_Turb": "NTU", "Avg_Chl": "ug/L", "Avg_CDOM": "ppb",
}


def read_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, comment="/", na_values=["NULL", "null", "", "NaN"])
    df["datetime_utc"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc").set_index("datetime_utc")
    for col in VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_raw()
    n_raw = len(raw)
    t0, t1 = raw.index.min(), raw.index.max()

    hourly = raw[VARIABLES].resample("1h").median()
    n_hours_possible = len(pd.date_range(t0.floor("h"), t1.ceil("h"), freq="1h"))

    rows = []
    for col in VARIABLES:
        s_raw = raw[col]
        s_hourly = hourly[col]
        rows.append(
            {
                "variavel": col,
                "unidade": UNITS[col],
                "n_valido_bruto": int(s_raw.notna().sum()),
                "cobertura_bruta_pct": round(100 * s_raw.notna().sum() / n_raw, 2),
                "cobertura_horaria_pct": round(100 * s_hourly.notna().sum() / n_hours_possible, 2),
                "media": round(float(s_raw.mean()), 3),
                "desvio_padrao": round(float(s_raw.std()), 3),
                "minimo": round(float(s_raw.min()), 3),
                "maximo": round(float(s_raw.max()), 3),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "estatisticas_descritivas_ba1.csv", index=False)

    coverage_by_year = hourly[VARIABLES].notna().groupby(hourly.index.year).mean() * 100
    coverage_by_year.to_csv(OUT_DIR / "cobertura_horaria_por_ano.csv")

    # Cobertura por ano restrita a variavel-alvo (Hsig). Um heatmap com as 11
    # variaveis lado a lado nao ajuda o leitor aqui, o estudo estatistico e
    # focado em Hsig, entao a cobertura relevante para contextualizar gaps e
    # o split treino/teste da LSTM tambem e a de Hsig.
    hsig_coverage = coverage_by_year["Hsig"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(hsig_coverage.index.astype(str), hsig_coverage.to_numpy(), color="#2f5f8a")
    for bar, value in zip(bars, hsig_coverage.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Cobertura horaria de Hsig (%)")
    ax.set_xlabel("Ano")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_eda_cobertura_por_ano.png", dpi=200)
    plt.close(fig)

    # Serie temporal principal do estudo (Hsig, o alvo de previsao da LSTM e
    # a variavel usada no experimento com anomalia sintetica), horaria, com
    # lacunas visiveis como cortes na linha (sem interpolacao aqui, e a serie
    # bruta reamostrada, nao o dado ja preenchido usado no pipeline de IA).
    # A serie horaria bruta e reamostrada ate o fim do arquivo (25/07/2026),
    # mas Hsig nao tem nenhum registro valido em 2026 (interrupcao do ADCP,
    # ver cobertura por ano). Cortar a cauda sem dado evita um trecho vazio
    # e um eixo x esticado sem informacao no final do grafico.
    hsig_hourly_full = hourly["Hsig"]
    last_valid = hsig_hourly_full.last_valid_index()
    hsig_hourly = hsig_hourly_full.loc[:last_valid]
    hsig_mean = float(hsig_hourly.mean())
    hsig_std = float(hsig_hourly.std())
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(hsig_hourly.index, hsig_hourly.to_numpy(), color="#2f5f8a", linewidth=0.5)
    ax.axhline(hsig_mean, color="#c9711f", linewidth=1.0, linestyle="--", label=f"media ({hsig_mean:.2f} m)")
    ax.fill_between(
        hsig_hourly.index,
        hsig_mean - hsig_std,
        hsig_mean + hsig_std,
        color="#c9711f",
        alpha=0.12,
        label="media +/- desvio padrao",
    )
    ax.set_xlim(hsig_hourly.index.min(), hsig_hourly.index.max())
    ax.set_ylabel("Hsig (m)")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_eda_serie_hsig.png", dpi=200)
    plt.close(fig)

    print(f"N registros brutos = {n_raw}, periodo = {t0.date()} a {t1.date()}")
    print(f"N horas possiveis no periodo = {n_hours_possible}")
    print(summary.to_string(index=False))
    print(f"Outputs salvos em {OUT_DIR}")


if __name__ == "__main__":
    main()
