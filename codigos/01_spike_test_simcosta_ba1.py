from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "spike_test_formula_principal"
FIG_DIR = OUT_DIR / "figures"

MAIN_VARIABLES = [
    "Hsig", "Tp", "Hmax", "HM0",
    "Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2",
    "Avg_Turb", "Avg_Chl", "Avg_CDOM",
]

RANGE_LIMITS = {
    "Avg_W_Tmp1": (15, 35), "Avg_W_Tmp2": (15, 35),
    "Avg_Sal": (0, 42), "Avg_DO": (0, 15),
    "Avg_Turb": (0, 2000), "Avg_Chl": (0, 200), "Avg_CDOM": (0, 500),
    "Hsig": (0, 8), "Hmax": (0, 15), "HM0": (0, 8), "Tp": (0, 30),
}

ROC_LIMITS_PER_HOUR = {
    "Avg_W_Tmp1": 1.5, "Avg_W_Tmp2": 1.5,
    "Avg_Sal": 3.0, "Avg_DO": 3.0,
    "Avg_Turb": 300.0, "Avg_Chl": 20.0, "Avg_CDOM": 80.0,
    "Hsig": 1.5, "Hmax": 3.0, "HM0": 1.5, "Tp": 8.0,
}


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, comment="/", na_values=["NULL", "null", "", "NaN"])
    df["datetime_utc"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc").set_index("datetime_utc")
    for col in MAIN_VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "verao"
    if month in (3, 4, 5):
        return "outono"
    if month in (6, 7, 8):
        return "inverno"
    return "primavera"


def safe_mad(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window, center=True, min_periods=max(7, window // 4)).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=max(7, window // 4)).median()
    return 1.4826 * mad.replace(0, np.nan)


def classify_qi(qi: pd.Series) -> pd.Series:
    return pd.cut(
        qi,
        bins=[-np.inf, 50, 75, 90, np.inf],
        labels=["Bad", "Suspect", "Good", "Excellent"],
    ).astype("string")


def qc_variable(df: pd.DataFrame, var: str) -> pd.DataFrame:
    x = df[var]
    dt_hours = df.index.to_series().diff().dt.total_seconds().div(3600).median()
    dt_hours = float(dt_hours) if pd.notna(dt_hours) and dt_hours > 0 else 0.5

    classic = (x - (x.shift(1) + x.shift(-1)) / 2).abs()
    mad_scale = safe_mad(classic, 49)
    mad_score = classic / mad_scale
    second_diff = (x.shift(-1) - 2 * x + x.shift(1)).abs()
    second_score = second_diff / safe_mad(second_diff, 49)
    roc = x.diff().abs() / dt_hours
    local_mean = classic.rolling(49, center=True, min_periods=12).mean()
    local_std = classic.rolling(49, center=True, min_periods=12).std().replace(0, np.nan)
    local_std_score = (classic - local_mean).abs() / local_std

    p995 = classic.dropna().quantile(0.995)
    p99_month = classic.groupby(df.index.month).transform(lambda s: s.dropna().quantile(0.99))
    second_p995 = second_diff.dropna().quantile(0.995)

    lo, hi = RANGE_LIMITS.get(var, (x.dropna().quantile(0.001), x.dropna().quantile(0.999)))
    roc_limit = ROC_LIMITS_PER_HOUR.get(var, roc.dropna().quantile(0.995))

    flag_classic = classic > p995
    flag_mad = mad_score > 6
    flag_second = second_diff > second_p995
    flag_qartod = mad_score > 8
    flag_roc = roc > roc_limit
    flag_local_std = local_std_score > 4
    flag_range = (x < lo) | (x > hi)

    score_components = pd.concat(
        [
            (classic / p995).rename("classic"),
            (mad_score / 8).rename("mad"),
            (second_diff / second_p995).rename("second_diff"),
            (roc / roc_limit).rename("roc"),
            (local_std_score / 4).rename("local_std"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    composite_score = score_components.max(axis=1).clip(lower=0)
    score_max = composite_score.dropna().quantile(0.999)
    if pd.isna(score_max) or score_max == 0:
        score_max = 1.0
    qi = (100 * (1 - composite_score / score_max)).clip(0, 100)

    flags = pd.concat(
        [
            flag_classic.rename("classic"),
            flag_mad.rename("mad"),
            flag_second.rename("second_diff"),
            flag_qartod.rename("qartod"),
            flag_roc.rename("roc"),
            flag_local_std.rename("local_std"),
            flag_range.rename("range"),
        ],
        axis=1,
    ).fillna(False)

    out = pd.DataFrame(index=df.index)
    out["variable"] = var
    out["value"] = x
    out["classic_score"] = classic
    out["mad_score"] = mad_score
    out["second_difference_score"] = second_diff
    out["roc_per_hour"] = roc
    out["local_std_score"] = local_std_score
    out["flag_classic_spike"] = flags["classic"]
    out["flag_mad"] = flags["mad"]
    out["flag_second_difference"] = flags["second_diff"]
    out["flag_qartod_spike"] = flags["qartod"]
    out["flag_roc"] = flags["roc"]
    out["flag_local_std"] = flags["local_std"]
    out["flag_range"] = flags["range"]
    out["flag_count"] = flags.sum(axis=1)
    out["removed"] = out["flag_range"] | (out["flag_count"] >= 3)
    out["suspect"] = out["flag_count"] >= 1
    out["quality_index"] = qi
    out["quality_class"] = classify_qi(qi)
    out["classic_threshold_p995"] = p995
    out["classic_threshold_p99_month"] = p99_month
    out["second_diff_threshold_p995"] = second_p995
    out["roc_limit_per_hour"] = roc_limit
    out["range_min"] = lo
    out["range_max"] = hi
    return out


def summarize(flags_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    perf = []
    tests = {
        "Range": "flag_range",
        "Spike classico": "flag_classic_spike",
        "MAD": "flag_mad",
        "Segunda diferenca": "flag_second_difference",
        "QARTOD spike": "flag_qartod_spike",
        "Rate of Change": "flag_roc",
        "Desvio-padrao local": "flag_local_std",
    }
    for var, g in flags_long.groupby("variable"):
        valid = g["value"].notna()
        n = int(valid.sum())
        rows.append({
            "Variavel": var,
            "N": n,
            "Removed (%)": 100 * g.loc[valid, "removed"].sum() / n if n else np.nan,
            "Spike (%)": 100 * g.loc[valid, "flag_classic_spike"].sum() / n if n else np.nan,
            "MAD (%)": 100 * g.loc[valid, "flag_mad"].sum() / n if n else np.nan,
            "SecondDiff (%)": 100 * g.loc[valid, "flag_second_difference"].sum() / n if n else np.nan,
            "QARTOD (%)": 100 * g.loc[valid, "flag_qartod_spike"].sum() / n if n else np.nan,
            "Range (%)": 100 * g.loc[valid, "flag_range"].sum() / n if n else np.nan,
            "ROC (%)": 100 * g.loc[valid, "flag_roc"].sum() / n if n else np.nan,
            "LocalSTD (%)": 100 * g.loc[valid, "flag_local_std"].sum() / n if n else np.nan,
            "QI median": g.loc[valid, "quality_index"].median(),
        })
    bool_cols = [v for v in tests.values()]
    all_flags = flags_long[bool_cols].fillna(False)
    for name, col in tests.items():
        flag = all_flags[col]
        perf.append({
            "Teste": name,
            "Flags unicas": int((flag & (all_flags.sum(axis=1) == 1)).sum()),
            "Flags compartilhadas": int((flag & (all_flags.sum(axis=1) > 1)).sum()),
            "Flags totais": int(flag.sum()),
        })

    annual = flags_long.assign(year=flags_long.index.year).groupby(["variable", "year"]).agg(
        N=("value", "count"), removed=("removed", "sum"), suspect=("suspect", "sum"), qi_median=("quality_index", "median")
    ).reset_index()
    annual["removed_pct"] = 100 * annual["removed"] / annual["N"]
    annual["suspect_pct"] = 100 * annual["suspect"] / annual["N"]

    seasonal = flags_long.assign(season=[season(m) for m in flags_long.index.month]).groupby(["variable", "season"]).agg(
        N=("value", "count"), removed=("removed", "sum"), suspect=("suspect", "sum"), qi_median=("quality_index", "median")
    ).reset_index()
    seasonal["removed_pct"] = 100 * seasonal["removed"] / seasonal["N"]
    seasonal["suspect_pct"] = 100 * seasonal["suspect"] / seasonal["N"]
    return pd.DataFrame(rows), pd.DataFrame(perf), annual, seasonal


def make_figures(df: pd.DataFrame, flags_long: pd.DataFrame, article: pd.DataFrame, perf: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Figure 1: QC flow survival
    total = flags_long["value"].notna().sum()
    steps = [
        ("Raw valid", total),
        ("Range pass", total - flags_long["flag_range"].sum()),
        ("Classic pass", total - flags_long["flag_classic_spike"].sum()),
        ("MAD pass", total - flags_long["flag_mad"].sum()),
        ("ROC pass", total - flags_long["flag_roc"].sum()),
        ("Final retained", total - flags_long["removed"].sum()),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([s[0] for s in steps], [100 * s[1] / total for s in steps], marker="o")
    ax.set_ylim(0, 101)
    ax.set_ylabel("Sobrevivencia dos dados (%)")
    ax.set_title("Figura 1 - Fluxograma de Controle de Qualidade")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_fluxograma_qc_sobrevivencia.png", dpi=180)
    plt.close(fig)

    season_bands = [
        ("verao", "2000-01-01", "2000-03-01", "#FEE8C8"),
        ("outono", "2000-03-01", "2000-06-01", "#E5F5E0"),
        ("inverno", "2000-06-01", "2000-09-01", "#DEEBF7"),
        ("primavera", "2000-09-01", "2000-12-01", "#F2E5FF"),
        ("verao", "2000-12-01", "2001-01-01", "#FEE8C8"),
    ]
    for var in MAIN_VARIABLES:
        g = flags_long[flags_long["variable"] == var]
        if g.empty:
            continue
        # Figure 2: full series flags
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(g.index, g["value"], color="0.35", lw=0.5)
        mask = g["suspect"].fillna(False)
        ax.scatter(g.index[mask], g.loc[mask, "value"], s=8, color="crimson")
        ax.set_title(f"Figura 2 - Serie temporal com flags - {var}")
        ax.set_ylabel(var)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig02_serie_flags_{var}.png", dpi=180)
        plt.close(fig)

        # Annual stack, Jan-Dec
        years = sorted(g.index.year.unique())
        fig, axes = plt.subplots(len(years), 1, figsize=(14, max(5, 2.0 * len(years))), sharex=True, squeeze=False)
        for ax, year in zip(axes.ravel(), years):
            gy = g[g.index.year == year]
            xplot = pd.to_datetime({"year": 2000, "month": gy.index.month, "day": gy.index.day,
                                    "hour": gy.index.hour, "minute": gy.index.minute, "second": gy.index.second})
            for _, start, end, color in season_bands:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.55, lw=0)
            ax.plot(xplot, gy["value"], color="0.35", lw=0.45)
            m = gy["suspect"].fillna(False).to_numpy()
            ax.scatter(xplot[m], gy.loc[m, "value"], s=8, color="crimson")
            ax.set_ylabel(str(year), rotation=0, labelpad=25)
            ax.grid(axis="y", alpha=0.2)
        axes.ravel()[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes.ravel()[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes.ravel()[-1].set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"))
        fig.suptitle(f"Comparacao anual Jan-Dez com estacoes - {var}")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(FIG_DIR / f"fig02_anual_empilhado_{var}.png", dpi=180)
        plt.close(fig)

        # Figure 3: MAD score distribution
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vals = g["mad_score"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals.clip(upper=30), bins=80, color="#4C78A8")
        ax.axvline(6, color="crimson", lw=1.4, label="MAD > 6")
        ax.axvline(8, color="black", lw=1.2, ls="--", label="QARTOD > 8")
        ax.set_title(f"Figura 3 - Distribuicao dos Scores MAD - {var}")
        ax.set_xlabel("Score MAD")
        ax.set_ylabel("Frequencia")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig03_mad_scores_{var}.png", dpi=180)
        plt.close(fig)

        # Scientific scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        sample = g[["value", "mad_score", "suspect"]].dropna()
        if len(sample) > 15000:
            sample = sample.sample(15000, random_state=42)
        ax.scatter(sample["value"], sample["mad_score"].clip(upper=30), s=4, alpha=0.25, c=np.where(sample["suspect"], "crimson", "0.35"))
        ax.axhline(6, color="crimson", lw=1)
        ax.set_title(f"Dispersao valor x Score MAD - {var}")
        ax.set_xlabel("x_i")
        ax.set_ylabel("Score MAD")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_scatter_xi_madscore_{var}.png", dpi=180)
        plt.close(fig)

    # Figure 4: heatmap QC by month/year
    heat = flags_long.assign(year=flags_long.index.year, month=flags_long.index.month).groupby(["year", "month"])["removed"].mean().mul(100).unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heat.fillna(0).values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Figura 4 - Heatmap de QC: removidos (%) por mes/ano")
    fig.colorbar(im, ax=ax, label="Removed (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_heatmap_qc_month_year.png", dpi=180)
    plt.close(fig)


def write_md(article: pd.DataFrame, perf: pd.DataFrame, annual: pd.DataFrame, seasonal: pd.DataFrame) -> None:
    lines = [
        "# Spike Test e QC Robusto - SIMCOSTA_BA-1",
        "",
        "## Objetivo",
        "",
        "Avaliar valores isolados incompatíveis com a evolução temporal dos dados da boia SIMCOSTA_BA-1. A formula principal usada foi `S_i = |x_i - (x_{i-1}+x_{i+1})/2|`. O teste classico responde: existe um valor isolado incompatível com os vizinhos?",
        "",
        "A rotina tambem aplica testes em ordem crescente/complementar de robustez: Mediana + MAD, segunda diferenca, QARTOD spike, gradiente fisico maximo, range test e desvio-padrao local. A decisao final nao remove automaticamente todo alerta: `removed` e verdadeiro quando o valor viola range fisico ou quando tres ou mais testes concordam.",
        "",
        "## Parametros",
        "",
        "- Janela robusta MAD: 49 registros, aproximadamente 24,5 h para intervalo de 30 min.",
        "- Spike classico: limiar regional P99.5 de `S_i` por variavel.",
        "- MAD: flag quando `Score_MAD > 6`.",
        "- QARTOD spike: flag mais severa quando `Score_MAD > 8`.",
        "- Segunda diferenca: limiar P99.5 da curvatura temporal por variavel.",
        "- Gradiente fisico maximo: limites por hora definidos por variavel no script.",
        "- Range: limites fisicos amplos definidos por variavel no script.",
        "- Desvio-padrao local: flag quando o score local excede 4 desvios-padrao.",
        "- QI: `QI = 100 * (1 - Score/Scoremax)`, com classes Excellent, Good, Suspect e Bad.",
        "",
        "## Tabela principal do artigo",
        "",
        article.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Tabela de performance dos testes",
        "",
        perf.to_markdown(index=False),
        "",
        "## Interpretacao das saidas",
        "",
        "A Figura 1 mostra a sobrevivencia dos dados ao longo do fluxo de QC. Quedas pequenas indicam que a serie e majoritariamente consistente; quedas concentradas em um teste indicam o tipo dominante de problema.",
        "",
        "As Figuras 2 mostram a serie temporal com flags para cada variavel principal. Pontos isolados em vermelho sugerem ruido eletronico, erro de transmissao ou medicao incompatível com a dinamica local. Blocos coerentes em mais de uma variavel podem representar evento oceanografico real.",
        "",
        "As figuras anuais empilhadas colocam cada ano em uma linha, de janeiro a dezembro, com cores de fundo por estacao. Elas servem para comparar recorrencia sazonal, anos problematicos e trechos de possivel instabilidade instrumental.",
        "",
        "A Figura 3 mostra a distribuicao dos scores MAD. Caudas longas indicam presenca de valores extremos; concentracao abaixo do limiar sugere boa estabilidade local.",
        "",
        "A Figura 4 e um heatmap de QC por mes e ano. Padroes verticais sugerem meses/estacoes recorrentes; padroes horizontais sugerem anos com maior problema de qualidade.",
        "",
        "Os diagramas `fig_scatter_xi_madscore_*.png` mostram valor observado versus Score MAD. Essa figura e forte para publicacao porque separa visualmente valores extremos reais de valores com anomalia local.",
        "",
        "## Interpretacao por tipo de problema",
        "",
        "- Ruido eletronico: spike isolado, sem recorrencia sazonal e sem concordancia com outros testes.",
        "- Falha do sensor: concentracao em uma unica variavel, em periodo continuo ou em um ano especifico.",
        "- Erro de transmissao: flags proximas a lacunas ou simultaneas em variaveis sem relacao fisica direta.",
        "- Valor fisicamente impossivel: flag de `Range`; deve ser removido ou tratado como `bad`.",
        "- Medicao isolada incompatível: falha no spike classico/MAD sem suporte das variaveis relacionadas.",
        "- Evento oceanografico real: flags simultaneas e coerentes entre onda, turbidez, salinidade, temperatura ou variaveis opticas.",
        "",
        "## Comparacao anual",
        "",
        annual.sort_values("removed_pct", ascending=False).head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Comparacao por estacao",
        "",
        seasonal.sort_values("removed_pct", ascending=False).head(20).to_markdown(index=False, floatfmt=".4f"),
    ]
    (OUT_DIR / "spike_qc_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    vars_available = [v for v in MAIN_VARIABLES if v in df.columns]
    flags = pd.concat([qc_variable(df, v) for v in vars_available])
    article, perf, annual, seasonal = summarize(flags)
    flags.to_csv(OUT_DIR / "spike_qc_flags_long.csv")
    article.to_csv(OUT_DIR / "tabela_principal_artigo.csv", index=False)
    perf.to_csv(OUT_DIR / "tabela_performance_testes.csv", index=False)
    annual.to_csv(OUT_DIR / "spike_qc_por_ano.csv", index=False)
    seasonal.to_csv(OUT_DIR / "spike_qc_por_estacao.csv", index=False)
    make_figures(df, flags, article, perf)
    write_md(article, perf, annual, seasonal)
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
