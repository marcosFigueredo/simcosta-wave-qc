from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "dixon_4sigma"

VARIABLES = [
    "Hsig", "Tp", "Hmax", "HM0",
    "Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2",
    "Avg_Turb", "Avg_Chl", "Avg_CDOM",
]


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, comment="/", na_values=["NULL", "null", "", "NaN"])
    df["Timestamp"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = df.drop_duplicates("Timestamp").set_index("Timestamp")
    for col in VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify(z: pd.Series) -> pd.Series:
    abs_z = z.abs()
    return np.select(
        [abs_z <= 3, (abs_z > 3) & (abs_z <= 4), abs_z > 4],
        ["GOOD", "SUSPECT", "BAD"],
        default="MISSING",
    )


def consecutive_bad_events(result: pd.DataFrame) -> pd.DataFrame:
    bad = result["Flag"].eq("BAD")
    if not bad.any():
        return pd.DataFrame(columns=["start", "end", "n_points", "min_value", "max_value", "max_abs_z", "event_type"])

    event_id = (bad.ne(bad.shift(fill_value=False))).cumsum()
    rows = []
    for _, group in result[bad].groupby(event_id[bad]):
        n = len(group)
        rows.append(
            {
                "start": group.index.min(),
                "end": group.index.max(),
                "n_points": n,
                "min_value": group["Valor"].min(),
                "max_value": group["Valor"].max(),
                "max_abs_z": group["z-score"].abs().max(),
                "event_type": "possible_physical_event" if n >= 2 else "possible_instrumental_spike",
            }
        )
    return pd.DataFrame(rows)


def interpretation_from_bad_pct(bad_pct: float) -> str:
    if bad_pct < 0.1:
        return "Excelente qualidade dos dados."
    if bad_pct < 1:
        return "Boa qualidade com poucos outliers."
    if bad_pct < 5:
        return "Serie requer investigacao."
    return "Possivel problema instrumental ou processamento inadequado."


def run_dixon(df: pd.DataFrame, var: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[var].dropna()
    mu = x.mean()
    sigma = x.std(ddof=1)
    z = (x - mu) / sigma if sigma and not np.isnan(sigma) else pd.Series(np.nan, index=x.index)

    result = pd.DataFrame({"Valor": x, "z-score": z, "Flag": classify(z)}, index=x.index)
    result.index.name = "Timestamp"

    total = len(result)
    good = int(result["Flag"].eq("GOOD").sum())
    suspect = int(result["Flag"].eq("SUSPECT").sum())
    bad = int(result["Flag"].eq("BAD").sum())
    bad_pct = 100 * bad / total if total else np.nan

    summary = {
        "Variavel": var,
        "Número total de observações": total,
        "Média (μ)": mu,
        "Desvio padrão (σ)": sigma,
        "Quantidade de GOOD": good,
        "Quantidade de SUSPECT": suspect,
        "Quantidade de BAD": bad,
        "Percentual de observações BAD": bad_pct,
        "Valor mínimo observado": x.min(),
        "Valor máximo observado": x.max(),
        "Limite inferior Dixon 4σ": mu - 4 * sigma,
        "Limite superior Dixon 4σ": mu + 4 * sigma,
        "Interpretação": interpretation_from_bad_pct(bad_pct),
    }

    top_positive = result.sort_values("z-score", ascending=False).head(10)
    top_negative = result.sort_values("z-score", ascending=True).head(10)
    events = consecutive_bad_events(result)
    return result, summary, top_positive, top_negative, events


def write_markdown(summaries: pd.DataFrame, events_all: pd.DataFrame) -> None:
    lines = [
        "# Dixon 4σ - SIMCOSTA_BA-1",
        "",
        "## Metodologia",
        "",
        "Para cada variavel foi calculada a media global `μ` e o desvio padrao amostral `σ`. Em seguida, cada observacao foi padronizada por `z_i = (x_i - μ) / σ`.",
        "",
        "Classificacao usada:",
        "",
        "- GOOD: `|z_i| <= 3`",
        "- SUSPECT: `3 < |z_i| <= 4`",
        "- BAD: `|z_i| > 4`",
        "",
        "O Dixon 4σ e simples e transparente, mas deve ser interpretado com cautela em series oceanograficas costeiras, porque a distribuicao pode ser assimetrica, sazonal e influenciada por eventos fisicos reais.",
        "",
        "## Resumo estatistico",
        "",
        summaries.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretacao geral",
        "",
    ]

    for _, row in summaries.iterrows():
        lines.append(
            f"- `{row['Variavel']}`: {row['Percentual de observações BAD']:.4f}% BAD. {row['Interpretação']}"
        )

    lines.extend(
        [
            "",
            "## Eventos consecutivos BAD",
            "",
            "Eventos com `n_points >= 2` sao candidatos a eventos fisicos reais ou periodos persistentes de anomalia. Eventos com `n_points = 1` sao candidatos mais fortes a spikes instrumentais isolados, especialmente se nao forem confirmados por outras variaveis.",
            "",
            events_all.head(50).to_markdown(index=False, floatfmt=".4f") if not events_all.empty else "Nenhum evento BAD encontrado.",
            "",
            "## Conclusao cientifica resumida",
            "",
            "O Dixon 4σ fornece uma primeira avaliacao global da compatibilidade das series com sua variabilidade estatistica central. Percentuais baixos de BAD indicam que os extremos sao raros em relacao a media e ao desvio padrao da serie; percentuais elevados indicam necessidade de investigacao, podendo refletir assimetria natural, eventos costeiros extremos, falhas instrumentais ou processamento inadequado. A classificacao final deve ser combinada com Spike test, range fisico, rate-of-change, lacunas temporais e coerencia multivariada.",
            "",
            "## Arquivos gerados",
            "",
            "- `dixon_4sigma_resumo.csv`",
            "- `dixon_4sigma_eventos_bad.csv`",
            "- `dixon_4sigma_top_positivos.csv`",
            "- `dixon_4sigma_top_negativos.csv`",
            "- `dixon_4sigma_<variavel>.csv` para a tabela Timestamp/Valor/z-score/Flag de cada variavel.",
        ]
    )
    (OUT_DIR / "dixon_4sigma_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    summaries = []
    events = []
    positives = []
    negatives = []

    for var in [v for v in VARIABLES if v in df.columns]:
        result, summary, top_pos, top_neg, ev = run_dixon(df, var)
        result.reset_index().to_csv(OUT_DIR / f"dixon_4sigma_{var}.csv", index=False)
        summaries.append(summary)

        top_pos = top_pos.reset_index()
        top_pos.insert(0, "Variavel", var)
        positives.append(top_pos)

        top_neg = top_neg.reset_index()
        top_neg.insert(0, "Variavel", var)
        negatives.append(top_neg)

        if not ev.empty:
            ev.insert(0, "Variavel", var)
            events.append(ev)

    summaries_df = pd.DataFrame(summaries)
    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    positives_df = pd.concat(positives, ignore_index=True)
    negatives_df = pd.concat(negatives, ignore_index=True)

    summaries_df.to_csv(OUT_DIR / "dixon_4sigma_resumo.csv", index=False)
    events_df.to_csv(OUT_DIR / "dixon_4sigma_eventos_bad.csv", index=False)
    positives_df.to_csv(OUT_DIR / "dixon_4sigma_top_positivos.csv", index=False)
    negatives_df.to_csv(OUT_DIR / "dixon_4sigma_top_negativos.csv", index=False)
    write_markdown(summaries_df, events_df)
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
