from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "q_dixon"

VARIABLES = [
    "Hsig", "Tp", "Hmax", "HM0",
    "Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2",
    "Avg_Turb", "Avg_Chl", "Avg_CDOM",
]

WINDOW_SIZE = 11
Q_SUSPECT = 0.412
Q_BAD = 0.517


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


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "verao"
    if month in (3, 4, 5):
        return "outono"
    if month in (6, 7, 8):
        return "inverno"
    return "primavera"


def q_dixon_center(window: np.ndarray) -> float:
    vals = window[~np.isnan(window)]
    if len(vals) < 5:
        return np.nan
    center = window[len(window) // 2]
    if np.isnan(center):
        return np.nan
    sorted_vals = np.sort(vals)
    data_range = sorted_vals[-1] - sorted_vals[0]
    if data_range == 0:
        return 0.0

    if center == sorted_vals[-1]:
        gap = sorted_vals[-1] - sorted_vals[-2]
        return gap / data_range
    if center == sorted_vals[0]:
        gap = sorted_vals[1] - sorted_vals[0]
        return gap / data_range
    return 0.0


def classify(q: pd.Series) -> pd.Series:
    return np.select(
        [q < Q_SUSPECT, (q >= Q_SUSPECT) & (q <= Q_BAD), q > Q_BAD],
        ["GOOD", "SUSPECT", "BAD"],
        default="MISSING",
    )


def consecutive_bad_events(result: pd.DataFrame) -> pd.DataFrame:
    bad = result["Flag"].eq("BAD")
    if not bad.any():
        return pd.DataFrame(columns=["start", "end", "n_points", "min_value", "max_value", "max_q", "event_type"])
    event_id = bad.ne(bad.shift(fill_value=False)).cumsum()
    rows = []
    for _, group in result[bad].groupby(event_id[bad]):
        n = len(group)
        rows.append({
            "start": group.index.min(),
            "end": group.index.max(),
            "n_points": n,
            "min_value": group["Valor"].min(),
            "max_value": group["Valor"].max(),
            "max_q": group["Q-Dixon"].max(),
            "event_type": "possible_physical_event" if n >= 2 else "possible_instrumental_spike",
        })
    return pd.DataFrame(rows)


def interpretation_from_bad_pct(bad_pct: float) -> str:
    if bad_pct < 0.1:
        return "Excelente qualidade; poucos extremos isolados."
    if bad_pct < 1:
        return "Boa qualidade com poucos extremos isolados."
    if bad_pct < 5:
        return "Serie requer investigacao local."
    return "Possivel problema instrumental, processamento inadequado ou limiar muito sensivel para esta variavel."


def run_q_dixon(df: pd.DataFrame, var: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[var].dropna()
    q = x.rolling(WINDOW_SIZE, center=True, min_periods=5).apply(q_dixon_center, raw=True)
    result = pd.DataFrame({"Valor": x, "Q-Dixon": q, "Flag": classify(q)}, index=x.index)
    result.index.name = "Timestamp"

    total = int(result["Valor"].notna().sum())
    good = int(result["Flag"].eq("GOOD").sum())
    suspect = int(result["Flag"].eq("SUSPECT").sum())
    bad = int(result["Flag"].eq("BAD").sum())
    bad_pct = 100 * bad / total if total else np.nan

    summary = {
        "Variavel": var,
        "N": total,
        "Janela": WINDOW_SIZE,
        "Q_suspect": Q_SUSPECT,
        "Q_bad": Q_BAD,
        "GOOD": good,
        "SUSPECT": suspect,
        "BAD": bad,
        "BAD (%)": bad_pct,
        "Valor mínimo": x.min(),
        "Valor máximo": x.max(),
        "Q mediano": q.median(),
        "Q P95": q.quantile(0.95),
        "Q máximo": q.max(),
        "Interpretação": interpretation_from_bad_pct(bad_pct),
    }

    top_positive = result.sort_values("Valor", ascending=False).head(10)
    top_negative = result.sort_values("Valor", ascending=True).head(10)
    events = consecutive_bad_events(result)
    return result, summary, top_positive, top_negative, events


def period_summaries(result: pd.DataFrame, var: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = result.copy()
    base["variable"] = var
    base["year"] = base.index.year
    base["month"] = base.index.month
    base["season"] = [season(m) for m in base.index.month]
    base["is_bad"] = base["Flag"].eq("BAD")
    base["is_suspect"] = base["Flag"].eq("SUSPECT")

    def agg(group_cols: list[str]) -> pd.DataFrame:
        out = base.groupby(group_cols).agg(
            N=("Valor", "count"),
            SUSPECT=("is_suspect", "sum"),
            BAD=("is_bad", "sum"),
            Q_median=("Q-Dixon", "median"),
            Q_p95=("Q-Dixon", lambda s: s.quantile(0.95)),
        ).reset_index()
        out["BAD (%)"] = 100 * out["BAD"] / out["N"]
        out["SUSPECT (%)"] = 100 * out["SUSPECT"] / out["N"]
        return out

    return agg(["variable", "year"]), agg(["variable", "month"]), agg(["variable", "season"])


def write_markdown(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    lines = [
        "# Q-Dixon - SIMCOSTA_BA-1",
        "",
        "## Objetivo",
        "",
        "Aplicar o Q-Dixon como teste complementar para identificar valores extremos isolados em janelas locais da serie temporal. Diferente do Dixon 4σ, que compara cada observacao com a media e o desvio-padrao globais, o Q-Dixon pergunta se o ponto central de uma janela e um extremo isolado em relacao aos demais valores daquela janela.",
        "",
        "## Metodologia",
        "",
        f"Foi usada janela movel centrada de `{WINDOW_SIZE}` observacoes. Para cada janela, o ponto central so recebe Q-Dixon diferente de zero se for o maior ou menor valor da janela. O score e calculado como `Q = gap / range`, onde `gap` e a distancia entre o extremo e seu vizinho mais proximo na serie ordenada, e `range` e a amplitude da janela.",
        "",
        "Classificacao usada:",
        "",
        f"- GOOD: `Q < {Q_SUSPECT}`",
        f"- SUSPECT: `{Q_SUSPECT} <= Q <= {Q_BAD}`",
        f"- BAD: `Q > {Q_BAD}`",
        "",
        "Esses limiares sao uma adaptacao operacional para janela local. O resultado deve ser interpretado como triagem, nao como remocao automatica.",
        "",
        "## Resumo estatistico",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretacao geral",
        "",
    ]

    for _, row in summary.iterrows():
        lines.append(f"- `{row['Variavel']}`: {row['BAD (%)']:.4f}% BAD. {row['Interpretação']}")

    lines.extend([
        "",
        "## Eventos consecutivos BAD",
        "",
        "Eventos com `n_points = 1` sao candidatos a spikes instrumentais isolados. Eventos com `n_points >= 2` podem representar extremos fisicos persistentes, mas tambem podem indicar trecho instrumental problematico.",
        "",
        events.head(50).to_markdown(index=False, floatfmt=".4f") if not events.empty else "Nenhum evento BAD encontrado.",
        "",
        "## Relevancia cientifica",
        "",
        "O Q-Dixon e especialmente util para separar extremos isolados de eventos persistentes. Quando o Dixon 4σ marca um valor como BAD, mas o Q-Dixon nao marca, isso sugere que o valor extremo pode estar dentro de um bloco coerente de variabilidade. Quando ambos marcam o mesmo ponto, a evidencia de spike instrumental ou extremo local isolado fica mais forte.",
        "",
        "## Arquivos gerados",
        "",
        "- `q_dixon_resumo.csv`",
        "- `q_dixon_eventos_bad.csv`",
        "- `q_dixon_top_positivos.csv`",
        "- `q_dixon_top_negativos.csv`",
        "- `q_dixon_por_ano.csv`",
        "- `q_dixon_por_mes.csv`",
        "- `q_dixon_por_estacao.csv`",
        "- `q_dixon_<variavel>.csv` com Timestamp, Valor, Q-Dixon e Flag.",
    ])
    (OUT_DIR / "q_dixon_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    summaries, events, positives, negatives = [], [], [], []
    yearly, monthly, seasonal = [], [], []

    for var in [v for v in VARIABLES if v in df.columns]:
        result, summary, top_pos, top_neg, ev = run_q_dixon(df, var)
        result.reset_index().to_csv(OUT_DIR / f"q_dixon_{var}.csv", index=False)
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

        y, m, s = period_summaries(result, var)
        yearly.append(y)
        monthly.append(m)
        seasonal.append(s)

    summary_df = pd.DataFrame(summaries)
    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    positives_df = pd.concat(positives, ignore_index=True)
    negatives_df = pd.concat(negatives, ignore_index=True)
    yearly_df = pd.concat(yearly, ignore_index=True)
    monthly_df = pd.concat(monthly, ignore_index=True)
    seasonal_df = pd.concat(seasonal, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "q_dixon_resumo.csv", index=False)
    events_df.to_csv(OUT_DIR / "q_dixon_eventos_bad.csv", index=False)
    positives_df.to_csv(OUT_DIR / "q_dixon_top_positivos.csv", index=False)
    negatives_df.to_csv(OUT_DIR / "q_dixon_top_negativos.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "q_dixon_por_ano.csv", index=False)
    monthly_df.to_csv(OUT_DIR / "q_dixon_por_mes.csv", index=False)
    seasonal_df.to_csv(OUT_DIR / "q_dixon_por_estacao.csv", index=False)
    write_markdown(summary_df, events_df)
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
