from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SPIKE_QC_PATH = BASE_DIR / "resultados_qc_ba1" / "spike_test_formula_principal" / "spike_qc_flags_long.csv"
DIXON_DIR = BASE_DIR / "resultados_qc_ba1" / "dixon_4sigma"
Q_DIXON_DIR = BASE_DIR / "resultados_qc_ba1" / "q_dixon"
OUT_DIR = BASE_DIR / "resultados_qc_ba1" / "base_qc_ready"

VARIABLES = [
    "Hsig", "Tp", "Hmax", "HM0",
    "Avg_Sal", "Avg_DO", "Avg_W_Tmp1", "Avg_W_Tmp2",
    "Avg_Turb", "Avg_Chl", "Avg_CDOM",
]


def normalize_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def read_spike_qc() -> pd.DataFrame:
    df = pd.read_csv(SPIKE_QC_PATH)
    df = df.rename(columns={"datetime_utc": "Timestamp", "value": "observed_value"})
    df["Timestamp"] = normalize_timestamp(df["Timestamp"])
    return df


def read_dixon() -> pd.DataFrame:
    frames = []
    for var in VARIABLES:
        path = DIXON_DIR / f"dixon_4sigma_{var}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Timestamp"] = normalize_timestamp(df["Timestamp"])
        df["variable"] = var
        df = df.rename(columns={"z-score": "dixon_4sigma_z", "Flag": "dixon_4sigma_flag"})
        frames.append(df[["Timestamp", "variable", "dixon_4sigma_z", "dixon_4sigma_flag"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_q_dixon() -> pd.DataFrame:
    frames = []
    for var in VARIABLES:
        path = Q_DIXON_DIR / f"q_dixon_{var}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Timestamp"] = normalize_timestamp(df["Timestamp"])
        df["variable"] = var
        df = df.rename(columns={"Q-Dixon": "q_dixon_score", "Flag": "q_dixon_flag"})
        frames.append(df[["Timestamp", "variable", "q_dixon_score", "q_dixon_flag"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def final_decision(row: pd.Series) -> str:
    if pd.isna(row.get("observed_value")):
        return "remove"
    if (
        bool(row.get("flag_range", False))
        or row.get("quality_class") == "Bad"
        or row.get("dixon_4sigma_flag") == "BAD"
        or row.get("q_dixon_flag") == "BAD"
        or bool(row.get("removed", False))
    ):
        return "remove"

    review_flags = [
        "flag_classic_spike",
        "flag_mad",
        "flag_second_difference",
        "flag_qartod_spike",
        "flag_roc",
        "flag_local_std",
    ]
    if any(bool(row.get(col, False)) for col in review_flags):
        return "review"
    if row.get("dixon_4sigma_flag") == "SUSPECT" or row.get("q_dixon_flag") == "SUSPECT":
        return "review"
    if pd.notna(row.get("quality_index")) and row.get("quality_index") < 75:
        return "review"
    return "use"


def final_quality_class(row: pd.Series) -> str:
    if row["final_decision"] == "remove":
        return "Bad"
    if row["final_decision"] == "review":
        return "Suspect"
    return str(row.get("quality_class")) if pd.notna(row.get("quality_class")) else "Excellent"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qc = read_spike_qc()
    dixon = read_dixon()
    q_dixon = read_q_dixon()

    final = qc.merge(dixon, on=["Timestamp", "variable"], how="left")
    final = final.merge(q_dixon, on=["Timestamp", "variable"], how="left")
    final["dixon_4sigma_flag"] = final["dixon_4sigma_flag"].fillna("MISSING")
    final["q_dixon_flag"] = final["q_dixon_flag"].fillna("MISSING")
    final["final_decision"] = final.apply(final_decision, axis=1)
    final["final_quality_class"] = final.apply(final_quality_class, axis=1)

    ordered_cols = [
        "Timestamp",
        "variable",
        "observed_value",
        "quality_index",
        "quality_class",
        "final_quality_class",
        "final_decision",
        "flag_range",
        "flag_roc",
        "flag_classic_spike",
        "flag_mad",
        "flag_second_difference",
        "flag_qartod_spike",
        "flag_local_std",
        "flag_count",
        "dixon_4sigma_z",
        "dixon_4sigma_flag",
        "q_dixon_score",
        "q_dixon_flag",
        "classic_score",
        "mad_score",
        "second_difference_score",
        "roc_per_hour",
        "local_std_score",
        "range_min",
        "range_max",
    ]
    final = final[[c for c in ordered_cols if c in final.columns]]
    final = final.sort_values(["Timestamp", "variable"])

    summary = final.groupby(["variable", "final_decision"]).size().unstack(fill_value=0)
    for col in ["use", "review", "remove"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["N"] = summary[["use", "review", "remove"]].sum(axis=1)
    summary["use_pct"] = 100 * summary["use"] / summary["N"]
    summary["review_pct"] = 100 * summary["review"] / summary["N"]
    summary["remove_pct"] = 100 * summary["remove"] / summary["N"]
    summary = summary.reset_index()

    final.to_csv(OUT_DIR / "base_qc_ready_long.csv", index=False)
    summary.to_csv(OUT_DIR / "base_qc_ready_summary_by_variable.csv", index=False)

    lines = [
        "# Base QC-ready - SIMCOSTA_BA-1",
        "",
        "Esta tabela consolida os testes de QC em uma base unica por `Timestamp` e `variable`.",
        "",
        "## Regra de decisao",
        "",
        "- `remove`: falha em range fisico, classe `Bad`, Dixon 4σ `BAD`, Q-Dixon `BAD` ou decisao `removed` da suite robusta.",
        "- `review`: qualquer flag estatistica, Dixon/Q-Dixon `SUSPECT` ou `QI < 75`.",
        "- `use`: observacao sem alertas relevantes.",
        "",
        "## Resumo por variavel",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Arquivos",
        "",
        "- `base_qc_ready_long.csv`: base oficial longa para o Digital Twin.",
        "- `base_qc_ready_summary_by_variable.csv`: resumo por variavel.",
    ]
    (OUT_DIR / "base_qc_ready_interpretacao.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
