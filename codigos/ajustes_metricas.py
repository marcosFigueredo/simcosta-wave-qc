"""Pure helpers for the quantitative audit of the Ocean Engineering manuscript."""

from __future__ import annotations

import numpy as np
import pandas as pd


def streaming_operational_metrics(log: pd.DataFrame) -> dict[str, float | int]:
    """Return false-alarm and episode-coverage metrics from one simulation log."""
    clean = log["rotulo_verdadeiro"].eq("GOOD")
    false_alarms = int((clean & log["rotulo_previsto"].ne("GOOD")).sum())
    n_clean = int(clean.sum())

    anomaly = ~clean
    episode_id = (~anomaly).cumsum()
    episode_id = episode_id.where(anomaly, -1)
    coverages: list[float] = []
    for _, group in log[episode_id.ne(-1)].groupby(episode_id[episode_id.ne(-1)]):
        coverages.append(float(group["rotulo_previsto"].ne("GOOD").mean()))

    return {
        "n_clean_hours": n_clean,
        "false_alarms": false_alarms,
        "false_alarm_rate_pct": 100.0 * false_alarms / max(1, n_clean),
        "n_episodes": len(coverages),
        "mean_episode_coverage_pct": 100.0 * float(np.mean(coverages)) if coverages else 0.0,
    }


def descriptive_variance_allocation(values: pd.DataFrame, metric: str) -> dict[str, float]:
    """Allocate crossed 3x5 variation descriptively; no interaction p-value is claimed."""
    table = values.pivot(index="model_seed", columns="anomaly_seed_base", values=metric).to_numpy()
    grand = float(np.mean(table))
    n_models, n_anomaly = table.shape
    ss_total = float(np.sum((table - grand) ** 2))
    ss_model = float(n_anomaly * np.sum((table.mean(axis=1) - grand) ** 2))
    ss_anomaly = float(n_models * np.sum((table.mean(axis=0) - grand) ** 2))
    ss_remaining = max(0.0, ss_total - ss_model - ss_anomaly)
    denom = max(ss_total, 1e-12)
    return {
        "pct_var_model_seed": 100.0 * ss_model / denom,
        "pct_var_anomaly_seed": 100.0 * ss_anomaly / denom,
        "pct_var_remaining_interaction_or_error": 100.0 * ss_remaining / denom,
    }
