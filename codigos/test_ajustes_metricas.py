import numpy as np
import pandas as pd

from ajustes_metricas import descriptive_variance_allocation, streaming_operational_metrics


def test_streaming_metrics_count_clean_hours_and_episodes():
    log = pd.DataFrame(
        {
            "rotulo_verdadeiro": ["GOOD", "GOOD", "BAD", "BAD", "GOOD", "SUSPECT"],
            "rotulo_previsto": ["GOOD", "BAD", "GOOD", "BAD", "SUSPECT", "GOOD"],
        }
    )
    result = streaming_operational_metrics(log)
    assert result["n_clean_hours"] == 3
    assert result["false_alarms"] == 2
    assert result["n_episodes"] == 2
    assert np.isclose(result["mean_episode_coverage_pct"], 25.0)


def test_variance_allocation_exposes_remaining_term_without_inferential_claim():
    rows = []
    for model_seed in [1, 2, 3]:
        for anomaly_seed_base in [10, 20, 30, 40, 50]:
            rows.append({"model_seed": model_seed, "anomaly_seed_base": anomaly_seed_base,
                         "metric": model_seed + anomaly_seed_base / 100.0})
    result = descriptive_variance_allocation(pd.DataFrame(rows), "metric")
    assert set(result) == {
        "pct_var_model_seed", "pct_var_anomaly_seed",
        "pct_var_remaining_interaction_or_error",
    }
    assert np.isclose(sum(result.values()), 100.0)
