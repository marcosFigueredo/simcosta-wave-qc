# A portable causal ordinal classifier for real-time quality control of significant wave height in coastal buoy networks

Code, trained model and curated result summaries accompanying the manuscript submitted to
*Ocean Engineering* (Elsevier). The pipeline combines an established statistical quality-control
protocol with a causal ordinal classifier (GOOD / SUSPECT / BAD) trained once on six pooled
SiMCosta buoys, and evaluates it under pooled, leave-one-buoy-out, and streaming protocols.

## Citation

If you use this code or the released model, please cite the paper (full citation and DOI to be
added on publication) and, if applicable, this repository's own archival DOI (see badge below
once minted).

<!-- [![DOI](https://zenodo.org/badge/DOI/XXXXXXX.svg)](https://doi.org/XXXXXXX) -->

## Repository structure

```
simcosta-wave-qc/
├── codigos/                    Numbered pipeline scripts, run in the order below
├── resultados_qc_ba1/          Curated result summaries and the final trained model
│   ├── qc_lstm_geral_multiboia/
│   │   └── modelo_final/       Trained predictor + ordinal classifier + scalers (E2, "the" model)
│   ├── lstm_peak_qc/           E1 comparator reproduction outputs
│   ├── ajustes_vagner/         Predictive error (E2), census, QI firing rates, variance decomposition
│   ├── dixon_4sigma/, q_dixon/, spike_test_formula_principal/, base_qc_ready/
│   │                           Classical QC protocol summaries (large per-timestamp flag
│   │                           dumps are excluded, see below)
│   └── ...
├── dadosSimcosta/               Empty here; place the raw SiMCosta OCEAN CSVs before running
├── requirements.txt
└── LICENSE
```

Large, fully regenerable intermediate files (per-timestamp flag dumps from the classical QC
protocol, on the order of hundreds of MB) are excluded via `.gitignore` and are not needed to
verify any number reported in the paper; only their summary/aggregate outputs are tracked here.
Re-running the relevant numbered script reproduces them from the raw data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Developed against Python 3.11/3.12. TensorFlow 2.19 is CPU-only in the reference environment;
no GPU is required to reproduce any result in the paper.

## Data

- **SiMCosta buoy records** (BA-1, ES-1, PR-1, RJ-1, RJ-2, RJ-4): download the OCEAN CSV files
  for each buoy from the [SiMCosta portal](https://www.simcosta.furg.br) and place them in
  `dadosSimcosta/`, matching the filenames referenced in `codigos/` (e.g.
  `SIMCOSTA_BA-1_OCEAN_<start>_<end>.csv`). SiMCosta is coordinated by the Instituto de
  Oceanografia, Universidade Federal do Rio Grande (FURG); consult the portal for current
  terms of use.
- **ERA5 wind speed and gust at 10 m**: retrieved automatically via the
  [Open-Meteo historical API](https://open-meteo.com/) by `codigos/00_fetch_wind_era5_ba1.py`
  (no API key required). This is a reanalysis proxy at the nearest grid node, not an in-situ
  buoy measurement.

The raw buoy files are not redistributed in this repository; the retrieval described above lets
the analysed subset be regenerated from source rather than from a static copy, consistent with
the manuscript's data availability statement.

## Reproducing the paper's results

Run the numbered scripts in `codigos/` in order. Each stage's outputs land under
`resultados_qc_ba1/<stage>/`, matching the curated files already included here.

| Script | Produces | Paper reference |
|---|---|---|
| `00_fetch_wind_era5_ba1.py` | ERA5 wind/gust proxy for BA-1 | Section 2.2 |
| `01_spike_test_simcosta_ba1.py` | Classical spike/QI index, all 11 variables | Table 5, Supp. Table S1 |
| `02_dixon_4sigma_simcosta_ba1.py`, `03_q_dixon_simcosta_ba1.py` | Dixon 4σ / Q-Dixon outlier tests | Table 5 |
| `05_consolidar_base_qc_ready_ba1.py` | Consolidated reference label $R_t$ | Section 3.2 |
| `04_lstm_peak_qc_ba1.py` | E1: LSTM-Peak reproduction + 5 comparators | Table 6, Fig. 6, Supp. Figs. S2–S3 |
| `06_ablacao_lstm_ba1.py` | Predictive-network ablation | Supp. Table S6, Fig. S1 |
| `09_eda_basica_ba1.py` | Descriptive statistics, coverage | Table 2 |
| `16_qc_lstm_univariado_ba1.py` | Univariate causal predictor (shared by E2/E3) | Section 2.4 |
| `18_qc_lstm_geral_multiboia.py` | **E2**: general multi-buoy model, trains and saves `modelo_final/` | Table 7 (pooled) |
| `25_reavaliar_modelo_geral_rotulos_fortes.py` | Re-evaluates the saved E2 model on strong labels | Table 7 (pooled), Supp. Table S12, Fig. S5 |
| `24_leave_one_buoy_out_e3.py` | **E3**: leave-one-buoy-out transfer, 6 folds | Table 7 (LOBO), Supp. Table S8 |
| `20_estabilidade_modelo_geral.py` | 15-run stability grid (3 model seeds × 5 injection seeds) | Table 8 |
| `19_simulador_tempo_real.py` | **E4**: streaming simulator, reference run | Table 9, Fig. 7 |
| `21_estabilidade_simulador.py` | Streaming stability, 10 seeds | Table 9 (aggregated) |
| `22_reconciliacao_hsig.py` | **E5**: reconciliation of $R_t$ and $Q_t$ on the real series | Table 10 |
| `23_executar_ajustes_vagner.py` | E2 predictive error, buoy census, QI firing rates/co-occurrence, variance decomposition, streaming operational metrics | Table 4 (E2 row), Supp. Tables S4, S5, S9, S11 |
| `10`–`15`, `17` | Earlier single-buoy / softmax variant, superseded by the causal ordinal design | Supp. Section S7 |

Scripts import one another via `importlib` rather than package-relative imports (e.g. `18` builds
on `16`, `25` builds on `18` and `19`); run them from inside `codigos/` so the relative model and
data paths resolve correctly.

The trained model in `resultados_qc_ba1/qc_lstm_geral_multiboia/modelo_final/` lets any script
that only evaluates (`19`, `22`, `24`, `25`) run without repeating the E2 training step.

## Authors and correspondence

Marcos Batista Figueredo (mbfigueredo@uneb.br) and Vagner Fonseca (vagner.fonseca@gmail.com),
corresponding authors. Full author list, affiliations and CRediT contributions are given in the
manuscript.

## License

Code released under the MIT License (see `LICENSE`). This licence does not extend to the
third-party SiMCosta and ERA5 data described above.
