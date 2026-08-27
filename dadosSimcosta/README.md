# Raw data goes here

Place the six SiMCosta OCEAN CSV files here before running the pipeline, named exactly as
`codigos/18_qc_lstm_geral_multiboia.py`'s `BUOYS` dict expects:

- `SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv`
- `SIMCOSTA_ES-1_OCEAN_2023-05-30_2026-07-25.csv`
- `SIMCOSTA_PR-1_OCEAN_2013-11-20_2025-05-23.csv`
- `SIMCOSTA_RJ-1_OCEAN_2015-07-29_2016-10-13.csv`
- `SIMCOSTA_RJ-2_OCEAN_2015-07-29_2016-12-20.csv`
- `SIMCOSTA_RJ-4_OCEAN_2017-08-28_2026-07-25.csv`

Download them from the [SiMCosta portal](https://www.simcosta.furg.br). If the portal now
serves different date ranges for the same buoys, update the filenames in the `BUOYS` dict
(and in the earlier per-buoy scripts that reference BA-1 directly) to match.

`codigos/00_fetch_wind_era5_ba1.py` populates this folder automatically with the ERA5
wind/gust proxy (`ERA5_BA-1_WIND_*.csv`); no manual download needed for that file.

These CSVs are intentionally excluded from version control (see `.gitignore`) to avoid
redistributing third-party data without a confirmed licence.
