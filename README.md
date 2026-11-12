# Pump.fun Top-5 Forecast

A deterministic, standard-library-only Python forecaster for make.fun's `pumpfun-top5-forecast-01` competition.

## Run

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

The grader runs `python3 /bot/main.py`; defaults are `/input` and `/output`. It writes `/output/forecast.json`.

The forecast combines observed market-cap ranks and robust recent log-cap momentum with horizon-scaled uncertainty. It leaves explicit probability mass unassigned for coins outside the observed candidate set, including coins launched after the input cutoff. Random draws use a fixed seed, and only the Python standard library is required.
