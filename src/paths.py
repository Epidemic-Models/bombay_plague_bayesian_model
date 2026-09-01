# =============================================================================
# paths.py
# =============================================================================
"""
Single source of truth for every path the scripts use. Import from here
instead of redefining PROJECT_ROOT / results paths in each script --
that's what let them drift out of sync before.

*** EDIT MERGED_CSV / DATE_COL below to match your actual prepared_data
output. *** Everything else is derived and shouldn't need touching.

Layout this enforces:

    prepared_data/
        merged_target_climate.csv      <- input (edit MERGED_CSV if renamed)

    results/
        nested_sampling/
            <betaH_mode>__<fit_target>.pkl   <- raw dynesty results
            <betaH_mode>__<fit_target>.json  <- run metadata (settings, timestamp)
        posterior_summaries/
            posterior_summary_<betaH_mode>__<fit_target>.csv
        figures/
            posterior_predictive_<betaH_mode>__<fit_target>.png
            corner_<betaH_mode>__<fit_target>.png
        model_comparison_bayesian.csv        <- cross-model table (top level: spans all models)
"""
from __future__ import annotations
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# ------------------------------------------------------------
# Raw input data -- EDIT if your filenames differ
# ------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_XLSX = DATA_DIR / "bombay_plague_data.xlsx"
RAW_WEATHER_CSV = DATA_DIR / "bombay_weather_daily_1896_1899.csv"

# ------------------------------------------------------------
# Input data -- EDIT to match your actual prepared_data output
# ------------------------------------------------------------
PREPARED_DATA_DIR = PROJECT_ROOT / "prepared_data"
MERGED_CSV = PREPARED_DATA_DIR / "merged_target_climate.csv"
DATE_COL = "week_ending"

# ------------------------------------------------------------
# Outputs -- one subfolder per artifact type, nothing loose at top
# level except the one table that genuinely spans all models
# ------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
NESTED_SAMPLING_DIR = RESULTS_DIR / "nested_sampling"
POSTERIOR_SUMMARY_DIR = RESULTS_DIR / "posterior_summaries"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison_bayesian.csv"


def run_stem(betaH_mode: str, fit_target: str) -> str:
    """
    Canonical file-stem for one (model, fit-target) run. Every script
    that reads or writes a per-run file should build its filename from
    this, so a rerun against a different fit_target (or later, a
    different delay-distribution setting) can never silently overwrite
    an unrelated run.
    """
    return f"{betaH_mode}__{fit_target}"


def ensure_output_dirs() -> None:
    for d in (NESTED_SAMPLING_DIR, POSTERIOR_SUMMARY_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)