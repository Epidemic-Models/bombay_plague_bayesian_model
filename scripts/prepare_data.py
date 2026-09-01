# =============================================================================
# prepare_data.py
# =============================================================================
"""
Prepare mortality targets and climate covariates for the Bayesian Bombay
plague model.

This script combines the two raw data sources:

    data/bombay_plague_data.xlsx
        Weekly plague cases, reported plague deaths, and all-cause mortality.

    data/bombay_weather_daily_1896_1899.csv
        Daily temperature, vapour pressure, and rainfall observations.

The script produces a single analysis-ready file:

    prepared_data/merged_target_climate.csv

Every Bayesian fitting script should read this prepared file rather than
repeating the data-cleaning and climate-preprocessing steps independently.

Processing steps
----------------
1. Read and clean the weekly Bombay plague data.

2. Construct two mortality targets:

       plague_deaths_reported_fit

   which is the reported plague mortality series used for fitting, with the
   documented reporting artifact for the week ending 19 February 1897
   replaced by linear interpolation;

   and

       plague_deaths_proxy

   which follows the definition used in the previous analysis:

       if excess mortality is available:
           max(reported plague deaths, nonnegative excess mortality)

       otherwise:
           reported plague deaths.

3. Build weekly climate covariates using src/climate.py.

   Temperature:
       7-day rolling mean
       14-day ecological lag

   Vapour pressure:
       7-day rolling mean
       14-day ecological lag

   Rainfall:
       28-day cumulative rainfall
       28-day ecological lag

   Climate variables are then aggregated to weeks ending Friday and
   standardized.

4. Merge mortality and climate data on week_ending.

Usage
-----
Run from the project root with:

    python scripts/prepare_data.py

The climate lags are fixed assumptions of the current mechanistic model.
They are not estimated from the data in this preprocessing step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports from src/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
from climate import build_weekly_climate  # noqa: E402


# =============================================================================
# Raw-data column names
# =============================================================================

CASES_COL = "Bombay_city_plague_cases"
DEATHS_COL = "Bombay_city_plague_deaths"
TOTAL_DEATHS_COL = "total_actual_mortality_all_causes"
BASELINE_DEATHS_COL = "total_average_mortality_all_causes"


# =============================================================================
# Known reporting artifact
# =============================================================================

OUTLIER_WEEK = "1897-02-19"


# =============================================================================
# Helpers
# =============================================================================

def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a potentially messy spreadsheet column to numeric values.

    Handles:
        commas,
        blank strings,
        dashes,
        em dashes,
        textual missing values.
    """
    s = series.astype(str).str.strip()

    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "-": np.nan,
            "—": np.nan,
        }
    )

    s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# Weekly mortality targets
# =============================================================================

def build_weekly_targets(raw_xlsx_path: str | Path) -> pd.DataFrame:
    """
    Read the raw Bombay plague spreadsheet and construct weekly fitting targets.

    Returns
    -------
    pd.DataFrame
        Columns:

        week_ending
            Week-ending date.

        week_index
            Sequential week index.

        plague_cases_reported
            Official reported Bombay plague cases.

        plague_deaths_reported
            Original reported plague deaths.

        plague_deaths_reported_fit
            Reported plague deaths used in fitting, after correction of the
            documented reporting artifact.

        total_mortality_actual
            Observed weekly all-cause mortality.

        total_mortality_average
            Historical/baseline average all-cause mortality.

        excess_mortality_raw
            Actual minus baseline all-cause mortality.

        excess_mortality_nonneg
            max(0, excess_mortality_raw), retaining NaN when excess mortality
            cannot be calculated.

        plague_deaths_proxy
            Alternative mortality target:

                if excess mortality is available:
                    max(
                        plague_deaths_reported,
                        excess_mortality_nonneg
                    )

                otherwise:
                    plague_deaths_reported.
    """
    raw_xlsx_path = Path(raw_xlsx_path)

    df = pd.read_excel(raw_xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    needed = [
        paths.DATE_COL,
        CASES_COL,
        DEATHS_COL,
        TOTAL_DEATHS_COL,
        BASELINE_DEATHS_COL,
    ]

    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(
            f"{raw_xlsx_path} is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------------------

    df[paths.DATE_COL] = pd.to_datetime(
        df[paths.DATE_COL],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Numeric cleaning
    # -------------------------------------------------------------------------

    for col in [
        CASES_COL,
        DEATHS_COL,
        TOTAL_DEATHS_COL,
        BASELINE_DEATHS_COL,
    ]:
        df[col] = _clean_numeric(df[col])

    # -------------------------------------------------------------------------
    # Keep usable epidemic observations
    # -------------------------------------------------------------------------

    df = (
        df[
            [
                paths.DATE_COL,
                CASES_COL,
                DEATHS_COL,
                TOTAL_DEATHS_COL,
                BASELINE_DEATHS_COL,
            ]
        ]
        .dropna(
            subset=[
                paths.DATE_COL,
                CASES_COL,
                DEATHS_COL,
            ]
        )
        .sort_values(paths.DATE_COL)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Construct output table
    # -------------------------------------------------------------------------

    out = pd.DataFrame(
        {
            paths.DATE_COL: df[paths.DATE_COL],
            "week_index": np.arange(len(df), dtype=int),
            "plague_cases_reported": df[CASES_COL],
            "plague_deaths_reported": df[DEATHS_COL],
            "total_mortality_actual": df[TOTAL_DEATHS_COL],
            "total_mortality_average": df[BASELINE_DEATHS_COL],
        }
    )

    # -------------------------------------------------------------------------
    # Excess mortality
    # -------------------------------------------------------------------------

    out["excess_mortality_raw"] = (
        out["total_mortality_actual"]
        - out["total_mortality_average"]
    )

    out["excess_mortality_nonneg"] = (
        out["excess_mortality_raw"]
        .clip(lower=0.0)
    )

    # -------------------------------------------------------------------------
    # Proxy target
    #
    # Preserve the definition used in the previous fitting workflow:
    #
    # if excess mortality is available:
    #     max(reported plague deaths, nonnegative excess mortality)
    #
    # otherwise:
    #     reported plague deaths
    # -------------------------------------------------------------------------

    reported = out["plague_deaths_reported"].to_numpy(dtype=float)
    excess_nonneg = out["excess_mortality_nonneg"].to_numpy(dtype=float)

    out["plague_deaths_proxy"] = np.where(
        np.isfinite(excess_nonneg),
        np.maximum(reported, excess_nonneg),
        reported,
    )

    # -------------------------------------------------------------------------
    # Correct documented reporting artifact for reported-deaths fitting target
    # -------------------------------------------------------------------------

    out["plague_deaths_reported_fit"] = (
        out["plague_deaths_reported"].copy()
    )

    mask = (
        out[paths.DATE_COL]
        == pd.Timestamp(OUTLIER_WEEK)
    )

    if mask.any():

        out.loc[
            mask,
            "plague_deaths_reported_fit",
        ] = np.nan

        out["plague_deaths_reported_fit"] = (
            out["plague_deaths_reported_fit"]
            .interpolate(method="linear")
        )

        print(
            f"[OK] Corrected reporting artifact at week ending "
            f"{OUTLIER_WEEK} by linear interpolation."
        )

    else:

        print(
            f"[WARNING] Expected outlier week {OUTLIER_WEEK} "
            "was not found."
        )

    return out


# =============================================================================
# Main preparation pipeline
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not paths.RAW_XLSX.exists():
        raise SystemExit(
            f"Could not find {paths.RAW_XLSX}. "
            "Check RAW_XLSX in src/paths.py."
        )

    if not paths.RAW_WEATHER_CSV.exists():
        raise SystemExit(
            f"Could not find {paths.RAW_WEATHER_CSV}. "
            "Check RAW_WEATHER_CSV in src/paths.py."
        )

    # -------------------------------------------------------------------------
    # Weekly epidemic targets
    # -------------------------------------------------------------------------

    print(
        "[1/3] Building weekly plague targets "
        "from raw spreadsheet ..."
    )

    target_df = build_weekly_targets(
        paths.RAW_XLSX
    )

    print(
        f"      {len(target_df)} weeks: "
        f"{target_df[paths.DATE_COL].min().date()} "
        f"to "
        f"{target_df[paths.DATE_COL].max().date()}"
    )

    # -------------------------------------------------------------------------
    # Weekly climate
    # -------------------------------------------------------------------------

    print(
        "[2/3] Building weekly climate covariates ..."
    )

    climate_weekly = build_weekly_climate(
        daily_csv_path=str(paths.RAW_WEATHER_CSV),

        date_col="date",

        temp_col="Temperature_F",
        vapour_col="VapourPressure_inHg",
        rain_col="Rain_in",

        # rolling windows
        T_win=7,
        e_win=7,
        R_win=28,

        # fixed ecological lags
        ell_T=14,
        ell_e=14,
        ell_R=28,

        # Bombay mortality series uses Friday-ending weeks
        week_rule="W-FRI",

        standardize=True,
    )

    # -------------------------------------------------------------------------
    # Ensure consistent date naming
    # -------------------------------------------------------------------------

    if paths.DATE_COL not in climate_weekly.columns:

        date_like = [
            c
            for c in climate_weekly.columns
            if pd.api.types.is_datetime64_any_dtype(
                climate_weekly[c]
            )
        ]

        if len(date_like) == 1:

            climate_weekly = climate_weekly.rename(
                columns={
                    date_like[0]: paths.DATE_COL
                }
            )

        else:

            raise ValueError(
                f"Could not identify '{paths.DATE_COL}' "
                "in climate output.\n"
                f"Columns returned: "
                f"{list(climate_weekly.columns)}"
            )

    climate_weekly[paths.DATE_COL] = pd.to_datetime(
        climate_weekly[paths.DATE_COL]
    )

    print(
        f"      {len(climate_weekly)} climate weeks"
    )

    # -------------------------------------------------------------------------
    # Merge mortality + climate
    # -------------------------------------------------------------------------

    print(
        "[3/3] Merging mortality and climate data ..."
    )

    merged = pd.merge(
        target_df,
        climate_weekly,
        on=paths.DATE_COL,
        how="inner",
    )

    merged = (
        merged
        .sort_values(paths.DATE_COL)
        .reset_index(drop=True)
    )

    if merged.empty:
        raise SystemExit(
            "No overlapping weeks between mortality "
            "and climate data. Check week alignment."
        )

    # Recompute sequential index AFTER the merge.
    # This avoids gaps caused by climate-lag burn-in.
    merged["week_index"] = np.arange(
        len(merged),
        dtype=int,
    )

    dropped = len(target_df) - len(merged)

    if dropped > 0:

        print(
            f"      NOTE: {dropped} target week(s) "
            "had no valid climate covariates and were dropped."
        )

    # -------------------------------------------------------------------------
    # Validate fitting targets
    # -------------------------------------------------------------------------

    for fit_col in [
        "plague_deaths_reported_fit",
        "plague_deaths_proxy",
    ]:

        n_nan = int(
            merged[fit_col]
            .isna()
            .sum()
        )

        if n_nan > 0:

            bad_weeks = (
                merged.loc[
                    merged[fit_col].isna(),
                    paths.DATE_COL,
                ]
                .dt.date
                .tolist()
            )

            print(
                f"[WARNING] {fit_col} contains "
                f"{n_nan} missing observation(s): "
                f"{bad_weeks}"
            )

    # -------------------------------------------------------------------------
    # Save prepared dataset
    # -------------------------------------------------------------------------

    paths.PREPARED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        paths.MERGED_CSV,
        index=False,
    )

    print(
        f"[OK] Saved prepared dataset:\n"
        f"     {paths.MERGED_CSV}"
    )

    print(
        f"[OK] Number of fitted weeks: "
        f"{len(merged)}"
    )

    print(
        "[OK] Columns:"
    )

    for col in merged.columns:
        print(f"     {col}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()