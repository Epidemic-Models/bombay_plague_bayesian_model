# climate.py
# ------------------------------------------------------------
# Climate preprocessing for the plague model.
#
# Input: daily CSV with columns like:
#   date, Temperature_F, VapourPressure_inHg, Rain_in
#
# Output: weekly covariates aligned to week-ending dates (e.g. W-SUN):
#   week_ending, xT, xe, xR, zT, ze, zR
#
# Steps:
#   1) rolling summaries (biological short-term memory):
#        T7  = 7-day mean temperature
#        e7  = 7-day mean vapour pressure
#        R28 = 28-day cumulative rainfall (sum, not mean)
#   2) lag each summary by ell_* days (ecological delay)
#   3) resample to weekly (week-ending) using mean (T,e) and sum/mean choice for R
#   4) standardize across the study period (optional)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score with ddof=0; safe for constant series."""
    x = series.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return (x - mu) * 0.0
    return (x - mu) / sd


# -----------------------------
# Main API
# -----------------------------
def build_weekly_climate(
    daily_csv_path: str,
    date_col: str = "date",
    temp_col: str = "Temperature_F",
    vapour_col: str = "VapourPressure_inHg",
    rain_col: str = "Rain_in",
    # rolling windows (days)
    T_win: int = 7,
    e_win: int = 7,
    R_win: int = 28,
    # lags (days)
    ell_T: int = 14,
    ell_e: int = 14,
    ell_R: int = 28,
    # weekly alignment
    week_rule: str = "W-SUN",   # "week ending Sunday" (matches many historical weekly series)
    # standardize?
    standardize: bool = True,
    # how to aggregate rainfall at weekly stage?
    # NOTE: xR is already a 28-day accumulation at the DAILY level.
    # When resampling to weeks, "mean" gives the typical 28-day accumulation during that week.
    rain_week_agg: str = "mean",  # {"mean","last"} are most common here
) -> pd.DataFrame:
    """
    Returns a weekly dataframe with columns:
      week_ending, xT, xe, xR, (optional) zT, ze, zR

    Interpretation:
      xT, xe, xR are lagged rolling summaries evaluated through time.
      If standardize=True, z* are z-scores over the returned weekly series.

    rain_week_agg:
      - "mean": typical 28-day accumulation during that week
      - "last": value on week-ending day (also reasonable)
    """
    # --- load ---
    df = pd.read_csv(daily_csv_path)
    df.columns = [c.strip() for c in df.columns]
    _require_cols(df, [date_col, temp_col, vapour_col, rain_col])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # numeric
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[vapour_col] = pd.to_numeric(df[vapour_col], errors="coerce")
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0)

    # set index for rolling + resample
    df = df.set_index(date_col)

    # --- rolling summaries (daily) ---
    # Means for T and e
    T7 = df[temp_col].rolling(window=T_win, min_periods=T_win).mean()
    e7 = df[vapour_col].rolling(window=e_win, min_periods=e_win).mean()
    # Sum for R (cumulative total rainfall)
    R28 = df[rain_col].rolling(window=R_win, min_periods=R_win).sum()

    # --- lags (days) ---
    T7_lag = T7.shift(ell_T)
    e7_lag = e7.shift(ell_e)
    R28_lag = R28.shift(ell_R)

    # assemble daily processed frame
    daily_proc = pd.DataFrame(
        {
            "T7_lag": T7_lag,
            "e7_lag": e7_lag,
            "R28_lag": R28_lag,
        },
        index=df.index,
    ).dropna()

    # --- resample to weekly ---
    # For temperature and vapour pressure, weekly mean of the daily lagged series is standard.
    wk_T = daily_proc["T7_lag"].resample(week_rule).mean()
    wk_e = daily_proc["e7_lag"].resample(week_rule).mean()

    # For rainfall (already a 28-day accumulation), either:
    #   - take weekly mean (typical state during week), or
    #   - take last value on week-ending day
    if rain_week_agg == "mean":
        wk_R = daily_proc["R28_lag"].resample(week_rule).mean()
    elif rain_week_agg == "last":
        wk_R = daily_proc["R28_lag"].resample(week_rule).last()
    else:
        raise ValueError("rain_week_agg must be 'mean' or 'last'")

    weekly = pd.DataFrame(
        {
            "xT": wk_T,
            "xe": wk_e,
            "xR": wk_R,
        }
    ).dropna()

    weekly = weekly.reset_index().rename(columns={date_col: "week_ending"})

    # --- standardize (optional) ---
    if standardize:
        weekly["zT"] = _zscore(weekly["xT"])
        weekly["ze"] = _zscore(weekly["xe"])
        weekly["zR"] = _zscore(weekly["xR"])

    return weekly


def save_weekly_climate_csv(
    daily_csv_path: str,
    out_csv_path: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: builds weekly climate covariates and writes CSV.
    """
    weekly = build_weekly_climate(daily_csv_path, **kwargs)
    weekly.to_csv(out_csv_path, index=False)
    return weekly