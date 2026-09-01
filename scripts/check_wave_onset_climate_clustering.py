# =============================================================================
# check_wave_onset_climate_clustering.py
# =============================================================================
"""
Tests whether a single, recurring climatic condition could explain the
repeated onset of spillover across the three epidemic waves, or whether
each wave's onset was preceded by distinct conditions.

Method
------
1. Load the canonical wave-boundary weeks from
   wave_boundary_weeks.csv (produced by
   analyse_epidemic_waves_and_amplification.py), rather than running an
   independent local-minima detection pass on the mortality series.

   *** FIXED: earlier versions of this script detected their own
   candidate troughs via scipy.signal.find_peaks(-observed, distance=8,
   prominence=50), using different detection parameters (raw death
   counts, distance=8 weeks, prominence=50 deaths) than the smoothed-log,
   distance=20-week, prominence=0.6-log-scale detection used everywhere
   else in the paper (analyse_epidemic_waves_and_amplification.py). There
   is no guarantee these two independent detection passes land on
   exactly the same weeks -- even a one-week mismatch would mean this
   script's "wave-onset troughs" are not the same troughs the rest of
   the manuscript calls the inter-wave troughs, silently undermining any
   claim phrased as being about "the" inter-wave troughs. This version
   loads the already-validated boundary weeks directly instead, so the
   climate-clustering claim is guaranteed to be about the same three
   boundaries (series start, and the two genuine internal troughs) used
   throughout the rest of the analysis. ***

2. Pull the standardized climate covariates (zT, ze, zR) at each of the
   three boundary weeks: the series-start boundary (1896-10-02) and the
   two genuine internal inter-wave troughs (1897-07-23, 1898-06-24).
3. Report the spread (range, std) across the two genuine internal
   troughs, and note explicitly that the series-start boundary is not a
   true pre-onset trough (no real pre-onset history is available before
   the start of the fitted series) and should be treated with caution
   relative to the two internal troughs, which are genuine, fully-observed
   inter-wave lulls.

This is a descriptive check, not a formal statistical test -- with only
two genuine internal trough comparisons, clustering or non-clustering
should be reported as suggestive, not conclusive.

Requires
--------
The wave-boundary weeks CSV, produced by
analyse_epidemic_waves_and_amplification.py:
    results/final_figures/transmission_dynamics/wave_analysis/
        wave_boundary_weeks.csv
Run that script first if this file does not exist.

Outputs
-------
results/final_figures/
    wave_onset_climate_conditions.csv

Usage
-----
    python scripts/check_wave_onset_climate_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402

FIT_TARGET = "plague_deaths_reported_fit"

# The canonical wave-boundary weeks, produced by
# analyse_epidemic_waves_and_amplification.py. This script uses these
# directly rather than running its own independent trough-detection pass.
BOUNDARY_CSV = (
    PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
    / "wave_analysis" / "wave_boundary_weeks.csv"
)

# The two genuine internal inter-wave troughs (same dates used throughout
# the rest of the analysis, e.g. check_spillover_reseeding_robustness.py
# and sweep_gamma_reseeding_falsification.py).
TROUGH_1_DATE = pd.Timestamp("1897-07-23")  # between Wave 1 and Wave 2
TROUGH_2_DATE = pd.Timestamp("1898-06-24")  # between Wave 2 and Wave 3

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
df = df.sort_values(paths.DATE_COL).reset_index(drop=True)

if FIT_TARGET not in df.columns:
    raise ValueError(f"{FIT_TARGET!r} not found in prepared data.")

observed = df[FIT_TARGET].to_numpy(dtype=float)
week_ending_dt = pd.to_datetime(df[paths.DATE_COL])

print("=" * 90)
print("WAVE-ONSET CLIMATE CLUSTERING CHECK")
print("=" * 90)
print(f"\nFitted weeks: {len(df)}  ({df[paths.DATE_COL].min().date()} to "
      f"{df[paths.DATE_COL].max().date()})")


# =============================================================================
# Load the canonical wave-boundary weeks (not an independent detection pass)
# =============================================================================

if not BOUNDARY_CSV.exists():
    raise FileNotFoundError(
        f"Wave boundary weeks not found:\n{BOUNDARY_CSV}\n\n"
        "Run analyse_epidemic_waves_and_amplification.py first "
        "(it exports wave_boundary_weeks.csv as part of its normal output)."
    )

boundary_df = pd.read_csv(BOUNDARY_CSV, parse_dates=["week_ending"])

print(f"\nLoaded {len(boundary_df)} wave boundaries from {BOUNDARY_CSV.name}:")
for _, row in boundary_df.iterrows():
    print(f"  {row['week_ending'].date()}  deaths={row['observed_deaths']:.0f}  "
          f"({row['boundary_type']})")

if len(boundary_df) != 4:
    raise RuntimeError(
        f"Expected 4 wave boundaries (series start, two internal troughs, "
        f"series end), found {len(boundary_df)}. Inspect {BOUNDARY_CSV} "
        "before proceeding."
    )

# Validate the two internal-trough dates match what the rest of the
# analysis uses (same check pattern as check_spillover_reseeding_robustness.py
# and sweep_gamma_reseeding_falsification.py).
internal_troughs = boundary_df[boundary_df["boundary_type"] == "internal_trough"]
internal_dates = set(pd.to_datetime(internal_troughs["week_ending"]))

for label, d in (("Trough 1 (1897-07-23)", TROUGH_1_DATE),
                  ("Trough 2 (1898-06-24)", TROUGH_2_DATE)):
    if d not in internal_dates:
        raise ValueError(
            f"{label} not found among the internal-trough dates in "
            f"{BOUNDARY_CSV}. Check that the hardcoded trough dates "
            f"still match the current wave-boundary detection output."
        )

series_start_row = boundary_df[boundary_df["boundary_type"] == "series_boundary"].iloc[0]


# =============================================================================
# Build the onset table using the canonical boundary weeks
# =============================================================================

def climate_row_for_date(date: pd.Timestamp) -> pd.Series:
    mask = (week_ending_dt == date).to_numpy()
    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one fitted week matching {date.date()}, "
            f"found {mask.sum()}."
        )
    return df.iloc[np.where(mask)[0][0]]


rows = []

# Series-start boundary: not a true pre-onset trough (no real pre-onset
# history is available before the start of the fitted series).
start_row = climate_row_for_date(pd.Timestamp(series_start_row["week_ending"]))
rows.append({
    "label": "Series-start boundary (CAUTION: no pre-onset history available)",
    "week": start_row[paths.DATE_COL].date(), "deaths": start_row[FIT_TARGET],
    "zT": start_row["zT"], "ze": start_row["ze"], "zR": start_row["zR"],
    "is_edge_confounded": True,
})

# The two genuine internal inter-wave troughs.
for label, date in (
    ("Internal trough 1 (Wave 1 -> Wave 2)", TROUGH_1_DATE),
    ("Internal trough 2 (Wave 2 -> Wave 3)", TROUGH_2_DATE),
):
    row = climate_row_for_date(date)
    rows.append({
        "label": label,
        "week": row[paths.DATE_COL].date(), "deaths": row[FIT_TARGET],
        "zT": row["zT"], "ze": row["ze"], "zR": row["zR"],
        "is_edge_confounded": False,
    })

result_df = pd.DataFrame(rows)

print("\n" + "-" * 90)
print("CLIMATE CONDITIONS AT EACH WAVE-BOUNDARY WEEK")
print("-" * 90)
print(result_df.to_string(index=False))

csv_path = OUTPUT_DIR / "wave_onset_climate_conditions.csv"
result_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")

print("\n" + "-" * 90)
print("SPREAD ACROSS THE TWO GENUINE INTERNAL TROUGHS (excludes series-start boundary)")
print("-" * 90)
clean = result_df[~result_df["is_edge_confounded"]]
for col in ["zT", "ze", "zR"]:
    vals = clean[col].round(3).tolist()
    rng = clean[col].max() - clean[col].min()
    std = clean[col].std()
    print(f"  {col}: values={vals}  range={rng:.3f}  std={std:.3f}")
    if rng < 0.5:
        print(f"    -> TIGHT clustering: consistent with a shared recurring trigger.")
    elif rng < 1.2:
        print(f"    -> Moderate clustering: partial consistency, worth reporting cautiously.")
    else:
        print(f"    -> NOT clustered: no consistent shared signature across onsets.")

print("\nCAUTION: with only two genuine internal troughs, this is a descriptive "
      "check, not a statistical test. The series-start boundary row is included "
      "for reference only and should be treated as unreliable, since there is "
      "no true pre-onset climate history available before the start of the "
      "fitted series.")
print("\nDone.")