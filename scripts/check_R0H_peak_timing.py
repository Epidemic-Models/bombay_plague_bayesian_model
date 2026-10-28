# =============================================================================
# check_R0H_peak_timing.py
# =============================================================================
"""
Where does R0_H(t) peak within each wave, relative to the mortality peak?

Checks whether the wave-1 timing ("R0_H peaked 18-32 weeks before the
mortality peak") comes from the spike in the first weeks of the series.

Reads only the saved output of final_transmission_dynamics.py:
    results/final_figures/transmission_dynamics/
        posterior_transmission_dynamics_all_climate.csv

Usage
-----
    python scripts/check_R0H_peak_timing.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = (PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
       / "posterior_transmission_dynamics_all_climate.csv")

# Wave windows: series start, the two internal troughs, series end.
BOUNDARIES = [pd.Timestamp("1896-09-25"), pd.Timestamp("1897-07-23"),
              pd.Timestamp("1898-06-24"), pd.Timestamp("1899-06-02")]
MORTALITY_PEAKS = [pd.Timestamp("1897-02-12"), pd.Timestamp("1898-03-18"),
                   pd.Timestamp("1899-03-24")]

# Weeks at the start of the series to leave out in the second check.
SKIP_FIRST_WEEKS = 4

df = pd.read_csv(CSV, parse_dates=["week_ending"])

print("First 8 weeks of R0_H (posterior median):")
print(df[["week_ending", "R0_H_median", "R0_H_q025", "R0_H_q975"]]
      .head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

first_kept = df["week_ending"].iloc[SKIP_FIRST_WEEKS]

print(f"\n{'Wave':<6}{'Mortality peak':<16}{'R0_H peak (all weeks)':<34}"
      f"{'R0_H peak (first ' + str(SKIP_FIRST_WEEKS) + ' weeks left out)':<40}")
print("-" * 96)
for w in range(3):
    lo, hi = BOUNDARIES[w], BOUNDARIES[w + 1]
    win = df[(df["week_ending"] >= lo) & (df["week_ending"] <= hi)]
    win_skip = win[win["week_ending"] >= first_kept]

    def describe(sub):
        r = sub.loc[sub["R0_H_median"].idxmax()]
        weeks = (r["week_ending"] - MORTALITY_PEAKS[w]).days / 7
        return f"{r['week_ending'].date()} ({r['R0_H_median']:.2f}), {weeks:+.0f} wk"

    print(f"{w + 1:<6}{str(MORTALITY_PEAKS[w].date()):<16}"
          f"{describe(win):<34}{describe(win_skip):<40}")

print("\nIf the two columns differ for wave 1, the published '18-32 weeks' range")
print("comes from the start-of-series spike.")


# =============================================================================
# Table S12 phase frequencies, with and without the first weeks
# =============================================================================

def phase_table(data: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"{'Wave':<6}{'Rising: median>1 / q025>1':<32}{'Falling: median>1 / q025>1':<32}"
          f"{'Median R0_H rising / falling'}")
    print("-" * 100)
    for w in range(3):
        lo, hi, pk = BOUNDARIES[w], BOUNDARIES[w + 1], MORTALITY_PEAKS[w]
        rise = data[(data["week_ending"] >= lo) & (data["week_ending"] <= pk)]
        fall = data[(data["week_ending"] > pk) & (data["week_ending"] <= hi)]

        def freq(sub):
            return (f"{100 * (sub['R0_H_median'] > 1).mean():5.1f} / "
                    f"{100 * (sub['R0_H_q025'] > 1).mean():5.1f} (n={len(sub)})")

        print(f"{w + 1:<6}{freq(rise):<32}{freq(fall):<32}"
              f"{rise['R0_H_median'].median():.3f} / {fall['R0_H_median'].median():.3f}")


phase_table(df, "Phase frequencies, all weeks (should reproduce Table S12):")
phase_table(df[df["week_ending"] >= first_kept],
            f"Phase frequencies, first {SKIP_FIRST_WEEKS} weeks left out:")