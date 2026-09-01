# =============================================================================
# check_R0_vs_growth_and_FOI_from_transmission_dynamics.py
# =============================================================================
"""
Uses the per-week posterior summaries written by final_transmission_dynamics.py
(posterior_transmission_dynamics_<betaH_mode>.csv) to check, for the
all-climate model:

  1. How often R0_H_median > 1 occurs during weeks with increasing deaths
     vs weeks without increase.
  2. How often "strong evidence" (R0_H_q025 > 1) occurs in growth vs
     non-growth weeks.
  3. During growth vs non-growth weeks, the mean fraction of total FOI
     attributable to the human component
         human_median / (spillover_median + human_median).

This matches the descriptive analysis in the manuscript.

Usage
-----
    python scripts/check_R0_vs_growth_and_FOI_from_transmission_dynamics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Adjust this if your project root is different.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default: use the all-climate posterior summary CSV produced by
# final_transmission_dynamics.py.
DEFAULT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "posterior_transmission_dynamics_all_climate.csv"
)

DEATHS_COL = "observed_deaths"


def main():
    csv_path = DEFAULT_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at:\n  {csv_path}\n\n"
            "Run scripts/final_transmission_dynamics.py first so that this file is created."
        )

    df = pd.read_csv(csv_path, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    required_cols = {
        "week_ending",
        DEATHS_COL,
        "spillover_median",
        "human_median",
        "R0_H_q025",
        "R0_H_median",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Check that {csv_path.name} really comes from final_transmission_dynamics.py."
        )

    # -------------------------------------------------------------------------
    # Define growth vs non-growth weeks based on deaths
    # -------------------------------------------------------------------------
    deaths = df[DEATHS_COL].astype(float)
    df["deaths_prev"] = deaths.shift(1)

    valid = df["deaths_prev"].notna()
    df_valid = df.loc[valid].copy()

    df_valid["growth_week"] = df_valid[DEATHS_COL] > df_valid["deaths_prev"]

    n_growth = int(df_valid["growth_week"].sum())
    n_non_growth = int((~df_valid["growth_week"]).sum())

    # -------------------------------------------------------------------------
    # R0_H median > 1
    # -------------------------------------------------------------------------
    df_valid["R0_median_gt1"] = df_valid["R0_H_median"] > 1.0

    growth_R0_median_gt1 = int(df_valid.loc[df_valid["growth_week"], "R0_median_gt1"].sum())
    non_growth_R0_median_gt1 = int(df_valid.loc[~df_valid["growth_week"], "R0_median_gt1"].sum())

    pct_growth_R0_median_gt1 = (
        100.0 * growth_R0_median_gt1 / n_growth if n_growth > 0 else np.nan
    )
    pct_non_growth_R0_median_gt1 = (
        100.0 * non_growth_R0_median_gt1 / n_non_growth if n_non_growth > 0 else np.nan
    )

    # -------------------------------------------------------------------------
    # Strong evidence: lower 95% credible bound > 1
    # -------------------------------------------------------------------------
    df_valid["R0_lower_gt1"] = df_valid["R0_H_q025"] > 1.0

    growth_R0_lower_gt1 = int(df_valid.loc[df_valid["growth_week"], "R0_lower_gt1"].sum())
    non_growth_R0_lower_gt1 = int(df_valid.loc[~df_valid["growth_week"], "R0_lower_gt1"].sum())

    pct_growth_R0_lower_gt1 = (
        100.0 * growth_R0_lower_gt1 / n_growth if n_growth > 0 else np.nan
    )
    pct_non_growth_R0_lower_gt1 = (
        100.0 * non_growth_R0_lower_gt1 / n_non_growth if n_non_growth > 0 else np.nan
    )

    # -------------------------------------------------------------------------
    # Human fraction of total FOI, by week
    # -------------------------------------------------------------------------
    df_valid["total_FOI_median"] = df_valid["spillover_median"] + df_valid["human_median"]
    df_valid = df_valid[df_valid["total_FOI_median"] > 0].copy()

    df_valid["human_fraction"] = df_valid["human_median"] / df_valid["total_FOI_median"]

    mean_human_frac_growth = float(
        df_valid.loc[df_valid["growth_week"], "human_fraction"].mean()
    )
    mean_human_frac_non_growth = float(
        df_valid.loc[~df_valid["growth_week"], "human_fraction"].mean()
    )

    # -------------------------------------------------------------------------
    # Print results in manuscript-like prose
    # -------------------------------------------------------------------------
    print("=" * 90)
    print(f"R0_H vs epidemic growth (from {csv_path})")
    print("=" * 90)

    print(
        f"\nWeeks classified (excluding first, which has no previous week): "
        f"{n_growth + n_non_growth}"
    )
    print(f"  Growth weeks (deaths_t > deaths_(t-1)): {n_growth}")
    print(f"  Non-growth weeks: {n_non_growth}")

    print("\n--- R0_H_median > 1 ---")
    print(
        f"During growth weeks, R0_H_median > 1 in "
        f"{growth_R0_median_gt1} of {n_growth} weeks "
        f"({pct_growth_R0_median_gt1:.1f}%)."
    )
    print(
        f"During non-growth weeks, R0_H_median > 1 in "
        f"{non_growth_R0_median_gt1} of {n_non_growth} weeks "
        f"({pct_non_growth_R0_median_gt1:.1f}%)."
    )

    print("\n--- Strong evidence: R0_H_q025 > 1 ---")
    print(
        f"During growth weeks, the lower 95% credible bound exceeds 1 in "
        f"{growth_R0_lower_gt1} of {n_growth} weeks "
        f"({pct_growth_R0_lower_gt1:.1f}%)."
    )
    print(
        f"During non-growth weeks, R0_H_q025 > 1 in "
        f"{non_growth_R0_lower_gt1} of {n_non_growth} weeks "
        f"({pct_non_growth_R0_lower_gt1:.1f}%)."
    )

    print("\n--- Human fraction of the total force of infection ---")
    print(
        f"Mean human fraction during growth weeks: {mean_human_frac_growth * 100:.1f}%"
    )
    print(
        f"Mean human fraction during non-growth weeks: {mean_human_frac_non_growth * 100:.1f}%"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
