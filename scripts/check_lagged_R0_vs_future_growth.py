# =============================================================================
# check_lagged_R0_vs_future_growth.py
# =============================================================================
"""
Tests whether supercritical human transmission at week t is associated with
increasing observed deaths at week t + lag.

For each lag from 0 to MAX_LAG weeks, the script compares:

    P(future growth | R0_H_median > 1)
vs
    P(future growth | R0_H_median <= 1)

and also the stricter criterion:

    R0_H_q025 > 1

where the full 95% posterior credible interval is above 1.

Input
-----
posterior_transmission_dynamics_all_climate.csv

Expected columns
----------------
week_ending
observed_deaths
R0_H_median
R0_H_q025
spillover_median
human_median

Output
------
1. Printed lag-by-lag summary
2. CSV table with results
3. Plot of future-growth probability vs lag
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "posterior_transmission_dynamics_all_climate.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "lagged_R0_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "lagged_R0_vs_future_growth.csv"
OUTPUT_FIG = OUTPUT_DIR / "lagged_R0_vs_future_growth.png"


# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------

DEATHS_COL = "observed_deaths"

# Test same week through 6 weeks later
MAX_LAG = 6


def safe_prop(x: pd.Series) -> float:
    """Return proportion of True values, or NaN if empty."""
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def main():

    # ---------------------------------------------------------------------
    # Read input
    # ---------------------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    required_cols = {
        "week_ending",
        DEATHS_COL,
        "R0_H_median",
        "R0_H_q025",
        "spillover_median",
        "human_median",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------------------
    # Current-week R0 classifications
    # ---------------------------------------------------------------------

    df["R0_median_gt1"] = df["R0_H_median"] > 1.0

    # Strong posterior evidence:
    # even the lower 95% credible bound is above 1
    df["R0_lower_gt1"] = df["R0_H_q025"] > 1.0

    # Current FOI decomposition
    df["total_FOI_median"] = (
        df["spillover_median"] + df["human_median"]
    )

    df["human_fraction"] = np.where(
        df["total_FOI_median"] > 0,
        df["human_median"] / df["total_FOI_median"],
        np.nan,
    )

    # ---------------------------------------------------------------------
    # Lag analysis
    # ---------------------------------------------------------------------

    results = []

    print("=" * 100)
    print("Lagged R0_H versus future epidemic growth")
    print("=" * 100)

    for lag in range(MAX_LAG + 1):

        # -------------------------------------------------------------
        # Define deaths at future week t + lag
        #
        # Growth at future week means:
        #
        # D_(t+lag) > D_(t+lag-1)
        #
        # For lag = 0, this is the original same-week comparison.
        # -------------------------------------------------------------

        future_deaths = df[DEATHS_COL].shift(-lag)

        future_prev_deaths = df[DEATHS_COL].shift(-(lag - 1))

        # Special case for lag = 0:
        # compare D_t with D_(t-1)
        if lag == 0:
            future_prev_deaths = df[DEATHS_COL].shift(1)

        future_growth = future_deaths > future_prev_deaths

        valid = (
            future_deaths.notna()
            & future_prev_deaths.notna()
        )

        temp = df.loc[valid].copy()

        temp["future_growth"] = future_growth.loc[valid].astype(bool)

        # -------------------------------------------------------------
        # Median R0 criterion
        # -------------------------------------------------------------

        supercritical = temp["R0_median_gt1"]
        subcritical = ~temp["R0_median_gt1"]

        p_growth_super = safe_prop(
            temp.loc[supercritical, "future_growth"]
        )

        p_growth_sub = safe_prop(
            temp.loc[subcritical, "future_growth"]
        )

        risk_ratio_median = (
            p_growth_super / p_growth_sub
            if np.isfinite(p_growth_super)
            and np.isfinite(p_growth_sub)
            and p_growth_sub > 0
            else np.nan
        )

        risk_difference_median = (
            p_growth_super - p_growth_sub
            if np.isfinite(p_growth_super)
            and np.isfinite(p_growth_sub)
            else np.nan
        )

        # -------------------------------------------------------------
        # Strong-evidence R0 criterion
        # -------------------------------------------------------------

        strong_supercritical = temp["R0_lower_gt1"]
        not_strong_supercritical = ~temp["R0_lower_gt1"]

        p_growth_strong = safe_prop(
            temp.loc[
                strong_supercritical,
                "future_growth"
            ]
        )

        p_growth_not_strong = safe_prop(
            temp.loc[
                not_strong_supercritical,
                "future_growth"
            ]
        )

        risk_ratio_strong = (
            p_growth_strong / p_growth_not_strong
            if np.isfinite(p_growth_strong)
            and np.isfinite(p_growth_not_strong)
            and p_growth_not_strong > 0
            else np.nan
        )

        risk_difference_strong = (
            p_growth_strong - p_growth_not_strong
            if np.isfinite(p_growth_strong)
            and np.isfinite(p_growth_not_strong)
            else np.nan
        )

        # -------------------------------------------------------------
        # Human FOI fraction among weeks classified by R0
        # -------------------------------------------------------------

        mean_human_frac_super = temp.loc[
            supercritical,
            "human_fraction"
        ].mean()

        mean_human_frac_sub = temp.loc[
            subcritical,
            "human_fraction"
        ].mean()

        # -------------------------------------------------------------
        # Counts
        # -------------------------------------------------------------

        n_valid = len(temp)

        n_super = int(supercritical.sum())
        n_sub = int(subcritical.sum())

        n_strong = int(strong_supercritical.sum())

        n_growth_super = int(
            temp.loc[
                supercritical,
                "future_growth"
            ].sum()
        )

        n_growth_sub = int(
            temp.loc[
                subcritical,
                "future_growth"
            ].sum()
        )

        # -------------------------------------------------------------
        # Store results
        # -------------------------------------------------------------

        results.append(
            {
                "lag_weeks": lag,

                "n_valid": n_valid,

                "n_R0_median_gt1": n_super,
                "n_R0_median_le1": n_sub,

                "n_growth_given_R0_gt1": n_growth_super,
                "n_growth_given_R0_le1": n_growth_sub,

                "p_future_growth_given_R0_gt1": p_growth_super,
                "p_future_growth_given_R0_le1": p_growth_sub,

                "risk_difference_median": risk_difference_median,
                "risk_ratio_median": risk_ratio_median,

                "n_R0_q025_gt1": n_strong,

                "p_future_growth_given_R0_q025_gt1": p_growth_strong,
                "p_future_growth_given_R0_q025_le1": p_growth_not_strong,

                "risk_difference_strong": risk_difference_strong,
                "risk_ratio_strong": risk_ratio_strong,

                "mean_human_fraction_R0_gt1": mean_human_frac_super,
                "mean_human_fraction_R0_le1": mean_human_frac_sub,
            }
        )

        # -------------------------------------------------------------
        # Print readable output
        # -------------------------------------------------------------

        print(f"\nLag = {lag} week(s)")
        print("-" * 100)

        print(
            f"P(future growth | R0 median > 1)  = "
            f"{100 * p_growth_super:.1f}% "
            f"({n_growth_super}/{n_super})"
        )

        print(
            f"P(future growth | R0 median <= 1) = "
            f"{100 * p_growth_sub:.1f}% "
            f"({n_growth_sub}/{n_sub})"
        )

        print(
            f"Risk difference = "
            f"{100 * risk_difference_median:.1f} percentage points"
        )

        print(
            f"Risk ratio = "
            f"{risk_ratio_median:.2f}"
        )

        print(
            f"\nStrong evidence criterion:"
        )

        print(
            f"P(future growth | R0 q025 > 1) = "
            f"{100 * p_growth_strong:.1f}%"
        )

        print(
            f"P(future growth | R0 q025 <= 1) = "
            f"{100 * p_growth_not_strong:.1f}%"
        )

        print(
            f"Strong-evidence risk ratio = "
            f"{risk_ratio_strong:.2f}"
        )

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ---------------------------------------------------------------------
    # Identify lag with strongest median-R0 association
    # ---------------------------------------------------------------------

    valid_rr = results_df[
        np.isfinite(results_df["risk_ratio_median"])
    ]

    if len(valid_rr) > 0:

        best_row = valid_rr.loc[
            valid_rr["risk_ratio_median"].idxmax()
        ]

        print("\n" + "=" * 100)
        print("Strongest lagged association")
        print("=" * 100)

        print(
            f"Maximum risk ratio occurs at "
            f"{int(best_row['lag_weeks'])} week(s)."
        )

        print(
            f"Risk ratio = "
            f"{best_row['risk_ratio_median']:.2f}"
        )

        print(
            f"P(growth | R0 > 1) = "
            f"{100 * best_row['p_future_growth_given_R0_gt1']:.1f}%"
        )

        print(
            f"P(growth | R0 <= 1) = "
            f"{100 * best_row['p_future_growth_given_R0_le1']:.1f}%"
        )

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results_df["lag_weeks"],
        100 * results_df["p_future_growth_given_R0_gt1"],
        marker="o",
        label=r"$R_{0,H}$ median > 1",
    )

    ax.plot(
        results_df["lag_weeks"],
        100 * results_df["p_future_growth_given_R0_le1"],
        marker="o",
        label=r"$R_{0,H}$ median $\leq$ 1",
    )

    ax.set_xlabel("Lag from R0_H to mortality growth (weeks)")
    ax.set_ylabel("Probability of increasing deaths (%)")

    ax.set_title(
        "Does supercritical human transmission precede epidemic growth?"
    )

    ax.set_xticks(
        range(MAX_LAG + 1)
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_FIG}")

    print("\nDone.")


if __name__ == "__main__":
    main()