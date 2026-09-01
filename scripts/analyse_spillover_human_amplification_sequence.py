# =============================================================================
# analyse_spillover_human_amplification_sequence.py
# =============================================================================
"""
Test the proposed temporal mechanism:

    spillover seeding
        ->
    temporary human amplification
        ->
    mortality growth

using posterior weekly summaries from the preferred all-climate model.

Analyses
--------
1. Calculate continuous log mortality growth:
       g(t) = log(D(t) + 1) - log(D(t-1) + 1)

2. Calculate lagged Spearman correlations for:

       spillover(t)  -> R0_H(t + lag)
       spillover(t)  -> human_FOI(t + lag)
       R0_H(t)       -> mortality_growth(t + lag)
       human_FOI(t)  -> mortality_growth(t + lag)

3. Also examine:
       spillover(t) -> mortality_growth(t + lag)

4. Produce:
       - CSV of lagged correlations
       - lag-correlation figure
       - four-panel temporal mechanism figure

Important
---------
This is a descriptive temporal analysis. It does NOT by itself establish
causality.

Input
-----
posterior_transmission_dynamics_all_climate.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# =============================================================================
# PATHS
# =============================================================================

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
    / "temporal_mechanism_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CORR_CSV = OUTPUT_DIR / "lagged_mechanism_correlations.csv"

OUTPUT_CORR_FIG = OUTPUT_DIR / "lagged_mechanism_correlations.png"

OUTPUT_TIME_FIG = OUTPUT_DIR / "temporal_mechanism_over_time.png"

OUTPUT_WEEKLY_CSV = OUTPUT_DIR / "weekly_mechanism_variables.csv"


# =============================================================================
# SETTINGS
# =============================================================================

MAX_LAG = 8

DEATHS_COL = "observed_deaths"


# =============================================================================
# FUNCTIONS
# =============================================================================

def lagged_spearman(
    x: pd.Series,
    y: pd.Series,
    lag: int,
):
    """
    Calculate Spearman correlation between:

        x(t)

    and

        y(t + lag)

    Positive lag therefore means x precedes y.

    Example:
        lag = 3

    tests whether x this week is associated with y three weeks later.
    """

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # y(t + lag), aligned onto row t
    y_future = y.shift(-lag)

    valid = x.notna() & y_future.notna()

    x_valid = x.loc[valid]
    y_valid = y_future.loc[valid]

    if len(x_valid) < 5:
        return np.nan, np.nan, len(x_valid)

    # Avoid problems if either variable has no variation
    if x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return np.nan, np.nan, len(x_valid)

    rho, p = spearmanr(x_valid, y_valid)

    return float(rho), float(p), len(x_valid)


def add_analysis(
    results: list,
    relationship: str,
    predictor_name: str,
    outcome_name: str,
    predictor: pd.Series,
    outcome: pd.Series,
    lag: int,
):
    """
    Run one lagged Spearman correlation and append to results.
    """

    rho, p, n = lagged_spearman(
        predictor,
        outcome,
        lag,
    )

    results.append(
        {
            "relationship": relationship,
            "predictor": predictor_name,
            "outcome": outcome_name,
            "lag_weeks": lag,
            "spearman_rho": rho,
            "p_value_descriptive": p,
            "n_pairs": n,
        }
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        parse_dates=["week_ending"],
    )

    df = (
        df
        .sort_values("week_ending")
        .reset_index(drop=True)
    )

    required_cols = {
        "week_ending",
        DEATHS_COL,
        "spillover_median",
        "human_median",
        "R0_H_median",
        "R0_H_q025",
        "R0_H_q975",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    print("=" * 100)
    print("TEMPORAL MECHANISM ANALYSIS")
    print("=" * 100)

    print(f"\nInput:")
    print(INPUT_CSV)

    print(f"\nNumber of weeks: {len(df)}")


    # =========================================================================
    # DERIVED VARIABLES
    # =========================================================================

    deaths = df[DEATHS_COL].astype(float)

    # -------------------------------------------------------------------------
    # Continuous epidemic growth
    #
    # log(D_t + 1) - log(D_(t-1) + 1)
    #
    # > 0 : deaths increasing
    # = 0 : unchanged
    # < 0 : deaths decreasing
    # -------------------------------------------------------------------------

    df["log_deaths"] = np.log1p(deaths)

    df["mortality_growth"] = (
        df["log_deaths"]
        -
        df["log_deaths"].shift(1)
    )

    # -------------------------------------------------------------------------
    # Total force of infection
    # -------------------------------------------------------------------------

    df["total_FOI_median"] = (
        df["spillover_median"]
        +
        df["human_median"]
    )

    # -------------------------------------------------------------------------
    # Human fraction
    # -------------------------------------------------------------------------

    df["human_fraction"] = np.where(
        df["total_FOI_median"] > 0,
        df["human_median"] / df["total_FOI_median"],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # R0 classifications
    # -------------------------------------------------------------------------

    df["R0_median_gt1"] = (
        df["R0_H_median"] > 1
    )

    df["R0_q025_gt1"] = (
        df["R0_H_q025"] > 1
    )


    # =========================================================================
    # LAGGED CORRELATION ANALYSIS
    # =========================================================================

    results = []

    for lag in range(MAX_LAG + 1):

        # ---------------------------------------------------------------------
        # 1. Spillover -> future R0_H
        # ---------------------------------------------------------------------

        add_analysis(
            results=results,
            relationship="spillover -> R0_H",
            predictor_name="spillover_median",
            outcome_name="R0_H_median",
            predictor=df["spillover_median"],
            outcome=df["R0_H_median"],
            lag=lag,
        )

        # ---------------------------------------------------------------------
        # 2. Spillover -> future human FOI
        # ---------------------------------------------------------------------

        add_analysis(
            results=results,
            relationship="spillover -> human_FOI",
            predictor_name="spillover_median",
            outcome_name="human_median",
            predictor=df["spillover_median"],
            outcome=df["human_median"],
            lag=lag,
        )

        # ---------------------------------------------------------------------
        # 3. R0_H -> future mortality growth
        # ---------------------------------------------------------------------

        add_analysis(
            results=results,
            relationship="R0_H -> mortality_growth",
            predictor_name="R0_H_median",
            outcome_name="mortality_growth",
            predictor=df["R0_H_median"],
            outcome=df["mortality_growth"],
            lag=lag,
        )

        # ---------------------------------------------------------------------
        # 4. Human FOI -> future mortality growth
        # ---------------------------------------------------------------------

        add_analysis(
            results=results,
            relationship="human_FOI -> mortality_growth",
            predictor_name="human_median",
            outcome_name="mortality_growth",
            predictor=df["human_median"],
            outcome=df["mortality_growth"],
            lag=lag,
        )

        # ---------------------------------------------------------------------
        # 5. Spillover -> future mortality growth
        # ---------------------------------------------------------------------

        add_analysis(
            results=results,
            relationship="spillover -> mortality_growth",
            predictor_name="spillover_median",
            outcome_name="mortality_growth",
            predictor=df["spillover_median"],
            outcome=df["mortality_growth"],
            lag=lag,
        )


    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CORR_CSV,
        index=False,
    )


    # =========================================================================
    # PRINT FULL RESULTS
    # =========================================================================

    print("\n" + "=" * 100)
    print("LAGGED SPEARMAN CORRELATIONS")
    print("=" * 100)

    relationships = results_df["relationship"].unique()

    for relationship in relationships:

        sub = results_df[
            results_df["relationship"] == relationship
        ].copy()

        print("\n")
        print(relationship)
        print("-" * 80)

        for _, row in sub.iterrows():

            print(
                f"Lag {int(row['lag_weeks']):2d}: "
                f"rho = {row['spearman_rho']: .3f}, "
                f"p = {row['p_value_descriptive']:.4g}, "
                f"n = {int(row['n_pairs'])}"
            )


    # =========================================================================
    # FIND STRONGEST POSITIVE ASSOCIATION FOR EACH RELATIONSHIP
    # =========================================================================

    print("\n" + "=" * 100)
    print("STRONGEST POSITIVE LAGGED ASSOCIATIONS")
    print("=" * 100)

    for relationship in relationships:

        sub = results_df[
            results_df["relationship"] == relationship
        ].dropna(
            subset=["spearman_rho"]
        )

        if len(sub) == 0:
            continue

        best = sub.loc[
            sub["spearman_rho"].idxmax()
        ]

        print(
            f"\n{relationship}"
        )

        print(
            f"  strongest positive association at "
            f"lag = {int(best['lag_weeks'])} week(s)"
        )

        print(
            f"  Spearman rho = "
            f"{best['spearman_rho']:.3f}"
        )

        print(
            f"  descriptive p = "
            f"{best['p_value_descriptive']:.4g}"
        )


    # =========================================================================
    # PARTICULARLY IMPORTANT TESTS
    # =========================================================================

    print("\n" + "=" * 100)
    print("INTERPRETATION TARGETS")
    print("=" * 100)

    print(
        """
The proposed mechanism would be supported descriptively if we see a
coherent ordering such as:

    spillover(t)
        -> future human FOI / R0_H
        -> future mortality growth

Specifically look for:

1. Positive spillover -> human_FOI correlations at short positive lags.

2. Positive R0_H -> mortality_growth correlations at biologically
   plausible positive lags.

3. Positive human_FOI -> mortality_growth correlations at plausible
   positive lags.

4. The peaks should form a coherent temporal sequence rather than
   appearing at arbitrary unrelated lags.

Do NOT interpret the smallest p-value across lags as a formal hypothesis
test. Weekly observations and neighbouring lags are temporally correlated,
and multiple lags are being examined.
"""
    )


    # =========================================================================
    # FIGURE 1:
    # LAGGED CORRELATIONS
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for relationship in relationships:

        sub = results_df[
            results_df["relationship"] == relationship
        ]

        ax.plot(
            sub["lag_weeks"],
            sub["spearman_rho"],
            marker="o",
            label=relationship,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Lag (weeks): predictor at t → outcome at t + lag"
    )

    ax.set_ylabel(
        "Spearman correlation"
    )

    ax.set_title(
        "Temporal ordering of spillover, human transmission, and mortality growth"
    )

    ax.set_xticks(
        range(MAX_LAG + 1)
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_CORR_FIG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # =========================================================================
    # FIGURE 2:
    # TEMPORAL MECHANISM
    # =========================================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        sharex=True,
    )

    # -------------------------------------------------------------------------
    # Panel A: observed deaths
    # -------------------------------------------------------------------------

    axes[0].plot(
        df["week_ending"],
        df[DEATHS_COL],
        linewidth=1.5,
    )

    axes[0].set_ylabel(
        "Observed deaths"
    )

    axes[0].set_title(
        "Observed plague mortality"
    )


    # -------------------------------------------------------------------------
    # Panel B: spillover FOI
    # -------------------------------------------------------------------------

    axes[1].plot(
        df["week_ending"],
        df["spillover_median"],
        linewidth=1.5,
    )

    axes[1].set_ylabel(
        "Spillover FOI"
    )

    axes[1].set_title(
        "Inferred zoonotic spillover"
    )


    # -------------------------------------------------------------------------
    # Panel C: human FOI
    # -------------------------------------------------------------------------

    axes[2].plot(
        df["week_ending"],
        df["human_median"],
        linewidth=1.5,
    )

    axes[2].set_ylabel(
        "Human FOI"
    )

    axes[2].set_title(
        "Inferred human transmission"
    )


    # -------------------------------------------------------------------------
    # Panel D: R0_H
    # -------------------------------------------------------------------------

    axes[3].fill_between(
        df["week_ending"],
        df["R0_H_q025"],
        df["R0_H_q975"],
        alpha=0.2,
    )

    axes[3].plot(
        df["week_ending"],
        df["R0_H_median"],
        linewidth=1.5,
    )

    axes[3].axhline(
        1,
        linestyle="--",
        linewidth=1,
    )

    axes[3].set_ylabel(
        r"$R_{0,H}$"
    )

    axes[3].set_xlabel(
        "Week"
    )

    axes[3].set_title(
        "Human reproduction number"
    )


    fig.tight_layout()

    fig.savefig(
        OUTPUT_TIME_FIG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # =========================================================================
    # SAVE WEEKLY DERIVED VARIABLES
    # =========================================================================

    weekly_cols = [
        "week_ending",
        DEATHS_COL,
        "mortality_growth",
        "spillover_median",
        "human_median",
        "total_FOI_median",
        "human_fraction",
        "R0_H_q025",
        "R0_H_median",
        "R0_H_q975",
        "R0_median_gt1",
        "R0_q025_gt1",
    ]

    df[weekly_cols].to_csv(
        OUTPUT_WEEKLY_CSV,
        index=False,
    )


    # =========================================================================
    # DONE
    # =========================================================================

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"\nLagged correlations:\n  {OUTPUT_CORR_CSV}"
    )

    print(
        f"\nCorrelation figure:\n  {OUTPUT_CORR_FIG}"
    )

    print(
        f"\nTemporal mechanism figure:\n  {OUTPUT_TIME_FIG}"
    )

    print(
        f"\nWeekly derived variables:\n  {OUTPUT_WEEKLY_CSV}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()