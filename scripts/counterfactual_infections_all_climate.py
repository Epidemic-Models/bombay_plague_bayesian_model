# =============================================================================
# counterfactual_infections_all_climate.py
# =============================================================================
"""
Counterfactual simulations for the all-climate beta_H model using existing
posterior draws.

This script does NOT refit the model.

For each posterior draw, it simulates:

    1. Full fitted model
       lambda(t) = beta_R(t) + beta_H(t) * I(t) / N

    2. Remove human-associated amplification (prevalence-associated term)
       human_multiplier = 0.0

    3. No recurrent zoonotic spillover after an initial seeding period
       spillover_multiplier = 0.0
       spillover_cutoff_week = SPILLOVER_CUTOFF_WEEK

For each scenario the script:

    - calculates weekly new exposures/infections;
    - stores weekly trajectories for every posterior draw;
    - calculates cumulative infections WITHIN each posterior draw;
    - calculates paired counterfactual reductions WITHIN each draw;
    - computes posterior medians and 95% credible intervals.

Outputs
-------
1. Weekly posterior trajectories:
   counterfactual_infections_all_climate_only.csv

2. One row per posterior draw:
   counterfactual_draw_level_totals.csv

3. Posterior summaries of cumulative effects:
   counterfactual_cumulative_posterior_summary.csv

4. Exact selected posterior parameter vectors (for full traceability):
   counterfactual_selected_posterior_draws.csv

IMPORTANT
---------
The cumulative posterior summaries are calculated by summing infections
within each posterior draw FIRST, and only then calculating posterior
quantiles. This is different from summing weekly posterior medians.

Reproducibility note
---------------------
Two distinct stochastic steps are involved: (1) converting dynesty's
weighted posterior sample cloud into an equal-weight posterior via
resample_equal, and (2) subsampling N_POSTERIOR_DRAWS draws from that
equal-weight posterior. These use two separate, independently specified
random seeds (RANDOM_SEED_RESAMPLE and RANDOM_SEED_DRAWS) rather than a
single shared seed, so that changing one step's sampling behaviour in the
future cannot silently change the other's.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dynesty.utils import resample_equal


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
BETAH_MODE = "all_climate"

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
)

FINAL_NESTED_DIR = (
    FINAL_ROOT
    / "nested_sampling"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

OUT_WEEKLY_CSV = (
    OUTPUT_DIR
    / "counterfactual_infections_all_climate_only.csv"
)

OUT_DRAW_CSV = (
    OUTPUT_DIR
    / "counterfactual_draw_level_totals.csv"
)

OUT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "counterfactual_cumulative_posterior_summary.csv"
)

OUT_SELECTED_DRAWS_CSV = (
    OUTPUT_DIR
    / "counterfactual_selected_posterior_draws.csv"
)


# -----------------------------------------------------------------------------
# Posterior settings
# -----------------------------------------------------------------------------

N_POSTERIOR_DRAWS = 1000

# Two separate, independently specified seeds for the two distinct
# stochastic steps below (previously a single RANDOM_SEED was reused for
# both, which works but conflates two independent sampling decisions).
RANDOM_SEED_RESAMPLE = 12345  # for converting weighted -> equal-weight posterior
RANDOM_SEED_DRAWS = 54321     # for subsampling N_POSTERIOR_DRAWS from that posterior


# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7

I0 = 0.0
E0 = 0.0
R0_INIT = 0.0

WEEK_RULE = "W-FRI"


# -----------------------------------------------------------------------------
# Counterfactual setting
# -----------------------------------------------------------------------------

# Spillover is retained during the initial four model weeks.
# After this point, beta_R is switched off in the no-recurrent-spillover
# counterfactual.
SPILLOVER_CUTOFF_WEEK = 4.0


# =============================================================================
# Helper functions
# =============================================================================

def load_final_result(betaH_mode: str) -> dict:
    """
    Load the saved final_v2 nested-sampling result.
    """

    pkl_path = (
        FINAL_NESTED_DIR
        / f"{betaH_mode}__{FIT_TARGET}__final_v2.pkl"
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find final_v2 result for "
            f"{betaH_mode!r}:\n"
            f"{pkl_path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    return saved


def get_equal_weight_posterior(
    saved: dict,
    seed: int,
) -> np.ndarray:
    """
    Convert dynesty's weighted posterior sample cloud into
    equal-weight posterior samples, using a seeded random state
    so this step is reproducible across runs.
    """

    results = saved["results"]

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    rstate = np.random.default_rng(seed)

    posterior = resample_equal(
        results.samples,
        weights,
        rstate=rstate,
    )

    return posterior


def weekly_sum_from_sim(
    sim: pd.DataFrame,
    week_ending_dates: pd.Series,
    col: str = "new_exposed",
) -> np.ndarray:
    """
    Aggregate a simulation flow column to weekly sums and align
    the result with the prepared epidemiological week grid.

    For new_exposed, the values returned are weekly numbers of
    newly exposed/infected individuals.
    """

    df = sim.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    wk = (
        df
        .set_index("date")[col]
        .resample(WEEK_RULE)
        .sum()
        .reset_index()
        .rename(
            columns={
                col: f"{col}_week",
                "date": "week_ending",
            }
        )
    )

    target = pd.DataFrame({
        "week_ending":
            pd.to_datetime(week_ending_dates)
    })

    aligned = pd.merge(
        target,
        wk,
        on="week_ending",
        how="left",
    )

    values = aligned[
        f"{col}_week"
    ].to_numpy(dtype=float)

    return values


def posterior_quantiles(
    arr: np.ndarray,
):
    """
    Calculate posterior 2.5%, 50%, and 97.5% quantiles
    along the posterior-draw dimension.
    """

    q025 = np.nanquantile(
        arr,
        0.025,
        axis=0,
    )

    median = np.nanquantile(
        arr,
        0.50,
        axis=0,
    )

    q975 = np.nanquantile(
        arr,
        0.975,
        axis=0,
    )

    return q025, median, q975


def summarize_vector(
    x: np.ndarray,
) -> dict:
    """
    Return posterior median and 95% credible interval
    for a one-dimensional posterior quantity.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    return {
        "q025":
            np.nanquantile(x, 0.025),

        "median":
            np.nanquantile(x, 0.50),

        "q975":
            np.nanquantile(x, 0.975),
    }


# =============================================================================
# Main analysis
# =============================================================================

def main():

    # =========================================================================
    # 1. Load prepared epidemiological and climate data
    # =========================================================================

    if not paths.MERGED_CSV.exists():

        raise FileNotFoundError(
            f"Prepared data not found:\n"
            f"{paths.MERGED_CSV}\n\n"
            "Run scripts/prepare_data.py first."
        )

    merged_df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[paths.DATE_COL],
    )

    merged_df = (
        merged_df
        .sort_values(paths.DATE_COL)
        .reset_index(drop=True)
    )


    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    required_columns = [
        paths.DATE_COL,
        "zT",
        "ze",
        "zR",
        FIT_TARGET,
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in merged_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns in prepared data: "
            + ", ".join(missing_columns)
        )


    # -------------------------------------------------------------------------
    # Extract data
    # -------------------------------------------------------------------------

    week_ending = pd.to_datetime(
        merged_df[paths.DATE_COL]
    )

    zT = merged_df[
        "zT"
    ].to_numpy(dtype=float)

    ze = merged_df[
        "ze"
    ].to_numpy(dtype=float)

    zR = merged_df[
        "zR"
    ].to_numpy(dtype=float)

    observed = merged_df[
        FIT_TARGET
    ].to_numpy(dtype=float)

    start_date = str(
        pd.Timestamp(
            week_ending.iloc[0]
        ).date()
    )

    n_weeks = len(
        merged_df
    )


    # -------------------------------------------------------------------------
    # Print basic information
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COUNTERFACTUAL INFECTIONS — "
        "ALL-CLIMATE MODEL (beta_H)"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFitted weeks: {n_weeks}"
    )

    print(
        f"Study period: "
        f"{week_ending.min().date()} "
        f"to "
        f"{week_ending.max().date()}"
    )

    print(
        f"Spillover cutoff week: "
        f"{SPILLOVER_CUTOFF_WEEK:g}"
    )


    # =========================================================================
    # 2. Load posterior
    # =========================================================================

    saved = load_final_result(
        BETAH_MODE
    )

    param_names = list(
        saved["param_names"]
    )

    posterior_samples = (
        get_equal_weight_posterior(
            saved,
            seed=RANDOM_SEED_RESAMPLE,
        )
    )

    n_available = len(
        posterior_samples
    )

    n_draws = min(
        N_POSTERIOR_DRAWS,
        n_available,
    )


    # -------------------------------------------------------------------------
    # Reproducible posterior subsample
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED_DRAWS
    )

    draw_idx = rng.choice(
        n_available,
        size=n_draws,
        replace=False,
    )

    draws = posterior_samples[
        draw_idx
    ]


    print(
        f"\nEqual-weight posterior samples available: "
        f"{n_available}"
    )

    print(
        f"Posterior draws used: "
        f"{n_draws}"
    )


    # -------------------------------------------------------------------------
    # Save the exact selected posterior parameter vectors, for full
    # traceability. counterfactual_draw_level_totals.csv (saved later)
    # only records "posterior_sample_index", i.e. which row of the
    # equal-weight resample was used -- this file records the equal-weight
    # sample's actual parameter values themselves.
    # -------------------------------------------------------------------------

    selected_draws_df = pd.DataFrame(
        draws,
        columns=param_names,
    )

    selected_draws_df.insert(
        0,
        "equal_weight_sample_index",
        draw_idx,
    )

    selected_draws_df.insert(
        0,
        "posterior_draw",
        np.arange(1, n_draws + 1),
    )

    selected_draws_df.to_csv(
        OUT_SELECTED_DRAWS_CSV,
        index=False,
    )

    print(
        f"[OK] Saved exact posterior draws: "
        f"{OUT_SELECTED_DRAWS_CSV}"
    )


    # =========================================================================
    # 3. Allocate weekly trajectory arrays
    # =========================================================================

    full_C = np.full(
        (n_draws, n_weeks),
        np.nan,
        dtype=float,
    )

    no_human_C = np.full(
        (n_draws, n_weeks),
        np.nan,
        dtype=float,
    )

    no_spill_C = np.full(
        (n_draws, n_weeks),
        np.nan,
        dtype=float,
    )


    # =========================================================================
    # 4. Counterfactual simulations
    # =========================================================================

    print(
        "\nRunning posterior counterfactual simulations..."
    )


    for i, theta in enumerate(draws):

        # ---------------------------------------------------------------------
        # Convert posterior parameter vector into dictionary
        # ---------------------------------------------------------------------

        sample = dict(
            zip(
                param_names,
                np.asarray(
                    theta,
                    dtype=float,
                ),
            )
        )


        # ---------------------------------------------------------------------
        # Build simulation parameters
        # ---------------------------------------------------------------------

        sim_params = build_sim_params(
            sample,
            BETAH_MODE,
        )


        # =====================================================================
        # Scenario 1: Full fitted model
        # =====================================================================

        sim_full = simulate_seir_dde(
            zT=zT,
            ze=ze,
            zR=zR,

            N=sample["N0"],

            params=sim_params,

            substeps_per_week=
                SUBSTEPS_PER_WEEK,

            tau0_days=
                TAU0_DAYS,

            I0=I0,
            E0=E0,
            R0=R0_INIT,

            C_seed=
                sample["C_seed"],

            betaH_mode=
                BETAH_MODE,

            start_date=
                start_date,
        )


        full_C[i, :] = (
            weekly_sum_from_sim(
                sim_full,
                week_ending,
                col="new_exposed",
            )
        )


        # =====================================================================
        # Scenario 2: Remove human-associated amplification
        # (the prevalence-associated beta_H(t) * I(t) / N term)
        # =====================================================================

        sim_no_human = simulate_seir_dde(
            zT=zT,
            ze=ze,
            zR=zR,

            N=sample["N0"],

            params=sim_params,

            substeps_per_week=
                SUBSTEPS_PER_WEEK,

            tau0_days=
                TAU0_DAYS,

            I0=I0,
            E0=E0,
            R0=R0_INIT,

            C_seed=
                sample["C_seed"],

            betaH_mode=
                BETAH_MODE,

            start_date=
                start_date,

            human_multiplier=0.0,
        )


        no_human_C[i, :] = (
            weekly_sum_from_sim(
                sim_no_human,
                week_ending,
                col="new_exposed",
            )
        )


        # =====================================================================
        # Scenario 3: Remove recurrent zoonotic spillover
        #
        # Spillover is retained during the initial seeding period and
        # switched off after SPILLOVER_CUTOFF_WEEK.
        # =====================================================================

        sim_no_spill = simulate_seir_dde(
            zT=zT,
            ze=ze,
            zR=zR,

            N=sample["N0"],

            params=sim_params,

            substeps_per_week=
                SUBSTEPS_PER_WEEK,

            tau0_days=
                TAU0_DAYS,

            I0=I0,
            E0=E0,
            R0=R0_INIT,

            C_seed=
                sample["C_seed"],

            betaH_mode=
                BETAH_MODE,

            start_date=
                start_date,

            spillover_multiplier=0.0,

            spillover_cutoff_week=
                SPILLOVER_CUTOFF_WEEK,
        )


        no_spill_C[i, :] = (
            weekly_sum_from_sim(
                sim_no_spill,
                week_ending,
                col="new_exposed",
            )
        )


        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            (i + 1) % 50 == 0
            or
            (i + 1) == n_draws
        ):

            print(
                f"  processed "
                f"{i + 1} / {n_draws} draws"
            )


    # =========================================================================
    # 5. Validate simulations
    # =========================================================================

    for name, arr in [
        ("full", full_C),
        ("no_human", no_human_C),
        ("no_spillover", no_spill_C),
    ]:

        if np.isnan(arr).any():

            n_missing = int(
                np.isnan(arr).sum()
            )

            raise RuntimeError(
                f"{name} scenario contains "
                f"{n_missing} NaN values."
            )

        if not np.isfinite(arr).all():

            raise RuntimeError(
                f"{name} scenario contains "
                "non-finite values."
            )

        if (arr < -1e-10).any():

            raise RuntimeError(
                f"{name} scenario contains "
                "negative weekly infections."
            )


    # =========================================================================
    # 6. DRAW-LEVEL cumulative infections
    #
    # IMPORTANT:
    # Sum within each posterior draw FIRST.
    # =========================================================================

    full_total_by_draw = np.sum(
        full_C,
        axis=1,
    )

    no_human_total_by_draw = np.sum(
        no_human_C,
        axis=1,
    )

    no_spill_total_by_draw = np.sum(
        no_spill_C,
        axis=1,
    )


    # -------------------------------------------------------------------------
    # Full-model totals must be positive before calculating ratios
    # -------------------------------------------------------------------------

    if np.any(
        full_total_by_draw <= 0
    ):

        bad = np.where(
            full_total_by_draw <= 0
        )[0]

        raise RuntimeError(
            "At least one posterior draw produced "
            "a non-positive full-model cumulative "
            f"infection burden. Bad draw indices: {bad}"
        )


    # =========================================================================
    # 7. Paired counterfactual effects within each posterior draw
    # =========================================================================

    no_human_pct_full = (
        100.0
        * no_human_total_by_draw
        / full_total_by_draw
    )

    no_spill_pct_full = (
        100.0
        * no_spill_total_by_draw
        / full_total_by_draw
    )


    # -------------------------------------------------------------------------
    # Percent reduction in cumulative infections
    # -------------------------------------------------------------------------

    human_removal_reduction = (
        100.0
        * (
            1.0
            -
            no_human_total_by_draw
            / full_total_by_draw
        )
    )

    spillover_removal_reduction = (
        100.0
        * (
            1.0
            -
            no_spill_total_by_draw
            / full_total_by_draw
        )
    )


    # =========================================================================
    # 8. Save one row per posterior draw
    # =========================================================================

    draw_df = pd.DataFrame({

        "draw":
            np.arange(
                1,
                n_draws + 1,
            ),

        "posterior_sample_index":
            draw_idx,

        "full_total_infections":
            full_total_by_draw,

        "no_human_total_infections":
            no_human_total_by_draw,

        "no_spillover_total_infections":
            no_spill_total_by_draw,

        "no_human_percent_of_full":
            no_human_pct_full,

        "no_spillover_percent_of_full":
            no_spill_pct_full,

        "human_removal_reduction_percent":
            human_removal_reduction,

        "spillover_removal_reduction_percent":
            spillover_removal_reduction,
    })


    draw_df.to_csv(
        OUT_DRAW_CSV,
        index=False,
    )


    # =========================================================================
    # 9. Posterior summaries of cumulative quantities
    # =========================================================================

    quantities = {

        "full_total_infections":
            full_total_by_draw,

        "no_human_total_infections":
            no_human_total_by_draw,

        "no_spillover_total_infections":
            no_spill_total_by_draw,

        "no_human_percent_of_full":
            no_human_pct_full,

        "no_spillover_percent_of_full":
            no_spill_pct_full,

        "human_removal_reduction_percent":
            human_removal_reduction,

        "spillover_removal_reduction_percent":
            spillover_removal_reduction,
    }


    summary_rows = []


    for quantity, values in quantities.items():

        summary = summarize_vector(
            values
        )

        summary_rows.append({

            "quantity":
                quantity,

            "q025":
                summary["q025"],

            "median":
                summary["median"],

            "q975":
                summary["q975"],
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )


    # =========================================================================
    # 10. Weekly posterior trajectories
    # =========================================================================

    (
        full_lo,
        full_med,
        full_hi,
    ) = posterior_quantiles(
        full_C
    )


    (
        nh_lo,
        nh_med,
        nh_hi,
    ) = posterior_quantiles(
        no_human_C
    )


    (
        ns_lo,
        ns_med,
        ns_hi,
    ) = posterior_quantiles(
        no_spill_C
    )


    weekly_df = pd.DataFrame({

        "week_ending":
            week_ending,

        "observed_deaths":
            observed,

        # Full model
        "full_infections_q025":
            full_lo,

        "full_infections_median":
            full_med,

        "full_infections_q975":
            full_hi,

        # Remove human-associated amplification
        "no_human_infections_q025":
            nh_lo,

        "no_human_infections_median":
            nh_med,

        "no_human_infections_q975":
            nh_hi,

        # No recurrent spillover
        "no_spillover_infections_q025":
            ns_lo,

        "no_spillover_infections_median":
            ns_med,

        "no_spillover_infections_q975":
            ns_hi,
    })


    weekly_df.to_csv(
        OUT_WEEKLY_CSV,
        index=False,
    )


    # =========================================================================
    # 11. Print main scientific results
    # =========================================================================

    full_summary = summarize_vector(
        full_total_by_draw
    )

    nh_summary = summarize_vector(
        no_human_total_by_draw
    )

    ns_summary = summarize_vector(
        no_spill_total_by_draw
    )

    human_reduction_summary = summarize_vector(
        human_removal_reduction
    )

    spill_reduction_summary = summarize_vector(
        spillover_removal_reduction
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "CUMULATIVE COUNTERFACTUAL RESULTS"
    )

    print(
        "=" * 80
    )


    # -------------------------------------------------------------------------
    # Full model
    # -------------------------------------------------------------------------

    print(
        "\nFull model cumulative infections:"
    )

    print(
        f"  Median = "
        f"{full_summary['median']:,.1f}"
    )

    print(
        f"  95% CrI = "
        f"[{full_summary['q025']:,.1f}, "
        f"{full_summary['q975']:,.1f}]"
    )


    # -------------------------------------------------------------------------
    # Remove human-associated amplification
    # -------------------------------------------------------------------------

    print(
        "\nRemove human-associated amplification "
        "(no prevalence-associated amplification):"
    )

    print(
        f"  Cumulative infections = "
        f"{nh_summary['median']:,.1f} "
        f"(95% CrI: "
        f"{nh_summary['q025']:,.1f}–"
        f"{nh_summary['q975']:,.1f})"
    )

    print(
        f"  Reduction relative to full model = "
        f"{human_reduction_summary['median']:.2f}% "
        f"(95% CrI: "
        f"{human_reduction_summary['q025']:.2f}%–"
        f"{human_reduction_summary['q975']:.2f}%)"
    )


    # -------------------------------------------------------------------------
    # No recurrent spillover
    # -------------------------------------------------------------------------

    print(
        "\nNo recurrent spillover "
        f"after week {SPILLOVER_CUTOFF_WEEK:g}:"
    )

    print(
        f"  Cumulative infections = "
        f"{ns_summary['median']:,.1f} "
        f"(95% CrI: "
        f"{ns_summary['q025']:,.1f}–"
        f"{ns_summary['q975']:,.1f})"
    )

    print(
        f"  Reduction relative to full model = "
        f"{spill_reduction_summary['median']:.2f}% "
        f"(95% CrI: "
        f"{spill_reduction_summary['q025']:.2f}%–"
        f"{spill_reduction_summary['q975']:.2f}%)"
    )


    # =========================================================================
    # 12. Output paths
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 80
    )

    print(
        "\nWeekly posterior trajectories:"
    )

    print(
        f"  {OUT_WEEKLY_CSV}"
    )

    print(
        "\nDraw-level cumulative results:"
    )

    print(
        f"  {OUT_DRAW_CSV}"
    )

    print(
        "\nPosterior cumulative summary:"
    )

    print(
        f"  {OUT_SUMMARY_CSV}"
    )

    print(
        "\nSelected posterior parameter vectors:"
    )

    print(
        f"  {OUT_SELECTED_DRAWS_CSV}"
    )

    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()