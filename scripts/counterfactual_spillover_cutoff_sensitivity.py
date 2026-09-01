# =============================================================================
# counterfactual_spillover_cutoff_sensitivity.py
# =============================================================================
"""
Sensitivity analysis for the duration of initial zoonotic spillover.

This script uses the existing posterior from the fitted all-climate model
and DOES NOT refit anything.

For each posterior draw, it:

    1. Simulates the full fitted model.
    2. Re-simulates the model after removing recurrent zoonotic spillover
       following several alternative cutoff times.

Primary sensitivity cutoffs:
    0, 2, 4, 8, and 12 model weeks.

For every posterior draw and cutoff, cumulative infections are calculated
first. The percentage reduction relative to the corresponding full-model
simulation is then calculated WITHIN the same posterior draw.

Finally, posterior medians and 95% credible intervals are calculated across
draws.

Outputs
-------
results/final_figures/transmission_dynamics/
    spillover_cutoff_sensitivity/

        spillover_cutoff_draw_level.csv
        spillover_cutoff_posterior_summary.csv
        spillover_cutoff_weekly_medians.csv
        spillover_cutoff_selected_posterior_draws.csv

        spillover_cutoff_reduction.png
        spillover_cutoff_reduction.pdf

        spillover_cutoff_trajectories.png
        spillover_cutoff_trajectories.pdf

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    / "spillover_cutoff_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUT_DRAW_CSV = (
    OUTPUT_DIR
    / "spillover_cutoff_draw_level.csv"
)

OUT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "spillover_cutoff_posterior_summary.csv"
)

OUT_WEEKLY_CSV = (
    OUTPUT_DIR
    / "spillover_cutoff_weekly_medians.csv"
)

OUT_SELECTED_DRAWS_CSV = (
    OUTPUT_DIR
    / "spillover_cutoff_selected_posterior_draws.csv"
)

OUT_REDUCTION_PNG = (
    OUTPUT_DIR
    / "spillover_cutoff_reduction.png"
)

OUT_REDUCTION_PDF = (
    OUTPUT_DIR
    / "spillover_cutoff_reduction.pdf"
)

OUT_TRAJECTORY_PNG = (
    OUTPUT_DIR
    / "spillover_cutoff_trajectories.png"
)

OUT_TRAJECTORY_PDF = (
    OUTPUT_DIR
    / "spillover_cutoff_trajectories.pdf"
)


# -----------------------------------------------------------------------------
# Posterior sampling
# -----------------------------------------------------------------------------

N_POSTERIOR_DRAWS = 1000

# Two separate, independently specified seeds for the two distinct
# stochastic steps below (previously a single RANDOM_SEED was reused for
# both, which works but conflates two independent sampling decisions).
RANDOM_SEED_RESAMPLE = 12345  # for converting weighted -> equal-weight posterior
RANDOM_SEED_DRAWS = 54321     # for subsampling N_POSTERIOR_DRAWS from that posterior


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7

I0 = 0.0
E0 = 0.0
R0_INIT = 0.0

WEEK_RULE = "W-FRI"


# -----------------------------------------------------------------------------
# Sensitivity cutoffs
# -----------------------------------------------------------------------------

SPILLOVER_CUTOFF_WEEKS = [
    0.0,
    2.0,
    4.0,
    8.0,
    12.0,
]


# =============================================================================
# Plot settings
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


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
            f"Could not find final_v2 result:\n"
            f"{pkl_path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_equal_weight_posterior(
    saved: dict,
    seed: int,
) -> np.ndarray:
    """
    Convert dynesty weighted samples to an equal-weight posterior sample,
    using a seeded random state so this step is reproducible across runs.
    """

    results = saved["results"]

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    rstate = np.random.default_rng(seed)

    return resample_equal(
        results.samples,
        weights,
        rstate=rstate,
    )

def weekly_sum_from_sim(
    sim: pd.DataFrame,
    week_ending_dates: pd.Series,
    col: str = "new_exposed",
) -> np.ndarray:
    """
    Aggregate simulated new exposures to weekly totals aligned to
    the prepared epidemiological week grid.
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
                "date": "week_ending",
                col: f"{col}_week",
            }
        )
    )

    target = pd.DataFrame({
        "week_ending":
            pd.to_datetime(
                week_ending_dates
            )
    })

    aligned = pd.merge(
        target,
        wk,
        on="week_ending",
        how="left",
    )

    return aligned[
        f"{col}_week"
    ].to_numpy(dtype=float)


def summarize_vector(
    x: np.ndarray,
) -> dict:
    """
    Posterior median and 95% credible interval.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    return {
        "q025":
            np.nanquantile(
                x,
                0.025,
            ),

        "median":
            np.nanquantile(
                x,
                0.50,
            ),

        "q975":
            np.nanquantile(
                x,
                0.975,
            ),
    }


def posterior_quantiles(
    arr: np.ndarray,
):
    """
    Weekly posterior 2.5%, 50%, 97.5% quantiles.
    """

    return (
        np.nanquantile(
            arr,
            0.025,
            axis=0,
        ),

        np.nanquantile(
            arr,
            0.50,
            axis=0,
        ),

        np.nanquantile(
            arr,
            0.975,
            axis=0,
        ),
    )


# =============================================================================
# Main analysis
# =============================================================================

def main():

    # =========================================================================
    # 1. Load data
    # =========================================================================

    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Prepared data not found:\n"
            f"{paths.MERGED_CSV}"
        )

    merged_df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[
            paths.DATE_COL
        ],
    )

    merged_df = (
        merged_df
        .sort_values(
            paths.DATE_COL
        )
        .reset_index(
            drop=True
        )
    )


    required_cols = [
        paths.DATE_COL,
        "zT",
        "ze",
        "zR",
        FIT_TARGET,
    ]

    missing = [
        c
        for c in required_cols
        if c not in merged_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )


    week_ending = pd.to_datetime(
        merged_df[
            paths.DATE_COL
        ]
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

    observed_deaths = merged_df[
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


    print(
        "\n"
        + "=" * 90
    )

    print(
        "SPILLOVER CUTOFF SENSITIVITY — ALL-CLIMATE MODEL"
    )

    print(
        "=" * 90
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
        "Cutoffs tested: "
        + ", ".join(
            f"{c:g}"
            for c
            in SPILLOVER_CUTOFF_WEEKS
        )
        + " weeks"
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
        f"\nPosterior draws used: "
        f"{n_draws} / {n_available}"
    )


    # -------------------------------------------------------------------------
    # Save the exact selected posterior parameter vectors, for full
    # traceability. spillover_cutoff_draw_level.csv (saved later) only
    # records "posterior_sample_index", i.e. which row of the equal-weight
    # resample was used -- this file records the equal-weight sample's
    # actual parameter values themselves.
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
    # 3. Allocate arrays
    # =========================================================================

    full_C = np.full(
        (
            n_draws,
            n_weeks,
        ),
        np.nan,
        dtype=float,
    )


    cutoff_arrays = {
        cutoff:
            np.full(
                (
                    n_draws,
                    n_weeks,
                ),
                np.nan,
                dtype=float,
            )

        for cutoff
        in SPILLOVER_CUTOFF_WEEKS
    }


    # =========================================================================
    # 4. Run simulations
    # =========================================================================

    print(
        "\nRunning simulations..."
    )


    for i, theta in enumerate(draws):

        sample = dict(
            zip(
                param_names,
                np.asarray(
                    theta,
                    dtype=float,
                ),
            )
        )


        sim_params = build_sim_params(
            sample,
            BETAH_MODE,
        )


        # =====================================================================
        # Full model
        # =====================================================================

        sim_full = simulate_seir_dde(
            zT=zT,
            ze=ze,
            zR=zR,

            N=
                sample["N0"],

            params=
                sim_params,

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


        full_C[
            i,
            :
        ] = weekly_sum_from_sim(
            sim_full,
            week_ending,
            col="new_exposed",
        )


        # =====================================================================
        # Spillover-cutoff scenarios
        # =====================================================================

        for cutoff in SPILLOVER_CUTOFF_WEEKS:

            sim_cf = simulate_seir_dde(
                zT=zT,
                ze=ze,
                zR=zR,

                N=
                    sample["N0"],

                params=
                    sim_params,

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

                spillover_multiplier=
                    0.0,

                spillover_cutoff_week=
                    cutoff,
            )


            cutoff_arrays[
                cutoff
            ][
                i,
                :
            ] = weekly_sum_from_sim(
                sim_cf,
                week_ending,
                col="new_exposed",
            )


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
    # 5. Validate arrays
    # =========================================================================

    if np.isnan(
        full_C
    ).any():

        raise RuntimeError(
            "Full-model array contains NaNs."
        )


    if not np.isfinite(
        full_C
    ).all():

        raise RuntimeError(
            "Full-model array contains "
            "non-finite values."
        )


    if (
        full_C < -1e-10
    ).any():

        raise RuntimeError(
            "Full-model array contains "
            "negative weekly infections."
        )


    for cutoff, arr in cutoff_arrays.items():

        if np.isnan(
            arr
        ).any():

            raise RuntimeError(
                f"Cutoff {cutoff:g} "
                "contains NaNs."
            )

        if not np.isfinite(
            arr
        ).all():

            raise RuntimeError(
                f"Cutoff {cutoff:g} "
                "contains non-finite values."
            )

        if (
            arr < -1e-10
        ).any():

            raise RuntimeError(
                f"Cutoff {cutoff:g} "
                "contains negative weekly infections."
            )


    # =========================================================================
    # 6. Full-model cumulative infections
    # =========================================================================

    full_total_by_draw = np.sum(
        full_C,
        axis=1,
    )


    if np.any(
        full_total_by_draw <= 0
    ):

        raise RuntimeError(
            "At least one full-model posterior draw "
            "has non-positive cumulative infections."
        )


    # =========================================================================
    # 7. Draw-level sensitivity results
    # =========================================================================

    draw_rows = []


    for i in range(
        n_draws
    ):

        full_total = (
            full_total_by_draw[
                i
            ]
        )


        for cutoff in SPILLOVER_CUTOFF_WEEKS:

            cf_total = np.sum(
                cutoff_arrays[
                    cutoff
                ][
                    i,
                    :
                ]
            )


            pct_full = (
                100.0
                * cf_total
                / full_total
            )


            reduction = (
                100.0
                * (
                    1.0
                    -
                    cf_total
                    / full_total
                )
            )


            draw_rows.append({

                "draw":
                    i + 1,

                "posterior_sample_index":
                    int(
                        draw_idx[
                            i
                        ]
                    ),

                "spillover_cutoff_week":
                    cutoff,

                "full_total_infections":
                    full_total,

                "counterfactual_total_infections":
                    cf_total,

                "counterfactual_percent_of_full":
                    pct_full,

                "reduction_percent":
                    reduction,
            })


    draw_df = pd.DataFrame(
        draw_rows
    )


    draw_df.to_csv(
        OUT_DRAW_CSV,
        index=False,
    )


    # =========================================================================
    # 8. Posterior summary by cutoff
    # =========================================================================

    summary_rows = []


    for cutoff in SPILLOVER_CUTOFF_WEEKS:

        block = draw_df[
            draw_df[
                "spillover_cutoff_week"
            ]
            == cutoff
        ]


        total_summary = summarize_vector(
            block[
                "counterfactual_total_infections"
            ].to_numpy()
        )


        pct_summary = summarize_vector(
            block[
                "counterfactual_percent_of_full"
            ].to_numpy()
        )


        reduction_summary = summarize_vector(
            block[
                "reduction_percent"
            ].to_numpy()
        )


        summary_rows.append({

            "spillover_cutoff_week":
                cutoff,

            "counterfactual_total_q025":
                total_summary["q025"],

            "counterfactual_total_median":
                total_summary["median"],

            "counterfactual_total_q975":
                total_summary["q975"],

            "percent_of_full_q025":
                pct_summary["q025"],

            "percent_of_full_median":
                pct_summary["median"],

            "percent_of_full_q975":
                pct_summary["q975"],

            "reduction_q025":
                reduction_summary["q025"],

            "reduction_median":
                reduction_summary["median"],

            "reduction_q975":
                reduction_summary["q975"],
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )


    # =========================================================================
    # 9. Weekly posterior medians for plotting
    # =========================================================================

    weekly_out = pd.DataFrame({

        "week_ending":
            week_ending,

        "observed_deaths":
            observed_deaths,

        "full_infections_median":
            np.nanmedian(
                full_C,
                axis=0,
            ),
    })


    for cutoff in SPILLOVER_CUTOFF_WEEKS:

        _, med, _ = posterior_quantiles(
            cutoff_arrays[
                cutoff
            ]
        )


        label = (
            f"cutoff_{cutoff:g}_weeks_median"
            .replace(
                ".",
                "p",
            )
        )


        weekly_out[
            label
        ] = med


    weekly_out.to_csv(
        OUT_WEEKLY_CSV,
        index=False,
    )


    # =========================================================================
    # 10. Figure: percentage reduction vs cutoff
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(
            6.5,
            4.5,
        )
    )


    x = summary_df[
        "spillover_cutoff_week"
    ].to_numpy(dtype=float)

    y = summary_df[
        "reduction_median"
    ].to_numpy(dtype=float)

    lo = summary_df[
        "reduction_q025"
    ].to_numpy(dtype=float)

    hi = summary_df[
        "reduction_q975"
    ].to_numpy(dtype=float)


    yerr = np.vstack([
        y - lo,
        hi - y,
    ])


    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=3,
        linewidth=1.5,
    )


    ax.set_xlabel(
        "Duration of initial spillover "
        "before removal (weeks)"
    )

    ax.set_ylabel(
        "Reduction in cumulative infections (%)"
    )

    ax.set_xticks(
        SPILLOVER_CUTOFF_WEEKS
    )

    ax.set_ylim(
        0,
        101,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.set_title(
        "Sensitivity of epidemic size to "
        "spillover cutoff"
    )


    fig.tight_layout()


    fig.savefig(
        OUT_REDUCTION_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_REDUCTION_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # =========================================================================
    # 11. Figure: weekly trajectories
    # =========================================================================

    fig, ax1 = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )


    # Full model
    ax1.plot(
        week_ending,
        np.nanmedian(
            full_C,
            axis=0,
        ),
        linewidth=2.3,
        label="Full model",
    )


    # Counterfactuals
    for cutoff in SPILLOVER_CUTOFF_WEEKS:

        med = np.nanmedian(
            cutoff_arrays[
                cutoff
            ],
            axis=0,
        )


        ax1.plot(
            week_ending,
            med,
            linewidth=1.4,
            label=(
                f"Spillover removed "
                f"after {cutoff:g} weeks"
            ),
        )


    ax1.set_xlabel(
        "Week ending"
    )

    ax1.set_ylabel(
        "Weekly modelled infections"
    )


    ax1.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=4
        )
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m"
        )
    )


    ax1.spines[
        "top"
    ].set_visible(
        False
    )


    # Observed deaths
    ax2 = ax1.twinx()


    ax2.scatter(
        week_ending,
        observed_deaths,
        s=10,
        alpha=0.25,
        label="Observed plague deaths",
    )


    ax2.set_ylabel(
        "Observed weekly plague deaths"
    )

    ax2.spines[
        "top"
    ].set_visible(
        False
    )


    # Combined legend
    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )


    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper left",
    )


    ax1.set_title(
        "Spillover-cutoff sensitivity "
        "of counterfactual epidemic trajectories"
    )


    fig.tight_layout()


    fig.savefig(
        OUT_TRAJECTORY_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_TRAJECTORY_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # =========================================================================
    # 12. Print summary
    # =========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "POSTERIOR SPILLOVER-CUTOFF SENSITIVITY"
    )

    print(
        "=" * 90
    )


    display = summary_df[
        [
            "spillover_cutoff_week",
            "counterfactual_total_median",
            "counterfactual_total_q025",
            "counterfactual_total_q975",
            "reduction_median",
            "reduction_q025",
            "reduction_q975",
        ]
    ].copy()


    print(
        "\n"
        + display.to_string(
            index=False,
            formatters={

                "spillover_cutoff_week":
                    "{:.0f}".format,

                "counterfactual_total_median":
                    "{:,.1f}".format,

                "counterfactual_total_q025":
                    "{:,.1f}".format,

                "counterfactual_total_q975":
                    "{:,.1f}".format,

                "reduction_median":
                    "{:.2f}".format,

                "reduction_q025":
                    "{:.2f}".format,

                "reduction_q975":
                    "{:.2f}".format,
            },
        )
    )


    print(
        "\nInterpretation:"
    )

    print(
        "Each row compares a counterfactual "
        "spillover-cutoff simulation with the "
        "corresponding full-model simulation "
        "within the same posterior draw."
    )

    print(
        "A consistently large reduction across "
        "cutoffs would indicate that the inferred "
        "dependence on recurrent zoonotic spillover "
        "is not specific to the four-week cutoff."
    )


    print(
        "\nSaved:"
    )

    print(
        f"  {OUT_DRAW_CSV}"
    )

    print(
        f"  {OUT_SUMMARY_CSV}"
    )

    print(
        f"  {OUT_WEEKLY_CSV}"
    )

    print(
        f"  {OUT_SELECTED_DRAWS_CSV}"
    )

    print(
        f"  {OUT_REDUCTION_PNG}"
    )

    print(
        f"  {OUT_REDUCTION_PDF}"
    )

    print(
        f"  {OUT_TRAJECTORY_PNG}"
    )

    print(
        f"  {OUT_TRAJECTORY_PDF}"
    )

    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()