# =============================================================================
# make_bayesian_model_fits_figure.py
# =============================================================================
"""
Consolidated Bayesian posterior model-fit figure and diagnostics.

This script reads the CORRECT final_v2 posteriors from

    results/final_model_comparison_v2/nested_sampling/

and does NOT refit the model.

For each beta_H formulation, it calculates:

    1. Posterior expected-death trajectory:
       - reproducibly resample dynesty's weighted posterior;
       - reproducibly select N_POSTERIOR_DRAWS parameter vectors;
       - simulate the expected-death trajectory for each draw;
       - calculate the weekly posterior median and 95% credible interval.

    2. Highest-likelihood trajectory:
       - use the sampled theta with the largest stored log-likelihood.

    3. Posterior-median-parameter trajectory:
       - replace each parameter by its marginal posterior median;
       - simulate once.
       This is a diagnostic trajectory, not the posterior median trajectory.

The script also saves the EXACT posterior parameter vectors used to construct
the credible bands. This makes the downstream figure reproducible even if a
future dynesty version changes the implementation of resample_equal().

IMPORTANT
---------
The 95% band shown here is uncertainty in the latent expected-death trajectory
mu(t). It is NOT the full posterior-predictive interval.

The posterior-predictive interval would additionally include observation
noise from the negative-binomial observation model and would therefore be
wider.

Outputs
-------
results/final_figures/bayesian_model_fits/

    bayesian_model_fits_grid.png
    bayesian_model_fits_grid.pdf

    bayesian_model_fits_<model>.csv

    selected_model_fit_draws_<model>.csv

    mle_vs_median_divergence.csv

Usage
-----
    python scripts/make_bayesian_model_fits_figure.py
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
from mechanistic_model import simulate_seir_dde, weekly_sum  # noqa: E402
from observation import make_delay_pmf_weeks, deaths_from_infections_fast  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
RAW_TARGET = "plague_deaths_reported"

OUTLIER_WEEK = "1897-02-19"

MODELS = [
    "constant",
    "humidity_only",
    "all_climate",
    "sigmoid",
]

MODEL_LABELS = {
    "constant":
        r"(a) Constant $\beta_H$",

    "humidity_only":
        r"(b) Humidity-only $\beta_H$",

    "all_climate":
        r"(c) Climate-driven $\beta_H$",

    "sigmoid":
        r"(d) Sigmoid $\beta_H$",
}


# -----------------------------------------------------------------------------
# Final posterior location
# -----------------------------------------------------------------------------

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
    / "nested_sampling"
)


def result_path(
    model: str,
) -> Path:
    return (
        RESULT_DIR
        / f"{model}__{FIT_TARGET}__final_v2.pkl"
    )


# -----------------------------------------------------------------------------
# Posterior settings
# -----------------------------------------------------------------------------

N_POSTERIOR_DRAWS = 1000
RANDOM_SEED = 12345


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
# Infection-to-death delay settings
# -----------------------------------------------------------------------------

DELAY_KIND = "gamma"
DELAY_TAU_WEEKS = 2
DELAY_MAX_WEEKS = 12
DELAY_SHAPE = 2.0
DELAY_SCALE = 0.6


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "bayesian_model_fits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Plot styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


COLOR_OBSERVED = "#1f4e8c"
COLOR_MEDIAN = "#E08214"
COLOR_MLE = "#2C7C2C"
COLOR_OUTLIER = "#C0392B"


# =============================================================================
# Helper functions
# =============================================================================

def load_final_result(
    model: str,
) -> dict:
    """
    Load one final_v2 nested-sampling posterior.
    """

    path = result_path(
        model
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find final_v2 result for {model!r}:\n"
            f"{path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(
        path,
        "rb",
    ) as f:

        saved = pickle.load(f)

    return saved


def get_result_array(
    results,
    name: str,
) -> np.ndarray:
    """
    Read an array from dynesty Results regardless of whether it is exposed
    as an attribute or dictionary-like field.
    """

    if hasattr(
        results,
        name,
    ):
        return np.asarray(
            getattr(
                results,
                name,
            )
        )

    return np.asarray(
        results[
            name
        ]
    )


def get_equal_weight_posterior(
    samples: np.ndarray,
    logwt: np.ndarray,
    logz: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Convert dynesty's weighted posterior into an equal-weight posterior
    reproducibly.
    """

    weights = np.exp(
        logwt
        -
        logz[-1]
    )

    rstate = np.random.default_rng(
        seed
    )

    posterior = resample_equal(
        samples,
        weights,
        rstate=rstate,
    )

    return np.asarray(
        posterior,
        dtype=float,
    )


def compute_mu_trajectory(
    theta: np.ndarray,
    param_names: list[str],
    betaH_mode: str,
    merged_df: pd.DataFrame,
    zT: np.ndarray,
    ze: np.ndarray,
    zR: np.ndarray,
    start_date: str,
    f_delay: np.ndarray,
) -> np.ndarray:
    """
    Simulate one posterior parameter vector and return the expected weekly
    plague-death trajectory mu(t).

    This function calls simulate_seir_dde directly rather than constructing
    a PlagueLikelihood object, so no downstream prior-bound logic is involved.
    """

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
        betaH_mode,
    )

    sim = simulate_seir_dde(
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
            betaH_mode,

        start_date=
            start_date,
    )


    # -------------------------------------------------------------------------
    # Aggregate modelled infections to epidemiological weeks
    # -------------------------------------------------------------------------

    weekly_inf = weekly_sum(
        sim,
        col="C",
        week_ending=WEEK_RULE,
    )

    weekly_inf[
        "week_ending"
    ] = pd.to_datetime(
        weekly_inf[
            "week_ending"
        ]
    )


    compare = pd.merge(
        merged_df[
            [
                "week_ending"
            ]
        ],
        weekly_inf,
        on="week_ending",
        how="left",
    )


    # -------------------------------------------------------------------------
    # Find the weekly infection-count column
    # -------------------------------------------------------------------------

    if "C_week" in compare.columns:

        c_col = "C_week"

    else:

        candidate_cols = [
            c
            for c in compare.columns
            if c.startswith(
                "C_"
            )
        ]

        if not candidate_cols:

            raise RuntimeError(
                "Could not identify weekly infection-count column."
            )

        c_col = candidate_cols[
            0
        ]


    C_week = compare[
        c_col
    ].to_numpy(
        dtype=float
    )


    # -------------------------------------------------------------------------
    # Fail rather than silently fill missing fitted weeks.
    # -------------------------------------------------------------------------

    if np.isnan(
        C_week
    ).any():

        bad = np.where(
            np.isnan(
                C_week
            )
        )[0]

        raise RuntimeError(
            "Weekly infection trajectory contains missing fitted weeks. "
            f"Indices: {bad.tolist()}"
        )


    if not np.isfinite(
        C_week
    ).all():

        raise RuntimeError(
            "Weekly infection trajectory contains non-finite values."
        )


    if (
        C_week
        <
        -1e-10
    ).any():

        raise RuntimeError(
            "Weekly infection trajectory contains negative values."
        )


    # -------------------------------------------------------------------------
    # Convert infections to expected deaths
    # -------------------------------------------------------------------------

    mu = deaths_from_infections_fast(
        C_week=C_week,
        psi=sample["psi"],
        f_delay=f_delay,
    )


    mu = np.asarray(
        mu,
        dtype=float,
    )


    if len(
        mu
    ) != len(
        merged_df
    ):

        raise RuntimeError(
            "Expected-death trajectory has the wrong length: "
            f"{len(mu)} versus {len(merged_df)} fitted weeks."
        )


    if not np.isfinite(
        mu
    ).all():

        raise RuntimeError(
            "Expected-death trajectory contains non-finite values."
        )


    return np.maximum(
        mu,
        1e-9,
    )


def sse(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Sum of squared errors used only as a descriptive trajectory diagnostic.
    """

    return float(
        np.sum(
            (
                observed
                -
                predicted
            )
            ** 2
        )
    )


# =============================================================================
# Main analysis
# =============================================================================

def main():

    # =========================================================================
    # 1. Load prepared data
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


    # -------------------------------------------------------------------------
    # Required columns
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
            "Prepared data are missing required columns: "
            +
            ", ".join(
                missing_columns
            )
        )


    week_ending = pd.to_datetime(
        merged_df[
            paths.DATE_COL
        ]
    )


    zT = merged_df[
        "zT"
    ].to_numpy(
        dtype=float
    )


    ze = merged_df[
        "ze"
    ].to_numpy(
        dtype=float
    )


    zR = merged_df[
        "zR"
    ].to_numpy(
        dtype=float
    )


    start_date = str(
        pd.Timestamp(
            week_ending.iloc[
                0
            ]
        ).date()
    )


    n_weeks = len(
        merged_df
    )


    observed_fit = merged_df[
        FIT_TARGET
    ].to_numpy(
        dtype=float
    )


    if RAW_TARGET in merged_df.columns:

        observed_raw = merged_df[
            RAW_TARGET
        ].to_numpy(
            dtype=float
        )

    else:

        observed_raw = observed_fit.copy()


    outlier_mask = (
        week_ending
        ==
        pd.Timestamp(
            OUTLIER_WEEK
        )
    ).to_numpy()


    if not outlier_mask.any():

        print(
            f"[WARNING] "
            f"Outlier week "
            f"{OUTLIER_WEEK} "
            "not found in prepared data."
        )


    # =========================================================================
    # 2. Infection-to-death delay distribution
    # =========================================================================

    f_delay = make_delay_pmf_weeks(
        kind=
            DELAY_KIND,

        tau_weeks=
            DELAY_TAU_WEEKS,

        max_weeks=
            DELAY_MAX_WEEKS,

        shape=
            DELAY_SHAPE,

        scale=
            DELAY_SCALE,
    )


    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "BAYESIAN POSTERIOR MODEL-FIT TRAJECTORIES "
        "(final_v2)"
    )

    print(
        "=" * 90
    )


    print(
        f"\nFitted weeks: "
        f"{n_weeks}"
    )


    print(
        f"Study period: "
        f"{week_ending.min().date()} "
        f"to "
        f"{week_ending.max().date()}"
    )


    print(
        f"Posterior draws per model: "
        f"{N_POSTERIOR_DRAWS}"
    )


    # =========================================================================
    # 3. Main model loop
    # =========================================================================

    all_model_summaries: dict[
        str,
        pd.DataFrame,
    ] = {}


    divergence_rows = []


    for model_index, model in enumerate(
        MODELS
    ):

        print(
            f"\n--- {model} ---"
        )


        # =====================================================================
        # 3a. Load final posterior
        # =====================================================================

        saved = load_final_result(
            model
        )


        results = saved[
            "results"
        ]


        param_names = list(
            saved[
                "param_names"
            ]
        )


        logwt = get_result_array(
            results,
            "logwt",
        )


        logz = get_result_array(
            results,
            "logz",
        )


        logl = get_result_array(
            results,
            "logl",
        )


        samples = get_result_array(
            results,
            "samples",
        )


        # =====================================================================
        # 3b. Reproducible equal-weight posterior
        # =====================================================================

        resample_seed = (
            RANDOM_SEED
            +
            100 * model_index
        )


        posterior = get_equal_weight_posterior(
            samples=samples,
            logwt=logwt,
            logz=logz,
            seed=resample_seed,
        )


        n_available = len(
            posterior
        )


        n_draws = min(
            N_POSTERIOR_DRAWS,
            n_available,
        )


        # =====================================================================
        # 3c. Reproducible posterior subsample
        # =====================================================================

        subsample_seed = (
            RANDOM_SEED
            +
            10000
            +
            model_index
        )


        subsample_rng = np.random.default_rng(
            subsample_seed
        )


        draw_idx = subsample_rng.choice(
            n_available,
            size=n_draws,
            replace=False,
        )


        draws = posterior[
            draw_idx
        ]


        print(
            f"  Equal-weight posterior samples available: "
            f"{n_available}"
        )


        print(
            f"  Posterior draws used: "
            f"{n_draws}"
        )


        # =====================================================================
        # 3d. Save exact selected posterior parameter vectors
        # =====================================================================

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
            np.arange(
                1,
                n_draws + 1,
            ),
        )


        selected_draws_path = (
            OUTPUT_DIR
            /
            f"selected_model_fit_draws_{model}.csv"
        )


        selected_draws_df.to_csv(
            selected_draws_path,
            index=False,
        )


        print(
            f"  [OK] Saved exact selected posterior draws:\n"
            f"    {selected_draws_path}"
        )


        # =====================================================================
        # 3e. Posterior expected-death trajectories
        # =====================================================================

        mu_draws = np.full(
            (
                n_draws,
                n_weeks,
            ),
            np.nan,
            dtype=float,
        )


        for i, theta in enumerate(
            draws
        ):

            mu_draws[
                i,
                :
            ] = compute_mu_trajectory(
                theta=
                    theta,

                param_names=
                    param_names,

                betaH_mode=
                    model,

                merged_df=
                    merged_df,

                zT=
                    zT,

                ze=
                    ze,

                zR=
                    zR,

                start_date=
                    start_date,

                f_delay=
                    f_delay,
            )


            if (
                (i + 1) % 50 == 0
                or
                (i + 1) == n_draws
            ):

                print(
                    f"  processed "
                    f"{i + 1} / "
                    f"{n_draws}"
                )


        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------

        if np.isnan(
            mu_draws
        ).any():

            raise RuntimeError(
                f"{model}: posterior expected-death trajectories "
                "contain NaN values."
            )


        if not np.isfinite(
            mu_draws
        ).all():

            raise RuntimeError(
                f"{model}: posterior expected-death trajectories "
                "contain non-finite values."
            )


        # ---------------------------------------------------------------------
        # Posterior weekly summaries
        # ---------------------------------------------------------------------

        mu_q025 = np.quantile(
            mu_draws,
            0.025,
            axis=0,
        )


        mu_band_median = np.quantile(
            mu_draws,
            0.50,
            axis=0,
        )


        mu_q975 = np.quantile(
            mu_draws,
            0.975,
            axis=0,
        )


        # =====================================================================
        # 3f. Highest-likelihood trajectory
        # =====================================================================

        best_idx = int(
            np.nanargmax(
                logl
            )
        )


        theta_mle = np.asarray(
            samples[
                best_idx
            ],
            dtype=float,
        )


        max_loglik = float(
            logl[
                best_idx
            ]
        )


        mu_mle = compute_mu_trajectory(
            theta=
                theta_mle,

            param_names=
                param_names,

            betaH_mode=
                model,

            merged_df=
                merged_df,

            zT=
                zT,

            ze=
                ze,

            zR=
                zR,

            start_date=
                start_date,

            f_delay=
                f_delay,
        )


        mle_sse = sse(
            observed_fit,
            mu_mle,
        )


        # =====================================================================
        # 3g. Posterior-median-parameter trajectory
        # =====================================================================

        theta_median_params = np.median(
            posterior,
            axis=0,
        )


        mu_median_params = compute_mu_trajectory(
            theta=
                theta_median_params,

            param_names=
                param_names,

            betaH_mode=
                model,

            merged_df=
                merged_df,

            zT=
                zT,

            ze=
                ze,

            zR=
                zR,

            start_date=
                start_date,

            f_delay=
                f_delay,
        )


        median_sse = sse(
            observed_fit,
            mu_median_params,
        )


        if mle_sse > 0:

            sse_ratio = (
                median_sse
                /
                mle_sse
            )

        else:

            sse_ratio = np.nan


        print(
            f"  MLE trajectory: "
            f"logL={max_loglik:.3f}, "
            f"SSE={mle_sse:.1f}"
        )


        print(
            f"  Posterior-median-parameter trajectory: "
            f"SSE={median_sse:.1f}, "
            f"ratio={sse_ratio:.2f}x MLE SSE"
        )


        if (
            np.isfinite(
                sse_ratio
            )
            and
            sse_ratio > 1.5
        ):

            print(
                "  [DIAGNOSTIC] Median-parameter trajectory fits "
                "notably worse than the MLE trajectory. This is "
                "consistent with correlated posterior parameters."
            )


        divergence_rows.append({

            "betaH_mode":
                model,

            "max_loglik":
                max_loglik,

            "mle_sse":
                mle_sse,

            "median_param_sse":
                median_sse,

            "median_over_mle_sse_ratio":
                sse_ratio,
        })


        # =====================================================================
        # 3h. Save model trajectories
        # =====================================================================

        model_df = pd.DataFrame({

            "week_ending":
                week_ending,

            "observed_fit":
                observed_fit,

            "observed_raw":
                observed_raw,

            "mu_q025":
                mu_q025,

            "mu_band_median":
                mu_band_median,

            "mu_q975":
                mu_q975,

            "mu_mle":
                mu_mle,

            "mu_median_params":
                mu_median_params,
        })


        csv_path = (
            OUTPUT_DIR
            /
            f"bayesian_model_fits_{model}.csv"
        )


        model_df.to_csv(
            csv_path,
            index=False,
        )


        print(
            f"  [OK] Saved:\n"
            f"    {csv_path}"
        )


        all_model_summaries[
            model
        ] = model_df


    # =========================================================================
    # 4. Save divergence diagnostics
    # =========================================================================

    divergence_df = pd.DataFrame(
        divergence_rows
    )


    divergence_csv = (
        OUTPUT_DIR
        /
        "mle_vs_median_divergence.csv"
    )


    divergence_df.to_csv(
        divergence_csv,
        index=False,
    )


    print(
        "\n"
        +
        "=" * 90
    )


    print(
        "MLE vs. POSTERIOR-MEDIAN-PARAMETER DIVERGENCE"
    )


    print(
        "=" * 90
    )


    print(
        divergence_df.to_string(
            index=False
        )
    )


    print(
        f"\n[OK] Saved:\n"
        f"  {divergence_csv}"
    )


    # =========================================================================
    # 5. Figure: 2x2 model-fit grid
    # =========================================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            8,
        ),
        sharex=True,
    )


    for ax, model in zip(
        axes.flat,
        MODELS,
    ):

        df = all_model_summaries[
            model
        ]


        # ---------------------------------------------------------------------
        # Posterior latent-mean credible interval
        # ---------------------------------------------------------------------

        ax.fill_between(
            df[
                "week_ending"
            ],
            df[
                "mu_q025"
            ],
            df[
                "mu_q975"
            ],
            color=
                COLOR_MEDIAN,
            alpha=
                0.20,
            linewidth=
                0,
            label=
                "Posterior 95% CrI of expected trajectory",
        )


        # ---------------------------------------------------------------------
        # Observed fitted mortality series
        # ---------------------------------------------------------------------

        ax.plot(
            df[
                "week_ending"
            ],
            df[
                "observed_fit"
            ],
            marker=
                "o",
            markersize=
                2.5,
            color=
                COLOR_OBSERVED,
            linewidth=
                1.0,
            label=
                "Observed reported deaths",
        )


        # ---------------------------------------------------------------------
        # Highest-likelihood trajectory
        # ---------------------------------------------------------------------

        ax.plot(
            df[
                "week_ending"
            ],
            df[
                "mu_mle"
            ],
            color=
                COLOR_MLE,
            linewidth=
                1.6,
            label=
                "Highest-likelihood trajectory",
        )


        # ---------------------------------------------------------------------
        # Posterior-median-parameter diagnostic
        # ---------------------------------------------------------------------

        ax.plot(
            df[
                "week_ending"
            ],
            df[
                "mu_median_params"
            ],
            color=
                COLOR_MEDIAN,
            linewidth=
                1.3,
            linestyle=
                "--",
            label=
                "Posterior-median-parameter trajectory",
        )


        # ---------------------------------------------------------------------
        # Original outlier
        # ---------------------------------------------------------------------

        if outlier_mask.any():

            ax.scatter(
                week_ending[
                    outlier_mask
                ],
                observed_raw[
                    outlier_mask
                ],
                color=
                    COLOR_OUTLIER,
                s=
                    25,
                zorder=
                    5,
                label=
                    "Original reported outlier",
            )


        ax.set_title(
            MODEL_LABELS[
                model
            ]
        )


        ax.set_ylabel(
            "Weekly plague deaths"
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


        ax.xaxis.set_major_locator(
            mdates.MonthLocator(
                interval=6
            )
        )


        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%Y-%m"
            )
        )


        ax.grid(
            False
        )


    # -------------------------------------------------------------------------
    # Shared legend
    # -------------------------------------------------------------------------

    handles, labels = (
        axes[
            0,
            0
        ].get_legend_handles_labels()
    )


    fig.legend(
        handles,
        labels,
        loc=
            "upper center",
        ncol=
            3,
        frameon=
            False,
        bbox_to_anchor=
            (
                0.5,
                1.05,
            ),
    )


    fig.tight_layout()


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    fig_path_png = (
        OUTPUT_DIR
        /
        "bayesian_model_fits_grid.png"
    )


    fig_path_pdf = (
        OUTPUT_DIR
        /
        "bayesian_model_fits_grid.pdf"
    )


    fig.savefig(
        fig_path_png,
        dpi=
            600,
        bbox_inches=
            "tight",
    )


    fig.savefig(
        fig_path_pdf,
        bbox_inches=
            "tight",
    )


    plt.close(
        fig
    )


    print(
        f"\n[OK] Saved figure:\n"
        f"  {fig_path_png}"
    )


    print(
        f"[OK] Saved figure:\n"
        f"  {fig_path_pdf}"
    )


    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()