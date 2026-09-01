# =============================================================================
# plot_model_fits.py
# =============================================================================
"""
Compare fitted mortality trajectories from Bayesian nested-sampling models.

For each beta_H formulation, this script loads the saved dynesty result and
constructs two fitted trajectories:

    1. Posterior-median parameter trajectory
       ------------------------------------------------------------
       Each parameter is replaced by its marginal posterior median, and the
       mechanistic model is simulated once using that combined parameter vector.

       NOTE:
       This is NOT necessarily the best-fitting parameter combination because
       parameters may be correlated and the component-wise median vector may
       never have been sampled by dynesty.

    2. Highest-likelihood dynesty trajectory
       ------------------------------------------------------------
       The sampled parameter vector having the largest negative-binomial
       log-likelihood is extracted directly from the dynesty results.

       This is the closest Bayesian-run analogue to the "best sample" used in
       the previous LHS analysis.

For both trajectories the script calculates:

    - negative-binomial log-likelihood
    - sum of squared errors (SSE)

The publication-style comparison plot uses the highest-likelihood trajectory
for each model.

Models:
    constant
    sigmoid
    humidity_only
    all_climate

Input:
    prepared_data/merged_target_climate.csv

Nested-sampling results:
    results/nested_sampling/
        constant__plague_deaths_reported_fit.pkl
        sigmoid__plague_deaths_reported_fit.pkl
        humidity_only__plague_deaths_reported_fit.pkl
        all_climate__plague_deaths_reported_fit.pkl

Outputs:
    results/figures/bayesian_model_fit_comparison__plague_deaths_reported_fit.png

    results/posterior_summaries/
        model_fit_diagnostics__plague_deaths_reported_fit.csv

Usage:
    python scripts/plot_model_fits.py
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
from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

BETA_H_MODES = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]

MODEL_LABELS = {
    "constant": r"Constant $\beta_H$",
    "sigmoid": r"Sigmoid $\beta_H$",
    "humidity_only": r"Humidity-dependent $\beta_H$",
    "all_climate": r"Climate-driven $\beta_H$",
}


# =============================================================================
# Plot styling
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


COLOR_OBS = "black"

MODEL_COLORS = {
    "constant": "#E08214",
    "sigmoid": "#7B3294",
    "humidity_only": "#008837",
    "all_climate": "#B3222B",
}


# =============================================================================
# Helpers
# =============================================================================

def load_dynesty_result(betaH_mode: str) -> dict:
    """
    Load the saved nested-sampling result for one beta_H model.
    """

    stem = paths.run_stem(
        betaH_mode,
        FIT_TARGET,
    )

    pkl_path = (
        paths.NESTED_SAMPLING_DIR
        / f"{stem}.pkl"
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find nested-sampling result:\n"
            f"{pkl_path}"
        )

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    return saved


def get_equal_weight_posterior_samples(
    saved: dict,
) -> np.ndarray:
    """
    Convert the weighted dynesty sample cloud to approximately
    equally weighted posterior samples.
    """

    results = saved["results"]

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    samples_equal = resample_equal(
        results.samples,
        weights,
    )

    return samples_equal


def get_posterior_median_theta(
    saved: dict,
) -> np.ndarray:
    """
    Construct the component-wise marginal posterior median parameter vector.
    """

    samples_equal = get_equal_weight_posterior_samples(
        saved
    )

    theta_median = np.median(
        samples_equal,
        axis=0,
    )

    return theta_median


def get_maximum_likelihood_theta(
    saved: dict,
) -> tuple[np.ndarray, float]:
    """
    Return the sampled parameter vector with the highest dynesty
    negative-binomial log-likelihood.
    """

    results = saved["results"]

    best_idx = int(
        np.argmax(results.logl)
    )

    theta_mle = (
        results.samples[best_idx]
        .copy()
    )

    max_loglik = float(
        results.logl[best_idx]
    )

    return theta_mle, max_loglik


def sum_squared_errors(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Sum of squared errors:

        SSE = sum_t (y_t - mu_t)^2
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted arrays "
            "must have the same shape."
        )

    return float(
        np.sum(
            (observed - predicted) ** 2
        )
    )


# =============================================================================
# Load prepared data
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(
        f"Prepared data file not found:\n"
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


# =============================================================================
# Observed mortality
# =============================================================================

if FIT_TARGET not in merged_df.columns:
    raise ValueError(
        f"{FIT_TARGET!r} not found in prepared dataset."
    )


weeks = merged_df[
    paths.DATE_COL
]

observed = merged_df[
    FIT_TARGET
].to_numpy(dtype=float)


# =============================================================================
# Generate fits and diagnostics
# =============================================================================

mle_predictions = {}
median_predictions = {}

summary_rows = []


for betaH_mode in BETA_H_MODES:

    print("\n" + "=" * 78)
    print(f"MODEL: {betaH_mode}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Construct likelihood object using the exact same model settings
    # as the nested-sampling fit
    # -------------------------------------------------------------------------

    lik = PlagueLikelihood(
        merged_df=merged_df,
        betaH_mode=betaH_mode,
        fit_target=FIT_TARGET,
    )

    # -------------------------------------------------------------------------
    # Load nested-sampling result
    # -------------------------------------------------------------------------

    saved = load_dynesty_result(
        betaH_mode
    )

    param_names = saved["param_names"]

    # -------------------------------------------------------------------------
    # Posterior median parameter vector
    # -------------------------------------------------------------------------

    theta_median = (
        get_posterior_median_theta(
            saved
        )
    )

    mu_median = (
        lik.predict_weekly_deaths(
            theta_median
        )
    )

    median_loglik = float(
        lik.log_likelihood(
            theta_median
        )
    )

    median_sse = (
        sum_squared_errors(
            observed,
            mu_median,
        )
    )

    median_predictions[
        betaH_mode
    ] = mu_median

    # -------------------------------------------------------------------------
    # Highest-likelihood dynesty sample
    # -------------------------------------------------------------------------

    theta_mle, max_loglik = (
        get_maximum_likelihood_theta(
            saved
        )
    )

    mu_mle = (
        lik.predict_weekly_deaths(
            theta_mle
        )
    )

    mle_sse = (
        sum_squared_errors(
            observed,
            mu_mle,
        )
    )

    mle_predictions[
        betaH_mode
    ] = mu_mle

    # -------------------------------------------------------------------------
    # AIC from highest likelihood encountered by dynesty
    # -------------------------------------------------------------------------

    k = len(param_names)

    aic = (
        2 * k
        - 2 * max_loglik
    )

    # -------------------------------------------------------------------------
    # Bayesian evidence
    # -------------------------------------------------------------------------

    logz = float(
        saved["logz"]
    )

    logzerr = float(
        saved["logzerr"]
    )

    # -------------------------------------------------------------------------
    # Print diagnostics
    # -------------------------------------------------------------------------

    print("\nPosterior-median parameter vector")
    print("---------------------------------")

    for name, value in zip(
        param_names,
        theta_median,
    ):
        print(
            f"{name:12s} = {value:.6g}"
        )

    print(
        f"\nPosterior-median logL = "
        f"{median_loglik:.3f}"
    )

    print(
        f"Posterior-median SSE  = "
        f"{median_sse:.3f}"
    )


    print("\nHighest-likelihood dynesty sample")
    print("---------------------------------")

    for name, value in zip(
        param_names,
        theta_mle,
    ):
        print(
            f"{name:12s} = {value:.6g}"
        )

    print(
        f"\nMaximum logL = "
        f"{max_loglik:.3f}"
    )

    print(
        f"MLE-sample SSE = "
        f"{mle_sse:.3f}"
    )

    print(
        f"AIC = "
        f"{aic:.3f}"
    )

    print(
        f"log(Z) = "
        f"{logz:.3f} +/- "
        f"{logzerr:.3f}"
    )

    # -------------------------------------------------------------------------
    # Store summary
    # -------------------------------------------------------------------------

    summary_rows.append({
        "betaH_mode": betaH_mode,
        "n_parameters": k,

        "posterior_median_loglik":
            median_loglik,

        "posterior_median_sse":
            median_sse,

        "max_loglik":
            max_loglik,

        "mle_sse":
            mle_sse,

        "AIC":
            aic,

        "logZ":
            logz,

        "logZ_error":
            logzerr,
    })


# =============================================================================
# Model-comparison table
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


# -------------------------------------------------------------------------
# Delta AIC
# -------------------------------------------------------------------------

summary_df["delta_AIC"] = (
    summary_df["AIC"]
    - summary_df["AIC"].min()
)


# -------------------------------------------------------------------------
# Delta log evidence relative to best model
#
# Best model has largest logZ.
# -------------------------------------------------------------------------

best_logz = (
    summary_df["logZ"].max()
)

summary_df["delta_logZ"] = (
    best_logz
    - summary_df["logZ"]
)


# -------------------------------------------------------------------------
# Approximate Bayes factor relative to best model
#
# BF_best/model = exp(delta_logZ)
# -------------------------------------------------------------------------

summary_df[
    "BF_best_over_model"
] = np.exp(
    summary_df["delta_logZ"]
)


# -------------------------------------------------------------------------
# Sort by Bayesian evidence
# -------------------------------------------------------------------------

summary_df = (
    summary_df
    .sort_values(
        "logZ",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# Save diagnostics table
# =============================================================================

paths.ensure_output_dirs()

diagnostic_path = (
    paths.POSTERIOR_SUMMARY_DIR
    / f"model_fit_diagnostics__{FIT_TARGET}.csv"
)

summary_df.to_csv(
    diagnostic_path,
    index=False,
)

print(
    f"\n[OK] Saved diagnostics table:\n"
    f"{diagnostic_path}"
)


# =============================================================================
# Figure 1:
# Highest-likelihood dynesty trajectories
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.2, 4.8)
)


# -------------------------------------------------------------------------
# Observed mortality
# -------------------------------------------------------------------------

ax.scatter(
    weeks,
    observed,
    s=20,
    color=COLOR_OBS,
    alpha=0.75,
    label="Observed deaths",
    zorder=5,
)


# -------------------------------------------------------------------------
# Model trajectories
# -------------------------------------------------------------------------

for betaH_mode in BETA_H_MODES:

    ax.plot(
        weeks,
        mle_predictions[
            betaH_mode
        ],
        color=MODEL_COLORS[
            betaH_mode
        ],
        linewidth=2.0,
        label=MODEL_LABELS[
            betaH_mode
        ],
    )


# -------------------------------------------------------------------------
# Axis styling
# -------------------------------------------------------------------------

ax.set_xlabel(
    "Week ending"
)

ax.set_ylabel(
    "Weekly plague deaths"
)


ax.xaxis.set_major_locator(
    mdates.MonthLocator(
        interval=4
    )
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%Y-%m"
    )
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


ax.tick_params(
    axis="both",
    which="major",
    length=4,
)


ax.grid(False)


ax.legend(
    frameon=False,
    loc="upper left",
)


fig.tight_layout()


# =============================================================================
# Save MLE figure
# =============================================================================

fig_path = (
    paths.FIGURES_DIR
    / f"bayesian_maximum_likelihood_fits__{FIT_TARGET}.png"
)


fig.savefig(
    fig_path,
    dpi=600,
    bbox_inches="tight",
)


print(
    f"\n[OK] Saved maximum-likelihood figure:\n"
    f"{fig_path}"
)


plt.show()
plt.close()


# =============================================================================
# Figure 2:
# Posterior-median-parameter trajectories
#
# This is included mainly as a diagnostic to show why component-wise
# posterior median parameters may not reproduce the maximum-likelihood fit.
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.2, 4.8)
)


ax.scatter(
    weeks,
    observed,
    s=20,
    color=COLOR_OBS,
    alpha=0.75,
    label="Observed deaths",
    zorder=5,
)


for betaH_mode in BETA_H_MODES:

    ax.plot(
        weeks,
        median_predictions[
            betaH_mode
        ],
        color=MODEL_COLORS[
            betaH_mode
        ],
        linewidth=2.0,
        label=MODEL_LABELS[
            betaH_mode
        ],
    )


ax.set_xlabel(
    "Week ending"
)

ax.set_ylabel(
    "Weekly plague deaths"
)


ax.xaxis.set_major_locator(
    mdates.MonthLocator(
        interval=4
    )
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%Y-%m"
    )
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


ax.tick_params(
    axis="both",
    which="major",
    length=4,
)


ax.grid(False)


ax.legend(
    frameon=False,
    loc="upper left",
)


fig.tight_layout()


median_fig_path = (
    paths.FIGURES_DIR
    / f"bayesian_posterior_median_parameter_fits__{FIT_TARGET}.png"
)


fig.savefig(
    median_fig_path,
    dpi=600,
    bbox_inches="tight",
)


print(
    f"[OK] Saved posterior-median parameter figure:\n"
    f"{median_fig_path}"
)


plt.show()
plt.close()