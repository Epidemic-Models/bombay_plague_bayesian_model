# =============================================================================
# posterior_predictive_round2.py
# =============================================================================
"""
Posterior predictive analysis for the final round-2 all-climate model.

This script:

1. Loads the round-2 all-climate dynesty result.
2. Converts dynesty weighted samples to equal-weight posterior samples.
3. Randomly selects posterior parameter vectors.
4. Runs the mechanistic model for every selected posterior draw.
5. Calculates the deterministic expected weekly deaths mu_t.
6. Generates negative-binomial posterior predictive death counts.
7. Calculates week-by-week posterior intervals.
8. Plots:
       - observed deaths
       - posterior median expected deaths
       - 95% credible interval for expected deaths
       - 95% posterior predictive interval for observed deaths
9. Saves numerical posterior predictive summaries.

Usage
-----
python scripts/posterior_predictive_round2.py
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

BETA_H_MODE = "all_climate"
FIT_TARGET = "plague_deaths_reported_fit"

# Number of posterior parameter vectors used for prediction.
#
# 1000 is normally plenty for a smooth 95% ribbon.
# Increase to 2000 later if desired.
N_POSTERIOR_DRAWS = 1000

RANDOM_SEED = 12345


# =============================================================================
# Paths
# =============================================================================

ROUND2_ROOT = (
    paths.RESULTS_DIR
    / "prior_sensitivity_round2"
)

ROUND2_NESTED_DIR = (
    ROUND2_ROOT
    / "nested_sampling"
)

ROUND2_FIG_DIR = (
    ROUND2_ROOT
    / "figures"
)

ROUND2_SUMMARY_DIR = (
    ROUND2_ROOT
    / "posterior_predictive"
)

ROUND2_FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ROUND2_SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULT_PKL = (
    ROUND2_NESTED_DIR
    / "all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl"
)

OUTPUT_CSV = (
    ROUND2_SUMMARY_DIR
    / "posterior_predictive_summary_round2.csv"
)

OUTPUT_NPZ = (
    ROUND2_SUMMARY_DIR
    / "posterior_predictive_draws_round2.npz"
)

FIG_PATH = (
    ROUND2_FIG_DIR
    / "posterior_predictive_all_climate_round2.png"
)


# =============================================================================
# Figure styling
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


# Your preferred figure colours
COLOR_MODEL = "#B3222B"
COLOR_OBS = "black"


# =============================================================================
# Load nested-sampling result
# =============================================================================

if not RESULT_PKL.exists():
    raise FileNotFoundError(
        f"Could not find round-2 nested-sampling result:\n{RESULT_PKL}"
    )


print("\n" + "=" * 88)
print("POSTERIOR PREDICTIVE ANALYSIS: ROUND-2 ALL-CLIMATE MODEL")
print("=" * 88)

print(f"\nLoading:\n{RESULT_PKL}")


with open(RESULT_PKL, "rb") as f:
    saved = pickle.load(f)


results = saved["results"]
param_names = saved["param_names"]

print(f"\nNumber of parameters: {len(param_names)}")
print(f"Parameters: {param_names}")

print(
    f"\nlog(Z) = "
    f"{results.logz[-1]:.3f} "
    f"+/- {results.logzerr[-1]:.3f}"
)


# =============================================================================
# Convert dynesty samples to equal-weight posterior samples
# =============================================================================

weights = np.exp(
    results.logwt
    - results.logz[-1]
)

posterior_samples = resample_equal(
    results.samples,
    weights,
)

print(
    f"\nAvailable equal-weight posterior samples: "
    f"{len(posterior_samples)}"
)


# =============================================================================
# Select posterior draws
# =============================================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

n_available = len(
    posterior_samples
)

n_draws = min(
    N_POSTERIOR_DRAWS,
    n_available,
)

draw_indices = rng.choice(
    n_available,
    size=n_draws,
    replace=False,
)

posterior_draws = posterior_samples[
    draw_indices
]


print(
    f"Using {n_draws} posterior draws "
    f"for posterior predictive analysis."
)


# =============================================================================
# Load prepared data
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(
        f"Prepared data file not found:\n"
        f"{paths.MERGED_CSV}"
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


weeks = merged_df[
    paths.DATE_COL
]

observed = merged_df[
    FIT_TARGET
].to_numpy(
    dtype=float
)


n_weeks = len(
    merged_df
)

print(
    f"\nNumber of fitted weeks: "
    f"{n_weeks}"
)


# =============================================================================
# Construct likelihood object
# =============================================================================

lik = PlagueLikelihood(
    merged_df=merged_df,
    betaH_mode=BETA_H_MODE,
    fit_target=FIT_TARGET,
)


# =============================================================================
# Locate NB dispersion parameter
# =============================================================================

if "nb_kappa" not in param_names:
    raise ValueError(
        "Could not find nb_kappa in parameter names."
    )


kappa_index = param_names.index(
    "nb_kappa"
)


# =============================================================================
# Allocate prediction arrays
# =============================================================================

# Deterministic expected deaths for every posterior parameter draw
mu_draws = np.full(
    shape=(n_draws, n_weeks),
    fill_value=np.nan,
    dtype=float,
)

# Actual posterior predictive observed counts,
# including negative-binomial observation noise
predictive_draws = np.full(
    shape=(n_draws, n_weeks),
    fill_value=np.nan,
    dtype=float,
)


# =============================================================================
# Negative-binomial simulation helper
# =============================================================================

def simulate_negative_binomial(
    mu: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw observations from the same NB mean-dispersion parameterization
    normally used in plague likelihoods:

        E[Y] = mu
        Var[Y] = mu + mu^2 / kappa

    numpy.random.negative_binomial uses parameters n and p:

        mean = n * (1-p) / p

    Setting:

        n = kappa
        p = kappa / (kappa + mu)

    gives the desired mean and variance.
    """

    mu = np.asarray(
        mu,
        dtype=float,
    )

    mu = np.maximum(
        mu,
        1e-12,
    )

    if kappa <= 0:
        raise ValueError(
            "nb_kappa must be positive."
        )

    p = (
        kappa
        / (kappa + mu)
    )

    return rng.negative_binomial(
        n=kappa,
        p=p,
    )


# =============================================================================
# Generate posterior predictions
# =============================================================================

print("\nGenerating posterior predictions...")
print("-" * 88)


n_success = 0
n_failed = 0


for i, theta in enumerate(
    posterior_draws
):

    try:

        # --------------------------------------------------------------
        # Expected weekly deaths for this parameter vector
        # --------------------------------------------------------------

        mu = lik.predict_weekly_deaths(
            theta
        )

        mu = np.asarray(
            mu,
            dtype=float,
        )

        if len(mu) != n_weeks:
            raise ValueError(
                f"Prediction length {len(mu)} "
                f"does not match data length {n_weeks}."
            )

        if not np.all(
            np.isfinite(mu)
        ):
            raise ValueError(
                "Non-finite predicted deaths."
            )

        mu = np.maximum(
            mu,
            1e-12,
        )

        mu_draws[i, :] = mu


        # --------------------------------------------------------------
        # Negative-binomial observation noise
        # --------------------------------------------------------------

        nb_kappa = float(
            theta[kappa_index]
        )

        y_rep = simulate_negative_binomial(
            mu=mu,
            kappa=nb_kappa,
            rng=rng,
        )

        predictive_draws[
            i, :
        ] = y_rep

        n_success += 1


    except Exception as e:

        n_failed += 1

        print(
            f"[WARNING] Posterior draw {i} failed: {e}"
        )


    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1} / {n_draws}"
        )


print("\nPrediction complete.")

print(
    f"Successful draws: {n_success}"
)

print(
    f"Failed draws:     {n_failed}"
)


# =============================================================================
# Remove failed draws
# =============================================================================

valid = np.all(
    np.isfinite(mu_draws),
    axis=1,
)

mu_draws = mu_draws[
    valid
]

predictive_draws = predictive_draws[
    valid
]


if len(mu_draws) == 0:
    raise RuntimeError(
        "All posterior predictive simulations failed."
    )


print(
    f"\nPosterior draws retained: "
    f"{len(mu_draws)}"
)


# =============================================================================
# Expected-death posterior summaries
# =============================================================================

mu_q025 = np.quantile(
    mu_draws,
    0.025,
    axis=0,
)

mu_q25 = np.quantile(
    mu_draws,
    0.25,
    axis=0,
)

mu_median = np.quantile(
    mu_draws,
    0.50,
    axis=0,
)

mu_q75 = np.quantile(
    mu_draws,
    0.75,
    axis=0,
)

mu_q975 = np.quantile(
    mu_draws,
    0.975,
    axis=0,
)


# =============================================================================
# Full posterior predictive summaries
# =============================================================================

pred_q025 = np.quantile(
    predictive_draws,
    0.025,
    axis=0,
)

pred_q25 = np.quantile(
    predictive_draws,
    0.25,
    axis=0,
)

pred_median = np.quantile(
    predictive_draws,
    0.50,
    axis=0,
)

pred_q75 = np.quantile(
    predictive_draws,
    0.75,
    axis=0,
)

pred_q975 = np.quantile(
    predictive_draws,
    0.975,
    axis=0,
)


# =============================================================================
# Posterior predictive diagnostics
# =============================================================================

inside_mu_95 = (
    (observed >= mu_q025)
    & (observed <= mu_q975)
)

inside_predictive_95 = (
    (observed >= pred_q025)
    & (observed <= pred_q975)
)


mu_coverage = np.mean(
    inside_mu_95
)

predictive_coverage = np.mean(
    inside_predictive_95
)


print("\n" + "=" * 88)
print("POSTERIOR PREDICTIVE DIAGNOSTICS")
print("=" * 88)

print(
    f"Observed weeks inside 95% expected-mean interval: "
    f"{100 * mu_coverage:.1f}%"
)

print(
    f"Observed weeks inside 95% posterior predictive interval: "
    f"{100 * predictive_coverage:.1f}%"
)


# =============================================================================
# SSE of posterior median expected trajectory
# =============================================================================

posterior_median_sse = float(
    np.sum(
        (observed - mu_median) ** 2
    )
)


print(
    f"\nSSE of posterior median expected-death trajectory: "
    f"{posterior_median_sse:.3f}"
)


# =============================================================================
# Save numerical summaries
# =============================================================================

summary_df = pd.DataFrame({
    "week_ending":
        weeks,

    "observed_deaths":
        observed,

    "mu_q025":
        mu_q025,

    "mu_q25":
        mu_q25,

    "mu_median":
        mu_median,

    "mu_q75":
        mu_q75,

    "mu_q975":
        mu_q975,

    "predictive_q025":
        pred_q025,

    "predictive_q25":
        pred_q25,

    "predictive_median":
        pred_median,

    "predictive_q75":
        pred_q75,

    "predictive_q975":
        pred_q975,

    "observed_inside_mu_95":
        inside_mu_95,

    "observed_inside_predictive_95":
        inside_predictive_95,
})


summary_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# Save raw draws too.
# This is useful later without having to rerun 1000 model simulations.

np.savez_compressed(
    OUTPUT_NPZ,
    mu_draws=mu_draws,
    predictive_draws=predictive_draws,
)


print(
    f"\n[OK] Saved posterior predictive summary:\n"
    f"{OUTPUT_CSV}"
)

print(
    f"\n[OK] Saved raw posterior predictive draws:\n"
    f"{OUTPUT_NPZ}"
)


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.2, 4.8)
)


# -----------------------------------------------------------------------------
# Full posterior predictive interval
# -----------------------------------------------------------------------------

ax.fill_between(
    weeks,
    pred_q025,
    pred_q975,
    color=COLOR_MODEL,
    alpha=0.12,
    linewidth=0,
    label="95% posterior predictive interval",
)


# -----------------------------------------------------------------------------
# Parameter uncertainty in expected deaths
# -----------------------------------------------------------------------------

ax.fill_between(
    weeks,
    mu_q025,
    mu_q975,
    color=COLOR_MODEL,
    alpha=0.28,
    linewidth=0,
    label="95% credible interval",
)


# -----------------------------------------------------------------------------
# Posterior median expected deaths
# -----------------------------------------------------------------------------

ax.plot(
    weeks,
    mu_median,
    color=COLOR_MODEL,
    linewidth=2.2,
    label="Posterior median",
    zorder=4,
)


# -----------------------------------------------------------------------------
# Observed deaths
# -----------------------------------------------------------------------------

ax.scatter(
    weeks,
    observed,
    s=16,
    color=COLOR_OBS,
    alpha=0.80,
    edgecolors="none",
    label="Observed deaths",
    zorder=5,
)


# =============================================================================
# Axes
# =============================================================================

ax.set_xlabel(
    "Week ending"
)

ax.set_ylabel(
    "Weekly plague deaths"
)


ax.set_xlim(
    weeks.min(),
    weeks.max(),
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
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


ax.tick_params(
    axis="both",
    which="major",
    length=4,
)


ax.grid(
    False
)


ax.legend(
    frameon=False,
    loc="upper left",
)


fig.tight_layout()


# =============================================================================
# Save figure
# =============================================================================

fig.savefig(
    FIG_PATH,
    dpi=600,
    bbox_inches="tight",
)


print(
    f"\n[OK] Saved posterior predictive figure:\n"
    f"{FIG_PATH}"
)


plt.show()
plt.close(fig)


# =============================================================================
# Final summary
# =============================================================================

print("\n" + "=" * 88)
print("FINAL SUMMARY")
print("=" * 88)

print(
    f"Posterior draws used: {len(mu_draws)}"
)

print(
    f"Expected-mean 95% interval coverage: "
    f"{100 * mu_coverage:.1f}%"
)

print(
    f"Posterior predictive 95% interval coverage: "
    f"{100 * predictive_coverage:.1f}%"
)

print(
    f"Posterior-median trajectory SSE: "
    f"{posterior_median_sse:.3f}"
)

print(
    "\nPosterior predictive analysis completed successfully."
)