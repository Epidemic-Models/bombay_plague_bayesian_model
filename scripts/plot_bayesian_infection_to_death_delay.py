# =============================================================================
# plot_bayesian_infection_to_death_delay.py
# =============================================================================
"""
Plot the infection-to-death delay kernel used by the final Bayesian
plague models.

IMPORTANT
---------
The delay kernel is fixed in the current Bayesian model. Its Gamma shape,
scale, and maximum represented lag are not estimated from the posterior.

The final Bayesian fits use:

    delay_kind = "gamma"
    delay_shape = 2.0
    delay_scale = 0.6 weeks
    delay_max_weeks = 12

For kind="gamma", make_delay_pmf_weeks() evaluates the Gamma density at
weekly midpoints (j + 0.5) and normalizes those weights over the discrete
lags j = 0,...,12.

The plotted discrete probabilities are generated with the exact same
make_delay_pmf_weeks() function used by the Bayesian likelihood.

The observation model applies the kernel through

    mu_D[w] = psi * sum_j C_week[w-j] * f_delay[j].

Accordingly, f_delay[j] is attached operationally to convolution lag index
j, while the underlying Gamma density used to construct that weight is
evaluated at the midpoint j + 0.5 weeks.

For interpretation, this script therefore distinguishes:

    1. The integer convolution lag index j.
    2. The midpoint-based delay j + 0.5 used to construct the Gamma weight.

The midpoint-weighted mean is the more interpretable summary of the
infection-to-death delay represented by this discretized Gamma kernel.

Outputs
-------
results/final_figures/observation_model/
    bayesian_infection_to_death_delay.png
    bayesian_infection_to_death_delay.pdf

Usage
-----
    python scripts/plot_bayesian_infection_to_death_delay.py
"""

from __future__ import annotations

import sys
from math import gamma as gamma_function
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from observation import make_delay_pmf_weeks  # noqa: E402


# =============================================================================
# SETTINGS
# =============================================================================

# These must match the defaults used by PlagueLikelihood in likelihood.py
# for the final Bayesian model fits.
DELAY_KIND = "gamma"
DELAY_MAX_WEEKS = 12
DELAY_SHAPE = 2.0
DELAY_SCALE = 0.6


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "observation_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PNG = (
    OUTPUT_DIR
    / "bayesian_infection_to_death_delay.png"
)

OUT_PDF = (
    OUTPUT_DIR
    / "bayesian_infection_to_death_delay.pdf"
)


# =============================================================================
# Plot styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 11,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
})


# =============================================================================
# Construct EXACT discrete delay PMF used by the Bayesian model
# =============================================================================

f_delay = make_delay_pmf_weeks(
    kind=DELAY_KIND,
    max_weeks=DELAY_MAX_WEEKS,
    shape=DELAY_SHAPE,
    scale=DELAY_SCALE,
)

f_delay = np.asarray(
    f_delay,
    dtype=float,
)


# =============================================================================
# Discrete lag-index grid and midpoint grid
# =============================================================================

# Integer lag indices used by the weekly convolution.
lags = np.arange(
    len(f_delay),
    dtype=float,
)

# Midpoints at which the continuous Gamma density is evaluated to construct
# the discrete weights.
midpoints = (
    lags
    + 0.5
)


# =============================================================================
# Validation
# =============================================================================

if not np.all(
    np.isfinite(f_delay)
):
    raise RuntimeError(
        "Delay PMF contains non-finite values."
    )

if np.any(
    f_delay < 0
):
    raise RuntimeError(
        "Delay PMF contains negative probabilities."
    )

pmf_sum = float(
    np.sum(f_delay)
)

if not np.isclose(
    pmf_sum,
    1.0,
    atol=1e-10,
):
    raise RuntimeError(
        f"Delay PMF does not sum to 1 "
        f"(sum={pmf_sum:.12f})."
    )


# =============================================================================
# Integer convolution-lag statistics
#
# These summarize the indices j used operationally in
#
#     mu_D[w] = psi * sum_j C_week[w-j] * f_delay[j]
#
# They are retained as diagnostics, but should not be interpreted directly
# as the mean biological infection-to-death delay.
# =============================================================================

mean_lag_index = float(
    np.sum(
        lags
        * f_delay
    )
)

variance_lag_index = float(
    np.sum(
        (
            lags
            - mean_lag_index
        ) ** 2
        * f_delay
    )
)

sd_lag_index = float(
    np.sqrt(
        variance_lag_index
    )
)

cdf = np.cumsum(
    f_delay
)

median_lag_index = float(
    lags[
        np.searchsorted(
            cdf,
            0.50,
        )
    ]
)

q025_lag_index = float(
    lags[
        np.searchsorted(
            cdf,
            0.025,
        )
    ]
)

q975_lag_index = float(
    lags[
        np.searchsorted(
            cdf,
            0.975,
        )
    ]
)


# =============================================================================
# Midpoint-based delay statistics
#
# These summarize the actual midpoint locations j + 0.5 weeks at which the
# Gamma density was evaluated to construct each discrete weight.
#
# This is the more interpretable summary of the infection-to-death delay
# represented by the discretized Gamma kernel.
# =============================================================================

mean_midpoint_delay = float(
    np.sum(
        midpoints
        * f_delay
    )
)

variance_midpoint_delay = float(
    np.sum(
        (
            midpoints
            - mean_midpoint_delay
        ) ** 2
        * f_delay
    )
)

sd_midpoint_delay = float(
    np.sqrt(
        variance_midpoint_delay
    )
)

median_midpoint_delay = float(
    midpoints[
        np.searchsorted(
            cdf,
            0.50,
        )
    ]
)

q025_midpoint_delay = float(
    midpoints[
        np.searchsorted(
            cdf,
            0.025,
        )
    ]
)

q975_midpoint_delay = float(
    midpoints[
        np.searchsorted(
            cdf,
            0.975,
        )
    ]
)


# =============================================================================
# Continuous Gamma distribution for reference
#
# IMPORTANT:
# This curve is shown only to illustrate the underlying continuous Gamma
# shape from which the model's discrete weekly weights are constructed.
#
# The fitted Bayesian likelihood itself uses f_delay above.
# =============================================================================

x = np.linspace(
    0.001,
    DELAY_MAX_WEEKS + 1.0,
    1000,
)

continuous_pdf = (
    (
        x ** (DELAY_SHAPE - 1.0)
    )
    *
    np.exp(
        -x / DELAY_SCALE
    )
    /
    (
        gamma_function(DELAY_SHAPE)
        *
        DELAY_SCALE ** DELAY_SHAPE
    )
)

continuous_mean = (
    DELAY_SHAPE
    * DELAY_SCALE
)


# =============================================================================
# Print model quantities
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BAYESIAN INFECTION-TO-DEATH DELAY KERNEL"
)

print(
    "=" * 80
)

print(
    f"\nDelay kind: "
    f"{DELAY_KIND}"
)

print(
    f"Gamma shape: "
    f"{DELAY_SHAPE}"
)

print(
    f"Gamma scale: "
    f"{DELAY_SCALE} weeks"
)

print(
    f"Maximum discrete lag index represented: "
    f"{int(lags[-1])}"
)

print(
    f"\nPMF sum: "
    f"{pmf_sum:.12f}"
)


# -----------------------------------------------------------------------------
# Midpoint-based interpretation
# -----------------------------------------------------------------------------

print(
    "\nMidpoint-based delay summary:"
)

print(
    f"  Mean midpoint-based delay: "
    f"{mean_midpoint_delay:.3f} weeks"
)

print(
    f"  Median midpoint-based delay: "
    f"{median_midpoint_delay:.3f} weeks"
)

print(
    f"  SD midpoint-based delay: "
    f"{sd_midpoint_delay:.3f} weeks"
)

print(
    f"  Central 95% midpoint interval: "
    f"{q025_midpoint_delay:.3f}–"
    f"{q975_midpoint_delay:.3f} weeks"
)


# -----------------------------------------------------------------------------
# Integer convolution-index diagnostics
# -----------------------------------------------------------------------------

print(
    "\nOperational convolution-index summary:"
)

print(
    f"  Mean lag index: "
    f"{mean_lag_index:.3f}"
)

print(
    f"  Median lag index: "
    f"{median_lag_index:g}"
)

print(
    f"  SD lag index: "
    f"{sd_lag_index:.3f}"
)

print(
    f"  Central 95% lag-index interval: "
    f"{q025_lag_index:g}–"
    f"{q975_lag_index:g}"
)


# -----------------------------------------------------------------------------
# Continuous Gamma reference
# -----------------------------------------------------------------------------

print(
    "\nContinuous Gamma reference:"
)

print(
    f"  Continuous Gamma mean: "
    f"{continuous_mean:.3f} weeks"
)

print(
    "\nDifference between continuous mean "
    "and midpoint-weighted discrete mean:"
)

print(
    f"  {continuous_mean - mean_midpoint_delay:.3f} weeks"
)


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9.5,
        5.8,
    )
)


# -----------------------------------------------------------------------------
# Continuous Gamma density
# -----------------------------------------------------------------------------

ax.plot(
    x,
    continuous_pdf,
    linestyle="-",
    linewidth=2.0,
    label="Underlying continuous Gamma density",
)


# -----------------------------------------------------------------------------
# Exact discrete probabilities used by the Bayesian likelihood
# -----------------------------------------------------------------------------

ax.bar(
    lags,
    f_delay,
    width=0.65,
    alpha=0.55,
    label="Discrete weekly weights used in likelihood",
)

ax.plot(
    lags,
    f_delay,
    marker="o",
    linestyle="--",
    linewidth=1.4,
)


# -----------------------------------------------------------------------------
# Show midpoint locations used to construct the discrete weights
# -----------------------------------------------------------------------------

ax.scatter(
    midpoints,
    np.zeros_like(
        midpoints
    ),
    marker="|",
    s=90,
    alpha=0.55,
    label="Gamma evaluation midpoints",
)


# -----------------------------------------------------------------------------
# Mean midpoint-based delay
# -----------------------------------------------------------------------------

ax.axvline(
    mean_midpoint_delay,
    linestyle="--",
    linewidth=1.4,
    label=(
        f"Mean midpoint-based delay = "
        f"{mean_midpoint_delay:.2f} weeks"
    ),
)


# =============================================================================
# Labels
# =============================================================================

ax.set_xlabel(
    "Weekly infection-to-death lag interval"
)

ax.set_ylabel(
    "Probability / density"
)

ax.set_title(
    "Infection-to-death delay kernel used in the Bayesian observation model"
)

ax.set_xlim(
    -0.5,
    DELAY_MAX_WEEKS
    + 0.5,
)

ax.set_xticks(
    np.arange(
        0,
        DELAY_MAX_WEEKS + 1,
        1,
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

ax.grid(
    axis="y",
    alpha=0.22,
    linewidth=0.6,
)

ax.legend(
    frameon=False,
)


# =============================================================================
# Save
# =============================================================================

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.show()

plt.close(
    fig
)


# =============================================================================
# Report
# =============================================================================

print(
    "\nSaved:"
)

print(
    f"  {OUT_PNG}"
)

print(
    f"  {OUT_PDF}"
)

print(
    "\nDone."
)