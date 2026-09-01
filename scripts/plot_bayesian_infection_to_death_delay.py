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

No additional two-week shift is applied.

The plotted discrete probabilities are generated with the exact same
make_delay_pmf_weeks() function used by the Bayesian likelihood.

Because the observation model applies the kernel through

    mu_D[w] = psi * sum_j C_week[w-j] * f_delay[j],

f_delay[j] is operationally associated with a discrete lag of j weeks.

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
# Discrete model-lag grid
# =============================================================================

lags = np.arange(
    len(f_delay),
    dtype=float,
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
# Effective discrete-lag statistics
# =============================================================================

mean_effective_lag = float(
    np.sum(
        lags
        * f_delay
    )
)

variance_effective_lag = float(
    np.sum(
        (
            lags
            - mean_effective_lag
        ) ** 2
        * f_delay
    )
)

sd_effective_lag = float(
    np.sqrt(
        variance_effective_lag
    )
)

cdf = np.cumsum(
    f_delay
)

median_effective_lag = float(
    lags[
        np.searchsorted(
            cdf,
            0.50,
        )
    ]
)

q025_effective_lag = float(
    lags[
        np.searchsorted(
            cdf,
            0.025,
        )
    ]
)

q975_effective_lag = float(
    lags[
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
# This curve is shown only to illustrate the continuous Gamma shape from
# which the model's weekly weights are constructed.
#
# The fitted model itself uses the discrete PMF above.
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


# =============================================================================
# Weekly midpoint locations used to CONSTRUCT the discrete Gamma weights
# =============================================================================

midpoints = (
    lags
    + 0.5
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
    f"\nDelay kind: {DELAY_KIND}"
)

print(
    f"Gamma shape: {DELAY_SHAPE}"
)

print(
    f"Gamma scale: {DELAY_SCALE} weeks"
)

print(
    f"Maximum discrete lag represented: "
    f"{int(lags[-1])} weeks"
)

print(
    "\nNo additional tau_weeks shift is used "
    "for the gamma kernel."
)

print(
    f"\nPMF sum: "
    f"{pmf_sum:.12f}"
)

print(
    "\nEffective discrete lag statistics "
    "(as used in the weekly convolution):"
)

print(
    f"  Mean effective lag: "
    f"{mean_effective_lag:.3f} weeks"
)

print(
    f"  Median effective lag: "
    f"{median_effective_lag:g} weeks"
)

print(
    f"  SD effective lag: "
    f"{sd_effective_lag:.3f} weeks"
)

print(
    f"  Central 95% probability interval: "
    f"{q025_effective_lag:g}–"
    f"{q975_effective_lag:g} weeks"
)

print(
    "\nContinuous Gamma reference:"
)

print(
    f"  Continuous Gamma mean = "
    f"{DELAY_SHAPE * DELAY_SCALE:.3f} weeks"
)


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(9.5, 5.8))

# Continuous Gamma density -- reference only
ax.plot(
    x,
    continuous_pdf,
    linestyle="-",
    linewidth=2.0,
    label="Continuous Gamma density (reference only)",
)

# Exact discrete probabilities used by the likelihood
ax.bar(
    lags,
    f_delay,
    width=0.65,
    alpha=0.55,
    label="Discrete lag probabilities used in likelihood",
)

ax.plot(
    lags,
    f_delay,
    marker="o",
    linestyle="--",
    linewidth=1.4,
)

# Effective mean lag of the discrete convolution kernel
ax.axvline(
    mean_effective_lag,
    linestyle="--",
    linewidth=1.4,
    label=f"Mean effective lag = {mean_effective_lag:.2f} weeks",
)

ax.set_xlabel("Discrete infection-to-death lag (weeks)")
ax.set_ylabel("Probability / density")

ax.set_title(
    "Infection-to-death delay kernel used in the Bayesian observation model"
)

ax.set_xlim(-0.5, DELAY_MAX_WEEKS + 0.5)
ax.set_xticks(np.arange(0, DELAY_MAX_WEEKS + 1, 1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.22,
    linewidth=0.6,
)

ax.legend(frameon=False)

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

plt.close(fig)

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