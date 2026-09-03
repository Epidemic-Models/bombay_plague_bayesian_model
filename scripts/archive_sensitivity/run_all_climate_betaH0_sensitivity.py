# =============================================================================
# run_all_climate_betaH0_sensitivity.py
# =============================================================================
"""
Targeted prior-sensitivity analysis for betaH0 in the all-climate model.

Purpose
-------
The round-2 all-climate posterior showed some accumulation near the upper
prior bound for betaH0:

    betaH0 ~ Uniform(0.01, 1.5)

This script repeats the all-climate nested-sampling analysis while changing
ONLY the betaH0 upper bound:

    betaH0 ~ Uniform(0.01, 3.0)

All other priors are kept identical to the round-2 prior-sensitivity run.

This allows us to determine whether the posterior for betaH0 was being
artificially truncated by the previous upper bound of 1.5.

Outputs are saved separately under:

    results/betaH0_prior_sensitivity/
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import dynesty
import numpy as np
import pandas as pd


# =============================================================================
# Project imports
# =============================================================================

# This script is now located in:
# project_root/scripts/archive_sensitivity/
#
# parents[0] = archive_sensitivity
# parents[1] = scripts
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import paths
from likelihood import PlagueLikelihood

# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
BETA_H_MODE = "all_climate"

# Nested-sampling settings.
# These should match the settings used in your previous runs.
NLIVE = 1000
DLOGZ = 0.1

# Random seed for reproducibility
RANDOM_SEED = 12345


# =============================================================================
# Output directories
# =============================================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "betaH0_prior_sensitivity"
)

NESTED_DIR = OUTPUT_ROOT / "nested_sampling"

NESTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Round-2 priors
# =============================================================================
#
# IMPORTANT:
# These are the priors from your round-2 all-climate analysis.
#
# The ONLY change in this experiment is:
#
#       betaH0: (0.01, 1.5) --> (0.01, 3.0)
#
# =============================================================================

PRIORS = {
    "N0":       (800000.0, 850000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 3.0),
    "a0":       (-11.0, -5.5),
    "aT":       (-2.0, 1.0),
    "aTT":      (-2.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-4.0, 1.0),

    # ---------------------------------------------------------
    # TARGETED SENSITIVITY CHANGE
    # ---------------------------------------------------------
    "betaH0":   (0.01, 3.0),

    # Keep psi bounded by 1.
    "psi":      (0.05, 1.0),

    "nb_kappa": (0.1, 80.0),

    # Human-to-human climate coefficients
    "bT_H":     (-2.0, 2.0),
    "be_H":     (-2.0, 2.0),
    "bR_H":     (-2.0, 2.0),
}


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


print()
print("Loaded prepared data:")
print(paths.MERGED_CSV)
print(f"Number of fitted weeks: {len(merged_df)}")
print(f"Target: {FIT_TARGET}")


# =============================================================================
# Construct likelihood
# =============================================================================

lik = PlagueLikelihood(
    merged_df=merged_df,
    betaH_mode=BETA_H_MODE,
    fit_target=FIT_TARGET,
)


# =============================================================================
# Parameter names
# =============================================================================

# Use the likelihood object's own parameter ordering.
#
# Depending on how your PlagueLikelihood class is written, the parameter
# names may be stored under one of these attributes.
#
# We try the common possibilities automatically.

if hasattr(lik, "param_names"):
    param_names = list(lik.param_names)

elif hasattr(lik, "parameter_names"):
    param_names = list(lik.parameter_names)

else:
    # Explicit ordering from your previous all-climate runs
    param_names = [
        "N0",
        "C_seed",
        "gamma",
        "a0",
        "aT",
        "aTT",
        "ae",
        "aR",
        "betaH0",
        "psi",
        "nb_kappa",
        "bT_H",
        "be_H",
        "bR_H",
    ]


ndim = len(param_names)


# =============================================================================
# Sanity checks
# =============================================================================

missing_priors = [
    name
    for name in param_names
    if name not in PRIORS
]

if missing_priors:
    raise KeyError(
        "Missing prior definitions for:\n"
        + "\n".join(missing_priors)
    )


if ndim != 14:
    print(
        f"[WARNING] Expected 14 all-climate parameters "
        f"but found {ndim}."
    )


# =============================================================================
# Prior transform
# =============================================================================

def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
    """
    Transform a point from the unit hypercube [0,1]^D into the physical
    parameter space using independent uniform priors.
    """

    theta = np.empty(ndim, dtype=float)

    for i, name in enumerate(param_names):

        lower, upper = PRIORS[name]

        theta[i] = (
            lower
            + unit_cube[i] * (upper - lower)
        )

    return theta


# =============================================================================
# Log-likelihood
# =============================================================================

def loglikelihood(theta: np.ndarray) -> float:
    """
    Evaluate the plague model log-likelihood.
    """

    try:

        value = lik.log_likelihood(theta)

    except AttributeError:

        # Some versions of the project may make the likelihood object
        # directly callable.
        value = lik(theta)

    if not np.isfinite(value):
        return -1.0e300

    return float(value)


# =============================================================================
# Print run information
# =============================================================================

print()
print("=" * 88)
print("TARGETED betaH0 PRIOR-SENSITIVITY RUN: all_climate")
print("=" * 88)

print()
print(f"Free parameters ({ndim}):")
print(param_names)

print()
print("Prior ranges")
print("-" * 88)

for name in param_names:

    lower, upper = PRIORS[name]

    marker = ""

    if name == "betaH0":
        marker = "  <-- TARGETED WIDENING: previous upper bound = 1.5"

    print(
        f"{name:<12s} "
        f"({lower:g}, {upper:g})"
        f"{marker}"
    )


print()
print("=" * 88)
print("IMPORTANT")
print("=" * 88)

print(
    """
This experiment changes ONLY betaH0 relative to the round-2 analysis:

    previous:
        betaH0 ~ Uniform(0.01, 1.5)

    current:
        betaH0 ~ Uniform(0.01, 3.0)

The purpose is to determine whether the betaH0 posterior was truncated by
the previous upper prior boundary.
"""
)


# =============================================================================
# Random-number generator
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# Run nested sampling
# =============================================================================

print("=" * 88)
print("Starting dynesty...")
print("=" * 88)
print()

start_time = time.time()


sampler = dynesty.NestedSampler(
    loglikelihood,
    prior_transform,
    ndim,
    nlive=NLIVE,
    bound="multi",
    sample="rwalk",
    rstate=rng,
)


sampler.run_nested(
    dlogz=DLOGZ,
    print_progress=True,
)


results = sampler.results


elapsed_seconds = time.time() - start_time
elapsed_minutes = elapsed_seconds / 60.0


# =============================================================================
# Results
# =============================================================================

logZ = float(results.logz[-1])
logZ_error = float(results.logzerr[-1])

max_index = int(np.argmax(results.logl))

max_loglik = float(
    results.logl[max_index]
)

theta_max = np.asarray(
    results.samples[max_index],
    dtype=float,
)


print()
print("=" * 88)
print("RUN COMPLETE")
print("=" * 88)

print(
    f"Finished in {elapsed_minutes:.1f} min"
)

print(
    f"Maximum sampled log-likelihood = "
    f"{max_loglik:.3f}"
)

print(
    f"log(Z) = "
    f"{logZ:.3f} +/- {logZ_error:.3f}"
)


# =============================================================================
# Highest-likelihood parameter vector
# =============================================================================

print()
print("Highest-likelihood parameter vector")
print("-" * 60)

for name, value in zip(
    param_names,
    theta_max,
):

    print(
        f"{name:<12s} = {value:.6g}"
    )


# =============================================================================
# Save result
# =============================================================================

stem = (
    "all_climate__"
    f"{FIT_TARGET}__"
    "betaH0_upper3"
)

pkl_path = (
    NESTED_DIR
    / f"{stem}.pkl"
)

json_path = (
    NESTED_DIR
    / f"{stem}.json"
)


saved = {
    "results": results,
    "param_names": param_names,
    "betaH_mode": BETA_H_MODE,
    "fit_target": FIT_TARGET,
    "priors": PRIORS,
    "experiment": "betaH0_upper_bound_sensitivity",
    "previous_betaH0_prior": (0.01, 1.5),
    "current_betaH0_prior": (0.01, 3.0),
}


with open(
    pkl_path,
    "wb",
) as f:

    pickle.dump(
        saved,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


# =============================================================================
# Save metadata
# =============================================================================

metadata = {
    "betaH_mode": BETA_H_MODE,
    "fit_target": FIT_TARGET,
    "n_parameters": ndim,
    "param_names": param_names,

    "n_fitted_weeks": int(
        len(merged_df)
    ),

    "nlive": NLIVE,
    "dlogz": DLOGZ,
    "random_seed": RANDOM_SEED,

    "elapsed_minutes": elapsed_minutes,

    "max_loglik": max_loglik,
    "logZ": logZ,
    "logZ_error": logZ_error,

    "experiment": (
        "targeted betaH0 upper-prior "
        "sensitivity"
    ),

    "previous_betaH0_prior": [
        0.01,
        1.5,
    ],

    "current_betaH0_prior": [
        0.01,
        3.0,
    ],

    "priors": {
        name: [
            float(bounds[0]),
            float(bounds[1]),
        ]
        for name, bounds
        in PRIORS.items()
    },

    "highest_likelihood_parameters": {
        name: float(value)
        for name, value
        in zip(
            param_names,
            theta_max,
        )
    },
}


with open(
    json_path,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print()
print("[OK] Saved nested-sampling result:")
print(pkl_path)

print()
print("[OK] Saved metadata:")
print(json_path)


# =============================================================================
# Very quick betaH0 check
# =============================================================================

beta_index = param_names.index(
    "betaH0"
)

weights = np.exp(
    results.logwt
    - results.logz[-1]
)

weights /= np.sum(weights)


# Weighted quantile helper
def weighted_quantile(
    values,
    quantiles,
    sample_weight,
):

    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)

    sorter = np.argsort(values)

    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cumulative = np.cumsum(
        sample_weight
    )

    cumulative /= cumulative[-1]

    return np.interp(
        quantiles,
        cumulative,
        values,
    )


beta_samples = results.samples[
    :,
    beta_index
]


beta_q025, beta_median, beta_q975 = (
    weighted_quantile(
        beta_samples,
        [0.025, 0.5, 0.975],
        weights,
    )
)


prob_gt_15 = np.sum(
    weights[
        beta_samples > 1.5
    ]
)

prob_gt_20 = np.sum(
    weights[
        beta_samples > 2.0
    ]
)

prob_gt_25 = np.sum(
    weights[
        beta_samples > 2.5
    ]
)

prob_gt_29 = np.sum(
    weights[
        beta_samples > 2.9
    ]
)


print()
print("=" * 88)
print("QUICK betaH0 POSTERIOR CHECK")
print("=" * 88)

print(
    f"Posterior median betaH0 = "
    f"{beta_median:.4f}"
)

print(
    f"95% CrI = "
    f"({beta_q025:.4f}, "
    f"{beta_q975:.4f})"
)

print()
print(
    f"P(betaH0 > 1.5 | data) = "
    f"{prob_gt_15:.4f}"
)

print(
    f"P(betaH0 > 2.0 | data) = "
    f"{prob_gt_20:.4f}"
)

print(
    f"P(betaH0 > 2.5 | data) = "
    f"{prob_gt_25:.4f}"
)

print(
    f"P(betaH0 > 2.9 | data) = "
    f"{prob_gt_29:.4f}"
)


print()
print("=" * 88)
print("INTERPRETATION GUIDE")
print("=" * 88)

print(
    """
After this run, look especially at the betaH0 posterior:

1. If the posterior peaks below 1.5 and declines well before 3.0:
       Good.
       The old 1.5 boundary was restrictive, but the new posterior is
       internally contained.

2. If substantial posterior probability is above 1.5:
       This confirms that the old betaH0 prior truncated the posterior.

3. If the posterior now piles up near 3.0:
       Do NOT simply keep widening it indefinitely.
       That would suggest betaH0 is weakly identified or that the model
       contains a parameter trade-off that needs investigation.

4. Compare log(Z) cautiously.
       Because the prior volume has changed, Bayesian evidence itself
       changes when the betaH0 prior is widened. The main purpose of this
       run is posterior sensitivity, not declaring a new winning model.
"""
)

print()
print(
    "Targeted betaH0 prior-sensitivity "
    "analysis completed successfully."
)