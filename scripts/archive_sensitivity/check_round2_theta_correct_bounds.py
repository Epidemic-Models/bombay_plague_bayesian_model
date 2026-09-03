# =============================================================================
# check_round2_theta_correct_bounds.py
# =============================================================================
"""
Diagnostic test:

Take the exact highest-likelihood theta from the successful round-2
all-climate run and evaluate it again using:

    - the same data
    - the same mechanistic model
    - the same NB likelihood
    - ALL round-2 widened bounds
    - betaH0 additionally widened from 1.5 to 3.0

If everything is internally consistent, the recalculated log-likelihood
should be approximately the stored round-2 value:

    logL ~= -780.25928
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

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

ROUND2_PKL = (
    PROJECT_ROOT
    / "results"
    / "prior_sensitivity_round2"
    / "nested_sampling"
    / "all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl"
)


# =============================================================================
# Correct sensitivity bounds
# =============================================================================
#
# These are the ROUND-2 bounds, except betaH0 is widened from 1.5 to 3.0.
#
# IMPORTANT:
# The ordering must correspond exactly to lik.param_names.
# =============================================================================

SENSITIVITY_BOUNDS = {
    "N0":       (800000.0, 850000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 3.0),
    "a0":       (-11.0, -5.5),
    "aT":       (-2.0, 1.0),
    "aTT":      (-2.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-4.0, 1.0),

    # Only additional change relative to round 2:
    "betaH0":   (0.01, 3.0),

    "psi":      (0.05, 1.0),
    "nb_kappa": (0.1, 80.0),

    "bT_H":     (-2.0, 2.0),
    "be_H":     (-2.0, 2.0),
    "bR_H":     (-2.0, 2.0),
}


# =============================================================================
# Load prepared data
# =============================================================================

print("=" * 90)
print("CHECK ROUND-2 THETA WITH CORRECT SENSITIVITY BOUNDS")
print("=" * 90)

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(
        f"Prepared data not found:\n{paths.MERGED_CSV}"
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

print(f"\nData:")
print(paths.MERGED_CSV)

print(f"\nNumber of fitted weeks: {len(merged_df)}")


# =============================================================================
# Load round-2 result
# =============================================================================

if not ROUND2_PKL.exists():
    raise FileNotFoundError(
        f"Round-2 result not found:\n{ROUND2_PKL}"
    )

print("\nLoading successful round-2 result:")
print(ROUND2_PKL)

with open(ROUND2_PKL, "rb") as f:
    saved = pickle.load(f)

results = saved["results"]
old_param_names = list(saved["param_names"])


# =============================================================================
# Extract exact highest-likelihood sample
# =============================================================================

best_index = int(np.argmax(results.logl))

theta_old = np.asarray(
    results.samples[best_index],
    dtype=float,
)

stored_logL = float(results.logl[best_index])


print("\n" + "=" * 90)
print("ROUND-2 BEST SAMPLE")
print("=" * 90)

print(f"\nStored maximum logL = {stored_logL:.10f}")

for name, value in zip(old_param_names, theta_old):
    print(f"{name:12s} = {value:.10g}")


# =============================================================================
# Construct current likelihood
# =============================================================================

lik = PlagueLikelihood(
    merged_df=merged_df,
    betaH_mode=BETA_H_MODE,
    fit_target=FIT_TARGET,
)


# =============================================================================
# Check parameter ordering
# =============================================================================

print("\n" + "=" * 90)
print("PARAMETER ORDER CHECK")
print("=" * 90)

print("\nSaved round-2 ordering:")
print(old_param_names)

print("\nCurrent likelihood ordering:")
print(lik.param_names)

if list(old_param_names) != list(lik.param_names):
    raise RuntimeError(
        "Parameter ordering differs. Stop here."
    )

print("\n[OK] Parameter ordering is identical.")


# =============================================================================
# Display bounds currently coming from priors.py
# =============================================================================

print("\n" + "=" * 90)
print("DEFAULT BOUNDS CURRENTLY USED BY PlagueLikelihood")
print("=" * 90)

for name, bound in zip(lik.param_names, lik.bounds):
    print(f"{name:12s} {bound}")


# =============================================================================
# Replace bounds with complete sensitivity bounds
# =============================================================================

missing = [
    name
    for name in lik.param_names
    if name not in SENSITIVITY_BOUNDS
]

if missing:
    raise RuntimeError(
        f"Missing sensitivity bounds for: {missing}"
    )

correct_bounds = [
    SENSITIVITY_BOUNDS[name]
    for name in lik.param_names
]

lik.bounds = correct_bounds


print("\n" + "=" * 90)
print("BOUNDS USED FOR THIS DIAGNOSTIC")
print("=" * 90)

for name, bound in zip(lik.param_names, lik.bounds):
    print(f"{name:12s} {bound}")


# =============================================================================
# Explicitly check every old parameter against new bounds
# =============================================================================

print("\n" + "=" * 90)
print("BOUND CHECK FOR ROUND-2 BEST THETA")
print("=" * 90)

outside = []

for name, value, (lo, hi) in zip(
    lik.param_names,
    theta_old,
    lik.bounds,
):
    inside = lo <= value <= hi

    status = "OK" if inside else "OUTSIDE"

    print(
        f"{name:12s} "
        f"value={value:14.7g}   "
        f"bounds=({lo:g}, {hi:g})   "
        f"{status}"
    )

    if not inside:
        outside.append(
            (name, value, lo, hi)
        )


if outside:

    print("\nERROR: Some parameters are still outside the bounds:")

    for name, value, lo, hi in outside:
        print(
            f"    {name}: {value} not in ({lo}, {hi})"
        )

    raise RuntimeError(
        "Cannot perform likelihood comparison because theta is outside bounds."
    )

else:
    print("\n[OK] EVERY parameter is inside the sensitivity bounds.")


# =============================================================================
# Evaluate exact old theta
# =============================================================================

print("\n" + "=" * 90)
print("LIKELIHOOD TEST")
print("=" * 90)

recalculated_logL = lik.log_likelihood(theta_old)

print(f"\nStored round-2 logL:")
print(f"    {stored_logL:.10f}")

print(f"\nRecalculated logL:")
print(f"    {recalculated_logL:.10f}")


if np.isfinite(recalculated_logL):

    difference = recalculated_logL - stored_logL

    print(f"\nDifference:")
    print(f"    {difference:.10f}")

else:

    difference = np.nan

    print("\nDifference cannot be calculated because logL = -inf.")


# =============================================================================
# Final interpretation
# =============================================================================

print("\n" + "=" * 90)
print("FINAL INTERPRETATION")
print("=" * 90)


if not np.isfinite(recalculated_logL):

    print(
        """
The parameter vector is now inside ALL specified sensitivity bounds,
but log_likelihood() still returns -inf.

Therefore the problem is NOT simply the initial bound check.

The next step would be to inspect the simulation itself and determine
where log_likelihood() is producing/rejecting an invalid result.
"""
    )

elif abs(difference) < 1e-6:

    print(
        """
SUCCESS: EXACT MATCH.

The old round-2 parameter vector reproduces its original likelihood.

This demonstrates that the mechanistic model + observation model +
negative-binomial likelihood are consistent.

It also demonstrates that this high-likelihood point remains valid when
betaH0 is widened to 3.0.

Therefore, if the betaH0-sensitivity nested-sampling run only found
maximum logL around -805, the problem is in how that sensitivity run
handled the priors/bounds or explored parameter space.

Do NOT interpret the betaH0-sensitivity logZ = -836.740 as a valid
comparison until that run is corrected.
"""
    )

elif abs(difference) < 0.1:

    print(
        """
The likelihood is essentially reproduced.

The tiny numerical difference cannot explain the approximately 25-unit
difference in maximum log-likelihood between the two nested-sampling runs.

The old high-likelihood region remains valid under the wider betaH0 prior.
"""
    )

else:

    print(
        """
The parameter vector is accepted, but its likelihood is substantially
different from the stored round-2 likelihood.

This means something besides the bounds changed between the analyses.

The next step is to compare the simulator and observation-model settings
used in the two runs.
"""
    )


print("\nDiagnostic completed.")