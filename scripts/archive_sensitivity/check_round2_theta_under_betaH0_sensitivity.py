# =============================================================================
# check_round2_theta_under_betaH0_sensitivity.py
# =============================================================================
"""
Diagnostic:

Take the highest-likelihood parameter vector from the previous ROUND-2
all-climate nested-sampling run and evaluate exactly that parameter vector
under the likelihood used for the betaH0 sensitivity analysis.

Expected logic:

Round-2 best:
    logL ~= -780.259

Because betaH0 was only widened from [0.01, 1.5] to [0.01, 3.0],
the old best parameter vector is still legal under the new prior.

Therefore, if the likelihood/model is otherwise unchanged, evaluating the
old theta should still give approximately -780.259.

If it does:
    -> likelihood/model are consistent
    -> the new dynesty run failed to locate the high-likelihood region

If it does NOT:
    -> something differs between the two likelihood/model configurations
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


# =============================================================================
# File containing the successful ROUND-2 result
# =============================================================================

ROUND2_PKL = (
    PROJECT_ROOT
    / "results"
    / "prior_sensitivity_round2"
    / "nested_sampling"
    / "all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl"
)


# =============================================================================
# Load data
# =============================================================================

print("=" * 90)
print("CHECK ROUND-2 BEST THETA UNDER betaH0-SENSITIVITY LIKELIHOOD")
print("=" * 90)

print("\nLoading prepared data:")
print(paths.MERGED_CSV)

merged_df = pd.read_csv(
    paths.MERGED_CSV,
    parse_dates=[paths.DATE_COL],
)

merged_df = (
    merged_df
    .sort_values(paths.DATE_COL)
    .reset_index(drop=True)
)

print(f"Number of fitted weeks: {len(merged_df)}")


# =============================================================================
# Load old round-2 nested-sampling result
# =============================================================================

if not ROUND2_PKL.exists():
    raise FileNotFoundError(
        f"Could not find round-2 result:\n{ROUND2_PKL}"
    )

print("\nLoading round-2 result:")
print(ROUND2_PKL)

with open(ROUND2_PKL, "rb") as f:
    saved = pickle.load(f)

results = saved["results"]
param_names_old = list(saved["param_names"])


# =============================================================================
# Find highest-likelihood sample from round 2
# =============================================================================

best_index = int(np.argmax(results.logl))

theta_old = np.asarray(
    results.samples[best_index],
    dtype=float,
)

stored_best_logL = float(results.logl[best_index])

print("\n" + "=" * 90)
print("ROUND-2 HIGHEST-LIKELIHOOD SAMPLE")
print("=" * 90)

print(f"\nStored round-2 maximum logL = {stored_best_logL:.6f}")

print("\nParameter vector:")
print("-" * 60)

for name, value in zip(param_names_old, theta_old):
    print(f"{name:12s} = {value:.10g}")


# =============================================================================
# Check betaH0
# =============================================================================

beta_index = param_names_old.index("betaH0")
beta_old = theta_old[beta_index]

print("\n" + "=" * 90)
print("betaH0 CHECK")
print("=" * 90)

print(f"\nRound-2 best betaH0 = {beta_old:.10g}")

print("\nOld bound:")
print("    betaH0 in [0.01, 1.5]")

print("\nNew sensitivity bound:")
print("    betaH0 in [0.01, 3.0]")

if 0.01 <= beta_old <= 3.0:
    print("\n[OK] Old betaH0 is inside the new wider range.")
else:
    print("\n[ERROR] Old betaH0 is outside the new range.")


# =============================================================================
# Construct likelihood object
# =============================================================================
#
# IMPORTANT:
#
# PlagueLikelihood normally obtains its bounds from priors.py.
#
# We want the same mechanistic model and same NB likelihood, but with the
# betaH0 upper bound changed to 3.0.
#
# We therefore construct the normal likelihood object and then change ONLY
# the betaH0 bound locally in this diagnostic object.
#
# =============================================================================

lik = PlagueLikelihood(
    merged_df=merged_df,
    betaH_mode=BETA_H_MODE,
    fit_target=FIT_TARGET,
)


# =============================================================================
# Make sure parameter ordering is identical
# =============================================================================

print("\n" + "=" * 90)
print("PARAMETER ORDER CHECK")
print("=" * 90)

print("\nOld result:")
print(param_names_old)

print("\nCurrent likelihood:")
print(lik.param_names)

if list(param_names_old) != list(lik.param_names):
    raise RuntimeError(
        "\nParameter ordering differs!\n"
        "Do NOT evaluate theta until this is resolved."
    )

print("\n[OK] Parameter ordering is identical.")


# =============================================================================
# Change ONLY betaH0 upper bound in this likelihood object
# =============================================================================

new_bounds = list(lik.bounds)

beta_index_new = lik.param_names.index("betaH0")

old_bound_current = new_bounds[beta_index_new]

new_bounds[beta_index_new] = (0.01, 3.0)

lik.bounds = new_bounds

print("\n" + "=" * 90)
print("LIKELIHOOD BOUND USED FOR DIAGNOSTIC")
print("=" * 90)

print(f"\nOriginal betaH0 bound in likelihood object: {old_bound_current}")
print(f"Diagnostic betaH0 bound:               {lik.bounds[beta_index_new]}")


# =============================================================================
# Evaluate EXACT old theta
# =============================================================================

print("\n" + "=" * 90)
print("REEVALUATING EXACT ROUND-2 BEST THETA")
print("=" * 90)

value = lik.log_likelihood(theta_old)

print(f"\nStored round-2 logL:")
print(f"    {stored_best_logL:.10f}")

print(f"\nRecalculated logL:")
print(f"    {value:.10f}")

difference = value - stored_best_logL

print(f"\nDifference:")
print(f"    {difference:.10f}")


# =============================================================================
# Interpretation
# =============================================================================

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)

if not np.isfinite(value):

    print(
        """
RESULT: The old parameter vector gives -inf.

This means the old parameter vector is being rejected somewhere in the
current likelihood/model configuration.

We need to identify which parameter/bound or simulation condition causes it.
"""
    )

elif abs(difference) < 1e-6:

    print(
        """
RESULT: PERFECT MATCH.

The old round-2 parameter vector gives essentially exactly the same
log-likelihood under the current likelihood.

Therefore the likelihood calculation itself is consistent.

Because this parameter vector is inside betaH0 <= 3, the new parameter
space definitely contains a solution with logL approximately equal to the
old maximum.

If the new dynesty run only found maximum logL around -805, then dynesty
failed to discover this high-likelihood region in that run.
"""
    )

elif abs(difference) < 0.1:

    print(
        """
RESULT: VERY CLOSE MATCH.

The tiny difference is unlikely to explain the approximately 25-log-unit
difference between the two nested-sampling runs.

The old high-likelihood region still exists under the widened betaH0 range.
The new nested-sampling run most likely failed to locate it adequately.
"""
    )

else:

    print(
        """
RESULT: IMPORTANT DIFFERENCE.

The same parameter vector does NOT reproduce its old log-likelihood.

Therefore something besides the betaH0 prior range differs between the
round-2 analysis and the current likelihood/model configuration.

Do NOT rerun dynesty yet.

We should next compare:
    * model settings
    * initial conditions
    * delay distribution
    * fitted data
    * climate arrays
    * parameter mapping
    * likelihood implementation
"""
    )


print("\nDiagnostic finished.")