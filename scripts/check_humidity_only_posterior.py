# =============================================================================
# check_humidity_only_posterior.py
# =============================================================================
"""
Extract the posterior summary for the vapour-pressure coefficient be_H
from the existing humidity_only final_v2 Dynesty result.

NO MODEL IS REFITTED.

Reports:
    posterior median
    95% credible interval
    P(be_H < 0 | data)
"""

from pathlib import Path
import pickle

import numpy as np
from dynesty.utils import resample_equal


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIT_TARGET = "plague_deaths_reported_fit"

PKL_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
    / "nested_sampling"
    / f"humidity_only__{FIT_TARGET}__final_v2.pkl"
)


# =============================================================================
# Reproducible posterior-resampling seed
# =============================================================================

RANDOM_SEED_POSTERIOR = 55321


# =============================================================================
# Load existing Dynesty result
# =============================================================================

if not PKL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find:\n{PKL_PATH}"
    )

print("\nLoading:")
print(PKL_PATH)

with open(PKL_PATH, "rb") as f:
    saved = pickle.load(f)


results = saved["results"]
param_names = list(saved["param_names"])


print("\nParameters:")
print(param_names)


# =============================================================================
# Check that be_H is present
# =============================================================================

if "be_H" not in param_names:
    raise RuntimeError(
        "'be_H' was not found in the humidity_only parameter list."
    )

be_idx = param_names.index("be_H")


# =============================================================================
# Construct equal-weight posterior
# =============================================================================

weights = np.exp(
    results.logwt
    - results.logz[-1]
)

rng = np.random.default_rng(
    RANDOM_SEED_POSTERIOR
)

posterior = resample_equal(
    results.samples,
    weights,
    rstate=rng,
)

posterior = np.asarray(
    posterior,
    dtype=float,
)

print(
    f"\nEqual-weight posterior samples: "
    f"{len(posterior)}"
)


# =============================================================================
# Extract be_H
# =============================================================================

be_vals = posterior[:, be_idx]

if not np.all(np.isfinite(be_vals)):
    raise RuntimeError(
        "be_H posterior contains non-finite values."
    )


# =============================================================================
# Posterior summary
# =============================================================================

median = float(
    np.quantile(be_vals, 0.50)
)

q025 = float(
    np.quantile(be_vals, 0.025)
)

q975 = float(
    np.quantile(be_vals, 0.975)
)

p_negative = float(
    np.mean(be_vals < 0.0)
)

p_positive = float(
    np.mean(be_vals > 0.0)
)


# =============================================================================
# Also extract MLE for comparison with Table S3
# =============================================================================

logl = np.asarray(
    results.logl,
    dtype=float,
)

samples = np.asarray(
    results.samples,
    dtype=float,
)

best_idx = int(
    np.nanargmax(logl)
)

mle_be = float(
    samples[best_idx, be_idx]
)

best_loglik = float(
    logl[best_idx]
)


# =============================================================================
# Report
# =============================================================================

print("\n" + "=" * 80)
print("VAPOUR-PRESSURE-ONLY POSTERIOR")
print("=" * 80)

print(
    f"\nPosterior median be_H = "
    f"{median:.4f}"
)

print(
    f"95% CrI = "
    f"({q025:.4f}, {q975:.4f})"
)

print(
    f"P(be_H < 0 | data) = "
    f"{p_negative:.4f}"
)

print(
    f"P(be_H > 0 | data) = "
    f"{p_positive:.4f}"
)


print("\n" + "-" * 80)
print("MLE / TABLE S3 CHECK")
print("-" * 80)

print(
    f"\nMLE be_H = "
    f"{mle_be:.4f}"
)

print(
    f"Maximum logL = "
    f"{best_loglik:.6f}"
)

print(
    "\nNOTE:"
    "\n  The MLE is the value appropriate for Table S3."
    "\n  The posterior median + 95% CrI + sign probability are"
    "\n  the values appropriate for the Results/Discussion."
)

print("\nDone.")