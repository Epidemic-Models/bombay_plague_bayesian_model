# =============================================================================
# run_all_climate_betaH0_sensitivity_v2.py
# =============================================================================
"""
Second attempt at the betaH0 upper-bound sensitivity run for the all-climate
model. The first attempt (run_all_climate_betaH0_sensitivity.py) converged
to the wrong mode: it found max logL ~ -805.22, ~25 log-units below the
value we independently proved is legal and reachable under these exact
bounds (check_round2_theta_correct_bounds.py got an EXACT match at
logL = -780.2592797235 for the round-2 best theta, re-evaluated under
betaH0 <= 3.0).

Two changes relative to the first attempt:

  1. sample="rslice" instead of "rwalk". Slice sampling tends to be more
     robust than random-walk proposals for higher-dimensional / strongly
     curved-and-correlated posteriors -- which is exactly what this model
     has (a0, aR, betaH0, bR_H trade off against each other).

  2. One live point is SEEDED at the known-good theta (the round-2 best
     sample, independently verified to give logL = -780.2592797235 under
     these bounds). This guarantees the high-likelihood ridge is
     represented in the live-point ensemble from iteration 0, so the
     question this run answers is "does the posterior stay there and
     settle" rather than "can dynesty find it from a cold start."

nlive is kept at 1000 (matching the original runs) as a first, cheap
check. If this still doesn't resolve cleanly, increase nlive (e.g. to
2000) and rerun.

Everything else -- bounds, data, likelihood, fit target -- is identical to
run_final_all_models.py's all_climate configuration, except betaH0's upper
bound is widened from 1.5 to 3.0.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dynesty import NestedSampler


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
BETA_H_MODE = "all_climate"

NLIVE = 1000
DLOGZ = 0.1
RANDOM_SEED = 12345

# The SAME bounds used in run_final_all_models.py for all_climate, except
# betaH0's upper bound widened from 1.5 to 3.0. This must match
# check_round2_theta_correct_bounds.py's SENSITIVITY_BOUNDS exactly, since
# that's the bound set under which the seed point was verified.
BOUNDS = {
    "N0":       (800000.0, 850000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 3.0),
    "a0":       (-11.0, -5.5),
    "aT":       (-2.0, 1.0),
    "aTT":      (-2.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-4.0, 1.0),
    "betaH0":   (0.01, 3.0),   # <-- widened from 1.5
    "psi":      (0.05, 1.0),
    "nb_kappa": (0.1, 80.0),
    "bT_H":     (-2.0, 2.0),
    "be_H":     (-2.0, 2.0),
    "bR_H":     (-2.0, 2.0),
}

# Known-good point: the round-2 best sample. Independently verified via
# check_round2_theta_correct_bounds.py to give EXACT logL = -780.2592797235
# under these bounds (difference = 0.0000000000). This is our seed.
SEED_THETA = {
    "N0": 815121.5743,
    "C_seed": 53.95384919,
    "gamma": 1.998171949,
    "a0": -10.00563272,
    "aT": -0.5288233686,
    "aTT": -0.7443846599,
    "ae": 0.06675601819,
    "aR": -2.15965454,
    "betaH0": 1.237526606,
    "psi": 0.9915830056,
    "nb_kappa": 6.651471449,
    "bT_H": 0.2691541929,
    "be_H": -0.9072867122,
    "bR_H": 0.7940067809,
}
EXPECTED_SEED_LOGL = -780.2592797235

OUTPUT_ROOT = PROJECT_ROOT / "results" / "betaH0_prior_sensitivity_v2"
NESTED_DIR = OUTPUT_ROOT / "nested_sampling"
NESTED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load data
# =============================================================================

print("\n" + "=" * 100)
print("betaH0 SENSITIVITY RERUN v2 -- rslice sampling + seeded live point")
print("=" * 100)

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

print(f"\nNumber of fitted weeks: {len(merged_df)}")
print(f"Target: {FIT_TARGET}")


# =============================================================================
# Likelihood -- explicitly set lik.bounds (avoids the priors.py mismatch bug)
# =============================================================================

lik = PlagueLikelihood(merged_df=merged_df, betaH_mode=BETA_H_MODE, fit_target=FIT_TARGET)
param_names = list(lik.param_names)

missing = [name for name in param_names if name not in BOUNDS]
if missing:
    raise KeyError(f"Missing bounds for: {missing}")

final_bounds = [tuple(BOUNDS[name]) for name in param_names]
lik.bounds = list(final_bounds)  # <-- critical, matches run_final_all_models.py's fix

ndim = len(param_names)

print(f"\nFree parameters ({ndim}):")
print(param_names)
print("\nBounds used (prior transform AND likelihood -- must match):")
for name, b in zip(param_names, final_bounds):
    marker = "  <-- widened from (0.01, 1.5)" if name == "betaH0" else ""
    print(f"  {name:<12s} {b}{marker}")


def prior_transform(u):
    u = np.asarray(u, dtype=float)
    lower = np.array([b[0] for b in final_bounds])
    upper = np.array([b[1] for b in final_bounds])
    return lower + u * (upper - lower)


def to_unit_cube(theta_dict):
    """Inverse of prior_transform, for constructing the seeded live point."""
    u = np.empty(ndim, dtype=float)
    for i, name in enumerate(param_names):
        lo, hi = BOUNDS[name]
        u[i] = (theta_dict[name] - lo) / (hi - lo)
    return u


# =============================================================================
# Verify the seed point BEFORE spending any sampling time
# =============================================================================

print("\n" + "=" * 100)
print("VERIFYING SEED POINT")
print("=" * 100)

seed_theta_vec = np.array([SEED_THETA[name] for name in param_names], dtype=float)
seed_logl = lik.log_likelihood(seed_theta_vec)

print(f"\nExpected seed logL: {EXPECTED_SEED_LOGL:.10f}")
print(f"Recomputed seed logL: {seed_logl:.10f}")
print(f"Difference: {seed_logl - EXPECTED_SEED_LOGL:.2e}")

if not np.isfinite(seed_logl) or abs(seed_logl - EXPECTED_SEED_LOGL) > 1e-6:
    raise RuntimeError(
        "Seed point does not reproduce the expected likelihood under the "
        "current bounds. STOP -- do not proceed to sampling until this is "
        "resolved (bounds, param ordering, or likelihood code has changed)."
    )

print("[OK] Seed point verified. Proceeding to build the seeded live-point set.")


# =============================================================================
# Build the seeded live-point ensemble
# =============================================================================
# dynesty accepts live_points=(live_u, live_v, live_logl):
#   live_u    : (nlive, ndim) points in the unit cube
#   live_v    : (nlive, ndim) points in parameter space (prior_transform(live_u))
#   live_logl : (nlive,) log-likelihood at each live_v
#
# We draw nlive-1 points uniformly at random and overwrite one slot with the
# verified seed point, so the high-likelihood ridge is represented in the
# initial live-point set from iteration 0.
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING SEEDED LIVE-POINT SET")
print("=" * 100)

rng = np.random.default_rng(RANDOM_SEED)

live_u = rng.random((NLIVE, ndim))
live_v = np.array([prior_transform(u) for u in live_u])

print(f"\nEvaluating log-likelihood for {NLIVE} initial live points "
      "(this is the slow part of setup)...")

t0 = time.time()
live_logl = np.array([lik.log_likelihood(v) for v in live_v])
print(f"  done in {time.time() - t0:.1f}s")

n_bad = np.sum(~np.isfinite(live_logl))
print(f"  {n_bad} / {NLIVE} initial points had -inf log-likelihood "
      f"(dynesty will resample these internally if needed)")

# Overwrite slot 0 with the verified seed point.
live_u[0] = to_unit_cube(SEED_THETA)
live_v[0] = seed_theta_vec
live_logl[0] = seed_logl

print(f"\n[OK] Seed point placed at live-point index 0 "
      f"(logL = {live_logl[0]:.6f})")
print(f"     Best of the {NLIVE - 1} random live points: "
      f"logL = {np.nanmax(live_logl[1:]):.6f}")


# =============================================================================
# Run dynesty with rslice sampling
# =============================================================================

print("\n" + "=" * 100)
print("STARTING DYNESTY (sample='rslice', seeded live points)")
print("=" * 100)
print(f"\nnlive = {NLIVE}, dlogz = {DLOGZ}, sample = 'rslice', bound = 'multi'")

rstate = np.random.default_rng(RANDOM_SEED + 777)

sampler = NestedSampler(
    loglikelihood=lik.log_likelihood,
    prior_transform=prior_transform,
    ndim=ndim,
    nlive=NLIVE,
    bound="multi",
    sample="rslice",
    rstate=rstate,
    live_points=(live_u, live_v, live_logl),
)

t0 = time.time()
sampler.run_nested(dlogz=DLOGZ, print_progress=True)
elapsed_minutes = (time.time() - t0) / 60.0

results = sampler.results

logl = np.asarray(results.logl, dtype=float)
samples = np.asarray(results.samples, dtype=float)
best_index = int(np.nanargmax(logl))
best_loglik = float(logl[best_index])
best_theta = samples[best_index].copy()
logZ = float(results.logz[-1])
logZ_error = float(results.logzerr[-1])

print("\n" + "=" * 100)
print("RUN COMPLETE")
print("=" * 100)
print(f"\nFinished in {elapsed_minutes:.1f} min")
print(f"Maximum sampled log-likelihood = {best_loglik:.6f}")
print(f"log(Z) = {logZ:.6f} +/- {logZ_error:.6f}")

print(f"\nFor comparison:")
print(f"  Known-achievable optimum (seed):        logL = {EXPECTED_SEED_LOGL:.6f}")
print(f"  This run's max sampled logL:             logL = {best_loglik:.6f}")
print(f"  Gap:                                            {best_loglik - EXPECTED_SEED_LOGL:.6f}")
if best_loglik >= EXPECTED_SEED_LOGL - 0.5:
    print("  -> This run FOUND (or improved on) the known high-likelihood region. Good.")
else:
    print("  -> This run STILL did not reach the known region. Consider nlive=2000, "
          "or investigate further before trusting this posterior.")

print("\nHighest-likelihood parameter vector")
print("-" * 70)
for name, value in zip(param_names, best_theta):
    print(f"  {name:<12s} = {value:.10g}")


# =============================================================================
# Consistency check (same pattern as run_final_all_models.py)
# =============================================================================

reevaluated_loglik = float(lik.log_likelihood(best_theta))
difference = reevaluated_loglik - best_loglik

print("\nBest-theta likelihood consistency check")
print("-" * 70)
print(f"Stored dynesty logL  = {best_loglik:.10f}")
print(f"Reevaluated logL     = {reevaluated_loglik:.10f}")
print(f"Difference           = {difference:.12g}")

if not np.isfinite(reevaluated_loglik) or abs(difference) > 1e-8:
    raise RuntimeError("Likelihood consistency check FAILED. Do not trust this result.")

print("[OK] Exact likelihood consistency confirmed.")


# =============================================================================
# Quick betaH0 posterior check
# =============================================================================

weights = np.exp(results.logwt - results.logz[-1])
weights = weights / weights.sum()


def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values)
    sorter = np.argsort(values)
    values, w = values[sorter], np.asarray(sample_weight)[sorter]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(quantiles, cw, values)


beta_idx = param_names.index("betaH0")
beta_samples = samples[:, beta_idx]
b_lo, b_med, b_hi = weighted_quantile(beta_samples, [0.025, 0.5, 0.975], weights)

print("\n" + "=" * 100)
print("betaH0 POSTERIOR")
print("=" * 100)
print(f"median = {b_med:.4f}   95% CrI = ({b_lo:.4f}, {b_hi:.4f})")
for thresh in (1.5, 2.0, 2.5, 2.9):
    p = weights[beta_samples > thresh].sum()
    print(f"P(betaH0 > {thresh}) = {p:.4f}")


# =============================================================================
# Save
# =============================================================================

stem = f"all_climate__{FIT_TARGET}__betaH0_upper3_v2_rslice_seeded"
pkl_path = NESTED_DIR / f"{stem}.pkl"
json_path = NESTED_DIR / f"{stem}.json"

with open(pkl_path, "wb") as f:
    pickle.dump({
        "results": results,
        "param_names": param_names,
        "betaH_mode": BETA_H_MODE,
        "fit_target": FIT_TARGET,
        "bounds": BOUNDS,
        "experiment": "betaH0_upper_bound_sensitivity_v2_rslice_seeded",
        "seed_theta": SEED_THETA,
        "seed_logl_expected": EXPECTED_SEED_LOGL,
    }, f)

with open(json_path, "w") as f:
    json.dump({
        "betaH_mode": BETA_H_MODE,
        "fit_target": FIT_TARGET,
        "param_names": param_names,
        "nlive": NLIVE,
        "dlogz": DLOGZ,
        "sample": "rslice",
        "seeded_live_point": True,
        "elapsed_minutes": elapsed_minutes,
        "best_loglik": best_loglik,
        "seed_logl_expected": EXPECTED_SEED_LOGL,
        "logZ": logZ,
        "logZ_error": logZ_error,
        "betaH0_median": float(b_med),
        "betaH0_ci": [float(b_lo), float(b_hi)],
    }, f, indent=2)

print(f"\n[OK] Saved: {pkl_path}")
print(f"[OK] Saved: {json_path}")
print("\nDone.")