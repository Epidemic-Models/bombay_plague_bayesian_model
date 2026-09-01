# =============================================================================
# run_all_climate_gamma_sensitivity.py
# =============================================================================
"""
Targeted prior-sensitivity analysis for gamma in the all-climate model.

Why this run exists
--------------------
Fixing the betaH0 pinning (round 1 -> round 2, betaH0 upper bound
0.01-1.5 -> 0.01-3.0) had a side effect: gamma's posterior shifted up
and is now itself pinned against ITS upper bound in the round-2
all-climate result.

    round 1: gamma median=2.03, 95% CrI=(1.08, 2.58) -- interior
    round 2: gamma median=2.44, 95% CrI=(1.30, 2.98) -- touching wall
             edge-mass ratio (upper) = 2.5x

This is the same trade-off pattern seen before with aR/a0 and betaH0:
widening one bound can shift posterior mass into a different parameter
that trades off against it. This script repeats the all-climate analysis
with ONLY gamma's upper bound changed, following the same rslice +
seeded-live-point protocol that successfully resolved the betaH0 issue
(see run_all_climate_betaH0_sensitivity_v2.py).

IMPORTANT -- decide before interpreting the result
----------------------------------------------------
gamma is the removal/recovery rate; gamma=3.0/week implies a mean
infectious period of ~2.3 days. Before treating a widened bound as
straightforwardly "more correct," check whether there is a genuine
biological/historical lower limit on the infectious period for
pneumonic/septicemic plague that would argue AGAINST widening gamma
much further. If 3.0 is close to a real biological ceiling, a posterior
piling up near it is a substantive finding (rapid removal), not
necessarily a prior artifact -- unlike betaH0's bound, which was
acknowledged as a modeling choice rather than a hard constraint.

This script widens gamma's upper bound from 3.0 to 6.0 (allowing a mean
infectious period down to ~0.83 weeks / ~1.2 days) purely as a diagnostic
probe. Do not adopt gamma<=6.0 as the new "final" bound without checking
that against what's biologically defensible for this disease.

Everything else matches the round-2 final all_climate configuration
(betaH0 <= 3.0, all other bounds identical).
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

# Round-2 final bounds, with ONLY gamma's upper bound changed: 3.0 -> 6.0
BOUNDS = {
    "N0":       (800000.0, 850000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 6.0),    # <-- widened from 3.0 (diagnostic probe)
    "a0":       (-11.0, -5.5),
    "aT":       (-2.0, 1.0),
    "aTT":      (-2.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-4.0, 1.0),
    "betaH0":   (0.01, 3.0),
    "psi":      (0.05, 1.0),
    "nb_kappa": (0.1, 80.0),
    "bT_H":     (-2.0, 2.0),
    "be_H":     (-2.0, 2.0),
    "bR_H":     (-2.0, 2.0),
}

# Seed: exact best theta from the round-2 (v2) all_climate result, extracted
# directly from the uploaded .pkl (argmax logl), verified logL below.
SEED_THETA = {
    "N0": 847147.1241052961,
    "C_seed": 45.46412824853358,
    "gamma": 2.7594435708006366,
    "a0": -10.073045091632157,
    "aT": -0.6082436937953069,
    "aTT": -0.6844148082093953,
    "ae": 0.13827360046742365,
    "aR": -2.139813921992762,
    "betaH0": 1.6996953492852156,
    "psi": 0.993549838769308,
    "nb_kappa": 6.697431515981066,
    "bT_H": 0.23770891916478698,
    "be_H": -0.8540293425921641,
    "bR_H": 0.7735783778330729,
}
EXPECTED_SEED_LOGL = -779.370198054828

OUTPUT_ROOT = PROJECT_ROOT / "results" / "gamma_prior_sensitivity"
NESTED_DIR = OUTPUT_ROOT / "nested_sampling"
NESTED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load data
# =============================================================================

print("\n" + "=" * 100)
print("gamma SENSITIVITY RERUN -- all_climate, rslice sampling + seeded live point")
print("=" * 100)

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

print(f"\nNumber of fitted weeks: {len(merged_df)}")
print(f"Target: {FIT_TARGET}")


# =============================================================================
# Likelihood -- explicitly set lik.bounds
# =============================================================================

lik = PlagueLikelihood(merged_df=merged_df, betaH_mode=BETA_H_MODE, fit_target=FIT_TARGET)
param_names = list(lik.param_names)

missing = [name for name in param_names if name not in BOUNDS]
if missing:
    raise KeyError(f"Missing bounds for: {missing}")

final_bounds = [tuple(BOUNDS[name]) for name in param_names]
lik.bounds = list(final_bounds)

ndim = len(param_names)

print(f"\nFree parameters ({ndim}):")
print(param_names)
print("\nBounds used (prior transform AND likelihood -- must match):")
for name, b in zip(param_names, final_bounds):
    marker = "  <-- widened from (0.5, 3.0) -- DIAGNOSTIC PROBE, see docstring" if name == "gamma" else ""
    print(f"  {name:<12s} {b}{marker}")


def prior_transform(u):
    u = np.asarray(u, dtype=float)
    lower = np.array([b[0] for b in final_bounds])
    upper = np.array([b[1] for b in final_bounds])
    return lower + u * (upper - lower)


def to_unit_cube(theta_dict):
    u = np.empty(ndim, dtype=float)
    for i, name in enumerate(param_names):
        lo, hi = BOUNDS[name]
        u[i] = (theta_dict[name] - lo) / (hi - lo)
    return u


# =============================================================================
# Verify seed point before spending sampling time
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
        "current bounds. STOP before spending time on sampling."
    )

print("[OK] Seed point verified. Proceeding to build the seeded live-point set.")


# =============================================================================
# Build the seeded live-point ensemble
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING SEEDED LIVE-POINT SET")
print("=" * 100)

rng = np.random.default_rng(RANDOM_SEED)

live_u = rng.random((NLIVE, ndim))
live_v = np.array([prior_transform(u) for u in live_u])

print(f"\nEvaluating log-likelihood for {NLIVE} initial live points...")
t0 = time.time()
live_logl = np.array([lik.log_likelihood(v) for v in live_v])
print(f"  done in {time.time() - t0:.1f}s")

n_bad = np.sum(~np.isfinite(live_logl))
print(f"  {n_bad} / {NLIVE} initial points had -inf log-likelihood")

live_u[0] = to_unit_cube(SEED_THETA)
live_v[0] = seed_theta_vec
live_logl[0] = seed_logl

print(f"\n[OK] Seed point placed at live-point index 0 (logL = {live_logl[0]:.6f})")
print(f"     Best of the {NLIVE - 1} random live points: logL = {np.nanmax(live_logl[1:]):.6f}")


# =============================================================================
# Run dynesty
# =============================================================================

print("\n" + "=" * 100)
print("STARTING DYNESTY (sample='rslice', seeded live points)")
print("=" * 100)
print(f"\nnlive = {NLIVE}, dlogz = {DLOGZ}, sample = 'rslice', bound = 'multi'")

rstate = np.random.default_rng(RANDOM_SEED + 888)

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
    print("  -> This run STILL did not reach the known region. Increase nlive and rerun "
          "before trusting this posterior.")

print("\nHighest-likelihood parameter vector")
print("-" * 70)
for name, value in zip(param_names, best_theta):
    print(f"  {name:<12s} = {value:.10g}")


# =============================================================================
# Consistency check
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
# Quick gamma posterior check
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


gamma_idx = param_names.index("gamma")
gamma_samples = samples[:, gamma_idx]
g_lo, g_med, g_hi = weighted_quantile(gamma_samples, [0.025, 0.5, 0.975], weights)

print("\n" + "=" * 100)
print("gamma POSTERIOR")
print("=" * 100)
print(f"median = {g_med:.4f}   95% CrI = ({g_lo:.4f}, {g_hi:.4f})")
for thresh in (3.0, 4.0, 5.0, 5.8):
    p = weights[gamma_samples > thresh].sum()
    print(f"P(gamma > {thresh}) = {p:.4f}")

print(f"\nImplied mean infectious period at posterior median: "
      f"{7.0 / g_med:.2f} days")
print(f"Implied mean infectious period at 95% CrI bounds: "
      f"({7.0 / g_hi:.2f}, {7.0 / g_lo:.2f}) days")


# =============================================================================
# Save
# =============================================================================

stem = f"all_climate__{FIT_TARGET}__gamma_upper6_rslice_seeded"
pkl_path = NESTED_DIR / f"{stem}.pkl"
json_path = NESTED_DIR / f"{stem}.json"

with open(pkl_path, "wb") as f:
    pickle.dump({
        "results": results,
        "param_names": param_names,
        "betaH_mode": BETA_H_MODE,
        "fit_target": FIT_TARGET,
        "bounds": BOUNDS,
        "experiment": "gamma_upper_bound_sensitivity",
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
        "gamma_median": float(g_med),
        "gamma_ci": [float(g_lo), float(g_hi)],
    }, f, indent=2)

print(f"\n[OK] Saved: {pkl_path}")
print(f"[OK] Saved: {json_path}")
print("\nDone.")
print("\nInterpretation reminder: if gamma piles up again near the new wall (6.0),")
print("that points toward weak identifiability / a structural trade-off rather than")
print("simple truncation -- do NOT just keep widening. Check against biologically")
print("plausible infectious-period ranges before deciding how to report this.")