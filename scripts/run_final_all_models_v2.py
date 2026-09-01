# =============================================================================
# run_final_all_models_v2.py
# =============================================================================
"""
Final Bayesian model comparison, ROUND 2 -- fixes two issues found in the
first final run (run_final_all_models.py):

  1. betaH0's upper bound was pinned at 1.5 for the all-climate model
     (95% CrI reached up to 1.487, P(betaH0>1.5)=0.447 once widened).
     That bound is a shared, common prior, so ALL FOUR models must use
     the same widened bound -- otherwise the model comparison is no
     longer apples-to-apples. betaH0 upper bound is now 3.0 for every
     model.

  2. The first betaH0-widening attempt (run_all_climate_betaH0_sensitivity.py)
     converged to the wrong mode with sample="rwalk" (best logL ~ -805 vs.
     the achievable -780). This run uses sample="rslice" plus a seeded
     live point at each model's known best theta from the round-1 final
     run, so every model starts with the previously-found high-likelihood
     region already represented in its live-point set.

*** UPDATED: bounds are now read directly from src/priors.py
(priors.get_bounds / priors.get_param_names) instead of a local
COMMON_BOUNDS/EXTRA_BOUNDS dict duplicated inside this script. priors.py
was itself updated to match this run's bounds exactly (see its docstring
and ORIGINAL_LHS_BOUNDS), so this removes the last place those bounds
were defined twice -- previously, this script's local dict and
src/priors.py's SHARED_BOUNDS/EXTRA_BOUNDS had to be kept in sync by
hand, which is exactly the kind of duplication that caused the earlier
priors.py staleness bugs (see project README, Section 5). Now there is
exactly one place these bounds are declared. ***

For constant / sigmoid / humidity_only, betaH0 was NOT pinned in round 1,
so the seed is just their round-1 best_theta (still legal under the wider
bound, since only the upper bound moved). For all_climate, the seed is the
IMPROVED point found by run_all_climate_betaH0_sensitivity_v2.py
(logL = -779.687078, better than round 1's -779.788032).

Outputs are written to:
    results/final_model_comparison_v2/
"""

from __future__ import annotations

import sys
import time
import json
import pickle
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
import priors  # noqa: E402 -- single source of truth for bounds, see docstring above
from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
MODELS = ["constant", "sigmoid", "humidity_only", "all_climate"]

NLIVE = 1000
DLOGZ = 0.1
RANDOM_SEED = 12345

OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_model_comparison_v2"
NESTED_DIR = OUTPUT_ROOT / "nested_sampling"
NESTED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Seed points -- round-1 best_theta for each model (all_climate uses the
# IMPROVED v2 point). These are legal under the widened bounds because only
# betaH0's upper bound moved (up), and none of these values exceed it.
# =============================================================================

SEED_THETAS = {
    "constant": {
        "N0": 847529.7048718277, "C_seed": 59.23434744009101, "gamma": 2.9933226553976553,
        "a0": -9.014074989553155, "aT": -0.8318642987831972, "aTT": -0.2799008929005635,
        "ae": -0.589771442569321, "aR": -0.10756424903524264, "betaH0": 1.3249861352428438,
        "psi": 0.9510498632053883, "nb_kappa": 3.821788563607029,
    },
    "sigmoid": {
        "N0": 834582.8344997002, "C_seed": 47.51822942150178, "gamma": 1.6426929714002383,
        "a0": -9.16503152285721, "aT": -1.1508129334098567, "aTT": -0.34164137549964835,
        "ae": -0.3941818902210267, "aR": -0.062065152972822446, "betaH0": 1.2103587403780371,
        "psi": 0.9793562337046177, "nb_kappa": 3.703496054045582,
        "theta": -0.44766311432173467, "t_star": 12.111230757510876, "kappa": 2.1990393075947687,
    },
    "humidity_only": {
        "N0": 839192.0484133193, "C_seed": 40.81976005323069, "gamma": 0.5016868697755253,
        "a0": -8.888707098150286, "aT": -0.4616242087571232, "aTT": -0.6506698799900783,
        "ae": -0.31633187071584534, "aR": -0.10384998243423071, "betaH0": 0.14978445538060242,
        "psi": 0.9944885942316771, "nb_kappa": 4.353040392399838, "be_H": -0.854060155638092,
    },
    "all_climate": {
        # Improved point from run_all_climate_betaH0_sensitivity_v2.py
        "N0": 835974.1264, "C_seed": 53.05633583, "gamma": 2.956666956,
        "a0": -9.932871573, "aT": -0.5772857829, "aTT": -0.8164787758,
        "ae": 0.2052556445, "aR": -2.113581693, "betaH0": 1.890874457,
        "psi": 0.9167862041, "nb_kappa": 6.62786897,
        "bT_H": 0.2597646708, "be_H": -0.862945601, "bR_H": 0.7397069532,
    },
}


# =============================================================================
# Load data
# =============================================================================

print("\n" + "=" * 100)
print("FINAL BAYESIAN MODEL COMPARISON -- ROUND 2 (betaH0 widened, rslice + seeded)")
print("Bounds sourced from src/priors.py")
print("=" * 100)

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

df = pd.read_csv(paths.MERGED_CSV, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

if FIT_TARGET not in df.columns:
    raise ValueError(f"{FIT_TARGET!r} not found in prepared data.")

print(f"\nNumber of fitted weeks: {len(df)}")
print(f"Target: {FIT_TARGET}")


# =============================================================================
# Run models
# =============================================================================

comparison_rows = []

for model_index, model in enumerate(MODELS):

    print("\n\n" + "=" * 100)
    print(f"MODEL: {model}")
    print("=" * 100)

    lik = PlagueLikelihood(merged_df=df, betaH_mode=model, fit_target=FIT_TARGET)
    param_names = list(lik.param_names)
    ndim = len(param_names)

    # Bounds come straight from priors.py, in the same order as param_names
    # (priors.get_param_names and priors.get_bounds are always built from
    # the same SHARED_PARAM_ORDER + MODEL_SPECS, so they're guaranteed
    # consistent with each other by construction).
    final_bounds = priors.get_bounds(model)

    # This assignment is technically redundant now that priors.py itself
    # declares the correct bounds by default (PlagueLikelihood.__init__
    # already sets self.bounds = priors.get_bounds(betaH_mode)) -- kept
    # explicit anyway as a guard: if priors.py and this script's own
    # priors.get_bounds(model) call ever disagree (e.g. a future edit),
    # this line still guarantees the prior_transform and the likelihood's
    # bound-check use IDENTICAL values, which is the property that matters.
    lik.bounds = list(final_bounds)

    print(f"\nFree parameters ({ndim}): {param_names}")
    print("\nBounds (from src/priors.py):")
    for name, b in zip(param_names, final_bounds):
        marker = "  <-- widened from (0.01, 1.5)" if name == "betaH0" else ""
        print(f"  {name:<12s} {b}{marker}")

    def prior_transform(u, _bounds=final_bounds):
        u = np.asarray(u, dtype=float)
        lower = np.array([b[0] for b in _bounds])
        upper = np.array([b[1] for b in _bounds])
        return lower + u * (upper - lower)

    def to_unit_cube(theta_dict, _names=param_names, _bounds=final_bounds):
        u = np.empty(len(_names), dtype=float)
        for i, (name, (lo, hi)) in enumerate(zip(_names, _bounds)):
            u[i] = (theta_dict[name] - lo) / (hi - lo)
        return u

    # -------------------------------------------------------------------
    # Verify seed point before spending sampling time
    # -------------------------------------------------------------------
    seed_dict = SEED_THETAS[model]
    seed_theta_vec = np.array([seed_dict[name] for name in param_names], dtype=float)
    seed_logl = lik.log_likelihood(seed_theta_vec)

    print(f"\nSeed point logL = {seed_logl:.6f}  (round-1 best for this model)")
    if not np.isfinite(seed_logl):
        raise RuntimeError(
            f"Seed point for {model!r} is not finite under the priors.py bounds. "
            "Check SEED_THETAS / priors.py before proceeding -- this likely means "
            "priors.py's bounds no longer match what this seed point was verified "
            "against."
        )

    # -------------------------------------------------------------------
    # Build seeded live-point ensemble
    # -------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED + model_index)

    live_u = rng.random((NLIVE, ndim))
    live_v = np.array([prior_transform(u) for u in live_u])

    print(f"Evaluating log-likelihood for {NLIVE} initial live points...")
    t0 = time.time()
    live_logl = np.array([lik.log_likelihood(v) for v in live_v])
    print(f"  done in {time.time() - t0:.1f}s "
          f"({np.sum(~np.isfinite(live_logl))} / {NLIVE} initial points had -inf)")

    live_u[0] = to_unit_cube(seed_dict)
    live_v[0] = seed_theta_vec
    live_logl[0] = seed_logl
    print(f"[OK] Seed placed at live-point index 0 (logL={live_logl[0]:.4f}); "
          f"best of the {NLIVE-1} random points: logL={np.nanmax(live_logl[1:]):.4f}")

    # -------------------------------------------------------------------
    # Run dynesty
    # -------------------------------------------------------------------
    print(f"\nStarting dynesty (sample='rslice', nlive={NLIVE}, dlogz={DLOGZ})...")

    rstate = np.random.default_rng(RANDOM_SEED + 1000 + model_index)

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

    print(f"\nFinished in {elapsed_minutes:.1f} min")
    print(f"Max sampled logL = {best_loglik:.6f}  (seed was {seed_logl:.6f}, "
          f"gap = {best_loglik - seed_logl:+.6f})")
    print(f"log(Z) = {logZ:.6f} +/- {logZ_error:.6f}")

    # -------------------------------------------------------------------
    # Consistency check
    # -------------------------------------------------------------------
    reevaluated = float(lik.log_likelihood(best_theta))
    diff = reevaluated - best_loglik
    print(f"\nConsistency check: stored={best_loglik:.10f}  reeval={reevaluated:.10f}  diff={diff:.2e}")
    if not np.isfinite(reevaluated) or abs(diff) > 1e-8:
        raise RuntimeError(f"Likelihood consistency check FAILED for {model!r}.")
    print("[OK] Consistency confirmed.")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    stem = f"{model}__{FIT_TARGET}__final_v2"
    pkl_path = NESTED_DIR / f"{stem}.pkl"
    json_path = NESTED_DIR / f"{stem}.json"

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "results": results, "betaH_mode": model, "fit_target": FIT_TARGET,
            "param_names": param_names, "bounds": final_bounds,
            "best_theta": best_theta, "best_loglik": best_loglik,
            "logZ": logZ, "logZ_error": logZ_error,
            "nlive": NLIVE, "dlogz": DLOGZ, "sample": "rslice",
            "seeded": True, "elapsed_minutes": elapsed_minutes,
            "bounds_source": "src/priors.py",
        }, f)

    with open(json_path, "w") as f:
        json.dump({
            "betaH_mode": model, "fit_target": FIT_TARGET, "param_names": param_names,
            "bounds": {n: list(b) for n, b in zip(param_names, final_bounds)},
            "bounds_source": "src/priors.py",
            "best_theta": {n: float(v) for n, v in zip(param_names, best_theta)},
            "best_loglik": best_loglik, "logZ": logZ, "logZ_error": logZ_error,
            "nlive": NLIVE, "dlogz": DLOGZ, "sample": "rslice", "seeded": True,
            "elapsed_minutes": elapsed_minutes,
        }, f, indent=2)

    print(f"[OK] Saved: {pkl_path}")
    print(f"[OK] Saved: {json_path}")

    comparison_rows.append({
        "betaH_mode": model, "n_parameters": ndim,
        "max_loglik": best_loglik, "logZ": logZ, "logZ_error": logZ_error,
        "elapsed_minutes": elapsed_minutes,
    })


# =============================================================================
# Final comparison table
# =============================================================================

comparison = pd.DataFrame(comparison_rows).sort_values("logZ", ascending=False).reset_index(drop=True)

best_logZ = float(comparison["logZ"].max())
comparison["delta_logZ"] = best_logZ - comparison["logZ"]
comparison["BF_best_over_model"] = np.exp(np.minimum(comparison["delta_logZ"], 700.0))

n = len(df)
k = comparison["n_parameters"].astype(float)
comparison["AIC"] = 2.0 * k - 2.0 * comparison["max_loglik"]
comparison["AICc"] = comparison["AIC"] + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
comparison["BIC"] = k * np.log(n) - 2.0 * comparison["max_loglik"]
comparison["delta_AIC"] = comparison["AIC"] - comparison["AIC"].min()
comparison["delta_AICc"] = comparison["AICc"] - comparison["AICc"].min()
comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()

print("\n\n" + "=" * 120)
print("FINAL MODEL COMPARISON -- ROUND 2 (consistent betaH0 <= 3.0 across all models, bounds from priors.py)")
print("=" * 120)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(comparison.to_string(index=False))

comparison_file = OUTPUT_ROOT / "final_model_comparison_v2.csv"
comparison.to_csv(comparison_file, index=False)
print(f"\n[OK] Saved: {comparison_file}")

print("\nDone.")