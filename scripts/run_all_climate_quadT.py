# =============================================================================
# run_all_climate_quadT.py
# =============================================================================
"""
Stand-alone sensitivity run: climate-driven amplification with a temperature
peak.

    beta_H(t) = betaH0 * exp( bT_H*zT + bTT_H*zT^2 + be_H*ze + bR_H*zR )

This is the climate-driven ("all_climate") formulation plus a quadratic
temperature term in amplification, so that amplification can peak at an
intermediate temperature in the same way as spillover. The prior for bTT_H is
(-2, 0), the same as aTT: a negative value gives a single peak, and
bTT_H = 0 is exactly the climate-driven model. k = 15.

Requires
--------
The formulation is defined in src/, like every other formulation:
  * src/priors.py            -- "bTT_H" in EXTRA_BOUNDS, "all_climate_quadT"
                                in MODEL_SPECS
  * src/mechanistic_model.py -- the "all_climate_quadT" branch in eta_H
The script stops with a clear message if either is missing.

Seeding
-------
Same procedure as the main formulations: one live point at a preliminary best
fit, the other NLIVE-1 drawn from the prior. The seed is the climate-driven
best fit with bTT_H = 0, which reproduces the climate-driven curve exactly,
so the run starts from log L = -779.37. The script checks this before
sampling.

Reading the result
------------------
Compare log Z with the climate-driven value (-813.63):
  * lower, or within about 1: a temperature peak in amplification is not
    supported (reported as a sensitivity check);
  * clearly higher (> about 2-3): the peak is supported, and this formulation
    would compete with the current best model.

Usage
-----
    python scripts/run_all_climate_quadT.py

Output
------
results/all_climate_quadT_check/nested_sampling/
    all_climate_quadT__plague_deaths_reported_fit__seeded.pkl
    all_climate_quadT__plague_deaths_reported_fit__seeded.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
import priors  # noqa: E402
from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings (match run_final_all_models_v2.py)
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
MODE = "all_climate_quadT"
NLIVE = 1000
DLOGZ = 0.1

SEED_LIVE = 52345
SEED_SAMPLER = 53345

SEED_PKL = (PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
            / f"all_climate__{FIT_TARGET}__final_v2.pkl")

OUT_DIR = PROJECT_ROOT / "results" / f"{MODE}_check" / "nested_sampling"
OUT_STEM = f"{MODE}__{FIT_TARGET}__seeded"


def weighted_quantiles(vals, w, qs):
    order = np.argsort(vals)
    return np.interp(qs, np.cumsum(w[order]), vals[order])


# =============================================================================
# Main
# =============================================================================

def main():

    if MODE not in priors.MODEL_SPECS or "bTT_H" not in priors.EXTRA_BOUNDS:
        raise SystemExit(
            f"src/priors.py does not define {MODE!r} and 'bTT_H'. Use the updated "
            "src/priors.py and src/mechanistic_model.py before running this script."
        )
    btt_bounds = priors.EXTRA_BOUNDS["bTT_H"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("SENSITIVITY RUN: climate-driven amplification with a temperature peak")
    print("=" * 90)

    df = (pd.read_csv(paths.MERGED_CSV, parse_dates=["week_ending"])
          .sort_values("week_ending").reset_index(drop=True))

    lik = PlagueLikelihood(merged_df=df, betaH_mode=MODE, fit_target=FIT_TARGET)
    param_names = list(lik.param_names)
    ndim = len(param_names)
    bounds = list(priors.get_bounds(MODE))
    lik.bounds = list(bounds)
    print(f"Parameters ({ndim}): {param_names}")

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    def prior_transform(u):
        return lower + np.asarray(u, dtype=float) * (upper - lower)

    # ------------------------------------------------------------- seed point
    with open(SEED_PKL, "rb") as f:
        src = pickle.load(f)
    seed_dict = dict(zip(src["param_names"], np.asarray(src["best_theta"], float)))
    seed_dict["bTT_H"] = 0.0
    seed_theta = np.array([seed_dict[n] for n in param_names], dtype=float)
    seed_logl = float(lik.log_likelihood(seed_theta))

    print(f"Seed logL = {seed_logl:.6f}  (climate-driven best fit = {src['best_loglik']:.6f})")
    if not np.isfinite(seed_logl) or abs(seed_logl - float(src["best_loglik"])) > 1e-6:
        raise RuntimeError(
            "The seed does not reproduce the climate-driven best fit. Check the "
            f"{MODE!r} branch in src/mechanistic_model.py; stopping before sampling."
        )
    seed_u = (seed_theta - lower) / (upper - lower)

    # ------------------------------------------------------------ live points
    rng = np.random.default_rng(SEED_LIVE)
    live_u = rng.random((NLIVE, ndim))
    live_v = np.array([prior_transform(u) for u in live_u])
    print(f"Evaluating {NLIVE} initial live points...")
    t0 = time.time()
    live_logl = np.array([lik.log_likelihood(v) for v in live_v])
    print(f"  done in {time.time() - t0:.1f}s "
          f"({np.sum(~np.isfinite(live_logl))} / {NLIVE} had -inf)")
    live_u[0], live_v[0], live_logl[0] = seed_u, seed_theta, seed_logl

    # -------------------------------------------------------- nested sampling
    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(SEED_SAMPLER),
        live_points=(live_u, live_v, live_logl),
    )
    t0 = time.time()
    sampler.run_nested(dlogz=DLOGZ, print_progress=True)
    elapsed_minutes = (time.time() - t0) / 60.0

    results = sampler.results
    best_idx = int(np.nanargmax(results.logl))
    best_theta = np.asarray(results.samples[best_idx], dtype=float)
    best_loglik = float(results.logl[best_idx])
    logZ = float(results.logz[-1])
    logZ_error = float(results.logzerr[-1])

    reeval = float(lik.log_likelihood(best_theta))
    if not np.isfinite(reeval) or abs(reeval - best_loglik) > 1e-8:
        raise RuntimeError("Likelihood consistency check FAILED.")

    # ---------------------------------------------------------------- summary
    w = np.exp(results.logwt - logZ)
    w /= w.sum()
    coef = {}
    for name in ("bT_H", "bTT_H", "be_H", "bR_H", "aT", "aTT", "ae", "aR"):
        v = results.samples[:, param_names.index(name)]
        lo, md, hi = weighted_quantiles(v, w, [0.025, 0.5, 0.975])
        coef[name] = {"median": float(md), "ci": [float(lo), float(hi)],
                      "resolved": bool(lo > 0 or hi < 0),
                      "p_positive": float(w[v > 0].sum())}

    bTT = results.samples[:, param_names.index("bTT_H")]
    edge_mass_upper = float(w[bTT >= btt_bounds[1] - 0.05 * (btt_bounds[1] - btt_bounds[0])].sum()) / 0.05

    print("\n" + "-" * 90)
    print(f"Finished in {elapsed_minutes:.1f} min")
    print(f"max logL : climate-driven {src['best_loglik']:.3f}  ->  quadT {best_loglik:.3f}")
    print(f"log Z    : climate-driven {src['logZ']:.3f}  ->  quadT {logZ:.3f} +/- {logZ_error:.3f}")
    print(f"Delta log Z (quadT - climate-driven) = {logZ - float(src['logZ']):+.3f}")
    print("\nAmplification coefficients:")
    for name in ("bT_H", "bTT_H", "be_H", "bR_H"):
        c = coef[name]
        flag = "resolved" if c["resolved"] else "includes 0"
        print(f"  {name:<6} median {c['median']:+.3f}, 95% CrI ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f})  [{flag}]")
    print(f"  bTT_H mass near 0 (straight-line end) vs flat prior: {edge_mass_upper:.2f}x")
    print("Spillover coefficients:")
    for name in ("aT", "aTT", "ae", "aR"):
        c = coef[name]
        flag = "resolved" if c["resolved"] else "includes 0"
        print(f"  {name:<6} median {c['median']:+.3f}, 95% CrI ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f})  [{flag}]")
    print("-" * 90)

    # ------------------------------------------------------------------- save
    record = {
        "results": results,
        "betaH_mode": MODE,
        "fit_target": FIT_TARGET,
        "param_names": param_names,
        "bounds": bounds,
        "bounds_source": "src/priors.py",
        "best_theta": best_theta,
        "best_loglik": best_loglik,
        "logZ": logZ,
        "logZ_error": logZ_error,
        "nlive": NLIVE,
        "dlogz": DLOGZ,
        "sample": "rslice",
        "seeded": True,
        "seed_source": str(SEED_PKL.relative_to(PROJECT_ROOT)) + " + bTT_H = 0",
        "seed_loglik": seed_logl,
        "reference_logZ_all_climate": float(src["logZ"]),
        "coefficients": coef,
        "elapsed_minutes": elapsed_minutes,
    }
    with open(OUT_DIR / f"{OUT_STEM}.pkl", "wb") as f:
        pickle.dump(record, f)

    summary = {k: v for k, v in record.items() if k != "results"}
    summary["best_theta"] = dict(zip(param_names, map(float, best_theta)))
    summary["bounds"] = {n: [float(b[0]), float(b[1])] for n, b in zip(param_names, bounds)}
    with open(OUT_DIR / f"{OUT_STEM}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved {OUT_DIR / (OUT_STEM + '.pkl')}")


if __name__ == "__main__":
    main()