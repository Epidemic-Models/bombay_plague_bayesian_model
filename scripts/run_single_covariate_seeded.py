# =============================================================================
# run_single_covariate_seeded.py
# =============================================================================
"""
Seeded rerun of a single-covariate beta_H formulation (rainfall_only or
temperature_only), using the same procedure as the four main formulations in
run_final_all_models_v2.py.

Why
---
The main formulations were each fitted with one live point placed at that
formulation's best fit from an earlier round; the other NLIVE-1 live points
were drawn at random from the prior. The single-covariate formulations were
first fitted without that seeded point (run_rainfall_only_model.py,
run_temperature_only_model.py). This script reruns them with the same
safeguard, so all six formulations are fitted the same way.

The seed is the best-fitting parameter vector from the formulation's own
unseeded run (*__check.pkl). That is the same choice made for the main
formulations, whose seeds were their own round-1 best fits.

Widened-prior sensitivity run
-----------------------------
In the unseeded rainfall_only run, the posteriors of a0 and aR reach the lower
edges of their priors (95% CrI lower ends -10.93 vs bound -11, and -3.93 vs
bound -4). The main comparison keeps the same priors for all six formulations,
so the evidence comparison stays fair. As a sensitivity check, this script also
refits rainfall_only with those two lower bounds widened (WIDENED_BOUNDS). If
log Z changes by much less than the 11.9-unit gap to the climate-driven
formulation, the ranking does not depend on the bounds.

Usage
-----
    python scripts/run_single_covariate_seeded.py               # all three runs
    python scripts/run_single_covariate_seeded.py rainfall_only # one run only
    python scripts/run_single_covariate_seeded.py rainfall_only_wide

Runs (keys of RUNS):
    rainfall_only       seeded, standard priors   -> ..._seeded.pkl
    temperature_only    seeded, standard priors   -> ..._seeded.pkl
    rainfall_only_wide  seeded, widened a0 and aR -> ..._seeded_wide.pkl

Output
------
results/<mode>_check/nested_sampling/<mode>__plague_deaths_reported_fit__<suffix>.pkl / .json
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
NLIVE = 1000
DLOGZ = 0.1

# Widened lower bounds for the sensitivity run (standard: a0 -11, aR -4).
WIDENED_BOUNDS = {
    "a0": (-13.0, -5.5),
    "aR": (-6.0, 1.0),
}

# One entry per run. Seeds differ from the unseeded runs and from each other
# so no two runs share a random stream.
RUNS = {
    "rainfall_only": {
        "mode": "rainfall_only",
        "coef": "bR_H",
        "bound_overrides": {},
        "suffix": "seeded",
        "seed_live": 22345,
        "seed_sampler": 23345,
    },
    "temperature_only": {
        "mode": "temperature_only",
        "coef": "bT_H",
        "bound_overrides": {},
        "suffix": "seeded",
        "seed_live": 32345,
        "seed_sampler": 33345,
    },
    "rainfall_only_wide": {
        "mode": "rainfall_only",
        "coef": "bR_H",
        "bound_overrides": WIDENED_BOUNDS,
        "suffix": "seeded_wide",
        "seed_live": 42345,
        "seed_sampler": 43345,
    },
}


def weighted_quantiles(vals, w, qs):
    order = np.argsort(vals)
    cw = np.cumsum(w[order])
    return np.interp(qs, cw, vals[order])


def run_one(run_name: str) -> dict:
    """Seeded fit of one run in RUNS. Returns a short summary."""

    cfg = RUNS[run_name]
    mode = cfg["mode"]

    if mode not in priors.MODEL_SPECS:
        raise RuntimeError(f"priors.py does not define {mode!r}.")

    nested_dir = PROJECT_ROOT / "results" / f"{mode}_check" / "nested_sampling"
    unseeded_pkl = nested_dir / f"{mode}__{FIT_TARGET}__check.pkl"
    out_stem = f"{mode}__{FIT_TARGET}__{cfg['suffix']}"

    if not unseeded_pkl.exists():
        raise FileNotFoundError(
            f"Unseeded run not found:\n  {unseeded_pkl}\n"
            f"Run the unseeded script for {mode} first; its best fit is the seed."
        )

    print("\n" + "=" * 90)
    print(f"SEEDED RUN: {run_name}")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Data and likelihood
    # -------------------------------------------------------------------------
    df = (pd.read_csv(paths.MERGED_CSV, parse_dates=["week_ending"])
          .sort_values("week_ending").reset_index(drop=True))

    lik = PlagueLikelihood(merged_df=df, betaH_mode=mode, fit_target=FIT_TARGET)
    param_names = list(lik.param_names)
    ndim = len(param_names)
    bounds = list(priors.get_bounds(mode))
    for name, new in cfg["bound_overrides"].items():
        if name not in param_names:
            raise RuntimeError(f"Cannot widen {name!r}: not a parameter of {mode!r}.")
        i = param_names.index(name)
        print(f"Widened bound for {name}: {tuple(bounds[i])} -> {new}")
        bounds[i] = tuple(new)
    lik.bounds = list(bounds)

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    def prior_transform(u):
        return lower + np.asarray(u, dtype=float) * (upper - lower)

    # -------------------------------------------------------------------------
    # Seed point: best fit from the unseeded run
    # -------------------------------------------------------------------------
    with open(unseeded_pkl, "rb") as f:
        unseeded = pickle.load(f)

    if list(unseeded["param_names"]) != param_names:
        raise RuntimeError("Parameter order in the unseeded run does not match the "
                           "current model. Check priors.py.")

    seed_theta = np.asarray(unseeded["best_theta"], dtype=float)
    seed_logl = float(lik.log_likelihood(seed_theta))

    if not np.isfinite(seed_logl):
        raise RuntimeError("Seed point is not finite under the current bounds.")
    if abs(seed_logl - float(unseeded["best_loglik"])) > 1e-6:
        raise RuntimeError(
            f"Seed logL {seed_logl:.6f} differs from the stored value "
            f"{unseeded['best_loglik']:.6f}. The model or data have changed "
            "since the unseeded run."
        )

    seed_u = (seed_theta - lower) / (upper - lower)
    print(f"Seed logL = {seed_logl:.6f}  (best fit of the unseeded run; "
          f"unseeded log Z = {unseeded['logZ']:.3f})")

    # -------------------------------------------------------------------------
    # Live points: one seed + NLIVE-1 random prior draws
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(cfg["seed_live"])
    live_u = rng.random((NLIVE, ndim))
    live_v = np.array([prior_transform(u) for u in live_u])

    print(f"Evaluating {NLIVE} initial live points...")
    t0 = time.time()
    live_logl = np.array([lik.log_likelihood(v) for v in live_v])
    print(f"  done in {time.time() - t0:.1f}s "
          f"({np.sum(~np.isfinite(live_logl))} / {NLIVE} had -inf)")

    live_u[0], live_v[0], live_logl[0] = seed_u, seed_theta, seed_logl

    # -------------------------------------------------------------------------
    # Nested sampling
    # -------------------------------------------------------------------------
    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(cfg["seed_sampler"]),
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

    # Coefficient summary from the weighted posterior
    w = np.exp(results.logwt - logZ)
    w /= w.sum()
    vals = results.samples[:, param_names.index(cfg["coef"])]
    q025, q50, q975 = weighted_quantiles(vals, w, [0.025, 0.5, 0.975])
    p_pos = float(w[vals > 0].sum())

    # Where do the parameters that touched the walls now sit?
    # Reported for a0 and aR in every run, so standard and widened runs compare.
    edge_report = {}
    for name in ("a0", "aR"):
        if name not in param_names:
            continue
        i = param_names.index(name)
        lo, hi = bounds[i]
        v = results.samples[:, i]
        ql, qm, qh = weighted_quantiles(v, w, [0.025, 0.5, 0.975])
        mass_low = float(w[v <= lo + 0.05 * (hi - lo)].sum())
        edge_report[name] = {
            "bounds": [float(lo), float(hi)],
            "median": float(qm), "ci": [float(ql), float(qh)],
            "mass_in_lowest_5pct_over_flat": mass_low / 0.05,
        }

    print("\n" + "-" * 90)
    print(f"Finished in {elapsed_minutes:.1f} min")
    print(f"max logL : unseeded {unseeded['best_loglik']:.3f}  ->  seeded {best_loglik:.3f}")
    print(f"log Z    : unseeded {unseeded['logZ']:.3f}  ->  seeded {logZ:.3f} +/- {logZ_error:.3f}")
    print(f"{cfg['coef']}: median {q50:.3f}, 95% CrI ({q025:.3f}, {q975:.3f}), "
          f"P(>0) = {p_pos:.3f}")
    for name, e in edge_report.items():
        print(f"{name:<5}: prior {tuple(e['bounds'])}, median {e['median']:.3f}, "
              f"95% CrI ({e['ci'][0]:.3f}, {e['ci'][1]:.3f}), "
              f"lower-edge mass ratio {e['mass_in_lowest_5pct_over_flat']:.2f}")
    print("-" * 90)

    # -------------------------------------------------------------------------
    # Save (same keys as the main runs, so the analysis scripts read it as-is)
    # -------------------------------------------------------------------------
    record = {
        "results": results,
        "betaH_mode": mode,
        "fit_target": FIT_TARGET,
        "param_names": param_names,
        "bounds": bounds,
        "bounds_source": ("src/priors.py" if not cfg["bound_overrides"]
                          else f"src/priors.py with overrides {cfg['bound_overrides']}"),
        "bound_overrides": {k: list(v) for k, v in cfg["bound_overrides"].items()},
        "run_name": run_name,
        "edge_report": edge_report,
        "best_theta": best_theta,
        "best_loglik": best_loglik,
        "logZ": logZ,
        "logZ_error": logZ_error,
        "nlive": NLIVE,
        "dlogz": DLOGZ,
        "sample": "rslice",
        "seeded": True,
        "seed_source": str(unseeded_pkl.relative_to(PROJECT_ROOT)),
        "seed_loglik": seed_logl,
        "unseeded_logZ": float(unseeded["logZ"]),
        "elapsed_minutes": elapsed_minutes,
    }
    with open(nested_dir / f"{out_stem}.pkl", "wb") as f:
        pickle.dump(record, f)

    summary = {k: v for k, v in record.items() if k != "results"}
    summary["best_theta"] = dict(zip(param_names, map(float, best_theta)))
    summary["bounds"] = {n: [float(b[0]), float(b[1])] for n, b in zip(param_names, bounds)}
    summary[f"{cfg['coef']}_median"] = float(q50)
    summary[f"{cfg['coef']}_ci"] = [float(q025), float(q975)]
    summary[f"{cfg['coef']}_p_positive"] = p_pos
    with open(nested_dir / f"{out_stem}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved {nested_dir / (out_stem + '.pkl')}")

    return {
        "run": run_name,
        "unseeded_max_logL": float(unseeded["best_loglik"]),
        "seeded_max_logL": best_loglik,
        "unseeded_logZ": float(unseeded["logZ"]),
        "seeded_logZ": logZ,
        "seeded_logZ_err": logZ_error,
        "minutes": elapsed_minutes,
    }


def main():
    # No argument (e.g. the PyCharm Run button): all runs in RUNS.
    # With arguments: only those, e.g. `... run_single_covariate_seeded.py rainfall_only`.
    run_names = sys.argv[1:] or list(RUNS)
    unknown = [r for r in run_names if r not in RUNS]
    if unknown:
        raise SystemExit(f"Unknown run(s) {unknown}. Use: {', '.join(RUNS)}")

    summaries = [run_one(r) for r in run_names]

    print("\n" + "=" * 90)
    print("SEEDED vs UNSEEDED (unseeded = original run with standard priors)")
    print("=" * 90)
    for s in summaries:
        print(f"{s['run']:<19} max logL {s['unseeded_max_logL']:9.3f} -> {s['seeded_max_logL']:9.3f}   "
              f"log Z {s['unseeded_logZ']:9.3f} -> {s['seeded_logZ']:9.3f} "
              f"(+/- {s['seeded_logZ_err']:.3f})   [{s['minutes']:.0f} min]")

    by_run = {s["run"]: s for s in summaries}
    if "rainfall_only" in by_run and "rainfall_only_wide" in by_run:
        d = by_run["rainfall_only_wide"]["seeded_logZ"] - by_run["rainfall_only"]["seeded_logZ"]
        print(f"\nWidening a0 and aR changed rainfall_only log Z by {d:+.3f} "
              f"(gap to climate-driven was 11.85).")


if __name__ == "__main__":
    main()