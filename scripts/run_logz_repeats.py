# =============================================================================
# run_logz_repeats.py
# =============================================================================
"""
Repeat runs to measure run-to-run variation in log Z for the two closest
formulations:

    all_climate        climate-driven (current best,   log Z -813.63)
    all_climate_quadT  climate-driven + temperature peak (log Z -809.71)

Why
---
The gap between them (+3.9) is small enough that ordinary run-to-run
variation of nested sampling could matter. dynesty's reported error (+/-0.2)
does not include all sources of that variation. Refitting each formulation
with different random seeds shows how much log Z really moves. If the gap
stays near +3.9 in every repeat, it is real; if it shrinks or varies a lot,
it is sampling noise.

The other four formulations are not repeated: their gaps to the
climate-driven formulation are 11.3 or more, far larger than any plausible
run-to-run variation.

Procedure (identical to the original runs except for the random seeds)
--------------------------------------------------------------------
* NLIVE = 1000, dlogz = 0.1, bound = "multi", sample = "rslice".
* One live point at a preliminary best fit, NLIVE-1 drawn from the prior:
    - all_climate:       its final best fit (final_v2 run);
    - all_climate_quadT: the climate-driven best fit with bTT_H = 0, exactly
                         as in run_all_climate_quadT.py.
* Each repeat uses its own pair of random seeds (REPEAT_SEEDS).

Nothing existing is read for anything except the seed point, and nothing
existing is overwritten. Each repeat is saved as soon as it finishes, and
repeats that already exist are skipped, so the script can be stopped and
restarted.

Requires
--------
src/priors.py and src/mechanistic_model.py with the "all_climate_quadT"
formulation (the same files used by run_all_climate_quadT.py).

Usage
-----
    python scripts/run_logz_repeats.py

Running time: about 3-4 hours per repeat; with the defaults (2 repeats of
each formulation) about 12-16 hours in total.

Output
------
results/logz_repeats/<mode>__rep<k>.pkl / .json
results/logz_repeats/logz_repeats_summary.csv
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
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
NLIVE = 1000
DLOGZ = 0.1

RESULTS = PROJECT_ROOT / "results"
MAIN_DIR = RESULTS / "final_model_comparison_v2" / "nested_sampling"
OUT_DIR = RESULTS / "logz_repeats"

CLIMATE_PKL = MAIN_DIR / f"all_climate__{FIT_TARGET}__final_v2.pkl"
QUADT_PKL = (RESULTS / "all_climate_quadT_check" / "nested_sampling"
             / f"all_climate_quadT__{FIT_TARGET}__seeded.pkl")

# The original runs, included in the summary as repeat 0.
ORIGINAL_RUNS = {
    "all_climate": CLIMATE_PKL,
    "all_climate_quadT": QUADT_PKL,
}

# Formulations to repeat, and the extra values needed to build each seed
# from the climate-driven best fit.
MODELS = {
    "all_climate": {},
    "all_climate_quadT": {"bTT_H": 0.0},
}

# One (live-point seed, sampler seed) pair per repeat. Different from every
# seed used before. Add a third pair to run a third repeat.
REPEAT_SEEDS = [
    (61001, 61002),
    (62001, 62002),
]

# Order of runs: alternate the formulations so that, if you stop early, you
# still have a repeat of each.
RUN_ORDER = [(mode, k) for k in range(1, len(REPEAT_SEEDS) + 1) for mode in MODELS]


# =============================================================================
# Helpers
# =============================================================================

def weighted_quantiles(vals, w, qs):
    order = np.argsort(vals)
    return np.interp(qs, np.cumsum(w[order]), vals[order])


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_one(mode: str, rep: int, df: pd.DataFrame, climate_src: dict) -> dict:
    out_stem = f"{mode}__rep{rep}"
    pkl_path = OUT_DIR / f"{out_stem}.pkl"
    if pkl_path.exists():
        print(f"\n[SKIP] {out_stem} already exists.")
        d = load_pkl(pkl_path)
        return {"mode": mode, "repeat": rep, "logZ": d["logZ"],
                "logZ_err": d["logZ_error"], "max_logL": d["best_loglik"],
                "bTT_H_median": d.get("bTT_H_median")}

    seed_live, seed_sampler = REPEAT_SEEDS[rep - 1]

    print("\n" + "=" * 90)
    print(f"REPEAT {rep}: {mode}   (seeds {seed_live}, {seed_sampler})")
    print("=" * 90)

    lik = PlagueLikelihood(merged_df=df, betaH_mode=mode, fit_target=FIT_TARGET)
    param_names = list(lik.param_names)
    ndim = len(param_names)
    bounds = list(priors.get_bounds(mode))
    lik.bounds = list(bounds)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    def prior_transform(u):
        return lower + np.asarray(u, dtype=float) * (upper - lower)

    # Seed point: climate-driven best fit (+ bTT_H = 0 for quadT)
    seed_dict = dict(zip(climate_src["param_names"],
                         np.asarray(climate_src["best_theta"], float)))
    seed_dict.update(MODELS[mode])
    seed_theta = np.array([seed_dict[n] for n in param_names], dtype=float)
    seed_logl = float(lik.log_likelihood(seed_theta))
    if not np.isfinite(seed_logl) or abs(seed_logl - float(climate_src["best_loglik"])) > 1e-6:
        raise RuntimeError(f"Seed for {mode} does not reproduce the climate-driven best fit "
                           f"({seed_logl:.6f} vs {climate_src['best_loglik']:.6f}).")
    print(f"Seed logL = {seed_logl:.6f}")

    rng = np.random.default_rng(seed_live)
    live_u = rng.random((NLIVE, ndim))
    live_v = np.array([prior_transform(u) for u in live_u])
    live_logl = np.array([lik.log_likelihood(v) for v in live_v])
    live_u[0] = (seed_theta - lower) / (upper - lower)
    live_v[0], live_logl[0] = seed_theta, seed_logl

    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(seed_sampler),
        live_points=(live_u, live_v, live_logl),
    )
    t0 = time.time()
    sampler.run_nested(dlogz=DLOGZ, print_progress=True)
    minutes = (time.time() - t0) / 60.0

    res = sampler.results
    best_idx = int(np.nanargmax(res.logl))
    best_theta = np.asarray(res.samples[best_idx], dtype=float)
    best_loglik = float(res.logl[best_idx])
    logZ = float(res.logz[-1])
    logZ_err = float(res.logzerr[-1])

    reeval = float(lik.log_likelihood(best_theta))
    if not np.isfinite(reeval) or abs(reeval - best_loglik) > 1e-8:
        raise RuntimeError("Likelihood consistency check FAILED.")

    bTT_median = None
    if "bTT_H" in param_names:
        w = np.exp(res.logwt - logZ)
        w /= w.sum()
        bTT_median = float(weighted_quantiles(res.samples[:, param_names.index("bTT_H")],
                                              w, [0.5])[0])

    record = {
        "results": res, "betaH_mode": mode, "fit_target": FIT_TARGET,
        "param_names": param_names, "bounds": bounds, "bounds_source": "src/priors.py",
        "best_theta": best_theta, "best_loglik": best_loglik,
        "logZ": logZ, "logZ_error": logZ_err,
        "nlive": NLIVE, "dlogz": DLOGZ, "sample": "rslice", "seeded": True,
        "seed_source": str(CLIMATE_PKL.relative_to(PROJECT_ROOT))
                       + (f" + {MODELS[mode]}" if MODELS[mode] else ""),
        "repeat": rep, "seed_live": seed_live, "seed_sampler": seed_sampler,
        "bTT_H_median": bTT_median, "elapsed_minutes": minutes,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(record, f)
    summary = {k: v for k, v in record.items() if k != "results"}
    summary["best_theta"] = dict(zip(param_names, map(float, best_theta)))
    summary["bounds"] = {n: [float(b[0]), float(b[1])] for n, b in zip(param_names, bounds)}
    with open(OUT_DIR / f"{out_stem}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{mode} repeat {rep}: log Z = {logZ:.3f} +/- {logZ_err:.3f}, "
          f"max logL = {best_loglik:.3f}  [{minutes:.0f} min]")
    print(f"[OK] Saved {pkl_path}")

    return {"mode": mode, "repeat": rep, "logZ": logZ, "logZ_err": logZ_err,
            "max_logL": best_loglik, "bTT_H_median": bTT_median}


# =============================================================================
# Main
# =============================================================================

def main():
    if "all_climate_quadT" not in priors.MODEL_SPECS:
        raise SystemExit("src/priors.py does not define 'all_climate_quadT'. Use the "
                         "updated src/priors.py and src/mechanistic_model.py.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = (pd.read_csv(paths.MERGED_CSV, parse_dates=["week_ending"])
          .sort_values("week_ending").reset_index(drop=True))
    climate_src = load_pkl(CLIMATE_PKL)

    rows = []
    for mode, path in ORIGINAL_RUNS.items():
        if path.exists():
            d = load_pkl(path)
            rows.append({"mode": mode, "repeat": 0, "logZ": float(d["logZ"]),
                         "logZ_err": float(d.get("logZ_error", np.nan)),
                         "max_logL": float(d["best_loglik"]),
                         "bTT_H_median": (d.get("coefficients", {}).get("bTT_H", {})
                                          .get("median"))})
        else:
            print(f"[WARNING] original run not found: {path}")

    for mode, rep in RUN_ORDER:
        rows.append(run_one(mode, rep, df, climate_src))
        # Write the summary after every run, so partial results are kept.
        pd.DataFrame(rows).to_csv(OUT_DIR / "logz_repeats_summary.csv", index=False)

    table = pd.DataFrame(rows).sort_values(["mode", "repeat"])
    print("\n" + "=" * 90)
    print("LOG Z ACROSS REPEATS (repeat 0 = original run)")
    print("=" * 90)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    stats = table.groupby("mode")["logZ"].agg(["count", "mean", "std", "min", "max"])
    print("\nPer formulation:")
    print(stats.to_string(float_format=lambda x: f"{x:.3f}"))

    if {"all_climate", "all_climate_quadT"} <= set(stats.index):
        a = table.loc[table["mode"] == "all_climate", "logZ"].to_numpy()
        q = table.loc[table["mode"] == "all_climate_quadT", "logZ"].to_numpy()
        gaps = q[:, None] - a[None, :]
        print(f"\nGap (quadT - climate-driven), all pairs of runs: "
              f"min {gaps.min():+.2f}, max {gaps.max():+.2f}, "
              f"mean of means {q.mean() - a.mean():+.2f}")
        print("Rule of thumb: if every pairwise gap stays well above 0 and the "
              "spread within each formulation is small compared with the gap, "
              "the difference is real.")

    print(f"\n[OK] Summary: {OUT_DIR / 'logz_repeats_summary.csv'}")


if __name__ == "__main__":
    main()