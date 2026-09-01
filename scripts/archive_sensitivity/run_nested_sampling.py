# =============================================================================
# run_nested_sampling.py
# =============================================================================
"""
Run nested sampling (dynesty) for ONE beta_H formulation, saving both the
posterior samples and the Bayesian evidence log(Z) for later comparison.

Usage:
    python scripts/run_nested_sampling.py --betaH_mode humidity_only
    python scripts/run_nested_sampling.py --betaH_mode constant --nlive 500 --dlogz 0.1

Start with small --nlive / --dlogz for a fast approximate run before
committing to a slow, precise one; runtime scales roughly with
nlive x (iterations to convergence) x (cost of one simulate_seir_dde call).

*** Path config lives in src/paths.py, not here -- edit MERGED_CSV there
if your prepared_data output is named differently. ***
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
import priors  # noqa: E402
from likelihood import PlagueLikelihood  # noqa: E402


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--betaH_mode",
        required=True,
        choices=list(priors.MODEL_SPECS.keys()),
    )

    ap.add_argument(
        "--fit-target",
        default="plague_deaths_reported_fit",
    )

    ap.add_argument(
        "--nlive",
        type=int,
        default=1000,
        help="number of live points",
    )

    ap.add_argument(
        "--dlogz",
        type=float,
        default=0.1,
        help="stopping criterion",
    )

    ap.add_argument(
        "--sample",
        default="auto",
        help="dynesty sampling method",
    )

    ap.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="parallel worker processes",
    )

    args = ap.parse_args()

    try:
        import dynesty
    except ImportError as e:
        raise SystemExit(f"dynesty is not installed. Run: pip install dynesty\n(original error: {e})")

    if not paths.MERGED_CSV.exists():
        raise SystemExit(
            f"Could not find {paths.MERGED_CSV}. Edit MERGED_CSV in src/paths.py "
            "to point at your actual prepared_data/ merged target+climate file."
        )

    merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
    lik = PlagueLikelihood(merged_df, betaH_mode=args.betaH_mode, fit_target=args.fit_target)

    print(f"beta_H mode: {args.betaH_mode}")
    print(f"Free parameters ({lik.ndim}): {lik.param_names}")
    print(f"Fitting to: {args.fit_target}  ({len(lik.y_obs)} weeks)")

    pool = None
    if args.nprocs > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.nprocs)

    sampler_kwargs = dict(
        loglikelihood=lik.log_likelihood,
        prior_transform=lik.prior_transform,
        ndim=lik.ndim,
        nlive=args.nlive,
        sample=args.sample,
    )
    if pool is not None:
        sampler_kwargs["pool"] = pool
        sampler_kwargs["queue_size"] = args.nprocs

    sampler = dynesty.NestedSampler(**sampler_kwargs)

    t0 = time.time()
    sampler.run_nested(dlogz=args.dlogz, print_progress=True)
    elapsed = time.time() - t0

    if pool is not None:
        pool.close()
        pool.join()

    results = sampler.results
    print(f"\n[OK] Finished in {elapsed / 60:.1f} min")
    print(f"log(Z) = {results.logz[-1]:.3f} +/- {results.logzerr[-1]:.3f}")

    paths.ensure_output_dirs()
    stem = paths.run_stem(args.betaH_mode, args.fit_target)
    out_path = paths.NESTED_SAMPLING_DIR / f"{stem}.pkl"
    meta_path = paths.NESTED_SAMPLING_DIR / f"{stem}.json"

    if out_path.exists():
        print(f"[WARNING] {out_path} already exists and will be overwritten "
              f"(previous run's .pkl + .json for this exact betaH_mode+fit_target combo).")

    with open(out_path, "wb") as f:
        pickle.dump({
            "betaH_mode": args.betaH_mode,
            "param_names": lik.param_names,
            "bounds": lik.bounds,
            "fit_target": args.fit_target,
            "results": results,
            "logz": results.logz[-1],
            "logzerr": results.logzerr[-1],
        }, f)

    with open(meta_path, "w") as f:
        json.dump({
            "betaH_mode": args.betaH_mode,
            "fit_target": args.fit_target,
            "nlive": args.nlive,
            "dlogz": args.dlogz,
            "sample": args.sample,
            "nprocs": args.nprocs,
            "ndim": lik.ndim,
            "param_names": lik.param_names,
            "n_weeks_fit": len(lik.y_obs),
            "logz": float(results.logz[-1]),
            "logzerr": float(results.logzerr[-1]),
            "elapsed_minutes": elapsed / 60,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, f, indent=2)

    print(f"[OK] Saved: {out_path}")
    print(f"[OK] Saved run metadata: {meta_path}")


if __name__ == "__main__":
    main()