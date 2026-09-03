# =============================================================================
# run_all_climate_prior_sensitivity2.py
# =============================================================================
"""
Second targeted prior-sensitivity analysis for the all-climate beta_H model.

WHY THIS RUN EXISTS
-------------------
The first wide-prior sensitivity analysis showed that most previously
constrained parameters moved comfortably away from their original bounds.

However, in the all-climate model:

    a0:
        prior = [-10, -5.5]
        posterior median ≈ -9.65
        95% CrI ≈ [-9.94, -9.17]

    aR:
        prior = [-2, 1]
        posterior median ≈ -1.93
        95% CrI ≈ [-2.00, -1.72]

The aR posterior remained strongly concentrated against its lower bound,
and a0 also retained substantial posterior mass close to its lower bound.

This script therefore performs a SECOND, TARGETED sensitivity analysis,
widening only those remaining problematic ranges.

SECOND-ROUND PRIORS
-------------------

Relative to the first wide-prior run:

    a0:
        [-10, -5.5]
        ->
        [-11, -5.5]

    aR:
        [-2, 1]
        ->
        [-4, 1]

All other sensitivity priors remain unchanged:

    C_seed = [0, 150]
    aT     = [-2, 1]
    aTT    = [-2, 0]
    psi    = [0.05, 1.0]

All remaining parameters retain their original priors.

MODEL
-----
Only:

    all_climate

is run.

OUTPUT
------
Results are saved separately under:

    results/prior_sensitivity_round2/nested_sampling/

so neither the primary analysis nor the first sensitivity analysis is
overwritten.

USAGE
-----
    python scripts/run_all_climate_prior_sensitivity2.py

Optional:
    python scripts/run_all_climate_prior_sensitivity2.py --nlive 1000 --dlogz 0.1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
import priors  # noqa: E402

from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

BETA_H_MODE = "all_climate"

FIT_TARGET = "plague_deaths_reported_fit"


# =============================================================================
# Second-round sensitivity priors
# =============================================================================

PRIOR_OVERRIDES_ROUND2 = {

    # Already widened in sensitivity round 1
    "C_seed": (
        0.0,
        150.0,
    ),

    "aT": (
        -2.0,
        1.0,
    ),

    "aTT": (
        -2.0,
        0.0,
    ),

    "psi": (
        0.05,
        1.0,
    ),

    # ---------------------------------------------------------
    # Further widened in sensitivity round 2
    # ---------------------------------------------------------

    "a0": (
        -11.0,
        -5.5,
    ),

    "aR": (
        -4.0,
        1.0,
    ),
}


# =============================================================================
# Output directories
# =============================================================================

ROUND2_ROOT = (
    paths.RESULTS_DIR
    / "prior_sensitivity_round2"
)

ROUND2_NESTED_DIR = (
    ROUND2_ROOT
    / "nested_sampling"
)

ROUND2_NESTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Helpers
# =============================================================================

def build_round2_bounds():
    """
    Construct the second-round prior bounds for the all-climate model.
    """

    param_names = priors.get_param_names(
        BETA_H_MODE
    )

    original_bounds = priors.get_bounds(
        BETA_H_MODE
    )

    if len(param_names) != len(original_bounds):
        raise ValueError(
            "Parameter/bounds length mismatch."
        )

    round2_bounds = []

    for name, old_bound in zip(
        param_names,
        original_bounds,
    ):

        if name in PRIOR_OVERRIDES_ROUND2:

            new_bound = (
                PRIOR_OVERRIDES_ROUND2[name]
            )

        else:

            new_bound = (
                tuple(old_bound)
            )

        round2_bounds.append(
            tuple(
                float(x)
                for x in new_bound
            )
        )

    return (
        param_names,
        original_bounds,
        round2_bounds,
    )


def make_prior_transform(
    bounds,
):
    """
    Uniform prior transform for dynesty.

    Maps u in [0,1]^D into the requested parameter ranges.
    """

    lower = np.array(
        [b[0] for b in bounds],
        dtype=float,
    )

    upper = np.array(
        [b[1] for b in bounds],
        dtype=float,
    )

    width = (
        upper
        - lower
    )

    def prior_transform(u):

        u = np.asarray(
            u,
            dtype=float,
        )

        return (
            lower
            + u * width
        )

    return prior_transform


def print_prior_table(
    param_names,
    original_bounds,
    round2_bounds,
):
    """
    Print original and round-2 prior ranges.
    """

    print("\nPrior ranges")
    print("-" * 82)

    for name, old, new in zip(
        param_names,
        original_bounds,
        round2_bounds,
    ):

        changed = (
            tuple(old)
            != tuple(new)
        )

        marker = (
            "  <-- sensitivity prior"
            if changed
            else ""
        )

        print(
            f"{name:12s} "
            f"original=({old[0]:.6g}, {old[1]:.6g}) "
            f"round2=({new[0]:.6g}, {new[1]:.6g})"
            f"{marker}"
        )


# =============================================================================
# Main
# =============================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--nlive",
        type=int,
        default=1000,
        help="number of dynesty live points",
    )

    ap.add_argument(
        "--dlogz",
        type=float,
        default=0.1,
        help="dynesty stopping criterion",
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
        help="number of worker processes",
    )

    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # dynesty
    # -------------------------------------------------------------------------

    try:

        import dynesty

    except ImportError as e:

        raise SystemExit(
            "dynesty is not installed.\n"
            "Run:\n"
            "    python -m pip install dynesty\n"
            f"\nOriginal error: {e}"
        )

    # -------------------------------------------------------------------------
    # Load prepared data
    # -------------------------------------------------------------------------

    if not paths.MERGED_CSV.exists():

        raise SystemExit(
            f"Could not find:\n"
            f"{paths.MERGED_CSV}\n\n"
            "Run scripts/prepare_data.py first."
        )

    merged_df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[
            paths.DATE_COL
        ],
    )

    merged_df = (
        merged_df
        .sort_values(
            paths.DATE_COL
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nLoaded prepared data:\n"
        f"{paths.MERGED_CSV}"
    )

    print(
        f"Number of fitted weeks: "
        f"{len(merged_df)}"
    )

    print(
        f"Target: "
        f"{FIT_TARGET}"
    )

    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------

    (
        param_names,
        original_bounds,
        round2_bounds,
    ) = build_round2_bounds()

    print("\n" + "=" * 84)
    print(
        "SECOND PRIOR-SENSITIVITY RUN: all_climate"
    )
    print("=" * 84)

    print(
        f"Free parameters ({len(param_names)}): "
        f"{param_names}"
    )

    print_prior_table(
        param_names,
        original_bounds,
        round2_bounds,
    )

    # -------------------------------------------------------------------------
    # Likelihood object
    # -------------------------------------------------------------------------

    lik = PlagueLikelihood(
        merged_df=merged_df,
        betaH_mode=BETA_H_MODE,
        fit_target=FIT_TARGET,
    )

    # IMPORTANT:
    # log_likelihood() checks lik.bounds internally.
    # Replace them with the round-2 sensitivity bounds.
    lik.bounds = round2_bounds

    if lik.param_names != param_names:

        raise ValueError(
            "Parameter order mismatch between "
            "PlagueLikelihood and priors.py."
        )

    prior_transform = make_prior_transform(
        round2_bounds
    )

    # -------------------------------------------------------------------------
    # Optional multiprocessing
    # -------------------------------------------------------------------------

    pool = None

    if args.nprocs > 1:

        import multiprocessing as mp

        pool = mp.Pool(
            args.nprocs
        )

    # -------------------------------------------------------------------------
    # Dynesty sampler
    # -------------------------------------------------------------------------

    sampler_kwargs = dict(

        loglikelihood=
            lik.log_likelihood,

        prior_transform=
            prior_transform,

        ndim=
            lik.ndim,

        nlive=
            args.nlive,

        sample=
            args.sample,
    )

    if pool is not None:

        sampler_kwargs[
            "pool"
        ] = pool

        sampler_kwargs[
            "queue_size"
        ] = args.nprocs

    sampler = dynesty.NestedSampler(
        **sampler_kwargs
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    t0 = time.time()

    sampler.run_nested(
        dlogz=args.dlogz,
        print_progress=True,
    )

    elapsed = (
        time.time()
        - t0
    )

    if pool is not None:

        pool.close()
        pool.join()

    results = sampler.results

    # -------------------------------------------------------------------------
    # Extract useful values
    # -------------------------------------------------------------------------

    max_idx = int(
        np.argmax(
            results.logl
        )
    )

    max_loglik = float(
        results.logl[
            max_idx
        ]
    )

    theta_max = (
        results.samples[
            max_idx
        ]
        .copy()
    )

    logz = float(
        results.logz[-1]
    )

    logzerr = float(
        results.logzerr[-1]
    )

    print(
        f"\n[OK] Finished in "
        f"{elapsed / 60:.1f} min"
    )

    print(
        f"Maximum sampled log-likelihood = "
        f"{max_loglik:.3f}"
    )

    print(
        f"log(Z) = "
        f"{logz:.3f} +/- "
        f"{logzerr:.3f}"
    )

    print(
        "\nHighest-likelihood parameter vector"
    )
    print("-" * 50)

    for name, value in zip(
        param_names,
        theta_max,
    ):

        print(
            f"{name:12s} = "
            f"{value:.6g}"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    stem = (
        "all_climate__"
        "plague_deaths_reported_fit__"
        "prior_sensitivity_round2"
    )

    pkl_path = (
        ROUND2_NESTED_DIR
        / f"{stem}.pkl"
    )

    json_path = (
        ROUND2_NESTED_DIR
        / f"{stem}.json"
    )

    saved = {

        "analysis":
            "prior_sensitivity_round2",

        "betaH_mode":
            BETA_H_MODE,

        "fit_target":
            FIT_TARGET,

        "param_names":
            param_names,

        "original_bounds":
            original_bounds,

        "bounds":
            round2_bounds,

        "prior_overrides":
            PRIOR_OVERRIDES_ROUND2,

        "results":
            results,

        "max_loglik":
            max_loglik,

        "theta_max":
            theta_max,

        "logz":
            logz,

        "logzerr":
            logzerr,
    }

    with open(
        pkl_path,
        "wb",
    ) as f:

        pickle.dump(
            saved,
            f,
        )

    metadata = {

        "analysis":
            "prior_sensitivity_round2",

        "betaH_mode":
            BETA_H_MODE,

        "fit_target":
            FIT_TARGET,

        "nlive":
            args.nlive,

        "dlogz":
            args.dlogz,

        "sample":
            args.sample,

        "nprocs":
            args.nprocs,

        "ndim":
            lik.ndim,

        "param_names":
            param_names,

        "round2_bounds":
            [
                list(b)
                for b in round2_bounds
            ],

        "prior_overrides":
            {
                key: list(value)
                for key, value
                in PRIOR_OVERRIDES_ROUND2.items()
            },

        "n_weeks_fit":
            len(lik.y_obs),

        "max_loglik":
            max_loglik,

        "logz":
            logz,

        "logzerr":
            logzerr,

        "elapsed_minutes":
            elapsed / 60,

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(
                timespec="seconds"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\n[OK] Saved nested-sampling result:\n"
        f"{pkl_path}"
    )

    print(
        f"\n[OK] Saved metadata:\n"
        f"{json_path}"
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()