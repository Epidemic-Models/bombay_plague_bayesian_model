# =============================================================================
# run_prior_sensitivity.py
# =============================================================================
"""
Prior-sensitivity analysis for the Bayesian Bombay plague model.

This script repeats nested sampling for the four beta_H formulations using
wider priors for parameters whose posterior distributions approached the
boundaries of the original prior ranges.

IMPORTANT
---------
This sensitivity analysis does NOT overwrite the primary nested-sampling
results.

Primary analysis:
    results/nested_sampling/

Sensitivity analysis:
    results/prior_sensitivity/nested_sampling/

The mechanistic model, mortality target, observation model, climate
covariates, integration settings, and negative-binomial likelihood remain
unchanged.

Only selected prior bounds are widened.


WHY THIS ANALYSIS IS PERFORMED
------------------------------

Posterior-boundary diagnostics from the primary analysis indicated that some
parameters had appreciable posterior mass close to their original prior
limits.

The principal parameters of concern were:

    a0
    aT
    aTT
    aR
    C_seed
    psi

The purpose of this analysis is to test whether:

    1. posterior distributions move away from the original boundaries;
    2. climate-effect estimates remain qualitatively stable;
    3. Bayesian model ranking remains stable;
    4. the evidence advantage of climate-dependent beta_H formulations
       persists under wider priors.


WIDER PRIOR RANGES
------------------

Original -> sensitivity:

    C_seed:
        0 -- 100
        ->
        0 -- 150

    a0:
        -9 -- -5.5
        ->
        -10 -- -5.5

    aT:
        -1 -- 1
        ->
        -2 -- 1

    aTT:
        -1 -- 0
        ->
        -2 -- 0

    aR:
        -1 -- 1
        ->
        -2 -- 1

    psi:
        0.05 -- 0.8
        ->
        0.05 -- 1.0

All other priors are inherited unchanged from src/priors.py.


MODELS
------

    constant
    sigmoid
    humidity_only
    all_climate


DEFAULT SAMPLING SETTINGS
-------------------------

    nlive = 1000
    dlogz = 0.1

These match the high-resolution primary Bayesian analysis.


USAGE
-----

Run all four models:

    python scripts/run_prior_sensitivity.py

Run only one model:

    python scripts/run_prior_sensitivity.py --betaH_mode all_climate

Alternative sampling settings:

    python scripts/run_prior_sensitivity.py \
        --nlive 500 \
        --dlogz 0.2
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

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

import paths  # noqa: E402
import priors  # noqa: E402

from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Analysis settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"


BETA_H_MODES = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]


# =============================================================================
# Sensitivity-prior overrides
# =============================================================================
#
# Only parameters listed here are changed.
#
# Every parameter not listed here inherits its original prior directly
# from src/priors.py.
#
# =============================================================================

WIDE_PRIOR_OVERRIDES = {

    # Pre-epidemic incidence history
    "C_seed": (
        0.0,
        150.0,
    ),

    # Reservoir/spillover baseline
    "a0": (
        -10.0,
        -5.5,
    ),

    # Linear temperature effect on beta_R
    "aT": (
        -2.0,
        1.0,
    ),

    # Quadratic temperature effect on beta_R
    "aTT": (
        -2.0,
        0.0,
    ),

    # Rainfall effect on beta_R
    "aR": (
        -2.0,
        1.0,
    ),

    # Infection-to-observed-death scaling
    #
    # If psi is interpreted as a fraction/probability,
    # 1.0 remains the natural upper physical limit.
    "psi": (
        0.05,
        1.0,
    ),
}


# =============================================================================
# Output directories
# =============================================================================

SENSITIVITY_ROOT = (
    paths.RESULTS_DIR
    / "prior_sensitivity"
)

SENSITIVITY_NESTED_DIR = (
    SENSITIVITY_ROOT
    / "nested_sampling"
)

SENSITIVITY_NESTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SENSITIVITY_COMPARISON_CSV = (
    SENSITIVITY_ROOT
    / "prior_sensitivity_model_comparison.csv"
)


# =============================================================================
# Helpers
# =============================================================================

def get_original_bounds(
    betaH_mode: str,
) -> tuple[list[str], list[tuple[float, float]]]:
    """
    Return parameter names and original prior bounds from src/priors.py.
    """

    param_names = priors.get_param_names(
        betaH_mode
    )

    original_bounds = priors.get_bounds(
        betaH_mode
    )

    if len(param_names) != len(original_bounds):
        raise ValueError(
            f"Parameter/bounds mismatch for {betaH_mode}: "
            f"{len(param_names)} names but "
            f"{len(original_bounds)} bounds."
        )

    return param_names, original_bounds


def build_wide_bounds(
    betaH_mode: str,
) -> tuple[
    list[str],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """
    Construct sensitivity bounds.

    Returns
    -------
    param_names
        Parameter order used by the model.

    original_bounds
        Original priors from priors.py.

    wide_bounds
        Same priors except selected parameters are replaced by
        WIDE_PRIOR_OVERRIDES.
    """

    param_names, original_bounds = (
        get_original_bounds(
            betaH_mode
        )
    )

    wide_bounds = []

    for name, original in zip(
        param_names,
        original_bounds,
    ):

        if name in WIDE_PRIOR_OVERRIDES:

            new_bound = (
                WIDE_PRIOR_OVERRIDES[name]
            )

        else:

            new_bound = (
                tuple(original)
            )

        wide_bounds.append(
            tuple(
                float(x)
                for x in new_bound
            )
        )

    return (
        param_names,
        original_bounds,
        wide_bounds,
    )


def make_prior_transform(
    bounds: list[tuple[float, float]],
):
    """
    Construct the dynesty prior transform for uniform bounded priors.

    Dynesty samples u_j in [0,1].

    This function maps:

        u_j

    to:

        theta_j =
            lower_j
            + u_j * (upper_j - lower_j)
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

    def prior_transform(
        u: np.ndarray,
    ) -> np.ndarray:

        u = np.asarray(
            u,
            dtype=float,
        )

        return (
            lower
            + u * width
        )

    return prior_transform


def print_prior_comparison(
    param_names,
    original_bounds,
    wide_bounds,
):
    """
    Print original and sensitivity prior ranges.
    """

    print("\nPrior ranges")
    print("-" * 76)

    for name, old, new in zip(
        param_names,
        original_bounds,
        wide_bounds,
    ):

        changed = (
            tuple(old)
            != tuple(new)
        )

        marker = (
            "  <-- widened"
            if changed
            else ""
        )

        print(
            f"{name:12s} "
            f"original=({old[0]:.6g}, {old[1]:.6g}) "
            f"sensitivity=({new[0]:.6g}, {new[1]:.6g})"
            f"{marker}"
        )


# =============================================================================
# Run one model
# =============================================================================

def run_one_model(
    betaH_mode: str,
    merged_df: pd.DataFrame,
    nlive: int,
    dlogz: float,
    sample_method: str,
    nprocs: int,
) -> dict:
    """
    Run one beta_H formulation with widened sensitivity priors.
    """

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
    # Build original + sensitivity bounds
    # -------------------------------------------------------------------------

    (
        param_names,
        original_bounds,
        wide_bounds,
    ) = build_wide_bounds(
        betaH_mode
    )

    print("\n" + "=" * 80)
    print(
        f"PRIOR-SENSITIVITY RUN: {betaH_mode}"
    )
    print("=" * 80)

    print(
        f"Free parameters ({len(param_names)}): "
        f"{param_names}"
    )

    print(
        f"Fitting to: {FIT_TARGET} "
        f"({len(merged_df)} weeks)"
    )

    print_prior_comparison(
        param_names,
        original_bounds,
        wide_bounds,
    )

    # -------------------------------------------------------------------------
    # Construct likelihood
    # -------------------------------------------------------------------------

    lik = PlagueLikelihood(
        merged_df=merged_df,
        betaH_mode=betaH_mode,
        fit_target=FIT_TARGET,
    )

    # IMPORTANT:
    #
    # PlagueLikelihood.log_likelihood() performs an internal
    # bounds check using lik.bounds.
    #
    # Therefore it must use the sensitivity bounds here.
    #
    # Parameter order remains unchanged.
    # -------------------------------------------------------------------------

    lik.bounds = (
        wide_bounds
    )

    if lik.param_names != param_names:

        raise ValueError(
            "Parameter ordering mismatch between "
            "PlagueLikelihood and priors.py."
        )

    # -------------------------------------------------------------------------
    # Sensitivity prior transform
    # -------------------------------------------------------------------------

    prior_transform = (
        make_prior_transform(
            wide_bounds
        )
    )

    # -------------------------------------------------------------------------
    # Optional multiprocessing
    # -------------------------------------------------------------------------

    pool = None

    if nprocs > 1:

        import multiprocessing as mp

        pool = mp.Pool(
            nprocs
        )

    # -------------------------------------------------------------------------
    # Construct nested sampler
    # -------------------------------------------------------------------------

    sampler_kwargs = dict(

        loglikelihood=
            lik.log_likelihood,

        prior_transform=
            prior_transform,

        ndim=
            lik.ndim,

        nlive=
            nlive,

        sample=
            sample_method,
    )

    if pool is not None:

        sampler_kwargs[
            "pool"
        ] = pool

        sampler_kwargs[
            "queue_size"
        ] = nprocs

    sampler = dynesty.NestedSampler(
        **sampler_kwargs
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    t0 = time.time()

    sampler.run_nested(
        dlogz=dlogz,
        print_progress=True,
    )

    elapsed = (
        time.time()
        - t0
    )

    if pool is not None:

        pool.close()
        pool.join()

    results = (
        sampler.results
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    max_loglik = float(
        np.max(
            results.logl
        )
    )

    logz = float(
        results.logz[-1]
    )

    logzerr = float(
        results.logzerr[-1]
    )

    print(
        f"\n[OK] Finished {betaH_mode} "
        f"in {elapsed / 60:.1f} min"
    )

    print(
        f"Maximum sampled log-likelihood = "
        f"{max_loglik:.3f}"
    )

    print(
        f"log(Z) = "
        f"{logz:.3f} +/- {logzerr:.3f}"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    stem = (
        f"{paths.run_stem(betaH_mode, FIT_TARGET)}"
        f"__wide_priors"
    )

    pkl_path = (
        SENSITIVITY_NESTED_DIR
        / f"{stem}.pkl"
    )

    json_path = (
        SENSITIVITY_NESTED_DIR
        / f"{stem}.json"
    )

    saved_object = {

        "analysis":
            "prior_sensitivity",

        "prior_set":
            "wide",

        "betaH_mode":
            betaH_mode,

        "fit_target":
            FIT_TARGET,

        "param_names":
            param_names,

        "original_bounds":
            original_bounds,

        "bounds":
            wide_bounds,

        "prior_overrides":
            WIDE_PRIOR_OVERRIDES,

        "results":
            results,

        "logz":
            logz,

        "logzerr":
            logzerr,

        "max_loglik":
            max_loglik,
    }

    with open(
        pkl_path,
        "wb",
    ) as f:

        pickle.dump(
            saved_object,
            f,
        )

    metadata = {

        "analysis":
            "prior_sensitivity",

        "prior_set":
            "wide",

        "betaH_mode":
            betaH_mode,

        "fit_target":
            FIT_TARGET,

        "nlive":
            nlive,

        "dlogz":
            dlogz,

        "sample":
            sample_method,

        "nprocs":
            nprocs,

        "ndim":
            lik.ndim,

        "param_names":
            param_names,

        "original_bounds":
            [
                list(b)
                for b in original_bounds
            ],

        "wide_bounds":
            [
                list(b)
                for b in wide_bounds
            ],

        "prior_overrides":
            {
                key: list(value)
                for key, value
                in WIDE_PRIOR_OVERRIDES.items()
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
        f"[OK] Saved:\n"
        f"     {pkl_path}"
    )

    print(
        f"[OK] Saved metadata:\n"
        f"     {json_path}"
    )

    return {

        "betaH_mode":
            betaH_mode,

        "n_parameters":
            lik.ndim,

        "max_loglik":
            max_loglik,

        "logZ":
            logz,

        "logZ_error":
            logzerr,

        "elapsed_minutes":
            elapsed / 60,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Run prior-sensitivity nested sampling "
            "with widened priors."
        )
    )

    ap.add_argument(
        "--betaH_mode",
        default="all",
        choices=[
            "all",
            *BETA_H_MODES,
        ],
        help=(
            "Model to run. "
            "Default 'all' runs all four models."
        ),
    )

    ap.add_argument(
        "--nlive",
        type=int,
        default=1000,
        help=(
            "Number of dynesty live points."
        ),
    )

    ap.add_argument(
        "--dlogz",
        type=float,
        default=0.1,
        help=(
            "Nested-sampling evidence stopping criterion."
        ),
    )

    ap.add_argument(
        "--sample",
        default="auto",
        help=(
            "dynesty constrained-sampling method."
        ),
    )

    ap.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes."
        ),
    )

    args = ap.parse_args()

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
    # Decide which models to run
    # -------------------------------------------------------------------------

    if args.betaH_mode == "all":

        modes = (
            BETA_H_MODES
        )

    else:

        modes = [
            args.betaH_mode
        ]

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    summary_rows = []

    for betaH_mode in modes:

        result = run_one_model(

            betaH_mode=
                betaH_mode,

            merged_df=
                merged_df,

            nlive=
                args.nlive,

            dlogz=
                args.dlogz,

            sample_method=
                args.sample,

            nprocs=
                args.nprocs,
        )

        summary_rows.append(
            result
        )

    # -------------------------------------------------------------------------
    # Model comparison table
    # -------------------------------------------------------------------------

    comparison = pd.DataFrame(
        summary_rows
    )

    if len(comparison) > 0:

        best_logz = (
            comparison[
                "logZ"
            ].max()
        )

        comparison[
            "delta_logZ"
        ] = (
            best_logz
            - comparison[
                "logZ"
            ]
        )

        comparison[
            "BF_best_over_model"
        ] = np.exp(
            comparison[
                "delta_logZ"
            ]
        )

        comparison = (
            comparison
            .sort_values(
                "logZ",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    # -------------------------------------------------------------------------
    # Save comparison
    # -------------------------------------------------------------------------

    comparison.to_csv(
        SENSITIVITY_COMPARISON_CSV,
        index=False,
    )

    print("\n")
    print("=" * 100)
    print(
        "PRIOR-SENSITIVITY MODEL COMPARISON"
    )
    print("=" * 100)

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\n[OK] Saved comparison table:\n"
        f"{SENSITIVITY_COMPARISON_CSV}"
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()