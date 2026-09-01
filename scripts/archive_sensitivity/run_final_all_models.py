# =============================================================================
# run_final_all_models.py
# =============================================================================
"""
FINAL BAYESIAN MODEL COMPARISON

Runs all four beta_H formulations using one consistent set of priors:

    1. constant
    2. sigmoid
    3. humidity_only
    4. all_climate

IMPORTANT:
The same final bounds are used BOTH:
    - to transform the dynesty unit cube
    - inside PlagueLikelihood.log_likelihood()

This avoids the prior/bound mismatch discovered during the betaH0
sensitivity experiment.

Outputs are written to:
    results/final_model_comparison/
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
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from likelihood import PlagueLikelihood


DATA_FILE = (
    PROJECT_ROOT
    / "prepared_data"
    / "merged_target_climate.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison"
)

NESTED_DIR = OUTPUT_ROOT / "nested_sampling"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
NESTED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

MODELS = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]

NLIVE = 1000
DLOGZ = 0.1

# Fixed random seed for reproducibility.
RANDOM_SEED = 12345


# =============================================================================
# FINAL COMMON PRIORS
# =============================================================================
#
# These are based on the prior-sensitivity work already performed.
#
# Shared parameters receive identical bounds in every model.
#
# NOTE:
# betaH0 remains at [0.01, 1.5] for the final comparison for now.
# The attempted betaH0 <= 3 experiment was invalid because of the
# prior/likelihood mismatch and should NOT be used.
#
# psi <= 1 is retained because psi is a proportion/probability.
#
# =============================================================================

COMMON_BOUNDS = {
    "N0":       (800000.0, 850000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 3.0),

    "a0":       (-11.0, -5.5),
    "aT":       (-2.0, 1.0),
    "aTT":      (-2.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-4.0, 1.0),

    "betaH0":   (0.01, 1.5),

    "psi":      (0.05, 1.0),
    "nb_kappa": (0.1, 80.0),
}


EXTRA_BOUNDS = {

    "constant": {},

    "sigmoid": {
        "theta":  (-1.5, 1.5),
        "t_star": (4.0, 20.0),
        "kappa":  (0.2, 3.0),
    },

    "humidity_only": {
        "be_H": (-2.0, 2.0),
    },

    "all_climate": {
        "bT_H": (-2.0, 2.0),
        "be_H": (-2.0, 2.0),
        "bR_H": (-2.0, 2.0),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_final_bounds(param_names, model):
    """
    Construct the final bounds in exactly the same order as param_names.
    """

    model_bounds = dict(COMMON_BOUNDS)
    model_bounds.update(EXTRA_BOUNDS[model])

    bounds = []

    for name in param_names:

        if name not in model_bounds:
            raise KeyError(
                f"No final prior specified for parameter {name!r} "
                f"in model {model!r}."
            )

        bounds.append(tuple(model_bounds[name]))

    return bounds


def make_prior_transform(bounds):
    """
    Return a dynesty prior transform corresponding exactly to `bounds`.

    Every parameter has a uniform prior:

        theta_i ~ Uniform(lower_i, upper_i)
    """

    lower = np.asarray(
        [b[0] for b in bounds],
        dtype=float,
    )

    upper = np.asarray(
        [b[1] for b in bounds],
        dtype=float,
    )

    width = upper - lower

    def prior_transform(u):

        u = np.asarray(u, dtype=float)

        return lower + u * width

    return prior_transform


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 100)
print("FINAL BAYESIAN MODEL COMPARISON")
print("=" * 100)

print("\nLoading prepared data:")
print(DATA_FILE)

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["week_ending"],
)

if FIT_TARGET not in df.columns:
    raise ValueError(
        f"{FIT_TARGET!r} not found in prepared data."
    )

print(f"Number of fitted weeks: {len(df)}")
print(f"Target: {FIT_TARGET}")


# =============================================================================
# RUN MODELS
# =============================================================================

comparison_rows = []


for model_index, model in enumerate(MODELS):

    print("\n\n")
    print("=" * 100)
    print(f"FINAL MODEL RUN: {model}")
    print("=" * 100)

    # ---------------------------------------------------------------------
    # Construct likelihood
    # ---------------------------------------------------------------------

    lik = PlagueLikelihood(
        merged_df=df,
        betaH_mode=model,
        fit_target=FIT_TARGET,
    )

    param_names = list(lik.param_names)

    final_bounds = get_final_bounds(
        param_names,
        model,
    )

    # CRITICAL:
    #
    # log_likelihood() checks self.bounds.
    # Therefore we explicitly replace it with exactly the same bounds
    # used by the dynesty prior transform below.
    #
    lik.bounds = list(final_bounds)

    prior_transform = make_prior_transform(
        final_bounds
    )

    ndim = len(param_names)

    print(f"\nFree parameters ({ndim}):")
    print(param_names)

    print("\nFinal prior ranges")
    print("-" * 80)

    for name, bounds in zip(
        param_names,
        final_bounds,
    ):
        print(
            f"{name:<12s} "
            f"({bounds[0]:g}, {bounds[1]:g})"
        )

    # ---------------------------------------------------------------------
    # Internal consistency check
    # ---------------------------------------------------------------------

    if len(lik.bounds) != ndim:
        raise RuntimeError(
            "Likelihood bounds dimension mismatch."
        )

    for name, bound in zip(
        param_names,
        lik.bounds,
    ):
        lo, hi = bound

        if not np.isfinite(lo):
            raise RuntimeError(
                f"Non-finite lower bound for {name}"
            )

        if not np.isfinite(hi):
            raise RuntimeError(
                f"Non-finite upper bound for {name}"
            )

        if hi <= lo:
            raise RuntimeError(
                f"Invalid bound for {name}: {bound}"
            )

    # ---------------------------------------------------------------------
    # Test prior transform before expensive sampling
    # ---------------------------------------------------------------------

    print("\nChecking prior-transform / likelihood-bound consistency...")

    rng_check = np.random.default_rng(
        RANDOM_SEED + model_index
    )

    for _ in range(100):

        u_test = rng_check.random(ndim)

        theta_test = prior_transform(u_test)

        for value, (lo, hi), name in zip(
            theta_test,
            final_bounds,
            param_names,
        ):

            if not (lo <= value <= hi):
                raise RuntimeError(
                    f"Prior transform generated invalid value:\n"
                    f"{name} = {value}\n"
                    f"bounds = ({lo}, {hi})"
                )

    print("[OK] Prior transform and likelihood use identical bounds.")

    # ---------------------------------------------------------------------
    # Start dynesty
    # ---------------------------------------------------------------------

    print("\nStarting dynesty...")
    print(
        f"nlive = {NLIVE}, "
        f"dlogz = {DLOGZ}"
    )

    start_time = time.time()

    # Different deterministic random stream for each model.
    rstate = np.random.default_rng(
        RANDOM_SEED + 1000 + model_index
    )

    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rwalk",
        rstate=rstate,
    )

    sampler.run_nested(
        dlogz=DLOGZ,
        print_progress=True,
    )

    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60.0

    results = sampler.results

    # ---------------------------------------------------------------------
    # Extract results
    # ---------------------------------------------------------------------

    logl = np.asarray(
        results.logl,
        dtype=float,
    )

    samples = np.asarray(
        results.samples,
        dtype=float,
    )

    best_index = int(
        np.nanargmax(logl)
    )

    best_loglik = float(
        logl[best_index]
    )

    best_theta = samples[
        best_index
    ].copy()

    logZ = float(
        results.logz[-1]
    )

    logZ_error = float(
        results.logzerr[-1]
    )

    print("\n" + "=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)

    print(
        f"\nFinished in "
        f"{elapsed_minutes:.1f} min"
    )

    print(
        f"Maximum sampled log-likelihood = "
        f"{best_loglik:.6f}"
    )

    print(
        f"log(Z) = "
        f"{logZ:.6f} +/- {logZ_error:.6f}"
    )

    print(
        "\nHighest-likelihood "
        "parameter vector"
    )

    print("-" * 70)

    for name, value in zip(
        param_names,
        best_theta,
    ):
        print(
            f"{name:<12s} = {value:.10g}"
        )

    # ---------------------------------------------------------------------
    # VERY IMPORTANT:
    # reevaluate best theta
    # ---------------------------------------------------------------------

    reevaluated_loglik = float(
        lik.log_likelihood(best_theta)
    )

    difference = (
        reevaluated_loglik
        - best_loglik
    )

    print(
        "\nBest-theta likelihood "
        "consistency check"
    )

    print("-" * 70)

    print(
        f"Stored dynesty logL      = "
        f"{best_loglik:.10f}"
    )

    print(
        f"Reevaluated logL         = "
        f"{reevaluated_loglik:.10f}"
    )

    print(
        f"Difference               = "
        f"{difference:.12g}"
    )

    if (
        not np.isfinite(reevaluated_loglik)
        or abs(difference) > 1e-8
    ):
        raise RuntimeError(
            "\nLikelihood consistency check FAILED.\n"
            "Do not continue to model comparison."
        )

    print("[OK] Exact likelihood consistency confirmed.")

    # ---------------------------------------------------------------------
    # Save result
    # ---------------------------------------------------------------------

    result_file = (
        NESTED_DIR
        / f"{model}__{FIT_TARGET}__final.pkl"
    )

    metadata_file = (
        NESTED_DIR
        / f"{model}__{FIT_TARGET}__final.json"
    )

    save_object = {
        "results": results,
        "betaH_mode": model,
        "fit_target": FIT_TARGET,
        "param_names": param_names,
        "bounds": final_bounds,
        "best_theta": best_theta,
        "best_loglik": best_loglik,
        "logZ": logZ,
        "logZ_error": logZ_error,
        "nlive": NLIVE,
        "dlogz": DLOGZ,
        "elapsed_minutes": elapsed_minutes,
    }

    with open(
        result_file,
        "wb",
    ) as f:
        pickle.dump(
            save_object,
            f,
        )

    metadata = {
        "betaH_mode": model,
        "fit_target": FIT_TARGET,
        "param_names": param_names,
        "bounds": {
            name: [
                float(lo),
                float(hi),
            ]
            for name, (lo, hi)
            in zip(
                param_names,
                final_bounds,
            )
        },
        "best_theta": {
            name: float(value)
            for name, value
            in zip(
                param_names,
                best_theta,
            )
        },
        "best_loglik": best_loglik,
        "logZ": logZ,
        "logZ_error": logZ_error,
        "nlive": NLIVE,
        "dlogz": DLOGZ,
        "elapsed_minutes": elapsed_minutes,
    }

    with open(
        metadata_file,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n[OK] Saved nested-sampling result:")
    print(result_file)

    print("\n[OK] Saved metadata:")
    print(metadata_file)

    # ---------------------------------------------------------------------
    # Store comparison row
    # ---------------------------------------------------------------------

    comparison_rows.append(
        {
            "betaH_mode": model,
            "n_parameters": ndim,
            "max_loglik": best_loglik,
            "logZ": logZ,
            "logZ_error": logZ_error,
            "elapsed_minutes": elapsed_minutes,
        }
    )


# =============================================================================
# FINAL MODEL COMPARISON
# =============================================================================

comparison = pd.DataFrame(
    comparison_rows
)

comparison = comparison.sort_values(
    "logZ",
    ascending=False,
).reset_index(drop=True)

best_logZ = float(
    comparison["logZ"].max()
)

comparison["delta_logZ"] = (
    best_logZ
    - comparison["logZ"]
)

# Bayes factor:
#
# BF(best/model) = exp(logZ_best - logZ_model)
#
comparison["BF_best_over_model"] = np.exp(
    np.minimum(
        comparison["delta_logZ"],
        700.0,
    )
)


# =============================================================================
# ALSO COMPUTE AIC / AICc / BIC
# =============================================================================

n = len(df)

comparison["AIC"] = (
    2.0 * comparison["n_parameters"]
    - 2.0 * comparison["max_loglik"]
)

k = comparison["n_parameters"].astype(float)

comparison["AICc"] = (
    comparison["AIC"]
    +
    (
        2.0 * k * (k + 1.0)
        /
        (n - k - 1.0)
    )
)

comparison["BIC"] = (
    k * np.log(n)
    - 2.0 * comparison["max_loglik"]
)

comparison["delta_AIC"] = (
    comparison["AIC"]
    - comparison["AIC"].min()
)

comparison["delta_AICc"] = (
    comparison["AICc"]
    - comparison["AICc"].min()
)

comparison["delta_BIC"] = (
    comparison["BIC"]
    - comparison["BIC"].min()
)


# =============================================================================
# PRINT
# =============================================================================

print("\n\n")
print("=" * 120)
print("FINAL MODEL COMPARISON")
print("=" * 120)

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

print(
    comparison.to_string(
        index=False
    )
)


# =============================================================================
# SAVE TABLE
# =============================================================================

comparison_file = (
    OUTPUT_ROOT
    / "final_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False,
)

print("\n[OK] Saved final comparison table:")
print(comparison_file)


# =============================================================================
# FINAL MESSAGE
# =============================================================================

print("\n" + "=" * 120)
print("FINAL RUN FINISHED SUCCESSFULLY")
print("=" * 120)

print(
    "\nAll four models were fitted using the SAME final priors "
    "for every shared parameter."
)

print(
    "\nThe table above can now be used for the final "
    "Bayesian model comparison."
)