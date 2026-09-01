# =============================================================================
# run_delay_sensitivity_all_climate.py
# =============================================================================
"""
Sensitivity analysis for the infection-to-death delay discretization in the
best-supported all-climate model.

Purpose
-------
The final_v2 Bayesian fits used a fixed Gamma-shaped infection-to-death delay
kernel with:

    shape = 2.0
    scale = 0.6 weeks
    max_weeks = 12

The original implementation ("gamma") discretizes that continuous Gamma
distribution by evaluating its PDF at weekly midpoints j + 0.5 and
normalizing the resulting values.

This sensitivity analysis compares that original implementation against an
alternative CDF-bin discretization ("gamma_cdf"), which assigns each weekly
lag its integrated Gamma probability:

    P(j <= T < j+1) = F(j+1) - F(j).

Importantly, the underlying Gamma shape and scale remain identical. This
analysis therefore tests sensitivity to the NUMERICAL DISCRETIZATION of the
delay distribution rather than to a different biological delay assumption.

The original final_v2 model outputs are NOT overwritten.

Scenarios
---------
1. gamma_midpoint
       delay_kind = "gamma"
       Original final_v2 implementation.

2. gamma_cdf
       delay_kind = "gamma_cdf"
       Weekly-bin integrated Gamma probabilities.

Outputs
-------
results/delay_sensitivity_all_climate/
    nested_sampling/
        all_climate__gamma_midpoint.pkl
        all_climate__gamma_midpoint.json
        all_climate__gamma_cdf.pkl
        all_climate__gamma_cdf.json

    delay_sensitivity_model_comparison.csv

Usage
-----
    python scripts/run_delay_sensitivity_all_climate.py
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
import priors  # noqa: E402
from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
BETA_H_MODE = "all_climate"

NLIVE = 1000
DLOGZ = 0.1

# Separate deterministic seeds for:
#   1. constructing the initial live-point ensemble
#   2. dynesty's own internal random state
#
# Each delay scenario receives its own offset so the runs remain
# independently reproducible.
BASE_SEED_LIVE_POINTS = 31000
BASE_SEED_DYNESTY = 41000


# =============================================================================
# Delay scenarios
# =============================================================================

DELAY_SCENARIOS = {
    "gamma_midpoint": {
        "delay_kind": "gamma",
        "delay_shape": 2.0,
        "delay_scale": 0.6,
        "delay_max_weeks": 12,
    },

    "gamma_cdf": {
        "delay_kind": "gamma_cdf",
        "delay_shape": 2.0,
        "delay_scale": 0.6,
        "delay_max_weeks": 12,
    },
}


# =============================================================================
# Output paths
# =============================================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "delay_sensitivity_all_climate"
)

NESTED_DIR = (
    OUTPUT_ROOT
    / "nested_sampling"
)

NESTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPARISON_CSV = (
    OUTPUT_ROOT
    / "delay_sensitivity_model_comparison.csv"
)


# =============================================================================
# Seed point
#
# Use the same verified high-likelihood all-climate seed used in the final_v2
# model comparison. It is guaranteed to be legal under the same priors.
#
# For gamma_midpoint, this is the exact same observation kernel as final_v2.
#
# For gamma_cdf, this parameter vector may no longer be especially
# high-likelihood, but it is still a legal point in parameter space and gives
# the sampler a biologically/plausibly relevant starting region.
# =============================================================================

SEED_THETA = {
    "N0": 835974.1264,
    "C_seed": 53.05633583,
    "gamma": 2.956666956,

    "a0": -9.932871573,
    "aT": -0.5772857829,
    "aTT": -0.8164787758,
    "ae": 0.2052556445,
    "aR": -2.113581693,

    "betaH0": 1.890874457,

    "psi": 0.9167862041,
    "nb_kappa": 6.62786897,

    "bT_H": 0.2597646708,
    "be_H": -0.862945601,
    "bR_H": 0.7397069532,
}


# =============================================================================
# Helpers
# =============================================================================

def prior_transform_factory(bounds):
    """
    Return a dynesty prior-transform function for the supplied uniform bounds.
    """

    lower = np.asarray(
        [b[0] for b in bounds],
        dtype=float,
    )

    upper = np.asarray(
        [b[1] for b in bounds],
        dtype=float,
    )

    def prior_transform(u):
        u = np.asarray(
            u,
            dtype=float,
        )

        return (
            lower
            + u * (upper - lower)
        )

    return prior_transform


def theta_dict_to_unit_cube(
    theta_dict: dict,
    param_names: list[str],
    bounds,
) -> np.ndarray:
    """
    Convert a named parameter vector into its corresponding point in the unit
    prior cube.
    """

    u = np.empty(
        len(param_names),
        dtype=float,
    )

    for i, (
        name,
        (lo, hi),
    ) in enumerate(
        zip(
            param_names,
            bounds,
        )
    ):

        value = float(
            theta_dict[name]
        )

        if not (
            lo
            <= value
            <= hi
        ):
            raise ValueError(
                f"Seed parameter {name!r}={value} lies outside "
                f"its prior bounds ({lo}, {hi})."
            )

        u[i] = (
            value - lo
        ) / (
            hi - lo
        )

    return u


def get_result_field(
    results,
    name: str,
):
    """
    Small compatibility helper for dynesty Results objects.
    """

    if hasattr(
        results,
        name,
    ):
        return getattr(
            results,
            name,
        )

    return results[
        name
    ]


# =============================================================================
# Main
# =============================================================================

def main():

    # =========================================================================
    # Load prepared data
    # =========================================================================

    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Prepared data not found:\n"
            f"{paths.MERGED_CSV}\n\n"
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[
            paths.DATE_COL
        ],
    )

    df = (
        df
        .sort_values(
            paths.DATE_COL
        )
        .reset_index(
            drop=True
        )
    )

    if FIT_TARGET not in df.columns:
        raise ValueError(
            f"{FIT_TARGET!r} not found in prepared data."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ALL-CLIMATE INFECTION-TO-DEATH DELAY DISCRETIZATION SENSITIVITY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nFitted weeks: "
        f"{len(df)}"
    )

    print(
        f"Target: "
        f"{FIT_TARGET}"
    )

    print(
        f"Model: "
        f"{BETA_H_MODE}"
    )

    print(
        "\nDelay scenarios:"
    )

    for name, config in DELAY_SCENARIOS.items():

        print(
            f"  {name:<18s} "
            f"kind={config['delay_kind']}, "
            f"shape={config['delay_shape']}, "
            f"scale={config['delay_scale']}, "
            f"max_weeks={config['delay_max_weeks']}"
        )


    # =========================================================================
    # Shared prior information
    # =========================================================================

    final_bounds = priors.get_bounds(
        BETA_H_MODE
    )

    expected_param_names = priors.get_param_names(
        BETA_H_MODE
    )

    print(
        "\nShared parameter bounds from src/priors.py:"
    )

    for name, bound in zip(
        expected_param_names,
        final_bounds,
    ):

        print(
            f"  {name:<12s} "
            f"{bound}"
        )


    # =========================================================================
    # Run delay scenarios
    # =========================================================================

    comparison_rows = []


    for scenario_index, (
        scenario_name,
        delay_config,
    ) in enumerate(
        DELAY_SCENARIOS.items()
    ):

        print(
            "\n\n"
            + "=" * 100
        )

        print(
            f"SCENARIO: {scenario_name}"
        )

        print(
            "=" * 100
        )


        # =====================================================================
        # Build likelihood with this delay kernel
        # =====================================================================

        lik = PlagueLikelihood(
            merged_df=df,
            betaH_mode=BETA_H_MODE,

            fit_target=
                FIT_TARGET,

            delay_kind=
                delay_config[
                    "delay_kind"
                ],

            delay_shape=
                delay_config[
                    "delay_shape"
                ],

            delay_scale=
                delay_config[
                    "delay_scale"
                ],

            delay_max_weeks=
                delay_config[
                    "delay_max_weeks"
                ],
        )


        param_names = list(
            lik.param_names
        )

        ndim = len(
            param_names
        )


        # ---------------------------------------------------------------------
        # Guard against accidental prior/order mismatch
        # ---------------------------------------------------------------------

        if param_names != expected_param_names:

            raise RuntimeError(
                "Parameter order from PlagueLikelihood does not match "
                "priors.get_param_names(all_climate).\n\n"
                f"Likelihood:\n{param_names}\n\n"
                f"Priors:\n{expected_param_names}"
            )


        # Explicitly use the same bounds for both the likelihood and prior
        # transform.
        lik.bounds = list(
            final_bounds
        )


        print(
            f"\nFree parameters ({ndim}): "
            f"{param_names}"
        )


        # =====================================================================
        # Prior transform
        # =====================================================================

        prior_transform = prior_transform_factory(
            final_bounds
        )


        # =====================================================================
        # Evaluate shared seed under this delay assumption
        # =====================================================================

        seed_theta_vec = np.asarray(
            [
                SEED_THETA[
                    name
                ]
                for name
                in param_names
            ],
            dtype=float,
        )


        seed_logl = float(
            lik.log_likelihood(
                seed_theta_vec
            )
        )


        print(
            f"\nSeed-point logL under "
            f"{scenario_name}: "
            f"{seed_logl:.6f}"
        )


        if not np.isfinite(
            seed_logl
        ):

            raise RuntimeError(
                f"Seed point has non-finite likelihood for "
                f"{scenario_name!r}."
            )


        seed_u = theta_dict_to_unit_cube(
            SEED_THETA,
            param_names,
            final_bounds,
        )


        # =====================================================================
        # Reproducible initial live-point ensemble
        # =====================================================================

        live_seed = (
            BASE_SEED_LIVE_POINTS
            + scenario_index
        )

        live_rng = np.random.default_rng(
            live_seed
        )


        live_u = live_rng.random(
            (
                NLIVE,
                ndim,
            )
        )


        live_v = np.asarray(
            [
                prior_transform(
                    u
                )
                for u
                in live_u
            ],
            dtype=float,
        )


        print(
            f"\nEvaluating "
            f"{NLIVE} initial live points..."
        )


        t0 = time.time()


        live_logl = np.asarray(
            [
                lik.log_likelihood(
                    theta
                )
                for theta
                in live_v
            ],
            dtype=float,
        )


        initial_eval_seconds = (
            time.time()
            - t0
        )


        print(
            f"  done in "
            f"{initial_eval_seconds:.1f}s"
        )


        print(
            f"  non-finite initial points: "
            f"{np.sum(~np.isfinite(live_logl))} "
            f"/ {NLIVE}"
        )


        # ---------------------------------------------------------------------
        # Replace first live point with the verified all-climate seed
        # ---------------------------------------------------------------------

        live_u[
            0
        ] = seed_u

        live_v[
            0
        ] = seed_theta_vec

        live_logl[
            0
        ] = seed_logl


        random_live_best = float(
            np.nanmax(
                live_logl[
                    1:
                ]
            )
        )


        print(
            f"[OK] Seed placed at live-point 0 "
            f"(logL={seed_logl:.6f})"
        )

        print(
            f"     Best random initial live point: "
            f"logL={random_live_best:.6f}"
        )


        # =====================================================================
        # Run dynesty
        # =====================================================================

        dynesty_seed = (
            BASE_SEED_DYNESTY
            + scenario_index
        )


        rstate = np.random.default_rng(
            dynesty_seed
        )


        print(
            "\nStarting dynesty:"
        )

        print(
            f"  sample = rslice"
        )

        print(
            f"  nlive  = {NLIVE}"
        )

        print(
            f"  dlogz  = {DLOGZ}"
        )

        print(
            f"  live seed    = "
            f"{live_seed}"
        )

        print(
            f"  dynesty seed = "
            f"{dynesty_seed}"
        )


        sampler = NestedSampler(
            loglikelihood=
                lik.log_likelihood,

            prior_transform=
                prior_transform,

            ndim=
                ndim,

            nlive=
                NLIVE,

            bound=
                "multi",

            sample=
                "rslice",

            rstate=
                rstate,

            live_points=(
                live_u,
                live_v,
                live_logl,
            ),
        )


        t0 = time.time()


        sampler.run_nested(
            dlogz=
                DLOGZ,

            print_progress=
                True,
        )


        elapsed_minutes = (
            time.time()
            - t0
        ) / 60.0


        # =====================================================================
        # Extract results
        # =====================================================================

        results = sampler.results


        logl = np.asarray(
            get_result_field(
                results,
                "logl",
            ),
            dtype=float,
        )


        samples = np.asarray(
            get_result_field(
                results,
                "samples",
            ),
            dtype=float,
        )


        logz = np.asarray(
            get_result_field(
                results,
                "logz",
            ),
            dtype=float,
        )


        logzerr = np.asarray(
            get_result_field(
                results,
                "logzerr",
            ),
            dtype=float,
        )


        best_index = int(
            np.nanargmax(
                logl
            )
        )


        best_loglik = float(
            logl[
                best_index
            ]
        )


        best_theta = samples[
            best_index
        ].copy()


        logZ = float(
            logz[
                -1
            ]
        )


        logZ_error = float(
            logzerr[
                -1
            ]
        )


        print(
            f"\nFinished in "
            f"{elapsed_minutes:.1f} min"
        )

        print(
            f"Max sampled logL = "
            f"{best_loglik:.6f}"
        )

        print(
            f"Seed logL = "
            f"{seed_logl:.6f}"
        )

        print(
            f"Improvement over seed = "
            f"{best_loglik - seed_logl:+.6f}"
        )

        print(
            f"log(Z) = "
            f"{logZ:.6f} "
            f"+/- "
            f"{logZ_error:.6f}"
        )


        # =====================================================================
        # Likelihood consistency check
        # =====================================================================

        reevaluated = float(
            lik.log_likelihood(
                best_theta
            )
        )


        diff = (
            reevaluated
            - best_loglik
        )


        print(
            "\nLikelihood consistency check:"
        )

        print(
            f"  stored = "
            f"{best_loglik:.10f}"
        )

        print(
            f"  reeval = "
            f"{reevaluated:.10f}"
        )

        print(
            f"  diff   = "
            f"{diff:.2e}"
        )


        if (
            not np.isfinite(
                reevaluated
            )
            or
            abs(
                diff
            )
            > 1e-8
        ):

            raise RuntimeError(
                f"Likelihood consistency check FAILED "
                f"for {scenario_name!r}."
            )


        print(
            "[OK] Consistency confirmed."
        )


        # =====================================================================
        # Save result
        # =====================================================================

        stem = (
            f"{BETA_H_MODE}"
            f"__{scenario_name}"
        )


        pkl_path = (
            NESTED_DIR
            / f"{stem}.pkl"
        )


        json_path = (
            NESTED_DIR
            / f"{stem}.json"
        )


        save_payload = {
            "results":
                results,

            "betaH_mode":
                BETA_H_MODE,

            "fit_target":
                FIT_TARGET,

            "delay_scenario":
                scenario_name,

            "delay_kind":
                delay_config[
                    "delay_kind"
                ],

            "delay_shape":
                delay_config[
                    "delay_shape"
                ],

            "delay_scale":
                delay_config[
                    "delay_scale"
                ],

            "delay_max_weeks":
                delay_config[
                    "delay_max_weeks"
                ],

            "param_names":
                param_names,

            "bounds":
                final_bounds,

            "best_theta":
                best_theta,

            "best_loglik":
                best_loglik,

            "logZ":
                logZ,

            "logZ_error":
                logZ_error,

            "nlive":
                NLIVE,

            "dlogz":
                DLOGZ,

            "sample":
                "rslice",

            "seeded":
                True,

            "seed_loglik":
                seed_logl,

            "live_points_seed":
                live_seed,

            "dynesty_seed":
                dynesty_seed,

            "elapsed_minutes":
                elapsed_minutes,
        }


        with open(
            pkl_path,
            "wb",
        ) as f:

            pickle.dump(
                save_payload,
                f,
            )


        with open(
            json_path,
            "w",
        ) as f:

            json.dump(
                {
                    "betaH_mode":
                        BETA_H_MODE,

                    "fit_target":
                        FIT_TARGET,

                    "delay_scenario":
                        scenario_name,

                    "delay_kind":
                        delay_config[
                            "delay_kind"
                        ],

                    "delay_shape":
                        delay_config[
                            "delay_shape"
                        ],

                    "delay_scale":
                        delay_config[
                            "delay_scale"
                        ],

                    "delay_max_weeks":
                        delay_config[
                            "delay_max_weeks"
                        ],

                    "param_names":
                        param_names,

                    "bounds":
                        {
                            name:
                                list(
                                    bound
                                )
                            for (
                                name,
                                bound
                            )
                            in zip(
                                param_names,
                                final_bounds,
                            )
                        },

                    "best_theta":
                        {
                            name:
                                float(
                                    value
                                )
                            for (
                                name,
                                value
                            )
                            in zip(
                                param_names,
                                best_theta,
                            )
                        },

                    "best_loglik":
                        best_loglik,

                    "logZ":
                        logZ,

                    "logZ_error":
                        logZ_error,

                    "nlive":
                        NLIVE,

                    "dlogz":
                        DLOGZ,

                    "sample":
                        "rslice",

                    "seeded":
                        True,

                    "seed_loglik":
                        seed_logl,

                    "live_points_seed":
                        live_seed,

                    "dynesty_seed":
                        dynesty_seed,

                    "elapsed_minutes":
                        elapsed_minutes,
                },
                f,
                indent=2,
            )


        print(
            f"\n[OK] Saved:\n"
            f"  {pkl_path}"
        )

        print(
            f"[OK] Saved:\n"
            f"  {json_path}"
        )


        # =====================================================================
        # Comparison row
        # =====================================================================

        comparison_rows.append(
            {
                "delay_scenario":
                    scenario_name,

                "delay_kind":
                    delay_config[
                        "delay_kind"
                    ],

                "delay_shape":
                    delay_config[
                        "delay_shape"
                    ],

                "delay_scale":
                    delay_config[
                        "delay_scale"
                    ],

                "delay_max_weeks":
                    delay_config[
                        "delay_max_weeks"
                    ],

                "n_parameters":
                    ndim,

                "max_loglik":
                    best_loglik,

                "logZ":
                    logZ,

                "logZ_error":
                    logZ_error,

                "seed_loglik":
                    seed_logl,

                "elapsed_minutes":
                    elapsed_minutes,
            }
        )


    # =========================================================================
    # Comparison table
    # =========================================================================

    comparison = pd.DataFrame(
        comparison_rows
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


    best_logZ = float(
        comparison[
            "logZ"
        ].max()
    )


    comparison[
        "delta_logZ"
    ] = (
        best_logZ
        -
        comparison[
            "logZ"
        ]
    )


    comparison[
        "BF_best_over_scenario"
    ] = np.exp(
        np.minimum(
            comparison[
                "delta_logZ"
            ],
            700.0,
        )
    )


    # Since both scenarios have the same parameter count, AIC/AICc/BIC
    # differences are driven only by their fitted likelihoods.
    n = len(
        df
    )


    k = comparison[
        "n_parameters"
    ].astype(
        float
    )


    comparison[
        "AIC"
    ] = (
        2.0 * k
        -
        2.0
        * comparison[
            "max_loglik"
        ]
    )


    comparison[
        "AICc"
    ] = (
        comparison[
            "AIC"
        ]
        +
        (
            2.0
            * k
            * (
                k + 1.0
            )
        )
        /
        (
            n - k - 1.0
        )
    )


    comparison[
        "BIC"
    ] = (
        k
        * np.log(
            n
        )
        -
        2.0
        * comparison[
            "max_loglik"
        ]
    )


    comparison[
        "delta_AICc"
    ] = (
        comparison[
            "AICc"
        ]
        -
        comparison[
            "AICc"
        ].min()
    )


    comparison[
        "delta_BIC"
    ] = (
        comparison[
            "BIC"
        ]
        -
        comparison[
            "BIC"
        ].min()
    )


    comparison.to_csv(
        COMPARISON_CSV,
        index=False,
    )


    # =========================================================================
    # Print comparison
    # =========================================================================

    print(
        "\n\n"
        + "=" * 120
    )

    print(
        "DELAY DISCRETIZATION SENSITIVITY — ALL-CLIMATE MODEL"
    )

    print(
        "=" * 120
    )


    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )


    print(
        "\n"
        + comparison.to_string(
            index=False
        )
    )


    print(
        f"\n[OK] Saved comparison:\n"
        f"  {COMPARISON_CSV}"
    )


    print(
        "\nInterpretation:"
    )

    print(
        "This comparison changes only the numerical discretization of the "
        "same Gamma(shape=2.0, scale=0.6 weeks) infection-to-death delay. "
        "It does not yet test alternative biological mean delays."
    )


    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()