# =============================================================================
# run_temperature_only_model.py
# =============================================================================
"""
Fit the "temperature_only" beta_H formulation and summarize the posterior for
bT_H.

Purpose
-------
This model is used as a sensitivity / independence check for the temperature
effect on prevalence-associated amplification, motivated directly by the
climate-covariate correlation analysis (check_climate_covariate_correlations.py):
in the full all-climate posterior, bT_H and be_H (the temperature and vapour
pressure coefficients on beta_H) are very strongly correlated
(Pearson r = -0.9245, Spearman rho = -0.9178), consistent with the raw
temperature/vapour-pressure climate series themselves being strongly
correlated (Pearson r = 0.9016). This raises the possibility that
temperature's effect on amplification -- reported as "directionally similar
(positive) but not resolved" in the full climate-driven model -- is left
unresolved partly because it is competing with the near-collinear vapour
pressure term for the same variance in the likelihood, rather than because
temperature has no meaningful independent effect.

The temperature-only formulation excludes vapour pressure and rainfall from
beta_H:

    beta_H(t) = betaH0 * exp(bT_H * zT(t))

so the posterior for bT_H obtained here, with the collinear vapour-pressure
term entirely removed, can be compared with the corresponding (unresolved)
coefficient under the full all-climate formulation.

Two possible outcomes, both informative:
  (a) bT_H becomes resolved (95% CrI excludes zero) once vapour pressure is
      excluded -- consistent with the full-model non-resolution being a
      collinearity artifact rather than evidence of no independent effect.
  (b) bT_H remains unresolved even alone -- strengthens the case that
      temperature has a genuinely weak/absent independent effect on
      amplification, not merely a collinearity-masked one.

IMPORTANT
---------
This is a new model formulation with no previously verified high-likelihood
parameter vector available for manual live-point seeding, exactly as with
run_rainfall_only_model.py.

The Dynesty random state IS seeded for reproducibility. What is absent is a
manually inserted high-likelihood live point.

Posterior equal-weight resampling is also given its own independent random
seed so that posterior summaries are reproducible and are not coupled to the
nested-sampling random stream.

Requires
--------
The "temperature_only" formulation must already be supported by:

    src/mechanistic_model.py
    src/priors.py
    src/likelihood.py

(mirroring exactly how "rainfall_only" and "humidity_only" are already
supported -- add an analogous single-covariate branch for temperature if one
does not already exist.)

Usage
-----
    python scripts/run_temperature_only_model.py
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
from dynesty.utils import resample_equal


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
BETA_H_MODE = "temperature_only"

NLIVE = 1000
DLOGZ = 0.1

# Separate random streams, exactly mirroring run_rainfall_only_model.py:
#
# 1. Dynesty's stochastic nested-sampling exploration.
# 2. Equal-weight posterior resampling used for posterior summaries.
#
# Keeping these separate makes the two stochastic stages independently
# reproducible. Different seed values from the rainfall-only run, so the two
# checks are not accidentally coupled to the same random stream.
RANDOM_SEED_SAMPLER = 13345
RANDOM_SEED_POSTERIOR = 55321

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "temperature_only_check"
)

NESTED_DIR = (
    OUTPUT_ROOT
    / "nested_sampling"
)

NESTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Helpers
# =============================================================================

def equal_weight_posterior(
    results,
    seed: int,
) -> np.ndarray:
    """
    Convert Dynesty's weighted posterior samples to equal-weight samples
    reproducibly.
    """

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    rstate = np.random.default_rng(
        seed
    )

    posterior = resample_equal(
        results.samples,
        weights,
        rstate=rstate,
    )

    return np.asarray(
        posterior,
        dtype=float,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FITTING temperature_only MODEL "
        "(vapour pressure and rainfall excluded from beta_H)"
    )

    print(
        "=" * 100
    )

    # =========================================================================
    # 1. Verify temperature_only formulation exists
    # =========================================================================

    if BETA_H_MODE not in priors.MODEL_SPECS:

        raise RuntimeError(
            "priors.py does not define 'temperature_only'. "
            "Add this formulation to src/priors.py, src/mechanistic_model.py, "
            "and src/likelihood.py -- mirroring exactly how 'rainfall_only' "
            "is already implemented -- before running this script."
        )

    # =========================================================================
    # 2. Load data
    # =========================================================================

    if not paths.MERGED_CSV.exists():

        raise FileNotFoundError(
            f"Prepared data not found:\n"
            f"{paths.MERGED_CSV}"
        )

    df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=["week_ending"],
    )

    df = (
        df
        .sort_values("week_ending")
        .reset_index(drop=True)
    )

    if FIT_TARGET not in df.columns:

        raise ValueError(
            f"{FIT_TARGET!r} not found in prepared data."
        )

    print(
        f"\nFitted weeks: "
        f"{len(df)}"
    )

    print(
        f"Study period: "
        f"{df['week_ending'].min().date()} "
        f"to "
        f"{df['week_ending'].max().date()}"
    )

    # =========================================================================
    # 3. Build likelihood and prior bounds
    # =========================================================================

    lik = PlagueLikelihood(
        merged_df=df,
        betaH_mode=BETA_H_MODE,
        fit_target=FIT_TARGET,
    )

    param_names = list(
        lik.param_names
    )

    ndim = len(
        param_names
    )

    final_bounds = list(
        priors.get_bounds(
            BETA_H_MODE
        )
    )

    # Explicitly keep the likelihood's bound checking and prior transform
    # synchronized.
    lik.bounds = list(
        final_bounds
    )

    if len(final_bounds) != ndim:

        raise RuntimeError(
            "Number of prior bounds does not match "
            "number of model parameters."
        )

    print(
        f"\nFree parameters ({ndim}): "
        f"{param_names}"
    )

    print(
        "\nBounds "
        "(from src/priors.py):"
    )

    for name, bounds in zip(
        param_names,
        final_bounds,
    ):

        print(
            f"  {name:<12s} "
            f"{bounds}"
        )

    # =========================================================================
    # 4. Prior transform
    # =========================================================================

    lower = np.asarray(
        [
            b[0]
            for b
            in final_bounds
        ],
        dtype=float,
    )

    upper = np.asarray(
        [
            b[1]
            for b
            in final_bounds
        ],
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

    # =========================================================================
    # 5. Dynesty nested sampling
    # =========================================================================

    print(
        "\nStarting Dynesty "
        "(sample='rslice', random state seeded, "
        "no manually seeded high-likelihood live point)..."
    )

    sampler_rstate = np.random.default_rng(
        RANDOM_SEED_SAMPLER
    )

    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=sampler_rstate,
    )

    t0 = time.time()

    sampler.run_nested(
        dlogz=DLOGZ,
        print_progress=True,
    )

    elapsed_minutes = (
        time.time() - t0
    ) / 60.0

    results = sampler.results

    logl = np.asarray(
        results.logl,
        dtype=float,
    )

    samples = np.asarray(
        results.samples,
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

    best_theta = (
        samples[
            best_index
        ]
        .copy()
    )

    logZ = float(
        results.logz[-1]
    )

    logZ_error = float(
        results.logzerr[-1]
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
        f"log(Z) = "
        f"{logZ:.6f} "
        f"+/- "
        f"{logZ_error:.6f}"
    )

    # =========================================================================
    # 6. Likelihood consistency check
    # =========================================================================

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
        "\nConsistency check: "
        f"stored={best_loglik:.10f}  "
        f"reeval={reevaluated:.10f}  "
        f"diff={diff:.2e}"
    )

    if (
        not np.isfinite(
            reevaluated
        )
        or
        abs(diff) > 1e-8
    ):

        raise RuntimeError(
            "Likelihood consistency check FAILED."
        )

    print(
        "[OK] Consistency confirmed."
    )

    # =========================================================================
    # 7. Reproducible equal-weight posterior
    # =========================================================================

    posterior = equal_weight_posterior(
        results,
        seed=RANDOM_SEED_POSTERIOR,
    )

    if (
        posterior.ndim != 2
        or
        posterior.shape[1] != ndim
    ):

        raise RuntimeError(
            "Unexpected posterior sample shape: "
            f"{posterior.shape}"
        )

    print(
        f"\nEqual-weight posterior samples: "
        f"{len(posterior)}"
    )

    print(
        "Posterior resampling seed: "
        f"{RANDOM_SEED_POSTERIOR}"
    )

    # =========================================================================
    # 8. bT_H posterior
    # =========================================================================

    if "bT_H" not in param_names:

        raise RuntimeError(
            "'bT_H' not found in temperature_only "
            "parameter list."
        )

    bT_H_idx = param_names.index(
        "bT_H"
    )

    bT_H_vals = posterior[
        :,
        bT_H_idx
    ]

    if not np.isfinite(
        bT_H_vals
    ).all():

        raise RuntimeError(
            "bT_H posterior contains "
            "non-finite values."
        )

    median = float(
        np.quantile(
            bT_H_vals,
            0.50,
        )
    )

    q025 = float(
        np.quantile(
            bT_H_vals,
            0.025,
        )
    )

    q975 = float(
        np.quantile(
            bT_H_vals,
            0.975,
        )
    )

    p_pos = float(
        np.mean(
            bT_H_vals > 0.0
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "bT_H POSTERIOR UNDER temperature_only "
        "(vapour pressure and rainfall excluded from beta_H)"
    )

    print(
        "=" * 100
    )

    print(
        f"median = "
        f"{median:.4f}"
    )

    print(
        f"95% CrI = "
        f"({q025:.4f}, "
        f"{q975:.4f})"
    )

    print(
        "P(bT_H > 0 | data) = "
        f"{p_pos:.4f}"
    )

    if q025 > 0:

        print(
            "\n[RESULT] The posterior sign remains "
            "positive and is now resolved when vapour pressure "
            "is excluded: the 95% CrI lies entirely above zero. "
            "This is consistent with the full model's non-resolution "
            "of bT_H being (at least partly) a collinearity artifact "
            "with be_H, rather than evidence of no independent effect."
        )

    elif q975 < 0:

        print(
            "\n[RESULT] The posterior lies entirely "
            "below zero when vapour pressure is excluded -- "
            "a sign reversal from the full model's (unresolved, "
            "nominally positive) point estimate. Interpret with caution "
            "and check for other structural changes before over-reading "
            "this reversal."
        )

    else:

        print(
            "\n[RESULT] The temperature-only 95% CrI "
            "still includes zero, so the direction remains "
            "unresolved even with vapour pressure excluded. "
            "This strengthens the case that temperature has a "
            "genuinely weak/absent independent effect on "
            "prevalence-associated amplification, rather than "
            "an effect masked purely by collinearity with humidity."
        )

    # =========================================================================
    # 9. Save posterior samples used for the summary
    # =========================================================================

    posterior_df = pd.DataFrame(
        posterior,
        columns=param_names,
    )

    posterior_df.insert(
        0,
        "equal_weight_draw",
        np.arange(
            1,
            len(posterior_df) + 1,
        ),
    )

    posterior_csv = (
        OUTPUT_ROOT
        / "temperature_only_equal_weight_posterior.csv"
    )

    posterior_df.to_csv(
        posterior_csv,
        index=False,
    )

    print(
        f"\n[OK] Saved equal-weight posterior: "
        f"{posterior_csv}"
    )

    # =========================================================================
    # 10. Save nested-sampling result
    # =========================================================================

    stem = (
        f"{BETA_H_MODE}"
        f"__{FIT_TARGET}"
        f"__check"
    )

    pkl_path = (
        NESTED_DIR
        / f"{stem}.pkl"
    )

    with open(
        pkl_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "results":
                    results,

                "betaH_mode":
                    BETA_H_MODE,

                "fit_target":
                    FIT_TARGET,

                "param_names":
                    param_names,

                "bounds":
                    final_bounds,

                "bounds_source":
                    "src/priors.py",

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

                "bound_method":
                    "multi",

                "sample_method":
                    "rslice",

                "manually_seeded_live_point":
                    False,

                "sampler_random_seed":
                    RANDOM_SEED_SAMPLER,

                "posterior_resample_seed":
                    RANDOM_SEED_POSTERIOR,

                "elapsed_minutes":
                    elapsed_minutes,

                "bT_H_median":
                    median,

                "bT_H_q025":
                    q025,

                "bT_H_q975":
                    q975,

                "bT_H_p_positive":
                    p_pos,
            },
            f,
        )

    # =========================================================================
    # 11. Save JSON summary
    # =========================================================================

    json_path = (
        NESTED_DIR
        / f"{stem}.json"
    )

    best_theta_dict = {
        name:
            float(value)
        for name, value
        in zip(
            param_names,
            best_theta,
        )
    }

    bounds_dict = {
        name:
            [
                float(bounds[0]),
                float(bounds[1]),
            ]
        for name, bounds
        in zip(
            param_names,
            final_bounds,
        )
    }

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

                "param_names":
                    param_names,

                "bounds":
                    bounds_dict,

                "bounds_source":
                    "src/priors.py",

                "best_theta":
                    best_theta_dict,

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

                "bound_method":
                    "multi",

                "sample_method":
                    "rslice",

                "manually_seeded_live_point":
                    False,

                "sampler_random_seed":
                    RANDOM_SEED_SAMPLER,

                "posterior_resample_seed":
                    RANDOM_SEED_POSTERIOR,

                "elapsed_minutes":
                    elapsed_minutes,

                "bT_H_median":
                    median,

                "bT_H_ci":
                    [
                        q025,
                        q975,
                    ],

                "bT_H_p_positive":
                    p_pos,
            },
            f,
            indent=2,
        )

    print(
        f"[OK] Saved nested result: "
        f"{pkl_path}"
    )

    print(
        f"[OK] Saved JSON summary: "
        f"{json_path}"
    )

    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()