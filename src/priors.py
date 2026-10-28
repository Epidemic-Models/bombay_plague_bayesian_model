# =============================================================================
# priors.py
# =============================================================================
"""
Prior parameter bounds for the beta_H formulations implemented in
mechanistic_model.py: "constant", "sigmoid", "humidity_only",
"rainfall_only", "temperature_only", "all_climate", "all_climate_quadT".

*** UPDATED to match the FINAL bounds used in run_final_all_models_v2.py. ***

These are no longer the original LHS-screening bounds. Two rounds of
prior-sensitivity analysis (documented in the project README) found that
the original bounds structurally excluded the true posterior for several
parameters:

  - aR, a0: the original (-1,1) / (-9,-5.5) bounds walled off the
    all-climate model's real optimum (posterior aR ~ -2.2, well outside
    the old box). This was the primary cause of the LHS/AIC -> Bayesian
    evidence reversal in model ranking.
  - betaH0: the original (0.01, 1.5) bound genuinely truncated the
    all-climate posterior (P(betaH0 > 1.5 | data) = 0.447 once widened;
    verified via run_all_climate_betaH0_sensitivity_v2.py, rslice
    sampling + a seeded live point, confirmed the posterior settles
    interior to the new (0.01, 3.0) box rather than piling up again).
  - C_seed, psi: widened during the same round of fixes
    (see run_final_all_models.py's docstring).

Widening gamma further was tested (run_all_climate_gamma_sensitivity.py,
gamma <= 6.0) and did NOT resolve cleanly -- gamma and betaH0 are
genuinely degenerate in this model (the data constrain their ratio,
betaH0/gamma, i.e. the human-transmission reproduction number, much
better than either rate constant individually). gamma's bound is
therefore left at its original (0.5, 3.0) here; do not widen it further
expecting a clean interior peak to appear -- see
check_spillover_reseeding_robustness.py for why this degeneracy does not
undermine the model's main scientific conclusions.

psi's upper bound is retained at 1.0 deliberately: it is a genuine
biological probability (infection-to-death scaling), not an arbitrary
computational choice, so it should not be widened past 1 even though it
is pinned there in all four models -- see the project README / Discussion
notes on this point (Gani & Leach 2004; ECDC plague fact sheet, on
untreated case-fatality rates).

"rainfall_only" and "temperature_only" are single-covariate ablation
formulations used as sensitivity/independence checks for the rainfall and
temperature effects on prevalence-associated amplification respectively
(see run_rainfall_only_model.py and run_temperature_only_model.py). Both
use only the single relevant EXTRA_BOUNDS entry (bR_H or bT_H) on top of
SHARED_BOUNDS, keeping k=12 for both -- the same complexity as
"humidity_only" -- so that Bayesian-evidence comparisons between them are
at equal model complexity.

"all_climate_quadT" is a sensitivity formulation: the all-climate
amplification term plus a quadratic temperature term, bTT_H * zT^2, so that
amplification can peak at an intermediate temperature in the same way as
spillover. Its bound (-2, 0) matches aTT: a non-positive quadratic term
gives a single peak (or, at bTT_H = 0, the linear all_climate form), never a
U-shape. It has k = 15.

If you need the ORIGINAL (round-0 / LHS-screening) bounds for a specific
diagnostic or historical-comparison purpose, they are preserved as
ORIGINAL_LHS_BOUNDS at the bottom of this file -- do not use them for
fitting, only for the explicit before/after diff in
make_supp_prior_table.py.

One deliberate change from sample.py's original naming: the sigmoid
steepness parameter is named `kappa` here (matching what
mechanistic_model.py reads out of `params`), not `k_sig`.
"""

from __future__ import annotations
import numpy as np


# ------------------------------------------------------------
# Bounds shared by every beta_H formulation
# (matches COMMON_BOUNDS in run_final_all_models_v2.py -- keep these two
# in sync if either changes)
# ------------------------------------------------------------
SHARED_BOUNDS = {
    "N0":       (800_000.0, 850_000.0),
    "C_seed":   (0.0, 150.0),
    "gamma":    (0.5, 3.0),        # see docstring: NOT widened -- degenerate with betaH0, not truncated
    "a0":       (-11.0, -5.5),     # widened from (-9.0, -5.5)
    "aT":       (-2.0, 1.0),       # widened from (-1.0, 1.0)
    "aTT":      (-2.0, 0.0),       # widened from (-1.0, 0.0)
    "ae":       (-1.0, 1.0),       # unchanged
    "aR":       (-4.0, 1.0),       # widened from (-1.0, 1.0) -- see docstring, this was the main fix
    "betaH0":   (0.01, 3.0),       # widened from (0.01, 1.5) -- confirmed genuine truncation, now resolved
    "psi":      (0.05, 1.0),       # widened from (0.05, 0.8); upper bound of 1.0 is a real constraint, keep it
    "nb_kappa": (0.1, 80.0),       # unchanged
}

SHARED_PARAM_ORDER = [
    "N0", "C_seed", "gamma",
    "a0", "aT", "aTT", "ae", "aR",
    "betaH0", "psi", "nb_kappa",
]

# ------------------------------------------------------------
# Extra parameters, only needed by specific beta_H formulations
# (matches EXTRA_BOUNDS in run_final_all_models_v2.py; unchanged from
# the original LHS-screening bounds -- these were never found to be
# truncated)
# ------------------------------------------------------------
EXTRA_BOUNDS = {
    "theta":  (-1.5, 1.5),   # sigmoid: amplitude of the human-transmission shift
    "t_star": (4.0, 20.0),   # sigmoid: transition week
    "kappa":  (0.2, 3.0),    # sigmoid: transition steepness (sample.py's "k_sig")
    "be_H":   (-2.0, 2.0),   # humidity_only, all_climate: vapour-pressure effect
    "bT_H":   (-2.0, 2.0),   # temperature_only, all_climate: temperature effect on beta_H
    "bR_H":   (-2.0, 2.0),   # rainfall_only, all_climate: rainfall effect on beta_H
    "bTT_H":  (-2.0, 0.0),   # all_climate_quadT: quadratic temperature effect on beta_H (same bound as aTT)
}

# ------------------------------------------------------------
# One entry per betaH_mode string accepted by mechanistic_model.py --
# the dict key IS the betaH_mode value, so there's no separate name
# mapping to keep in sync.
# ------------------------------------------------------------
MODEL_SPECS = {
    "constant": {
        "extra_params": [],
    },
    "sigmoid": {
        "extra_params": ["theta", "t_star", "kappa"],
    },
    "humidity_only": {
        "extra_params": ["be_H"],
    },
    "rainfall_only": {
        "extra_params": ["bR_H"]
    },
    "temperature_only": {
        "extra_params": ["bT_H"]
    },
    "all_climate": {
        "extra_params": ["bT_H", "be_H", "bR_H"],
    },
    "all_climate_quadT": {
        "extra_params": ["bT_H", "bTT_H", "be_H", "bR_H"],
    },
}


def get_param_names(betaH_mode: str) -> list[str]:
    """Ordered list of free-parameter names for this beta_H formulation."""
    if betaH_mode not in MODEL_SPECS:
        raise ValueError(f"Unknown betaH_mode={betaH_mode!r}. Use one of {list(MODEL_SPECS)}.")
    return SHARED_PARAM_ORDER + list(MODEL_SPECS[betaH_mode]["extra_params"])


def get_bounds(betaH_mode: str) -> list[tuple[float, float]]:
    """Ordered list of (low, high) bounds matching get_param_names()."""
    names = get_param_names(betaH_mode)
    bounds = []
    for name in names:
        if name in SHARED_BOUNDS:
            bounds.append(SHARED_BOUNDS[name])
        else:
            bounds.append(EXTRA_BOUNDS[name])
    return bounds


def prior_transform(u: np.ndarray, betaH_mode: str) -> np.ndarray:
    """
    Map a point in the unit hypercube to physical parameter values by
    linear scaling within each parameter's uniform bounds.

    This is the function nested samplers (dynesty, UltraNest) call
    directly -- they explore the unit cube internally and use this to
    convert each proposal into an actual parameter vector before your
    simulator ever sees it.
    """
    u = np.asarray(u, dtype=float)
    bounds = get_bounds(betaH_mode)
    theta = np.empty_like(u)
    for i, (lo, hi) in enumerate(bounds):
        theta[i] = lo + u[i] * (hi - lo)
    return theta


def log_prior(theta: np.ndarray, betaH_mode: str) -> float:
    """
    Unnormalized flat log-prior: 0.0 inside the box, -inf outside.

    Used by MCMC samplers (e.g. emcee) that need log_prior +
    log_likelihood combined into a single log_posterior function,
    rather than dynesty's separate prior_transform approach.
    """
    bounds = get_bounds(betaH_mode)
    for val, (lo, hi) in zip(theta, bounds):
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def params_dict_from_theta(theta: np.ndarray, betaH_mode: str) -> dict:
    """
    Convenience: turn a flat theta vector into the {name: value} dict
    form used elsewhere (e.g. for printing, or building the `params`
    dict mechanistic_model.simulate_seir_dde expects).
    """
    names = get_param_names(betaH_mode)
    return dict(zip(names, np.asarray(theta, dtype=float)))


# =============================================================================
# Historical reference only -- the ORIGINAL LHS-screening bounds, preserved
# for the explicit before/after diff in make_supp_prior_table.py. DO NOT
# use these for fitting; SHARED_BOUNDS above is the current, correct set.
# =============================================================================
ORIGINAL_LHS_BOUNDS = {
    "N0":       (800_000.0, 850_000.0),
    "C_seed":   (0.0, 100.0),
    "gamma":    (0.5, 3.0),
    "a0":       (-9.0, -5.5),
    "aT":       (-1.0, 1.0),
    "aTT":      (-1.0, 0.0),
    "ae":       (-1.0, 1.0),
    "aR":       (-1.0, 1.0),
    "betaH0":   (0.01, 1.5),
    "psi":      (0.05, 0.8),
    "nb_kappa": (0.1, 80.0),
    "theta":    (-1.5, 1.5),
    "t_star":   (4.0, 20.0),
    "kappa":    (0.2, 3.0),
    "be_H":     (-1.0, 1.0),
    "bT_H":     (-1.0, 1.0),
    "bR_H":     (-1.0, 1.0),
}