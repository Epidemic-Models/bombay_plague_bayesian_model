# =============================================================================
# run_tau_delay_sensitivity_all_climate.py
# =============================================================================
"""
Two-axis sensitivity analysis for the all-climate model:

  Axis 1 (tau_only)
      Vary the fixed incubation/exposed duration tau0_days over
      {2, 4, 7, 9} days. The infection-to-death delay kernel is left
      at the ORIGINAL final_v2 "gamma" (midpoint-discretized,
      shape=2.0, scale=0.6 weeks) for every value of tau0_days. This
      tests sensitivity to tau alone, with the death-delay kernel
      unchanged.

  Axis 2 (tau_plus_shifted_kernel)
      Vary tau0_days over {2, 4, 7} days, each paired with the
      "gamma_shifted" delay kernel, whose hard lower floor
      (shift_weeks) is set equal to tau0_days / 7 weeks -- enforcing
      "no death before the person has mechanistically left the
      exposed compartment." The post-shift Gamma scale is solved so
      the TOTAL mean delay (shift_weeks + shape * scale) stays fixed
      at 1.2 weeks (8.4 days) for every tau0_days in this axis.

      tau0_days = 9 is EXCLUDED from axis 2: 9/7 ~= 1.286 weeks
      already exceeds the 1.2-week target total mean, so no positive
      post-shift scale satisfies the equal-mean constraint. It is
      still covered by axis 1 (plain "gamma").

tau0_days = 7 with delay_kind = "gamma" (shape=2.0, scale=0.6) is the
exact final_v2 baseline and is included in axis 1 as a sanity check
(should reproduce the final_v2 all-climate logZ to within sampler
noise).

The original final_v2 model outputs are NOT overwritten.

Outputs
-------
results/tau_delay_sensitivity_all_climate/
    nested_sampling/
        all_climate__tau{N}d_gamma.pkl / .json             (axis 1)
        all_climate__tau{N}d_gamma_shifted.pkl / .json      (axis 2)

    tau_delay_sensitivity_model_comparison.csv

Usage
-----
    python scripts/run_tau_delay_sensitivity_all_climate.py
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
# Each scenario receives its own offset so the runs remain
# independently reproducible.
BASE_SEED_LIVE_POINTS = 51000
BASE_SEED_DYNESTY = 61000


# =============================================================================
# tau / delay-kernel scenarios
# =============================================================================

ORIGINAL_SHAPE = 2.0
ORIGINAL_SCALE = 0.6
ORIGINAL_MAX_WEEKS = 12
TARGET_TOTAL_MEAN_WEEKS = 1.2  # shift_weeks + shape * scale, held fixed across axis 2


def shifted_scale_for_tau(tau_days: float, shape: float = ORIGINAL_SHAPE,
                           target_total_mean_weeks: float = TARGET_TOTAL_MEAN_WEEKS) -> float:
    """
    Solve for the post-shift Gamma scale that keeps

        shift_weeks + shape * scale == target_total_mean_weeks

    where shift_weeks = tau_days / 7. Raises ValueError if tau_days
    alone already exceeds the target total mean (no positive scale
    can satisfy the constraint).
    """
    shift_weeks = tau_days / 7.0
    remaining_mean = target_total_mean_weeks - shift_weeks
    if remaining_mean <= 0:
        raise ValueError(
            f"tau_days={tau_days} (shift_weeks={shift_weeks:.3f} weeks) already "
            f"exceeds target_total_mean_weeks={target_total_mean_weeks}; no "
            "positive scale satisfies the equal-mean constraint for this tau."
        )
    return remaining_mean / shape


TAU_GRID_DAYS_AXIS1 = [2, 4, 7, 9]
TAU_GRID_DAYS_AXIS2 = [2, 4, 7]  # 9 excluded -- see module docstring

SCENARIOS: dict[str, dict] = {}

for _tau in TAU_GRID_DAYS_AXIS1:
    SCENARIOS[f"tau{_tau}d_gamma"] = {
        "axis": "tau_only",
        "tau0_days": _tau,
        "delay_kind": "gamma",
        "delay_shape": ORIGINAL_SHAPE,
        "delay_scale": ORIGINAL_SCALE,
        "delay_max_weeks": ORIGINAL_MAX_WEEKS,
        "delay_shift_weeks": 1.0,  # ignored by "gamma"
    }

for _tau in TAU_GRID_DAYS_AXIS2:
    SCENARIOS[f"tau{_tau}d_gamma_shifted"] = {
        "axis": "tau_plus_shifted_kernel",
        "tau0_days": _tau,
        "delay_kind": "gamma_shifted",
        "delay_shape": ORIGINAL_SHAPE,
        "delay_scale": shifted_scale_for_tau(_tau),
        "delay_max_weeks": ORIGINAL_MAX_WEEKS,
        "delay_shift_weeks": _tau / 7.0,
    }

del _tau


# =============================================================================
# Output paths
# =============================================================================

OUTPUT_ROOT = PROJECT_ROOT / "results" / "tau_delay_sensitivity_all_climate"
NESTED_DIR = OUTPUT_ROOT / "nested_sampling"
NESTED_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = OUTPUT_ROOT / "tau_delay_sensitivity_model_comparison.csv"


# =============================================================================
# Seed point
#
# Same verified high-likelihood all-climate seed used in the final_v2
# model comparison and in the gamma_cdf discretization sensitivity run.
# It is guaranteed legal under the shared priors (box bounds don't
# depend on tau0_days or the delay kernel).
#
# IMPORTANT: unlike the gamma_cdf sensitivity (which left tau0_days
# at 7 and only changed the death-reporting kernel), this sweep also
# changes tau0_days itself, which changes the DDE incubation delay in
# the mechanistic simulation. The seed is therefore NOT guaranteed to
# have a finite likelihood away from tau0_days=7. Scenarios where it
# fails fall back to a purely random live-point ensemble -- see
# build_live_points().
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
    """Return a dynesty prior-transform function for the supplied uniform bounds."""
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)

    def prior_transform(u):
        u = np.asarray(u, dtype=float)
        return lower + u * (upper - lower)

    return prior_transform


def theta_dict_to_unit_cube(theta_dict: dict, param_names: list[str], bounds) -> np.ndarray:
    """Convert a named parameter vector into its point in the unit prior cube."""
    u = np.empty(len(param_names), dtype=float)
    for i, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        value = float(theta_dict[name])
        if not (lo <= value <= hi):
            raise ValueError(
                f"Seed parameter {name!r}={value} lies outside its prior bounds ({lo}, {hi})."
            )
        u[i] = (value - lo) / (hi - lo)
    return u


def get_result_field(results, name: str):
    """Small compatibility helper for dynesty Results objects."""
    if hasattr(results, name):
        return getattr(results, name)
    return results[name]


def build_live_points(lik, prior_transform, param_names, bounds, live_seed: int):
    """
    Build the initial live-point ensemble for one scenario.

    Tries to place the shared SEED_THETA at live-point 0, as in the
    gamma_cdf sensitivity script. If the seed is non-finite under
    this scenario's (tau0_days, delay_kind) combination -- which can
    happen here because tau0_days also changes the mechanistic DDE,
    not just the observation kernel -- falls back to a fully random
    ensemble instead of raising, and says so.
    """
    ndim = len(param_names)
    live_rng = np.random.default_rng(live_seed)
    live_u = live_rng.random((NLIVE, ndim))
    live_v = np.asarray([prior_transform(u) for u in live_u], dtype=float)

    print(f"\nEvaluating {NLIVE} initial live points...")
    t0 = time.time()
    live_logl = np.asarray([lik.log_likelihood(theta) for theta in live_v], dtype=float)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  non-finite initial points: {np.sum(~np.isfinite(live_logl))} / {NLIVE}")

    seed_theta_vec = np.asarray([SEED_THETA[name] for name in param_names], dtype=float)
    seed_logl = float(lik.log_likelihood(seed_theta_vec))
    print(f"\nShared seed-point logL under this scenario: {seed_logl:.6f}")

    if np.isfinite(seed_logl):
        seed_u = theta_dict_to_unit_cube(SEED_THETA, param_names, bounds)
        live_u[0] = seed_u
        live_v[0] = seed_theta_vec
        live_logl[0] = seed_logl
        random_live_best = float(np.nanmax(live_logl[1:]))
        print(f"[OK] Seed placed at live-point 0 (logL={seed_logl:.6f})")
        print(f"     Best random initial live point: logL={random_live_best:.6f}")
        used_seed = True
    else:
        print(
            "[WARN] Shared seed is non-finite under this scenario's tau0_days / "
            "delay kernel (expected away from tau0_days=7 -- the seed was "
            "calibrated against the final_v2 DDE structure). Falling back to a "
            "purely random live-point ensemble for this scenario."
        )
        used_seed = False

    if not np.any(np.isfinite(live_logl)):
        raise RuntimeError(
            "No finite-likelihood live points (seed or random) for this scenario. "
            "Prior bounds or model configuration likely need revisiting before "
            "sampling can proceed."
        )

    return live_u, live_v, live_logl, seed_logl, used_seed


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load prepared data
    # -------------------------------------------------------------------------
    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Prepared data not found:\n{paths.MERGED_CSV}\n\nRun scripts/prepare_data.py first."
        )

    df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
    df = df.sort_values(paths.DATE_COL).reset_index(drop=True)

    if FIT_TARGET not in df.columns:
        raise ValueError(f"{FIT_TARGET!r} not found in prepared data.")

    print("\n" + "=" * 100)
    print("ALL-CLIMATE TAU / DELAY-KERNEL SENSITIVITY")
    print("=" * 100)
    print(f"\nFitted weeks: {len(df)}")
    print(f"Target: {FIT_TARGET}")
    print(f"Model: {BETA_H_MODE}")

    print("\nScenarios:")
    for name, cfg in SCENARIOS.items():
        print(
            f"  {name:<24s} axis={cfg['axis']:<24s} "
            f"tau0_days={cfg['tau0_days']}, kind={cfg['delay_kind']}, "
            f"shape={cfg['delay_shape']}, scale={cfg['delay_scale']:.4f}, "
            f"shift_weeks={cfg['delay_shift_weeks']:.4f}, max_weeks={cfg['delay_max_weeks']}"
        )

    # -------------------------------------------------------------------------
    # Shared prior information
    # -------------------------------------------------------------------------
    final_bounds = priors.get_bounds(BETA_H_MODE)
    expected_param_names = priors.get_param_names(BETA_H_MODE)

    print("\nShared parameter bounds from src/priors.py:")
    for name, bound in zip(expected_param_names, final_bounds):
        print(f"  {name:<12s} {bound}")

    # -------------------------------------------------------------------------
    # Run scenarios
    # -------------------------------------------------------------------------
    comparison_rows = []

    for scenario_index, (scenario_name, cfg) in enumerate(SCENARIOS.items()):

        print("\n\n" + "=" * 100)
        print(f"SCENARIO: {scenario_name}  (axis={cfg['axis']})")
        print("=" * 100)

        lik = PlagueLikelihood(
            merged_df=df,
            betaH_mode=BETA_H_MODE,
            fit_target=FIT_TARGET,
            tau0_days=cfg["tau0_days"],
            delay_kind=cfg["delay_kind"],
            delay_shape=cfg["delay_shape"],
            delay_scale=cfg["delay_scale"],
            delay_max_weeks=cfg["delay_max_weeks"],
            delay_shift_weeks=cfg["delay_shift_weeks"],
        )

        param_names = list(lik.param_names)
        ndim = len(param_names)

        if param_names != expected_param_names:
            raise RuntimeError(
                "Parameter order from PlagueLikelihood does not match "
                "priors.get_param_names(all_climate).\n\n"
                f"Likelihood:\n{param_names}\n\nPriors:\n{expected_param_names}"
            )

        lik.bounds = list(final_bounds)
        print(f"\nFree parameters ({ndim}): {param_names}")

        prior_transform = prior_transform_factory(final_bounds)

        live_seed = BASE_SEED_LIVE_POINTS + scenario_index
        live_u, live_v, live_logl, seed_logl, used_seed = build_live_points(
            lik, prior_transform, param_names, final_bounds, live_seed,
        )

        dynesty_seed = BASE_SEED_DYNESTY + scenario_index
        rstate = np.random.default_rng(dynesty_seed)

        print("\nStarting dynesty:")
        print("  sample = rslice")
        print(f"  nlive  = {NLIVE}")
        print(f"  dlogz  = {DLOGZ}")
        print(f"  seeded = {used_seed}")
        print(f"  live seed    = {live_seed}")
        print(f"  dynesty seed = {dynesty_seed}")

        sampler = NestedSampler(
            loglikelihood=lik.log_likelihood,
            prior_transform=prior_transform,
            ndim=ndim,
            nlive=NLIVE,
            bound="multi",
            sample="rslice",
            rstate=rstate,
            live_points=(live_u, live_v, live_logl),
        )

        t0 = time.time()
        sampler.run_nested(dlogz=DLOGZ, print_progress=True)
        elapsed_minutes = (time.time() - t0) / 60.0

        # ---------------------------------------------------------------------
        # Extract results
        # ---------------------------------------------------------------------
        results = sampler.results
        logl = np.asarray(get_result_field(results, "logl"), dtype=float)
        samples = np.asarray(get_result_field(results, "samples"), dtype=float)
        logz = np.asarray(get_result_field(results, "logz"), dtype=float)
        logzerr = np.asarray(get_result_field(results, "logzerr"), dtype=float)

        best_index = int(np.nanargmax(logl))
        best_loglik = float(logl[best_index])
        best_theta = samples[best_index].copy()
        logZ = float(logz[-1])
        logZ_error = float(logzerr[-1])

        print(f"\nFinished in {elapsed_minutes:.1f} min")
        print(f"Max sampled logL = {best_loglik:.6f}")
        print(f"Seed logL = {seed_logl:.6f}" if used_seed else "Seed logL = n/a (not seeded)")
        if used_seed:
            print(f"Improvement over seed = {best_loglik - seed_logl:+.6f}")
        print(f"log(Z) = {logZ:.6f} +/- {logZ_error:.6f}")

        # ---------------------------------------------------------------------
        # Likelihood consistency check
        # ---------------------------------------------------------------------
        reevaluated = float(lik.log_likelihood(best_theta))
        diff = reevaluated - best_loglik

        print("\nLikelihood consistency check:")
        print(f"  stored = {best_loglik:.10f}")
        print(f"  reeval = {reevaluated:.10f}")
        print(f"  diff   = {diff:.2e}")

        if not np.isfinite(reevaluated) or abs(diff) > 1e-8:
            raise RuntimeError(f"Likelihood consistency check FAILED for {scenario_name!r}.")

        print("[OK] Consistency confirmed.")

        # ---------------------------------------------------------------------
        # Save result
        # ---------------------------------------------------------------------
        stem = f"{BETA_H_MODE}__{scenario_name}"
        pkl_path = NESTED_DIR / f"{stem}.pkl"
        json_path = NESTED_DIR / f"{stem}.json"

        save_payload = {
            "results": results,
            "betaH_mode": BETA_H_MODE,
            "fit_target": FIT_TARGET,
            "scenario": scenario_name,
            "axis": cfg["axis"],
            "tau0_days": cfg["tau0_days"],
            "delay_kind": cfg["delay_kind"],
            "delay_shape": cfg["delay_shape"],
            "delay_scale": cfg["delay_scale"],
            "delay_max_weeks": cfg["delay_max_weeks"],
            "delay_shift_weeks": cfg["delay_shift_weeks"],
            "param_names": param_names,
            "bounds": final_bounds,
            "best_theta": best_theta,
            "best_loglik": best_loglik,
            "logZ": logZ,
            "logZ_error": logZ_error,
            "nlive": NLIVE,
            "dlogz": DLOGZ,
            "sample": "rslice",
            "seeded": used_seed,
            "seed_loglik": seed_logl,
            "live_points_seed": live_seed,
            "dynesty_seed": dynesty_seed,
            "elapsed_minutes": elapsed_minutes,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(save_payload, f)

        with open(json_path, "w") as f:
            json.dump(
                {
                    "betaH_mode": BETA_H_MODE,
                    "fit_target": FIT_TARGET,
                    "scenario": scenario_name,
                    "axis": cfg["axis"],
                    "tau0_days": cfg["tau0_days"],
                    "delay_kind": cfg["delay_kind"],
                    "delay_shape": cfg["delay_shape"],
                    "delay_scale": cfg["delay_scale"],
                    "delay_max_weeks": cfg["delay_max_weeks"],
                    "delay_shift_weeks": cfg["delay_shift_weeks"],
                    "param_names": param_names,
                    "bounds": {name: list(bound) for name, bound in zip(param_names, final_bounds)},
                    "best_theta": {name: float(value) for name, value in zip(param_names, best_theta)},
                    "best_loglik": best_loglik,
                    "logZ": logZ,
                    "logZ_error": logZ_error,
                    "nlive": NLIVE,
                    "dlogz": DLOGZ,
                    "sample": "rslice",
                    "seeded": used_seed,
                    "seed_loglik": seed_logl,
                    "live_points_seed": live_seed,
                    "dynesty_seed": dynesty_seed,
                    "elapsed_minutes": elapsed_minutes,
                },
                f,
                indent=2,
            )

        print(f"\n[OK] Saved:\n  {pkl_path}")
        print(f"[OK] Saved:\n  {json_path}")

        # ---------------------------------------------------------------------
        # Comparison row
        # ---------------------------------------------------------------------
        comparison_rows.append({
            "scenario": scenario_name,
            "axis": cfg["axis"],
            "tau0_days": cfg["tau0_days"],
            "delay_kind": cfg["delay_kind"],
            "delay_shape": cfg["delay_shape"],
            "delay_scale": cfg["delay_scale"],
            "delay_shift_weeks": cfg["delay_shift_weeks"],
            "delay_max_weeks": cfg["delay_max_weeks"],
            "n_parameters": ndim,
            "max_loglik": best_loglik,
            "logZ": logZ,
            "logZ_error": logZ_error,
            "seeded": used_seed,
            "seed_loglik": seed_logl,
            "elapsed_minutes": elapsed_minutes,
        })

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values("logZ", ascending=False).reset_index(drop=True)

    best_logZ = float(comparison["logZ"].max())
    comparison["delta_logZ"] = best_logZ - comparison["logZ"]
    comparison["BF_best_over_scenario"] = np.exp(np.minimum(comparison["delta_logZ"], 700.0))

    n = len(df)
    k = comparison["n_parameters"].astype(float)

    comparison["AIC"] = 2.0 * k - 2.0 * comparison["max_loglik"]
    comparison["AICc"] = comparison["AIC"] + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    comparison["BIC"] = k * np.log(n) - 2.0 * comparison["max_loglik"]
    comparison["delta_AICc"] = comparison["AICc"] - comparison["AICc"].min()
    comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()

    comparison.to_csv(COMPARISON_CSV, index=False)

    print("\n\n" + "=" * 120)
    print("TAU / DELAY-KERNEL SENSITIVITY — ALL-CLIMATE MODEL")
    print("=" * 120)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n" + comparison.to_string(index=False))

    print(f"\n[OK] Saved comparison:\n  {COMPARISON_CSV}")

    print("\nInterpretation:")
    print(
        "Axis 'tau_only' varies the fixed incubation/exposed duration tau0_days "
        "while holding the original final_v2 death-delay kernel fixed. Axis "
        "'tau_plus_shifted_kernel' additionally enforces a hard death-delay floor "
        "at tau0_days (no death can be observed before the person has left the "
        "exposed compartment), re-solving the post-shift Gamma scale so the total "
        "mean delay stays fixed at 1.2 weeks across that axis. tau0_days=9 is "
        "covered only by the tau_only axis, since no positive post-shift scale "
        "satisfies the equal-mean constraint there (see module docstring)."
    )

    print("\nDone.")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()