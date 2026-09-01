# =============================================================================
# derived_quantities.py
# =============================================================================
"""
Tests one specific, checkable claim about the posterior: that gamma and
betaH0 are correlated (a diagonal/elongated cloud in their joint
posterior, not a round blob), and that their ratio
R0_H = betaH0 / gamma is more tightly constrained than either factor
alone -- the textbook signature of correlated parameters partially
cancelling in a derived ratio.

Neither posterior_analysis.py (marginal summaries only) nor
posterior_predictive.py's corner plot (visual, no numbers) currently
demonstrates this quantitatively. This script does: it reports the
weighted Pearson correlation between gamma and betaH0, and compares the
relative width (95% CI width / median) of gamma, betaH0, and R0_H
against each other.

*** FIXED: this now reads from results/final_model_comparison_v2/nested_sampling/,
the correct final posteriors (betaH0<=3.0 common prior across all four models,
rslice sampling, seeded live points). The previous version of this script
read via paths.NESTED_SAMPLING_DIR, which resolves to results/nested_sampling/
-- the FIRST-generation runs, from before even the aR/a0 bound-widening fix
and before betaH0's own bound was widened from 1.5 to 3.0 (see
posterior_analysis.py's docstring and run_all_climate_betaH0_sensitivity_v2.py
for the history of that fix). A run against the old directory would silently
report gamma/betaH0/correlation numbers from a superseded posterior -- for
example, the old all_climate gamma median of ~1.82 versus the confirmed
final_v2 posterior, which pins much closer to its upper bound (Table 1 MLE
gamma=2.7594). Always verify against EXPECTED_ALL_CLIMATE_LOGZ below before
trusting a printed correlation number. ***

Usage:
    python scripts/derived_quantities.py                       # analyzes every fitted model found
    python scripts/derived_quantities.py --betaH_mode humidity_only   # just one
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import paths  # noqa: E402

FIT_TARGET_DEFAULT = "plague_deaths_reported_fit"

# *** final_v2 -- the corrected results directory. Do NOT point this back at
# paths.NESTED_SAMPLING_DIR (results/nested_sampling/) -- that's the
# first-generation, most-superseded result set in the project. ***
RESULT_DIR = PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"

# Expected logZ for all_climate, for the sanity check below (from the
# verified final_v2 run). If this doesn't match, RESULT_DIR is pointing at
# the wrong posterior and nothing printed by this script should be trusted.
EXPECTED_ALL_CLIMATE_LOGZ = -813.629
LOGZ_TOLERANCE = 1.0


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    return float(cov / np.sqrt(vx * vy))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(weights)
    cw /= cw[-1]
    return np.interp(q, cw, values)


def relative_width(values: np.ndarray, weights: np.ndarray):
    lo, med, hi = weighted_quantile(values, weights, [0.025, 0.5, 0.975])
    rel = (hi - lo) / med if med != 0 else np.inf
    return rel, (lo, med, hi)


def analyze_one(betaH_mode: str, fit_target: str) -> bool:
    """Runs the correlation/ratio check for one saved run. Returns False
    (and prints why) if that run can't be analyzed, so the caller can
    keep going with any other models found rather than stopping entirely."""
    # final_v2 files are named "<betaH_mode>__<fit_target>__final_v2.pkl",
    # not paths.run_stem's older "<betaH_mode>__<fit_target>.pkl" pattern.
    pkl_path = RESULT_DIR / f"{betaH_mode}__{fit_target}__final_v2.pkl"
    if not pkl_path.exists():
        print(f"[skip] No saved final_v2 results at {pkl_path}.")
        return False

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    results = d["results"]
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()
    names = d["param_names"]

    if "gamma" not in names or "betaH0" not in names:
        print(f"[skip] {betaH_mode}: gamma and betaH0 must both be free parameters; got {names}")
        return False

    logZ = float(d["logZ"]) if "logZ" in d else float(results.logz[-1])

    print(f"betaH_mode: {betaH_mode}   fit_target: {fit_target}")
    print(f"logZ = {logZ:.3f}")

    if betaH_mode == "all_climate":
        if abs(logZ - EXPECTED_ALL_CLIMATE_LOGZ) > LOGZ_TOLERANCE:
            print(
                f"*** WARNING: all_climate logZ = {logZ:.3f}, expected approximately "
                f"{EXPECTED_ALL_CLIMATE_LOGZ:.3f}. RESULT_DIR may NOT be pointing at "
                f"the verified final_v2 results -- DO NOT TRUST the numbers below "
                f"until this is resolved. ***"
            )
        else:
            print(f"[OK] all_climate logZ matches the verified final_v2 value.")

    gamma = samples[:, names.index("gamma")]
    betaH0 = samples[:, names.index("betaH0")]
    R0_H = betaH0 / gamma

    rho = weighted_corr(gamma, betaH0, weights)
    rw_gamma, (glo, gmed, ghi) = relative_width(gamma, weights)
    rw_betaH0, (blo, bmed, bhi) = relative_width(betaH0, weights)
    rw_R0H, (rlo, rmed, rhi) = relative_width(R0_H, weights)

    print(f"weighted Pearson correlation(gamma, betaH0) = {rho:+.3f}")
    print()
    print(f"{'quantity':10s} {'median':>10s} {'95% CI':>24s} {'rel. width (CI/median)':>26s}")
    print(f"{'gamma':10s} {gmed:10.4f} [{glo:9.4f}, {ghi:9.4f}] {rw_gamma:26.3f}")
    print(f"{'betaH0':10s} {bmed:10.4f} [{blo:9.4f}, {bhi:9.4f}] {rw_betaH0:26.3f}")
    print(f"{'R0_H':10s} {rmed:10.4f} [{rlo:9.4f}, {rhi:9.4f}] {rw_R0H:26.3f}")
    print()

    if abs(rho) > 0.3 and rw_R0H < min(rw_gamma, rw_betaH0):
        print("CONFIRMED: gamma and betaH0 are correlated, and R0_H = betaH0/gamma "
              "is more tightly constrained (smaller relative width) than either "
              "factor alone -- the claim holds on this data.")
    elif abs(rho) <= 0.3:
        print("NOT CONFIRMED: |correlation| <= 0.3 -- the diagonal-cloud claim "
              "doesn't hold as stated; gamma and betaH0 look closer to independent "
              "in this posterior. Don't assert the correlated-parameters framing "
              "for this model without revisiting this number.")
    else:
        print("PARTIAL: gamma and betaH0 are correlated, but R0_H isn't actually "
              "tighter than the individually-diffuse factors -- the correlation "
              "exists but doesn't buy the constraint the follow-up sentence claims.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--betaH_mode",
        default=None,
        help="Which fitted model to analyze. If omitted, analyzes every "
             "<betaH_mode>__<fit-target>__final_v2.pkl found in "
             "results/final_model_comparison_v2/nested_sampling/.",
    )
    ap.add_argument("--fit-target", default=FIT_TARGET_DEFAULT)
    args = ap.parse_args()

    if args.betaH_mode is not None:
        ok = analyze_one(args.betaH_mode, args.fit_target)
        if not ok:
            raise SystemExit(1)
        return

    # No --betaH_mode given: discover every saved final_v2 run for this
    # fit_target and analyze each one in turn.
    suffix = f"__{args.fit_target}__final_v2.pkl"
    found = sorted(
        p.name[: -len(suffix)]
        for p in RESULT_DIR.glob(f"*{suffix}")
    ) if RESULT_DIR.exists() else []

    if not found:
        raise SystemExit(
            f"No saved final_v2 nested-sampling results found in {RESULT_DIR} "
            f"for fit_target={args.fit_target!r}. Run run_final_all_models_v2.py "
            "first (for at least one --betaH_mode)."
        )

    print(f"No --betaH_mode given -- analyzing all {len(found)} fitted model(s) found: {found}\n")
    for i, betaH_mode in enumerate(found):
        analyze_one(betaH_mode, args.fit_target)
        if i < len(found) - 1:
            print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()