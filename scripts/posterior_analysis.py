# =============================================================================
# posterior_analysis.py
# =============================================================================
"""
Summarize all saved nested-sampling runs found in
results/final_model_comparison_v2/nested_sampling/.

*** FIXED: this now reads from results/final_model_comparison_v2/, the
correct final posteriors (betaH0<=3.0 common prior across all four models,
rslice sampling, seeded live points). The previous version of this script
read via paths.NESTED_SAMPLING_DIR, which resolves to results/nested_sampling/
-- the FIRST-generation runs, from before even the aR/a0 bound-widening fix.
Its "Bayesian model comparison" table would have shown the original,
since-reversed ranking (constant/humidity-only ahead of all-climate).
If you have an old copy of model_comparison_bayesian.csv or any
posterior_summary_*.csv generated before this fix, regenerate them. ***

  1. Per-model, per-parameter posterior summary (weighted mean/median/95% CI),
     saved to results/final_model_comparison_v2/posterior_summaries/posterior_summary_<betaH_mode>.csv
  2. A model-comparison table (log Z, delta-log-Z, Bayes factor vs the
     best-supported model), saved to
     results/final_model_comparison_v2/model_comparison_bayesian_v2.csv
  3. An automatic boundary-pinning check based on posterior MASS near each
     edge, not just where the credible interval happens to land. A wide
     but genuinely diffuse posterior will have a 95% CI that reaches
     close to both walls without actually piling up against either one --
     checking CI-edge proximity alone would falsely flag it. Instead we
     compare the weighted probability mass in the outer `EDGE_BAND`
     fraction of the box to the mass a flat (prior-dominated, i.e.
     "the data told us nothing here") posterior would place there. A
     ratio well above 1 means real pinning; a ratio near 1 means the
     posterior is just wide, which is a separate (and much less
     concerning) thing.

     NOTE: as of the final_v2 results, gamma is known to be pinned at its
     upper bound in the all_climate model (a genuine gamma-betaH0
     degeneracy, not resolved by further widening -- see
     run_all_climate_gamma_sensitivity.py and the project README). Expect
     this script to flag it; that is correct behavior, not a bug.

Usage:
    python scripts/posterior_analysis.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIT_TARGET = "plague_deaths_reported_fit"

# *** final_v2 -- the corrected results directory. Do NOT point this back at
# paths.NESTED_SAMPLING_DIR (results/nested_sampling/) -- that's the
# first-generation, most-superseded result set in the project. ***
RESULT_DIR = PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_model_comparison_v2"
POSTERIOR_SUMMARY_DIR = OUTPUT_ROOT / "posterior_summaries"
COMPARISON_CSV = OUTPUT_ROOT / "model_comparison_bayesian_v2.csv"

# Expected logZ for all_climate, for the sanity check below (from the
# verified final_v2 run).
EXPECTED_ALL_CLIMATE_LOGZ = -813.629

EDGE_BAND = 0.05     # width of the "near the wall" band, as a fraction of the prior box
PIN_RATIO = 2.0       # flag if observed mass in that band exceeds this multiple of the null (EDGE_BAND)


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(weights)
    cw /= cw[-1]
    return np.interp(q, cw, values)


def edge_mass_ratio(vals, weights, lo_bound, hi_bound, band=EDGE_BAND):
    """
    (lower_ratio, upper_ratio): weighted mass in the outer `band` fraction
    of the box near each edge, divided by `band` itself (the null
    expectation under a flat/uninformative posterior). >> 1 means the
    posterior is piling up against that wall.
    """
    width = hi_bound - lo_bound
    lo_edge = lo_bound + band * width
    hi_edge = hi_bound - band * width
    w = weights / weights.sum()
    mass_lo = w[vals <= lo_edge].sum()
    mass_hi = w[vals >= hi_edge].sum()
    return mass_lo / band, mass_hi / band


def summarize_one(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    results = d["results"]
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()

    rows = []
    for i, name in enumerate(d["param_names"]):
        vals = samples[:, i]
        lo_bound, hi_bound = d["bounds"][i]

        mean = np.average(vals, weights=weights)
        ci_lo, med, ci_hi = weighted_quantile(vals, weights, [0.025, 0.5, 0.975])

        ratio_lo, ratio_hi = edge_mass_ratio(vals, weights, lo_bound, hi_bound)
        pinned = ""
        if ratio_lo > PIN_RATIO and ratio_lo >= ratio_hi:
            pinned = "LOWER"
        elif ratio_hi > PIN_RATIO:
            pinned = "UPPER"

        rows.append({
            "param": name, "mean": mean, "median": med,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "prior_lo": lo_bound, "prior_hi": hi_bound,
            "edge_mass_ratio_lower": ratio_lo, "edge_mass_ratio_upper": ratio_hi,
            "pinned_at_boundary": pinned,
        })

    param_df = pd.DataFrame(rows)

    summary = {
        "betaH_mode": d["betaH_mode"],
        "fit_target": d["fit_target"],
        "k": len(d["param_names"]),
        "logZ": d["logZ"] if "logZ" in d else results.logz[-1],
        "logZ_err": d["logZ_error"] if "logZ_error" in d else results.logzerr[-1],
        "n_params_pinned": (param_df["pinned_at_boundary"] != "").sum(),
    }
    return summary, param_df


def main():
    print(f"Reading posteriors from:\n{RESULT_DIR}\n")

    pkl_files = sorted(RESULT_DIR.glob(f"*__{FIT_TARGET}__final_v2.pkl"))
    if not pkl_files:
        raise SystemExit(
            f"No final_v2 nested-sampling results found in {RESULT_DIR}. "
            "Run scripts/run_final_all_models_v2.py first."
        )

    print(f"Found {len(pkl_files)} model results: "
          f"{[p.name for p in pkl_files]}\n")

    POSTERIOR_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for pkl_path in pkl_files:
        summary, param_df = summarize_one(pkl_path)
        summaries.append(summary)

        param_out = POSTERIOR_SUMMARY_DIR / f"posterior_summary_{summary['betaH_mode']}__final_v2.csv"
        param_df.to_csv(param_out, index=False)
        print(f"[OK] Saved: {param_out}")

        pinned = param_df[param_df["pinned_at_boundary"] != ""]
        if not pinned.empty:
            print(f"  NOTE -- {summary['betaH_mode']}: parameter(s) pinned at prior boundary:")
            for _, row in pinned.iterrows():
                ratio = row["edge_mass_ratio_lower"] if row["pinned_at_boundary"] == "LOWER" else row["edge_mass_ratio_upper"]
                extra = ""
                if summary["betaH_mode"] == "all_climate" and row["param"] == "gamma":
                    extra = (
                        "  <-- EXPECTED: known gamma-betaH0 degeneracy, see "
                        "run_all_climate_gamma_sensitivity.py / README. Do not "
                        "widen further expecting this to resolve."
                    )
                if row["param"] == "psi":
                    extra = "  <-- check whether psi<=1 is a genuine biological constraint before widening."
                print(f"    {row['param']}: {ratio:.1f}x more posterior mass near the "
                      f"{row['pinned_at_boundary'].lower()} wall than a flat posterior would have "
                      f"(prior [{row['prior_lo']:.4g}, {row['prior_hi']:.4g}]){extra}")

    comparison = pd.DataFrame(summaries).sort_values("logZ", ascending=False).reset_index(drop=True)

    # Sanity check against the known-correct value
    all_climate_row = comparison[comparison["betaH_mode"] == "all_climate"]
    if not all_climate_row.empty:
        actual_logz = all_climate_row["logZ"].iloc[0]
        if abs(actual_logz - EXPECTED_ALL_CLIMATE_LOGZ) > 1.0:
            print(
                f"\n*** WARNING: all_climate logZ = {actual_logz:.3f}, expected approximately "
                f"{EXPECTED_ALL_CLIMATE_LOGZ:.3f}. This suggests RESULT_DIR is NOT pointing at "
                f"the verified final_v2 results -- check the path before trusting this table. ***"
            )
        else:
            print(f"\n[OK] all_climate logZ = {actual_logz:.3f}, matches the verified final_v2 value.")

    if comparison["fit_target"].nunique() > 1:
        print(
            "\nWARNING: results for more than one fit_target are present "
            f"({sorted(comparison['fit_target'].unique())}). log(Z) is only "
            "comparable across models fit to the SAME data -- comparing across "
            "different fit_targets here is apples-to-oranges."
        )

    best_logZ = comparison["logZ"].iloc[0]
    comparison["delta_logZ"] = best_logZ - comparison["logZ"]
    comparison["bayes_factor_vs_best"] = np.exp(-comparison["delta_logZ"])

    comparison.to_csv(COMPARISON_CSV, index=False)

    print("\n=== Bayesian model comparison (final_v2) ===")
    print(comparison.to_string(index=False))
    if (comparison["n_params_pinned"] > 0).any():
        print(
            "\nNOTE: one or more models have parameters pinned at a prior boundary "
            "(see notes above). For all_climate's gamma, this is a known, investigated "
            "structural degeneracy with betaH0 (not a truncation artifact -- widening "
            "gamma's bound further does not resolve it; see "
            "check_spillover_reseeding_robustness.py for why this doesn't undermine the "
            "main scientific conclusions). For psi, confirm whether the psi<=1 bound is a "
            "genuine biological constraint before treating pinning there as a problem."
        )
    print(f"\n[OK] Saved: {COMPARISON_CSV}")


if __name__ == "__main__":
    main()