# =============================================================================
# posterior_analysis.py
# =============================================================================
"""
Summarize the nested-sampling runs for all six beta_H formulations.

The four main formulations (constant, single shift / sigmoid, vapour pressure
only, climate-driven) live in

    results/final_model_comparison_v2/nested_sampling/

and the two single-covariate formulations (rainfall only, temperature only)
were fitted separately. Each formulation's file path is listed explicitly in
MODELS below, so nothing is picked up by accident from an older result set.

*** Do NOT point any entry at paths.NESTED_SAMPLING_DIR (results/nested_sampling/).
That is the first-generation result set, from before the aR/a0 bound-widening
fix, and its ranking was later reversed. ***

Outputs (results/final_model_comparison_v2/):

  1. Per-model, per-parameter posterior summary (weighted mean/median/95% CI)
     posterior_summaries/posterior_summary_<betaH_mode>__final_v2.csv

  2. Model-comparison table with log Z, its error, Delta log Z, Bayes factor
     vs the best model, maximum log-likelihood, AICc and BIC
     model_comparison_all_formulations.csv
     model_comparison_all_formulations.tex   (rows ready for the manuscript)

  3. Boundary-pinning check based on posterior MASS near each prior edge.
     The weighted mass in the outer EDGE_BAND fraction of the prior box is
     compared with the mass a flat posterior would put there (EDGE_BAND).
     A ratio well above 1 means real pinning; a ratio near 1 means the
     posterior is simply wide.

     NOTE: gamma is known to be pinned at its upper bound in the all_climate
     model (a gamma-betaH0 degeneracy, not resolved by further widening; see
     run_all_climate_gamma_sensitivity.py). Expect it to be flagged.

AICc and BIC are computed from the maximum log-likelihood stored in each run:

    AIC  = 2k - 2 logL
    AICc = AIC + 2k(k+1) / (n - k - 1)
    BIC  = k ln(n) - 2 logL

with k the number of fitted parameters and n the number of fitted weeks.

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

import paths  # noqa: E402

FIT_TARGET = "plague_deaths_reported_fit"

RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = RESULTS_ROOT / "final_model_comparison_v2"
POSTERIOR_SUMMARY_DIR = OUTPUT_ROOT / "posterior_summaries"
COMPARISON_CSV = OUTPUT_ROOT / "model_comparison_all_formulations.csv"
COMPARISON_TEX = OUTPUT_ROOT / "model_comparison_all_formulations.tex"

# -----------------------------------------------------------------------------
# One entry per formulation. Same paths as make_bayesian_model_fits_figure.py.
# *** EDIT the temperature_only path to match your file. ***
# -----------------------------------------------------------------------------
MAIN_DIR = RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"

MODELS = {
    "all_climate": {
        "label": "Climate-driven",
        "path": MAIN_DIR / f"all_climate__{FIT_TARGET}__final_v2.pkl",
    },
    "humidity_only": {
        "label": "Vapour pressure only",
        "path": MAIN_DIR / f"humidity_only__{FIT_TARGET}__final_v2.pkl",
    },
    "rainfall_only": {
        "label": "Rainfall only",
        "path": RESULTS_ROOT / "rainfall_only_check" / "nested_sampling"
                / f"rainfall_only__{FIT_TARGET}__check.pkl",
    },
    "temperature_only": {
        "label": "Temperature only",
        "path": RESULTS_ROOT / "temperature_only_check" / "nested_sampling"
                / f"temperature_only__{FIT_TARGET}__check.pkl",
    },
    "constant": {
        "label": "Constant",
        "path": MAIN_DIR / f"constant__{FIT_TARGET}__final_v2.pkl",
    },
    "sigmoid": {
        "label": "Single shift",
        "path": MAIN_DIR / f"sigmoid__{FIT_TARGET}__final_v2.pkl",
    },
}

# If True, formulations whose file is missing are skipped with a warning.
# Keep False for the final table.
SKIP_MISSING_MODELS = False

# Expected logZ for all_climate, for the sanity check (verified final_v2 run).
EXPECTED_ALL_CLIMATE_LOGZ = -813.629

EDGE_BAND = 0.05   # width of the "near the wall" band, as a fraction of the prior box
PIN_RATIO = 2.0    # flag if mass in that band exceeds this multiple of the flat-posterior mass


# =============================================================================
# Helpers
# =============================================================================

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(weights)
    cw /= cw[-1]
    return np.interp(q, cw, values)


def edge_mass_ratio(vals, weights, lo_bound, hi_bound, band=EDGE_BAND):
    """
    (lower_ratio, upper_ratio): weighted mass in the outer `band` fraction of
    the prior box near each edge, divided by `band` (the flat-posterior mass).
    Values well above 1 mean the posterior is piling up against that wall.
    """
    width = hi_bound - lo_bound
    lo_edge = lo_bound + band * width
    hi_edge = hi_bound - band * width
    w = weights / weights.sum()
    return w[vals <= lo_edge].sum() / band, w[vals >= hi_edge].sum() / band


def information_criteria(max_loglik: float, k: int, n: int) -> tuple[float, float]:
    """Return (AICc, BIC) from the maximum log-likelihood."""
    aic = 2 * k - 2 * max_loglik
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    bic = k * np.log(n) - 2 * max_loglik
    return aicc, bic


def summarize_one(model: str, pkl_path: Path, n_weeks: int):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    stored_mode = d.get("betaH_mode", model)
    if stored_mode != model:
        raise ValueError(f"{pkl_path.name}: file says betaH_mode={stored_mode!r}, "
                         f"but MODELS lists it as {model!r}.")
    if d.get("fit_target", FIT_TARGET) != FIT_TARGET:
        raise ValueError(f"{pkl_path.name}: fitted to {d['fit_target']!r}, not {FIT_TARGET!r}. "
                         "log Z is only comparable across models fitted to the same data.")

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
        p_positive = weights[vals > 0].sum()

        ratio_lo, ratio_hi = edge_mass_ratio(vals, weights, lo_bound, hi_bound)
        pinned = ""
        if ratio_lo > PIN_RATIO and ratio_lo >= ratio_hi:
            pinned = "LOWER"
        elif ratio_hi > PIN_RATIO:
            pinned = "UPPER"

        rows.append({
            "param": name, "mean": mean, "median": med,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "p_positive": p_positive,
            "prior_lo": lo_bound, "prior_hi": hi_bound,
            "edge_mass_ratio_lower": ratio_lo, "edge_mass_ratio_upper": ratio_hi,
            "pinned_at_boundary": pinned,
        })
    param_df = pd.DataFrame(rows)

    k = len(d["param_names"])
    max_loglik = float(np.max(results.logl))
    aicc, bic = information_criteria(max_loglik, k, n_weeks)

    summary = {
        "betaH_mode": model,
        "label": MODELS[model]["label"],
        "source_file": str(pkl_path.relative_to(PROJECT_ROOT)),
        "k": k,
        "max_loglik": max_loglik,
        "logZ": float(d["logZ"]) if "logZ" in d else float(results.logz[-1]),
        "logZ_err": float(d["logZ_error"]) if "logZ_error" in d else float(results.logzerr[-1]),
        "AICc": aicc,
        "BIC": bic,
        "nlive": d.get("nlive"),
        "seeded_live_point": d.get("seeded", d.get("manually_seeded_live_point")),
        "n_params_pinned": int((param_df["pinned_at_boundary"] != "").sum()),
    }
    return summary, param_df


def write_latex_rows(comparison: pd.DataFrame, path: Path) -> None:
    """Write the tabular body for Table 1 (paste between \\hline lines)."""
    lines = []
    for _, r in comparison.iterrows():
        lines.append(
            f"{r['label']:<22} & {r['k']:>2d} & ${r['max_loglik']:.2f}$ & "
            f"${r['logZ']:.2f}$ & {r['delta_logZ']:.2f} & "
            f"{r['AICc']:.2f} & {r['BIC']:.2f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    merged_df = pd.read_csv(paths.MERGED_CSV)
    n_weeks = int(merged_df[FIT_TARGET].notna().sum())
    print(f"Fitted weeks (n for AICc/BIC): {n_weeks}\n")

    available = []
    for model, spec in MODELS.items():
        if spec["path"].exists():
            available.append(model)
        elif SKIP_MISSING_MODELS:
            print(f"[WARNING] Skipping {model!r}: not found at\n  {spec['path']}")
        else:
            raise SystemExit(
                f"Result for {model!r} not found:\n  {spec['path']}\n"
                "Edit its path in MODELS, or set SKIP_MISSING_MODELS = True."
            )

    POSTERIOR_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in available:
        summary, param_df = summarize_one(model, MODELS[model]["path"], n_weeks)
        summaries.append(summary)

        param_out = POSTERIOR_SUMMARY_DIR / f"posterior_summary_{model}__final_v2.csv"
        param_df.to_csv(param_out, index=False)
        print(f"[OK] {MODELS[model]['label']}: {param_out.name}")

        pinned = param_df[param_df["pinned_at_boundary"] != ""]
        for _, row in pinned.iterrows():
            side = row["pinned_at_boundary"]
            ratio = row["edge_mass_ratio_lower"] if side == "LOWER" else row["edge_mass_ratio_upper"]
            extra = ""
            if model == "all_climate" and row["param"] == "gamma":
                extra = "  <-- EXPECTED: known gamma-betaH0 degeneracy."
            if row["param"] == "psi":
                extra = "  <-- check whether psi<=1 is a genuine constraint before widening."
            print(f"    pinned: {row['param']} has {ratio:.1f}x the flat-posterior mass near "
                  f"the {side.lower()} wall (prior [{row['prior_lo']:.4g}, "
                  f"{row['prior_hi']:.4g}]){extra}")

    comparison = (pd.DataFrame(summaries)
                  .sort_values("logZ", ascending=False)
                  .reset_index(drop=True))

    # Sanity check against the verified value
    ac = comparison[comparison["betaH_mode"] == "all_climate"]
    if not ac.empty:
        actual = ac["logZ"].iloc[0]
        if abs(actual - EXPECTED_ALL_CLIMATE_LOGZ) > 1.0:
            print(f"\n*** WARNING: all_climate logZ = {actual:.3f}, expected about "
                  f"{EXPECTED_ALL_CLIMATE_LOGZ:.3f}. Check the paths in MODELS. ***")
        else:
            print(f"\n[OK] all_climate logZ = {actual:.3f} matches the verified value.")

    best = comparison["logZ"].iloc[0]
    comparison["delta_logZ"] = best - comparison["logZ"]
    comparison["bayes_factor_vs_best"] = np.exp(-comparison["delta_logZ"])
    comparison["delta_AICc"] = comparison["AICc"] - comparison["AICc"].min()
    comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()

    # Do AICc and BIC agree with the evidence ranking?
    evidence_order = list(comparison["betaH_mode"])
    aicc_order = list(comparison.sort_values("AICc")["betaH_mode"])
    bic_order = list(comparison.sort_values("BIC")["betaH_mode"])
    agree_best = aicc_order[0] == bic_order[0] == evidence_order[0]
    agree_full = aicc_order == evidence_order and bic_order == evidence_order

    comparison.to_csv(COMPARISON_CSV, index=False)
    write_latex_rows(comparison, COMPARISON_TEX)

    cols = ["label", "k", "max_loglik", "logZ", "logZ_err", "delta_logZ",
            "AICc", "delta_AICc", "BIC", "delta_BIC", "n_params_pinned"]
    print("\n=== Model comparison, all formulations (ordered by log Z) ===")
    print(comparison[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSame best model under log Z, AICc and BIC: {agree_best}")
    print(f"Same full ranking under log Z, AICc and BIC: {agree_full}")
    if not agree_full:
        print(f"  log Z order: {evidence_order}")
        print(f"  AICc order:  {aicc_order}")
        print(f"  BIC order:   {bic_order}")

    print(f"\n[OK] Saved: {COMPARISON_CSV}")
    print(f"[OK] Saved: {COMPARISON_TEX}  (rows for Table 1)")


if __name__ == "__main__":
    main()