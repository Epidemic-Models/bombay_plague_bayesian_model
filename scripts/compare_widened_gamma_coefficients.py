# =============================================================================
# compare_widened_gamma_coefficients.py
# =============================================================================
"""
Does widening the prior bound on gamma (0.5-3.0 -> 0.5-6.0 per week) change
the climate effects of the main model?

Reads two existing nested-sampling results (nothing is refitted):
    MAIN_PKL     the main model (all_climate, final_v2)
    WIDE_PKL     the same model refitted with gamma's upper bound at 6.0

and prints, for every parameter, the posterior median and 95% credible
interval in both runs, and whether the interval excludes zero.

Usage
-----
    python scripts/compare_widened_gamma_coefficients.py
    python scripts/compare_widened_gamma_coefficients.py path/to/widened_gamma.pkl

If no path is given, the script searches results/ for a .pkl whose name
contains "gamma" and lists what it finds; set WIDE_PKL below if needed.

Output
------
results/final_figures/widened_gamma_comparison.csv
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
FIT_TARGET = "plague_deaths_reported_fit"

MAIN_PKL = (RESULTS / "final_model_comparison_v2" / "nested_sampling"
            / f"all_climate__{FIT_TARGET}__final_v2.pkl")

# *** Set this to your widened-gamma result if the search below does not find it.
WIDE_PKL: Path | None = (RESULTS / "gamma_prior_sensitivity" / "nested_sampling"
                         / f"all_climate__{FIT_TARGET}__gamma_upper6_rslice_seeded.pkl")

CLIMATE_PARAMS = ["aT", "aTT", "ae", "aR", "bT_H", "be_H", "bR_H"]
OUT_CSV = RESULTS / "final_figures" / "widened_gamma_comparison.csv"


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def summarise(saved: dict) -> dict:
    res = saved["results"]
    names = list(saved["param_names"])
    w = np.exp(res.logwt - res.logz[-1])
    w = w / w.sum()
    out = {}
    for j, name in enumerate(names):
        v = res.samples[:, j]
        order = np.argsort(v)
        cw = np.cumsum(w[order])
        lo, md, hi = np.interp([0.025, 0.5, 0.975], cw, v[order])
        out[name] = (md, lo, hi, float(w[v > 0].sum()))
    out["_logZ"] = float(res.logz[-1])
    out["_bounds"] = dict(zip(names, saved.get("bounds", [None] * len(names))))
    return out


def find_wide() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if WIDE_PKL is not None:
        return Path(WIDE_PKL)
    hits = sorted(p for p in RESULTS.rglob("*.pkl") if "gamma" in p.name.lower())
    if len(hits) == 1:
        return hits[0]
    print("Could not pick the widened-gamma result automatically. Candidates:")
    for p in hits:
        print("  ", p.relative_to(PROJECT_ROOT))
    raise SystemExit("Pass the path as an argument or set WIDE_PKL in the script.")


def main():
    wide_path = find_wide()
    print(f"Main model   : {MAIN_PKL.relative_to(PROJECT_ROOT)}")
    print(f"Widened gamma: {wide_path}")

    main_s = summarise(load(MAIN_PKL))
    wide_s = summarise(load(wide_path))
    print(f"gamma bound in widened run: {wide_s['_bounds'].get('gamma')}")
    print(f"log Z: main {main_s['_logZ']:.2f}, widened {wide_s['_logZ']:.2f}\n")

    rows = []
    params = [p for p in main_s if not p.startswith("_")]
    for p in params:
        if p not in wide_s:
            continue
        m, w = main_s[p], wide_s[p]
        res_m = m[1] > 0 or m[2] < 0
        res_w = w[1] > 0 or w[2] < 0
        rows.append({
            "parameter": p,
            "main_median": m[0], "main_lo": m[1], "main_hi": m[2],
            "main_resolved": res_m,
            "wide_median": w[0], "wide_lo": w[1], "wide_hi": w[2],
            "wide_resolved": res_w,
            "same_sign": np.sign(m[0]) == np.sign(w[0]),
            "same_resolution": res_m == res_w,
        })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    def cell(md, lo, hi, r):
        return f"{md:+.3f} ({lo:+.3f}, {hi:+.3f}){' *' if r else '  '}"

    print(f"{'Parameter':<10}{'Main model (gamma <= 3)':<34}{'Widened (gamma <= 6)':<34}Change")
    print("-" * 92)
    for _, r in df.iterrows():
        flag = ""
        if r["parameter"] in CLIMATE_PARAMS:
            if not r["same_sign"]:
                flag = "SIGN CHANGED"
            elif not r["same_resolution"]:
                flag = "RESOLUTION CHANGED"
            else:
                flag = "same"
        print(f"{r['parameter']:<10}"
              f"{cell(r['main_median'], r['main_lo'], r['main_hi'], r['main_resolved']):<34}"
              f"{cell(r['wide_median'], r['wide_lo'], r['wide_hi'], r['wide_resolved']):<34}"
              f"{flag}")
    print("\n* = 95% credible interval excludes zero")

    clim = df[df["parameter"].isin(CLIMATE_PARAMS)]
    if clim["same_sign"].all() and clim["same_resolution"].all():
        print("\nAll climate effects kept their direction and resolution.")
    else:
        print("\nAt least one climate effect changed; see the flags above.")
    print(f"[OK] Saved {OUT_CSV}")


if __name__ == "__main__":
    main()