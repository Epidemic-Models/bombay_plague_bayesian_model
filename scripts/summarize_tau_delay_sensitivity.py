# =============================================================================
# summarize_tau_delay_sensitivity.py
# =============================================================================
"""
Reads the posteriors written by run_tau_delay_sensitivity_all_climate.py and
answers the question the evidence table alone cannot: do the scientific
conclusions change with tau and the delay kernel?

For every scenario it reports, from the equal-weight posterior:
  * median and 95% CrI of every free parameter;
  * P(coefficient in the headline direction) for the climate coefficients
    (headline signs: aT<0, aTT<0, aR<0, be_H<0, bR_H>0; ae, bT_H unresolved);
  * derived quantities: baseline R0_H = betaH0 / gamma and the spillover
    temperature optimum z* = -aT / (2 aTT) (in SD units);
  * the discrete death-delay kernel actually used (first 4 weights and its
    mean lag), so degenerate kernels are visible.

Nothing is refitted.

Outputs (results/tau_delay_sensitivity_all_climate/):
    tau_delay_parameter_summary.csv   (long format, one row per scenario x parameter)
    tau_delay_sign_stability.csv      (one row per scenario, headline checks)

Usage
-----
    python scripts/summarize_tau_delay_sensitivity.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dynesty.utils import resample_equal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "tau_delay_sensitivity_all_climate"
NESTED_DIR = RESULTS_DIR / "nested_sampling"
COMPARISON_CSV = RESULTS_DIR / "tau_delay_sensitivity_model_comparison.csv"

SCENARIOS = [
    "tau7d_gamma", "tau9d_gamma", "tau4d_gamma", "tau2d_gamma",
    "tau4d_gamma_shifted", "tau2d_gamma_shifted", "tau7d_gamma_shifted",
]

# Headline direction for each climate coefficient (None = not resolved in headline).
HEADLINE_SIGN = {
    "aT": -1, "aTT": -1, "aR": -1, "ae": None,
    "bT_H": None, "be_H": -1, "bR_H": +1,
}

RANDOM_SEED = 12345


def load_posterior(scenario: str) -> tuple[np.ndarray, list[str]]:
    pkl = NESTED_DIR / f"all_climate__{scenario}.pkl"
    with open(pkl, "rb") as f:
        saved = pickle.load(f)

    results = saved["results"] if isinstance(saved, dict) and "results" in saved else saved
    if isinstance(saved, dict) and "param_names" in saved:
        names = list(saved["param_names"])
    else:
        meta = json.loads((NESTED_DIR / f"all_climate__{scenario}.json").read_text())
        names = list(meta["param_names"])

    weights = np.exp(results["logwt"] - results["logz"][-1])
    weights /= weights.sum()
    rng = np.random.default_rng(RANDOM_SEED)
    try:
        post = resample_equal(results["samples"], weights, rstate=rng)
    except TypeError:                      # older dynesty signature
        post = resample_equal(results["samples"], weights, rng)
    return post, names


def kernel_summary(row: pd.Series) -> dict:
    """Rebuild the delay kernel for one scenario, if observation.py supports it."""
    try:
        from observation import make_delay_pmf_weeks
        kwargs = dict(kind=row["delay_kind"], max_weeks=int(row["delay_max_weeks"]),
                      shape=float(row["delay_shape"]), scale=float(row["delay_scale"]))
        try:
            w = make_delay_pmf_weeks(shift_weeks=float(row["delay_shift_weeks"]), **kwargs)
        except TypeError:
            w = make_delay_pmf_weeks(**kwargs)
        w = np.asarray(w, dtype=float)
        lags = np.arange(len(w))
        return {
            "kernel_w0": w[0], "kernel_w1": w[1], "kernel_w2": w[2], "kernel_w3": w[3],
            "kernel_mean_lag_weeks": float((lags * w).sum()),
        }
    except Exception as exc:               # keep the summary even if this fails
        print(f"  [note] kernel not rebuilt for {row['scenario']}: {exc}")
        return {}


def main() -> None:
    comp = pd.read_csv(COMPARISON_CSV).set_index("scenario")

    long_rows, sign_rows = [], []
    for sc in SCENARIOS:
        if not (NESTED_DIR / f"all_climate__{sc}.pkl").exists():
            print(f"[skip] {sc}: no posterior file")
            continue
        post, names = load_posterior(sc)
        P = pd.DataFrame(post, columns=names)

        P["R0H_baseline"] = P["betaH0"] / P["gamma"]
        P["zT_opt_SD"] = -P["aT"] / (2.0 * P["aTT"])

        for p in list(names) + ["R0H_baseline", "zT_opt_SD"]:
            x = P[p].to_numpy()
            long_rows.append({
                "scenario": sc, "parameter": p,
                "median": np.median(x),
                "q025": np.quantile(x, 0.025), "q975": np.quantile(x, 0.975),
                "P_gt0": float(np.mean(x > 0)),
            })

        srow = {"scenario": sc,
                "tau0_days": comp.loc[sc, "tau0_days"],
                "delay_kind": comp.loc[sc, "delay_kind"],
                "logZ": comp.loc[sc, "logZ"],
                "delta_logZ": comp.loc[sc, "delta_logZ"],
                "max_loglik": comp.loc[sc, "max_loglik"]}
        all_ok = True
        for p, sign in HEADLINE_SIGN.items():
            p_dir = float(np.mean(P[p] < 0)) if sign == -1 else float(np.mean(P[p] > 0))
            if sign is None:
                lo, hi = np.quantile(P[p], [0.025, 0.975])
                srow[f"{p}_resolved"] = bool(lo > 0 or hi < 0)
            else:
                srow[f"P_{p}_headline_dir"] = round(p_dir, 3)
                all_ok &= p_dir >= 0.975
        srow["all_headline_signs_resolved"] = all_ok
        srow["R0H_baseline_median"] = float(np.median(P["R0H_baseline"]))
        srow["zT_opt_SD_median"] = float(np.median(P["zT_opt_SD"]))
        srow.update(kernel_summary(comp.loc[sc].rename(sc).to_frame().T.reset_index()
                                   .rename(columns={"index": "scenario"}).iloc[0]))
        sign_rows.append(srow)

    long_df = pd.DataFrame(long_rows)
    sign_df = pd.DataFrame(sign_rows)

    long_path = RESULTS_DIR / "tau_delay_parameter_summary.csv"
    sign_path = RESULTS_DIR / "tau_delay_sign_stability.csv"
    long_df.to_csv(long_path, index=False)
    sign_df.to_csv(sign_path, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== Headline-sign stability across scenarios ===")
    print(sign_df.to_string(index=False))

    key = ["aR", "bR_H", "be_H", "aT", "aTT", "gamma", "betaH0", "R0H_baseline", "zT_opt_SD"]
    wide = (long_df[long_df["parameter"].isin(key)]
            .assign(cell=lambda d: d.apply(
                lambda r: f"{r['median']:.3f} ({r['q025']:.3f}, {r['q975']:.3f})", axis=1))
            .pivot(index="parameter", columns="scenario", values="cell")
            .reindex(key)[[s for s in SCENARIOS if s in set(long_df["scenario"])]])
    print("\n=== Key parameters: median (95% CrI) ===")
    print(wide.to_string())

    print(f"\n[OK] Saved {long_path}\n[OK] Saved {sign_path}")


if __name__ == "__main__":
    main()