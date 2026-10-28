# =============================================================================
# counterfactual_compare_climate_models.py
# =============================================================================
"""
Do the counterfactual results depend on how temperature enters amplification?

Runs the counterfactual analyses of
    counterfactual_infections_all_climate.py         (remove amplification;
                                                      remove recurrent spillover
                                                      after week 4)
    counterfactual_spillover_cutoff_sensitivity.py   (spillover removed after
                                                      0, 2, 4, 8, 12 weeks)
for the two climate formulations, with identical settings:

    all_climate        climate, linear temperature   (main model)
    all_climate_quadT  climate, temperature peak      (bTT_H * zT^2 added)

and prints the results side by side. If the two agree closely, the main
counterfactual conclusions do not depend on the choice between the two
climate formulations.

This script does NOT refit anything, and it does not overwrite the outputs of
the two original scripts. For all_climate it uses the same random seeds as the
original scripts, so its numbers reproduce the published ones; that is a
built-in check that the comparison is like-for-like.

Requires
--------
src/priors.py and src/mechanistic_model.py with the "all_climate_quadT"
formulation, and the finished run from run_all_climate_quadT.py.

Usage
-----
    python scripts/counterfactual_compare_climate_models.py

Output
------
results/final_figures/transmission_dynamics/counterfactual_climate_model_comparison/
    counterfactual_comparison_summary.csv      (one row per model x quantity)
    counterfactual_comparison_draw_level.csv   (one row per model x draw x scenario)
    counterfactual_comparison_table.tex        (SI table)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dynesty.utils import resample_equal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
import priors  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings (identical to the two original counterfactual scripts)
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
RESULTS = PROJECT_ROOT / "results"

MODELS = {
    "all_climate": {
        "label": "Climate, linear temperature",
        "path": RESULTS / "final_model_comparison_v2" / "nested_sampling"
                / f"all_climate__{FIT_TARGET}__final_v2.pkl",
    },
    "all_climate_quadT": {
        "label": "Climate, temperature peak",
        "path": RESULTS / "all_climate_quadT_check" / "nested_sampling"
                / f"all_climate_quadT__{FIT_TARGET}__seeded.pkl",
    },
}

N_POSTERIOR_DRAWS = 1000
RANDOM_SEED_RESAMPLE = 12345
RANDOM_SEED_DRAWS = 54321

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0, E0, R0_INIT = 0.0, 0.0, 0.0
WEEK_RULE = "W-FRI"

# Main counterfactual uses week 4; the others are the cutoff sensitivity.
MAIN_CUTOFF_WEEK = 4.0
SPILLOVER_CUTOFF_WEEKS = [0.0, 2.0, 4.0, 8.0, 12.0]

OUTPUT_DIR = (RESULTS / "final_figures" / "transmission_dynamics"
              / "counterfactual_climate_model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Posterior not found:\n  {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def equal_weight(saved: dict, seed: int) -> np.ndarray:
    res = saved["results"]
    w = np.exp(res.logwt - res.logz[-1])
    return resample_equal(res.samples, w, rstate=np.random.default_rng(seed))


def weekly_new_exposed(sim: pd.DataFrame, week_ending: pd.Series) -> np.ndarray:
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    wk = (df.set_index("date")["new_exposed"].resample(WEEK_RULE).sum()
          .reset_index().rename(columns={"date": "week_ending",
                                         "new_exposed": "v"}))
    target = pd.DataFrame({"week_ending": pd.to_datetime(week_ending)})
    return pd.merge(target, wk, on="week_ending", how="left")["v"].to_numpy(float)


def q(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    return (float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.5)),
            float(np.nanquantile(x, 0.975)))


# =============================================================================
# One model
# =============================================================================

def run_model(mode: str, data: dict) -> tuple[list[dict], list[dict]]:
    spec = MODELS[mode]
    saved = load(spec["path"])
    if saved.get("betaH_mode", mode) != mode:
        raise ValueError(f"{spec['path'].name} is not a {mode!r} result.")
    param_names = list(saved["param_names"])

    posterior = equal_weight(saved, RANDOM_SEED_RESAMPLE)
    n_draws = min(N_POSTERIOR_DRAWS, len(posterior))
    idx = np.random.default_rng(RANDOM_SEED_DRAWS).choice(len(posterior), n_draws,
                                                          replace=False)
    draws = posterior[idx]

    print(f"\n--- {spec['label']} ({mode}): {n_draws} posterior draws ---")

    common = dict(zT=data["zT"], ze=data["ze"], zR=data["zR"],
                  substeps_per_week=SUBSTEPS_PER_WEEK, tau0_days=TAU0_DAYS,
                  I0=I0, E0=E0, R0=R0_INIT, betaH_mode=mode,
                  start_date=data["start_date"])

    full_tot = np.empty(n_draws)
    nohum_tot = np.empty(n_draws)
    cut_tot = {c: np.empty(n_draws) for c in SPILLOVER_CUTOFF_WEEKS}

    for i, theta in enumerate(draws):
        s = dict(zip(param_names, np.asarray(theta, float)))
        p = build_sim_params(s, mode)
        kw = dict(common, N=s["N0"], params=p, C_seed=s["C_seed"])

        full_tot[i] = weekly_new_exposed(simulate_seir_dde(**kw), data["week"]).sum()
        nohum_tot[i] = weekly_new_exposed(
            simulate_seir_dde(**kw, human_multiplier=0.0), data["week"]).sum()
        for c in SPILLOVER_CUTOFF_WEEKS:
            cut_tot[c][i] = weekly_new_exposed(
                simulate_seir_dde(**kw, spillover_multiplier=0.0,
                                  spillover_cutoff_week=c), data["week"]).sum()

        if (i + 1) % 100 == 0 or (i + 1) == n_draws:
            print(f"  processed {i + 1} / {n_draws}")

    if np.any(~np.isfinite(full_tot)) or np.any(full_tot <= 0):
        raise RuntimeError(f"{mode}: non-positive or non-finite full-model totals.")

    summary, draw_rows = [], []

    def add(quantity: str, values: np.ndarray):
        lo, md, hi = q(values)
        summary.append({"betaH_mode": mode, "label": spec["label"],
                        "quantity": quantity, "q025": lo, "median": md, "q975": hi})

    add("full_total_infections", full_tot)
    add("no_amplification_total_infections", nohum_tot)
    add("no_amplification_reduction_percent", 100 * (1 - nohum_tot / full_tot))
    for c in SPILLOVER_CUTOFF_WEEKS:
        add(f"no_spillover_after_{c:g}wk_reduction_percent",
            100 * (1 - cut_tot[c] / full_tot))

    for i in range(n_draws):
        draw_rows.append({"betaH_mode": mode, "draw": i + 1,
                          "posterior_sample_index": int(idx[i]),
                          "scenario": "no_amplification",
                          "full_total": full_tot[i], "counterfactual_total": nohum_tot[i]})
        for c in SPILLOVER_CUTOFF_WEEKS:
            draw_rows.append({"betaH_mode": mode, "draw": i + 1,
                              "posterior_sample_index": int(idx[i]),
                              "scenario": f"no_spillover_after_{c:g}wk",
                              "full_total": full_tot[i],
                              "counterfactual_total": cut_tot[c][i]})
    return summary, draw_rows


# =============================================================================
# Main
# =============================================================================

def main():
    if "all_climate_quadT" not in priors.MODEL_SPECS:
        raise SystemExit("src/priors.py does not define 'all_climate_quadT'. Use the "
                         "updated src/priors.py and src/mechanistic_model.py.")

    df = (pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
          .sort_values(paths.DATE_COL).reset_index(drop=True))
    week = pd.to_datetime(df[paths.DATE_COL])
    data = {"zT": df["zT"].to_numpy(float), "ze": df["ze"].to_numpy(float),
            "zR": df["zR"].to_numpy(float), "week": week,
            "start_date": str(pd.Timestamp(week.iloc[0]).date())}

    all_summary, all_draws = [], []
    for mode in MODELS:
        s, d = run_model(mode, data)
        all_summary += s
        all_draws += d

    summary = pd.DataFrame(all_summary)
    summary.to_csv(OUTPUT_DIR / "counterfactual_comparison_summary.csv", index=False)
    pd.DataFrame(all_draws).to_csv(OUTPUT_DIR / "counterfactual_comparison_draw_level.csv",
                                   index=False)

    # ----------------------------------------------------------- side by side
    def fmt(row, pct):
        if pct:
            return f"{row['median']:.1f}% ({row['q025']:.1f}-{row['q975']:.1f})"
        return f"{row['median']:,.0f} ({row['q025']:,.0f}-{row['q975']:,.0f})"

    quantities = summary["quantity"].unique()
    labels = [MODELS[m]["label"] for m in MODELS]
    print("\n" + "=" * 100)
    print("COUNTERFACTUAL RESULTS: LINEAR TEMPERATURE vs TEMPERATURE PEAK")
    print("=" * 100)
    print(f"{'Quantity':<44}" + "".join(f"{l:<30}" for l in labels))
    tex_rows = []
    for qty in quantities:
        pct = qty.endswith("percent")
        cells = []
        for m in MODELS:
            r = summary[(summary["betaH_mode"] == m) & (summary["quantity"] == qty)].iloc[0]
            cells.append(fmt(r, pct))
        print(f"{qty:<44}" + "".join(f"{c:<30}" for c in cells))

        nice = (qty.replace("_", " ")
                .replace("no amplification reduction percent", "Reduction without amplification")
                .replace("full total infections", "Cumulative infections, full model")
                .replace("no amplification total infections", "Cumulative infections without amplification"))
        if qty.startswith("no_spillover_after_"):
            wk = qty.split("_after_")[1].split("wk")[0]
            nice = f"Reduction without spillover after week {wk}"
        tex_cells = [c.replace("%", r"\%").replace("-", "--", 1) for c in cells]
        tex_rows.append(f"{nice} & " + " & ".join(tex_cells) + r" \\")

    tex = (r"""\begin{table}[ht]
\centering
\small
\caption{Counterfactual results under the two climate formulations of amplification. Values are posterior medians with 95\% credible intervals from 1{,}000 posterior draws; reductions are calculated within each draw relative to the full model. The main-text counterfactual removes spillover after week 4.}
\label{tab:si_counterfactual_climate_models}
\begin{tabular}{lll}
\toprule
Quantity & """ + " & ".join(labels) + r""" \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (OUTPUT_DIR / "counterfactual_comparison_table.tex").write_text(tex)
    print("\n" + tex)

    print("Check: the linear-temperature column should reproduce the published "
          "values (e.g. about 83% without amplification and 99% without "
          "spillover after week 4).")
    print(f"\n[OK] Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()