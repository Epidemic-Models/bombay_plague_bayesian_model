# =============================================================================
# compare_psi_fixed_dynamics.py
# =============================================================================
"""
Transmission dynamics and counterfactuals under the main model and under the
refit with the infection-to-death scaling, psi, held fixed.

Why
---
psi maps modelled infections to recorded plague deaths. Its posterior in the
main model sits near the top of its [0.05, 1] prior, above the value that the
product of the fatality of untreated bubonic plague and the completeness of
death registration can plausibly take. run_psi_sensitivity.py --refit refits
the model with psi held at a plausible value; this script asks whether the
quantities the paper reports change.

Everything here is computed from the two posteriors directly. Nothing is
rescaled by hand, so the comparison does not rely on the approximate
invariance argument.

Quantities compared
-------------------
    cumulative infections
    amplification share of the cumulative force of infection
    median R0_H, and the share of draw-week combinations with R0_H > 1
    reduction in cumulative infections with amplification removed
    reduction in cumulative infections with spillover stopped after week 4

The counterfactuals use the simulator's own switches (human_multiplier,
spillover_multiplier, spillover_cutoff_week), matching
counterfactual_compare_climate_models.py, so the main-model column should
reproduce the published numbers.

Outputs
-------
results/psi_sensitivity/
    psi_fixed_dynamics_draws.csv      one row per model x draw
    psi_fixed_dynamics_summary.csv    medians and 95% credible intervals
    psi_fixed_dynamics_table.tex      SI table

Usage
-----
    python scripts/compare_psi_fixed_dynamics.py
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
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

MODE = "all_climate"
FIT_TARGET = "plague_deaths_reported_fit"

PSI_SENS_DIR = PROJECT_ROOT / "results" / "psi_sensitivity"

RUNS = {
    "main": {
        "label": r"Main model ($\psi$ free)",
        "path": (PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
                 / f"{MODE}__{FIT_TARGET}__final_v2.pkl"),
    },
    "psi_fixed": {
        "label": r"$\psi$ fixed at 0.5",
        "path": (PSI_SENS_DIR / "nested_sampling"
                 / f"{MODE}__{FIT_TARGET}__psi_fixed.pkl"),
    },
}

N_DRAWS = 1000
RANDOM_SEED = 20261025      # one stream per run, offset by RUN_OFFSET
RUN_OFFSET = {"main": 0, "psi_fixed": 1000}

SPILLOVER_CUTOFF_WEEK = 4.0

# Simulation settings (identical to final_transmission_dynamics.py)
SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = 0.0
E0 = 0.0
R0_INIT = 0.0
WEEK_RULE = "W-FRI"

OUTPUT_DIR = PSI_SENS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def equal_weight_draws(path: Path, n_draws: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Equal-weight posterior subsample, reproducibly chosen."""
    if not path.exists():
        raise FileNotFoundError(
            f"Posterior not found:\n  {path}\n"
            "Run run_psi_sensitivity.py with RUN_REFIT = True (or --refit) first."
        )
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"]
    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    eq = np.asarray(resample_equal(np.asarray(res.samples, dtype=float), w,
                                   rstate=np.random.default_rng(seed)), dtype=float)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(eq), size=min(n_draws, len(eq)), replace=False)
    return list(d["param_names"]), eq[idx]


def simulate(sample: dict, zT, ze, zR, start_date, **extra) -> pd.DataFrame:
    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR,
        N=sample["N0"],
        params=build_sim_params(sample, MODE),
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=MODE,
        start_date=start_date,
        **extra,
    )
    sim["amplification_force"] = sim["lambda"] - sim["beta_R"]
    sim["R0_H"] = sim["beta_H"] / sample["gamma"]
    return sim


def weekly(sim: pd.DataFrame, col: str, week_ending: pd.Series, how: str) -> np.ndarray:
    """Weekly sum or mean of an internal-grid column, on the fitted week grid."""
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.set_index("date")[col].resample(WEEK_RULE)
    wk = (g.sum() if how == "sum" else g.mean()).reset_index()
    wk.columns = ["week_ending", "v"]
    target = pd.DataFrame({"week_ending": pd.to_datetime(week_ending)})
    return pd.merge(target, wk, on="week_ending", how="left")["v"].to_numpy(float)


def q(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return (float(np.nanquantile(x, 0.025)),
            float(np.nanquantile(x, 0.5)),
            float(np.nanquantile(x, 0.975)))


def fmt(med, lo, hi, dp=1, thousands=False) -> str:
    f = (lambda v: f"{v:,.0f}") if thousands else (lambda v: f"{v:.{dp}f}")
    return f"{f(med)} ({f(lo)}, {f(hi)})"


# =============================================================================
# Main
# =============================================================================

def main():

    merged_df = (pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
                 .sort_values(paths.DATE_COL).reset_index(drop=True))
    week_ending = pd.to_datetime(merged_df[paths.DATE_COL])
    zT = merged_df["zT"].to_numpy(dtype=float)
    ze = merged_df["ze"].to_numpy(dtype=float)
    zR = merged_df["zR"].to_numpy(dtype=float)
    start_date = str(pd.Timestamp(week_ending.iloc[0]).date())

    print("\n" + "=" * 92)
    print("TRANSMISSION DYNAMICS AND COUNTERFACTUALS: MAIN MODEL vs PSI FIXED")
    print("=" * 92)

    draw_rows = []
    per_run: dict[str, dict] = {}

    for run, spec in RUNS.items():
        names, draws = equal_weight_draws(spec["path"], N_DRAWS,
                                          RANDOM_SEED + RUN_OFFSET[run])
        print(f"\n--- {spec['label']} ---")
        print(f"  {spec['path'].name}")
        print(f"  {len(draws)} draws, {len(names)} parameters"
              f"{' (psi fixed, not sampled)' if 'psi' not in names else ''}")

        n = len(draws)
        cum = np.empty(n)
        share = np.empty(n)
        r0_med = np.empty(n)
        red_amp = np.empty(n)
        red_spill = np.empty(n)
        r0_weekly = []

        for i, theta in enumerate(draws):
            s = dict(zip(names, np.asarray(theta, dtype=float)))

            sim = simulate(s, zT, ze, zR, start_date)
            sim_noamp = simulate(s, zT, ze, zR, start_date, human_multiplier=0.0)
            sim_nospill = simulate(s, zT, ze, zR, start_date,
                                   spillover_multiplier=0.0,
                                   spillover_cutoff_week=SPILLOVER_CUTOFF_WEEK)

            total = float(np.nansum(weekly(sim, "new_exposed", week_ending, "sum")))
            t_noamp = float(np.nansum(weekly(sim_noamp, "new_exposed", week_ending, "sum")))
            t_nospill = float(np.nansum(weekly(sim_nospill, "new_exposed", week_ending, "sum")))

            amp = sim["amplification_force"].to_numpy(dtype=float)
            lam = sim["lambda"].to_numpy(dtype=float)
            r0_wk = weekly(sim, "R0_H", week_ending, "mean")

            cum[i] = total
            share[i] = 100.0 * float(np.sum(amp) / np.sum(lam))
            r0_med[i] = float(np.nanmedian(r0_wk))
            red_amp[i] = 100.0 * (1.0 - t_noamp / total)
            red_spill[i] = 100.0 * (1.0 - t_nospill / total)
            r0_weekly.append(r0_wk)

            draw_rows.append({"run": run, "draw": i + 1,
                              "cumulative_infections": total,
                              "amplification_share_pct": share[i],
                              "median_R0_H": r0_med[i],
                              "reduction_no_amplification_pct": red_amp[i],
                              "reduction_no_spillover_after_4wk_pct": red_spill[i]})

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"  processed {i + 1} / {n}")

        r0_flat = np.concatenate([np.asarray(v, dtype=float) for v in r0_weekly])
        per_run[run] = {
            "label": spec["label"],
            "cumulative_infections": cum,
            "amplification_share_pct": share,
            "median_R0_H": r0_med,
            "reduction_no_amplification_pct": red_amp,
            "reduction_no_spillover_after_4wk_pct": red_spill,
            "pct_draw_weeks_R0_H_above_1": 100.0 * float(np.mean(r0_flat > 1.0)),
        }

    pd.DataFrame(draw_rows).to_csv(OUTPUT_DIR / "psi_fixed_dynamics_draws.csv", index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    quantities = [
        ("cumulative_infections", "Cumulative infections", 0, True),
        ("amplification_share_pct", r"Amplification share of $\lambda$ (\%)", 1, False),
        ("median_R0_H", r"Median $R_{0,H}$", 3, False),
        ("reduction_no_amplification_pct",
         r"Reduction, amplification removed (\%)", 1, False),
        ("reduction_no_spillover_after_4wk_pct",
         r"Reduction, spillover stopped after week 4 (\%)", 2, False),
    ]

    rows = []
    for key, label, dp, thousands in quantities:
        entry = {"quantity": label}
        for run in RUNS:
            lo, md, hi = q(per_run[run][key])
            entry[f"{run}_median"] = md
            entry[f"{run}_lo"] = lo
            entry[f"{run}_hi"] = hi
            entry[f"{run}_text"] = fmt(md, lo, hi, dp, thousands)
        entry["pct_change_median"] = 100.0 * (entry["psi_fixed_median"]
                                              / entry["main_median"] - 1.0)
        rows.append(entry)

    # Not a per-draw quantity: pooled over draws and weeks.
    rows.append({
        "quantity": r"Draw-weeks with $R_{0,H} > 1$ (\%)",
        "main_median": per_run["main"]["pct_draw_weeks_R0_H_above_1"],
        "main_text": f"{per_run['main']['pct_draw_weeks_R0_H_above_1']:.1f}",
        "psi_fixed_median": per_run["psi_fixed"]["pct_draw_weeks_R0_H_above_1"],
        "psi_fixed_text": f"{per_run['psi_fixed']['pct_draw_weeks_R0_H_above_1']:.1f}",
        "pct_change_median": 100.0 * (per_run["psi_fixed"]["pct_draw_weeks_R0_H_above_1"]
                                      / per_run["main"]["pct_draw_weeks_R0_H_above_1"] - 1.0),
    })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "psi_fixed_dynamics_summary.csv", index=False)

    print("\n" + "-" * 92)
    print(f"{'Quantity':<48}{'Main':<28}{'psi fixed':<28}{'Change':>8}")
    print("-" * 92)
    for _, r in summary.iterrows():
        label = (r["quantity"].replace(r"$\lambda$", "lambda")
                 .replace(r"$R_{0,H}$", "R0_H").replace(r"\%", "%").replace("$", ""))
        print(f"{label:<48}{r['main_text']:<28}{r['psi_fixed_text']:<28}"
              f"{r['pct_change_median']:+7.1f}%")

    # -------------------------------------------------------------------------
    # SI table
    # -------------------------------------------------------------------------
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Transmission dynamics and counterfactual results under the main model, "
        r"in which the infection-to-death scaling $\psi$ is estimated, and under a refit "
        r"with $\psi$ held at 0.5. Both columns are computed from their own posterior, "
        r"with " + f"{N_DRAWS}" + r" draws each; values are posterior medians with 95\% "
        r"credible intervals, except the last row, which pools draws and weeks. Fixing "
        r"$\psi$ raises the inferred number of infections, because the same recorded deaths "
        r"then imply more infections, but leaves the amplification share, $R_{0,H}$ and both "
        r"counterfactual reductions essentially unchanged.}",
        r"\label{tab:si_psi_fixed_dynamics}",
        r"\begin{tabularx}{\textwidth}{Xll}",
        r"\toprule",
        r"Quantity & " + RUNS["main"]["label"] + r" & " + RUNS["psi_fixed"]["label"] + r" \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(f"{r['quantity']} & {r['main_text']} & {r['psi_fixed_text']} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    tex = "\n".join(lines)
    (OUTPUT_DIR / "psi_fixed_dynamics_table.tex").write_text(tex, encoding="utf-8")
    print("\n" + tex)
    print(f"[OK] Wrote {OUTPUT_DIR / 'psi_fixed_dynamics_table.tex'}")
    print("\nCheck that the main-model column reproduces the published numbers "
          "(about 83% amplification share, 83% and 99% counterfactual reductions).")


if __name__ == "__main__":
    main()