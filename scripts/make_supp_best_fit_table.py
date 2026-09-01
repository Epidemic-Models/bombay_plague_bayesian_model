# =============================================================================
# make_supp_best_fit_table.py
# =============================================================================
"""
Generates the supplementary "best-fit parameter comparison" table directly
from the final_v2 nested-sampling results -- replaces the old LHS-based
table, computing every number (parameters, SSE, logL) from the actual saved
posteriors rather than transcribing by hand.

For each of the four beta_H formulations:
    - "Best-fit" parameters = the single highest-likelihood (MLE) sampled
      theta from the final_v2 dynesty result (matches the old table's
      definition: "selected by maximum negative-binomial log-likelihood").
    - SSE = sum of squared errors between that trajectory's expected deaths
      and the observed (outlier-interpolated) reported-deaths fit target.
    - logL = the stored best_loglik for that MLE sample.

Outputs
-------
results/final_figures/
    supp_best_fit_comparison_reported.csv     (machine-readable)
    supp_best_fit_comparison_reported.tex     (LaTeX table, ready to paste)

Usage
-----
    python scripts/make_supp_best_fit_table.py
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
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde, weekly_sum  # noqa: E402
from observation import make_delay_pmf_weeks, deaths_from_infections_fast  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
MODELS = ["constant", "humidity_only", "all_climate", "sigmoid"]  # column order matches old table
MODEL_COL_LABELS = {
    "constant": r"Constant $\beta_H$",
    "humidity_only": r"Humidity-only $\beta_H$",
    "all_climate": r"Climate-driven $\beta_H$",
    "sigmoid": r"Sigmoid $\beta_H$",
}

RESULT_DIR = PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"

def result_path(model: str) -> Path:
    return RESULT_DIR / f"{model}__{FIT_TARGET}__final_v2.pkl"


SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = E0 = R0_INIT = 0.0
WEEK_RULE = "W-FRI"

DELAY_KIND = "gamma"
DELAY_TAU_WEEKS = 2
DELAY_MAX_WEEKS = 12
DELAY_SHAPE = 2.0
DELAY_SCALE = 0.6

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Row order + display names + formatting, matching the old table's structure.
# (fmt, label) -- fmt is a format spec or "int_comma" or "sci"
PARAM_ROWS = [
    ("N0", "int_comma", r"$N_0$"),
    ("C_seed", ".4f", r"$C_{\mathrm{seed}}$"),
    ("gamma", ".4f", r"$\gamma$"),
    ("a0", ".4f", r"$a_0$"),
    ("aT", ".4f", r"$a_T$"),
    ("aTT", ".4f", r"$a_{TT}$"),
    ("ae", ".4f", r"$a_e$"),
    ("aR", ".4f", r"$a_R$"),
    ("betaH0", ".4f", r"$\beta_{H0}$"),
    ("bT_H", ".4f", r"$b_T^{(H)}$"),
    ("be_H", ".4f", r"$b_e^{(H)}$"),
    ("bR_H", ".4f", r"$b_R^{(H)}$"),
    ("theta", ".4f", r"$\theta$"),
    ("t_star", ".4f", r"$t^{*}$"),
    ("kappa", ".4f", r"$\kappa_{\mathrm{sig}}$"),
    ("psi", ".4f", r"$\psi$"),
    ("nb_kappa", ".4f", r"$\kappa_{\mathrm{NB}}$"),
]


# =============================================================================
# Load prepared data
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

week_ending = merged_df[paths.DATE_COL]
zT = merged_df["zT"].to_numpy(dtype=float)
ze = merged_df["ze"].to_numpy(dtype=float)
zR = merged_df["zR"].to_numpy(dtype=float)
start_date = str(pd.Timestamp(week_ending.iloc[0]).date())
n_weeks = len(merged_df)

if FIT_TARGET not in merged_df.columns:
    raise ValueError(f"{FIT_TARGET!r} not found in prepared data.")

observed = merged_df[FIT_TARGET].to_numpy(dtype=float)

f_delay = make_delay_pmf_weeks(
    kind=DELAY_KIND, tau_weeks=DELAY_TAU_WEEKS, max_weeks=DELAY_MAX_WEEKS,
    shape=DELAY_SHAPE, scale=DELAY_SCALE,
)

print(f"Fitted weeks: {n_weeks}  ({week_ending.min().date()} to {week_ending.max().date()})")


# =============================================================================
# Helpers
# =============================================================================

def compute_mu_trajectory(theta: np.ndarray, param_names: list[str], betaH_mode: str) -> np.ndarray:
    sample = dict(zip(param_names, np.asarray(theta, dtype=float)))
    sim_params = build_sim_params(sample, betaH_mode)

    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR, N=sample["N0"], params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK, tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT, C_seed=sample["C_seed"],
        betaH_mode=betaH_mode, start_date=start_date,
    )

    weekly_inf = weekly_sum(sim, col="C", week_ending=WEEK_RULE)
    weekly_inf["week_ending"] = pd.to_datetime(weekly_inf["week_ending"])
    compare = pd.merge(merged_df[["week_ending"]], weekly_inf, on="week_ending", how="left")
    c_col = "C_week" if "C_week" in compare.columns else [
        c for c in compare.columns if c.startswith("C_")
    ][0]
    C_week = compare[c_col].fillna(0.0).to_numpy(dtype=float)

    mu = deaths_from_infections_fast(C_week=C_week, psi=sample["psi"], f_delay=f_delay)
    return np.maximum(mu, 1e-9)


def sse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((observed - predicted) ** 2))


def fmt_value(val, fmt) -> str:
    if val is None:
        return "--"
    if fmt == "int_comma":
        return f"{val:,.0f}"
    if fmt == "sci":
        return f"{val:.3e}"
    return f"{val:{fmt}}"


# =============================================================================
# Extract MLE theta + SSE + logL for each model
# =============================================================================

model_data = {}

print("\n" + "=" * 90)
print("EXTRACTING BEST-FIT (MLE) PARAMETERS FROM final_v2 RESULTS")
print("=" * 90)

for model in MODELS:
    path = result_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find final_v2 result for {model!r}:\n{path}\n"
            "Edit RESULT_DIR / result_path() if your files live elsewhere."
        )

    with open(path, "rb") as f:
        saved = pickle.load(f)

    results = saved["results"]
    param_names = list(saved["param_names"])

    logl = results.logl if hasattr(results, "logl") else results["logl"]
    samples = results.samples if hasattr(results, "samples") else results["samples"]

    best_idx = int(np.nanargmax(logl))
    theta_mle = np.asarray(samples[best_idx], dtype=float)
    max_loglik = float(logl[best_idx])

    param_dict = dict(zip(param_names, theta_mle))

    mu_mle = compute_mu_trajectory(theta_mle, param_names, model)
    mle_sse = sse(observed, mu_mle)

    print(f"\n{model}: logL={max_loglik:.3f}  SSE={mle_sse:.3e}")
    for name, val in param_dict.items():
        print(f"  {name:10s} = {val:.6g}")

    model_data[model] = {
        "params": param_dict,
        "logL": max_loglik,
        "SSE": mle_sse,
    }


# =============================================================================
# Assemble table (CSV, long format for easy checking)
# =============================================================================

rows_long = []
for model in MODELS:
    d = model_data[model]
    for pname, val in d["params"].items():
        rows_long.append({"betaH_mode": model, "parameter": pname, "value": val})
    rows_long.append({"betaH_mode": model, "parameter": "SSE_reported", "value": d["SSE"]})
    rows_long.append({"betaH_mode": model, "parameter": "logL_reported", "value": d["logL"]})

long_df = pd.DataFrame(rows_long)
csv_path = OUTPUT_DIR / "supp_best_fit_comparison_reported.csv"
long_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")


# =============================================================================
# Build LaTeX table, same structure as the original
# =============================================================================

lines = []
lines.append(r"\begin{center}")
lines.append(r"\captionof{table}{Best-fit parameter values for the alternative human-transmission "
             r"formulations, selected by maximum negative-binomial log-likelihood for reported deaths "
             r"(final Bayesian nested-sampling posteriors, betaH0 $\le 3.0$ common prior). Parameters "
             r"not active in a given formulation are indicated by --.}")
lines.append(r"\label{tab:supp_best_fit_comparison_reported}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{l" + "c" * len(MODELS) + "}")
lines.append(r"\toprule")
lines.append("Parameter & " + " & ".join(MODEL_COL_LABELS[m] for m in MODELS) + r" \\")
lines.append(r"\midrule")

for pname, fmt, label in PARAM_ROWS:
    cells = []
    for model in MODELS:
        val = model_data[model]["params"].get(pname, None)
        cells.append(fmt_value(val, fmt))
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")

lines.append(r"\midrule")
sse_cells = [fmt_value(model_data[m]["SSE"], "sci") for m in MODELS]
lines.append(r"$\mathrm{SSE}_{\mathrm{reported}}$ & " + " & ".join(sse_cells) + r" \\")

logl_cells = [fmt_value(model_data[m]["logL"], ".3f") for m in MODELS]
lines.append(r"$\log L_{\mathrm{reported}}$ & " + " & ".join(logl_cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{center}")

latex_table = "\n".join(lines)

tex_path = OUTPUT_DIR / "supp_best_fit_comparison_reported.tex"
with open(tex_path, "w") as f:
    f.write(latex_table)

print(f"[OK] Saved: {tex_path}")
print("\n" + "=" * 90)
print("LATEX TABLE PREVIEW")
print("=" * 90)
print(latex_table)

print("\nDone.")