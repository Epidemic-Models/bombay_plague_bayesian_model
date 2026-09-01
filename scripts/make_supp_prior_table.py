# =============================================================================
# make_supp_prior_table.py
# =============================================================================
"""
Generates the supplementary "prior ranges" table by reading directly from
src/priors.py -- the single declared source of truth for what bounds this
project uses -- rather than from the .pkl files.

*** Cross-checks against the bounds actually saved inside the final_v2
.pkl files as a SAFETY NET, not the primary source: if priors.py and the
saved results ever disagree, this script WARNS loudly rather than silently
using whichever one happens to be read. This catches the failure mode
where someone edits priors.py later without re-running the fit (or vice
versa) -- the exact kind of staleness that caused the final_climate_effects.py
/ posterior_analysis.py path bugs earlier in this project. ***

Also prints an explicit diff against the ORIGINAL LHS-screening bounds
(priors.ORIGINAL_LHS_BOUNDS), so the paper can state precisely which
bounds changed and by how much -- concrete evidence for the "old bounds
excluded the true optimum" argument in the Discussion.

Outputs
-------
results/final_figures/
    supp_prior_ranges.csv
    supp_prior_ranges.tex

Usage
-----
    python scripts/make_supp_prior_table.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import priors  # noqa: E402 -- the primary source of truth


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
RESULT_DIR = PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"

ALL_MODELS = ["constant", "humidity_only", "all_climate", "sigmoid"]

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_META = [
    ("N0",       "individuals",      "Initial population size",                         "Initial conditions"),
    ("C_seed",   "infections/week",  "Initial incidence history before $t=0$",           "Initial conditions"),
    ("gamma",    "week$^{-1}$",      "Recovery rate (inverse infectious period)",        "Epidemic dynamics"),
    ("a0",       "log-scale",        "Baseline spillover intensity",                     r"Spillover transmission ($\beta_R$)"),
    ("aT",       "--",               "Temperature effect",                               r"Spillover transmission ($\beta_R$)"),
    ("aTT",      "--",               "Temperature quadratic effect",                     r"Spillover transmission ($\beta_R$)"),
    ("ae",       "--",               "Vapour pressure (humidity) effect",                r"Spillover transmission ($\beta_R$)"),
    ("aR",       "--",               "Rainfall effect",                                  r"Spillover transmission ($\beta_R$)"),
    ("betaH0",   "week$^{-1}$",      "Baseline human transmission rate",                 r"Human transmission ($\beta_H$)"),
    ("bT_H",     "--",               "Temperature effect on human transmission",         r"Human transmission ($\beta_H$)"),
    ("be_H",     "--",               "Humidity effect on human transmission",            r"Human transmission ($\beta_H$)"),
    ("bR_H",     "--",               "Rainfall effect on human transmission",            r"Human transmission ($\beta_H$)"),
    ("theta",    "--",               "Magnitude/direction of temporal change (sigmoid)", r"Human transmission ($\beta_H$, sigmoid)"),
    ("t_star",   "weeks",            "Transition time (sigmoid)",                        r"Human transmission ($\beta_H$, sigmoid)"),
    ("kappa",    "week$^{-1}$",      "Steepness of logistic transition (sigmoid)",       r"Human transmission ($\beta_H$, sigmoid)"),
    ("psi",      "probability",      "Infection-to-death scaling parameter",             "Observation model"),
    ("nb_kappa", "--",               "Negative binomial dispersion parameter",           "Observation model"),
]

# Proper LaTeX math symbols per parameter -- matches the notation used in
# make_supp_best_fit_table.py and make_supp_model_summary_table.py, so all
# three supplementary tables use identical symbols for the same parameter.
LATEX_SYMBOLS = {
    "N0": r"$N_0$",
    "C_seed": r"$C_{\mathrm{seed}}$",
    "gamma": r"$\gamma$",
    "a0": r"$a_0$",
    "aT": r"$a_T$",
    "aTT": r"$a_{TT}$",
    "ae": r"$a_e$",
    "aR": r"$a_R$",
    "betaH0": r"$\beta_{H0}$",
    "bT_H": r"$b_T^{(H)}$",
    "be_H": r"$b_e^{(H)}$",
    "bR_H": r"$b_R^{(H)}$",
    "theta": r"$\theta$",
    "t_star": r"$t^{*}$",
    "kappa": r"$\kappa_{\mathrm{sig}}$",
    "psi": r"$\psi$",
    "nb_kappa": r"$\kappa_{\mathrm{NB}}$",
}


# =============================================================================
# PRIMARY SOURCE: read bounds directly from priors.py
# =============================================================================

print("=" * 90)
print("READING BOUNDS FROM src/priors.py (primary source of truth)")
print("=" * 90)

declared_bounds = {}
for pname, _, _, _ in PARAM_META:
    if pname in priors.SHARED_BOUNDS:
        declared_bounds[pname] = priors.SHARED_BOUNDS[pname]
    elif pname in priors.EXTRA_BOUNDS:
        declared_bounds[pname] = priors.EXTRA_BOUNDS[pname]
    else:
        raise KeyError(f"{pname!r} not found in priors.py's SHARED_BOUNDS or EXTRA_BOUNDS.")

print(f"Loaded {len(declared_bounds)} declared bounds.")


# =============================================================================
# SAFETY NET: cross-check against what's actually saved in the final_v2 .pkl
# files. Disagreement here means priors.py has drifted from what was
# actually fit -- warn loudly, don't silently proceed as if nothing's wrong.
# =============================================================================

def load_saved_bounds(model: str) -> dict:
    path = RESULT_DIR / f"{model}__{FIT_TARGET}__final_v2.pkl"
    if not path.exists():
        print(f"  [skip] No final_v2 result for {model!r} at {path} -- cannot cross-check this model.")
        return {}
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return dict(zip(saved["param_names"], saved["bounds"]))


print("\n" + "=" * 90)
print("SAFETY-NET CROSS-CHECK: priors.py vs. what was actually saved in final_v2 results")
print("=" * 90)

drift_warnings = []
for model in ALL_MODELS:
    saved_bounds = load_saved_bounds(model)
    for pname, saved_bound in saved_bounds.items():
        declared = declared_bounds.get(pname)
        if declared is not None and tuple(declared) != tuple(saved_bound):
            drift_warnings.append((model, pname, declared, saved_bound))

if drift_warnings:
    print("\n*** WARNING: priors.py DISAGREES with what was actually used for fitting: ***")
    for model, pname, declared, saved_bound in drift_warnings:
        print(f"  {model}.{pname}: priors.py says {declared}, but final_v2 .pkl says {saved_bound}")
    print(
        "\nThis means priors.py has drifted from the bounds that actually produced your "
        "reported results (or the .pkl files predate the latest priors.py edit). The table "
        "below is built from priors.py (the DECLARED source of truth) -- resolve this "
        "discrepancy before trusting either the table or the results as mutually consistent."
    )
else:
    print("[OK] priors.py matches the bounds actually saved in every checked final_v2 result. "
          "No drift detected.")


# =============================================================================
# Build table + diff against original LHS bounds
# =============================================================================

rows = []
print("\n" + "=" * 90)
print("PRIOR BOUNDS: current (priors.py) vs. original LHS screening")
print("=" * 90)

for pname, units, desc, section in PARAM_META:
    new_lo, new_hi = declared_bounds[pname]
    old = priors.ORIGINAL_LHS_BOUNDS.get(pname)

    changed = ""
    if old is not None and tuple(old) != (new_lo, new_hi):
        changed = "CHANGED"

    if old:
        print(f"  {pname:10s}  old=({old[0]:>10g}, {old[1]:>10g})  "
              f"new=({new_lo:>10g}, {new_hi:>10g})  {changed}")
    else:
        print(f"  {pname:10s}  new=({new_lo:>10g}, {new_hi:>10g})  (no old bound on record)")

    rows.append({
        "parameter": pname, "section": section, "units": units, "description": desc,
        "old_lhs_lower": old[0] if old else None, "old_lhs_upper": old[1] if old else None,
        "declared_lower": new_lo, "declared_upper": new_hi,
        "bound_changed": bool(changed),
    })

table_df = pd.DataFrame(rows)
csv_path = OUTPUT_DIR / "supp_prior_ranges.csv"
table_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")


# =============================================================================
# LaTeX table
# =============================================================================

lines = []
lines.append(r"\begin{table}[!htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Model parameters and prior ranges used in the final Bayesian nested-sampling "
             r"analysis (common priors across all four $\beta_H$ formulations, as declared in "
             r"\texttt{src/priors.py}). Time-related quantities are expressed in weeks unless "
             r"otherwise noted.}")
lines.append(r"\label{tab:supp_parameter_ranges}")
lines.append(r"\begin{tabularx}{\textwidth}{l l l X}")
lines.append(r"\hline")
lines.append(r"Parameter & Range & Units & Description \\")
lines.append(r"\hline")

current_section = None
for _, row in table_df.iterrows():
    if row["section"] != current_section:
        lines.append(r"\multicolumn{4}{l}{\textbf{" + row["section"] + r"}} \\")
        current_section = row["section"]

    symbol = LATEX_SYMBOLS.get(row["parameter"], f"${row['parameter']}$")
    lines.append(
        f"{symbol} & "
        f"$[{row['declared_lower']:g},\\,{row['declared_upper']:g}]$ & "
        f"{row['units']} & {row['description']} \\\\"
    )

lines.append(r"\hline")
lines.append(r"\end{tabularx}")
lines.append(r"\end{table}")

latex_table = "\n".join(lines)
tex_path = OUTPUT_DIR / "supp_prior_ranges.tex"
with open(tex_path, "w") as f:
    f.write(latex_table)

print(f"[OK] Saved: {tex_path}")
print("\n" + "=" * 90)
print("LATEX TABLE PREVIEW")
print("=" * 90)
print(latex_table)

n_changed = table_df["bound_changed"].sum()
print(f"\n{n_changed} of {len(table_df)} parameter bounds changed relative to the original LHS screening.")
if drift_warnings:
    print(f"\n*** {len(drift_warnings)} DRIFT WARNING(S) ABOVE -- resolve before treating this table as final. ***")
print("Done.")