# =============================================================================
# make_logz_repeats_table.py
# =============================================================================
"""
SI Table S16: log-evidence across repeated runs of the two climate
formulations.

Reads only the saved output of run_logz_repeats.py:
    results/logz_repeats/logz_repeats_summary.csv
(repeat 0 = the original run, repeats 1-2 = new random seeds)

Prints the LaTeX table and the numbers quoted in the Main Text (mean gap and
range over all pairings of runs), and saves the table to
    results/logz_repeats/table_si_logz_repeats.tex

Usage
-----
    python scripts/make_logz_repeats_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_CSV = PROJECT_ROOT / "results" / "logz_repeats" / "logz_repeats_summary.csv"
OUT_TEX = PROJECT_ROOT / "results" / "logz_repeats" / "table_si_logz_repeats.tex"

LINEAR, PEAK = "all_climate", "all_climate_quadT"

df = pd.read_csv(IN_CSV).drop_duplicates(subset=["mode", "repeat"], keep="last")
df = df.sort_values(["mode", "repeat"])

lin = df[df["mode"] == LINEAR].set_index("repeat")
pk = df[df["mode"] == PEAK].set_index("repeat")
repeats = sorted(set(lin.index) & set(pk.index))
if len(repeats) < 2:
    raise SystemExit(f"Need at least two runs of each formulation; found {repeats}.")

gaps = pk.loc[repeats, "logZ"].to_numpy()[:, None] - lin.loc[repeats, "logZ"].to_numpy()[None, :]


def run_label(r):
    return "Original" if r == 0 else f"Repeat {r}"


rows = []
for r in repeats:
    rows.append(
        f"{run_label(r)} & ${lin.loc[r, 'logZ']:.2f}$ & ${lin.loc[r, 'max_logL']:.2f}$ & "
        f"${pk.loc[r, 'logZ']:.2f}$ & ${pk.loc[r, 'max_logL']:.2f}$ & "
        f"{pk.loc[r, 'logZ'] - lin.loc[r, 'logZ']:.2f} \\\\"
    )
rows.append(r"\midrule")
rows.append(
    f"Mean (SD) & ${lin.loc[repeats, 'logZ'].mean():.2f}$ ({lin.loc[repeats, 'logZ'].std(ddof=1):.2f}) & "
    f"& ${pk.loc[repeats, 'logZ'].mean():.2f}$ ({pk.loc[repeats, 'logZ'].std(ddof=1):.2f}) & & "
    f"{gaps.mean():.2f} \\\\"
)

n = len(repeats)
tex = (r"""\begin{table}[!htbp]
\centering
\caption{Log-evidence, $\log Z$, and maximum log-likelihood, $\log L_{\max}$, across """ + f"{n}" +
       r""" independent nested-sampling runs of each climate formulation. Runs differ only in the random seeds for the initial live points and the sampler; all other settings are identical (Parameter inference). $\Delta\log Z$ is the log-evidence of the temperature-peak formulation minus that of the linear-temperature formulation within the same row. Across all """ + f"{n * n}" +
       r""" pairings of runs, $\Delta\log Z$ ranged from """ + f"{gaps.min():.1f} to {gaps.max():.1f}" +
       r""" (mean """ + f"{gaps.mean():.1f}" + r""").}
\label{tab:si_logz_repeats}
\begin{tabularx}{\textwidth}{X*{5}{>{\centering\arraybackslash}X}}
\toprule
 & \multicolumn{2}{c}{Climate, linear temperature} & \multicolumn{2}{c}{Climate, temperature peak} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Run & $\log Z$ & $\log L_{\max}$ & $\log Z$ & $\log L_{\max}$ & $\Delta\log Z$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

OUT_TEX.write_text(tex)
print(tex)
print(f"Gap over all {n * n} pairings: mean {gaps.mean():.2f}, "
      f"range {gaps.min():.2f} to {gaps.max():.2f}")
ll_gain = pk.loc[repeats, "max_logL"].to_numpy() - lin.loc[repeats, "max_logL"].to_numpy()
print(f"Gain in max logL (same row): {ll_gain.min():.2f} to {ll_gain.max():.2f}")
print(f"[OK] Saved {OUT_TEX}")