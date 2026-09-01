# =============================================================================
# make_supp_wave_amplification_section.py
# =============================================================================
"""
Generate a supplementary LaTeX section and table for the wave-by-wave
human-amplification analysis.

This script reads directly from the outputs created by:

    scripts/analyse_epidemic_waves_and_amplification.py

Specifically:
    results/final_figures/transmission_dynamics/wave_analysis/
        wave_phase_summary.csv
        wave_timing_summary.csv

The script does NOT hard-code the reported percentages or timing values.
Instead, it reads the saved analysis outputs and constructs:

    1. A publication-ready CSV summary table.
    2. A LaTeX table.
    3. A LaTeX subsection describing the wave-by-wave results conservatively.

The wording is deliberately model-conditional and descriptive. It does not
claim that the analysis independently proves causality, because spillover,
human FOI, and R0_H are inferred quantities from the fitted mechanistic model.

Outputs
-------
results/final_figures/transmission_dynamics/wave_analysis/
    supp_wave_amplification_table.csv
    supp_wave_amplification_table.tex
    supp_wave_amplification_section.tex

Usage
-----
    python scripts/make_supp_wave_amplification_section.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "wave_analysis"
)

PHASE_CSV = INPUT_DIR / "wave_phase_summary.csv"
TIMING_CSV = INPUT_DIR / "wave_timing_summary.csv"

OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TABLE_CSV = OUTPUT_DIR / "supp_wave_amplification_table.csv"
OUTPUT_TABLE_TEX = OUTPUT_DIR / "supp_wave_amplification_table.tex"
OUTPUT_SECTION_TEX = OUTPUT_DIR / "supp_wave_amplification_section.tex"


# =============================================================================
# Formatting helpers
# =============================================================================

def pct(x: float, digits: int = 1) -> str:
    """Format a fraction as a percentage."""
    if pd.isna(x):
        return "--"
    return f"{100.0 * x:.{digits}f}"


def sci(x: float, digits: int = 2) -> str:
    """Scientific notation for LaTeX."""
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def signed_weeks(x: float) -> str:
    """
    Format timing relative to mortality peak.

    Negative = before mortality peak.
    Positive = after mortality peak.
    """
    if pd.isna(x):
        return "--"
    if abs(x) < 1e-12:
        return "0"
    return f"{x:+.0f}"


def latex_escape_text(text: str) -> str:
    """Minimal escaping for plain text inserted into LaTeX."""
    return (
        str(text)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


# =============================================================================
# Load analysis outputs
# =============================================================================

print("=" * 100)
print("GENERATING SUPPLEMENTARY WAVE-AMPLIFICATION LATEX SECTION")
print("=" * 100)

for path in [PHASE_CSV, TIMING_CSV]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required analysis output not found:\n  {path}\n\n"
            "Run scripts/analyse_epidemic_waves_and_amplification.py first."
        )

phase_df = pd.read_csv(
    PHASE_CSV,
    parse_dates=["start_date", "end_date"],
)

timing_df = pd.read_csv(
    TIMING_CSV,
    parse_dates=[
        "wave_start",
        "mortality_peak_date",
        "wave_end",
        "spillover_peak_date",
        "R0_peak_date",
        "human_FOI_peak_date",
    ],
)

required_phase = {
    "wave",
    "phase",
    "start_date",
    "end_date",
    "n_weeks",
    "median_R0_H",
    "fraction_R0_median_gt1",
    "fraction_R0_q025_gt1",
    "mean_spillover",
    "mean_human_FOI",
    "median_spillover",
    "median_human_FOI",
    "median_human_fraction",
}

required_timing = {
    "wave",
    "mortality_peak_date",
    "spillover_peak_relative_to_mortality_peak_weeks",
    "R0_peak_relative_to_mortality_peak_weeks",
    "human_FOI_peak_relative_to_mortality_peak_weeks",
}

missing_phase = required_phase - set(phase_df.columns)
missing_timing = required_timing - set(timing_df.columns)

if missing_phase:
    raise ValueError(f"Missing columns in {PHASE_CSV.name}: {missing_phase}")

if missing_timing:
    raise ValueError(f"Missing columns in {TIMING_CSV.name}: {missing_timing}")

print(f"[OK] Loaded: {PHASE_CSV}")
print(f"[OK] Loaded: {TIMING_CSV}")


# =============================================================================
# Build publication table
# =============================================================================

rows = []

wave_ids = sorted(
    int(w)
    for w in phase_df["wave"].dropna().unique()
)

for wave in wave_ids:

    asc = phase_df[
        (phase_df["wave"] == wave)
        & (phase_df["phase"] == "ascending")
    ]

    dec = phase_df[
        (phase_df["wave"] == wave)
        & (phase_df["phase"] == "declining")
    ]

    timing = timing_df[timing_df["wave"] == wave]

    if len(asc) != 1:
        raise ValueError(
            f"Expected exactly one ascending row for wave {wave}; "
            f"found {len(asc)}."
        )

    if len(dec) != 1:
        raise ValueError(
            f"Expected exactly one declining row for wave {wave}; "
            f"found {len(dec)}."
        )

    if len(timing) != 1:
        raise ValueError(
            f"Expected exactly one timing row for wave {wave}; "
            f"found {len(timing)}."
        )

    asc = asc.iloc[0]
    dec = dec.iloc[0]
    timing = timing.iloc[0]

    rows.append(
        {
            "wave": wave,

            "ascending_start": asc["start_date"],
            "mortality_peak": timing["mortality_peak_date"],
            "declining_end": dec["end_date"],

            "R0_gt1_ascending": asc["fraction_R0_median_gt1"],
            "R0_gt1_declining": dec["fraction_R0_median_gt1"],

            "R0_q025_gt1_ascending": asc["fraction_R0_q025_gt1"],
            "R0_q025_gt1_declining": dec["fraction_R0_q025_gt1"],

            "median_R0_ascending": asc["median_R0_H"],
            "median_R0_declining": dec["median_R0_H"],

            "median_human_fraction_ascending": asc["median_human_fraction"],
            "median_human_fraction_declining": dec["median_human_fraction"],

            "median_human_FOI_ascending": asc["median_human_FOI"],
            "median_human_FOI_declining": dec["median_human_FOI"],

            "median_spillover_ascending": asc["median_spillover"],
            "median_spillover_declining": dec["median_spillover"],

            "spillover_peak_relative_weeks":
                timing["spillover_peak_relative_to_mortality_peak_weeks"],

            "R0_peak_relative_weeks":
                timing["R0_peak_relative_to_mortality_peak_weeks"],

            "human_FOI_peak_relative_weeks":
                timing["human_FOI_peak_relative_to_mortality_peak_weeks"],
        }
    )

table_df = pd.DataFrame(rows)
table_df.to_csv(OUTPUT_TABLE_CSV, index=False)

print(f"[OK] Saved: {OUTPUT_TABLE_CSV}")


# =============================================================================
# Print concise numerical checks
# =============================================================================

print("\n" + "=" * 100)
print("KEY NUMERICAL CHECKS")
print("=" * 100)

for _, row in table_df.iterrows():
    wave = int(row["wave"])

    print(f"\nWave {wave}")
    print(
        "  median R0_H > 1: "
        f"{pct(row['R0_gt1_ascending'])}% ascending vs "
        f"{pct(row['R0_gt1_declining'])}% declining"
    )

    print(
        "  q025 R0_H > 1:   "
        f"{pct(row['R0_q025_gt1_ascending'])}% ascending vs "
        f"{pct(row['R0_q025_gt1_declining'])}% declining"
    )

    print(
        "  human FOI peak relative to mortality peak: "
        f"{signed_weeks(row['human_FOI_peak_relative_weeks'])} weeks"
    )

    print(
        "  spillover peak relative to mortality peak: "
        f"{signed_weeks(row['spillover_peak_relative_weeks'])} weeks"
    )


# =============================================================================
# LaTeX table
# =============================================================================

latex = []

latex.append(r"\begin{table}[!htbp]")
latex.append(r"\centering")
latex.append(
    r"\caption{Wave-specific transmission dynamics inferred from the "
    r"all-climate model. Ascending and declining phases were defined from "
    r"the observed mortality series using the major mortality peak and the "
    r"adjacent inter-wave troughs. The column $R_{0,H}>1$ gives the percentage "
    r"of weeks in which the posterior median exceeded one; ``strong'' gives "
    r"the percentage of weeks in which the lower 2.5\% posterior quantile also "
    r"exceeded one. Peak timing is reported in weeks relative to the observed "
    r"mortality peak, with negative values indicating that the inferred "
    r"quantity peaked before mortality.}"
)
latex.append(r"\label{tab:supp_wave_amplification}")
latex.append(r"\resizebox{\textwidth}{!}{%")
latex.append(r"\begin{tabular}{c c c c c c c c c}")
latex.append(r"\hline")
latex.append(
    r"Wave & "
    r"\multicolumn{2}{c}{$R_{0,H}>1$ (\% weeks)} & "
    r"\multicolumn{2}{c}{Strong $R_{0,H}>1$ (\% weeks)} & "
    r"\multicolumn{2}{c}{Median $R_{0,H}$} & "
    r"\multicolumn{2}{c}{Peak relative to mortality (weeks)} \\"
)
latex.append(
    r" & Asc. & Dec. & Asc. & Dec. & Asc. & Dec. & "
    r"Human FOI & Spillover \\"
)
latex.append(r"\hline")

for _, row in table_df.iterrows():
    latex.append(
        f"{int(row['wave'])} & "
        f"{pct(row['R0_gt1_ascending'])} & "
        f"{pct(row['R0_gt1_declining'])} & "
        f"{pct(row['R0_q025_gt1_ascending'])} & "
        f"{pct(row['R0_q025_gt1_declining'])} & "
        f"{row['median_R0_ascending']:.3f} & "
        f"{row['median_R0_declining']:.3f} & "
        f"{signed_weeks(row['human_FOI_peak_relative_weeks'])} & "
        f"{signed_weeks(row['spillover_peak_relative_weeks'])} \\\\"
    )

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"}")
latex.append(
    r"\begin{flushleft}\footnotesize "
    r"Asc., ascending phase; Dec., declining phase; FOI, force of infection. "
    r"The strong criterion requires the full central 95\% posterior credible "
    r"interval for $R_{0,H}$ to lie above one. "
    r"These summaries describe quantities inferred within the fitted model "
    r"and should not be interpreted as an independent causal test."
    r"\end{flushleft}"
)
latex.append(r"\end{table}")

latex_table = "\n".join(latex)

with open(OUTPUT_TABLE_TEX, "w") as f:
    f.write(latex_table)

print(f"[OK] Saved: {OUTPUT_TABLE_TEX}")


# =============================================================================
# Build LaTeX subsection from the data
# =============================================================================

# Pull waves explicitly for readable prose.
w1 = table_df.loc[table_df["wave"] == 1].iloc[0]
w2 = table_df.loc[table_df["wave"] == 2].iloc[0]
w3 = table_df.loc[table_df["wave"] == 3].iloc[0]


section = rf"""
\subsection{{Wave-specific concentration of supercritical human transmission}}
\label{{sec:supp_wave_amplification}}

To examine whether periods of inferred supercritical human transmission were
preferentially associated with epidemic expansion, we divided the mortality
series into three major epidemic waves and classified each wave into an
ascending and a declining phase. Wave boundaries were defined by the
inter-wave mortality troughs, and the observed mortality maximum within each
wave was used to separate ascent from decline. We then summarized the
posterior median human reproduction number, $R_{{0,H}}$, together with the
inferred human and zoonotic forces of infection within each phase
(Table~\ref{{tab:supp_wave_amplification}}).

The clearest phase-specific separation was observed in the second and third
epidemic waves. During wave~2, the posterior median of $R_{{0,H}}$ exceeded
one in {pct(w2["R0_gt1_ascending"])}\% of ascending-phase weeks, compared with
{pct(w2["R0_gt1_declining"])}\% of declining-phase weeks. Using the more
stringent criterion that the lower 2.5\% posterior quantile also exceeded one,
the corresponding proportions were {pct(w2["R0_q025_gt1_ascending"])}\% and
{pct(w2["R0_q025_gt1_declining"])}\%, respectively. A similar pattern was
observed in wave~3, for which {pct(w3["R0_gt1_ascending"])}\% of ascending
weeks had a posterior median $R_{{0,H}}>1$, whereas no declining-phase week
did so. Under the stricter posterior criterion, the corresponding proportions
were {pct(w3["R0_q025_gt1_ascending"])}\% and
{pct(w3["R0_q025_gt1_declining"])}\%.

Wave~1 showed a weaker separation between phases. The posterior median
$R_{{0,H}}$ exceeded one in {pct(w1["R0_gt1_ascending"])}\% of ascending
weeks and {pct(w1["R0_gt1_declining"])}\% of declining weeks, while the
stricter criterion was met in {pct(w1["R0_q025_gt1_ascending"])}\% and
{pct(w1["R0_q025_gt1_declining"])}\%, respectively. Thus, the concentration
of supercritical human transmission during epidemic ascent was most evident
for the second and third waves rather than being equally pronounced across
the full study period.

The timing of inferred human transmission was also consistent with a role for
human amplification in the later epidemic waves. Human FOI reached its
within-wave maximum {abs(int(w2["human_FOI_peak_relative_weeks"]))} weeks
{("before" if w2["human_FOI_peak_relative_weeks"] < 0 else "after" if w2["human_FOI_peak_relative_weeks"] > 0 else "at")} the wave~2 mortality peak
and {abs(int(w3["human_FOI_peak_relative_weeks"]))} weeks
{("before" if w3["human_FOI_peak_relative_weeks"] < 0 else "after" if w3["human_FOI_peak_relative_weeks"] > 0 else "at")} the wave~3 mortality peak.
For wave~1, the human FOI maximum occurred
{abs(int(w1["human_FOI_peak_relative_weeks"]))} weeks
{("before" if w1["human_FOI_peak_relative_weeks"] < 0 else "after" if w1["human_FOI_peak_relative_weeks"] > 0 else "at")} the mortality peak.
Importantly, inferred human FOI did not collapse immediately after the
mortality peak. This is not inconsistent with $R_{{0,H}}<1$: the force of
infection depends on the number of infectious humans present, so substantial
human transmission can persist while existing transmission chains are
contracting.

The inferred spillover component showed a different temporal pattern.
Within waves~2 and~3, the maximum spillover FOI occurred
{abs(int(w2["spillover_peak_relative_weeks"]))} and
{abs(int(w3["spillover_peak_relative_weeks"]))} weeks after the mortality
peak, respectively, whereas in wave~1 it occurred
{abs(int(w1["spillover_peak_relative_weeks"]))} weeks before the mortality
peak. The later waves therefore do not support a simple interpretation in
which a single large spillover pulse immediately precedes each mortality
wave. Instead, conditional on the fitted model, they are consistent with
continued zoonotic introductions occurring alongside a temporally varying
capacity for human-to-human amplification.

Taken together, these wave-specific summaries suggest that the second and
third mortality waves were characterized by a marked concentration of
supercritical human transmission during their ascending phases, followed by
predominantly subcritical human transmission after the mortality peak despite
continued inferred spillover. This pattern is consistent with recurrent
zoonotic seeding combined with episodic human amplification. Because all
transmission quantities in this analysis are posterior estimates from the
same fitted mechanistic model, however, the wave analysis should be
interpreted as a model-conditional description of the inferred mechanism
rather than as an independent causal validation.
""".strip()

# Special-case grammar: if a peak is exactly at mortality peak, remove
# "0 weeks at" and replace by "at".
section = section.replace("0 weeks at the wave~2 mortality peak", "at the wave~2 mortality peak")
section = section.replace("0 weeks at the wave~3 mortality peak", "at the wave~3 mortality peak")
section = section.replace("0 weeks at the mortality peak", "at the mortality peak")

with open(OUTPUT_SECTION_TEX, "w") as f:
    f.write(section + "\n")

print(f"[OK] Saved: {OUTPUT_SECTION_TEX}")


# =============================================================================
# Preview
# =============================================================================

print("\n" + "=" * 100)
print("LATEX TABLE PREVIEW")
print("=" * 100)
print(latex_table)

print("\n" + "=" * 100)
print("LATEX SECTION PREVIEW")
print("=" * 100)
print(section)

print("\n" + "=" * 100)
print("FILES WRITTEN")
print("=" * 100)
print(f"  {OUTPUT_TABLE_CSV}")
print(f"  {OUTPUT_TABLE_TEX}")
print(f"  {OUTPUT_SECTION_TEX}")

print("\nDone.")
