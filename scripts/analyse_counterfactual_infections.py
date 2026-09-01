# =============================================================================
# analyse_counterfactual_infections.py
# =============================================================================
"""
Analyse counterfactual infection trajectories from the all-climate model.

Input
-----
results/final_figures/transmission_dynamics/
    counterfactual_infections_all_climate_only.csv

Outputs
-------
results/final_figures/transmission_dynamics/counterfactual_analysis/

    counterfactual_summary_table.csv
    counterfactual_peak_table.csv
    counterfactual_weekly_comparison.csv

    counterfactual_infections_timeseries.png
    counterfactual_infections_timeseries.pdf

    counterfactual_relative_infections.png
    counterfactual_relative_infections.pdf


IMPORTANT
---------
The input CSV contains WEEKLY posterior summaries.

Therefore:

    sum(full_infections_median)

is the sum of weekly posterior medians.

It is NOT the posterior median of total cumulative infections.

The summary produced here is therefore descriptive and useful for plotting
and interpretation, but posterior credible intervals for cumulative effects
must be calculated from the original draw-level trajectories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "counterfactual_infections_all_climate_only.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "counterfactual_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Plot settings
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


# Colours matching the conceptual mechanisms
COLOR_FULL = "#2C7C2C"
COLOR_NO_HUMAN = "#1f77b4"
COLOR_NO_SPILL = "#D55E00"
COLOR_DEATHS = "black"


# =============================================================================
# Load data
# =============================================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_CSV}\n\n"
        "Run counterfactual_infections_all_climate.py first."
    )

df = pd.read_csv(INPUT_CSV, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)


required_cols = [
    "week_ending",
    "observed_deaths",
    "full_infections_q025",
    "full_infections_median",
    "full_infections_q975",
    "no_human_infections_q025",
    "no_human_infections_median",
    "no_human_infections_q975",
    "no_spillover_infections_q025",
    "no_spillover_infections_median",
    "no_spillover_infections_q975",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(
        "Input CSV is missing required columns:\n"
        + "\n".join(missing)
    )


# =============================================================================
# 1. Overall descriptive summary
# =============================================================================

full_total = float(df["full_infections_median"].sum())
no_human_total = float(df["no_human_infections_median"].sum())
no_spill_total = float(df["no_spillover_infections_median"].sum())


def pct_of_full(value: float) -> float:
    if full_total <= 0:
        return np.nan
    return 100.0 * value / full_total


def pct_reduction(value: float) -> float:
    if full_total <= 0:
        return np.nan
    return 100.0 * (1.0 - value / full_total)


summary_table = pd.DataFrame({
    "scenario": [
        "Full model",
        "No human transmission",
        "No recurrent spillover after week 4",
    ],
    "sum_weekly_posterior_medians": [
        full_total,
        no_human_total,
        no_spill_total,
    ],
    "percent_of_full_model": [
        100.0,
        pct_of_full(no_human_total),
        pct_of_full(no_spill_total),
    ],
    "percent_reduction_vs_full": [
        0.0,
        pct_reduction(no_human_total),
        pct_reduction(no_spill_total),
    ],
})


summary_csv = OUTPUT_DIR / "counterfactual_summary_table.csv"
summary_table.to_csv(summary_csv, index=False)


# =============================================================================
# 2. Identify observed mortality peaks
# =============================================================================
#
# We use the three historically obvious epidemic peaks by dividing the
# observation series into calendar-year waves:
#
# Wave 1 = 1896-1897
# Wave 2 = 1897-1898
# Wave 3 = 1898-1899
#
# These date windows avoid accidentally selecting multiple nearby local peaks.
# =============================================================================

wave_windows = {
    "Wave 1": ("1896-09-25", "1897-09-01"),
    "Wave 2": ("1897-09-02", "1898-08-31"),
    "Wave 3": ("1898-09-01", "1899-06-02"),
}


peak_rows = []

for wave_name, (start, end) in wave_windows.items():

    block = df[
        (df["week_ending"] >= pd.Timestamp(start))
        & (df["week_ending"] <= pd.Timestamp(end))
    ].copy()

    if block.empty:
        continue

    peak_idx = block["observed_deaths"].idxmax()
    row = df.loc[peak_idx]

    peak_rows.append({
        "wave": wave_name,
        "peak_date": row["week_ending"],
        "observed_deaths": row["observed_deaths"],

        "full_infections_median":
            row["full_infections_median"],

        "no_human_infections_median":
            row["no_human_infections_median"],

        "no_spillover_infections_median":
            row["no_spillover_infections_median"],

        "no_human_percent_of_full":
            100.0
            * row["no_human_infections_median"]
            / row["full_infections_median"]
            if row["full_infections_median"] > 0
            else np.nan,

        "no_spillover_percent_of_full":
            100.0
            * row["no_spillover_infections_median"]
            / row["full_infections_median"]
            if row["full_infections_median"] > 0
            else np.nan,
    })


peak_table = pd.DataFrame(peak_rows)

peak_csv = OUTPUT_DIR / "counterfactual_peak_table.csv"
peak_table.to_csv(peak_csv, index=False)


# =============================================================================
# 3. Weekly relative trajectories
# =============================================================================

weekly = df.copy()

weekly["no_human_percent_of_full"] = np.where(
    weekly["full_infections_median"] > 0,
    100.0
    * weekly["no_human_infections_median"]
    / weekly["full_infections_median"],
    np.nan,
)

weekly["no_spillover_percent_of_full"] = np.where(
    weekly["full_infections_median"] > 0,
    100.0
    * weekly["no_spillover_infections_median"]
    / weekly["full_infections_median"],
    np.nan,
)

weekly["human_removal_reduction_percent"] = (
    100.0 - weekly["no_human_percent_of_full"]
)

weekly["spillover_removal_reduction_percent"] = (
    100.0 - weekly["no_spillover_percent_of_full"]
)

weekly_csv = OUTPUT_DIR / "counterfactual_weekly_comparison.csv"
weekly.to_csv(weekly_csv, index=False)


# =============================================================================
# 4. Figure: infection trajectories
# =============================================================================

fig, ax1 = plt.subplots(figsize=(10, 5.5))


# Full model CrI
ax1.fill_between(
    df["week_ending"],
    df["full_infections_q025"],
    df["full_infections_q975"],
    color=COLOR_FULL,
    alpha=0.15,
    linewidth=0,
)

ax1.plot(
    df["week_ending"],
    df["full_infections_median"],
    color=COLOR_FULL,
    linewidth=2.2,
    label="Full model",
)


# No human transmission
ax1.fill_between(
    df["week_ending"],
    df["no_human_infections_q025"],
    df["no_human_infections_q975"],
    color=COLOR_NO_HUMAN,
    alpha=0.12,
    linewidth=0,
)

ax1.plot(
    df["week_ending"],
    df["no_human_infections_median"],
    color=COLOR_NO_HUMAN,
    linewidth=1.8,
    label="No human transmission",
)


# No recurrent spillover
ax1.fill_between(
    df["week_ending"],
    df["no_spillover_infections_q025"],
    df["no_spillover_infections_q975"],
    color=COLOR_NO_SPILL,
    alpha=0.12,
    linewidth=0,
)

ax1.plot(
    df["week_ending"],
    df["no_spillover_infections_median"],
    color=COLOR_NO_SPILL,
    linewidth=1.8,
    label="No recurrent spillover",
)


ax1.set_xlabel("Week ending")
ax1.set_ylabel("Weekly modelled infections")

ax1.xaxis.set_major_locator(
    mdates.MonthLocator(interval=4)
)

ax1.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax1.spines["top"].set_visible(False)
ax1.grid(False)


# Observed mortality on secondary axis
ax2 = ax1.twinx()

ax2.scatter(
    df["week_ending"],
    df["observed_deaths"],
    s=11,
    color=COLOR_DEATHS,
    alpha=0.28,
    label="Observed plague deaths",
)

ax2.set_ylabel("Observed weekly plague deaths")
ax2.spines["top"].set_visible(False)


# Mark mortality peaks
for _, row in peak_table.iterrows():

    ax1.axvline(
        row["peak_date"],
        color="0.6",
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
    )


# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    loc="upper left",
)

ax1.set_title(
    "Counterfactual transmission simulations — all-climate model"
)

fig.tight_layout()


fig_png = OUTPUT_DIR / "counterfactual_infections_timeseries.png"
fig_pdf = OUTPUT_DIR / "counterfactual_infections_timeseries.pdf"

fig.savefig(fig_png, dpi=600, bbox_inches="tight")
fig.savefig(fig_pdf, bbox_inches="tight")

plt.close(fig)


# =============================================================================
# 5. Figure: counterfactual infections as percentage of full model
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(
    weekly["week_ending"],
    weekly["no_human_percent_of_full"],
    color=COLOR_NO_HUMAN,
    linewidth=1.9,
    label="No human transmission",
)

ax.plot(
    weekly["week_ending"],
    weekly["no_spillover_percent_of_full"],
    color=COLOR_NO_SPILL,
    linewidth=1.9,
    label="No recurrent spillover",
)

ax.axhline(
    100,
    color="0.35",
    linestyle="--",
    linewidth=1.0,
    label="Full model",
)

ax.set_xlabel("Week ending")
ax.set_ylabel("Counterfactual infections (% of full model)")

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=4)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax.set_ylim(bottom=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False)

ax.set_title(
    "Contribution of transmission pathways to inferred infections"
)

fig.tight_layout()


relative_png = OUTPUT_DIR / "counterfactual_relative_infections.png"
relative_pdf = OUTPUT_DIR / "counterfactual_relative_infections.pdf"

fig.savefig(relative_png, dpi=600, bbox_inches="tight")
fig.savefig(relative_pdf, bbox_inches="tight")

plt.close(fig)


# =============================================================================
# Report
# =============================================================================

print("\n" + "=" * 90)
print("COUNTERFACTUAL ANALYSIS — ALL-CLIMATE MODEL")
print("=" * 90)

print("\nDESCRIPTIVE CUMULATIVE SUMMARY")
print("-" * 90)

print(
    summary_table.to_string(
        index=False,
        formatters={
            "sum_weekly_posterior_medians": "{:,.1f}".format,
            "percent_of_full_model": "{:.2f}".format,
            "percent_reduction_vs_full": "{:.2f}".format,
        },
    )
)


print("\n\nOBSERVED MORTALITY PEAKS")
print("-" * 90)

if not peak_table.empty:

    display_peak = peak_table.copy()

    print(
        display_peak.to_string(
            index=False,
            formatters={
                "observed_deaths": "{:,.0f}".format,
                "full_infections_median": "{:,.2f}".format,
                "no_human_infections_median": "{:,.2f}".format,
                "no_spillover_infections_median": "{:,.6f}".format,
                "no_human_percent_of_full": "{:.2f}".format,
                "no_spillover_percent_of_full": "{:.4f}".format,
            },
        )
    )


print("\n\nINTERPRETATION")
print("-" * 90)

print(
    f"""
Full model:
    Sum of weekly posterior-median infections = {full_total:,.1f}

No human transmission:
    Sum = {no_human_total:,.1f}
    = {pct_of_full(no_human_total):.2f}% of full model
    = {pct_reduction(no_human_total):.2f}% reduction

No recurrent spillover after week 4:
    Sum = {no_spill_total:,.1f}
    = {pct_of_full(no_spill_total):.2f}% of full model
    = {pct_reduction(no_spill_total):.2f}% reduction

CAUTION:
These cumulative values are sums of weekly posterior medians.
They are not posterior medians of cumulative infections and therefore
should not be reported with posterior credible intervals.

For publication-quality cumulative-effect estimates, compute total infections
within EACH posterior draw first, then summarize those totals across draws.
"""
)


print("\nSaved outputs:")
print(f"  {summary_csv}")
print(f"  {peak_csv}")
print(f"  {weekly_csv}")
print(f"  {fig_png}")
print(f"  {fig_pdf}")
print(f"  {relative_png}")
print(f"  {relative_pdf}")

print("\nDone.")