#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# plot_climate_perturbation_effects.py
# =============================================================================
"""
Dark-theme forest plot of posterior climate-perturbation effects.

Input
-----
results/final_figures/scenario_analysis/
    climate_perturbation_effects_posterior.csv

Outputs
-------
results/final_figures/scenario_analysis/
    climate_perturbation_forest_plot.png
    climate_perturbation_forest_plot.pdf

Usage
-----
python scripts/plot_climate_perturbation_effects.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "scenario_analysis"
)

CSV_PATH = (
    OUTPUT_DIR
    / "climate_perturbation_effects_posterior.csv"
)

FIG_PATH_PNG = (
    OUTPUT_DIR
    / "climate_perturbation_forest_plot.png"
)

FIG_PATH_PDF = (
    OUTPUT_DIR
    / "climate_perturbation_forest_plot.pdf"
)


# =============================================================================
# 2. SCENARIOS
# =============================================================================

SCENARIO_ORDER = [
    (
        "temperature",
        "+",
        "Temperature +2 °C",
    ),
    (
        "temperature",
        "-",
        "Temperature −2 °C",
    ),
    (
        "vapour_pressure",
        "+",
        "Vapour pressure +1.69 hPa",
    ),
    (
        "vapour_pressure",
        "-",
        "Vapour pressure −1.69 hPa",
    ),
    (
        "rainfall",
        "+",
        "Rainfall +50.8 mm",
    ),
    (
        "rainfall",
        "-",
        "Rainfall −50.8 mm",
    ),
]


# =============================================================================
# 3. DARK THEME
# =============================================================================

BG_COLOR = "#0B1220"
AX_COLOR = "#111C2E"

TEXT_COLOR = "#F1F5F9"
MUTED_TEXT = "#CBD5E1"

GRID_COLOR = "#64748B"
SPINE_COLOR = "#64748B"

COLOR_R = "#38BDF8"
COLOR_H = "#FB923C"

ZERO_COLOR = "#E2E8F0"


plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.titlesize": 14,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 11,

    "axes.linewidth": 0.9,

    "figure.facecolor": BG_COLOR,

    "axes.facecolor": AX_COLOR,

    "savefig.facecolor": BG_COLOR,

    "text.color": TEXT_COLOR,

    "axes.labelcolor": TEXT_COLOR,

    "axes.edgecolor": SPINE_COLOR,

    "xtick.color": MUTED_TEXT,

    "ytick.color": TEXT_COLOR,

    "grid.color": GRID_COLOR,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# =============================================================================
# 4. MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"Could not find:\n{CSV_PATH}\n\n"
            "Run scripts/run_climate_perturbation_analysis.py first."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "covariate",
        "direction",
        "component",
        "statistic",
        "median_pct_change",
        "ci_lo",
        "ci_hi",
        "n_posterior_draws",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # -------------------------------------------------------------------------
    # Select time-averaged statistic
    # -------------------------------------------------------------------------

    df = (
        df[df["statistic"] == "mean"]
        .copy()
        .reset_index(drop=True)
    )

    if df.empty:

        raise RuntimeError(
            "No rows with statistic == 'mean' were found."
        )

    # -------------------------------------------------------------------------
    # Posterior draw count
    # -------------------------------------------------------------------------

    unique_draw_counts = (
        df["n_posterior_draws"]
        .dropna()
        .unique()
    )

    if len(unique_draw_counts) != 1:

        raise RuntimeError(
            "Expected exactly one posterior-draw count, "
            f"but found: {unique_draw_counts}"
        )

    n_draws = int(unique_draw_counts[0])

    print(
        f"Posterior draws used: {n_draws}"
    )

    # -------------------------------------------------------------------------
    # Validate scenarios
    # -------------------------------------------------------------------------

    for covariate, direction, _ in SCENARIO_ORDER:

        for component in ["beta_R", "beta_H"]:

            rows = df[
                (df["covariate"] == covariate)
                &
                (df["direction"] == direction)
                &
                (df["component"] == component)
            ]

            if len(rows) != 1:

                raise RuntimeError(
                    "Expected exactly one row for "
                    f"{covariate}, {direction}, {component}; "
                    f"found {len(rows)}."
                )

    # =========================================================================
    # 5. BUILD FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6.5),
        facecolor=BG_COLOR,
    )

    ax.set_facecolor(AX_COLOR)

    y_positions = np.arange(
        len(SCENARIO_ORDER)
    )

    offset = 0.16

    # -------------------------------------------------------------------------
    # Subtle alternating row backgrounds
    # -------------------------------------------------------------------------

    for i in range(len(SCENARIO_ORDER)):

        if i % 2 == 0:

            ax.axhspan(
                i - 0.45,
                i + 0.45,
                color="#FFFFFF",
                alpha=0.035,
                zorder=0,
            )

    # -------------------------------------------------------------------------
    # Forest plot
    # -------------------------------------------------------------------------

    for i, (
        covariate,
        direction,
        label,
    ) in enumerate(SCENARIO_ORDER):

        y = y_positions[i]

        for (
            component,
            color,
            y_offset,
            marker,
        ) in [

            (
                "beta_R",
                COLOR_R,
                offset,
                "o",
            ),

            (
                "beta_H",
                COLOR_H,
                -offset,
                "s",
            ),
        ]:

            row = df[
                (df["covariate"] == covariate)
                &
                (df["direction"] == direction)
                &
                (df["component"] == component)
            ].iloc[0]

            med = float(
                row["median_pct_change"]
            )

            lo = float(
                row["ci_lo"]
            )

            hi = float(
                row["ci_hi"]
            )

            if not all(
                np.isfinite([med, lo, hi])
            ):

                raise RuntimeError(
                    "Non-finite plotting value for "
                    f"{covariate}, {direction}, {component}."
                )

            if not lo <= med <= hi:

                raise RuntimeError(
                    "Credible interval does not contain median: "
                    f"{covariate}, {direction}, {component}"
                )

            ax.errorbar(

                med,

                y + y_offset,

                xerr=[
                    [med - lo],
                    [hi - med],
                ],

                fmt=marker,

                color=color,

                ecolor=color,

                capsize=5,

                capthick=1.8,

                elinewidth=2.0,

                markersize=8,

                markerfacecolor=color,

                markeredgecolor=BG_COLOR,

                markeredgewidth=1.0,

                zorder=4,
            )

    # =========================================================================
    # 6. ZERO REFERENCE LINE
    # =========================================================================

    ax.axvline(

        0,

        linestyle="--",

        color=ZERO_COLOR,

        linewidth=1.5,

        alpha=0.85,

        zorder=2,
    )

    # =========================================================================
    # 7. AXES
    # =========================================================================

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        [
            label
            for _, _, label
            in SCENARIO_ORDER
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Change in time-averaged transmission coefficient "
        "relative to baseline (%)",
        color=TEXT_COLOR,
        labelpad=14,
    )

    ax.set_title(
        "Climate perturbation effects on transmission\n"
        f"Posterior median and 95% credible interval "
        f"({n_draws:,} posterior draws)",
        color=TEXT_COLOR,
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="x",
        color=GRID_COLOR,
        alpha=0.25,
        linestyle="-",
        linewidth=0.8,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Spines
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE_COLOR)

    ax.spines["bottom"].set_color(SPINE_COLOR)

    # -------------------------------------------------------------------------
    # Ticks
    # -------------------------------------------------------------------------

    ax.tick_params(
        axis="both",
        length=0,
        pad=10,
    )

    # =========================================================================
    # 8. LEGEND
    # =========================================================================

    legend_elements = [

        Line2D(
            [0],
            [0],

            marker="o",

            color=COLOR_R,

            label=(
                r"$\beta_R$ "
                "(reservoir-associated spillover)"
            ),

            markerfacecolor=COLOR_R,

            markeredgecolor=BG_COLOR,

            markeredgewidth=1.0,

            linestyle="None",

            markersize=9,
        ),

        Line2D(
            [0],
            [0],

            marker="s",

            color=COLOR_H,

            label=(
                r"$\beta_H$ "
                "(human-associated amplification)"
            ),

            markerfacecolor=COLOR_H,

            markeredgecolor=BG_COLOR,

            markeredgewidth=1.0,

            linestyle="None",

            markersize=9,
        ),
    ]

    legend = ax.legend(

        handles=legend_elements,

        frameon=False,

        loc="upper center",

        bbox_to_anchor=(0.5, -0.13),

        ncol=2,

        columnspacing=2.5,

        handletextpad=0.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # =========================================================================
    # 9. LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.28,
        right=0.96,
        top=0.84,
        bottom=0.22,
    )

    # =========================================================================
    # 10. SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIG_PATH_PNG,
        dpi=600,
        facecolor=BG_COLOR,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_PATH_PDF,
        facecolor=BG_COLOR,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"\n[OK] Saved PNG: {FIG_PATH_PNG}"
    )

    print(
        f"[OK] Saved PDF: {FIG_PATH_PDF}"
    )

    print("\nDone.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()