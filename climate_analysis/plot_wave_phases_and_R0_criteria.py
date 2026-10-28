#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# plot_climate_perturbation_effects.py
# =============================================================================

"""
BOMBAY PLAGUE: POSTERIOR CLIMATE-PERTURBATION EFFECTS
DARK PRESENTATION VERSION — TIMES NEW ROMAN
=====================================================

Forest plot showing the posterior effects of climate perturbations
on the two transmission coefficients:

    beta_R : reservoir-associated spillover
    beta_H : human-associated amplification

For each climate scenario, the figure displays:

    - Posterior median percentage change
    - 95% credible interval

The plotted statistic is the relative change in the
time-averaged transmission coefficient.

This script DOES NOT refit or re-simulate the model.

INPUT
-----
results/final_figures/scenario_analysis/
    climate_perturbation_effects_posterior.csv

OUTPUT
------
climate_analysis/figures/
    fig_climate_perturbation_forest_dark.png
    fig_climate_perturbation_forest_dark.pdf

STYLE
-----
- Times New Roman
- STIX mathematical notation
- Dark background #0B0D10
- White text
- Bright blue reservoir-spillover markers
- Bright orange amplification markers
- Posterior median and 95% credible intervals
- High-resolution PNG and vector PDF

USAGE
-----
python climate_analysis/plot_climate_perturbation_effects.py

or

python scripts/plot_climate_perturbation_effects.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Works when the script is inside climate_analysis/
# or inside the original scripts/ directory.

if SCRIPT_DIR.name == "climate_analysis":

    PROJECT_ROOT = SCRIPT_DIR.parent

    CLIMATE_DIR = SCRIPT_DIR

elif SCRIPT_DIR.name == "scripts":

    PROJECT_ROOT = SCRIPT_DIR.parent

    CLIMATE_DIR = PROJECT_ROOT / "climate_analysis"

else:

    PROJECT_ROOT = SCRIPT_DIR.parent

    CLIMATE_DIR = PROJECT_ROOT / "climate_analysis"


# =============================================================================
# INPUT
# =============================================================================

SCENARIO_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "scenario_analysis"
)

INPUT_CSV = (
    SCENARIO_DIR
    / "climate_perturbation_effects_posterior.csv"
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = CLIMATE_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_PATH_PNG = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_dark.png"
)

FIG_PATH_PDF = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_dark.pdf"
)


# =============================================================================
# 2. CLIMATE SCENARIOS
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
# 3. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#FFFFFF"

GRID = "#414750"

SPINE = "#D1D5DB"


# Reservoir-associated spillover

COLOR_R = "#38BDF8"


# Human-associated amplification

COLOR_H = "#FFB000"


# Zero-change reference line

COLOR_ZERO = "#FFFFFF"


# Subtle alternating row background

COLOR_ROW = "#171B21"


# =============================================================================
# 4. TIMES NEW ROMAN STYLE
# =============================================================================

try:

    times_font_path = font_manager.findfont(
        "Times New Roman",
        fallback_to_default=False
    )

    print(
        f"[OK] Times New Roman found: {times_font_path}"
    )

except ValueError:

    print(
        "[WARNING] Times New Roman was not found. "
        "Matplotlib will use a fallback serif font."
    )


plt.rcParams.update({

    # -------------------------------------------------------------------------
    # FONT
    # -------------------------------------------------------------------------

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ],

    "mathtext.fontset": "stix",

    # -------------------------------------------------------------------------
    # FONT SIZES
    # -------------------------------------------------------------------------

    "font.size": 12,

    "axes.labelsize": 14,

    "axes.titlesize": 16,

    "legend.fontsize": 11,

    "xtick.labelsize": 11,

    "ytick.labelsize": 12,

    # -------------------------------------------------------------------------
    # LINES
    # -------------------------------------------------------------------------

    "lines.linewidth": 2.0,

    "axes.linewidth": 1.0,

    # -------------------------------------------------------------------------
    # DARK BACKGROUND
    # -------------------------------------------------------------------------

    "figure.facecolor": BG,

    "axes.facecolor": BG,

    "savefig.facecolor": BG,

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------

    "text.color": TEXT,

    "axes.labelcolor": TEXT,

    "axes.edgecolor": SPINE,

    "xtick.color": TEXT,

    "ytick.color": TEXT,

    # -------------------------------------------------------------------------
    # GRID
    # -------------------------------------------------------------------------

    "grid.color": GRID,

    # -------------------------------------------------------------------------
    # TICKS
    # -------------------------------------------------------------------------

    "xtick.direction": "out",

    "ytick.direction": "out",

    # -------------------------------------------------------------------------
    # PDF FONT EMBEDDING
    # -------------------------------------------------------------------------

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})


# =============================================================================
# 5. COMMON AXIS STYLE
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(BG)

    # -------------------------------------------------------------------------
    # Spines
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE)

    ax.spines["bottom"].set_color(SPINE)

    # -------------------------------------------------------------------------
    # Ticks
    # -------------------------------------------------------------------------

    ax.tick_params(
        axis="both",
        colors=TEXT,
        direction="out",
        length=5,
        width=1.0,
        pad=8
    )

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.8,
        alpha=0.40,
        linestyle="--",
        zorder=0
    )

    ax.set_axisbelow(True)


# =============================================================================
# 6. LOAD AND VALIDATE POSTERIOR RESULTS
# =============================================================================

def load_posterior_results() -> tuple[pd.DataFrame, int]:

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Posterior results not found:\n{INPUT_CSV}\n\n"
            "Run run_climate_perturbation_analysis.py first."
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

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
    # Select time-averaged coefficient statistic
    # -------------------------------------------------------------------------

    df = (
        df[
            df["statistic"] == "mean"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if df.empty:

        raise RuntimeError(
            "No rows with statistic == 'mean' were found."
        )

    # -------------------------------------------------------------------------
    # Read posterior-draw count
    # -------------------------------------------------------------------------

    unique_draw_counts = (
        df["n_posterior_draws"]
        .dropna()
        .unique()
    )

    if len(unique_draw_counts) != 1:

        raise RuntimeError(
            "Expected exactly one value of n_posterior_draws "
            "across the plotted rows, but found: "
            f"{unique_draw_counts}"
        )

    n_draws = int(
        unique_draw_counts[0]
    )

    if n_draws <= 0:

        raise RuntimeError(
            "n_posterior_draws must be positive."
        )

    # -------------------------------------------------------------------------
    # Validate all scenarios and components
    # -------------------------------------------------------------------------

    for covariate, direction, _ in SCENARIO_ORDER:

        for component in [
            "beta_R",
            "beta_H",
        ]:

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

            row = rows.iloc[0]

            med = float(
                row["median_pct_change"]
            )

            lo = float(
                row["ci_lo"]
            )

            hi = float(
                row["ci_hi"]
            )

            if not np.all(
                np.isfinite([med, lo, hi])
            ):

                raise RuntimeError(
                    "Non-finite posterior summary for "
                    f"{covariate}, {direction}, {component}."
                )

            if not (
                lo <= med <= hi
            ):

                raise RuntimeError(
                    "Credible interval does not contain "
                    "the posterior median for "
                    f"{covariate}, {direction}, {component}: "
                    f"lo={lo}, median={med}, hi={hi}"
                )

    return df, n_draws


# =============================================================================
# 7. FOREST PLOT
# =============================================================================

def draw_forest_plot(
    ax,
    df: pd.DataFrame
) -> None:

    y_positions = np.arange(
        len(SCENARIO_ORDER)
    )

    offset = 0.16

    # =========================================================================
    # ALTERNATING ROW SHADING
    # =========================================================================

    for i in range(
        len(SCENARIO_ORDER)
    ):

        if i % 2 == 0:

            ax.axhspan(
                i - 0.46,
                i + 0.46,
                facecolor=COLOR_ROW,
                alpha=0.55,
                linewidth=0,
                zorder=0
            )

    # =========================================================================
    # POSTERIOR MEDIANS AND 95% CREDIBLE INTERVALS
    # =========================================================================

    for i, (
        covariate,
        direction,
        label
    ) in enumerate(
        SCENARIO_ORDER
    ):

        y = y_positions[i]

        for (
            component,
            color,
            y_offset,
            marker
        ) in [

            (
                "beta_R",
                COLOR_R,
                -offset,
                "o",
            ),

            (
                "beta_H",
                COLOR_H,
                offset,
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

            # -----------------------------------------------------------------
            # Asymmetric 95% credible interval
            # -----------------------------------------------------------------

            xerr = [
                [med - lo],
                [hi - med],
            ]

            ax.errorbar(
                med,
                y + y_offset,

                xerr=xerr,

                fmt=marker,

                color=color,

                ecolor=color,

                elinewidth=2.2,

                capsize=5,

                capthick=1.8,

                markersize=9,

                markerfacecolor=color,

                markeredgecolor=BG,

                markeredgewidth=1.0,

                linestyle="None",

                zorder=5
            )

    # =========================================================================
    # ZERO-CHANGE REFERENCE LINE
    # =========================================================================

    ax.axvline(
        0.0,
        color=COLOR_ZERO,
        linestyle="--",
        linewidth=1.5,
        alpha=0.90,
        zorder=2
    )

    # =========================================================================
    # Y-AXIS LABELS
    # =========================================================================

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        [
            label
            for _, _, label
            in SCENARIO_ORDER
        ],
        fontsize=13,
        color=TEXT
    )

    ax.invert_yaxis()

    # =========================================================================
    # X-AXIS LABEL
    # =========================================================================

    ax.set_xlabel(
        "Change in time-averaged transmission coefficient "
        "relative to baseline (%)",
        fontsize=14,
        labelpad=15,
        color=TEXT
    )

    # =========================================================================
    # AXIS LIMITS
    # =========================================================================

    # Collect credible interval bounds to determine an appropriate x-range.

    relevant_rows = []

    for covariate, direction, _ in SCENARIO_ORDER:

        for component in ["beta_R", "beta_H"]:

            row = df[
                (df["covariate"] == covariate)
                &
                (df["direction"] == direction)
                &
                (df["component"] == component)
            ].iloc[0]

            relevant_rows.append(row)

    lower_values = [
        float(row["ci_lo"])
        for row in relevant_rows
    ]

    upper_values = [
        float(row["ci_hi"])
        for row in relevant_rows
    ]

    xmin = min(
        min(lower_values),
        0.0
    )

    xmax = max(
        max(upper_values),
        0.0
    )

    span = xmax - xmin

    if span <= 0:

        span = 1.0

    padding = 0.08 * span

    ax.set_xlim(
        xmin - padding,
        xmax + padding
    )

    # =========================================================================
    # Y-AXIS LIMITS
    # =========================================================================

    ax.set_ylim(
        len(SCENARIO_ORDER) - 0.5,
        -0.5
    )

    # =========================================================================
    # STYLE
    # =========================================================================

    style_axis(
        ax
    )


# =============================================================================
# 8. LEGEND
# =============================================================================

def add_legend(ax) -> None:

    legend_elements = [

        Line2D(
            [0],
            [0],

            marker="o",

            linestyle="None",

            color=COLOR_R,

            markerfacecolor=COLOR_R,

            markeredgecolor=BG,

            markeredgewidth=1.0,

            markersize=10,

            label=(
                r"$\beta_R$ "
                "(reservoir-associated spillover)"
            ),
        ),

        Line2D(
            [0],
            [0],

            marker="s",

            linestyle="None",

            color=COLOR_H,

            markerfacecolor=COLOR_H,

            markeredgecolor=BG,

            markeredgewidth=1.0,

            markersize=10,

            label=(
                r"$\beta_H$ "
                "(human-associated amplification)"
            ),
        ),

    ]

    legend = ax.legend(
        handles=legend_elements,

        loc="upper center",

        bbox_to_anchor=(0.5, -0.16),

        ncol=2,

        fontsize=12,

        frameon=False,

        columnspacing=2.0,

        handletextpad=0.7
    )

    for text in legend.get_texts():

        text.set_color(TEXT)


# =============================================================================
# 9. SAVE FIGURE
# =============================================================================

def save_figure(fig) -> None:

    # -------------------------------------------------------------------------
    # High-resolution PNG
    # -------------------------------------------------------------------------

    fig.savefig(
        FIG_PATH_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )

    # -------------------------------------------------------------------------
    # Vector PDF
    # -------------------------------------------------------------------------

    fig.savefig(
        FIG_PATH_PDF,
        bbox_inches="tight",
        facecolor=BG
    )

    print(
        f"[OK] Saved {FIG_PATH_PNG}"
    )

    print(
        f"[OK] Saved {FIG_PATH_PDF}"
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:

    print("\n==========================================")
    print("BOMBAY PLAGUE")
    print("CLIMATE-PERTURBATION FOREST PLOT")
    print("DARK PRESENTATION — TIMES NEW ROMAN")
    print("==========================================")

    print(
        f"\nInput: {INPUT_CSV}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    df, n_draws = load_posterior_results()

    print(
        f"\nPosterior draws per scenario: {n_draws:,}"
    )

    print(
        f"Number of plotted scenarios: {len(SCENARIO_ORDER)}"
    )

    print(
        "Transmission components: beta_R, beta_H"
    )

    # =========================================================================
    # BUILD FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(13.5, 7.5),
        dpi=300,
        facecolor=BG
    )

    ax.set_facecolor(BG)

    # =========================================================================
    # DRAW FOREST PLOT
    # =========================================================================

    draw_forest_plot(
        ax,
        df
    )

    # =========================================================================
    # TITLE
    # =========================================================================

    ax.set_title(
        "Climate perturbation effects on transmission coefficients",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        pad=22
    )

    # =========================================================================
    # SUBTITLE
    # =========================================================================

    ax.text(
        0.5,
        1.025,

        (
            "Posterior median and 95% credible interval  "
            f"({n_draws:,} posterior draws per scenario)"
        ),

        transform=ax.transAxes,

        ha="center",

        va="bottom",

        fontsize=12,

        color=SPINE
    )

    # =========================================================================
    # LEGEND
    # =========================================================================

    add_legend(
        ax
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.29,
        right=0.96,
        bottom=0.23,
        top=0.83
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_figure(
        fig
    )

    plt.show()

    plt.close(
        fig
    )

    print("\n==========================================")
    print("FIGURE GENERATED SUCCESSFULLY")
    print("==========================================")

    print(
        f"\nSaved in: {OUTPUT_DIR}"
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()