#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# plot_climate_perturbation_effects_updated.py
# =============================================================================

"""
BOMBAY PLAGUE: POSTERIOR CLIMATE-PERTURBATION EFFECTS
DARK PRESENTATION VERSION — TIMES NEW ROMAN
=====================================================

This script generates TWO SEPARATE forest plots:

    FIGURE 1: RAW-UNIT CLIMATE PERTURBATIONS

        Temperature:     +/- 2 degrees Celsius
        Vapour pressure: +/- 0.05 inHg
        Rainfall:        +/- 2 inches

    FIGURE 2: STANDARDIZED CLIMATE PERTURBATIONS

        Temperature:     +/- 1 SD
        Vapour pressure: +/- 1 SD
        Rainfall:        +/- 1 SD

Both figures display posterior effects on:

    beta_R : reservoir-associated spillover
    beta_H : human-associated amplification

For each climate scenario:

    - Posterior median percentage change
    - 95% credible interval

The plotted statistic is the percentage change in the
time-averaged transmission coefficient relative to baseline.

This script DOES NOT refit or re-simulate the model.

INPUT
-----
results/final_figures/scenario_analysis_updated/

    climate_perturbation_effects_posterior.csv

    climate_perturbation_effects_posterior_1sd.csv


OUTPUT
------
climate_analysis/figures/

    fig_climate_perturbation_forest_updated_dark.png
    fig_climate_perturbation_forest_updated_dark.pdf

    fig_climate_perturbation_forest_1sd_dark.png
    fig_climate_perturbation_forest_1sd_dark.pdf


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
python climate_analysis/plot_climate_perturbation_effects_updated.py
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
# 2. INPUT DIRECTORY
# =============================================================================

SCENARIO_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "scenario_analysis_updated"
)


# Original raw-unit perturbations

INPUT_RAW_CSV = (
    SCENARIO_DIR
    / "climate_perturbation_effects_posterior.csv"
)


# Standardized +/- 1 SD perturbations

INPUT_1SD_CSV = (
    SCENARIO_DIR
    / "climate_perturbation_effects_posterior_1sd.csv"
)


# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = CLIMATE_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. OUTPUT FILENAMES
# =============================================================================

# -------------------------------------------------------------------------
# Figure 1: Raw-unit perturbations
# -------------------------------------------------------------------------

FIG_RAW_PNG = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_updated_dark.png"
)

FIG_RAW_PDF = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_updated_dark.pdf"
)


# -------------------------------------------------------------------------
# Figure 2: +/- 1 SD perturbations
# -------------------------------------------------------------------------

FIG_1SD_PNG = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_1sd_dark.png"
)

FIG_1SD_PDF = (
    OUTPUT_DIR
    / "fig_climate_perturbation_forest_1sd_dark.pdf"
)


# =============================================================================
# 5. CLIMATE SCENARIOS
# =============================================================================

# -------------------------------------------------------------------------
# Original raw-unit perturbations
# -------------------------------------------------------------------------

SCENARIO_ORDER_RAW = [

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


# -------------------------------------------------------------------------
# Standardized +/- 1 SD perturbations
# -------------------------------------------------------------------------

SCENARIO_ORDER_1SD = [

    (
        "temperature",
        "+",
        "Temperature +1 SD",
    ),

    (
        "temperature",
        "-",
        "Temperature −1 SD",
    ),

    (
        "vapour_pressure",
        "+",
        "Vapour pressure +1 SD",
    ),

    (
        "vapour_pressure",
        "-",
        "Vapour pressure −1 SD",
    ),

    (
        "rainfall",
        "+",
        "Rainfall +1 SD",
    ),

    (
        "rainfall",
        "-",
        "Rainfall −1 SD",
    ),

]


# =============================================================================
# 6. DARK PRESENTATION COLOURS
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


# Alternating row background

COLOR_ROW = "#171B21"


# =============================================================================
# 7. TIMES NEW ROMAN STYLE
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
# 8. COMMON AXIS STYLE
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
# 9. LOAD AND VALIDATE POSTERIOR RESULTS
# =============================================================================

def load_posterior_results(
    input_csv: Path,
    scenario_order: list,
) -> tuple[pd.DataFrame, int]:

    # -------------------------------------------------------------------------
    # Check file
    # -------------------------------------------------------------------------

    if not input_csv.exists():

        raise FileNotFoundError(
            f"Posterior results not found:\n{input_csv}\n\n"
            "Run run_climate_perturbation_analysis.py first."
        )

    print(
        f"\nLoading posterior results:\n{input_csv}"
    )

    df = pd.read_csv(
        input_csv
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
    # Read posterior draw count
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

    for covariate, direction, _ in scenario_order:

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
# 10. FOREST PLOT
# =============================================================================

def draw_forest_plot(
    ax,
    df: pd.DataFrame,
    scenario_order: list,
) -> None:

    y_positions = np.arange(
        len(scenario_order)
    )

    offset = 0.16

    # =========================================================================
    # ALTERNATING ROW SHADING
    # =========================================================================

    for i in range(
        len(scenario_order)
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
        scenario_order
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
            in scenario_order
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

    relevant_rows = []

    for covariate, direction, _ in scenario_order:

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
        len(scenario_order) - 0.5,
        -0.5
    )

    # =========================================================================
    # STYLE
    # =========================================================================

    style_axis(
        ax
    )


# =============================================================================
# 11. LEGEND
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
# 12. SAVE FIGURE
# =============================================================================

def save_figure(
    fig,
    png_path: Path,
    pdf_path: Path,
) -> None:

    # -------------------------------------------------------------------------
    # High-resolution PNG
    # -------------------------------------------------------------------------

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )

    # -------------------------------------------------------------------------
    # Vector PDF
    # -------------------------------------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=BG
    )

    print(
        f"[OK] Saved {png_path}"
    )

    print(
        f"[OK] Saved {pdf_path}"
    )


# =============================================================================
# 13. GENERATE ONE FOREST PLOT
# =============================================================================

def generate_forest_plot(
    input_csv: Path,
    scenario_order: list,
    title: str,
    subtitle_prefix: str,
    png_path: Path,
    pdf_path: Path,
) -> None:

    print("\n" + "=" * 80)

    print(title)

    print("=" * 80)

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    df, n_draws = load_posterior_results(
        input_csv,
        scenario_order
    )

    print(
        f"\nPosterior draws per scenario: {n_draws:,}"
    )

    print(
        f"Number of plotted scenarios: {len(scenario_order)}"
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
        df,
        scenario_order
    )

    # =========================================================================
    # TITLE
    # =========================================================================

    ax.set_title(
        title,
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
            f"{subtitle_prefix}  |  "
            "Posterior median and 95% credible interval  "
            f"({n_draws:,} posterior draws)"
        ),

        transform=ax.transAxes,

        ha="center",

        va="bottom",

        fontsize=11.5,

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
        fig,
        png_path,
        pdf_path
    )

    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:

    print("\n==========================================")
    print("BOMBAY PLAGUE")
    print("CLIMATE-PERTURBATION FOREST PLOTS")
    print("DARK PRESENTATION — TIMES NEW ROMAN")
    print("==========================================")

    print(
        f"\nInput directory: {SCENARIO_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    # =========================================================================
    # FIGURE 1: RAW-UNIT PERTURBATIONS
    # =========================================================================

    generate_forest_plot(

        input_csv=INPUT_RAW_CSV,

        scenario_order=SCENARIO_ORDER_RAW,

        title=(
            "Climate perturbation effects on transmission coefficients"
        ),

        subtitle_prefix=(
            "Raw-unit climate perturbations"
        ),

        png_path=FIG_RAW_PNG,

        pdf_path=FIG_RAW_PDF,

    )

    # =========================================================================
    # FIGURE 2: +/- 1 SD PERTURBATIONS
    # =========================================================================

    generate_forest_plot(

        input_csv=INPUT_1SD_CSV,

        scenario_order=SCENARIO_ORDER_1SD,

        title=(
            "Standardized climate perturbation effects "
            "on transmission coefficients"
        ),

        subtitle_prefix=(
            "Each climate covariate shifted by ±1 SD"
        ),

        png_path=FIG_1SD_PNG,

        pdf_path=FIG_1SD_PDF,

    )

    print("\n==========================================")
    print("BOTH FIGURES GENERATED SUCCESSFULLY")
    print("==========================================")

    print(
        f"\nSaved in: {OUTPUT_DIR}"
    )

    print("\nGenerated figures:")

    print(
        f"  1. {FIG_RAW_PNG.name}"
    )

    print(
        f"  2. {FIG_RAW_PDF.name}"
    )

    print(
        f"  3. {FIG_1SD_PNG.name}"
    )

    print(
        f"  4. {FIG_1SD_PDF.name}"
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()