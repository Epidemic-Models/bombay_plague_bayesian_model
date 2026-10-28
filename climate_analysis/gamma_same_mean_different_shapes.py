#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gamma distributions with the same mean but different shapes.

DARK PRESENTATION VERSION — TIMES NEW ROMAN
============================================

Illustrates that different Gamma(shape, scale) combinations
can have the same mean infection-to-death delay.

All distributions satisfy:

    mean = shape * scale = 1.2 weeks

Selected model distribution:

    Gamma(shape=2.0, scale=0.6 weeks)

The x-axis extends to 12 weeks to illustrate the maximum
lag used in the observation model.

STYLE
-----
- Dark background: #0B0D10
- Times New Roman font
- STIX mathematical notation
- White text
- Bright contrasting curves
- Golden orange highlighting the selected model

OUTPUT
------
climate_analysis/figures/
    gamma_same_mean_different_shapes_dark.pdf
    gamma_same_mean_different_shapes_dark.png

The script should be located inside:

    bombay_plague_bayesian_model/climate_analysis/
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma
from matplotlib import font_manager


# =============================================================================
# 1. PATHS
# =============================================================================

# Directory containing this Python script
SCRIPT_DIR = Path(__file__).resolve().parent

# If script is directly inside climate_analysis
if SCRIPT_DIR.name == "climate_analysis":

    CLIMATE_DIR = SCRIPT_DIR

# If script is inside climate_analysis/scripts
elif SCRIPT_DIR.parent.name == "climate_analysis":

    CLIMATE_DIR = SCRIPT_DIR.parent

else:

    CLIMATE_DIR = SCRIPT_DIR


# Save figures inside climate_analysis/figures
FIG_DIR = CLIMATE_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_NAME = "gamma_same_mean_different_shapes_dark"


# =============================================================================
# 2. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#FFFFFF"

GRID = "#414750"

SPINE = "#D1D5DB"

MEAN_COLOR = "#FFFFFF"

MAX_LAG_COLOR = "#B0B7C3"


# =============================================================================
# 3. TIMES NEW ROMAN FONT SETTINGS
# =============================================================================

# Check whether Times New Roman is available on this computer.

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

    # Mathematical notation matching Times New Roman
    "mathtext.fontset": "stix",

    # -------------------------------------------------------------------------
    # FONT SIZES
    # -------------------------------------------------------------------------

    "font.size": 13,

    "axes.labelsize": 14,

    "axes.titlesize": 17,

    "legend.fontsize": 12,

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    # -------------------------------------------------------------------------
    # LINES AND AXES
    # -------------------------------------------------------------------------

    "lines.linewidth": 2.2,

    "axes.linewidth": 1.0,

    # -------------------------------------------------------------------------
    # DARK BACKGROUND
    # -------------------------------------------------------------------------

    "figure.facecolor": BG,

    "axes.facecolor": BG,

    "savefig.facecolor": BG,

    # -------------------------------------------------------------------------
    # WHITE TEXT
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
    # PDF FONT EMBEDDING
    # -------------------------------------------------------------------------

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})


# =============================================================================
# 4. GAMMA DISTRIBUTIONS
# =============================================================================

# Every combination satisfies:
#
#       shape × scale = 1.2 weeks
#
# Therefore all distributions have the same mean,
# but different variances and shapes.

parameter_sets = [

    (1.0, 1.20),

    (2.0, 0.60),   # Selected model

    (3.0, 0.40),

    (6.0, 0.20),

]


# Bright colours suitable for a dark background

colors = [

    "#60A5FA",    # Bright blue

    "#FFB000",    # Golden orange: selected model

    "#00C896",    # Bright green

    "#C084FC",    # Bright purple

]


# =============================================================================
# 5. MAIN
# =============================================================================

def main():

    # =========================================================================
    # CONTINUOUS TIME AXIS
    # =========================================================================

    x = np.linspace(
        0.001,
        12.0,
        4000
    )

    # =========================================================================
    # FIGURE
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(13.5, 5.5),
        dpi=300
    )

    fig.patch.set_facecolor(BG)

    ax.set_facecolor(BG)

    # =========================================================================
    # PLOT GAMMA DISTRIBUTIONS
    # =========================================================================

    print("\n==========================================")
    print("GAMMA DISTRIBUTIONS")
    print("DARK PRESENTATION — TIMES NEW ROMAN")
    print("==========================================")

    for (k, theta), color in zip(
        parameter_sets,
        colors
    ):

        y = gamma.pdf(
            x,
            a=k,
            scale=theta
        )

        mean = k * theta

        variance = k * theta**2

        sd = np.sqrt(variance)

        # ---------------------------------------------------------------------
        # HIGHLIGHT SELECTED MODEL
        # ---------------------------------------------------------------------

        if k == 2.0 and theta == 0.60:

            label = (
                rf"$k={k:.0f},\ \theta={theta:.2f}$ "
                "(model choice)"
            )

            linewidth = 3.5

            zorder = 6

        else:

            label = (
                rf"$k={k:.0f},\ \theta={theta:.2f}$"
            )

            linewidth = 2.3

            zorder = 4

        # ---------------------------------------------------------------------
        # PLOT CURVE
        # ---------------------------------------------------------------------

        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=zorder
        )

        print(
            f"k = {k:4.1f}, "
            f"theta = {theta:4.2f}, "
            f"mean = {mean:4.2f} weeks, "
            f"SD = {sd:4.2f} weeks"
        )

    # =========================================================================
    # COMMON MEAN
    # =========================================================================

    ax.axvline(
        1.2,
        color=MEAN_COLOR,
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label="Common mean = 1.2 weeks",
        zorder=3
    )

    # =========================================================================
    # MAXIMUM LAG
    # =========================================================================

    ax.axvline(
        12.0,
        color=MAX_LAG_COLOR,
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
        label="Maximum lag = 12 weeks",
        zorder=3
    )

    # =========================================================================
    # AXIS LABELS
    # =========================================================================

    ax.set_xlabel(
        "Infection-to-death delay (weeks)",
        fontsize=15,
        color=TEXT,
        labelpad=12
    )

    ax.set_ylabel(
        "Probability density",
        fontsize=15,
        color=TEXT,
        labelpad=12
    )

    # =========================================================================
    # TITLE
    # =========================================================================

    ax.set_title(
        "Different Gamma distributions can have the same mean",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        pad=22
    )

    # =========================================================================
    # AXIS LIMITS
    # =========================================================================

    ax.set_xlim(
        0,
        12.5
    )

    ax.set_ylim(
        bottom=0
    )

    # =========================================================================
    # AXIS STYLE
    # =========================================================================

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE)

    ax.spines["bottom"].set_color(SPINE)

    ax.spines["left"].set_linewidth(1.1)

    ax.spines["bottom"].set_linewidth(1.1)

    ax.tick_params(
        axis="both",
        colors=TEXT,
        labelsize=12,
        direction="out",
        length=5,
        width=1.0
    )

    # =========================================================================
    # GRID
    # =========================================================================

    ax.grid(
        True,
        color=GRID,
        linewidth=0.7,
        alpha=0.35,
        linestyle="--",
        zorder=0
    )

    ax.set_axisbelow(True)

    # =========================================================================
    # LEGEND
    # =========================================================================

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=12,
        handlelength=3.2,
        labelspacing=1.1,
        frameon=True,
        facecolor="#171B21",
        edgecolor="#414750",
        framealpha=0.95
    )

    for text in legend.get_texts():

        text.set_color(TEXT)

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.70,
        bottom=0.17,
        top=0.86
    )

    # =========================================================================
    # PROBABILITY BEYOND 12 WEEKS
    # =========================================================================

    print("\n==========================================")
    print("PROBABILITY BEYOND 12 WEEKS")
    print("==========================================")

    for k, theta in parameter_sets:

        tail_probability = gamma.sf(
            12.0,
            a=k,
            scale=theta
        )

        probability_before_12 = gamma.cdf(
            12.0,
            a=k,
            scale=theta
        )

        print(
            f"k={k:4.1f}, theta={theta:4.2f}: "
            f"P(X <= 12) = {probability_before_12:.10f}, "
            f"P(X > 12) = {tail_probability:.3e}"
        )

    # =========================================================================
    # MODEL CHOICE SUMMARY
    # =========================================================================

    k_model = 2.0

    theta_model = 0.60

    mean_model = k_model * theta_model

    variance_model = k_model * theta_model**2

    sd_model = np.sqrt(variance_model)

    mode_model = (k_model - 1) * theta_model

    print("\n==========================================")
    print("MODEL CHOICE: k = 2, theta = 0.6")
    print("==========================================")

    print(
        f"Mean     = {mean_model:.3f} weeks"
    )

    print(
        f"Mode     = {mode_model:.3f} weeks"
    )

    print(
        f"SD       = {sd_model:.3f} weeks"
    )

    print(
        f"Variance = {variance_model:.3f} weeks^2"
    )

    print(
        f"P(X > 12 weeks) = "
        f"{gamma.sf(12.0, a=k_model, scale=theta_model):.3e}"
    )

    # =========================================================================
    # SAVE FIGURES
    # =========================================================================

    pdf_path = (
        FIG_DIR
        / f"{OUT_NAME}.pdf"
    )

    png_path = (
        FIG_DIR
        / f"{OUT_NAME}.png"
    )

    # -------------------------------------------------------------------------
    # VECTOR PDF
    # -------------------------------------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=BG
    )

    # -------------------------------------------------------------------------
    # HIGH-RESOLUTION PNG
    # -------------------------------------------------------------------------

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )

    plt.close(fig)

    # =========================================================================
    # CONFIRM OUTPUT
    # =========================================================================

    print("\n==========================================")
    print("FIGURES SAVED SUCCESSFULLY")
    print("==========================================")

    print(f"\nFigures directory: {FIG_DIR}")

    print(f"\nPDF: {pdf_path}")

    print(f"PNG: {png_path}")


# =============================================================================
# 6. RUN
# =============================================================================

if __name__ == "__main__":

    main()