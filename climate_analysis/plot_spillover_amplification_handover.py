#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# plot_spillover_amplification_handover.py
# =============================================================================
"""
SPILLOVER–AMPLIFICATION HANDOVER
DARK PRESENTATION VERSION — TIMES NEW ROMAN
==========================================

Three stacked panels on a shared weekly time axis:

(a) Observed weekly plague deaths.

(b) Two forces of infection:
        Reservoir-associated spillover:
            beta_R(t)

        Prevalence-associated amplification:
            beta_H(t) * I(t) / N

(c) Amplification reproduction number:

        R0_H(t) = beta_H(t) / gamma

    with its 95% credible interval and threshold R0_H = 1.

Ascending and declining epidemic phases are shaded behind
all three panels.

IMPORTANT
---------
This script only visualizes existing posterior results.
No Bayesian model is refitted.

INPUT
-----
results/final_figures/transmission_dynamics/
    posterior_transmission_dynamics_all_climate.csv

OUTPUT
------
climate_analysis/figures/
    fig_spillover_amplification_handover_dark.png
    fig_spillover_amplification_handover_dark.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib import font_manager


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

# Current directory:
# bombay_plague_bayesian_model/climate_analysis/

CLIMATE_DIR = Path(__file__).resolve().parent

# Main project:
# bombay_plague_bayesian_model/

PROJECT_ROOT = CLIMATE_DIR.parent

# Original analysis scripts:
# bombay_plague_bayesian_model/scripts/

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Presentation figures:
# bombay_plague_bayesian_model/climate_analysis/figures/

FIG_DIR = CLIMATE_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_NAME = "fig_spillover_amplification_handover_dark"


# =============================================================================
# 2. IMPORT ORIGINAL WAVE ANALYSIS FUNCTIONS
# =============================================================================

WAVE_SCRIPT = (
    SCRIPTS_DIR
    / "plot_wave_phases_and_R0_criteria.py"
)

if not WAVE_SCRIPT.exists():

    raise FileNotFoundError(
        "Cannot find the original wave-analysis script:\n"
        f"{WAVE_SCRIPT}\n\n"
        "Check that plot_wave_phases_and_R0_criteria.py "
        "is inside the project's scripts directory."
    )

sys.path.insert(
    0,
    str(SCRIPTS_DIR)
)

from plot_wave_phases_and_R0_criteria import (  # noqa: E402
    DEATHS_COL,
    COLOR_ASCENDING,
    COLOR_DECLINING,
    detect_waves,
    cross_check_boundaries,
    shade_phases,
    format_time_axis,
)


# =============================================================================
# 3. INPUT POSTERIOR RESULTS
# =============================================================================

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "posterior_transmission_dynamics_all_climate.csv"
)


# =============================================================================
# 4. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#F5F5F5"

GRID = "#414750"

SPINE = "#D1D5DB"


# Main data colours

COLOR_SPILLOVER = "#38BDF8"       # Bright blue

COLOR_AMPLIFICATION = "#FFB000"   # Golden orange

COLOR_BARS = "#B8BDC7"            # Light grey


# Wave-phase colours

COLOR_ASCENDING_DARK = "#285443"

COLOR_DECLINING_DARK = "#493B56"


# Wave markers

COLOR_PEAK_DARK = "#FF6B6B"

COLOR_TROUGH_DARK = "#B0B7C3"


# Threshold

COLOR_THRESHOLD = "#FFFFFF"


# =============================================================================
# 5. TIMES NEW ROMAN FONT SETTINGS
# =============================================================================

# Verify that Times New Roman is installed.
# If it is not available, Matplotlib will use a fallback serif font.

try:

    times_font_path = font_manager.findfont(
        "Times New Roman",
        fallback_to_default=False
    )

    print(f"[OK] Times New Roman found: {times_font_path}")

except ValueError:

    print(
        "[WARNING] Times New Roman was not found. "
        "Matplotlib may substitute another serif font."
    )


plt.rcParams.update({

    # -------------------------------------------------------------------------
    # TIMES NEW ROMAN
    # -------------------------------------------------------------------------

    "font.family": "serif",

    "font.serif": ["Times New Roman"],

    # Mathematical notation matching Times New Roman
    "mathtext.fontset": "stix",

    "mathtext.rm": "Times New Roman",

    "mathtext.it": "STIXGeneral:italic",

    "mathtext.bf": "STIXGeneral:bold",

    # -------------------------------------------------------------------------
    # FONT SIZES
    # -------------------------------------------------------------------------

    "font.size": 13,

    "axes.labelsize": 14,

    "axes.titlesize": 16,

    "legend.fontsize": 11,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    # -------------------------------------------------------------------------
    # LINES
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
    # TEXT AND AXES
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

    "grid.alpha": 0.30,

    "grid.linewidth": 0.6,

    # -------------------------------------------------------------------------
    # PDF FONT EMBEDDING
    # -------------------------------------------------------------------------

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})


# =============================================================================
# 6. PANEL LABEL
# =============================================================================

def panel_label(ax, text: str) -> None:

    ax.text(
        0.01,
        0.96,
        text,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=15,

        fontweight="bold",

        fontfamily="Times New Roman",

        color=TEXT,

        bbox=dict(
            facecolor=BG,
            edgecolor="none",
            alpha=0.85,
            pad=3
        ),

        zorder=20
    )


# =============================================================================
# 7. DARK AXIS STYLE
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE)

    ax.spines["bottom"].set_color(SPINE)

    ax.spines["left"].set_linewidth(1.0)

    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        colors=TEXT,
        direction="out",
        length=5,
        width=1.0
    )

    ax.xaxis.label.set_color(TEXT)

    ax.yaxis.label.set_color(TEXT)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        color=GRID,
        alpha=0.30,
        zorder=0
    )

    ax.set_axisbelow(True)


# =============================================================================
# 8. WAVE PHASE SHADING
# =============================================================================

def shade_phases_dark(ax, df, peaks, bounds):
    """
    Reuse the original wave-detection and phase-shading logic.

    The original shade_phases() function determines the phase
    boundaries. We do not recalculate or change those boundaries.

    Newly created phase-shading artists are recoloured for
    the dark presentation.
    """

    old_collections = set(ax.collections)
    old_patches = set(ax.patches)
    old_lines = set(ax.lines)

    shade_phases(
        ax,
        df,
        peaks,
        bounds
    )

    new_collections = [
        artist
        for artist in ax.collections
        if artist not in old_collections
    ]

    new_patches = [
        artist
        for artist in ax.patches
        if artist not in old_patches
    ]

    def recolour_artist(artist):

        try:

            fc = artist.get_facecolor()

            fc = np.asarray(fc)

            if fc.size == 0:
                return

            rgba = fc.reshape(-1, 4)[0]

            asc_rgba = np.asarray(
                to_rgba(COLOR_ASCENDING)
            )

            dec_rgba = np.asarray(
                to_rgba(COLOR_DECLINING)
            )

            d_asc = np.linalg.norm(
                rgba[:3] - asc_rgba[:3]
            )

            d_dec = np.linalg.norm(
                rgba[:3] - dec_rgba[:3]
            )

            if d_asc <= d_dec:

                artist.set_facecolor(
                    COLOR_ASCENDING_DARK
                )

            else:

                artist.set_facecolor(
                    COLOR_DECLINING_DARK
                )

            artist.set_alpha(0.35)

        except (AttributeError, ValueError, TypeError):
            pass

    for artist in new_collections:

        recolour_artist(artist)

    for artist in new_patches:

        recolour_artist(artist)

    new_lines = [
        artist
        for artist in ax.lines
        if artist not in old_lines
    ]

    for line in new_lines:

        if line.get_linestyle() == "--":

            line.set_color(
                COLOR_PEAK_DARK
            )

        elif line.get_linestyle() == ":":

            line.set_color(
                COLOR_TROUGH_DARK
            )

        line.set_linewidth(1.2)


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:

    # =========================================================================
    # LOAD POSTERIOR RESULTS
    # =========================================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Posterior results not found:\n{INPUT_CSV}\n\n"
            "Run scripts/final_transmission_dynamics.py first."
        )

    df = pd.read_csv(
        INPUT_CSV,
        parse_dates=["week_ending"]
    )

    df = (
        df
        .sort_values("week_ending")
        .reset_index(drop=True)
    )

    print("\n==========================================")
    print("SPILLOVER–AMPLIFICATION HANDOVER")
    print("DARK PRESENTATION FIGURE")
    print("TIMES NEW ROMAN")
    print("==========================================")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Wave script: {WAVE_SCRIPT}")
    print(f"Posterior data: {INPUT_CSV}")
    print(f"Output directory: {FIG_DIR}")

    print(f"\nNumber of weeks: {len(df)}")

    print(
        f"Study period: "
        f"{df['week_ending'].min().date()} "
        f"to "
        f"{df['week_ending'].max().date()}"
    )

    # =========================================================================
    # WAVE DETECTION
    # =========================================================================

    peaks, bounds = detect_waves(df)

    cross_check_boundaries(
        df,
        bounds
    )

    t = df["week_ending"]

    # =========================================================================
    # FORCE OF INFECTION SCALING
    # =========================================================================

    scale = 1e4

    # =========================================================================
    # FIGURE
    # =========================================================================

    fig, (ax_d, ax_f, ax_r) = plt.subplots(

        3,
        1,

        figsize=(14.5, 10.5),

        sharex=True,

        dpi=300,

        gridspec_kw={
            "height_ratios": [0.9, 1.2, 1.0],
            "hspace": 0.18
        }
    )

    fig.patch.set_facecolor(BG)

    # =========================================================================
    # PANEL A: OBSERVED MORTALITY
    # =========================================================================

    shade_phases_dark(
        ax_d,
        df,
        peaks,
        bounds
    )

    ax_d.bar(
        t,
        df[DEATHS_COL],

        width=6,

        color=COLOR_BARS,

        alpha=0.90,

        edgecolor="none",

        zorder=3
    )

    ax_d.set_ylabel(
        "Weekly deaths",
        fontsize=14,
        labelpad=12
    )

    panel_label(
        ax_d,
        "a   Observed mortality"
    )

    style_axis(ax_d)

    format_time_axis(ax_d)

    # =========================================================================
    # PANEL B: SPILLOVER VS REALISED AMPLIFICATION
    # =========================================================================

    shade_phases_dark(
        ax_f,
        df,
        peaks,
        bounds
    )

    # Spillover 95% credible interval
    ax_f.fill_between(
        t,

        scale * df["spillover_q025"],

        scale * df["spillover_q975"],

        color=COLOR_SPILLOVER,

        alpha=0.20,

        linewidth=0,

        zorder=2
    )

    # Spillover posterior median
    ax_f.plot(
        t,

        scale * df["spillover_median"],

        color=COLOR_SPILLOVER,

        linewidth=2.4,

        zorder=4
    )

    # Amplification 95% credible interval
    ax_f.fill_between(
        t,

        scale * df["human_q025"],

        scale * df["human_q975"],

        color=COLOR_AMPLIFICATION,

        alpha=0.23,

        linewidth=0,

        zorder=2
    )

    # Amplification posterior median
    ax_f.plot(
        t,

        scale * df["human_median"],

        color=COLOR_AMPLIFICATION,

        linewidth=2.4,

        zorder=5
    )

    ax_f.set_ylabel(
        "Force of infection\n(per 10,000 per week)",
        fontsize=14,
        labelpad=12
    )

    ax_f.set_ylim(bottom=0)

    panel_label(
        ax_f,
        "b   Spillover vs realised amplification"
    )

    style_axis(ax_f)

    format_time_axis(ax_f)

    # =========================================================================
    # PANEL C: AMPLIFICATION POTENTIAL
    # =========================================================================

    shade_phases_dark(
        ax_r,
        df,
        peaks,
        bounds
    )

    # 95% credible interval
    ax_r.fill_between(
        t,

        df["R0_H_q025"],

        df["R0_H_q975"],

        color=COLOR_AMPLIFICATION,

        alpha=0.23,

        linewidth=0,

        zorder=2
    )

    # Posterior median
    ax_r.plot(
        t,

        df["R0_H_median"],

        color=COLOR_AMPLIFICATION,

        linewidth=2.4,

        zorder=4
    )

    # Threshold R0_H = 1
    ax_r.axhline(
        1.0,

        color=COLOR_THRESHOLD,

        linestyle="--",

        linewidth=1.6,

        alpha=0.90,

        zorder=3
    )

    ax_r.set_ylabel(
        r"$R_{0,H}(t)$",
        fontsize=15,
        labelpad=12
    )

    ax_r.set_ylim(bottom=0)

    panel_label(
        ax_r,
        r"c   Amplification potential: "
        r"$R_{0,H}(t)=\beta_H(t)/\gamma$"
    )

    style_axis(ax_r)

    format_time_axis(ax_r)

    # =========================================================================
    # SHARED X-AXIS
    # =========================================================================

    ax_r.set_xlabel(
        "Week",
        fontsize=15,
        labelpad=12
    )

    ax_d.tick_params(labelbottom=False)

    ax_f.tick_params(labelbottom=False)

    # =========================================================================
    # LEGEND
    # =========================================================================

    handles = [

        Patch(
            facecolor=COLOR_BARS,
            label="Observed deaths"
        ),

        Line2D(
            [0], [0],
            color=COLOR_SPILLOVER,
            linewidth=2.5,
            label=r"Spillover $\beta_R(t)$"
        ),

        Line2D(
            [0], [0],
            color=COLOR_AMPLIFICATION,
            linewidth=2.5,
            label=r"Amplification $\beta_H(t)I(t)/N$"
        ),

        Line2D(
            [0], [0],
            color=COLOR_THRESHOLD,
            linestyle="--",
            linewidth=1.6,
            label=r"$R_{0,H}=1$"
        ),

        Patch(
            facecolor=COLOR_ASCENDING_DARK,
            alpha=0.60,
            label="Ascending phase"
        ),

        Patch(
            facecolor=COLOR_DECLINING_DARK,
            alpha=0.60,
            label="Declining phase"
        ),

        Line2D(
            [0], [0],
            color=COLOR_PEAK_DARK,
            linestyle="--",
            linewidth=1.3,
            label="Mortality peak"
        ),

        Line2D(
            [0], [0],
            color=COLOR_TROUGH_DARK,
            linestyle=":",
            linewidth=1.3,
            label="Wave boundary"
        ),

    ]

    legend = fig.legend(
        handles=handles,

        loc="lower center",

        ncol=4,

        fontsize=11,

        frameon=False,

        bbox_to_anchor=(0.5, 0.005),

        columnspacing=1.8,

        handlelength=2.8,

        labelspacing=1.2,

        prop={
            "family": "Times New Roman",
            "size": 11
        }
    )

    for text in legend.get_texts():
        text.set_color(TEXT)

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.97,
        bottom=0.19,
        hspace=0.22
    )

    # =========================================================================
    # SAVE FIGURES
    # =========================================================================

    pdf_path = FIG_DIR / f"{OUT_NAME}.pdf"

    png_path = FIG_DIR / f"{OUT_NAME}.png"

    # Vector PDF
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=BG
    )

    # High-resolution PNG
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )

    plt.close(fig)

    print("\n==========================================")
    print("FIGURES SAVED SUCCESSFULLY")
    print("==========================================")

    print(f"\nPDF: {pdf_path}")
    print(f"PNG: {png_path}")


# =============================================================================
# 10. RUN
# =============================================================================

if __name__ == "__main__":
    main()