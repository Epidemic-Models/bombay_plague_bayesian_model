#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bombay weather analysis, 1896–1899.

DARK PRESENTATION VERSION — TIMES NEW ROMAN

Generates:
    1. Weather distributions (3 panels)
    2. Weather correlations (3 panels)
    3. Combined weather figure (6 panels)

All figures use:
    - Dark background (#0B0D10)
    - Times New Roman
    - White text
    - Bright contrasting colours

Outputs:
    climate_analysis/figures/
        bombay_weather_distributions_dark.png/pdf
        bombay_weather_correlations_dark.png/pdf
        bombay_weather_combined_dark.png/pdf

The script is intended to be located inside:

    bombay_plague_bayesian_model/climate_analysis/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager


# =============================================================================
# 1. PATHS
# =============================================================================

# Directory containing this Python script
SCRIPT_DIR = Path(__file__).resolve().parent

# Input weather data
WEATHER_CSV = (
    SCRIPT_DIR
    / "bombay_weather_daily_1896_1899.csv"
)

# Output directory
FIG_DIR = SCRIPT_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. COLOURS
# =============================================================================

# Dark presentation background
BG = "#0B0D10"

# Text and axes
TEXT = "#FFFFFF"
GRID = "#414750"
SPINE = "#D1D5DB"

# Histograms
C_T = "#FFB000"       # Temperature: orange
C_V = "#00C896"       # Vapour pressure: green
C_R = "#00BFFF"       # Rainfall: blue

# Scatter plots
C_TV = "#B58BE8"      # Temperature vs vapour pressure: purple
C_TR = "#FF8C42"      # Temperature vs rainfall: orange
C_VR = "#38BDF8"      # Vapour pressure vs rainfall: blue

# Reference lines
MEDIAN_COLOR = "#FFFFFF"
PERCENTILE_COLOR = "#B0B7C3"


# =============================================================================
# 3. GLOBAL STYLE — TIMES NEW ROMAN
# =============================================================================

# Check whether Times New Roman is available.
# This does not change any font files on your computer.

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
        "Matplotlib may substitute another serif font."
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

    # Mathematical notation matching Times
    "mathtext.fontset": "stix",

    # -------------------------------------------------------------------------
    # FONT SIZES
    # -------------------------------------------------------------------------

    "font.size": 12,

    "axes.labelsize": 12,

    "axes.titlesize": 14,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    # -------------------------------------------------------------------------
    # AXES
    # -------------------------------------------------------------------------

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
# 4. LOAD DATA
# =============================================================================

def load_weather():
    """
    Load daily Bombay weather observations.

    Convert temperature from Fahrenheit to Celsius.
    """

    if not WEATHER_CSV.exists():

        raise FileNotFoundError(
            "Weather CSV not found:\n"
            f"{WEATHER_CSV}\n\n"
            "Place bombay_weather_daily_1896_1899.csv "
            "inside the climate_analysis directory."
        )

    df = pd.read_csv(
        WEATHER_CSV
    )

    # Fahrenheit to Celsius
    df["Temperature_C"] = (
        df["Temperature_F"] - 32.0
    ) * 5.0 / 9.0

    return df


# =============================================================================
# 5. COMMON AXIS STYLE
# =============================================================================

def style_axis(ax):
    """
    Apply the common dark presentation style.
    """

    ax.set_facecolor(BG)

    # Remove top and right borders
    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    # Style remaining borders
    ax.spines["left"].set_color(SPINE)

    ax.spines["bottom"].set_color(SPINE)

    # Grid
    ax.grid(
        True,
        color=GRID,
        linewidth=0.65,
        alpha=0.35,
        linestyle="--"
    )

    ax.set_axisbelow(True)

    # Tick marks and labels
    ax.tick_params(
        axis="both",
        colors=TEXT,
        direction="out",
        length=4,
        width=0.8
    )


# =============================================================================
# 6. HISTOGRAM FUNCTION
# =============================================================================

def plot_histogram(
    ax,
    data,
    letter,
    title,
    xlabel,
    color,
    rainfall=False,
):
    """
    Plot a weather-variable distribution.

    Reference lines:
        Median: white dashed line
        5th percentile: grey dotted line
        95th percentile: grey dotted line

    For rainfall, the 5th-percentile line is omitted
    because many observations have zero rainfall.
    """

    x = pd.Series(data).dropna()

    # -------------------------------------------------------------------------
    # HISTOGRAM
    # -------------------------------------------------------------------------

    ax.hist(
        x,
        bins=30,
        color=color,
        alpha=0.90,
        edgecolor=BG,
        linewidth=0.5,
        zorder=3
    )

    # -------------------------------------------------------------------------
    # SUMMARY STATISTICS
    # -------------------------------------------------------------------------

    median = x.median()

    q05 = x.quantile(0.05)

    q95 = x.quantile(0.95)

    # -------------------------------------------------------------------------
    # MEDIAN
    # -------------------------------------------------------------------------

    ax.axvline(
        median,
        color=MEDIAN_COLOR,
        linestyle="--",
        linewidth=1.7,
        zorder=5
    )

    # -------------------------------------------------------------------------
    # 5TH PERCENTILE
    # -------------------------------------------------------------------------

    if not rainfall:

        ax.axvline(
            q05,
            color=PERCENTILE_COLOR,
            linestyle=":",
            linewidth=1.3,
            zorder=5
        )

    # -------------------------------------------------------------------------
    # 95TH PERCENTILE
    # -------------------------------------------------------------------------

    ax.axvline(
        q95,
        color=PERCENTILE_COLOR,
        linestyle=":",
        linewidth=1.3,
        zorder=5
    )

    # -------------------------------------------------------------------------
    # TITLE
    # -------------------------------------------------------------------------

    ax.set_title(
        f"{letter}   {title}",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    # -------------------------------------------------------------------------
    # X-AXIS
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        labelpad=9
    )

    style_axis(ax)

    # -------------------------------------------------------------------------
    # RAINFALL ANNOTATION
    # -------------------------------------------------------------------------

    if rainfall:

        n = len(x)

        n_zero = int(
            (x == 0).sum()
        )

        pct_zero = (
            100 * n_zero / n
        )

        ax.text(
            0.97,
            0.95,

            f"Zero rainfall: {pct_zero:.1f}%\n"
            f"(n = {n_zero:,})",

            transform=ax.transAxes,

            ha="right",
            va="top",

            fontsize=10,

            color=TEXT,

            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#222831",
                edgecolor="#6B7280",
                linewidth=0.8,
                alpha=0.95
            ),

            zorder=10
        )


# =============================================================================
# 7. SCATTER FUNCTION
# =============================================================================

def plot_scatter(
    ax,
    x,
    y,
    letter,
    title,
    xlabel,
    ylabel,
    color,
):
    """
    Plot the relationship between two weather variables.

    Calculate:
        Pearson correlation
        Spearman correlation

    Both correlations are displayed inside the panel.
    """

    # -------------------------------------------------------------------------
    # REMOVE MISSING PAIRS
    # -------------------------------------------------------------------------

    pair = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    # -------------------------------------------------------------------------
    # CORRELATIONS
    # -------------------------------------------------------------------------

    pearson = pair["x"].corr(
        pair["y"],
        method="pearson"
    )

    spearman = pair["x"].corr(
        pair["y"],
        method="spearman"
    )

    # -------------------------------------------------------------------------
    # SCATTER PLOT
    # -------------------------------------------------------------------------

    ax.scatter(
        pair["x"],
        pair["y"],
        s=13,
        alpha=0.45,
        color=color,
        edgecolors="none",
        rasterized=True,
        zorder=3
    )

    # -------------------------------------------------------------------------
    # TITLE
    # -------------------------------------------------------------------------

    ax.set_title(
        f"{letter}   {title}",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    # -------------------------------------------------------------------------
    # AXIS LABELS
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        labelpad=9
    )

    ax.set_ylabel(
        ylabel,
        labelpad=9
    )

    style_axis(ax)

    # -------------------------------------------------------------------------
    # CORRELATION ANNOTATION
    # -------------------------------------------------------------------------

    ax.text(
        0.04,
        0.96,

        f"Pearson $r$ = {pearson:.2f}\n"
        f"Spearman $\\rho$ = {spearman:.2f}",

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=10,

        color=TEXT,

        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#222831",
            edgecolor="#6B7280",
            linewidth=0.6,
            alpha=0.90
        ),

        zorder=10
    )

    return pearson, spearman


# =============================================================================
# 8. SAVE FUNCTION
# =============================================================================

def save_figure(fig, filename):
    """
    Save a figure in both PNG and PDF formats.

    PNG:
        600 dpi, suitable for PowerPoint.

    PDF:
        Vector output, suitable for publication.
    """

    png_path = (
        FIG_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIG_DIR
        / f"{filename}.pdf"
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

    # -------------------------------------------------------------------------
    # VECTOR PDF
    # -------------------------------------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=BG
    )

    print(f"[OK] Saved: {png_path}")

    print(f"[OK] Saved: {pdf_path}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():

    # =========================================================================
    # LOAD WEATHER DATA
    # =========================================================================

    df = load_weather()

    T = df["Temperature_C"]

    V = df["VapourPressure_inHg"]

    R = df["Rain_in"]

    print("\n==========================================")
    print("BOMBAY WEATHER ANALYSIS")
    print("1896–1899")
    print("DARK PRESENTATION — TIMES NEW ROMAN")
    print("==========================================")

    print(f"\nWeather data: {WEATHER_CSV}")

    print(f"Figure directory: {FIG_DIR}")

    print(f"Number of observations: {len(df)}")

    # =========================================================================
    # PRINT CORRELATIONS
    # =========================================================================

    climate = pd.DataFrame({

        "Temperature": T,

        "Vapour pressure": V,

        "Rainfall": R

    }).dropna()

    print("\nPEARSON CORRELATIONS")

    print(
        climate.corr(
            method="pearson"
        ).round(3)
    )

    print("\nSPEARMAN CORRELATIONS")

    print(
        climate.corr(
            method="spearman"
        ).round(3)
    )

    # =========================================================================
    # FIGURE 1: DISTRIBUTIONS
    # =========================================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        dpi=300
    )

    # Temperature
    plot_histogram(
        axes[0],
        T,
        "A",
        "Temperature",
        "Temperature (°C)",
        C_T
    )

    # Vapour pressure
    plot_histogram(
        axes[1],
        V,
        "B",
        "Vapour pressure",
        "Vapour pressure (inHg)",
        C_V
    )

    # Rainfall
    plot_histogram(
        axes[2],
        R,
        "C",
        "Rainfall",
        "Daily rainfall (in)",
        C_R,
        rainfall=True
    )

    axes[0].set_ylabel(
        "Number of days"
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.19,
        top=0.86,
        wspace=0.30
    )

    save_figure(
        fig,
        "bombay_weather_distributions_dark"
    )

    plt.close(fig)

    # =========================================================================
    # FIGURE 2: CORRELATIONS
    # =========================================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        dpi=300
    )

    # Temperature vs vapour pressure
    plot_scatter(
        axes[0],
        T,
        V,
        "A",
        "Temperature vs vapour pressure",
        "Temperature (°C)",
        "Vapour pressure (inHg)",
        C_TV
    )

    # Temperature vs rainfall
    plot_scatter(
        axes[1],
        T,
        R,
        "B",
        "Temperature vs rainfall",
        "Temperature (°C)",
        "Daily rainfall (in)",
        C_TR
    )

    # Vapour pressure vs rainfall
    plot_scatter(
        axes[2],
        V,
        R,
        "C",
        "Vapour pressure vs rainfall",
        "Vapour pressure (inHg)",
        "Daily rainfall (in)",
        C_VR
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.19,
        top=0.86,
        wspace=0.32
    )

    save_figure(
        fig,
        "bombay_weather_correlations_dark"
    )

    plt.close(fig)

    # =========================================================================
    # FIGURE 3: COMBINED 6-PANEL FIGURE
    # =========================================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        dpi=300
    )

    # -------------------------------------------------------------------------
    # TOP ROW: DISTRIBUTIONS
    # -------------------------------------------------------------------------

    # A: Temperature
    plot_histogram(
        axes[0, 0],
        T,
        "A",
        "Temperature",
        "Temperature (°C)",
        C_T
    )

    # B: Vapour pressure
    plot_histogram(
        axes[0, 1],
        V,
        "B",
        "Vapour pressure",
        "Vapour pressure (inHg)",
        C_V
    )

    # C: Rainfall
    plot_histogram(
        axes[0, 2],
        R,
        "C",
        "Rainfall",
        "Daily rainfall (in)",
        C_R,
        rainfall=True
    )

    axes[0, 0].set_ylabel(
        "Number of days"
    )

    # -------------------------------------------------------------------------
    # BOTTOM ROW: CORRELATIONS
    # -------------------------------------------------------------------------

    # D: Temperature vs vapour pressure
    plot_scatter(
        axes[1, 0],
        T,
        V,
        "D",
        "Temperature vs vapour pressure",
        "Temperature (°C)",
        "Vapour pressure (inHg)",
        C_TV
    )

    # E: Temperature vs rainfall
    plot_scatter(
        axes[1, 1],
        T,
        R,
        "E",
        "Temperature vs rainfall",
        "Temperature (°C)",
        "Daily rainfall (in)",
        C_TR
    )

    # F: Vapour pressure vs rainfall
    plot_scatter(
        axes[1, 2],
        V,
        R,
        "F",
        "Vapour pressure vs rainfall",
        "Vapour pressure (inHg)",
        "Daily rainfall (in)",
        C_VR
    )

    # -------------------------------------------------------------------------
    # LAYOUT
    # -------------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.10,
        top=0.92,
        wspace=0.32,
        hspace=0.48
    )

    save_figure(
        fig,
        "bombay_weather_combined_dark"
    )

    plt.close(fig)

    # =========================================================================
    # FINISHED
    # =========================================================================

    print("\n==========================================")
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("==========================================")

    print(f"\nSaved in: {FIG_DIR}")

    print("\nGenerated figures:")

    print("1. bombay_weather_distributions_dark.png/pdf")

    print("2. bombay_weather_correlations_dark.png/pdf")

    print("3. bombay_weather_combined_dark.png/pdf")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()