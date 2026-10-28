#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot distributions of Bombay weather variables, 1896–1899.

Panels:
    A. Temperature
    B. Vapour pressure
    C. Rainfall

Lines:
    Dashed black = median
    Dotted grey  = 5th and 95th percentiles

For rainfall, the 5th percentile is not drawn because both
the 5th percentile and median are zero.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

WEATHER_CSV = "bombay_weather_daily_1896_1899.csv"

FIG_DIR = "figures"
OUT_NAME = "bombay_weather_distributions"

os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 2. GLOBAL PLOT STYLE
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
    "lines.markersize": 4,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})


# =============================================================================
# 3. COLORS
# =============================================================================

C_T = "#D55E00"   # Temperature: orange/red
C_V = "#009E73"   # Vapour pressure: green
C_R = "#0072B2"   # Rainfall: blue


# =============================================================================
# 4. LOAD DATA
# =============================================================================

def load_weather(path):

    df = pd.read_csv(path)

    # Convert Fahrenheit to Celsius
    df["Temperature_C"] = (
        df["Temperature_F"] - 32.0
    ) * 5.0 / 9.0

    return df


# =============================================================================
# 5. SUMMARY FUNCTION
# =============================================================================

def describe_variable(x, name):

    x = pd.Series(x).dropna()

    print(f"\n{name}")
    print("-" * len(name))
    print(f"N          = {len(x)}")
    print(f"Minimum    = {x.min():.3f}")
    print(f"5th pct    = {x.quantile(0.05):.3f}")
    print(f"Median     = {x.median():.3f}")
    print(f"95th pct   = {x.quantile(0.95):.3f}")
    print(f"Maximum    = {x.max():.3f}")
    print(f"Skewness   = {x.skew():.3f}")
    print(f"Zero (%)   = {(x == 0).mean() * 100:.1f}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    df = load_weather(WEATHER_CSV)

    T = df["Temperature_C"]
    V = df["VapourPressure_inHg"]
    R = df["Rain_in"]

    # -------------------------------------------------------------------------
    # Rainfall statistics
    # -------------------------------------------------------------------------

    n_rain = R.notna().sum()
    n_zero = (R == 0).sum()
    pct_zero = 100 * n_zero / n_rain

    # -------------------------------------------------------------------------
    # Print diagnostics
    # -------------------------------------------------------------------------

    print("\n========================================")
    print("BOMBAY WEATHER DATA: DISTRIBUTION CHECK")
    print("========================================")

    describe_variable(T, "Temperature (°C)")
    describe_variable(V, "Vapour pressure (inHg)")
    describe_variable(R, "Daily rainfall (in)")

    corr_TV = df[
        ["Temperature_C", "VapourPressure_inHg"]
    ].corr().iloc[0, 1]

    print("\nTemperature–vapour pressure correlation")
    print("---------------------------------------")
    print(f"Pearson r = {corr_TV:.3f}")

    print("\nRainfall zeros")
    print("--------------")
    print(f"Zero-rainfall days = {n_zero:,} / {n_rain:,}")
    print(f"Percentage zero    = {pct_zero:.1f}%")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 2.7),
        dpi=300
    )

    variables = [
        (
            T,
            "A",
            "Temperature",
            "Temperature (°C)",
            30,
            C_T
        ),
        (
            V,
            "B",
            "Vapour pressure",
            "Vapour pressure (inHg)",
            30,
            C_V
        ),
        (
            R,
            "C",
            "Rainfall",
            "Daily rainfall (in)",
            30,
            C_R
        ),
    ]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    for i, (
        x,
        letter,
        title,
        xlabel,
        bins,
        color
    ) in enumerate(variables):

        ax = axes[i]
        x = x.dropna()

        # Histogram
        ax.hist(
            x,
            bins=bins,
            color=color,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.4,
            zorder=2
        )

        # Statistics
        median = x.median()
        q05 = x.quantile(0.05)
        q95 = x.quantile(0.95)

        # -------------------------------------------------------------
        # Median
        # -------------------------------------------------------------

        ax.axvline(
            median,
            color="black",
            linestyle="--",
            linewidth=1.3,
            zorder=4
        )

        # -------------------------------------------------------------
        # Percentiles
        # -------------------------------------------------------------

        # For temperature and vapour pressure:
        # show both 5th and 95th percentiles.
        #
        # For rainfall:
        # do NOT show the 5th percentile because it is identical
        # to the median (= 0).

        if i != 2:

            ax.axvline(
                q05,
                color="0.30",
                linestyle=":",
                linewidth=1.0,
                zorder=4
            )

        # 95th percentile for all variables
        ax.axvline(
            q95,
            color="0.30",
            linestyle=":",
            linewidth=1.0,
            zorder=4
        )

        # -------------------------------------------------------------
        # Titles and labels
        # -------------------------------------------------------------

        ax.set_title(
            f"{letter}   {title}",
            fontsize=9,
            fontweight="bold"
        )

        ax.set_xlabel(xlabel)

        # -------------------------------------------------------------
        # Style
        # -------------------------------------------------------------

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            linewidth=0.4,
            alpha=0.25,
            zorder=0
        )

        ax.tick_params(direction="out")

    # =========================================================================
    # RAINFALL ANNOTATION
    # =========================================================================

    axes[2].text(
        0.97,
        0.95,
        f"Zero rainfall: {pct_zero:.1f}%\n"
        f"($n$ = {n_zero:,})",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.6",
            alpha=0.9
        )
    )

    # -------------------------------------------------------------------------
    # Y-axis
    # -------------------------------------------------------------------------

    axes[0].set_ylabel("Number of days")

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    for ext, kw in (
        ("pdf", {}),
        ("png", {"dpi": 600})
    ):

        plt.savefig(
            os.path.join(
                FIG_DIR,
                f"{OUT_NAME}.{ext}"
            ),
            bbox_inches="tight",
            **kw
        )

    plt.close(fig)

    print(
        f"\nSaved "
        f"{os.path.join(FIG_DIR, OUT_NAME)}.pdf "
        f"and .png"
    )


# =============================================================================
# 7. RUN
# =============================================================================

if __name__ == "__main__":
    main()