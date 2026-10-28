#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation analysis of Bombay weather variables, 1896–1899.

Variables:
    - Temperature
    - Vapour pressure
    - Rainfall

Calculates:
    1. Pearson correlation
    2. Spearman correlation

Also produces pairwise scatter plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. PATHS
# =============================================================================

WEATHER_CSV = "bombay_weather_daily_1896_1899.csv"

FIG_DIR = "figures"
OUT_NAME = "bombay_weather_correlations"

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

C_TV = "#7A5195"
C_TR = "#D55E00"
C_VR = "#0072B2"


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
# 5. MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_weather(WEATHER_CSV)

    climate = df[
        [
            "Temperature_C",
            "VapourPressure_inHg",
            "Rain_in"
        ]
    ].copy()

    climate.columns = [
        "Temperature",
        "Vapour pressure",
        "Rainfall"
    ]

    # Remove rows with missing observations
    climate = climate.dropna()

    print("\nNumber of complete observations:")
    print(len(climate))

    # =========================================================================
    # PEARSON CORRELATION
    # =========================================================================

    pearson = climate.corr(method="pearson")

    print("\n==========================================")
    print("PEARSON CORRELATION")
    print("==========================================")
    print(pearson.round(3))

    # =========================================================================
    # SPEARMAN CORRELATION
    # =========================================================================

    spearman = climate.corr(method="spearman")

    print("\n==========================================")
    print("SPEARMAN CORRELATION")
    print("==========================================")
    print(spearman.round(3))

    # =========================================================================
    # EXTRACT INDIVIDUAL CORRELATIONS
    # =========================================================================

    r_TV = pearson.loc[
        "Temperature",
        "Vapour pressure"
    ]

    r_TR = pearson.loc[
        "Temperature",
        "Rainfall"
    ]

    r_VR = pearson.loc[
        "Vapour pressure",
        "Rainfall"
    ]

    rho_TV = spearman.loc[
        "Temperature",
        "Vapour pressure"
    ]

    rho_TR = spearman.loc[
        "Temperature",
        "Rainfall"
    ]

    rho_VR = spearman.loc[
        "Vapour pressure",
        "Rainfall"
    ]

    print("\n==========================================")
    print("PAIRWISE RESULTS")
    print("==========================================")

    print(
        f"Temperature vs vapour pressure: "
        f"Pearson r = {r_TV:.3f}, "
        f"Spearman rho = {rho_TV:.3f}"
    )

    print(
        f"Temperature vs rainfall: "
        f"Pearson r = {r_TR:.3f}, "
        f"Spearman rho = {rho_TR:.3f}"
    )

    print(
        f"Vapour pressure vs rainfall: "
        f"Pearson r = {r_VR:.3f}, "
        f"Spearman rho = {rho_VR:.3f}"
    )

    # =========================================================================
    # FIGURE
    # =========================================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.7),
        dpi=300
    )

    # -------------------------------------------------------------------------
    # A. Temperature vs vapour pressure
    # -------------------------------------------------------------------------

    axes[0].scatter(
        climate["Temperature"],
        climate["Vapour pressure"],
        s=8,
        alpha=0.40,
        color=C_TV,
        edgecolors="none"
    )

    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Vapour pressure (inHg)")

    axes[0].set_title(
        "A   Temperature vs vapour pressure",
        fontsize=9,
        fontweight="bold"
    )

    axes[0].text(
        0.05,
        0.95,
        f"Pearson $r$ = {r_TV:.2f}\n"
        f"Spearman $\\rho$ = {rho_TV:.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8
    )

    # -------------------------------------------------------------------------
    # B. Temperature vs rainfall
    # -------------------------------------------------------------------------

    axes[1].scatter(
        climate["Temperature"],
        climate["Rainfall"],
        s=8,
        alpha=0.40,
        color=C_TR,
        edgecolors="none"
    )

    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Daily rainfall (in)")

    axes[1].set_title(
        "B   Temperature vs rainfall",
        fontsize=9,
        fontweight="bold"
    )

    axes[1].text(
        0.05,
        0.95,
        f"Pearson $r$ = {r_TR:.2f}\n"
        f"Spearman $\\rho$ = {rho_TR:.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8
    )

    # -------------------------------------------------------------------------
    # C. Vapour pressure vs rainfall
    # -------------------------------------------------------------------------

    axes[2].scatter(
        climate["Vapour pressure"],
        climate["Rainfall"],
        s=8,
        alpha=0.40,
        color=C_VR,
        edgecolors="none"
    )

    axes[2].set_xlabel("Vapour pressure (inHg)")
    axes[2].set_ylabel("Daily rainfall (in)")

    axes[2].set_title(
        "C   Vapour pressure vs rainfall",
        fontsize=9,
        fontweight="bold"
    )

    axes[2].text(
        0.05,
        0.95,
        f"Pearson $r$ = {r_VR:.2f}\n"
        f"Spearman $\\rho$ = {rho_VR:.2f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8
    )

    # =========================================================================
    # STYLE
    # =========================================================================

    for ax in axes:

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.25
        )

        ax.tick_params(direction="out")

    plt.tight_layout()

    # =========================================================================
    # SAVE
    # =========================================================================

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
# 6. RUN
# =============================================================================

if __name__ == "__main__":
    main()