# =============================================================================
# plot_climate_perturbation_effects.py
# =============================================================================
"""
Forest plot of the posterior climate-perturbation results.

Reads directly from:

    climate_perturbation_effects_posterior.csv

produced by:

    run_climate_perturbation_analysis.py

No simulation or posterior sampling is performed here.

The number of posterior draws used in the upstream analysis is read directly
from the CSV rather than being hard-coded in the figure title.

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
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "scenario_analysis"
    / "climate_perturbation_effects_posterior.csv"
)

OUTPUT_DIR = CSV_PATH.parent

FIG_PATH_PNG = (
    OUTPUT_DIR
    / "climate_perturbation_forest_plot.png"
)

FIG_PATH_PDF = (
    OUTPUT_DIR
    / "climate_perturbation_forest_plot.pdf"
)


SCENARIO_ORDER = [
    (
        "temperature",
        "+",
        "Temperature +2 °C",
    ),
    (
        "temperature",
        "-",
        "Temperature \u22122 °C",
    ),
    (
        "vapour_pressure",
        "+",
        "Vapour pressure +1.69 hPa",
    ),
    (
        "vapour_pressure",
        "-",
        "Vapour pressure \u22121.69 hPa",
    ),
    (
        "rainfall",
        "+",
        "Rainfall +50.8 mm",
    ),
    (
        "rainfall",
        "-",
        "Rainfall \u221250.8 mm",
    ),
]


# =============================================================================
# Plot colours
# =============================================================================

COLOR_R = "#1f77b4"
COLOR_H = "#E08214"


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load input
    # -------------------------------------------------------------------------

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{CSV_PATH}\n\n"
            "Run scripts/run_climate_perturbation_analysis.py first."
        )

    df = pd.read_csv(
        CSV_PATH
    )

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

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing))
            + "\n\n"
            "Rerun run_climate_perturbation_analysis.py after adding "
            "'n_posterior_draws' to the saved result rows."
        )

    # -------------------------------------------------------------------------
    # Keep only the time-averaged coefficient statistic
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
    # Read posterior-draw count from the CSV
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

    print(
        f"Posterior draws used in upstream analysis: "
        f"{n_draws}"
    )

    # -------------------------------------------------------------------------
    # Validate that every expected scenario/component exists exactly once
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

    # =========================================================================
    # Plot styling
    # =========================================================================

    plt.rcParams.update({
        "font.family":
            "Times New Roman",

        "font.size":
            10,

        "axes.labelsize":
            11,

        "axes.titlesize":
            11,

        "legend.fontsize":
            9,

        "xtick.labelsize":
            9,

        "ytick.labelsize":
            9,

        "axes.linewidth":
            0.9,
    })

    # =========================================================================
    # Build figure
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    y_positions = np.arange(
        len(SCENARIO_ORDER)
    )

    offset = 0.15

    for i, (
        covariate,
        direction,
        label,
    ) in enumerate(
        SCENARIO_ORDER
    ):

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

            if not (
                np.isfinite(med)
                and
                np.isfinite(lo)
                and
                np.isfinite(hi)
            ):
                raise RuntimeError(
                    "Non-finite plotting value for "
                    f"{covariate}, {direction}, {component}."
                )

            if not (
                lo <= med <= hi
            ):
                raise RuntimeError(
                    "Credible interval does not contain the median for "
                    f"{covariate}, {direction}, {component}: "
                    f"lo={lo}, median={med}, hi={hi}"
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
                capsize=4,
                linewidth=1.6,
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

    # =========================================================================
    # Reference line
    # =========================================================================

    ax.axvline(
        0,
        linestyle="--",
        color="gray",
        linewidth=1.2,
        zorder=0,
    )

    # =========================================================================
    # Axis labels
    # =========================================================================

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        [
            label
            for _, _, label
            in SCENARIO_ORDER
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Change in time-averaged rate relative to historical climate (%)"
    )

    # No title: the figure caption in the manuscript describes the panel
    # (main model, number of posterior draws).

    # =========================================================================
    # Legend
    # =========================================================================

    legend_elements = [

        Line2D(
            [0],
            [0],

            marker="o",
            color=COLOR_R,

            label=(
                r"Spillover, $\beta_R$"
            ),

            markeredgecolor="black",
            markeredgewidth=0.5,

            linestyle="None",
            markersize=7,
        ),

        Line2D(
            [0],
            [0],

            marker="s",
            color=COLOR_H,

            label=(
                r"Amplification, $\beta_H$"
            ),

            markeredgecolor="black",
            markeredgewidth=0.5,

            linestyle="None",
            markersize=7,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )

    # =========================================================================
    # Styling
    # =========================================================================

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    # =========================================================================
    # Save
    # =========================================================================

    fig.savefig(
        FIG_PATH_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_PATH_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Saved: "
        f"{FIG_PATH_PNG}"
    )

    print(
        f"[OK] Saved: "
        f"{FIG_PATH_PDF}"
    )

    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()