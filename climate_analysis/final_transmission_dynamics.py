#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# final_transmission_dynamics.py
# =============================================================================

"""
POSTERIOR TRANSMISSION-DYNAMICS DECOMPOSITION
DARK PRESENTATION VERSION — TIMES NEW ROMAN
============================================

Posterior decomposition of the force of infection into:

    1. Reservoir-associated spillover:
           beta_R(t)

    2. Human-associated amplification:
           beta_H(t) * I(t) / N

    3. Total force of infection:
           lambda(t)

for the FOUR final Bayesian model-comparison runs.

This script reads the final_v2 posterior results and DOES NOT refit anything.

For each beta_H formulation:

    1. Convert dynesty's weighted posterior into an equal-weight posterior
       reproducibly using a seeded random state.

    2. Select N_POSTERIOR_DRAWS posterior parameter vectors reproducibly.

    3. Save the exact selected posterior parameter vectors used in the
       downstream simulations.

    4. Re-simulate the mechanistic model once per posterior draw.

    5. Decompose the total force of infection into:

           reservoir-associated spillover:
               beta_R(t)

           human-associated amplification:
               beta_H(t) * I(t) / N

           total force of infection:
               lambda(t)

    6. Calculate the amplification reproduction number:

           R0_H(t) = beta_H(t) / gamma

    7. Aggregate rate-like quantities to weekly means.

    8. Compute weekly posterior medians and 95% credible intervals.

IMPORTANT
---------
The amplification contribution is recovered as:

    amplification_force(t) = lambda(t) - beta_R(t)

rather than recomputing beta_H(t) * I(t) / N from the stored
post-update I(t).

The internal Euler solver evaluates lambda using I from the previous
substep, so this subtraction preserves exact additivity at the time
step used by the simulator:

    lambda(t)
        =
    beta_R(t)
        +
    amplification_force(t)

REPRODUCIBILITY
---------------
Two stochastic steps occur downstream of the fitted posterior:

    1. Equal-weight resampling of dynesty's weighted posterior.

    2. Selection of N_POSTERIOR_DRAWS from that equal-weight posterior.

Both are explicitly seeded.

The exact selected parameter vectors are saved to CSV.

VALIDATION
----------
Every posterior trajectory array is checked for:

    - NaN values
    - Non-finite values
    - Materially negative values

OUTPUT CSV FILES
----------------
results/final_figures/transmission_dynamics/

    posterior_transmission_dynamics_<betaH_mode>.csv

    selected_transmission_dynamics_draws_<betaH_mode>.csv

    posterior_human_R0_summary.csv

OUTPUT FIGURES
--------------
climate_analysis/figures/

    posterior_transmission_dynamics_grid_dark.png
    posterior_transmission_dynamics_grid_dark.pdf

    posterior_transmission_dynamics_all_climate_dark.png
    posterior_transmission_dynamics_all_climate_dark.pdf

STYLE
-----
- Times New Roman
- STIX mathematical notation
- Dark background: #0B0D10
- White text
- Bright blue: reservoir-associated spillover
- Golden orange: human-associated amplification
- Bright green: total force of infection
- Light grey: observed plague deaths
- 600-dpi PNG
- Vector PDF

USAGE
-----
    python scripts/final_transmission_dynamics.py
"""

from __future__ import annotations

import pickle
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import font_manager
from matplotlib.lines import Line2D

from dynesty.utils import resample_equal


# =============================================================================
# 1. PROJECT IMPORTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

import paths  # noqa: E402

from likelihood import build_sim_params  # noqa: E402

from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# 2. SETTINGS
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

MODELS = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]

MODEL_LABELS = {

    "constant":
        r"Constant $\beta_H$",

    "sigmoid":
        r"Sigmoid $\beta_H$",

    "humidity_only":
        r"Vapour-pressure-dependent $\beta_H$",

    "all_climate":
        r"Climate-driven $\beta_H$",

}


# =============================================================================
# 3. PATHS
# =============================================================================

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
)

FINAL_NESTED_DIR = (
    FINAL_ROOT
    / "nested_sampling"
)

# Original results directory.
# Posterior CSV files remain here.

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Presentation figures go alongside the other dark figures.

CLIMATE_DIR = (
    PROJECT_ROOT
    / "climate_analysis"
)

FIG_DIR = (
    CLIMATE_DIR
    / "figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. POSTERIOR SETTINGS
# =============================================================================

N_POSTERIOR_DRAWS = 1000

RANDOM_SEED = 12345


# =============================================================================
# 5. SIMULATION SETTINGS
# =============================================================================

SUBSTEPS_PER_WEEK = 7

TAU0_DAYS = 7

I0 = 0.0

E0 = 0.0

R0_INIT = 0.0

WEEK_RULE = "W-FRI"

NEGATIVE_TOL = -1e-10


# =============================================================================
# 6. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#FFFFFF"

GRID = "#414750"

SPINE = "#D1D5DB"

LEGEND_BG = "#171B21"


# Reservoir-associated spillover

COLOR_SPILLOVER = "#38BDF8"


# Human-associated amplification

COLOR_AMPLIFICATION = "#FFB000"


# Total force of infection

COLOR_TOTAL = "#00C896"


# Observed weekly plague deaths

COLOR_DEATHS = "#D1D5DB"


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

    "axes.labelsize": 13,

    "axes.titlesize": 15,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    # -------------------------------------------------------------------------
    # LINES
    # -------------------------------------------------------------------------

    "lines.linewidth": 2.0,

    "axes.linewidth": 1.0,

    # -------------------------------------------------------------------------
    # TICKS
    # -------------------------------------------------------------------------

    "xtick.major.width": 0.9,

    "ytick.major.width": 0.9,

    "xtick.direction": "out",

    "ytick.direction": "out",

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
    # PDF FONT EMBEDDING
    # -------------------------------------------------------------------------

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})


# =============================================================================
# 8. PLOT HELPERS
# =============================================================================

def style_axis(ax):
    """
    Apply the common dark presentation style.
    """

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
        width=0.9,
    )

    ax.grid(
        True,
        color=GRID,
        linewidth=0.65,
        alpha=0.30,
        linestyle="--",
        zorder=0,
    )

    ax.set_axisbelow(True)


def style_legend(legend):
    """
    Style legends for dark presentation figures.
    """

    if legend is None:
        return

    frame = legend.get_frame()

    frame.set_facecolor(
        LEGEND_BG
    )

    frame.set_edgecolor(
        GRID
    )

    frame.set_alpha(
        0.95
    )

    for item in legend.get_texts():

        item.set_color(
            TEXT
        )


def format_time_axis(ax, interval=4):
    """
    Format the date axis consistently.
    """

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=interval
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m"
        )
    )

    ax.tick_params(
        axis="x",
        colors=TEXT,
        labelrotation=0,
    )


def save_presentation_figure(
    fig,
    filename: str
):
    """
    Save a 600-dpi PNG and a vector PDF.
    """

    png_path = (
        FIG_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIG_DIR
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=BG,
    )

    plt.close(fig)

    print(
        f"[OK] Saved: {png_path}"
    )

    print(
        f"[OK] Saved: {pdf_path}"
    )


# =============================================================================
# 9. LOAD FINAL POSTERIOR RESULTS
# =============================================================================

def load_final_result(
    betaH_mode: str
) -> dict:

    """
    Load one final_v2 nested-sampling posterior result.
    """

    pkl_path = (
        FINAL_NESTED_DIR
        / f"{betaH_mode}__{FIT_TARGET}__final_v2.pkl"
    )

    if not pkl_path.exists():

        raise FileNotFoundError(
            f"Could not find final_v2 result for {betaH_mode!r}:\n"
            f"{pkl_path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(
        pkl_path,
        "rb"
    ) as f:

        saved = pickle.load(f)

    return saved


# =============================================================================
# 10. EQUAL-WEIGHT POSTERIOR
# =============================================================================

def get_equal_weight_posterior(
    saved: dict,
    seed: int
) -> np.ndarray:

    """
    Convert dynesty's weighted posterior sample cloud
    into an equal-weight posterior reproducibly.
    """

    results = saved["results"]

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    rstate = np.random.default_rng(
        seed
    )

    posterior = resample_equal(
        results.samples,
        weights,
        rstate=rstate,
    )

    return posterior


# =============================================================================
# 11. SIMULATE ONE POSTERIOR DRAW
# =============================================================================

def simulate_one_draw(
    theta: np.ndarray,
    param_names: list[str],
    betaH_mode: str,
    zT: np.ndarray,
    ze: np.ndarray,
    zR: np.ndarray,
    start_date: str,
) -> pd.DataFrame:

    """
    Run the mechanistic model once for one
    posterior parameter vector.
    """

    sample = dict(
        zip(
            param_names,
            np.asarray(
                theta,
                dtype=float
            )
        )
    )

    sim_params = build_sim_params(
        sample,
        betaH_mode
    )

    sim = simulate_seir_dde(
        zT=zT,
        ze=ze,
        zR=zR,
        N=sample["N0"],
        params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0,
        E0=E0,
        R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=betaH_mode,
        start_date=start_date,
    )

    # -------------------------------------------------------------------------
    # Recover the amplification contribution exactly as
    # evaluated by the internal simulator.
    # -------------------------------------------------------------------------

    sim["amplification_force"] = (
        sim["lambda"]
        - sim["beta_R"]
    )

    return sim


# =============================================================================
# 12. WEEKLY MEAN TRAJECTORY
# =============================================================================

def weekly_mean_trajectory(
    sim: pd.DataFrame,
    week_ending_dates: pd.Series,
    col: str,
) -> np.ndarray:

    """
    Calculate weekly means of internal-grid quantities.

    Appropriate for:

        beta_R
        amplification_force
        lambda
        R0_H
    """

    df = sim.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    wk = (
        df
        .set_index("date")[col]
        .resample(WEEK_RULE)
        .mean()
        .reset_index()
        .rename(
            columns={
                col: f"{col}_week",
                "date": "week_ending"
            }
        )
    )

    target = pd.DataFrame({

        "week_ending": pd.to_datetime(
            week_ending_dates
        )

    })

    aligned = pd.merge(
        target,
        wk,
        on="week_ending",
        how="left"
    )

    values = aligned[
        f"{col}_week"
    ].to_numpy(
        dtype=float
    )

    return values


# =============================================================================
# 13. POSTERIOR QUANTILES
# =============================================================================

def posterior_quantiles(
    arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Posterior 2.5%, 50%, and 97.5% quantiles
    across posterior draws.
    """

    return (

        np.nanquantile(
            arr,
            0.025,
            axis=0
        ),

        np.nanquantile(
            arr,
            0.50,
            axis=0
        ),

        np.nanquantile(
            arr,
            0.975,
            axis=0
        ),

    )


# =============================================================================
# 14. VALIDATION
# =============================================================================

def validate_array(
    name: str,
    arr: np.ndarray,
    check_nonnegative: bool = False
) -> None:

    """
    Fail explicitly if a posterior trajectory array
    contains invalid values.
    """

    n_nan = int(
        np.isnan(arr).sum()
    )

    if n_nan > 0:

        raise RuntimeError(
            f"{name} contains {n_nan} NaN values."
        )

    if not np.isfinite(arr).all():

        raise RuntimeError(
            f"{name} contains non-finite values."
        )

    if (
        check_nonnegative
        and np.any(arr < NEGATIVE_TOL)
    ):

        n_negative = int(
            np.sum(
                arr < NEGATIVE_TOL
            )
        )

        most_negative = float(
            np.min(arr)
        )

        raise RuntimeError(
            f"{name} contains {n_negative} materially negative values "
            f"(most negative = {most_negative:.3e}, "
            f"tolerance = {NEGATIVE_TOL:.1e})."
        )


# =============================================================================
# 15. FIGURE 1: FOUR-MODEL POSTERIOR DECOMPOSITION
# =============================================================================

def plot_four_model_grid(
    all_model_summaries: dict[str, pd.DataFrame]
):
    """
    Plot the posterior force-of-infection decomposition
    for all four beta_H formulations.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        sharex=True,
        dpi=300,
    )

    fig.patch.set_facecolor(BG)

    panel_letters = [
        "A",
        "B",
        "C",
        "D"
    ]

    for ax, betaH_mode, letter in zip(
        axes.flat,
        MODELS,
        panel_letters,
    ):

        df = all_model_summaries[
            betaH_mode
        ]

        t = df[
            "week_ending"
        ].to_numpy()

        # ---------------------------------------------------------------------
        # RESERVOIR-ASSOCIATED SPILLOVER
        # ---------------------------------------------------------------------

        ax.fill_between(
            t,
            df["spillover_q025"].to_numpy(dtype=float),
            df["spillover_q975"].to_numpy(dtype=float),
            color=COLOR_SPILLOVER,
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )

        ax.plot(
            t,
            df["spillover_median"].to_numpy(dtype=float),
            color=COLOR_SPILLOVER,
            linewidth=2.0,
            label=r"Spillover $\beta_R(t)$",
            zorder=4,
        )

        # ---------------------------------------------------------------------
        # HUMAN-ASSOCIATED AMPLIFICATION
        # ---------------------------------------------------------------------

        ax.fill_between(
            t,
            df["human_q025"].to_numpy(dtype=float),
            df["human_q975"].to_numpy(dtype=float),
            color=COLOR_AMPLIFICATION,
            alpha=0.23,
            linewidth=0,
            zorder=2,
        )

        ax.plot(
            t,
            df["human_median"].to_numpy(dtype=float),
            color=COLOR_AMPLIFICATION,
            linewidth=2.0,
            label=r"Amplification $\beta_H(t)I(t)/N$",
            zorder=5,
        )

        # ---------------------------------------------------------------------
        # TOTAL FORCE OF INFECTION
        # ---------------------------------------------------------------------

        ax.plot(
            t,
            df["total_median"].to_numpy(dtype=float),
            color=COLOR_TOTAL,
            linewidth=1.8,
            linestyle="--",
            label=r"Total $\lambda(t)$",
            zorder=3,
        )

        # ---------------------------------------------------------------------
        # AXES
        # ---------------------------------------------------------------------

        ax.set_title(
            f"{letter}   {MODEL_LABELS[betaH_mode]}",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        ax.set_ylabel(
            r"Force of infection (week$^{-1}$)",
            fontsize=12,
            labelpad=9,
        )

        ax.set_ylim(
            bottom=0
        )

        style_axis(
            ax
        )

        format_time_axis(
            ax,
            interval=6
        )

    # =========================================================================
    # SHARED LEGEND
    # =========================================================================

    handles = [

        Line2D(
            [0], [0],
            color=COLOR_SPILLOVER,
            linewidth=2.2,
            label=r"Reservoir-associated spillover $\beta_R(t)$",
        ),

        Line2D(
            [0], [0],
            color=COLOR_AMPLIFICATION,
            linewidth=2.2,
            label=r"Human-associated amplification $\beta_H(t)I(t)/N$",
        ),

        Line2D(
            [0], [0],
            color=COLOR_TOTAL,
            linewidth=2.0,
            linestyle="--",
            label=r"Total force of infection $\lambda(t)$",
        ),

    ]

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=11,
        frameon=True,
        handlelength=2.8,
        columnspacing=2.0,
    )

    style_legend(
        legend
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.13,
        top=0.94,
        wspace=0.24,
        hspace=0.30,
    )

    save_presentation_figure(
        fig,
        "posterior_transmission_dynamics_grid_dark"
    )


# =============================================================================
# 16. FIGURE 2: ALL-CLIMATE POSTERIOR DECOMPOSITION
# =============================================================================

def plot_all_climate_detail(
    df: pd.DataFrame
):
    """
    Detailed posterior decomposition for the
    climate-driven amplification model.

    Includes observed plague mortality on
    the secondary axis.
    """

    t = df[
        "week_ending"
    ].to_numpy()

    fig, ax1 = plt.subplots(
        figsize=(14, 6),
        dpi=300,
    )

    fig.patch.set_facecolor(BG)

    # =========================================================================
    # RESERVOIR-ASSOCIATED SPILLOVER
    # =========================================================================

    ax1.fill_between(
        t,
        df["spillover_q025"].to_numpy(dtype=float),
        df["spillover_q975"].to_numpy(dtype=float),
        color=COLOR_SPILLOVER,
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )

    ax1.plot(
        t,
        df["spillover_median"].to_numpy(dtype=float),
        color=COLOR_SPILLOVER,
        linewidth=2.4,
        label=r"Reservoir-associated spillover $\beta_R(t)$ (95% CrI)",
        zorder=4,
    )

    # =========================================================================
    # HUMAN-ASSOCIATED AMPLIFICATION
    # =========================================================================

    ax1.fill_between(
        t,
        df["human_q025"].to_numpy(dtype=float),
        df["human_q975"].to_numpy(dtype=float),
        color=COLOR_AMPLIFICATION,
        alpha=0.23,
        linewidth=0,
        zorder=2,
    )

    ax1.plot(
        t,
        df["human_median"].to_numpy(dtype=float),
        color=COLOR_AMPLIFICATION,
        linewidth=2.4,
        label=r"Human-associated amplification $\beta_H(t)I(t)/N$ (95% CrI)",
        zorder=5,
    )

    # =========================================================================
    # TOTAL FORCE OF INFECTION
    # =========================================================================

    ax1.plot(
        t,
        df["total_median"].to_numpy(dtype=float),
        color=COLOR_TOTAL,
        linewidth=2.0,
        linestyle="--",
        label=r"Total force of infection $\lambda(t)$",
        zorder=3,
    )

    # =========================================================================
    # PRIMARY AXIS
    # =========================================================================

    ax1.set_ylabel(
        r"Force of infection (week$^{-1}$)",
        fontsize=14,
        labelpad=12,
    )

    ax1.set_xlabel(
        "Week ending",
        fontsize=14,
        labelpad=12,
    )

    ax1.set_ylim(
        bottom=0
    )

    style_axis(
        ax1
    )

    format_time_axis(
        ax1,
        interval=4
    )

    # =========================================================================
    # SECONDARY AXIS: OBSERVED PLAGUE DEATHS
    # =========================================================================

    ax2 = ax1.twinx()

    ax2.scatter(
        t,
        df["observed_deaths"].to_numpy(dtype=float),
        s=15,
        color=COLOR_DEATHS,
        alpha=0.45,
        edgecolors="none",
        zorder=1,
        label="Observed deaths (right axis)",
    )

    ax2.set_ylabel(
        "Observed weekly plague deaths",
        fontsize=14,
        color=TEXT,
        labelpad=12,
    )

    ax2.tick_params(
        axis="y",
        colors=TEXT,
        labelsize=11,
    )

    ax2.spines["top"].set_visible(False)

    ax2.spines["left"].set_visible(False)

    ax2.spines["right"].set_color(SPINE)

    ax2.set_ylim(
        bottom=0
    )

    # =========================================================================
    # TITLE
    # =========================================================================

    ax1.set_title(
        "Posterior decomposition of the force of infection\n"
        "Climate-driven amplification model",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # =========================================================================
    # COMBINED LEGEND
    # =========================================================================

    handles1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    handles2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    legend = ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=11,
        frameon=True,
        handlelength=2.8,
        labelspacing=0.9,
    )

    style_legend(
        legend
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.88,
        bottom=0.16,
        top=0.82,
    )

    save_presentation_figure(
        fig,
        "posterior_transmission_dynamics_all_climate_dark"
    )


# =============================================================================
# 17. MAIN ANALYSIS
# =============================================================================

def main():

    # =========================================================================
    # 1. LOAD PREPARED DATA
    # =========================================================================

    if not paths.MERGED_CSV.exists():

        raise FileNotFoundError(
            f"Prepared data not found:\n{paths.MERGED_CSV}\n\n"
            "Run scripts/prepare_data.py first."
        )

    merged_df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[
            paths.DATE_COL
        ]
    )

    merged_df = (
        merged_df
        .sort_values(
            paths.DATE_COL
        )
        .reset_index(
            drop=True
        )
    )

    required_columns = [
        paths.DATE_COL,
        "zT",
        "ze",
        "zR",
        FIT_TARGET,
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in merged_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Prepared data are missing required columns: "
            + ", ".join(missing_columns)
        )

    week_ending = pd.to_datetime(
        merged_df[
            paths.DATE_COL
        ]
    )

    zT = merged_df[
        "zT"
    ].to_numpy(
        dtype=float
    )

    ze = merged_df[
        "ze"
    ].to_numpy(
        dtype=float
    )

    zR = merged_df[
        "zR"
    ].to_numpy(
        dtype=float
    )

    observed = merged_df[
        FIT_TARGET
    ].to_numpy(
        dtype=float
    )

    start_date = str(
        pd.Timestamp(
            week_ending.iloc[0]
        ).date()
    )

    n_weeks = len(
        merged_df
    )

    # =========================================================================
    # 2. MAIN LOOP OVER MODELS
    # =========================================================================

    print("\n" + "=" * 100)

    print(
        "FINAL POSTERIOR TRANSMISSION-DYNAMICS DECOMPOSITION"
    )

    print(
        "DARK PRESENTATION — TIMES NEW ROMAN"
    )

    print("=" * 100)

    print(
        f"\nFitted weeks: {n_weeks}"
    )

    print(
        f"Study period: "
        f"{week_ending.min().date()} "
        f"to {week_ending.max().date()}"
    )

    print(
        f"Posterior draws per model: {N_POSTERIOR_DRAWS}"
    )

    all_model_summaries: dict[
        str,
        pd.DataFrame
    ] = {}

    r0_summary_rows = []

    for model_index, betaH_mode in enumerate(
        MODELS
    ):

        print("\n" + "-" * 80)

        print(
            f"MODEL: {betaH_mode}"
        )

        print("-" * 80)

        # =====================================================================
        # 2a. LOAD POSTERIOR
        # =====================================================================

        saved = load_final_result(
            betaH_mode
        )

        param_names = list(
            saved["param_names"]
        )

        resample_seed = (
            RANDOM_SEED
            + 100 * model_index
        )

        posterior_samples = get_equal_weight_posterior(
            saved,
            seed=resample_seed
        )

        # ---------------------------------------------------------------------
        # ALL-CLIMATE SANITY CHECK
        # ---------------------------------------------------------------------

        if betaH_mode == "all_climate":

            loaded_logz = float(
                saved["results"].logz[-1]
            )

            expected_logz = -813.629

            if (
                abs(
                    loaded_logz
                    - expected_logz
                ) > 1.0
            ):

                print(
                    "*** WARNING: all_climate logZ="
                    f"{loaded_logz:.3f}, "
                    "expected approximately "
                    f"{expected_logz:.3f}. "
                    "Check FINAL_NESTED_DIR. ***"
                )

            else:

                print(
                    f"[OK] all_climate logZ="
                    f"{loaded_logz:.3f} "
                    "matches the verified final_v2 result."
                )

        # =====================================================================
        # 2b. SELECT POSTERIOR DRAWS
        # =====================================================================

        n_available = len(
            posterior_samples
        )

        n_draws = min(
            N_POSTERIOR_DRAWS,
            n_available
        )

        subsample_seed = (
            RANDOM_SEED
            + 10000
            + model_index
        )

        subsample_rng = np.random.default_rng(
            subsample_seed
        )

        draw_idx = subsample_rng.choice(
            n_available,
            size=n_draws,
            replace=False
        )

        draws = posterior_samples[
            draw_idx
        ]

        print(
            f"Equal-weight posterior samples available: "
            f"{n_available}"
        )

        print(
            f"Posterior draws used: {n_draws}"
        )

        # =====================================================================
        # 2c. SAVE EXACT PARAMETER VECTORS
        # =====================================================================

        selected_draws_df = pd.DataFrame(
            draws,
            columns=param_names
        )

        selected_draws_df.insert(
            0,
            "equal_weight_sample_index",
            draw_idx
        )

        selected_draws_df.insert(
            0,
            "posterior_draw",
            np.arange(
                1,
                n_draws + 1
            )
        )

        selected_draws_path = (
            OUTPUT_DIR
            / f"selected_transmission_dynamics_draws_{betaH_mode}.csv"
        )

        selected_draws_df.to_csv(
            selected_draws_path,
            index=False
        )

        print(
            "[OK] Saved exact selected posterior draws:\n"
            f"  {selected_draws_path}"
        )

        # =====================================================================
        # 2d. ALLOCATE TRAJECTORY ARRAYS
        # =====================================================================

        gamma_idx = param_names.index(
            "gamma"
        )

        spillover_draws = np.full(
            (n_draws, n_weeks),
            np.nan,
            dtype=float
        )

        amplification_draws = np.full(
            (n_draws, n_weeks),
            np.nan,
            dtype=float
        )

        total_draws = np.full(
            (n_draws, n_weeks),
            np.nan,
            dtype=float
        )

        r0_draws = np.full(
            (n_draws, n_weeks),
            np.nan,
            dtype=float
        )

        # =====================================================================
        # 2e. SIMULATE POSTERIOR DRAWS
        # =====================================================================

        for i, theta in enumerate(
            draws
        ):

            sim = simulate_one_draw(
                theta=theta,
                param_names=param_names,
                betaH_mode=betaH_mode,
                zT=zT,
                ze=ze,
                zR=zR,
                start_date=start_date,
            )

            spillover_draws[
                i, :
            ] = weekly_mean_trajectory(
                sim,
                week_ending,
                "beta_R"
            )

            amplification_draws[
                i, :
            ] = weekly_mean_trajectory(
                sim,
                week_ending,
                "amplification_force"
            )

            total_draws[
                i, :
            ] = weekly_mean_trajectory(
                sim,
                week_ending,
                "lambda"
            )

            gamma = float(
                theta[
                    gamma_idx
                ]
            )

            sim["R0_H"] = (
                sim["beta_H"]
                / gamma
            )

            r0_draws[
                i, :
            ] = weekly_mean_trajectory(
                sim,
                week_ending,
                "R0_H"
            )

            if (
                (i + 1) % 50 == 0
                or (i + 1) == n_draws
            ):

                print(
                    f"  processed {i + 1} / {n_draws}"
                )

        # =====================================================================
        # 2f. VALIDATE TRAJECTORIES
        # =====================================================================

        validate_array(
            "spillover_draws",
            spillover_draws,
            check_nonnegative=True
        )

        validate_array(
            "amplification_draws",
            amplification_draws,
            check_nonnegative=True
        )

        validate_array(
            "total_draws",
            total_draws,
            check_nonnegative=True
        )

        validate_array(
            "r0_draws",
            r0_draws,
            check_nonnegative=True
        )

        # =====================================================================
        # 2g. POSTERIOR WEEKLY SUMMARIES
        # =====================================================================

        spill_lo, spill_med, spill_hi = posterior_quantiles(
            spillover_draws
        )

        amp_lo, amp_med, amp_hi = posterior_quantiles(
            amplification_draws
        )

        total_lo, total_med, total_hi = posterior_quantiles(
            total_draws
        )

        r0_lo, r0_med, r0_hi = posterior_quantiles(
            r0_draws
        )

        # =====================================================================
        # 2h. SAVE WEEKLY POSTERIOR SUMMARY
        # =====================================================================

        model_df = pd.DataFrame({

            "week_ending": week_ending,

            "observed_deaths": observed,

            # Reservoir-associated spillover

            "spillover_q025": spill_lo,

            "spillover_median": spill_med,

            "spillover_q975": spill_hi,

            # Human-associated amplification

            "human_q025": amp_lo,

            "human_median": amp_med,

            "human_q975": amp_hi,

            # Total force of infection

            "total_q025": total_lo,

            "total_median": total_med,

            "total_q975": total_hi,

            # Amplification reproduction number

            "R0_H_q025": r0_lo,

            "R0_H_median": r0_med,

            "R0_H_q975": r0_hi,

        })

        csv_path = (
            OUTPUT_DIR
            / f"posterior_transmission_dynamics_{betaH_mode}.csv"
        )

        model_df.to_csv(
            csv_path,
            index=False
        )

        print(
            "\n[OK] Saved weekly posterior transmission dynamics:\n"
            f"  {csv_path}"
        )

        all_model_summaries[
            betaH_mode
        ] = model_df

        # =====================================================================
        # 2i. R0_H DIAGNOSTICS
        # =====================================================================

        r0_flat = r0_draws[
            np.isfinite(
                r0_draws
            )
        ]

        if r0_flat.size:

            prop_above_1 = float(
                np.mean(
                    r0_flat > 1.0
                )
            )

        else:

            prop_above_1 = np.nan

        r0_summary_rows.append({

            "betaH_mode":
                betaH_mode,

            "R0_H_posterior_mean":
                float(np.mean(r0_flat)),

            "R0_H_posterior_median":
                float(np.median(r0_flat)),

            "R0_H_q025":
                float(np.quantile(r0_flat, 0.025)),

            "R0_H_q975":
                float(np.quantile(r0_flat, 0.975)),

            "R0_H_min":
                float(np.min(r0_flat)),

            "R0_H_max":
                float(np.max(r0_flat)),

            "pct_draw_week_combinations_R0_H_above_1":
                100.0 * prop_above_1,

        })

        print(
            f"  R0_H pooled draw-week summary: "
            f"median={np.median(r0_flat):.3f}, "
            f"95% interval=("
            f"{np.quantile(r0_flat, 0.025):.3f}, "
            f"{np.quantile(r0_flat, 0.975):.3f}), "
            f"{100.0 * prop_above_1:.1f}% "
            "of draw-week combinations exceed 1"
        )

    # =========================================================================
    # 3. SAVE R0_H SUMMARY
    # =========================================================================

    r0_summary_df = pd.DataFrame(
        r0_summary_rows
    )

    r0_csv = (
        OUTPUT_DIR
        / "posterior_human_R0_summary.csv"
    )

    r0_summary_df.to_csv(
        r0_csv,
        index=False
    )

    print("\n" + "=" * 100)

    print(
        "R0_H SUMMARY — ALL MODELS"
    )

    print("=" * 100)

    print(
        r0_summary_df.to_string(
            index=False
        )
    )

    print(
        f"\n[OK] Saved:\n  {r0_csv}"
    )

    # =========================================================================
    # 4. FIGURE 1: FOUR-MODEL GRID
    # =========================================================================

    plot_four_model_grid(
        all_model_summaries
    )

    # =========================================================================
    # 5. FIGURE 2: ALL-CLIMATE DETAIL
    # =========================================================================

    plot_all_climate_detail(
        all_model_summaries[
            "all_climate"
        ]
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print("\n" + "=" * 100)

    print(
        "POSTERIOR TRANSMISSION-DYNAMICS DECOMPOSITION COMPLETE"
    )

    print("=" * 100)

    print(
        f"\nPosterior CSV files saved in:\n  {OUTPUT_DIR}"
    )

    print(
        f"\nDark presentation figures saved in:\n  {FIG_DIR}"
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()