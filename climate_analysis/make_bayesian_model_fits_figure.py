#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# make_bayesian_model_fits_figure.py
# =============================================================================

"""
BAYESIAN POSTERIOR MODEL FITS
DARK PRESENTATION VERSION — TIMES NEW ROMAN
============================================

Consolidated Bayesian posterior model-fit figure and diagnostics.

This script reads the CORRECT final_v2 posteriors from:

    results/final_model_comparison_v2/nested_sampling/

and DOES NOT refit the model.

For each beta_H formulation, it calculates:

1. Posterior expected-death trajectory:
   - Reproducibly resample dynesty's weighted posterior.
   - Reproducibly select N_POSTERIOR_DRAWS parameter vectors.
   - Simulate the expected-death trajectory for each draw.
   - Calculate the weekly posterior median and 95% credible interval.

2. Maximum-likelihood (MLE) trajectory:
   - Use the sampled theta with the largest stored log-likelihood.

3. Posterior-median-parameter trajectory:
   - Replace each parameter by its marginal posterior median.
   - Simulate once.
   - This is a diagnostic trajectory, NOT the posterior median trajectory.

The exact posterior parameter vectors used to construct the credible
bands are saved for reproducibility.

IMPORTANT
---------
The 95% credible band represents uncertainty in the latent expected-death
trajectory mu(t).

It is NOT the full posterior-predictive interval.

A posterior-predictive interval would additionally include observation
noise from the negative-binomial observation model.

OUTPUT CSV FILES
----------------
results/final_figures/bayesian_model_fits/

    bayesian_model_fits_<model>.csv
    selected_model_fit_draws_<model>.csv
    mle_vs_median_divergence.csv

OUTPUT FIGURES
--------------
climate_analysis/figures/

    bayesian_model_fits_grid_dark.png
    bayesian_model_fits_grid_dark.pdf

STYLE
-----
- Times New Roman
- STIX mathematical notation
- Dark background: #0B0D10
- White text
- Bright blue observed mortality
- Golden-orange posterior credible band
- Bright-green MLE trajectory
- Dashed golden-orange median-parameter trajectory
- Bright-red original outlier
- 600-dpi PNG
- Vector PDF

USAGE
-----
    python scripts/make_bayesian_model_fits_figure.py
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
from matplotlib.patches import Patch

from dynesty.utils import resample_equal


# =============================================================================
# 1. PROJECT IMPORTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Support running the script from either scripts/ or climate_analysis/.

if SCRIPT_DIR.name in ("scripts", "climate_analysis"):

    PROJECT_ROOT = SCRIPT_DIR.parent

else:

    PROJECT_ROOT = SCRIPT_DIR.parent


sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

import paths  # noqa: E402

from likelihood import build_sim_params  # noqa: E402

from mechanistic_model import simulate_seir_dde, weekly_sum  # noqa: E402

from observation import make_delay_pmf_weeks, deaths_from_infections_fast  # noqa: E402


# =============================================================================
# 2. SETTINGS
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

RAW_TARGET = "plague_deaths_reported"

OUTLIER_WEEK = "1897-02-19"


MODELS = [
    "constant",
    "humidity_only",
    "all_climate",
    "sigmoid",
]


MODEL_LABELS = {

    "constant":
        r"(a) Constant $\beta_H$",

    "humidity_only":
        r"(b) Vapour-pressure-only $\beta_H$",

    "all_climate":
        r"(c) Climate-driven $\beta_H$",

    "sigmoid":
        r"(d) Sigmoid $\beta_H$",

}


# =============================================================================
# 3. FINAL POSTERIOR LOCATION
# =============================================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
    / "nested_sampling"
)


def result_path(
    model: str
) -> Path:

    return (
        RESULT_DIR
        / f"{model}__{FIT_TARGET}__final_v2.pkl"
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


# =============================================================================
# 6. INFECTION-TO-DEATH DELAY SETTINGS
# =============================================================================

DELAY_KIND = "gamma"

DELAY_MAX_WEEKS = 12

DELAY_SHAPE = 2.0

DELAY_SCALE = 0.6


# =============================================================================
# 7. OUTPUT PATHS
# =============================================================================

# Keep all posterior CSV outputs in their original directory.

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "bayesian_model_fits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Save presentation figures alongside the other dark figures.

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
# 8. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#FFFFFF"

GRID = "#414750"

SPINE = "#D1D5DB"

LEGEND_BG = "#171B21"


# Observed plague mortality

COLOR_OBSERVED = "#38BDF8"


# Posterior credible band and median-parameter trajectory

COLOR_MEDIAN = "#FFB000"


# Maximum-likelihood trajectory

COLOR_MLE = "#00C896"


# Original reported outlier

COLOR_OUTLIER = "#FF6B6B"


# =============================================================================
# 9. TIMES NEW ROMAN STYLE
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

    "legend.fontsize": 11,

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
# 10. PLOT HELPERS
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
    Style the legend for the dark background.
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


def format_time_axis(ax, interval=6):
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
# 11. LOAD FINAL POSTERIOR
# =============================================================================

def load_final_result(
    model: str
) -> dict:

    """
    Load one final_v2 nested-sampling posterior.
    """

    path = result_path(
        model
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find final_v2 result for {model!r}:\n"
            f"{path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(
        path,
        "rb"
    ) as f:

        saved = pickle.load(f)

    return saved


# =============================================================================
# 12. READ DYNESTY RESULT ARRAYS
# =============================================================================

def get_result_array(
    results,
    name: str
) -> np.ndarray:

    """
    Read an array from dynesty Results regardless of whether
    it is exposed as an attribute or dictionary-like field.
    """

    if hasattr(
        results,
        name
    ):

        return np.asarray(
            getattr(
                results,
                name
            )
        )

    return np.asarray(
        results[
            name
        ]
    )


# =============================================================================
# 13. EQUAL-WEIGHT POSTERIOR
# =============================================================================

def get_equal_weight_posterior(
    samples: np.ndarray,
    logwt: np.ndarray,
    logz: np.ndarray,
    seed: int,
) -> np.ndarray:

    """
    Convert dynesty's weighted posterior into an
    equal-weight posterior reproducibly.
    """

    weights = np.exp(
        logwt
        -
        logz[-1]
    )

    rstate = np.random.default_rng(
        seed
    )

    posterior = resample_equal(
        samples,
        weights,
        rstate=rstate,
    )

    return np.asarray(
        posterior,
        dtype=float
    )


# =============================================================================
# 14. COMPUTE EXPECTED-DEATH TRAJECTORY
# =============================================================================

def compute_mu_trajectory(
    theta: np.ndarray,
    param_names: list[str],
    betaH_mode: str,
    merged_df: pd.DataFrame,
    zT: np.ndarray,
    ze: np.ndarray,
    zR: np.ndarray,
    start_date: str,
    f_delay: np.ndarray,
) -> np.ndarray:

    """
    Simulate one posterior parameter vector and return
    the expected weekly plague-death trajectory mu(t).

    Calls simulate_seir_dde directly rather than constructing
    a PlagueLikelihood object.
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

    # =========================================================================
    # AGGREGATE MODELLED INFECTIONS TO EPIDEMIOLOGICAL WEEKS
    # =========================================================================

    weekly_inf = weekly_sum(
        sim,
        col="C",
        week_ending=WEEK_RULE,
    )

    weekly_inf["week_ending"] = pd.to_datetime(
        weekly_inf["week_ending"]
    )

    compare = pd.merge(
        merged_df[["week_ending"]],
        weekly_inf,
        on="week_ending",
        how="left",
    )

    # =========================================================================
    # IDENTIFY WEEKLY INFECTION-COUNT COLUMN
    # =========================================================================

    if "C_week" in compare.columns:

        c_col = "C_week"

    else:

        candidate_cols = [
            c
            for c in compare.columns
            if c.startswith("C_")
        ]

        if not candidate_cols:

            raise RuntimeError(
                "Could not identify weekly infection-count column."
            )

        c_col = candidate_cols[0]

    C_week = compare[
        c_col
    ].to_numpy(
        dtype=float
    )

    # =========================================================================
    # VALIDATE WEEKLY INFECTIONS
    # =========================================================================

    if np.isnan(C_week).any():

        bad = np.where(
            np.isnan(C_week)
        )[0]

        raise RuntimeError(
            "Weekly infection trajectory contains missing fitted weeks. "
            f"Indices: {bad.tolist()}"
        )

    if not np.isfinite(C_week).all():

        raise RuntimeError(
            "Weekly infection trajectory contains non-finite values."
        )

    if (C_week < -1e-10).any():

        raise RuntimeError(
            "Weekly infection trajectory contains negative values."
        )

    # =========================================================================
    # CONVERT INFECTIONS TO EXPECTED DEATHS
    # =========================================================================

    mu = deaths_from_infections_fast(
        C_week=C_week,
        psi=sample["psi"],
        f_delay=f_delay,
    )

    mu = np.asarray(
        mu,
        dtype=float
    )

    if len(mu) != len(merged_df):

        raise RuntimeError(
            "Expected-death trajectory has the wrong length: "
            f"{len(mu)} versus {len(merged_df)} fitted weeks."
        )

    if not np.isfinite(mu).all():

        raise RuntimeError(
            "Expected-death trajectory contains non-finite values."
        )

    return np.maximum(
        mu,
        1e-9
    )


# =============================================================================
# 15. SUM OF SQUARED ERRORS
# =============================================================================

def sse(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:

    """
    Sum of squared errors used only as a
    descriptive trajectory diagnostic.
    """

    return float(
        np.sum(
            (observed - predicted) ** 2
        )
    )


# =============================================================================
# 16. PLOT BAYESIAN MODEL FITS
# =============================================================================

def plot_bayesian_model_fits(
    all_model_summaries: dict[str, pd.DataFrame],
    week_ending: pd.Series,
    observed_raw: np.ndarray,
    outlier_mask: np.ndarray,
):
    """
    Generate the four-panel dark presentation figure.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        sharex=True,
        dpi=300,
    )

    fig.patch.set_facecolor(BG)

    # =========================================================================
    # MODEL PANELS
    # =========================================================================

    for ax, model in zip(
        axes.flat,
        MODELS
    ):

        df = all_model_summaries[
            model
        ]

        t = df[
            "week_ending"
        ].to_numpy()

        # ---------------------------------------------------------------------
        # POSTERIOR 95% CREDIBLE INTERVAL
        # ---------------------------------------------------------------------

        ax.fill_between(
            t,
            df["mu_q025"].to_numpy(dtype=float),
            df["mu_q975"].to_numpy(dtype=float),
            color=COLOR_MEDIAN,
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )

        # ---------------------------------------------------------------------
        # OBSERVED REPORTED DEATHS
        # ---------------------------------------------------------------------

        ax.plot(
            t,
            df["observed_fit"].to_numpy(dtype=float),
            marker="o",
            markersize=3.0,
            color=COLOR_OBSERVED,
            linewidth=1.2,
            alpha=0.90,
            label="Observed reported deaths",
            zorder=4,
        )

        # ---------------------------------------------------------------------
        # MAXIMUM-LIKELIHOOD TRAJECTORY
        # ---------------------------------------------------------------------

        ax.plot(
            t,
            df["mu_mle"].to_numpy(dtype=float),
            color=COLOR_MLE,
            linewidth=2.2,
            label="Maximum-likelihood (MLE) trajectory",
            zorder=5,
        )

        # ---------------------------------------------------------------------
        # POSTERIOR-MEDIAN-PARAMETER TRAJECTORY
        # ---------------------------------------------------------------------

        ax.plot(
            t,
            df["mu_median_params"].to_numpy(dtype=float),
            color=COLOR_MEDIAN,
            linewidth=2.0,
            linestyle="--",
            label="Posterior-median-parameter trajectory",
            zorder=6,
        )

        # ---------------------------------------------------------------------
        # ORIGINAL REPORTED OUTLIER
        # ---------------------------------------------------------------------

        if outlier_mask.any():

            ax.scatter(
                week_ending[outlier_mask],
                observed_raw[outlier_mask],
                color=COLOR_OUTLIER,
                s=55,
                edgecolors=BG,
                linewidth=0.8,
                zorder=8,
                label="Original reported outlier",
            )

        # ---------------------------------------------------------------------
        # AXES AND TITLES
        # ---------------------------------------------------------------------

        ax.set_title(
            MODEL_LABELS[model],
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_ylabel(
            "Weekly plague deaths",
            fontsize=13,
            labelpad=10,
        )

        ax.set_ylim(
            bottom=0
        )

        style_axis(ax)

        format_time_axis(
            ax,
            interval=6
        )

    # =========================================================================
    # SHARED LEGEND
    # =========================================================================

    legend_handles = [

        Line2D(
            [0], [0],
            color=COLOR_OBSERVED,
            marker="o",
            markersize=5,
            linewidth=1.5,
            label="Observed reported deaths",
        ),

        Patch(
            facecolor=COLOR_MEDIAN,
            alpha=0.30,
            label="Posterior 95% CrI of expected deaths",
        ),

        Line2D(
            [0], [0],
            color=COLOR_MLE,
            linewidth=2.4,
            label="Maximum-likelihood (MLE) trajectory",
        ),

        Line2D(
            [0], [0],
            color=COLOR_MEDIAN,
            linewidth=2.2,
            linestyle="--",
            label="Posterior-median-parameter trajectory",
        ),

        Line2D(
            [0], [0],
            color=COLOR_OUTLIER,
            marker="o",
            markersize=7,
            linewidth=0,
            label="Original reported outlier",
        ),

    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=11,
        frameon=True,
        handlelength=2.8,
        columnspacing=2.0,
        labelspacing=1.0,
    )

    style_legend(
        legend
    )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.16,
        top=0.94,
        wspace=0.22,
        hspace=0.30,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_presentation_figure(
        fig,
        "bayesian_model_fits_grid_dark"
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
            f"Prepared data not found:\n{paths.MERGED_CSV}"
        )

    merged_df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[paths.DATE_COL],
    )

    merged_df = (
        merged_df
        .sort_values(paths.DATE_COL)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required_columns = [
        paths.DATE_COL,
        "zT",
        "ze",
        "zR",
        FIT_TARGET,
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in merged_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Prepared data are missing required columns: "
            + ", ".join(missing_columns)
        )

    week_ending = pd.to_datetime(
        merged_df[paths.DATE_COL]
    )

    zT = merged_df["zT"].to_numpy(dtype=float)

    ze = merged_df["ze"].to_numpy(dtype=float)

    zR = merged_df["zR"].to_numpy(dtype=float)

    start_date = str(
        pd.Timestamp(
            week_ending.iloc[0]
        ).date()
    )

    n_weeks = len(
        merged_df
    )

    observed_fit = merged_df[
        FIT_TARGET
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Original observed mortality
    # -------------------------------------------------------------------------

    if RAW_TARGET in merged_df.columns:

        observed_raw = merged_df[
            RAW_TARGET
        ].to_numpy(
            dtype=float
        )

    else:

        observed_raw = observed_fit.copy()

    # -------------------------------------------------------------------------
    # Original reported outlier
    # -------------------------------------------------------------------------

    outlier_mask = (
        week_ending
        == pd.Timestamp(OUTLIER_WEEK)
    ).to_numpy()

    if not outlier_mask.any():

        print(
            f"[WARNING] Outlier week {OUTLIER_WEEK} "
            "not found in prepared data."
        )

    # =========================================================================
    # 2. INFECTION-TO-DEATH DELAY
    # =========================================================================

    f_delay = make_delay_pmf_weeks(
        kind=DELAY_KIND,
        max_weeks=DELAY_MAX_WEEKS,
        shape=DELAY_SHAPE,
        scale=DELAY_SCALE,
    )

    print("\n" + "=" * 90)

    print(
        "BAYESIAN POSTERIOR MODEL-FIT TRAJECTORIES (final_v2)"
    )

    print(
        "DARK PRESENTATION — TIMES NEW ROMAN"
    )

    print("=" * 90)

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

    # =========================================================================
    # 3. MAIN MODEL LOOP
    # =========================================================================

    all_model_summaries: dict[str, pd.DataFrame] = {}

    divergence_rows = []

    for model_index, model in enumerate(
        MODELS
    ):

        print(
            f"\n--- {model} ---"
        )

        # =====================================================================
        # 3a. LOAD FINAL POSTERIOR
        # =====================================================================

        saved = load_final_result(
            model
        )

        results = saved[
            "results"
        ]

        param_names = list(
            saved["param_names"]
        )

        logwt = get_result_array(
            results,
            "logwt"
        )

        logz = get_result_array(
            results,
            "logz"
        )

        logl = get_result_array(
            results,
            "logl"
        )

        samples = get_result_array(
            results,
            "samples"
        )

        # =====================================================================
        # 3b. REPRODUCIBLE EQUAL-WEIGHT POSTERIOR
        # =====================================================================

        resample_seed = (
            RANDOM_SEED
            + 100 * model_index
        )

        posterior = get_equal_weight_posterior(
            samples=samples,
            logwt=logwt,
            logz=logz,
            seed=resample_seed,
        )

        n_available = len(
            posterior
        )

        n_draws = min(
            N_POSTERIOR_DRAWS,
            n_available
        )

        # =====================================================================
        # 3c. REPRODUCIBLE POSTERIOR SUBSAMPLE
        # =====================================================================

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
            replace=False,
        )

        draws = posterior[
            draw_idx
        ]

        print(
            f"  Equal-weight posterior samples available: "
            f"{n_available}"
        )

        print(
            f"  Posterior draws used: {n_draws}"
        )

        # =====================================================================
        # 3d. SAVE EXACT POSTERIOR PARAMETER VECTORS
        # =====================================================================

        selected_draws_df = pd.DataFrame(
            draws,
            columns=param_names,
        )

        selected_draws_df.insert(
            0,
            "equal_weight_sample_index",
            draw_idx,
        )

        selected_draws_df.insert(
            0,
            "posterior_draw",
            np.arange(1, n_draws + 1),
        )

        selected_draws_path = (
            OUTPUT_DIR
            / f"selected_model_fit_draws_{model}.csv"
        )

        selected_draws_df.to_csv(
            selected_draws_path,
            index=False,
        )

        print(
            f"  [OK] Saved exact selected posterior draws:\n"
            f"    {selected_draws_path}"
        )

        # =====================================================================
        # 3e. POSTERIOR EXPECTED-DEATH TRAJECTORIES
        # =====================================================================

        mu_draws = np.full(
            (n_draws, n_weeks),
            np.nan,
            dtype=float,
        )

        for i, theta in enumerate(
            draws
        ):

            mu_draws[i, :] = compute_mu_trajectory(
                theta=theta,
                param_names=param_names,
                betaH_mode=model,
                merged_df=merged_df,
                zT=zT,
                ze=ze,
                zR=zR,
                start_date=start_date,
                f_delay=f_delay,
            )

            if (
                (i + 1) % 50 == 0
                or (i + 1) == n_draws
            ):

                print(
                    f"  processed {i + 1} / {n_draws}"
                )

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------

        if np.isnan(mu_draws).any():

            raise RuntimeError(
                f"{model}: posterior expected-death trajectories "
                "contain NaN values."
            )

        if not np.isfinite(mu_draws).all():

            raise RuntimeError(
                f"{model}: posterior expected-death trajectories "
                "contain non-finite values."
            )

        # ---------------------------------------------------------------------
        # Posterior weekly summaries
        # ---------------------------------------------------------------------

        mu_q025 = np.quantile(
            mu_draws,
            0.025,
            axis=0,
        )

        mu_band_median = np.quantile(
            mu_draws,
            0.50,
            axis=0,
        )

        mu_q975 = np.quantile(
            mu_draws,
            0.975,
            axis=0,
        )

        # =====================================================================
        # 3f. MAXIMUM-LIKELIHOOD TRAJECTORY
        # =====================================================================

        best_idx = int(
            np.nanargmax(logl)
        )

        theta_mle = np.asarray(
            samples[best_idx],
            dtype=float,
        )

        max_loglik = float(
            logl[best_idx]
        )

        mu_mle = compute_mu_trajectory(
            theta=theta_mle,
            param_names=param_names,
            betaH_mode=model,
            merged_df=merged_df,
            zT=zT,
            ze=ze,
            zR=zR,
            start_date=start_date,
            f_delay=f_delay,
        )

        mle_sse = sse(
            observed_fit,
            mu_mle,
        )

        # =====================================================================
        # 3g. POSTERIOR-MEDIAN-PARAMETER TRAJECTORY
        # =====================================================================

        theta_median_params = np.median(
            posterior,
            axis=0,
        )

        mu_median_params = compute_mu_trajectory(
            theta=theta_median_params,
            param_names=param_names,
            betaH_mode=model,
            merged_df=merged_df,
            zT=zT,
            ze=ze,
            zR=zR,
            start_date=start_date,
            f_delay=f_delay,
        )

        median_sse = sse(
            observed_fit,
            mu_median_params,
        )

        if mle_sse > 0:

            sse_ratio = (
                median_sse
                / mle_sse
            )

        else:

            sse_ratio = np.nan

        print(
            f"  MLE trajectory: "
            f"logL={max_loglik:.3f}, "
            f"SSE={mle_sse:.1f}"
        )

        print(
            f"  Posterior-median-parameter trajectory: "
            f"SSE={median_sse:.1f}, "
            f"ratio={sse_ratio:.2f}x MLE SSE"
        )

        if (
            np.isfinite(sse_ratio)
            and sse_ratio > 1.5
        ):

            print(
                "  [DIAGNOSTIC] Median-parameter trajectory fits "
                "notably worse than the MLE trajectory. This is "
                "consistent with correlated posterior parameters."
            )

        divergence_rows.append({

            "betaH_mode": model,

            "max_loglik": max_loglik,

            "mle_sse": mle_sse,

            "median_param_sse": median_sse,

            "median_over_mle_sse_ratio": sse_ratio,

        })

        # =====================================================================
        # 3h. SAVE MODEL TRAJECTORIES
        # =====================================================================

        model_df = pd.DataFrame({

            "week_ending": week_ending,

            "observed_fit": observed_fit,

            "observed_raw": observed_raw,

            "mu_q025": mu_q025,

            "mu_band_median": mu_band_median,

            "mu_q975": mu_q975,

            "mu_mle": mu_mle,

            "mu_median_params": mu_median_params,

        })

        csv_path = (
            OUTPUT_DIR
            / f"bayesian_model_fits_{model}.csv"
        )

        model_df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"  [OK] Saved:\n"
            f"    {csv_path}"
        )

        all_model_summaries[
            model
        ] = model_df

    # =========================================================================
    # 4. SAVE DIVERGENCE DIAGNOSTICS
    # =========================================================================

    divergence_df = pd.DataFrame(
        divergence_rows
    )

    divergence_csv = (
        OUTPUT_DIR
        / "mle_vs_median_divergence.csv"
    )

    divergence_df.to_csv(
        divergence_csv,
        index=False,
    )

    print("\n" + "=" * 90)

    print(
        "MLE vs. POSTERIOR-MEDIAN-PARAMETER DIVERGENCE"
    )

    print("=" * 90)

    print(
        divergence_df.to_string(
            index=False
        )
    )

    print(
        f"\n[OK] Saved:\n"
        f"  {divergence_csv}"
    )

    # =========================================================================
    # 5. DARK PRESENTATION FIGURE
    # =========================================================================

    plot_bayesian_model_fits(
        all_model_summaries=all_model_summaries,
        week_ending=week_ending,
        observed_raw=observed_raw,
        outlier_mask=outlier_mask,
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print("\n" + "=" * 90)

    print(
        "BAYESIAN POSTERIOR MODEL-FIT ANALYSIS COMPLETE"
    )

    print("=" * 90)

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