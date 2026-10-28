#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# analyse_epidemic_waves_and_amplification.py
# =============================================================================

"""
BOMBAY PLAGUE: EPIDEMIC WAVES AND AMPLIFICATION
DARK PRESENTATION VERSION — TIMES NEW ROMAN
=====================================================

Wave-by-wave analysis of the Bombay plague transmission dynamics.

GOAL
----
Examine whether the major epidemic waves show a repeated pattern
consistent with:

    Recurrent reservoir-associated spillover
        +
    Temporary supercritical prevalence-associated amplification
        (R0_H > 1)
        ->
    Epidemic expansion and mortality peaks.

ANALYSIS
--------
1. Identify major mortality waves.
2. Define wave boundaries using mortality troughs.
3. Separate ascending and declining phases.
4. Summarize inferred transmission quantities by phase.
5. Calculate the frequency of R0_H > 1.
6. Compare the timing of transmission and mortality peaks.
7. Save numerical summaries and diagnostic figures.

The analysis is descriptive. It does not independently establish
the biological transmission mechanism.

INPUT
-----
results/final_figures/transmission_dynamics/
    posterior_transmission_dynamics_all_climate.csv

NUMERICAL OUTPUT
----------------
results/final_figures/transmission_dynamics/wave_analysis/
    wave_phase_summary.csv
    wave_weekly_classification.csv
    wave_timing_summary.csv
    wave_boundary_weeks.csv

PRESENTATION FIGURES
--------------------
climate_analysis/figures/
    fig_wave_identification_dark.png
    fig_wave_identification_dark.pdf
    fig_wave_mechanism_panels_dark_monsoon.png
    fig_wave_mechanism_panels_dark_monsoon.pdf

STYLE
-----
- Times New Roman
- STIX mathematical notation
- Dark background #0B0D10
- White text
- Blue reservoir-associated spillover
- Orange prevalence-associated amplification
- Gold amplification reproduction number
- Monsoon hatching (June–September)
- High-resolution PNG and vector PDF

UPDATED Y-AXIS LABELS
---------------------
Panel (a): Deaths

Panel (b):
    lambda_R(t) = beta_R(t)

Panel (c):
    lambda_H(t) = beta_H(t) I(t)/N

Panel (d):
    R_0,H(t)

USAGE
-----
python climate_analysis/analyse_epidemic_waves_and_amplification.py

or

python scripts/analyse_epidemic_waves_and_amplification.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.signal import find_peaks


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
# 2. INPUT
# =============================================================================

TD_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
)

INPUT_CSV = (
    TD_DIR
    / "posterior_transmission_dynamics_all_climate.csv"
)


# =============================================================================
# 3. NUMERICAL OUTPUT
# =============================================================================

ANALYSIS_DIR = (
    TD_DIR
    / "wave_analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WAVE_SUMMARY_CSV = (
    ANALYSIS_DIR
    / "wave_phase_summary.csv"
)

WEEKLY_CLASSIFICATION_CSV = (
    ANALYSIS_DIR
    / "wave_weekly_classification.csv"
)

TIMING_CSV = (
    ANALYSIS_DIR
    / "wave_timing_summary.csv"
)

BOUNDARY_CSV = (
    ANALYSIS_DIR
    / "wave_boundary_weeks.csv"
)


# =============================================================================
# 4. PRESENTATION FIGURES
# =============================================================================

OUTPUT_DIR = CLIMATE_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DIAGNOSTIC_FIG_PNG = (
    OUTPUT_DIR
    / "fig_wave_identification_dark.png"
)

DIAGNOSTIC_FIG_PDF = (
    OUTPUT_DIR
    / "fig_wave_identification_dark.pdf"
)

MECHANISM_FIG_PNG = (
    OUTPUT_DIR
    / "fig_wave_mechanism_panels_dark_monsoon.png"
)

MECHANISM_FIG_PDF = (
    OUTPUT_DIR
    / "fig_wave_mechanism_panels_dark_monsoon.pdf"
)


# =============================================================================
# 5. WAVE-DETECTION SETTINGS
# =============================================================================

DEATHS_COL = "observed_deaths"

SMOOTH_WINDOW_WEEKS = 5

MIN_PEAK_DISTANCE_WEEKS = 20

MIN_LOG_PROMINENCE = 0.60

MIN_MAJOR_PEAK_DEATHS = 300.0

PEAK_SEARCH_HALF_WIDTH = 2


# =============================================================================
# 6. DARK PRESENTATION COLOURS
# =============================================================================

BG = "#0B0D10"

TEXT = "#FFFFFF"

GRID = "#414750"

SPINE = "#D1D5DB"

ANNOTATION_BG = "#171B21"

COLOR_DEATHS = "#D1D5DB"

COLOR_SMOOTH = "#38BDF8"

COLOR_SPILLOVER = "#38BDF8"

COLOR_AMPLIFICATION = "#FF8C42"

COLOR_R0 = "#FFB000"

COLOR_PEAK = "#FF6B6B"

COLOR_TROUGH = "#B0B7C3"

COLOR_ASCENDING = "#285443"

COLOR_DECLINING = "#493B56"

COLOR_THRESHOLD = "#FFFFFF"

COLOR_MONSOON = "#7FB3D5"

MONSOON_START_MONTH = 6

MONSOON_END_MONTH = 9


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

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ],

    "mathtext.fontset": "stix",

    "font.size": 12,

    "axes.labelsize": 14,

    "axes.titlesize": 16,

    "legend.fontsize": 11,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    "lines.linewidth": 2.0,

    "axes.linewidth": 1.0,

    "figure.facecolor": BG,

    "axes.facecolor": BG,

    "savefig.facecolor": BG,

    "text.color": TEXT,

    "axes.labelcolor": TEXT,

    "axes.edgecolor": SPINE,

    "xtick.color": TEXT,

    "ytick.color": TEXT,

    "grid.color": GRID,

    "xtick.direction": "out",

    "ytick.direction": "out",

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# =============================================================================
# 8. COMMON AXIS STYLE
# =============================================================================

def style_axis(ax):

    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE)

    ax.spines["bottom"].set_color(SPINE)

    ax.tick_params(
        axis="both",
        colors=TEXT,
        direction="out",
        length=5,
        width=1.0
    )

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.7,
        alpha=0.35,
        linestyle="--",
        zorder=0
    )

    ax.set_axisbelow(True)


# =============================================================================
# 9. TIME-AXIS FORMAT
# =============================================================================

def format_time_axis(ax):

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=4)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m")
    )

    ax.tick_params(
        axis="x",
        colors=TEXT,
        labelrotation=0
    )


# =============================================================================
# 10. LEGEND STYLE
# =============================================================================

def style_legend(legend):

    if legend is None:
        return

    frame = legend.get_frame()

    frame.set_facecolor(
        ANNOTATION_BG
    )

    frame.set_edgecolor(
        GRID
    )

    frame.set_alpha(
        0.95
    )

    for text in legend.get_texts():

        text.set_color(TEXT)


# =============================================================================
# 11. SUMMARY HELPERS
# =============================================================================

def summarize_phase(
    block: pd.DataFrame,
    wave_id: int,
    phase: str,
) -> dict:

    """
    Return descriptive statistics for one wave phase.

    Historical internal column names such as 'human_median'
    are retained for downstream compatibility.
    """

    total_foi = (
        block["spillover_median"]
        + block["human_median"]
    )

    amplification_fraction = np.where(
        total_foi > 0,
        block["human_median"] / total_foi,
        np.nan,
    )

    return {

        "wave": wave_id,

        "phase": phase,

        "start_date":
            block["week_ending"].min(),

        "end_date":
            block["week_ending"].max(),

        "n_weeks":
            len(block),

        "mean_deaths":
            block[DEATHS_COL].mean(),

        "max_deaths":
            block[DEATHS_COL].max(),

        "mean_spillover":
            block["spillover_median"].mean(),

        "median_spillover":
            block["spillover_median"].median(),

        "max_spillover":
            block["spillover_median"].max(),

        "mean_human_FOI":
            block["human_median"].mean(),

        "median_human_FOI":
            block["human_median"].median(),

        "max_human_FOI":
            block["human_median"].max(),

        "mean_human_fraction":
            np.nanmean(amplification_fraction),

        "median_human_fraction":
            np.nanmedian(amplification_fraction),

        "mean_R0_H":
            block["R0_H_median"].mean(),

        "median_R0_H":
            block["R0_H_median"].median(),

        "max_R0_H":
            block["R0_H_median"].max(),

        "n_R0_median_gt1":
            int(
                (
                    block["R0_H_median"] > 1.0
                ).sum()
            ),

        "fraction_R0_median_gt1":
            float(
                (
                    block["R0_H_median"] > 1.0
                ).mean()
            ),

        "n_R0_q025_gt1":
            int(
                (
                    block["R0_H_q025"] > 1.0
                ).sum()
            ),

        "fraction_R0_q025_gt1":
            float(
                (
                    block["R0_H_q025"] > 1.0
                ).mean()
            ),
    }


def nearest_index_of_max(
    block: pd.DataFrame,
    column: str,
) -> int:

    return int(
        block[column].idxmax()
    )


def date_diff_weeks(
    date_a,
    date_b,
) -> float:

    """
    Return date_a - date_b in weeks.

    Negative: transmission quantity peaks before mortality.
    Positive: transmission quantity peaks after mortality.
    """

    return (
        pd.Timestamp(date_a)
        - pd.Timestamp(date_b)
    ).days / 7.0


# =============================================================================
# 12. WAVE PHASE SHADING
# =============================================================================

def shade_wave_phases(
    ax,
    df,
    major_peaks,
    boundaries
):

    half_week = pd.Timedelta(
        days=3.5
    )

    for wave_num, peak_idx in enumerate(
        major_peaks,
        start=1
    ):

        start_idx = boundaries[
            wave_num - 1
        ]

        end_idx = boundaries[
            wave_num
        ]

        t_start = (
            df.loc[
                start_idx,
                "week_ending"
            ]
            - half_week
        )

        t_peak = (
            df.loc[
                peak_idx,
                "week_ending"
            ]
            + half_week
        )

        t_end = (
            df.loc[
                end_idx,
                "week_ending"
            ]
            + half_week
        )

        ax.axvspan(
            t_start,
            t_peak,
            color=COLOR_ASCENDING,
            alpha=0.30,
            linewidth=0,
            zorder=0
        )

        ax.axvspan(
            t_peak,
            t_end,
            color=COLOR_DECLINING,
            alpha=0.30,
            linewidth=0,
            zorder=0
        )

        ax.axvline(
            df.loc[
                peak_idx,
                "week_ending"
            ],
            color=COLOR_PEAK,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1
        )

    for boundary_idx in boundaries:

        ax.axvline(
            df.loc[
                boundary_idx,
                "week_ending"
            ],
            color=COLOR_TROUGH,
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            zorder=1
        )


# =============================================================================
# 13. MONSOON SHADING
# =============================================================================

def shade_monsoon(ax, df):

    """
    Hatch the Bombay monsoon months (June–September) of every year
    covered by the series.

    Hatching without an opaque fill keeps the ascending and declining
    phase shading visible underneath.
    """

    first_year = df["week_ending"].min().year

    last_year = df["week_ending"].max().year

    for year in range(first_year, last_year + 1):

        start = pd.Timestamp(
            year=year,
            month=MONSOON_START_MONTH,
            day=1
        )

        end = (
            pd.Timestamp(
                year=year,
                month=MONSOON_END_MONTH,
                day=1
            )
            + pd.offsets.MonthEnd(0)
        )

        start = max(
            start,
            df["week_ending"].min()
        )

        end = min(
            end,
            df["week_ending"].max()
        )

        if start >= end:
            continue

        ax.axvspan(
            start,
            end,
            facecolor="none",
            edgecolor=COLOR_MONSOON,
            hatch="///",
            linewidth=0,
            alpha=0.55,
            zorder=1
        )


# =============================================================================
# 14. SAVE FIGURES
# =============================================================================

def save_figure(
    fig,
    png_path,
    pdf_path
):

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor=BG
    )

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

    plt.close(fig)


# =============================================================================
# 15. MAIN
# =============================================================================

def main():

    # =========================================================================
    # LOAD INPUT
    # =========================================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}\n\n"
            "Run final_transmission_dynamics.py first."
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

    required = {

        "week_ending",

        DEATHS_COL,

        "spillover_median",

        "human_median",

        "R0_H_q025",

        "R0_H_median",

        "R0_H_q975",

    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) < 3:

        raise RuntimeError(
            "Input series is too short for wave analysis."
        )

    print("\n==========================================")
    print("BOMBAY PLAGUE")
    print("EPIDEMIC WAVES AND AMPLIFICATION")
    print("DARK PRESENTATION — TIMES NEW ROMAN")
    print("==========================================")

    print(
        f"\nInput: {INPUT_CSV}"
    )

    print(
        f"Numerical output: {ANALYSIS_DIR}"
    )

    print(
        f"Presentation figures: {OUTPUT_DIR}"
    )

    print(
        f"Number of weeks: {len(df)}"
    )

    # =========================================================================
    # DERIVED SERIES
    # =========================================================================

    df["log_deaths"] = np.log1p(
        df[DEATHS_COL].astype(float)
    )

    df["log_deaths_smooth"] = (
        df["log_deaths"]
        .rolling(
            window=SMOOTH_WINDOW_WEEKS,
            center=True,
            min_periods=1
        )
        .mean()
    )

    df["total_FOI_median"] = (
        df["spillover_median"]
        + df["human_median"]
    )

    df["human_fraction"] = np.where(
        df["total_FOI_median"] > 0,
        df["human_median"] / df["total_FOI_median"],
        np.nan
    )

    df["mortality_log_growth"] = (
        df["log_deaths"]
        - df["log_deaths"].shift(1)
    )

    # =========================================================================
    # DETECT MAJOR MORTALITY PEAKS
    # =========================================================================

    y = df[
        "log_deaths_smooth"
    ].to_numpy(dtype=float)

    candidate_peaks, _ = find_peaks(
        y,
        distance=MIN_PEAK_DISTANCE_WEEKS,
        prominence=MIN_LOG_PROMINENCE
    )

    major_peaks = []

    for idx in candidate_peaks:

        lo = max(
            0,
            idx - PEAK_SEARCH_HALF_WIDTH
        )

        hi = min(
            len(df),
            idx + PEAK_SEARCH_HALF_WIDTH + 1
        )

        local_block = df.loc[
            lo:hi - 1
        ]

        local_max = float(
            local_block[DEATHS_COL].max()
        )

        if local_max >= MIN_MAJOR_PEAK_DEATHS:

            local_idx = int(
                local_block[DEATHS_COL].idxmax()
            )

            major_peaks.append(
                local_idx
            )

    major_peaks = sorted(
        set(major_peaks)
    )

    if len(major_peaks) < 2:

        raise RuntimeError(
            "Fewer than two major waves were detected."
        )

    print("\nDetected major mortality peaks:")

    for j, idx in enumerate(
        major_peaks,
        start=1
    ):

        print(
            f"  Wave {j}: "
            f"{df.loc[idx, 'week_ending'].date()} "
            f"(deaths = {df.loc[idx, DEATHS_COL]:.1f})"
        )

    # =========================================================================
    # DEFINE WAVE BOUNDARIES
    # =========================================================================

    internal_troughs = []

    for left_peak, right_peak in zip(
        major_peaks[:-1],
        major_peaks[1:]
    ):

        segment = df.loc[
            left_peak:right_peak,
            "log_deaths_smooth"
        ]

        trough_idx = int(
            segment.idxmin()
        )

        internal_troughs.append(
            trough_idx
        )

    first_boundary = int(
        df.loc[
            :major_peaks[0],
            "log_deaths_smooth"
        ].idxmin()
    )

    last_boundary = int(
        df.loc[
            major_peaks[-1]:,
            "log_deaths_smooth"
        ].idxmin()
    )

    boundaries = (
        [first_boundary]
        + internal_troughs
        + [last_boundary]
    )

    if len(boundaries) != len(major_peaks) + 1:

        raise RuntimeError(
            "Unexpected number of wave boundaries."
        )

    print("\nWave boundaries:")

    for j, idx in enumerate(boundaries):

        print(
            f"  Boundary {j}: "
            f"{df.loc[idx, 'week_ending'].date()}"
        )

    # =========================================================================
    # CLASSIFY WAVE PHASES
    # =========================================================================

    df["wave"] = np.nan

    df["phase"] = "outside"

    phase_summaries = []

    timing_rows = []

    for wave_num, peak_idx in enumerate(
        major_peaks,
        start=1
    ):

        start_idx = boundaries[
            wave_num - 1
        ]

        end_idx = boundaries[
            wave_num
        ]

        if not start_idx <= peak_idx <= end_idx:

            raise RuntimeError(
                f"Peak for wave {wave_num} "
                "does not lie within its boundaries."
            )

        ascending = df.loc[
            start_idx:peak_idx
        ].copy()

        declining = df.loc[
            peak_idx + 1:end_idx
        ].copy()

        whole_wave = df.loc[
            start_idx:end_idx
        ].copy()

        df.loc[
            start_idx:peak_idx,
            "wave"
        ] = wave_num

        df.loc[
            start_idx:peak_idx,
            "phase"
        ] = "ascending"

        if peak_idx + 1 <= end_idx:

            df.loc[
                peak_idx + 1:end_idx,
                "wave"
            ] = wave_num

            df.loc[
                peak_idx + 1:end_idx,
                "phase"
            ] = "declining"

        # ---------------------------------------------------------------------
        # PHASE SUMMARIES
        # ---------------------------------------------------------------------

        phase_summaries.append(
            summarize_phase(
                ascending,
                wave_num,
                "ascending"
            )
        )

        if len(declining) > 0:

            phase_summaries.append(
                summarize_phase(
                    declining,
                    wave_num,
                    "declining"
                )
            )

        phase_summaries.append(
            summarize_phase(
                whole_wave,
                wave_num,
                "whole_wave"
            )
        )

        # ---------------------------------------------------------------------
        # TIMING OF MAXIMA
        # ---------------------------------------------------------------------

        mortality_peak_idx = nearest_index_of_max(
            whole_wave,
            DEATHS_COL
        )

        spillover_peak_idx = nearest_index_of_max(
            whole_wave,
            "spillover_median"
        )

        amplification_peak_idx = nearest_index_of_max(
            whole_wave,
            "human_median"
        )

        r0_peak_idx = nearest_index_of_max(
            whole_wave,
            "R0_H_median"
        )

        mortality_peak_date = df.loc[
            mortality_peak_idx,
            "week_ending"
        ]

        spillover_peak_date = df.loc[
            spillover_peak_idx,
            "week_ending"
        ]

        amplification_peak_date = df.loc[
            amplification_peak_idx,
            "week_ending"
        ]

        r0_peak_date = df.loc[
            r0_peak_idx,
            "week_ending"
        ]

        timing_rows.append({

            "wave": wave_num,

            "wave_start":
                df.loc[start_idx, "week_ending"],

            "mortality_peak_date":
                mortality_peak_date,

            "wave_end":
                df.loc[end_idx, "week_ending"],

            "spillover_peak_date":
                spillover_peak_date,

            "spillover_peak_value":
                df.loc[
                    spillover_peak_idx,
                    "spillover_median"
                ],

            "spillover_peak_relative_to_mortality_peak_weeks":
                date_diff_weeks(
                    spillover_peak_date,
                    mortality_peak_date
                ),

            "R0_peak_date":
                r0_peak_date,

            "R0_peak_value":
                df.loc[
                    r0_peak_idx,
                    "R0_H_median"
                ],

            "R0_peak_relative_to_mortality_peak_weeks":
                date_diff_weeks(
                    r0_peak_date,
                    mortality_peak_date
                ),

            "human_FOI_peak_date":
                amplification_peak_date,

            "human_FOI_peak_value":
                df.loc[
                    amplification_peak_idx,
                    "human_median"
                ],

            "human_FOI_peak_relative_to_mortality_peak_weeks":
                date_diff_weeks(
                    amplification_peak_date,
                    mortality_peak_date
                ),
        })

    # =========================================================================
    # SAVE NUMERICAL TABLES
    # =========================================================================

    summary_df = pd.DataFrame(
        phase_summaries
    )

    timing_df = pd.DataFrame(
        timing_rows
    )

    summary_df.to_csv(
        WAVE_SUMMARY_CSV,
        index=False
    )

    timing_df.to_csv(
        TIMING_CSV,
        index=False
    )

    boundaries_df = pd.DataFrame({

        "boundary_index":
            boundaries,

        "week_ending":
            df.loc[
                boundaries,
                "week_ending"
            ].values,

        "observed_deaths":
            df.loc[
                boundaries,
                DEATHS_COL
            ].values,

        "boundary_type": [
            (
                "internal_trough"
                if 0 < j < len(boundaries) - 1
                else "series_boundary"
            )
            for j in range(len(boundaries))
        ],
    })

    boundaries_df.to_csv(
        BOUNDARY_CSV,
        index=False
    )

    weekly_cols = [

        "week_ending",

        DEATHS_COL,

        "log_deaths_smooth",

        "mortality_log_growth",

        "wave",

        "phase",

        "spillover_median",

        "human_median",

        "total_FOI_median",

        "human_fraction",

        "R0_H_q025",

        "R0_H_median",

        "R0_H_q975",
    ]

    df[weekly_cols].to_csv(
        WEEKLY_CLASSIFICATION_CSV,
        index=False
    )

    # =========================================================================
    # PRINT PHASE-SPECIFIC RESULTS
    # =========================================================================

    print("\n" + "=" * 100)
    print("PHASE-SPECIFIC RESULTS")
    print("=" * 100)

    for wave_num in range(
        1,
        len(major_peaks) + 1
    ):

        sub = summary_df[
            (summary_df["wave"] == wave_num)
            &
            (
                summary_df["phase"].isin(
                    ["ascending", "declining"]
                )
            )
        ]

        print(f"\nWAVE {wave_num}")
        print("-" * 100)

        for _, row in sub.iterrows():

            print(
                f"\n{row['phase'].upper()}: "
                f"{pd.Timestamp(row['start_date']).date()} "
                f"to {pd.Timestamp(row['end_date']).date()} "
                f"({int(row['n_weeks'])} weeks)"
            )

            print(
                f"  median R_0,H                      = "
                f"{row['median_R0_H']:.3f}"
            )

            print(
                f"  weeks with median R_0,H > 1       = "
                f"{int(row['n_R0_median_gt1'])}/"
                f"{int(row['n_weeks'])} "
                f"({100 * row['fraction_R0_median_gt1']:.1f}%)"
            )

            print(
                f"  weeks with lower CrI R_0,H > 1    = "
                f"{int(row['n_R0_q025_gt1'])}/"
                f"{int(row['n_weeks'])} "
                f"({100 * row['fraction_R0_q025_gt1']:.1f}%)"
            )

            print(
                f"  mean spillover FOI                = "
                f"{row['mean_spillover']:.6g}"
            )

            print(
                f"  mean amplification FOI            = "
                f"{row['mean_human_FOI']:.6g}"
            )

            print(
                f"  median amplification fraction     = "
                f"{100 * row['median_human_fraction']:.1f}%"
            )

    # =========================================================================
    # PRINT TIMING
    # =========================================================================

    print("\n" + "=" * 100)
    print("TIMING OF WITHIN-WAVE MAXIMA")
    print("=" * 100)

    print(
        "\nNegative timing means the transmission quantity "
        "peaked BEFORE the mortality peak.\n"
    )

    for _, row in timing_df.iterrows():

        print(
            f"WAVE {int(row['wave'])}"
        )

        print(
            f"  mortality peak      : "
            f"{pd.Timestamp(row['mortality_peak_date']).date()}"
        )

        print(
            f"  spillover peak      : "
            f"{pd.Timestamp(row['spillover_peak_date']).date()} "
            f"("
            f"{row['spillover_peak_relative_to_mortality_peak_weeks']:+.1f} "
            f"weeks)"
        )

        print(
            f"  R_0,H peak          : "
            f"{pd.Timestamp(row['R0_peak_date']).date()} "
            f"("
            f"{row['R0_peak_relative_to_mortality_peak_weeks']:+.1f} "
            f"weeks)"
        )

        print(
            f"  amplification peak  : "
            f"{pd.Timestamp(row['human_FOI_peak_date']).date()} "
            f"("
            f"{row['human_FOI_peak_relative_to_mortality_peak_weeks']:+.1f} "
            f"weeks)"
        )

        print()

    # =========================================================================
    # ASCENDING VS DECLINING COMPARISON
    # =========================================================================

    print("=" * 100)
    print("ASCENDING VS DECLINING: KEY CHECK")
    print("=" * 100)

    for wave_num in range(
        1,
        len(major_peaks) + 1
    ):

        asc = summary_df[
            (summary_df["wave"] == wave_num)
            &
            (summary_df["phase"] == "ascending")
        ].iloc[0]

        dec_rows = summary_df[
            (summary_df["wave"] == wave_num)
            &
            (summary_df["phase"] == "declining")
        ]

        if len(dec_rows) == 0:
            continue

        dec = dec_rows.iloc[0]

        print(f"\nWave {wave_num}:")

        print(
            f"  R_0,H > 1 fraction: "
            f"ascending "
            f"{100 * asc['fraction_R0_median_gt1']:.1f}% "
            f"vs declining "
            f"{100 * dec['fraction_R0_median_gt1']:.1f}%"
        )

        print(
            f"  strong R_0,H > 1 fraction: "
            f"ascending "
            f"{100 * asc['fraction_R0_q025_gt1']:.1f}% "
            f"vs declining "
            f"{100 * dec['fraction_R0_q025_gt1']:.1f}%"
        )

        print(
            f"  median amplification FOI: "
            f"ascending "
            f"{asc['median_human_FOI']:.6g} "
            f"vs declining "
            f"{dec['median_human_FOI']:.6g}"
        )

        print(
            f"  median spillover FOI: "
            f"ascending "
            f"{asc['median_spillover']:.6g} "
            f"vs declining "
            f"{dec['median_spillover']:.6g}"
        )

    # =========================================================================
    # FIGURE 1: WAVE IDENTIFICATION
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(14.5, 6.0),
        dpi=300,
        facecolor=BG
    )

    shade_wave_phases(
        ax,
        df,
        major_peaks,
        boundaries
    )

    ax.plot(
        df["week_ending"],
        df[DEATHS_COL],
        color=COLOR_DEATHS,
        linewidth=1.8,
        label="Observed deaths",
        zorder=3
    )

    smooth_deaths = np.expm1(
        df["log_deaths_smooth"]
    )

    ax.plot(
        df["week_ending"],
        smooth_deaths,
        color=COLOR_SMOOTH,
        linewidth=2.0,
        linestyle="-.",
        label="5-week smoothed mortality",
        zorder=4
    )

    for wave_num, peak_idx in enumerate(
        major_peaks,
        start=1
    ):

        peak_date = df.loc[
            peak_idx,
            "week_ending"
        ]

        peak_deaths = df.loc[
            peak_idx,
            DEATHS_COL
        ]

        ax.scatter(
            peak_date,
            peak_deaths,
            color=COLOR_PEAK,
            s=55,
            zorder=6
        )

        ax.annotate(
            f"Wave {wave_num}",
            (peak_date, peak_deaths),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color=TEXT,
            zorder=10
        )

    ax.set_xlabel(
        "Week",
        fontsize=14,
        labelpad=12
    )

    ax.set_ylabel(
        "Weekly plague deaths",
        fontsize=14,
        labelpad=12
    )

    ax.set_title(
        "Identification of major Bombay plague mortality waves",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    legend_handles = [

        Line2D(
            [0], [0],
            color=COLOR_DEATHS,
            linewidth=1.8,
            label="Observed deaths"
        ),

        Line2D(
            [0], [0],
            color=COLOR_SMOOTH,
            linewidth=2.0,
            linestyle="-.",
            label="5-week smoothed mortality"
        ),

        Patch(
            facecolor=COLOR_ASCENDING,
            alpha=0.65,
            label="Ascending phase"
        ),

        Patch(
            facecolor=COLOR_DECLINING,
            alpha=0.65,
            label="Declining phase"
        ),

        Line2D(
            [0], [0],
            color=COLOR_PEAK,
            linestyle="--",
            linewidth=1.2,
            label="Mortality peak"
        ),

        Line2D(
            [0], [0],
            color=COLOR_TROUGH,
            linestyle=":",
            linewidth=1.2,
            label="Wave boundary"
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=11,
        borderaxespad=0,
        labelspacing=1.0
    )

    style_legend(legend)

    style_axis(ax)

    format_time_axis(ax)

    fig.subplots_adjust(
        left=0.09,
        right=0.75,
        bottom=0.16,
        top=0.87
    )

    save_figure(
        fig,
        DIAGNOSTIC_FIG_PNG,
        DIAGNOSTIC_FIG_PDF
    )

    # =========================================================================
    # FIGURE 2: MECHANISM PANELS
    # =========================================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14.5, 12.0),
        sharex=True,
        dpi=300,
        facecolor=BG,
        gridspec_kw={
            "hspace": 0.30
        }
    )

    # -------------------------------------------------------------------------
    # PANEL A: OBSERVED MORTALITY
    # -------------------------------------------------------------------------

    axes[0].plot(
        df["week_ending"],
        df[DEATHS_COL],
        linewidth=1.8,
        color=COLOR_DEATHS,
        zorder=3
    )

    axes[0].set_ylabel(
        "Deaths",
        fontsize=13,
        labelpad=12
    )

    axes[0].set_title(
        "(a) Observed mortality",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    # -------------------------------------------------------------------------
    # PANEL B: RESERVOIR-ASSOCIATED FORCE OF INFECTION
    # -------------------------------------------------------------------------

    axes[1].plot(
        df["week_ending"],
        df["spillover_median"],
        linewidth=2.0,
        color=COLOR_SPILLOVER,
        zorder=3
    )

    # UPDATED Y-AXIS LABEL

    axes[1].set_ylabel(
        r"$\lambda_R(t)=\beta_R(t)$",
        fontsize=15,
        labelpad=12
    )

    axes[1].set_title(
        "(b) Reservoir-associated force of infection",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    axes[1].ticklabel_format(
        style="sci",
        axis="y",
        scilimits=(0, 0)
    )

    # -------------------------------------------------------------------------
    # PANEL C: PREVALENCE-ASSOCIATED AMPLIFICATION FORCE OF INFECTION
    # -------------------------------------------------------------------------

    axes[2].plot(
        df["week_ending"],
        df["human_median"],
        linewidth=2.0,
        color=COLOR_AMPLIFICATION,
        zorder=3
    )

    # UPDATED Y-AXIS LABEL
    #
    # This is AFTER multiplying beta_H(t) by I(t)/N.

    axes[2].set_ylabel(
        r"$\lambda_H(t)=\beta_H(t)\frac{I(t)}{N}$",
        fontsize=15,
        labelpad=12
    )

    axes[2].set_title(
        "(c) Prevalence-associated amplification force of infection",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    axes[2].ticklabel_format(
        style="sci",
        axis="y",
        scilimits=(0, 0)
    )

    # -------------------------------------------------------------------------
    # PANEL D: AMPLIFICATION REPRODUCTION NUMBER
    # -------------------------------------------------------------------------

    axes[3].fill_between(
        df["week_ending"],
        df["R0_H_q025"],
        df["R0_H_q975"],
        color=COLOR_R0,
        alpha=0.23,
        linewidth=0,
        zorder=2
    )

    axes[3].plot(
        df["week_ending"],
        df["R0_H_median"],
        linewidth=2.4,
        color=COLOR_R0,
        zorder=3
    )

    axes[3].axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        color=COLOR_THRESHOLD,
        zorder=4
    )

    axes[3].set_ylabel(
        r"$R_{0,H}(t)$",
        fontsize=15,
        labelpad=12
    )

    axes[3].set_xlabel(
        "Week",
        fontsize=14,
        labelpad=12
    )

    axes[3].set_title(
        "(d) Amplification reproduction number",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    axes[3].set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # SHADING, PEAKS, BOUNDARIES AND AXIS STYLE
    # -------------------------------------------------------------------------

    for ax in axes:

        shade_wave_phases(
            ax,
            df,
            major_peaks,
            boundaries
        )

        style_axis(ax)

        format_time_axis(ax)

    shade_monsoon(
        axes[1],
        df
    )

    axes[-1].set_xlim(
        df["week_ending"].min() - pd.Timedelta(days=7),
        df["week_ending"].max() + pd.Timedelta(days=7)
    )

    for ax in axes[:-1]:

        ax.tick_params(
            labelbottom=False
        )

    # -------------------------------------------------------------------------
    # FIGURE LEGEND
    # -------------------------------------------------------------------------

    legend_handles = [

        Patch(
            facecolor=COLOR_ASCENDING,
            alpha=0.65,
            label="Ascending phase"
        ),

        Patch(
            facecolor=COLOR_DECLINING,
            alpha=0.65,
            label="Declining phase"
        ),

        Line2D(
            [0], [0],
            color=COLOR_PEAK,
            linestyle="--",
            linewidth=1.2,
            label="Mortality peak"
        ),

        Line2D(
            [0], [0],
            color=COLOR_TROUGH,
            linestyle=":",
            linewidth=1.2,
            label="Wave boundary"
        ),

        Line2D(
            [0], [0],
            color=COLOR_THRESHOLD,
            linestyle="--",
            linewidth=1.5,
            label=r"$R_{0,H}=1$"
        ),

        Patch(
            facecolor=COLOR_R0,
            alpha=0.25,
            label=r"95% credible interval for $R_{0,H}$"
        ),

        Patch(
            facecolor="none",
            edgecolor=COLOR_MONSOON,
            hatch="///",
            label="Monsoon (Jun–Sep), panel (b)"
        ),
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.77, 0.55),
        ncol=1,
        frameon=True,
        fontsize=11,
        labelspacing=1.2
    )

    style_legend(legend)

    fig.suptitle(
        "Bombay plague: epidemic waves and inferred transmission dynamics",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    # UPDATED LEFT MARGIN FOR MATHEMATICAL Y-AXIS LABELS

    fig.subplots_adjust(
        left=0.16,
        right=0.74,
        bottom=0.08,
        top=0.90,
        hspace=0.35
    )

    save_figure(
        fig,
        MECHANISM_FIG_PNG,
        MECHANISM_FIG_PDF
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)

    print(
        """
The most relevant pattern for the temporary prevalence-associated
amplification hypothesis would be repeated evidence across major waves
that:

  * Supercritical amplification is more frequent during ascending phases
    than during declining phases.

  * Prevalence-associated amplification is substantial during epidemic
    expansion.

  * The amplification reproduction number and/or amplification FOI peaks
    at or before the mortality peak.

  * Reservoir-associated spillover remains present, consistent with
    recurrent infection input.

R_0,H describes the reproduction potential of the prevalence-associated
amplification component in the mathematical model.

It does not independently identify a specific biological transmission route.

The inferred patterns are descriptive and should not be interpreted as
independent causal validation of the transmission mechanism.
"""
    )

    print("\n" + "=" * 100)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 100)

    print("\nNumerical outputs:")

    for path in [
        WAVE_SUMMARY_CSV,
        TIMING_CSV,
        BOUNDARY_CSV,
        WEEKLY_CLASSIFICATION_CSV
    ]:

        print(f"  {path}")

    print("\nPresentation figures:")

    for path in [
        DIAGNOSTIC_FIG_PNG,
        DIAGNOSTIC_FIG_PDF,
        MECHANISM_FIG_PNG,
        MECHANISM_FIG_PDF
    ]:

        print(f"  {path}")

    print("\nDone.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()