# =============================================================================
# analyse_epidemic_waves_and_amplification.py
# =============================================================================
"""
Wave-by-wave analysis of the Bombay plague transmission dynamics.

Goal
----
Test whether major epidemic waves show a repeated pattern consistent with:

    recurrent reservoir-associated spillover
        +
    temporary periods of supercritical prevalence-associated amplification
    (R0_H > 1)
        ->
    epidemic expansion / mortality peak

The script:
1. Reads posterior_transmission_dynamics_all_climate.csv.
2. Smooths observed mortality only for the purpose of identifying major waves.
3. Detects major mortality peaks objectively using scipy.signal.find_peaks.
4. Uses mortality troughs between major peaks to define wave boundaries.
5. Splits each wave into:
       ascending phase: trough/start -> peak
       declining phase: peak -> next trough/end
6. Summarizes, for each phase:
       - observed deaths
       - reservoir-associated spillover FOI
       - prevalence-associated amplification FOI
       - amplification fraction of total FOI
       - R0_H
       - fraction of weeks with median R0_H > 1
       - fraction of weeks with R0_H_q025 > 1
7. Reports timing of:
       - maximum spillover
       - maximum amplification FOI
       - maximum R0_H
       - mortality peak
8. Saves CSV summaries and diagnostic figures, including the actual
   wave-boundary (trough) weeks, so downstream robustness scripts can use
   pre-specified troughs instead of an ad hoc mortality threshold.

Important
---------
This is a descriptive analysis of inferred model quantities. It does not
constitute independent causal validation of the fitted mechanism.

The default settings are chosen to identify the three major mortality waves
in the 141-week fitted series while ignoring smaller secondary bumps.

This script contains no random sampling. Given the same input CSV and the
same settings, it should return the same wave detection and summaries on
every run.

Usage
-----
    python scripts/analyse_epidemic_waves_and_amplification.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.signal import find_peaks


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "posterior_transmission_dynamics_all_climate.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "transmission_dynamics"
    / "wave_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WAVE_SUMMARY_CSV = (
    OUTPUT_DIR
    / "wave_phase_summary.csv"
)

WEEKLY_CLASSIFICATION_CSV = (
    OUTPUT_DIR
    / "wave_weekly_classification.csv"
)

TIMING_CSV = (
    OUTPUT_DIR
    / "wave_timing_summary.csv"
)

BOUNDARY_CSV = (
    OUTPUT_DIR
    / "wave_boundary_weeks.csv"
)

DIAGNOSTIC_FIG_PNG = (
    OUTPUT_DIR
    / "wave_identification_diagnostic.png"
)

DIAGNOSTIC_FIG_PDF = (
    OUTPUT_DIR
    / "wave_identification_diagnostic.pdf"
)

MECHANISM_FIG_PNG = (
    OUTPUT_DIR
    / "wave_mechanism_panels.png"
)

MECHANISM_FIG_PDF = (
    OUTPUT_DIR
    / "wave_mechanism_panels.pdf"
)


# =============================================================================
# SETTINGS
# =============================================================================

DEATHS_COL = "observed_deaths"

# Smooth log-mortality for wave detection only.
SMOOTH_WINDOW_WEEKS = 5

# Peaks must be at least this many weeks apart.
MIN_PEAK_DISTANCE_WEEKS = 20

# Minimum prominence on smoothed log(1 + deaths).
MIN_LOG_PROMINENCE = 0.60

# Require a major peak to reach at least this observed weekly mortality.
# This excludes the smaller autumn-1898 bump and retains the three major waves.
MIN_MAJOR_PEAK_DEATHS = 300.0

# R0_H in the first weeks of the series is excluded from the R0_H summaries
# and timing (and week 1, an averaging artefact of the first partial week, is
# not plotted). Heavy late-monsoon rainfall raised R0_H above two in October
# 1896 while almost no one was infected, so those weeks say nothing about
# amplification during the first wave. Matches SI Table S13.
R0H_EXCLUDE_FIRST_WEEKS = 4


# =============================================================================
# HELPERS
# =============================================================================

def summarize_phase(
    block: pd.DataFrame,
    wave_id: int,
    phase: str,
) -> dict:
    """
    Return descriptive statistics for one wave phase.

    Note
    ----
    The input CSV retains the historical internal column names
    'human_median', etc. These represent prevalence-associated
    amplification in the current interpretation.
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

    r0 = block.dropna(subset=["R0_H_median"])

    return {
        "wave":
            wave_id,

        "phase":
            phase,

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

        # Keep historical column names for downstream compatibility.
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

        "n_weeks_R0":
            len(r0),

        "mean_R0_H":
            r0["R0_H_median"].mean(),

        "median_R0_H":
            r0["R0_H_median"].median(),

        "max_R0_H":
            r0["R0_H_median"].max(),

        "n_R0_median_gt1":
            int(
                (
                    r0["R0_H_median"] > 1.0
                ).sum()
            ),

        "fraction_R0_median_gt1":
            float(
                (
                    r0["R0_H_median"] > 1.0
                ).mean()
            ),

        "n_R0_q025_gt1":
            int(
                (
                    r0["R0_H_q025"] > 1.0
                ).sum()
            ),

        "fraction_R0_q025_gt1":
            float(
                (
                    r0["R0_H_q025"] > 1.0
                ).mean()
            ),
    }


def nearest_index_of_max(
    block: pd.DataFrame,
    column: str,
) -> int:
    """
    Return the DataFrame index label of the maximum value.
    """

    return int(
        block[column].idxmax()
    )


def date_diff_weeks(
    date_a,
    date_b,
) -> float:
    """
    Return date_a - date_b in weeks.

    Positive:
        quantity peaks after mortality peak.

    Negative:
        quantity peaks before mortality peak.
    """

    return (
        pd.Timestamp(date_a)
        - pd.Timestamp(date_b)
    ).days / 7.0


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # 1. LOAD INPUT
    # =========================================================================

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"  {INPUT_CSV}\n\n"
            "Run final_transmission_dynamics.py first."
        )

    df = pd.read_csv(
        INPUT_CSV,
        parse_dates=["week_ending"],
    )

    df = (
        df
        .sort_values("week_ending")
        .reset_index(drop=True)
    )

    # Keep an unmasked copy of R0_H for plotting (week 1 dropped below).
    r0_plot = df[["R0_H_q025", "R0_H_median", "R0_H_q975"]].copy()
    r0_plot.iloc[0] = np.nan

    required = {
        "week_ending",
        DEATHS_COL,
        "spillover_median",
        "human_median",
        "R0_H_q025",
        "R0_H_median",
        "R0_H_q975",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) < 3:

        raise RuntimeError(
            "Input series is too short for wave analysis."
        )

    print(
        "=" * 100
    )

    print(
        "WAVE-BY-WAVE PREVALENCE-ASSOCIATED "
        "AMPLIFICATION ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"\nInput: {INPUT_CSV}"
    )

    print(
        f"Weeks: {len(df)}"
    )

    print(
        "\nWave-detection settings:"
    )

    print(
        f"  smoothing window          = "
        f"{SMOOTH_WINDOW_WEEKS} weeks"
    )

    print(
        f"  minimum peak distance     = "
        f"{MIN_PEAK_DISTANCE_WEEKS} weeks"
    )

    print(
        f"  minimum log prominence    = "
        f"{MIN_LOG_PROMINENCE}"
    )

    print(
        f"  minimum major peak deaths = "
        f"{MIN_MAJOR_PEAK_DEATHS:g}"
    )


    # R0_H: exclude the first weeks of the series (see R0H_EXCLUDE_FIRST_WEEKS).
    r0_cols = ["R0_H_q025", "R0_H_median", "R0_H_q975"]
    df.loc[: R0H_EXCLUDE_FIRST_WEEKS - 1, r0_cols] = np.nan
    print(f"\nR0_H excluded for the first {R0H_EXCLUDE_FIRST_WEEKS} weeks "
          f"(up to {df.loc[R0H_EXCLUDE_FIRST_WEEKS - 1, 'week_ending'].date()}).")

    # =========================================================================
    # 2. DERIVED SERIES
    # =========================================================================

    df["log_deaths"] = np.log1p(
        df[DEATHS_COL].astype(float)
    )

    df["log_deaths_smooth"] = (
        df["log_deaths"]
        .rolling(
            window=SMOOTH_WINDOW_WEEKS,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    df["total_FOI_median"] = (
        df["spillover_median"]
        + df["human_median"]
    )

    df["human_fraction"] = np.where(
        df["total_FOI_median"] > 0,
        (
            df["human_median"]
            / df["total_FOI_median"]
        ),
        np.nan,
    )

    df["mortality_log_growth"] = (
        df["log_deaths"]
        - df["log_deaths"].shift(1)
    )


    # =========================================================================
    # 3. DETECT MAJOR PEAKS
    # =========================================================================

    y = (
        df["log_deaths_smooth"]
        .to_numpy(dtype=float)
    )

    candidate_peaks, _ = find_peaks(
        y,
        distance=MIN_PEAK_DISTANCE_WEEKS,
        prominence=MIN_LOG_PROMINENCE,
    )

    major_peaks = []

    for idx in candidate_peaks:

        lo = max(
            0,
            idx - 2,
        )

        hi = min(
            len(df),
            idx + 3,
        )

        local_block = df.loc[
            lo:hi - 1
        ]

        local_max = float(
            local_block[
                DEATHS_COL
            ].max()
        )

        if (
            local_max
            >= MIN_MAJOR_PEAK_DEATHS
        ):

            local_idx = int(
                local_block[
                    DEATHS_COL
                ].idxmax()
            )

            major_peaks.append(
                local_idx
            )

    major_peaks = sorted(
        set(major_peaks)
    )

    if len(major_peaks) < 2:

        raise RuntimeError(
            "Fewer than two major waves were detected. "
            "Inspect the wave-detection settings."
        )

    print(
        "\nDetected major mortality peaks:"
    )

    for j, idx in enumerate(
        major_peaks,
        start=1,
    ):

        print(
            f"  Wave {j}: "
            f"{df.loc[idx, 'week_ending'].date()} "
            f"(observed deaths = "
            f"{df.loc[idx, DEATHS_COL]:.1f})"
        )


    # =========================================================================
    # 4. DEFINE TROUGHS / WAVE BOUNDARIES
    # =========================================================================

    internal_troughs = []

    for (
        left_peak,
        right_peak,
    ) in zip(
        major_peaks[:-1],
        major_peaks[1:],
    ):

        segment = df.loc[
            left_peak:right_peak,
            "log_deaths_smooth",
        ]

        trough_idx = int(
            segment.idxmin()
        )

        internal_troughs.append(
            trough_idx
        )

    # First boundary
    first_segment = df.loc[
        :major_peaks[0],
        "log_deaths_smooth",
    ]

    first_boundary = int(
        first_segment.idxmin()
    )

    # Last boundary
    last_segment = df.loc[
        major_peaks[-1]:,
        "log_deaths_smooth",
    ]

    last_boundary = int(
        last_segment.idxmin()
    )

    boundaries = (
        [first_boundary]
        + internal_troughs
        + [last_boundary]
    )

    if (
        len(boundaries)
        != len(major_peaks) + 1
    ):

        raise RuntimeError(
            "Unexpected number of wave boundaries."
        )

    print(
        "\nWave boundaries / troughs:"
    )

    for j, idx in enumerate(
        boundaries
    ):

        boundary_type = (
            "internal trough"
            if (
                0 < j < len(boundaries) - 1
            )
            else "series boundary"
        )

        print(
            f"  Boundary {j}: "
            f"{df.loc[idx, 'week_ending'].date()} "
            f"(deaths = "
            f"{df.loc[idx, DEATHS_COL]:.1f}; "
            f"{boundary_type})"
        )


    # =========================================================================
    # 5. CLASSIFY WEEKS
    # =========================================================================

    df["wave"] = np.nan
    df["phase"] = "outside"

    phase_summaries = []
    timing_rows = []

    for (
        wave_num,
        peak_idx,
    ) in enumerate(
        major_peaks,
        start=1,
    ):

        start_idx = boundaries[
            wave_num - 1
        ]

        end_idx = boundaries[
            wave_num
        ]

        if not (
            start_idx
            <= peak_idx
            <= end_idx
        ):

            raise RuntimeError(
                f"Peak for wave {wave_num} "
                "does not lie within its boundaries."
            )

        # Ascending includes the peak.
        ascending = df.loc[
            start_idx:peak_idx
        ].copy()

        # Declining begins after the peak.
        declining = df.loc[
            peak_idx + 1:end_idx
        ].copy()

        whole_wave = df.loc[
            start_idx:end_idx
        ].copy()

        df.loc[
            start_idx:peak_idx,
            "wave",
        ] = wave_num

        df.loc[
            start_idx:peak_idx,
            "phase",
        ] = "ascending"

        if (
            peak_idx + 1
            <= end_idx
        ):

            df.loc[
                peak_idx + 1:end_idx,
                "wave",
            ] = wave_num

            df.loc[
                peak_idx + 1:end_idx,
                "phase",
            ] = "declining"


        # ---------------------------------------------------------------------
        # Phase summaries
        # ---------------------------------------------------------------------

        phase_summaries.append(
            summarize_phase(
                ascending,
                wave_num,
                "ascending",
            )
        )

        if len(declining) > 0:

            phase_summaries.append(
                summarize_phase(
                    declining,
                    wave_num,
                    "declining",
                )
            )

        phase_summaries.append(
            summarize_phase(
                whole_wave,
                wave_num,
                "whole_wave",
            )
        )


        # ---------------------------------------------------------------------
        # Timing of maxima
        # ---------------------------------------------------------------------

        mortality_peak_idx = (
            nearest_index_of_max(
                whole_wave,
                DEATHS_COL,
            )
        )

        spillover_peak_idx = (
            nearest_index_of_max(
                whole_wave,
                "spillover_median",
            )
        )

        amplification_peak_idx = (
            nearest_index_of_max(
                whole_wave,
                "human_median",
            )
        )

        r0_peak_idx = (
            nearest_index_of_max(
                whole_wave,
                "R0_H_median",
            )
        )

        mortality_peak_date = df.loc[
            mortality_peak_idx,
            "week_ending",
        ]

        spillover_peak_date = df.loc[
            spillover_peak_idx,
            "week_ending",
        ]

        amplification_peak_date = df.loc[
            amplification_peak_idx,
            "week_ending",
        ]

        r0_peak_date = df.loc[
            r0_peak_idx,
            "week_ending",
        ]

        timing_rows.append(
            {
                "wave":
                    wave_num,

                "wave_start":
                    df.loc[
                        start_idx,
                        "week_ending",
                    ],

                "mortality_peak_date":
                    mortality_peak_date,

                "wave_end":
                    df.loc[
                        end_idx,
                        "week_ending",
                    ],

                "spillover_peak_date":
                    spillover_peak_date,

                "spillover_peak_value":
                    df.loc[
                        spillover_peak_idx,
                        "spillover_median",
                    ],

                "spillover_peak_relative_to_mortality_peak_weeks":
                    date_diff_weeks(
                        spillover_peak_date,
                        mortality_peak_date,
                    ),

                "R0_peak_date":
                    r0_peak_date,

                "R0_peak_value":
                    df.loc[
                        r0_peak_idx,
                        "R0_H_median",
                    ],

                "R0_peak_relative_to_mortality_peak_weeks":
                    date_diff_weeks(
                        r0_peak_date,
                        mortality_peak_date,
                    ),

                # Historical names retained for compatibility.
                "human_FOI_peak_date":
                    amplification_peak_date,

                "human_FOI_peak_value":
                    df.loc[
                        amplification_peak_idx,
                        "human_median",
                    ],

                "human_FOI_peak_relative_to_mortality_peak_weeks":
                    date_diff_weeks(
                        amplification_peak_date,
                        mortality_peak_date,
                    ),
            }
        )


    # =========================================================================
    # 6. SAVE TABLES
    # =========================================================================

    summary_df = pd.DataFrame(
        phase_summaries
    )

    timing_df = pd.DataFrame(
        timing_rows
    )

    summary_df.to_csv(
        WAVE_SUMMARY_CSV,
        index=False,
    )

    timing_df.to_csv(
        TIMING_CSV,
        index=False,
    )

    boundaries_df = pd.DataFrame(
        {
            "boundary_index":
                boundaries,

            "week_ending":
                df.loc[
                    boundaries,
                    "week_ending",
                ].values,

            "observed_deaths":
                df.loc[
                    boundaries,
                    DEATHS_COL,
                ].values,

            "boundary_type":
                [
                    (
                        "internal_trough"
                        if (
                            0 < j < len(boundaries) - 1
                        )
                        else "series_boundary"
                    )
                    for j in range(
                        len(boundaries)
                    )
                ],
        }
    )

    boundaries_df.to_csv(
        BOUNDARY_CSV,
        index=False,
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

    df[
        weekly_cols
    ].to_csv(
        WEEKLY_CLASSIFICATION_CSV,
        index=False,
    )


    # =========================================================================
    # 7. PRINT PHASE-SPECIFIC RESULTS
    # =========================================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE-SPECIFIC RESULTS"
    )

    print(
        "=" * 100
    )

    for wave_num in range(
        1,
        len(major_peaks) + 1,
    ):

        sub = summary_df[
            (
                summary_df["wave"]
                == wave_num
            )
            &
            (
                summary_df["phase"]
                .isin(
                    [
                        "ascending",
                        "declining",
                    ]
                )
            )
        ]

        print(
            f"\nWAVE {wave_num}"
        )

        print(
            "-" * 100
        )

        for _, row in sub.iterrows():

            print(
                f"\n{row['phase'].upper()}: "
                f"{pd.Timestamp(row['start_date']).date()} "
                f"to "
                f"{pd.Timestamp(row['end_date']).date()} "
                f"({int(row['n_weeks'])} weeks)"
            )

            print(
                f"  median R_0,H                      = "
                f"{row['median_R0_H']:.3f}"
            )

            print(
                f"  weeks with median R_0,H > 1       = "
                f"{int(row['n_R0_median_gt1'])}/"
                f"{int(row['n_weeks_R0'])} "
                f"({100 * row['fraction_R0_median_gt1']:.1f}%)"
            )

            print(
                f"  weeks with lower CrI R_0,H > 1    = "
                f"{int(row['n_R0_q025_gt1'])}/"
                f"{int(row['n_weeks_R0'])} "
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
    # 8. PRINT TIMING
    # =========================================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TIMING OF WITHIN-WAVE MAXIMA"
    )

    print(
        "=" * 100
    )

    print(
        "\nNegative relative timing means the "
        "transmission quantity peaked BEFORE "
        "the mortality peak.\n"
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
    # 9. ASCENDING VS DECLINING COMPARISON
    # =========================================================================

    print(
        "=" * 100
    )

    print(
        "ASCENDING VS DECLINING: KEY CHECK"
    )

    print(
        "=" * 100
    )

    for wave_num in range(
        1,
        len(major_peaks) + 1,
    ):

        asc = summary_df[
            (
                summary_df["wave"]
                == wave_num
            )
            &
            (
                summary_df["phase"]
                == "ascending"
            )
        ].iloc[0]

        dec_rows = summary_df[
            (
                summary_df["wave"]
                == wave_num
            )
            &
            (
                summary_df["phase"]
                == "declining"
            )
        ]

        if len(dec_rows) == 0:
            continue

        dec = dec_rows.iloc[0]

        print(
            f"\nWave {wave_num}:"
        )

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
    # 10. FIGURE 1: DIAGNOSTIC WAVE IDENTIFICATION
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        df["week_ending"],
        df[DEATHS_COL],
        label="Observed deaths",
        linewidth=1.2,
    )

    smooth_deaths = np.expm1(
        df["log_deaths_smooth"]
    )

    ax.plot(
        df["week_ending"],
        smooth_deaths,
        label=(
            f"{SMOOTH_WINDOW_WEEKS}-week "
            "smoothed mortality"
        ),
        linewidth=2,
    )

    for (
        wave_num,
        peak_idx,
    ) in enumerate(
        major_peaks,
        start=1,
    ):

        ax.axvline(
            df.loc[
                peak_idx,
                "week_ending",
            ],
            linestyle="--",
            linewidth=1,
        )

        ax.text(
            df.loc[
                peak_idx,
                "week_ending",
            ],
            df.loc[
                peak_idx,
                DEATHS_COL,
            ],
            f"  Wave {wave_num}",
            va="bottom",
        )

    for boundary_idx in boundaries:

        ax.axvline(
            df.loc[
                boundary_idx,
                "week_ending",
            ],
            linestyle=":",
            linewidth=1,
        )

    ax.set_xlabel(
        "Week"
    )

    ax.set_ylabel(
        "Observed plague deaths"
    )

    ax.set_title(
        "Identification of major Bombay plague mortality waves"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DIAGNOSTIC_FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        DIAGNOSTIC_FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # =========================================================================
    # 11. FIGURE 2: MECHANISM PANELS
    # =========================================================================

    plt.rcParams.update({
        "font.family":
            "Times New Roman",

        "font.size":
            10,

        "axes.labelsize":
            10,

        "axes.titlesize":
            10,

        "legend.fontsize":
            9,

        "xtick.labelsize":
            9,

        "ytick.labelsize":
            9,

        "axes.linewidth":
            0.9,
    })

    COLOR_DEATHS = "#333333"
    COLOR_SPILLOVER = "#1f77b4"
    COLOR_AMPLIFICATION = "#E08214"
    COLOR_R0 = "#8B2FA0"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 11),
        sharex=True,
    )


    # -------------------------------------------------------------------------
    # Panel a
    # -------------------------------------------------------------------------

    axes[0].plot(
        df["week_ending"],
        df[DEATHS_COL],
        linewidth=1.4,
        color=COLOR_DEATHS,
    )

    axes[0].set_ylabel(
        "Weekly deaths"
    )

    axes[0].set_title(
        "(A) Recorded deaths",
        loc="left",
        fontweight="bold",
    )


    # -------------------------------------------------------------------------
    # Panel b
    # -------------------------------------------------------------------------

    axes[1].plot(
        df["week_ending"],
        df["spillover_median"],
        linewidth=1.6,
        color=COLOR_SPILLOVER,
    )

    axes[1].set_ylabel(
        r"Force of infection (week$^{-1}$)"
    )

    axes[1].set_title(
        r"(B) Spillover, $\beta_R(t)$",
        loc="left",
        fontweight="bold",
    )

    axes[1].ticklabel_format(
        style="sci",
        axis="y",
        scilimits=(0, 0),
    )


    # -------------------------------------------------------------------------
    # Panel c
    # -------------------------------------------------------------------------

    axes[2].plot(
        df["week_ending"],
        df["human_median"],
        linewidth=1.6,
        color=COLOR_AMPLIFICATION,
    )

    axes[2].set_ylabel(
        r"Force of infection (week$^{-1}$)"
    )

    axes[2].set_title(
        r"(C) Amplification, $\beta_H(t)I(t)/N$",
        loc="left",
        fontweight="bold",
    )

    axes[2].ticklabel_format(
        style="sci",
        axis="y",
        scilimits=(0, 0),
    )


    # -------------------------------------------------------------------------
    # Panel d
    # -------------------------------------------------------------------------

    axes[3].fill_between(
        df["week_ending"],
        r0_plot["R0_H_q025"],
        r0_plot["R0_H_q975"],
        alpha=0.20,
        color=COLOR_R0,
        linewidth=0,
    )

    axes[3].plot(
        df["week_ending"],
        r0_plot["R0_H_median"],
        linewidth=1.6,
        color=COLOR_R0,
    )

    axes[3].axhline(
        1.0,
        linestyle="--",
        linewidth=1.1,
        color="gray",
        label=r"$\mathcal{R}_{0,H}=1$",
    )

    axes[3].set_ylabel(
        r"$\mathcal{R}_{0,H}$"
    )

    axes[3].set_title(
        "(D) Amplification reproduction number",
        loc="left",
        fontweight="bold",
    )

    axes[3].legend(
        loc="upper right",
        frameon=False,
    )


    # -------------------------------------------------------------------------
    # Peaks and boundaries
    # -------------------------------------------------------------------------

    for ax in axes:

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
            axis="y",
            alpha=0.25,
            linewidth=0.6,
        )

        for peak_idx in major_peaks:

            ax.axvline(
                df.loc[
                    peak_idx,
                    "week_ending",
                ],
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                color="black",
            )

        for boundary_idx in boundaries:

            ax.axvline(
                df.loc[
                    boundary_idx,
                    "week_ending",
                ],
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
                color="black",
            )


    legend_lines = [

        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="Mortality peak",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
            label="Wave boundary",
        ),
    ]

    fig.legend(
        handles=legend_lines,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()

    fig.savefig(
        MECHANISM_FIG_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        MECHANISM_FIG_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # =========================================================================
    # 12. FINAL NOTES
    # =========================================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INTERPRETATION GUIDE"
    )

    print(
        "=" * 100
    )

    print(
        """
The most relevant pattern for the temporary prevalence-associated
amplification hypothesis would be repeated evidence across the major waves
that:

  * supercritical amplification is more frequent during the ascending phase
    than during the declining phase;

  * prevalence-associated amplification is substantial during the
    ascending phase;

  * the amplification reproduction number and/or amplification FOI peaks
    at or before the mortality peak;

  * reservoir-associated spillover remains present, consistent with
    recurrent input into the epidemic system.

The quantity R_0,H describes the reproduction potential of the
prevalence-associated amplification component in the mathematical model.
It does not by itself identify a specific biological transmission route.

Do not require spillover to peak before R_0,H. Spillover supplies infections,
whereas R_0,H describes the capacity for prevalence-associated amplification.

If different waves show different patterns, report that heterogeneity rather
than forcing all waves into a single mechanism.
"""
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {WAVE_SUMMARY_CSV}"
    )

    print(
        f"  {TIMING_CSV}"
    )

    print(
        f"  {BOUNDARY_CSV}"
    )

    print(
        f"  {WEEKLY_CLASSIFICATION_CSV}"
    )

    print(
        f"  {DIAGNOSTIC_FIG_PNG}"
    )

    print(
        f"  {DIAGNOSTIC_FIG_PDF}"
    )

    print(
        f"  {MECHANISM_FIG_PNG}"
    )

    print(
        f"  {MECHANISM_FIG_PDF}"
    )

    print(
        "\nDone."
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()