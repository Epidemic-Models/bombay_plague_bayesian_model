# =============================================================================
# plot_wave_phases_and_R0_criteria.py
# =============================================================================
"""
Presentation figures explaining (1) how epidemic waves and their ascending /
declining phases are defined, and (2) the two criteria used to call a week
"supercritical" for prevalence-associated amplification.

This script does NOT refit or re-simulate anything. It reads the weekly
posterior summary written by final_transmission_dynamics.py:

    results/final_figures/transmission_dynamics/
        posterior_transmission_dynamics_all_climate.csv

and re-applies the SAME wave-detection rule as
analyse_epidemic_waves_and_amplification.py (same settings, below). If that
script's wave_boundary_weeks.csv exists, the boundaries found here are
checked against it, and the script stops if they disagree.

Wave definition (identical to the manuscript Methods)
-----------------------------------------------------
  1. y_t = log(1 + observed deaths), smoothed with a centred 5-week moving
     average (used ONLY to find waves, never for fitting).
  2. Candidate peaks: scipy.signal.find_peaks on the smoothed series,
     distance >= 20 weeks, prominence >= 0.60 (log scale).
  3. A candidate is a MAJOR wave only if observed deaths reach >= 300
     within +/-2 weeks; the peak week is the week of maximum observed
     deaths in that window.
  4. Boundaries: minimum of the smoothed series between consecutive major
     peaks; before the first and after the last peak for the outer ones.
  5. Ascending phase  = start boundary ... peak week (inclusive).
     Declining phase  = week after peak ... next boundary.

Supercritical criteria
----------------------
  lenient : posterior MEDIAN of R0_H(t) > 1
  strict  : lower 2.5% posterior quantile of R0_H(t) > 1
            (the whole 95% credible band lies above 1)

Outputs
-------
results/final_figures/transmission_dynamics/presentation/
    fig_wave_anatomy.(png|pdf)
    fig_R0H_criteria.(png|pdf)
    fig_waves_and_R0H_combined.(png|pdf)
    wave_phase_R0H_counts.csv      <- the percentages shown on the figures

Usage
-----
    python scripts/plot_wave_phases_and_R0_criteria.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TD_DIR = PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
INPUT_CSV = TD_DIR / "posterior_transmission_dynamics_all_climate.csv"
BOUNDARY_CSV = TD_DIR / "wave_analysis" / "wave_boundary_weeks.csv"   # optional cross-check

OUTPUT_DIR = TD_DIR / "presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Wave-detection settings -- MUST match analyse_epidemic_waves_and_amplification.py
# =============================================================================

DEATHS_COL = "observed_deaths"
SMOOTH_WINDOW_WEEKS = 5
MIN_PEAK_DISTANCE_WEEKS = 20
MIN_LOG_PROMINENCE = 0.60
MIN_MAJOR_PEAK_DEATHS = 300.0
PEAK_SEARCH_HALF_WIDTH = 2          # +/- weeks around a candidate peak


# =============================================================================
# Plot styling (matches final_transmission_dynamics.py)
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLOR_ASCENDING = "#F4A582"     # light orange
COLOR_DECLINING = "#92C5DE"     # light blue
COLOR_DEATHS = "#333333"
COLOR_SMOOTH = "#7F7F7F"
COLOR_R0 = "#E08214"            # same orange as amplification elsewhere
COLOR_PEAK = "#B2182B"
COLOR_TROUGH = "#555555"


# =============================================================================
# Wave detection (same rule as analyse_epidemic_waves_and_amplification.py)
# =============================================================================

def detect_waves(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return (major_peak_indices, boundary_indices) as row positions in df."""

    df["log_deaths"] = np.log1p(df[DEATHS_COL].astype(float))
    df["log_deaths_smooth"] = (
        df["log_deaths"]
        .rolling(window=SMOOTH_WINDOW_WEEKS, center=True, min_periods=1)
        .mean()
    )

    y = df["log_deaths_smooth"].to_numpy(dtype=float)
    candidates, _ = find_peaks(
        y, distance=MIN_PEAK_DISTANCE_WEEKS, prominence=MIN_LOG_PROMINENCE
    )

    major_peaks = []
    for idx in candidates:
        lo = max(0, idx - PEAK_SEARCH_HALF_WIDTH)
        hi = min(len(df), idx + PEAK_SEARCH_HALF_WIDTH + 1)
        block = df.loc[lo:hi - 1]
        if float(block[DEATHS_COL].max()) >= MIN_MAJOR_PEAK_DEATHS:
            major_peaks.append(int(block[DEATHS_COL].idxmax()))
    major_peaks = sorted(set(major_peaks))

    if len(major_peaks) < 2:
        raise RuntimeError("Fewer than two major waves detected; check settings.")

    internal = [
        int(df.loc[a:b, "log_deaths_smooth"].idxmin())
        for a, b in zip(major_peaks[:-1], major_peaks[1:])
    ]
    first = int(df.loc[:major_peaks[0], "log_deaths_smooth"].idxmin())
    last = int(df.loc[major_peaks[-1]:, "log_deaths_smooth"].idxmin())

    return major_peaks, [first] + internal + [last]


def cross_check_boundaries(df: pd.DataFrame, boundaries: list[int]) -> None:
    """Stop if boundaries differ from the wave-analysis script's saved output."""

    if not BOUNDARY_CSV.exists():
        print(f"[note] {BOUNDARY_CSV.name} not found -- skipping cross-check.")
        return

    saved = pd.read_csv(BOUNDARY_CSV, parse_dates=["week_ending"])
    mine = pd.to_datetime(df.loc[boundaries, "week_ending"]).reset_index(drop=True)
    theirs = saved["week_ending"].reset_index(drop=True)

    if len(mine) != len(theirs) or not (mine.values == theirs.values).all():
        raise RuntimeError(
            "Wave boundaries differ from wave_boundary_weeks.csv:\n"
            f"  here : {[d.date() for d in mine]}\n"
            f"  saved: {[d.date() for d in theirs]}\n"
            "Check that the settings above match "
            "analyse_epidemic_waves_and_amplification.py."
        )
    print("[OK] Wave boundaries match wave_boundary_weeks.csv.")


def classify_phases(df: pd.DataFrame, peaks: list[int], bounds: list[int]) -> pd.DataFrame:
    """Add 'wave' and 'phase' columns (ascending includes the peak week)."""

    df["wave"] = np.nan
    df["phase"] = "outside"
    for w, pk in enumerate(peaks, start=1):
        start, end = bounds[w - 1], bounds[w]
        df.loc[start:pk, ["wave", "phase"]] = [w, "ascending"]
        if pk + 1 <= end:
            df.loc[pk + 1:end, ["wave", "phase"]] = [w, "declining"]
    return df


def phase_blocks(df: pd.DataFrame, peaks: list[int], bounds: list[int]):
    """Yield (wave, phase, block) using index ranges, exactly as the wave script.

    A boundary (trough) week belongs to BOTH the declining phase of the
    earlier wave and the ascending phase of the next one, matching
    analyse_epidemic_waves_and_amplification.py and the manuscript numbers.
    """
    for w, pk in enumerate(peaks, start=1):
        start, end = bounds[w - 1], bounds[w]
        yield w, "ascending", df.loc[start:pk]
        if pk + 1 <= end:
            yield w, "declining", df.loc[pk + 1:end]


def phase_counts(df: pd.DataFrame, peaks: list[int], bounds: list[int]) -> pd.DataFrame:
    """Percentage of weeks meeting each supercritical criterion, per wave/phase."""

    rows = []
    for w, ph, block in phase_blocks(df, peaks, bounds):
        rows.append({
            "wave": w,
            "phase": ph,
            "start": block["week_ending"].min().date(),
            "end": block["week_ending"].max().date(),
            "n_weeks": len(block),
            "pct_median_gt1": 100.0 * (block["R0_H_median"] > 1).mean(),
            "pct_q025_gt1": 100.0 * (block["R0_H_q025"] > 1).mean(),
            "median_R0H": float(block["R0_H_median"].median()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================

def shade_phases(ax, df: pd.DataFrame, peaks: list[int], bounds: list[int]) -> None:
    """Shade ascending/declining phases and draw peak and trough lines."""

    half_week = pd.Timedelta(days=3.5)
    for w, pk in enumerate(peaks, start=1):
        t_start = df.loc[bounds[w - 1], "week_ending"] - half_week
        t_peak = df.loc[pk, "week_ending"] + half_week
        t_end = df.loc[bounds[w], "week_ending"] + half_week
        ax.axvspan(t_start, t_peak, color=COLOR_ASCENDING, alpha=0.35, lw=0, zorder=0)
        ax.axvspan(t_peak, t_end, color=COLOR_DECLINING, alpha=0.35, lw=0, zorder=0)
        ax.axvline(df.loc[pk, "week_ending"], color=COLOR_PEAK, ls="--", lw=1.0, zorder=1)
    for b in bounds:
        ax.axvline(df.loc[b, "week_ending"], color=COLOR_TROUGH, ls=":", lw=1.0, zorder=1)


def format_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def phase_legend_handles() -> list:
    return [
        Patch(facecolor=COLOR_ASCENDING, alpha=0.5, label="Ascending phase (trough → peak)"),
        Patch(facecolor=COLOR_DECLINING, alpha=0.5, label="Declining phase (peak → trough)"),
        Line2D([0], [0], color=COLOR_PEAK, ls="--", lw=1.0, label="Major mortality peak"),
        Line2D([0], [0], color=COLOR_TROUGH, ls=":", lw=1.0, label="Wave boundary (trough)"),
    ]


def draw_wave_anatomy(ax, df, peaks, bounds) -> None:
    shade_phases(ax, df, peaks, bounds)
    ax.plot(df["week_ending"], df[DEATHS_COL], color=COLOR_DEATHS, lw=1.4, zorder=3)
    smooth_deaths = np.expm1(df["log_deaths_smooth"])
    ax.plot(df["week_ending"], smooth_deaths, color=COLOR_SMOOTH, lw=1.0, ls="-.", zorder=2)
    ax.scatter(df.loc[peaks, "week_ending"], df.loc[peaks, DEATHS_COL],
               color=COLOR_PEAK, s=30, zorder=4)
    for w, pk in enumerate(peaks, start=1):
        ax.annotate(f"Wave {w}", (df.loc[pk, "week_ending"], df.loc[pk, DEATHS_COL]),
                    xytext=(6, 4), textcoords="offset points", fontsize=10)
    ax.set_ylabel("Weekly plague deaths")
    handles = [
        Line2D([0], [0], color=COLOR_DEATHS, lw=1.4, label="Observed deaths"),
        Line2D([0], [0], color=COLOR_SMOOTH, lw=1.0, ls="-.",
               label="5-week smoothed (wave detection only)"),
    ] + phase_legend_handles()
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=9)
    format_time_axis(ax)


def draw_R0_criteria(ax, df, peaks, bounds, counts: pd.DataFrame) -> None:
    shade_phases(ax, df, peaks, bounds)
    t = df["week_ending"]
    ax.fill_between(t, df["R0_H_q025"], df["R0_H_q975"], color=COLOR_R0, alpha=0.25,
                    lw=0, zorder=2)
    ax.plot(t, df["R0_H_median"], color=COLOR_R0, lw=1.6, zorder=3)
    ax.axhline(1.0, color="black", lw=1.0, zorder=3)

    lenient = df["R0_H_median"] > 1
    strict = df["R0_H_q025"] > 1
    ax.scatter(t[lenient], df.loc[lenient, "R0_H_median"], s=18, facecolors="none",
               edgecolors="black", lw=0.8, zorder=4)
    ax.scatter(t[strict], df.loc[strict, "R0_H_median"], s=18, color="black", zorder=5)

    # Per-phase percentages, written inside each shaded block
    ymax = float(np.nanmax(df["R0_H_q975"]))
    for _, r in counts.iterrows():
        block = df[(df["week_ending"].dt.date >= r["start"])
                   & (df["week_ending"].dt.date <= r["end"])]
        x_mid = block["week_ending"].iloc[len(block) // 2]
        ax.text(x_mid, ymax * 0.97,
                f"{r['pct_median_gt1']:.0f}%\n({r['pct_q025_gt1']:.0f}%)",
                ha="center", va="top", fontsize=9)

    ax.set_ylabel(r"$\mathscr{R}_{0,H}(t)$" if plt.rcParams.get("text.usetex") else r"$R_{0,H}(t)$")
    handles = [
        Line2D([0], [0], color=COLOR_R0, lw=1.6, label=r"Posterior median $R_{0,H}$"),
        Patch(facecolor=COLOR_R0, alpha=0.25, label="95% credible band"),
        Line2D([0], [0], color="black", lw=1.0, label=r"$R_{0,H}=1$"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor="black",
               label="Lenient: median > 1"),
        Line2D([0], [0], marker="o", ls="", color="black",
               label="Strict: lower 2.5% bound > 1"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.text(0.995, 0.02, "Numbers per block: % weeks lenient (strict)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, style="italic")
    format_time_axis(ax)


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"[OK] Saved {path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV}\nRun scripts/final_transmission_dynamics.py first."
        )

    df = pd.read_csv(INPUT_CSV, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    peaks, bounds = detect_waves(df)
    cross_check_boundaries(df, bounds)
    df = classify_phases(df, peaks, bounds)

    print("\nMajor peaks:", [df.loc[p, "week_ending"].date() for p in peaks])
    print("Boundaries :", [df.loc[b, "week_ending"].date() for b in bounds])

    counts = phase_counts(df, peaks, bounds)
    counts_path = OUTPUT_DIR / "wave_phase_R0H_counts.csv"
    counts.to_csv(counts_path, index=False)
    print("\n" + counts.to_string(index=False))
    print(f"[OK] Saved {counts_path}")

    # Figure 1: wave anatomy
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    draw_wave_anatomy(ax, df, peaks, bounds)
    ax.set_title("How waves and phases are defined")
    save(fig, "fig_wave_anatomy")

    # Figure 2: R0_H criteria
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    draw_R0_criteria(ax, df, peaks, bounds, counts)
    ax.set_title("When is amplification supercritical? Two criteria")
    save(fig, "fig_R0H_criteria")

    # Figure 3: both, stacked on a shared time axis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    draw_wave_anatomy(a1, df, peaks, bounds)
    draw_R0_criteria(a2, df, peaks, bounds, counts)
    a1.set_title("Wave phases (top) and supercritical amplification (bottom)")
    fig.tight_layout()
    save(fig, "fig_waves_and_R0H_combined")


if __name__ == "__main__":
    main()