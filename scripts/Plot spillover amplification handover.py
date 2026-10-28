# =============================================================================
# plot_spillover_amplification_handover.py
# =============================================================================
"""
Three stacked panels on one weekly time axis (climate-driven model), with the
ascending / declining wave phases shaded behind all of them:

  (a) Observed weekly plague deaths (bars) -- what we actually observe.
  (b) The two forces of infection on ONE axis (same units, week^-1):
        reservoir-associated spillover  beta_R(t)
        prevalence-associated amplification  beta_H(t) * I(t) / N
      Because the amplification force contains I(t), this panel shows WHEN
      amplification actually contributed (enough infectious people), and how
      it hands over from spillover during each wave.
  (c) The amplification reproduction number R0_H(t) = beta_H(t) / gamma with
      its 95% band and the line at 1 -- the climate-set POTENTIAL, which does
      not contain I(t).

Reading the figure: (c) says when conditions allowed amplification to be
self-sustaining; (b) says when it actually mattered, which requires both
R0_H > 1 and enough infectious people supplied by spillover.

Reads only results/final_figures/transmission_dynamics/
    posterior_transmission_dynamics_all_climate.csv
(written by final_transmission_dynamics.py). Nothing is refitted.
The wave rule is imported from plot_wave_phases_and_R0_criteria.py and
cross-checked against wave_analysis/wave_boundary_weeks.csv.

Usage
-----
    python scripts/plot_spillover_amplification_handover.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_wave_phases_and_R0_criteria import (  # noqa: E402
    INPUT_CSV,
    OUTPUT_DIR,
    DEATHS_COL,
    COLOR_ASCENDING,
    COLOR_DECLINING,
    COLOR_PEAK,
    COLOR_TROUGH,
    COLOR_R0,
    detect_waves,
    cross_check_boundaries,
    shade_phases,
    format_time_axis,
)

COLOR_SPILLOVER = "#1f77b4"      # blue, as in the paper's decomposition figures
COLOR_AMPLIFICATION = "#E08214"  # orange, as in the paper
COLOR_BARS = "#8C8C8C"


def panel_label(ax, text: str) -> None:
    ax.text(0.005, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV}\nRun scripts/final_transmission_dynamics.py first."
        )

    df = pd.read_csv(INPUT_CSV, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    peaks, bounds = detect_waves(df)
    cross_check_boundaries(df, bounds)
    t = df["week_ending"]

    # Forces of infection are ~1e-4 per week: show them per 10,000 per week
    # so the axis reads in plain numbers.
    scale = 1e4

    fig, (ax_d, ax_f, ax_r) = plt.subplots(
        3, 1, figsize=(12.5, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.2, 1.0]},
    )

    # (a) deaths
    shade_phases(ax_d, df, peaks, bounds)
    ax_d.bar(t, df[DEATHS_COL], width=6, color=COLOR_BARS, zorder=2)
    ax_d.set_ylabel("Weekly deaths")
    panel_label(ax_d, "a  Observed mortality")
    format_time_axis(ax_d)

    # (b) the two forces of infection, same axis
    shade_phases(ax_f, df, peaks, bounds)
    ax_f.fill_between(t, scale * df["spillover_q025"], scale * df["spillover_q975"],
                      color=COLOR_SPILLOVER, alpha=0.20, lw=0, zorder=2)
    ax_f.plot(t, scale * df["spillover_median"], color=COLOR_SPILLOVER, lw=1.6, zorder=3)
    ax_f.fill_between(t, scale * df["human_q025"], scale * df["human_q975"],
                      color=COLOR_AMPLIFICATION, alpha=0.25, lw=0, zorder=2)
    ax_f.plot(t, scale * df["human_median"], color=COLOR_AMPLIFICATION, lw=1.6, zorder=3)
    ax_f.set_ylabel("Force of infection\n(per 10,000 per week)")
    ax_f.set_ylim(bottom=0)
    panel_label(ax_f, "b  Spillover vs realised amplification")
    format_time_axis(ax_f)

    # (c) R0_H potential
    shade_phases(ax_r, df, peaks, bounds)
    ax_r.fill_between(t, df["R0_H_q025"], df["R0_H_q975"],
                      color=COLOR_AMPLIFICATION, alpha=0.25, lw=0, zorder=2)
    ax_r.plot(t, df["R0_H_median"], color=COLOR_AMPLIFICATION, lw=1.6, zorder=3)
    ax_r.axhline(1.0, color="black", lw=1.0, zorder=3)
    ax_r.set_ylabel(r"$R_{0,H}(t)$")
    panel_label(ax_r, r"c  Amplification potential $R_{0,H}=\beta_H/\gamma$")
    format_time_axis(ax_r)

    handles = [
        Patch(facecolor=COLOR_BARS, label="Observed deaths"),
        Line2D([0], [0], color=COLOR_SPILLOVER, lw=1.6,
               label=r"Spillover $\beta_R(t)$ (median, 95% band)"),
        Line2D([0], [0], color=COLOR_AMPLIFICATION, lw=1.6,
               label=r"Amplification $\beta_H I/N$; $R_{0,H}$ (median, 95% band)"),
        Line2D([0], [0], color="black", lw=1.0, label=r"$R_{0,H}=1$"),
        Patch(facecolor=COLOR_ASCENDING, alpha=0.5, label="Ascending phase"),
        Patch(facecolor=COLOR_DECLINING, alpha=0.5, label="Declining phase"),
        Line2D([0], [0], color=COLOR_PEAK, ls="--", lw=1.0, label="Mortality peak"),
        Line2D([0], [0], color=COLOR_TROUGH, ls=":", lw=1.0, label="Wave boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"fig_spillover_amplification_handover.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"[OK] Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()