# =============================================================================
# explain_r0_summary.py
# =============================================================================
"""
Reads posterior_human_R0_summary.csv (produced by
final_transmission_dynamics.py) and prints a plain-language interpretation
of what the numbers mean -- ranking, which models can/cannot sustain
human-to-human transmission on their own, and a flag for any model where
the mean and median diverge sharply (a sign of a right-skewed
distribution, meaning the mean alone would overstate the "typical" value).

This does NOT recompute anything -- it only reads and interprets the CSV
that final_transmission_dynamics.py already produced, so it's fast and has
no dependency on dynesty, the mechanistic model, or any .pkl file.

Usage
-----
    python scripts/explain_r0_summary.py
    python scripts/explain_r0_summary.py --csv path/to/posterior_human_R0_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "results" / "final_figures" / "transmission_dynamics"
    / "posterior_human_R0_summary.csv"
)

MODEL_LABELS = {
    "constant": "Constant beta_H",
    "sigmoid": "Sigmoid beta_H",
    "humidity_only": "Humidity-only beta_H",
    "all_climate": "Climate-driven beta_H",
}

# Below this, "essentially never self-sustaining" is a fair description.
SELF_SUSTAIN_THRESHOLD_PCT = 1.0

# If mean and median differ by more than this fraction of the mean, flag
# the distribution as meaningfully right-skewed.
SKEW_FLAG_RATIO = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(
            f"Could not find {args.csv}\n"
            "Run scripts/final_transmission_dynamics.py first, or pass "
            "--csv pointing at your posterior_human_R0_summary.csv."
        )

    df = pd.read_csv(args.csv)
    required = {
        "betaH_mode", "R0_H_posterior_mean", "R0_H_posterior_median",
        "R0_H_min", "R0_H_max", "pct_draw_week_combinations_R0_H_above_1",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected columns: {missing}")

    df = df.sort_values("R0_H_posterior_mean", ascending=False).reset_index(drop=True)

    print("=" * 90)
    print(f"R0_H POSTERIOR SUMMARY -- interpreting {args.csv}")
    print("=" * 90)

    print(
        "\nR0_H(t) = beta_H(t) / gamma measures whether human-to-human transmission\n"
        "ALONE (with no help from reservoir spillover) could sustain epidemic growth.\n"
        "R0_H > 1 in a given week/parameter-draw means: if spillover stopped entirely\n"
        "that week, human transmission alone would still be growing the epidemic.\n"
    )

    print("-" * 90)
    print(f"{'Model':<24s}{'mean':>8s}{'median':>8s}{'min':>8s}{'max':>8s}{'% weeks >1':>14s}")
    print("-" * 90)
    for _, row in df.iterrows():
        label = MODEL_LABELS.get(row["betaH_mode"], row["betaH_mode"])
        print(
            f"{label:<24s}"
            f"{row['R0_H_posterior_mean']:>8.3f}"
            f"{row['R0_H_posterior_median']:>8.3f}"
            f"{row['R0_H_min']:>8.3f}"
            f"{row['R0_H_max']:>8.3f}"
            f"{row['pct_draw_week_combinations_R0_H_above_1']:>13.1f}%"
        )

    print("\n" + "=" * 90)
    print("RANKING (highest human-transmission reproduction number first)")
    print("=" * 90)
    for i, row in df.iterrows():
        label = MODEL_LABELS.get(row["betaH_mode"], row["betaH_mode"])
        print(f"  {i + 1}. {label}  (mean R0_H = {row['R0_H_posterior_mean']:.3f})")

    print("\n" + "=" * 90)
    print("PER-MODEL INTERPRETATION")
    print("=" * 90)

    for _, row in df.iterrows():
        label = MODEL_LABELS.get(row["betaH_mode"], row["betaH_mode"])
        mean = row["R0_H_posterior_mean"]
        median = row["R0_H_posterior_median"]
        pct_above = row["pct_draw_week_combinations_R0_H_above_1"]

        print(f"\n{label}:")

        if pct_above < SELF_SUSTAIN_THRESHOLD_PCT:
            print(
                f"  -> Essentially never self-sustaining ({pct_above:.1f}% of draw-week "
                f"combinations exceed 1). Human transmission alone cannot maintain the "
                f"epidemic under this formulation; continued spillover is required "
                f"throughout."
            )
        elif pct_above < 20.0:
            print(
                f"  -> Occasionally self-sustaining ({pct_above:.1f}% of draw-week "
                f"combinations exceed 1). Human transmission can carry the epidemic on "
                f"its own during a minority of conditions, but spillover still matters "
                f"most of the time."
            )
        else:
            print(
                f"  -> Substantially self-sustaining ({pct_above:.1f}% of draw-week "
                f"combinations exceed 1). Human transmission alone can sustain the "
                f"epidemic under a meaningful fraction of the fitted period/posterior."
            )

        if mean > 0:
            skew = abs(mean - median) / mean
            if skew > SKEW_FLAG_RATIO:
                direction = "above" if mean > median else "below"
                print(
                    f"  -> NOTE: mean ({mean:.3f}) sits notably {direction} the median "
                    f"({median:.3f}) -- the distribution is right-skewed: most "
                    f"weeks/draws sit below the mean, with a smaller number of high "
                    f"values pulling it up. Report the median as the 'typical' value, "
                    f"not the mean, if you only have room for one number."
                )

    print("\n" + "=" * 90)
    print("BOTTOM LINE")
    print("=" * 90)

    best = df.iloc[0]
    worst = df.iloc[-1]
    best_label = MODEL_LABELS.get(best["betaH_mode"], best["betaH_mode"])
    worst_label = MODEL_LABELS.get(worst["betaH_mode"], worst["betaH_mode"])

    print(
        f"\n{best_label} has the largest human-transmission reproduction number "
        f"(mean {best['R0_H_posterior_mean']:.2f}, {best['pct_draw_week_combinations_R0_H_above_1']:.1f}% "
        f"of draw-weeks self-sustaining) and {worst_label} has the smallest "
        f"(mean {worst['R0_H_posterior_mean']:.2f}, "
        f"{worst['pct_draw_week_combinations_R0_H_above_1']:.1f}%).\n"
        f"\nIf these numbers came from final_v2 (check the console output of "
        f"final_transmission_dynamics.py for the logZ sanity-check line), this "
        f"ranking is the one to report in the paper."
    )


if __name__ == "__main__":
    main()