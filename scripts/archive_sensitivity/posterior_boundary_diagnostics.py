# =============================================================================
# posterior_boundary_diagnostics.py
# =============================================================================
"""
Check whether posterior distributions are being truncated by prior bounds.

For each beta_H formulation, this script loads the saved dynesty nested-sampling
result and examines the posterior distribution of every fitted parameter.

For each parameter it reports:

    - prior lower bound
    - prior upper bound
    - posterior median
    - posterior 2.5th percentile
    - posterior 97.5th percentile
    - fraction of posterior samples within 5% of the lower prior bound
    - fraction of posterior samples within 5% of the upper prior bound

A parameter is flagged when a substantial fraction of posterior mass lies
close to a prior boundary.

The script also saves posterior histograms with the prior bounds shown as
vertical dashed lines.

Usage:
    python scripts/posterior_boundary_diagnostics.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynesty.utils import resample_equal


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

BETA_H_MODES = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]

MODEL_LABELS = {
    "constant": r"Constant $\beta_H$",
    "sigmoid": r"Sigmoid $\beta_H$",
    "humidity_only": r"Humidity-dependent $\beta_H$",
    "all_climate": r"Climate-driven $\beta_H$",
}


# Fraction of the total prior width considered "near" a boundary.
#
# Example:
# prior = [-1, 1]
# width = 2
# 5% of width = 0.1
#
# therefore samples in [-1, -0.9] count as near the lower boundary.
BOUNDARY_FRACTION = 0.05


# Posterior fraction above which we flag a possible boundary issue.
#
# This is a diagnostic threshold, not a formal statistical test.
FLAG_THRESHOLD = 0.10


# =============================================================================
# Figure styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


# =============================================================================
# Output directories
# =============================================================================

paths.ensure_output_dirs()

BOUNDARY_FIG_DIR = (
    paths.FIGURES_DIR
    / "posterior_boundary_diagnostics"
)

BOUNDARY_FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Helpers
# =============================================================================

def load_saved_result(betaH_mode: str) -> dict:
    """
    Load one saved dynesty result.
    """

    stem = paths.run_stem(
        betaH_mode,
        FIT_TARGET,
    )

    pkl_path = (
        paths.NESTED_SAMPLING_DIR
        / f"{stem}.pkl"
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find nested-sampling result:\n{pkl_path}"
        )

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    return saved


def get_equal_weight_posterior(saved: dict) -> np.ndarray:
    """
    Convert weighted dynesty samples to approximately equal-weight
    posterior samples.
    """

    results = saved["results"]

    weights = np.exp(
        results.logwt
        - results.logz[-1]
    )

    posterior_samples = resample_equal(
        results.samples,
        weights,
    )

    return posterior_samples


def boundary_diagnostics_for_parameter(
    samples: np.ndarray,
    lower: float,
    upper: float,
) -> dict:
    """
    Calculate posterior summary and boundary-mass diagnostics for one parameter.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    width = upper - lower

    if width <= 0:
        raise ValueError(
            f"Invalid prior bounds: lower={lower}, upper={upper}"
        )

    boundary_width = (
        BOUNDARY_FRACTION
        * width
    )

    lower_cutoff = (
        lower
        + boundary_width
    )

    upper_cutoff = (
        upper
        - boundary_width
    )

    fraction_near_lower = float(
        np.mean(
            samples <= lower_cutoff
        )
    )

    fraction_near_upper = float(
        np.mean(
            samples >= upper_cutoff
        )
    )

    q025, median, q975 = np.quantile(
        samples,
        [0.025, 0.5, 0.975],
    )

    flag_lower = (
        fraction_near_lower
        >= FLAG_THRESHOLD
    )

    flag_upper = (
        fraction_near_upper
        >= FLAG_THRESHOLD
    )

    return {
        "prior_lower": lower,
        "prior_upper": upper,

        "posterior_q025": float(q025),
        "posterior_median": float(median),
        "posterior_q975": float(q975),

        "fraction_near_lower":
            fraction_near_lower,

        "fraction_near_upper":
            fraction_near_upper,

        "flag_lower":
            flag_lower,

        "flag_upper":
            flag_upper,

        "boundary_flag":
            flag_lower or flag_upper,
    }


def plot_parameter_posterior(
    samples: np.ndarray,
    parameter_name: str,
    betaH_mode: str,
    lower: float,
    upper: float,
    summary: dict,
) -> None:
    """
    Plot one posterior histogram with prior bounds and posterior median.
    """

    fig, ax = plt.subplots(
        figsize=(5.4, 3.8)
    )

    ax.hist(
        samples,
        bins=40,
        density=True,
        alpha=0.75,
        edgecolor="none",
    )

    # Prior lower bound
    ax.axvline(
        lower,
        linestyle="--",
        linewidth=1.2,
        label="Prior bounds",
    )

    # Prior upper bound
    ax.axvline(
        upper,
        linestyle="--",
        linewidth=1.2,
    )

    # Posterior median
    ax.axvline(
        summary["posterior_median"],
        linestyle="-",
        linewidth=1.5,
        label="Posterior median",
    )

    # 95% credible interval
    ax.axvline(
        summary["posterior_q025"],
        linestyle=":",
        linewidth=1.2,
        label="95% CrI",
    )

    ax.axvline(
        summary["posterior_q975"],
        linestyle=":",
        linewidth=1.2,
    )

    ax.set_xlabel(
        parameter_name
    )

    ax.set_ylabel(
        "Posterior density"
    )

    ax.set_title(
        f"{MODEL_LABELS[betaH_mode]} | {parameter_name}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    fig_path = (
        BOUNDARY_FIG_DIR
        / f"{betaH_mode}__{parameter_name}.png"
    )

    fig.savefig(
        fig_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

summary_rows = []


for betaH_mode in BETA_H_MODES:

    print("\n" + "=" * 80)
    print(f"POSTERIOR BOUNDARY CHECK: {betaH_mode}")
    print("=" * 80)

    saved = load_saved_result(
        betaH_mode
    )

    param_names = saved["param_names"]

    bounds = saved["bounds"]

    posterior = get_equal_weight_posterior(
        saved
    )

    if posterior.shape[1] != len(param_names):
        raise ValueError(
            f"Posterior dimension mismatch for {betaH_mode}"
        )

    if len(bounds) != len(param_names):
        raise ValueError(
            f"Bounds dimension mismatch for {betaH_mode}"
        )

    for j, parameter_name in enumerate(
        param_names
    ):

        lower, upper = bounds[j]

        samples_j = posterior[:, j]

        diag = boundary_diagnostics_for_parameter(
            samples=samples_j,
            lower=lower,
            upper=upper,
        )

        summary_rows.append({
            "betaH_mode":
                betaH_mode,

            "parameter":
                parameter_name,

            **diag,
        })

        print(
            f"{parameter_name:12s} "
            f"median={diag['posterior_median']:.5g} "
            f"95%CrI=({diag['posterior_q025']:.5g}, "
            f"{diag['posterior_q975']:.5g}) "
            f"near lower={100*diag['fraction_near_lower']:.1f}% "
            f"near upper={100*diag['fraction_near_upper']:.1f}% "
            f"{'*** FLAG ***' if diag['boundary_flag'] else ''}"
        )

        plot_parameter_posterior(
            samples=samples_j,
            parameter_name=parameter_name,
            betaH_mode=betaH_mode,
            lower=lower,
            upper=upper,
            summary=diag,
        )


# =============================================================================
# Combine results
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


# =============================================================================
# Save complete table
# =============================================================================

summary_path = (
    paths.POSTERIOR_SUMMARY_DIR
    / f"posterior_boundary_diagnostics__{FIT_TARGET}.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 100)
print("BOUNDARY DIAGNOSTIC SUMMARY")
print("=" * 100)

print(
    summary_df[
        [
            "betaH_mode",
            "parameter",
            "prior_lower",
            "prior_upper",
            "posterior_median",
            "posterior_q025",
            "posterior_q975",
            "fraction_near_lower",
            "fraction_near_upper",
            "boundary_flag",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# Print only flagged parameters
# =============================================================================

flagged = (
    summary_df[
        summary_df["boundary_flag"]
    ]
    .copy()
)


print("\n" + "=" * 100)
print("FLAGGED PARAMETERS")
print("=" * 100)


if flagged.empty:

    print(
        "No parameters exceeded the boundary-mass diagnostic threshold."
    )

else:

    print(
        flagged[
            [
                "betaH_mode",
                "parameter",
                "prior_lower",
                "prior_upper",
                "posterior_median",
                "posterior_q025",
                "posterior_q975",
                "fraction_near_lower",
                "fraction_near_upper",
            ]
        ].to_string(
            index=False
        )
    )


print(
    f"\n[OK] Saved boundary diagnostic table:\n"
    f"{summary_path}"
)

print(
    f"\n[OK] Saved posterior diagnostic figures to:\n"
    f"{BOUNDARY_FIG_DIR}"
)