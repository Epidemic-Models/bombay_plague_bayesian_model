# =============================================================================
# posterior_boundary_diagnostics_round2.py
# =============================================================================
"""
Posterior-boundary diagnostics for the SECOND all-climate prior-sensitivity run.

This script examines the posterior from:

    results/prior_sensitivity_round2/nested_sampling/
        all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl

The purpose is to determine whether the previously problematic parameters,
especially aR and a0, now lie comfortably inside their widened prior ranges.

For each parameter the script reports:

    - prior lower and upper bounds
    - posterior median
    - posterior 2.5% and 97.5% quantiles
    - posterior distance from each bound
    - fraction of posterior mass near each bound
    - a simple boundary flag

It also produces one posterior histogram per parameter.

Usage:
    python scripts/posterior_boundary_diagnostics_round2.py
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

BETA_H_MODE = "all_climate"
FIT_TARGET = "plague_deaths_reported_fit"

BOUNDARY_FRACTION = 0.05
FLAG_THRESHOLD = 0.10


# =============================================================================
# Paths
# =============================================================================

ROUND2_ROOT = (
    paths.RESULTS_DIR
    / "prior_sensitivity_round2"
)

ROUND2_NESTED_DIR = (
    ROUND2_ROOT
    / "nested_sampling"
)

ROUND2_FIG_DIR = (
    ROUND2_ROOT
    / "figures"
    / "posterior_boundary_diagnostics"
)

ROUND2_FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PKL = (
    ROUND2_NESTED_DIR
    / "all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl"
)

SUMMARY_CSV = (
    ROUND2_ROOT
    / "posterior_boundary_diagnostics_round2.csv"
)


# =============================================================================
# Plot styling
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
# Helpers
# =============================================================================

def load_round2_result() -> dict:
    """
    Load the round-2 all-climate nested-sampling result.
    """

    if not RESULT_PKL.exists():
        raise FileNotFoundError(
            f"Could not find round-2 result:\n{RESULT_PKL}"
        )

    with open(RESULT_PKL, "rb") as f:
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

    posterior = resample_equal(
        results.samples,
        weights,
    )

    return posterior


def diagnose_parameter(
    samples: np.ndarray,
    lower: float,
    upper: float,
) -> dict:
    """
    Calculate posterior summary and boundary diagnostics.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    width = (
        upper
        - lower
    )

    if width <= 0:
        raise ValueError(
            f"Invalid prior bounds: ({lower}, {upper})"
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

    q025, median, q975 = np.quantile(
        samples,
        [0.025, 0.5, 0.975],
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

    flag_lower = (
        fraction_near_lower
        >= FLAG_THRESHOLD
    )

    flag_upper = (
        fraction_near_upper
        >= FLAG_THRESHOLD
    )

    distance_median_from_lower = (
        median - lower
    )

    distance_median_from_upper = (
        upper - median
    )

    normalized_distance_lower = (
        distance_median_from_lower / width
    )

    normalized_distance_upper = (
        distance_median_from_upper / width
    )

    return {
        "prior_lower":
            float(lower),

        "prior_upper":
            float(upper),

        "posterior_q025":
            float(q025),

        "posterior_median":
            float(median),

        "posterior_q975":
            float(q975),

        "fraction_near_lower":
            fraction_near_lower,

        "fraction_near_upper":
            fraction_near_upper,

        "normalized_median_distance_lower":
            float(normalized_distance_lower),

        "normalized_median_distance_upper":
            float(normalized_distance_upper),

        "flag_lower":
            flag_lower,

        "flag_upper":
            flag_upper,

        "boundary_flag":
            flag_lower or flag_upper,
    }


def plot_parameter(
    samples: np.ndarray,
    parameter_name: str,
    lower: float,
    upper: float,
    diag: dict,
) -> None:
    """
    Plot posterior histogram with prior bounds and 95% credible interval.
    """

    fig, ax = plt.subplots(
        figsize=(5.4, 3.8)
    )

    ax.hist(
        samples,
        bins=45,
        density=True,
        alpha=0.75,
        edgecolor="none",
    )

    ax.axvline(
        lower,
        linestyle="--",
        linewidth=1.2,
        label="Prior bounds",
    )

    ax.axvline(
        upper,
        linestyle="--",
        linewidth=1.2,
    )

    ax.axvline(
        diag["posterior_median"],
        linestyle="-",
        linewidth=1.5,
        label="Posterior median",
    )

    ax.axvline(
        diag["posterior_q025"],
        linestyle=":",
        linewidth=1.2,
        label="95% CrI",
    )

    ax.axvline(
        diag["posterior_q975"],
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
        f"All-climate $\\beta_H$ | {parameter_name}"
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
        ROUND2_FIG_DIR
        / f"all_climate__{parameter_name}__round2.png"
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

saved = load_round2_result()

param_names = saved["param_names"]
bounds = saved["bounds"]

posterior = get_equal_weight_posterior(
    saved
)

if posterior.shape[1] != len(param_names):
    raise ValueError(
        "Posterior dimension does not match parameter-name list."
    )

if len(bounds) != len(param_names):
    raise ValueError(
        "Bounds list does not match parameter-name list."
    )


summary_rows = []

print("\n" + "=" * 88)
print("ROUND-2 POSTERIOR BOUNDARY CHECK: all_climate")
print("=" * 88)


for j, parameter_name in enumerate(
    param_names
):

    lower, upper = bounds[j]

    samples_j = posterior[:, j]

    diag = diagnose_parameter(
        samples=samples_j,
        lower=lower,
        upper=upper,
    )

    summary_rows.append({
        "betaH_mode":
            BETA_H_MODE,

        "parameter":
            parameter_name,

        **diag,
    })

    flag_text = (
        "*** FLAG ***"
        if diag["boundary_flag"]
        else ""
    )

    print(
        f"{parameter_name:12s} "
        f"median={diag['posterior_median']:.6g} "
        f"95%CrI=("
        f"{diag['posterior_q025']:.6g}, "
        f"{diag['posterior_q975']:.6g}) "
        f"near lower="
        f"{100 * diag['fraction_near_lower']:.1f}% "
        f"near upper="
        f"{100 * diag['fraction_near_upper']:.1f}% "
        f"{flag_text}"
    )

    plot_parameter(
        samples=samples_j,
        parameter_name=parameter_name,
        lower=lower,
        upper=upper,
        diag=diag,
    )


# =============================================================================
# Save table
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# =============================================================================
# Print complete table
# =============================================================================

print("\n" + "=" * 110)
print("ROUND-2 BOUNDARY DIAGNOSTIC SUMMARY")
print("=" * 110)

print(
    summary_df[
        [
            "parameter",
            "prior_lower",
            "prior_upper",
            "posterior_median",
            "posterior_q025",
            "posterior_q975",
            "fraction_near_lower",
            "fraction_near_upper",
            "normalized_median_distance_lower",
            "normalized_median_distance_upper",
            "boundary_flag",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# Print flagged parameters
# =============================================================================

flagged = (
    summary_df[
        summary_df["boundary_flag"]
    ]
    .copy()
)

print("\n" + "=" * 110)
print("FLAGGED PARAMETERS")
print("=" * 110)

if flagged.empty:

    print(
        "No parameters exceeded the boundary diagnostic threshold."
    )

else:

    print(
        flagged[
            [
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


# =============================================================================
# Focus on previously problematic parameters
# =============================================================================

focus_params = [
    "a0",
    "aR",
    "aTT",
    "aT",
    "psi",
]

focus = summary_df[
    summary_df["parameter"].isin(
        focus_params
    )
].copy()


print("\n" + "=" * 110)
print("FOCUS: PREVIOUSLY IMPORTANT PARAMETERS")
print("=" * 110)

print(
    focus[
        [
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


print(
    f"\n[OK] Saved round-2 boundary table:\n"
    f"{SUMMARY_CSV}"
)

print(
    f"\n[OK] Saved round-2 posterior figures to:\n"
    f"{ROUND2_FIG_DIR}"
)