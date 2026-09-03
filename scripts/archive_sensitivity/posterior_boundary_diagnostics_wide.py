# =============================================================================
# posterior_boundary_diagnostics_wide.py
# =============================================================================
"""
Posterior boundary diagnostics for the WIDE-PRIOR sensitivity analysis.

This script checks whether posterior distributions from the widened-prior
nested-sampling runs still pile up against their new prior boundaries.

It reads results from:

    results/prior_sensitivity/nested_sampling/

and writes:

    results/prior_sensitivity/posterior_boundary_diagnostics_wide.csv

plus one posterior histogram per parameter under:

    results/prior_sensitivity/figures/posterior_boundary_diagnostics/

For each parameter the script reports:

    - prior lower bound
    - prior upper bound
    - posterior median
    - posterior 2.5th percentile
    - posterior 97.5th percentile
    - fraction of posterior samples near the lower bound
    - fraction of posterior samples near the upper bound
    - whether the parameter is flagged as potentially boundary constrained

IMPORTANT
---------
The "near boundary" diagnostic is heuristic, not a formal statistical test.

Usage:
    python scripts/posterior_boundary_diagnostics_wide.py
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


# =============================================================================
# Boundary diagnostic settings
# =============================================================================

# Fraction of prior width considered "near" a boundary.
BOUNDARY_FRACTION = 0.05

# Fraction of posterior mass near a boundary needed to trigger a flag.
FLAG_THRESHOLD = 0.10


# =============================================================================
# Paths
# =============================================================================

SENSITIVITY_ROOT = (
    paths.RESULTS_DIR
    / "prior_sensitivity"
)

SENSITIVITY_NESTED_DIR = (
    SENSITIVITY_ROOT
    / "nested_sampling"
)

SENSITIVITY_FIG_DIR = (
    SENSITIVITY_ROOT
    / "figures"
    / "posterior_boundary_diagnostics"
)

SENSITIVITY_FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    SENSITIVITY_ROOT
    / "posterior_boundary_diagnostics_wide.csv"
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

def load_saved_result(betaH_mode: str) -> dict:
    """
    Load one wide-prior nested-sampling result.
    """

    stem = (
        f"{paths.run_stem(betaH_mode, FIT_TARGET)}"
        f"__wide_priors"
    )

    pkl_path = (
        SENSITIVITY_NESTED_DIR
        / f"{stem}.pkl"
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find wide-prior result:\n{pkl_path}"
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

    posterior = resample_equal(
        results.samples,
        weights,
    )

    return posterior


def boundary_diagnostics_for_parameter(
    samples: np.ndarray,
    lower: float,
    upper: float,
) -> dict:
    """
    Calculate posterior summaries and simple boundary diagnostics.
    """

    samples = np.asarray(samples, dtype=float)

    width = upper - lower

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
        "prior_lower": float(lower),
        "prior_upper": float(upper),

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
    diag: dict,
) -> None:
    """
    Plot one posterior distribution with prior bounds and 95% CrI.
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

    # Prior bounds
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

    # Posterior median
    ax.axvline(
        diag["posterior_median"],
        linestyle="-",
        linewidth=1.5,
        label="Posterior median",
    )

    # 95% credible interval
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
        SENSITIVITY_FIG_DIR
        / f"{betaH_mode}__{parameter_name}__wide.png"
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
    print(
        f"WIDE-PRIOR POSTERIOR BOUNDARY CHECK: "
        f"{betaH_mode}"
    )
    print("=" * 80)

    saved = load_saved_result(
        betaH_mode
    )

    param_names = saved["param_names"]

    # For the sensitivity runs, `bounds` contains the WIDE bounds.
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

        flag_text = (
            "*** FLAG ***"
            if diag["boundary_flag"]
            else ""
        )

        print(
            f"{parameter_name:12s} "
            f"median={diag['posterior_median']:.5g} "
            f"95%CrI=("
            f"{diag['posterior_q025']:.5g}, "
            f"{diag['posterior_q975']:.5g}) "
            f"near lower="
            f"{100 * diag['fraction_near_lower']:.1f}% "
            f"near upper="
            f"{100 * diag['fraction_near_upper']:.1f}% "
            f"{flag_text}"
        )

        plot_parameter_posterior(
            samples=samples_j,
            parameter_name=parameter_name,
            betaH_mode=betaH_mode,
            lower=lower,
            upper=upper,
            diag=diag,
        )


# =============================================================================
# Save summary
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


print("\n" + "=" * 100)
print(
    "WIDE-PRIOR BOUNDARY DIAGNOSTIC SUMMARY"
)
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
# Flagged parameters only
# =============================================================================

flagged = (
    summary_df[
        summary_df["boundary_flag"]
    ]
    .copy()
)


print("\n" + "=" * 100)
print(
    "FLAGGED PARAMETERS UNDER WIDE PRIORS"
)
print("=" * 100)


if flagged.empty:

    print(
        "No parameters exceeded the boundary diagnostic threshold."
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
    f"\n[OK] Saved wide-prior boundary table:\n"
    f"{SUMMARY_CSV}"
)

print(
    f"\n[OK] Saved wide-prior boundary figures to:\n"
    f"{SENSITIVITY_FIG_DIR}"
)