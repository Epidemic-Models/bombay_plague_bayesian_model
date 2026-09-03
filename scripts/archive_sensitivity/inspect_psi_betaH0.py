# =============================================================================
# inspect_psi_betaH0.py
# =============================================================================
"""
Posterior diagnostic for psi and betaH0 in the final round-2 all-climate model.

Purpose
-------
The round-2 posterior-boundary analysis showed some posterior mass close to:

    psi upper bound     = 1.0
    betaH0 upper bound  = 1.5

This script investigates whether those patterns reflect:

    1. genuine posterior concentration near the prior limits;
    2. a trade-off / correlation between psi and betaH0;
    3. correlation between psi and other mechanistic parameters;
    4. a potentially restrictive prior on psi.

The script does NOT modify priors or rerun nested sampling.

Input
-----
results/prior_sensitivity_round2/nested_sampling/
    all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl

Outputs
-------
results/prior_sensitivity_round2/psi_diagnostics/

    psi_betaH0_posterior_summary.csv
    psi_parameter_correlations.csv
    psi_posterior.png
    betaH0_posterior.png
    psi_vs_betaH0.png

Usage
-----
python scripts/inspect_psi_betaH0.py
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
BETA_H_MODE = "all_climate"


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

RESULT_PKL = (
    ROUND2_NESTED_DIR
    / "all_climate__plague_deaths_reported_fit__prior_sensitivity_round2.pkl"
)

OUTPUT_DIR = (
    ROUND2_ROOT
    / "psi_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "psi_betaH0_posterior_summary.csv"
)

CORRELATION_CSV = (
    OUTPUT_DIR
    / "psi_parameter_correlations.csv"
)

PSI_FIG = (
    OUTPUT_DIR
    / "psi_posterior.png"
)

BETAH0_FIG = (
    OUTPUT_DIR
    / "betaH0_posterior.png"
)

JOINT_FIG = (
    OUTPUT_DIR
    / "psi_vs_betaH0.png"
)


# =============================================================================
# Styling
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
# Load result
# =============================================================================

if not RESULT_PKL.exists():
    raise FileNotFoundError(
        f"Could not find round-2 result:\n{RESULT_PKL}"
    )


with open(RESULT_PKL, "rb") as f:
    saved = pickle.load(f)


results = saved["results"]
param_names = list(saved["param_names"])
bounds = list(saved["bounds"])


# =============================================================================
# Equal-weight posterior
# =============================================================================

weights = np.exp(
    results.logwt
    - results.logz[-1]
)

posterior = resample_equal(
    results.samples,
    weights,
)


posterior_df = pd.DataFrame(
    posterior,
    columns=param_names,
)


print("\n" + "=" * 90)
print("PSI / betaH0 POSTERIOR DIAGNOSTIC")
print("=" * 90)

print(
    f"\nPosterior samples: {len(posterior_df)}"
)

print(
    f"log(Z) = "
    f"{results.logz[-1]:.3f} "
    f"+/- {results.logzerr[-1]:.3f}"
)


# =============================================================================
# Helper
# =============================================================================

def get_bound(parameter: str):
    """
    Return prior lower and upper bounds for one parameter.
    """

    idx = param_names.index(parameter)

    return (
        float(bounds[idx][0]),
        float(bounds[idx][1]),
    )


def summarize_parameter(parameter: str) -> dict:
    """
    Posterior summary for one parameter.
    """

    values = posterior_df[
        parameter
    ].to_numpy(dtype=float)

    lower, upper = get_bound(
        parameter
    )

    q025, q25, median, q75, q975 = np.quantile(
        values,
        [
            0.025,
            0.25,
            0.50,
            0.75,
            0.975,
        ],
    )

    width = (
        upper
        - lower
    )

    # Fractions very close to the upper boundary.
    #
    # Use several thresholds rather than one arbitrary 5% rule.
    upper_10pct = (
        upper
        - 0.10 * width
    )

    upper_5pct = (
        upper
        - 0.05 * width
    )

    upper_1pct = (
        upper
        - 0.01 * width
    )

    return {
        "parameter":
            parameter,

        "prior_lower":
            lower,

        "prior_upper":
            upper,

        "posterior_mean":
            float(np.mean(values)),

        "posterior_sd":
            float(np.std(values)),

        "posterior_q025":
            float(q025),

        "posterior_q25":
            float(q25),

        "posterior_median":
            float(median),

        "posterior_q75":
            float(q75),

        "posterior_q975":
            float(q975),

        "fraction_upper_10pct":
            float(
                np.mean(
                    values >= upper_10pct
                )
            ),

        "fraction_upper_5pct":
            float(
                np.mean(
                    values >= upper_5pct
                )
            ),

        "fraction_upper_1pct":
            float(
                np.mean(
                    values >= upper_1pct
                )
            ),

        "distance_median_from_upper":
            float(
                upper
                - median
            ),

        "normalized_distance_median_from_upper":
            float(
                (upper - median)
                / width
            ),
    }


# =============================================================================
# Summaries
# =============================================================================

parameters_of_interest = [
    "psi",
    "betaH0",
    "nb_kappa",
    "a0",
    "aT",
    "aTT",
    "ae",
    "aR",
    "bT_H",
    "be_H",
    "bR_H",
]


summary_rows = []


for parameter in parameters_of_interest:

    if parameter not in posterior_df.columns:
        continue

    summary_rows.append(
        summarize_parameter(
            parameter
        )
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


print("\nPosterior summaries")
print("-" * 90)

print(
    summary_df[
        [
            "parameter",
            "prior_lower",
            "prior_upper",
            "posterior_median",
            "posterior_q025",
            "posterior_q975",
            "fraction_upper_5pct",
            "fraction_upper_1pct",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# Correlation of psi with every other parameter
# =============================================================================

psi_values = posterior_df[
    "psi"
].to_numpy(dtype=float)


correlation_rows = []


for parameter in param_names:

    if parameter == "psi":
        continue

    x = posterior_df[
        parameter
    ].to_numpy(dtype=float)

    pearson = np.corrcoef(
        psi_values,
        x,
    )[0, 1]

    # Spearman correlation using ranks without requiring scipy
    psi_rank = pd.Series(
        psi_values
    ).rank().to_numpy()

    x_rank = pd.Series(
        x
    ).rank().to_numpy()

    spearman = np.corrcoef(
        psi_rank,
        x_rank,
    )[0, 1]

    correlation_rows.append({
        "parameter":
            parameter,

        "pearson_correlation_with_psi":
            float(pearson),

        "spearman_correlation_with_psi":
            float(spearman),

        "absolute_pearson":
            float(abs(pearson)),
    })


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df = (
    correlation_df
    .sort_values(
        "absolute_pearson",
        ascending=False,
    )
    .reset_index(drop=True)
)


correlation_df.to_csv(
    CORRELATION_CSV,
    index=False,
)


print("\n" + "=" * 90)
print("PARAMETER CORRELATIONS WITH psi")
print("=" * 90)

print(
    correlation_df[
        [
            "parameter",
            "pearson_correlation_with_psi",
            "spearman_correlation_with_psi",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# Specific psi / betaH0 relationship
# =============================================================================

psi = posterior_df[
    "psi"
].to_numpy(dtype=float)

betaH0 = posterior_df[
    "betaH0"
].to_numpy(dtype=float)


pearson_psi_beta = np.corrcoef(
    psi,
    betaH0,
)[0, 1]


print("\n" + "=" * 90)
print("psi vs betaH0")
print("=" * 90)

print(
    f"Pearson correlation = "
    f"{pearson_psi_beta:.4f}"
)


# =============================================================================
# Additional psi boundary probabilities
# =============================================================================

print("\n" + "=" * 90)
print("PSI UPPER-BOUND CHECK")
print("=" * 90)

print(
    f"P(psi > 0.90 | data) = "
    f"{np.mean(psi > 0.90):.3f}"
)

print(
    f"P(psi > 0.95 | data) = "
    f"{np.mean(psi > 0.95):.3f}"
)

print(
    f"P(psi > 0.98 | data) = "
    f"{np.mean(psi > 0.98):.3f}"
)

print(
    f"P(psi > 0.99 | data) = "
    f"{np.mean(psi > 0.99):.3f}"
)


# =============================================================================
# betaH0 upper-bound probabilities
# =============================================================================

beta_lower, beta_upper = get_bound(
    "betaH0"
)


print("\n" + "=" * 90)
print("betaH0 UPPER-BOUND CHECK")
print("=" * 90)

print(
    f"P(betaH0 > 1.40 | data) = "
    f"{np.mean(betaH0 > 1.40):.3f}"
)

print(
    f"P(betaH0 > 1.45 | data) = "
    f"{np.mean(betaH0 > 1.45):.3f}"
)

print(
    f"P(betaH0 > 1.49 | data) = "
    f"{np.mean(betaH0 > 1.49):.3f}"
)


# =============================================================================
# Plot psi posterior
# =============================================================================

psi_lower, psi_upper = get_bound(
    "psi"
)

psi_median = np.median(
    psi
)

psi_q025, psi_q975 = np.quantile(
    psi,
    [0.025, 0.975],
)


fig, ax = plt.subplots(
    figsize=(5.4, 3.8)
)


ax.hist(
    psi,
    bins=50,
    density=True,
    alpha=0.75,
    edgecolor="none",
)


ax.axvline(
    psi_lower,
    linestyle="--",
    linewidth=1.2,
    label="Prior bounds",
)

ax.axvline(
    psi_upper,
    linestyle="--",
    linewidth=1.2,
)


ax.axvline(
    psi_median,
    linestyle="-",
    linewidth=1.6,
    label="Posterior median",
)


ax.axvline(
    psi_q025,
    linestyle=":",
    linewidth=1.2,
    label="95% CrI",
)

ax.axvline(
    psi_q975,
    linestyle=":",
    linewidth=1.2,
)


ax.set_xlabel(
    r"$\psi$"
)

ax.set_ylabel(
    "Posterior density"
)

ax.set_title(
    r"Posterior distribution of $\psi$"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    PSI_FIG,
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Plot betaH0 posterior
# =============================================================================

beta_median = np.median(
    betaH0
)

beta_q025, beta_q975 = np.quantile(
    betaH0,
    [0.025, 0.975],
)


fig, ax = plt.subplots(
    figsize=(5.4, 3.8)
)


ax.hist(
    betaH0,
    bins=50,
    density=True,
    alpha=0.75,
    edgecolor="none",
)


ax.axvline(
    beta_lower,
    linestyle="--",
    linewidth=1.2,
    label="Prior bounds",
)

ax.axvline(
    beta_upper,
    linestyle="--",
    linewidth=1.2,
)


ax.axvline(
    beta_median,
    linestyle="-",
    linewidth=1.6,
    label="Posterior median",
)


ax.axvline(
    beta_q025,
    linestyle=":",
    linewidth=1.2,
    label="95% CrI",
)

ax.axvline(
    beta_q975,
    linestyle=":",
    linewidth=1.2,
)


ax.set_xlabel(
    r"$\beta_{H0}$"
)

ax.set_ylabel(
    "Posterior density"
)

ax.set_title(
    r"Posterior distribution of $\beta_{H0}$"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    BETAH0_FIG,
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Joint psi / betaH0 posterior
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.3, 4.3)
)


hb = ax.hexbin(
    psi,
    betaH0,
    gridsize=45,
    mincnt=1,
)


ax.set_xlabel(
    r"$\psi$"
)

ax.set_ylabel(
    r"$\beta_{H0}$"
)

ax.set_title(
    r"Joint posterior: $\psi$ and $\beta_{H0}$"
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


cbar = fig.colorbar(
    hb,
    ax=ax,
)

cbar.set_label(
    "Posterior sample density"
)


fig.tight_layout()


fig.savefig(
    JOINT_FIG,
    dpi=400,
    bbox_inches="tight",
)


plt.close(fig)


# =============================================================================
# Final report
# =============================================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"\nPosterior summary:\n"
    f"{SUMMARY_CSV}"
)

print(
    f"\npsi correlations:\n"
    f"{CORRELATION_CSV}"
)

print(
    f"\npsi posterior figure:\n"
    f"{PSI_FIG}"
)

print(
    f"\nbetaH0 posterior figure:\n"
    f"{BETAH0_FIG}"
)

print(
    f"\nJoint posterior figure:\n"
    f"{JOINT_FIG}"
)

print(
    "\nDiagnostic completed successfully."
)