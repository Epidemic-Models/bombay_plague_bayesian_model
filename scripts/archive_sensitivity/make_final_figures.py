# =============================================================================
# make_final_figures.py
# =============================================================================
"""
Generate final paper figures from the FINAL Bayesian model-comparison runs.

Uses only:
    results/final_model_comparison/nested_sampling/

Outputs:
    results/final_figures/

Main outputs
------------
1. posterior_predictive_all_climate_final.png
2. posterior_parameter_summary_all_climate.csv
3. posterior_parameter_forest_all_climate.png
4. model_comparison_logZ.png
5. final_model_comparison_for_paper.csv

The all-climate model is used for posterior parameter inference and
posterior predictive analysis.

All four models are used for the model-comparison figure.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dynesty.utils import resample_equal


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

import paths  # noqa: E402
from likelihood import PlagueLikelihood  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison"
)

FINAL_NESTED_DIR = (
    FINAL_ROOT
    / "nested_sampling"
)

FINAL_COMPARISON_CSV = (
    FINAL_ROOT
    / "final_model_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALL_CLIMATE_PKL = (
    FINAL_NESTED_DIR
    / "all_climate__plague_deaths_reported_fit__final.pkl"
)


# Number of posterior draws for posterior predictive calculations
N_POSTERIOR_DRAWS = 1000

RANDOM_SEED = 12345


# =============================================================================
# Figure styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


COLOR_MODEL = "#B3222B"
COLOR_OBS = "black"


# =============================================================================
# Helper functions
# =============================================================================

def load_final_result(pkl_path: Path):
    """
    Load one saved final nested-sampling result.
    """

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{pkl_path}"
        )

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    return saved


def get_equal_weight_posterior(saved):
    """
    Convert dynesty weighted samples into approximately equal-weight
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


def simulate_negative_binomial(
    mu,
    kappa,
    rng,
):
    """
    Negative-binomial parameterization:

        E[Y] = mu
        Var[Y] = mu + mu^2 / kappa

    numpy parameterization:

        n = kappa
        p = kappa / (kappa + mu)
    """

    mu = np.asarray(
        mu,
        dtype=float,
    )

    mu = np.maximum(
        mu,
        1e-12,
    )

    p = (
        kappa
        / (kappa + mu)
    )

    return rng.negative_binomial(
        n=kappa,
        p=p,
    )


# =============================================================================
# Load prepared data
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(
        f"Prepared data not found:\n"
        f"{paths.MERGED_CSV}"
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

weeks = merged_df[
    paths.DATE_COL
]

observed = merged_df[
    FIT_TARGET
].to_numpy(dtype=float)

n_weeks = len(
    merged_df
)


# =============================================================================
# Load final all-climate posterior
# =============================================================================

saved = load_final_result(
    ALL_CLIMATE_PKL
)

results = saved["results"]
param_names = list(
    saved["param_names"]
)

posterior_samples = (
    get_equal_weight_posterior(
        saved
    )
)


print("\n" + "=" * 100)
print("FINAL ALL-CLIMATE POSTERIOR")
print("=" * 100)

print(
    f"Posterior samples available: "
    f"{len(posterior_samples)}"
)

print(
    f"log(Z) = "
    f"{results.logz[-1]:.3f} "
    f"+/- {results.logzerr[-1]:.3f}"
)


# =============================================================================
# Posterior parameter summary
# =============================================================================

summary_rows = []

for j, name in enumerate(
    param_names
):

    values = posterior_samples[
        :,
        j
    ]

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

    summary_rows.append({
        "parameter":
            name,

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
    })


posterior_summary_df = pd.DataFrame(
    summary_rows
)

POSTERIOR_SUMMARY_CSV = (
    OUTPUT_DIR
    / "posterior_parameter_summary_all_climate.csv"
)

posterior_summary_df.to_csv(
    POSTERIOR_SUMMARY_CSV,
    index=False,
)


print(
    f"\n[OK] Saved posterior summary:\n"
    f"{POSTERIOR_SUMMARY_CSV}"
)


# =============================================================================
# Posterior parameter forest plot
# =============================================================================

# Parameters most useful for interpretation.
# You can add/remove later if desired.

FOREST_PARAMETERS = [
    "gamma",
    "a0",
    "aT",
    "aTT",
    "ae",
    "aR",
    "betaH0",
    "psi",
    "nb_kappa",
    "bT_H",
    "be_H",
    "bR_H",
]


forest_df = (
    posterior_summary_df[
        posterior_summary_df[
            "parameter"
        ].isin(
            FOREST_PARAMETERS
        )
    ]
    .copy()
)


# Preserve desired ordering
forest_df["order"] = forest_df[
    "parameter"
].map({
    name: i
    for i, name
    in enumerate(
        FOREST_PARAMETERS
    )
})

forest_df = (
    forest_df
    .sort_values(
        "order",
        ascending=False,
    )
    .reset_index(drop=True)
)


fig, ax = plt.subplots(
    figsize=(6.5, 5.6)
)


y = np.arange(
    len(forest_df)
)


median = forest_df[
    "posterior_median"
].to_numpy()

lower = forest_df[
    "posterior_q025"
].to_numpy()

upper = forest_df[
    "posterior_q975"
].to_numpy()


xerr = np.vstack([
    median - lower,
    upper - median,
])


ax.errorbar(
    median,
    y,
    xerr=xerr,
    fmt="o",
    capsize=3,
    linewidth=1.5,
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)


ax.set_yticks(
    y
)

ax.set_yticklabels(
    forest_df[
        "parameter"
    ]
)


ax.set_xlabel(
    "Posterior median and 95% credible interval"
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


fig.tight_layout()


FOREST_FIG = (
    OUTPUT_DIR
    / "posterior_parameter_forest_all_climate.png"
)

fig.savefig(
    FOREST_FIG,
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


print(
    f"\n[OK] Saved parameter forest plot:\n"
    f"{FOREST_FIG}"
)


# =============================================================================
# Posterior predictive analysis
# =============================================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

n_available = len(
    posterior_samples
)

n_draws = min(
    N_POSTERIOR_DRAWS,
    n_available,
)


draw_indices = rng.choice(
    n_available,
    size=n_draws,
    replace=False,
)

posterior_draws = posterior_samples[
    draw_indices
]


lik = PlagueLikelihood(
    merged_df=merged_df,
    betaH_mode="all_climate",
    fit_target=FIT_TARGET,
)


# CRITICAL:
# Replace likelihood bounds with the final bounds stored in the final run.

lik.bounds = list(
    saved["bounds"]
)


kappa_index = param_names.index(
    "nb_kappa"
)


mu_draws = np.full(
    (
        n_draws,
        n_weeks,
    ),
    np.nan,
)

predictive_draws = np.full(
    (
        n_draws,
        n_weeks,
    ),
    np.nan,
)


print("\n" + "=" * 100)
print("FINAL POSTERIOR PREDICTIVE ANALYSIS")
print("=" * 100)


for i, theta in enumerate(
    posterior_draws
):

    mu = lik.predict_weekly_deaths(
        theta
    )

    mu = np.asarray(
        mu,
        dtype=float,
    )

    if len(mu) != n_weeks:
        raise RuntimeError(
            "Prediction length mismatch."
        )

    mu_draws[
        i,
        :
    ] = mu

    kappa = float(
        theta[
            kappa_index
        ]
    )

    predictive_draws[
        i,
        :
    ] = simulate_negative_binomial(
        mu=mu,
        kappa=kappa,
        rng=rng,
    )

    if (
        (i + 1) % 100
        == 0
    ):
        print(
            f"Processed "
            f"{i + 1} / "
            f"{n_draws}"
        )


# =============================================================================
# Posterior predictive summaries
# =============================================================================

mu_q025 = np.quantile(
    mu_draws,
    0.025,
    axis=0,
)

mu_median = np.quantile(
    mu_draws,
    0.50,
    axis=0,
)

mu_q975 = np.quantile(
    mu_draws,
    0.975,
    axis=0,
)


pred_q025 = np.quantile(
    predictive_draws,
    0.025,
    axis=0,
)

pred_median = np.quantile(
    predictive_draws,
    0.50,
    axis=0,
)

pred_q975 = np.quantile(
    predictive_draws,
    0.975,
    axis=0,
)


inside_mu = (
    (observed >= mu_q025)
    &
    (observed <= mu_q975)
)

inside_pred = (
    (observed >= pred_q025)
    &
    (observed <= pred_q975)
)


mean_coverage = float(
    np.mean(
        inside_mu
    )
)

predictive_coverage = float(
    np.mean(
        inside_pred
    )
)


posterior_median_sse = float(
    np.sum(
        (
            observed
            - mu_median
        ) ** 2
    )
)


print(
    f"\nExpected-mean 95% interval coverage: "
    f"{100 * mean_coverage:.1f}%"
)

print(
    f"Posterior predictive 95% interval coverage: "
    f"{100 * predictive_coverage:.1f}%"
)

print(
    f"Posterior-median SSE: "
    f"{posterior_median_sse:.3f}"
)


# =============================================================================
# Save predictive summary
# =============================================================================

predictive_summary_df = pd.DataFrame({
    "week_ending":
        weeks,

    "observed_deaths":
        observed,

    "mu_q025":
        mu_q025,

    "mu_median":
        mu_median,

    "mu_q975":
        mu_q975,

    "predictive_q025":
        pred_q025,

    "predictive_median":
        pred_median,

    "predictive_q975":
        pred_q975,

    "observed_inside_mu95":
        inside_mu,

    "observed_inside_predictive95":
        inside_pred,
})


PREDICTIVE_CSV = (
    OUTPUT_DIR
    / "posterior_predictive_summary_all_climate_final.csv"
)

predictive_summary_df.to_csv(
    PREDICTIVE_CSV,
    index=False,
)


# =============================================================================
# Posterior predictive figure
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.2, 4.8)
)


# Full predictive uncertainty
ax.fill_between(
    weeks,
    pred_q025,
    pred_q975,
    color=COLOR_MODEL,
    alpha=0.12,
    linewidth=0,
    label="95% posterior predictive interval",
)


# Parameter uncertainty in expected mean
ax.fill_between(
    weeks,
    mu_q025,
    mu_q975,
    color=COLOR_MODEL,
    alpha=0.28,
    linewidth=0,
    label="95% credible interval",
)


# Posterior median
ax.plot(
    weeks,
    mu_median,
    color=COLOR_MODEL,
    linewidth=2.2,
    label="Posterior median",
    zorder=4,
)


# Observations
ax.scatter(
    weeks,
    observed,
    s=17,
    color=COLOR_OBS,
    alpha=0.80,
    edgecolors="none",
    label="Observed deaths",
    zorder=5,
)


ax.set_xlabel(
    "Week ending"
)

ax.set_ylabel(
    "Weekly plague deaths"
)


ax.xaxis.set_major_locator(
    mdates.MonthLocator(
        interval=4
    )
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%Y-%m"
    )
)


ax.set_xlim(
    weeks.min(),
    weeks.max(),
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


ax.grid(False)


ax.legend(
    frameon=False,
    loc="upper left",
)


fig.tight_layout()


PREDICTIVE_FIG = (
    OUTPUT_DIR
    / "posterior_predictive_all_climate_final.png"
)

fig.savefig(
    PREDICTIVE_FIG,
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


print(
    f"\n[OK] Saved posterior predictive figure:\n"
    f"{PREDICTIVE_FIG}"
)


# =============================================================================
# Final model comparison figure
# =============================================================================

if not FINAL_COMPARISON_CSV.exists():
    raise FileNotFoundError(
        f"Could not find:\n"
        f"{FINAL_COMPARISON_CSV}"
    )


comparison_df = pd.read_csv(
    FINAL_COMPARISON_CSV
)


# Ensure best model first
comparison_df = (
    comparison_df
    .sort_values(
        "delta_logZ",
        ascending=True,
    )
    .reset_index(drop=True)
)


MODEL_LABELS = {
    "constant":
        r"Constant $\beta_H$",

    "sigmoid":
        r"Sigmoid $\beta_H$",

    "humidity_only":
        r"Vapour-pressure-dependent $\beta_H$",

    "all_climate":
        r"Climate-driven $\beta_H$",
}


comparison_df[
    "label"
] = comparison_df[
    "betaH_mode"
].map(
    MODEL_LABELS
)


fig, ax = plt.subplots(
    figsize=(6.4, 4.2)
)


y = np.arange(
    len(
        comparison_df
    )
)


ax.barh(
    y,
    comparison_df[
        "delta_logZ"
    ],
)


ax.set_yticks(
    y
)

ax.set_yticklabels(
    comparison_df[
        "label"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    r"$\Delta \log Z$ relative to best model"
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


ax.grid(
    False
)


fig.tight_layout()


MODEL_COMPARISON_FIG = (
    OUTPUT_DIR
    / "model_comparison_delta_logZ.png"
)

fig.savefig(
    MODEL_COMPARISON_FIG,
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


print(
    f"\n[OK] Saved model comparison figure:\n"
    f"{MODEL_COMPARISON_FIG}"
)


# =============================================================================
# Save paper comparison table
# =============================================================================

PAPER_COMPARISON_CSV = (
    OUTPUT_DIR
    / "final_model_comparison_for_paper.csv"
)

comparison_df.to_csv(
    PAPER_COMPARISON_CSV,
    index=False,
)


print(
    f"\n[OK] Saved model comparison table:\n"
    f"{PAPER_COMPARISON_CSV}"
)


# =============================================================================
# Final report
# =============================================================================

print("\n" + "=" * 100)
print("FINAL FIGURES COMPLETE")
print("=" * 100)

print(
    f"\nPosterior predictive coverage: "
    f"{100 * predictive_coverage:.1f}%"
)

print(
    f"Expected-mean interval coverage: "
    f"{100 * mean_coverage:.1f}%"
)

print(
    f"Posterior median SSE: "
    f"{posterior_median_sse:.3f}"
)

print(
    f"\nAll outputs saved in:\n"
    f"{OUTPUT_DIR}"
)