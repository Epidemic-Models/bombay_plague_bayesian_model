# =============================================================================
# final_climate_response_curves.py
# =============================================================================
"""
Final posterior climate-response curves for the all-climate plague model.

Purpose
-------
This script takes posterior samples from the FINAL all-climate nested-sampling
fit and translates the climate coefficients into interpretable multiplicative
effects on beta_H.

For each standardized climate covariate z:

    Temperature:
        M_T(z) = exp(bT_H * z)

    Humidity / vapour-pressure-related covariate:
        M_e(z) = exp(be_H * z)

    Rainfall:
        M_R(z) = exp(bR_H * z)

The other climate covariates are held at their standardized mean (z = 0).

For each response curve the script calculates:
    - posterior median
    - 50% credible interval
    - 95% credible interval

It also calculates posterior distributions of the multiplicative effect
associated with a +1 SD change in each climate covariate.

IMPORTANT
---------
This script DOES NOT refit the model.
It reads the posterior from the final all-climate model comparison.

Expected final result file:
    results/final_model_comparison/nested_sampling/
        all_climate__plague_deaths_reported_fit__final.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/bornalid/PycharmProjects/bombay_plague_bayesian_model"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison"
    / "nested_sampling"
    / "all_climate__plague_deaths_reported_fit__final.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "climate_response_curves"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL INFORMATION
# =============================================================================

# Parameter order used in your final all-climate model.
PARAM_NAMES = [
    "N0",
    "C_seed",
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

CLIMATE_PARAMETERS = {
    "Temperature": "bT_H",
    "Humidity": "be_H",
    "Rainfall": "bR_H",
}


# Range of standardized climate covariate values.
Z_MIN = -2.5
Z_MAX = 2.5
N_GRID = 301

z_grid = np.linspace(Z_MIN, Z_MAX, N_GRID)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def weighted_quantile(values, quantiles, weights):
    """
    Compute weighted quantiles.

    Parameters
    ----------
    values : array-like
    quantiles : array-like
        Values between 0 and 1.
    weights : array-like

    Returns
    -------
    np.ndarray
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)

    sorter = np.argsort(values)

    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(quantiles, cumulative, values)


def get_equal_weight_samples(results, rng):
    """
    Convert dynesty weighted posterior samples into approximately
    equal-weight posterior samples.

    This function avoids depending on dynesty.utils.resample_equal,
    so the script is robust to slightly different saved-result formats.
    """

    samples = np.asarray(results.samples, dtype=float)

    logwt = np.asarray(results.logwt, dtype=float)
    logz_final = float(np.asarray(results.logz)[-1])

    weights = np.exp(logwt - logz_final)
    weights = weights / np.sum(weights)

    n_samples = len(samples)

    indices = rng.choice(
        np.arange(n_samples),
        size=n_samples,
        replace=True,
        p=weights,
    )

    return samples[indices], weights


def summarize_curve(coefficient_samples, z_values):
    """
    Calculate posterior climate-response curve:

        multiplier = exp(coefficient * z)

    Returns posterior median, 50% CrI and 95% CrI.
    """

    coefficient_samples = np.asarray(coefficient_samples)

    # Shape:
    # posterior samples x z-grid
    response = np.exp(
        coefficient_samples[:, None] * z_values[None, :]
    )

    q025 = np.quantile(response, 0.025, axis=0)
    q25 = np.quantile(response, 0.25, axis=0)
    median = np.quantile(response, 0.50, axis=0)
    q75 = np.quantile(response, 0.75, axis=0)
    q975 = np.quantile(response, 0.975, axis=0)

    return {
        "q025": q025,
        "q25": q25,
        "median": median,
        "q75": q75,
        "q975": q975,
    }


# =============================================================================
# LOAD POSTERIOR
# =============================================================================

print("=" * 90)
print("FINAL CLIMATE RESPONSE CURVES")
print("=" * 90)
print()

print("Loading final all-climate posterior:")
print(RESULT_FILE)
print()

if not RESULT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find final posterior file:\n{RESULT_FILE}"
    )

with open(RESULT_FILE, "rb") as f:
    loaded = pickle.load(f)


# Your saved file may either contain the dynesty Results object directly
# or contain it inside a dictionary.
if hasattr(loaded, "samples"):
    results = loaded

elif isinstance(loaded, dict):

    possible_keys = [
        "results",
        "result",
        "dynesty_results",
        "sampler_results",
    ]

    results = None

    for key in possible_keys:
        if key in loaded and hasattr(loaded[key], "samples"):
            results = loaded[key]
            break

    if results is None:
        raise ValueError(
            "The pickle was loaded, but I could not find a dynesty "
            "Results object inside it."
        )

else:
    raise ValueError(
        "Unrecognized saved-result format."
    )


samples = np.asarray(results.samples)

if samples.shape[1] != len(PARAM_NAMES):
    raise ValueError(
        "\nParameter-count mismatch.\n"
        f"Posterior contains {samples.shape[1]} parameters, "
        f"but PARAM_NAMES contains {len(PARAM_NAMES)}.\n"
        "Check the parameter ordering before continuing."
    )


print(f"Raw nested-sampling samples: {len(samples)}")
print(f"Number of parameters: {samples.shape[1]}")

if hasattr(results, "logz"):
    print(
        f"log(Z) = {float(np.asarray(results.logz)[-1]):.3f}"
    )

print()


# =============================================================================
# GENERATE EQUAL-WEIGHT POSTERIOR
# =============================================================================

rng = np.random.default_rng(2026)

posterior_samples, original_weights = get_equal_weight_samples(
    results,
    rng,
)

print(
    f"Equal-weight posterior samples generated: "
    f"{len(posterior_samples)}"
)
print()


posterior = pd.DataFrame(
    posterior_samples,
    columns=PARAM_NAMES,
)


# =============================================================================
# CHECK CLIMATE COEFFICIENTS
# =============================================================================

print("=" * 90)
print("POSTERIOR CLIMATE COEFFICIENT CHECK")
print("=" * 90)
print()

for label, parameter in CLIMATE_PARAMETERS.items():

    x = posterior[parameter].to_numpy()

    q025, median, q975 = np.quantile(
        x,
        [0.025, 0.50, 0.975],
    )

    p_positive = np.mean(x > 0)
    p_negative = np.mean(x < 0)

    print(label)
    print("-" * 50)
    print(f"Parameter: {parameter}")
    print(f"Median = {median:.4f}")
    print(f"95% CrI = ({q025:.4f}, {q975:.4f})")
    print(f"P(effect > 0 | data) = {p_positive:.4f}")
    print(f"P(effect < 0 | data) = {p_negative:.4f}")
    print()


# =============================================================================
# CALCULATE RESPONSE CURVES
# =============================================================================

print("=" * 90)
print("CALCULATING POSTERIOR RESPONSE CURVES")
print("=" * 90)
print()

curve_results = {}

for label, parameter in CLIMATE_PARAMETERS.items():

    coefficient_samples = posterior[parameter].to_numpy()

    curve_results[label] = summarize_curve(
        coefficient_samples,
        z_grid,
    )

    print(f"[OK] {label}")


# =============================================================================
# SAVE CURVE DATA
# =============================================================================

rows = []

for label, parameter in CLIMATE_PARAMETERS.items():

    curve = curve_results[label]

    for i, z in enumerate(z_grid):

        rows.append(
            {
                "climate_variable": label,
                "parameter": parameter,
                "z": z,
                "multiplier_q025": curve["q025"][i],
                "multiplier_q25": curve["q25"][i],
                "multiplier_median": curve["median"][i],
                "multiplier_q75": curve["q75"][i],
                "multiplier_q975": curve["q975"][i],
            }
        )


curve_df = pd.DataFrame(rows)

curve_csv = OUTPUT_DIR / "posterior_climate_response_curves.csv"

curve_df.to_csv(
    curve_csv,
    index=False,
)

print()
print("[OK] Saved response-curve data:")
print(curve_csv)


# =============================================================================
# +1 SD EFFECTS
# =============================================================================

print()
print("=" * 90)
print("EFFECT OF A +1 SD CHANGE IN EACH CLIMATE COVARIATE")
print("=" * 90)
print()

effect_rows = []

for label, parameter in CLIMATE_PARAMETERS.items():

    coefficient_samples = posterior[parameter].to_numpy()

    # Because z changes from 0 to +1:
    #
    # multiplier = exp(b * 1)
    #
    multipliers = np.exp(coefficient_samples)

    q025, q25, median, q75, q975 = np.quantile(
        multipliers,
        [0.025, 0.25, 0.50, 0.75, 0.975],
    )

    percent_change = (multipliers - 1.0) * 100.0

    pc025, pc50, pc975 = np.quantile(
        percent_change,
        [0.025, 0.50, 0.975],
    )

    probability_increase = np.mean(multipliers > 1.0)
    probability_decrease = np.mean(multipliers < 1.0)

    effect_rows.append(
        {
            "climate_variable": label,
            "parameter": parameter,
            "multiplier_q025": q025,
            "multiplier_q25": q25,
            "multiplier_median": median,
            "multiplier_q75": q75,
            "multiplier_q975": q975,
            "percent_change_q025": pc025,
            "percent_change_median": pc50,
            "percent_change_q975": pc975,
            "probability_increase": probability_increase,
            "probability_decrease": probability_decrease,
        }
    )

    print(label)
    print("-" * 60)

    print(
        f"Median multiplier for +1 SD = "
        f"{median:.3f}"
    )

    print(
        f"95% CrI multiplier = "
        f"({q025:.3f}, {q975:.3f})"
    )

    print(
        f"Median percentage change = "
        f"{pc50:+.1f}%"
    )

    print(
        f"95% CrI percentage change = "
        f"({pc025:+.1f}%, {pc975:+.1f}%)"
    )

    print(
        f"P(increase | data) = "
        f"{probability_increase:.4f}"
    )

    print(
        f"P(decrease | data) = "
        f"{probability_decrease:.4f}"
    )

    print()


effect_df = pd.DataFrame(effect_rows)

effect_csv = OUTPUT_DIR / "posterior_plus1SD_climate_effects.csv"

effect_df.to_csv(
    effect_csv,
    index=False,
)

print("[OK] Saved +1 SD effects:")
print(effect_csv)


# =============================================================================
# FIGURE 1:
# THREE RESPONSE CURVES
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.8),
    sharey=False,
)

for ax, (label, parameter) in zip(
    axes,
    CLIMATE_PARAMETERS.items(),
):

    curve = curve_results[label]

    # 95% credible interval
    ax.fill_between(
        z_grid,
        curve["q025"],
        curve["q975"],
        alpha=0.20,
        label="95% CrI",
    )

    # 50% credible interval
    ax.fill_between(
        z_grid,
        curve["q25"],
        curve["q75"],
        alpha=0.30,
        label="50% CrI",
    )

    # posterior median
    ax.plot(
        z_grid,
        curve["median"],
        linewidth=2.2,
        label="Posterior median",
    )

    # no-effect line
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
    )

    # standardized mean
    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_title(label)

    ax.set_xlabel(
        "Standardized climate covariate (SD)"
    )

    ax.set_ylabel(
        r"Relative multiplier on $\beta_H$"
    )

    ax.grid(
        alpha=0.20,
    )


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Posterior climate-response functions",
    fontsize=14,
    y=1.03,
)

fig.tight_layout()

figure_path = (
    OUTPUT_DIR
    / "posterior_climate_response_curves.png"
)

fig.savefig(
    figure_path,
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("[OK] Saved response-curve figure:")
print(figure_path)


# =============================================================================
# FIGURE 2:
# ALL THREE MEDIAN RESPONSE CURVES TOGETHER
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8, 6),
)

for label in CLIMATE_PARAMETERS:

    curve = curve_results[label]

    ax.plot(
        z_grid,
        curve["median"],
        linewidth=2.2,
        label=label,
    )


ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
)

ax.axvline(
    0.0,
    linestyle=":",
    linewidth=1.0,
)

ax.set_xlabel(
    "Standardized climate covariate (SD)"
)

ax.set_ylabel(
    r"Relative multiplier on $\beta_H$"
)

ax.set_title(
    "Posterior median climate-response functions"
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.20,
)

fig.tight_layout()

combined_path = (
    OUTPUT_DIR
    / "posterior_climate_response_curves_combined.png"
)

fig.savefig(
    combined_path,
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)

print("[OK] Saved combined response figure:")
print(combined_path)


# =============================================================================
# FIGURE 3:
# +1 SD MULTIPLICATIVE EFFECTS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7, 5),
)

labels_plot = []
medians = []
lower_errors = []
upper_errors = []

for row in effect_rows:

    labels_plot.append(
        row["climate_variable"]
    )

    median = row["multiplier_median"]
    lower = row["multiplier_q025"]
    upper = row["multiplier_q975"]

    medians.append(median)

    lower_errors.append(
        median - lower
    )

    upper_errors.append(
        upper - median
    )


x_positions = np.arange(
    len(labels_plot)
)

ax.errorbar(
    x_positions,
    medians,
    yerr=[
        lower_errors,
        upper_errors,
    ],
    fmt="o",
    capsize=5,
    linewidth=1.5,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    labels_plot
)

ax.set_ylabel(
    r"Multiplier on $\beta_H$ for +1 SD"
)

ax.set_title(
    "Posterior effect of a +1 SD climate change"
)

ax.grid(
    axis="y",
    alpha=0.20,
)

fig.tight_layout()

plus1_path = (
    OUTPUT_DIR
    / "posterior_plus1SD_climate_effects.png"
)

fig.savefig(
    plus1_path,
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)

print("[OK] Saved +1 SD figure:")
print(plus1_path)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print()

for row in effect_rows:

    print(
        f"{row['climate_variable']:12s}: "
        f"+1 SD -> median beta_H multiplier = "
        f"{row['multiplier_median']:.3f} "
        f"[95% CrI "
        f"{row['multiplier_q025']:.3f}, "
        f"{row['multiplier_q975']:.3f}]"
    )


print()
print("Interpretation:")
print("-" * 90)
print(
    "A multiplier of 1 means no change relative to the standardized "
    "climate mean."
)

print(
    "A multiplier > 1 means higher beta_H relative to the mean."
)

print(
    "A multiplier < 1 means lower beta_H relative to the mean."
)

print()
print(
    "These are conditional climate-response relationships from the "
    "fitted model. They should not automatically be interpreted as "
    "causal effects."
)

print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)
print()

print(curve_csv)
print(effect_csv)
print(figure_path)
print(combined_path)
print(plus1_path)

print()
print("Climate-response analysis completed successfully.")