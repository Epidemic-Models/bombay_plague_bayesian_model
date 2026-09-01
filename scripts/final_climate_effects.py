# =============================================================================
# final_climate_effects.py
# =============================================================================
"""
Final posterior climate-effect analysis for the all-climate model --
BOTH transmission pathways.

*** Reads from results/final_model_comparison_v2/, the correct final
posterior (betaH0<=3.0, rslice sampling, seeded live points). Do NOT
point this back at results/final_model_comparison/ (no _v2 suffix) --
that's the superseded round-1 result. ***

*** EXTENDED: earlier versions of this script only analyzed the three
human-transmission climate coefficients (bT_H, be_H, bR_H). That left the
spillover pathway's climate coefficients (aT, aTT, ae, aR) reported only
as single best-fit point estimates, with no posterior uncertainty
quantification -- an asymmetry that meant "which mechanism, how strongly,
in general" was only half-answered. This version reports full posterior
summaries for BOTH pathways, so the climate-effect picture is complete
and symmetric across spillover and human transmission. ***

For each coefficient (in both pathways) this script reports:

    - posterior mean
    - posterior median
    - posterior 95% credible interval
    - P(coefficient > 0 | data)
    - P(coefficient < 0 | data)

It generates:

    1. posterior_climate_coefficients_human.png
       Horizontal posterior median + 95% CrI figure, human-transmission
       pathway (bT_H, be_H, bR_H).

    2. posterior_climate_coefficients_spillover.png
       Same, for the spillover pathway (aT, aTT, ae, aR). Note aTT is a
       quadratic term -- its sign indicates curvature direction, not a
       simple "more/less" effect; see the printed interpretation notes.

    3. posterior_climate_densities_human.png
       Posterior density distributions, human-transmission coefficients.

    4. posterior_climate_densities_spillover.png
       Posterior density distributions, spillover coefficients.

    5. posterior_climate_effects_summary.csv
       Combined summary table, both pathways, with a "pathway" column.

Outputs:
    results/final_figures/climate_effects/
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynesty.utils import resample_equal


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIT_TARGET = "plague_deaths_reported_fit"
RANDOM_SEED = 12345

# *** final_v2 -- the corrected run. Do NOT point this back at
# results/final_model_comparison/ (no suffix) -- that's the superseded
# round-1 result, where betaH0 was still pinned at its old 1.5 ceiling. ***
RESULT_PKL = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
    / "nested_sampling"
    / f"all_climate__{FIT_TARGET}__final_v2.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "climate_effects"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "posterior_climate_effects_summary.csv"

COEFFICIENT_FIG_HUMAN = OUTPUT_DIR / "posterior_climate_coefficients_human.png"
COEFFICIENT_FIG_SPILLOVER = OUTPUT_DIR / "posterior_climate_coefficients_spillover.png"
DENSITY_FIG_HUMAN = OUTPUT_DIR / "posterior_climate_densities_human.png"
DENSITY_FIG_SPILLOVER = OUTPUT_DIR / "posterior_climate_densities_spillover.png"


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
# Climate coefficients -- BOTH pathways
# =============================================================================

HUMAN_PARAMETERS = ["bT_H", "be_H", "bR_H"]
SPILLOVER_PARAMETERS = ["aT", "aTT", "ae", "aR"]

PARAMETER_LABELS = {
    "bT_H": r"Temperature effect $b_{T,H}$ (human)",
    "be_H": r"Vapour-pressure effect $b_{e,H}$ (human)",
    "bR_H": r"Rainfall effect $b_{R,H}$ (human)",
    "aT": r"Temperature effect $a_T$ (spillover)",
    "aTT": r"Temperature$^2$ effect $a_{TT}$ (spillover)",
    "ae": r"Vapour-pressure effect $a_e$ (spillover)",
    "aR": r"Rainfall effect $a_R$ (spillover)",
}

PARAMETER_PATHWAY = {
    "bT_H": "human", "be_H": "human", "bR_H": "human",
    "aT": "spillover", "aTT": "spillover", "ae": "spillover", "aR": "spillover",
}

ALL_PARAMETERS = HUMAN_PARAMETERS + SPILLOVER_PARAMETERS


# =============================================================================
# Load final posterior
# =============================================================================

if not RESULT_PKL.exists():
    raise FileNotFoundError(
        f"Could not find final all-climate result:\n{RESULT_PKL}\n"
        "This script expects results/final_model_comparison_v2/ -- run "
        "scripts/run_final_all_models_v2.py first if this directory doesn't exist."
    )

print(f"Reading posterior from:\n{RESULT_PKL}\n")

with open(RESULT_PKL, "rb") as f:
    saved = pickle.load(f)

results = saved["results"]
param_names = list(saved["param_names"])

weights = np.exp(results.logwt - results.logz[-1])
rstate = np.random.default_rng(RANDOM_SEED)
posterior = resample_equal(results.samples, weights, rstate=rstate)
posterior_df = pd.DataFrame(posterior, columns=param_names)

print("\n" + "=" * 90)
print("FINAL CLIMATE-EFFECT POSTERIOR ANALYSIS (final_v2, both pathways)")
print("=" * 90)

print(f"\nPosterior samples: {len(posterior_df)}")
print(f"log(Z) = {results.logz[-1]:.3f} +/- {results.logzerr[-1]:.3f}")
print(
    "\n[Sanity check: this logZ should be approximately -813.6, matching "
    "the final_v2 model-comparison table. If it instead reads approximately "
    "-814.2, this script is STILL reading the superseded round-1 result --"
    " check RESULT_PKL above.]"
)


# =============================================================================
# Summaries -- both pathways, identical treatment
# =============================================================================

summary_rows = []

for parameter in ALL_PARAMETERS:

    if parameter not in posterior_df.columns:
        raise KeyError(f"{parameter!r} not found in posterior.")

    values = posterior_df[parameter].to_numpy(dtype=float)

    q025, q25, median, q75, q975 = np.quantile(values, [0.025, 0.25, 0.50, 0.75, 0.975])

    p_positive = float(np.mean(values > 0))
    p_negative = float(np.mean(values < 0))

    summary_rows.append({
        "parameter": parameter,
        "pathway": PARAMETER_PATHWAY[parameter],
        "label": PARAMETER_LABELS[parameter],
        "posterior_mean": float(np.mean(values)),
        "posterior_sd": float(np.std(values)),
        "posterior_q025": float(q025),
        "posterior_q25": float(q25),
        "posterior_median": float(median),
        "posterior_q75": float(q75),
        "posterior_q975": float(q975),
        "probability_positive": p_positive,
        "probability_negative": p_negative,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)


# =============================================================================
# Print interpretation table -- both pathways
# =============================================================================

for pathway_name, param_list in [("HUMAN TRANSMISSION", HUMAN_PARAMETERS),
                                   ("SPILLOVER", SPILLOVER_PARAMETERS)]:
    print(f"\n{'='*90}\n{pathway_name} PATHWAY COEFFICIENTS\n{'='*90}")

    for parameter in param_list:
        row = summary_df.loc[summary_df["parameter"] == parameter].iloc[0]

        print(f"\n{parameter}")
        print(f"  median = {row['posterior_median']:.4f}")
        print(f"  95% CrI = ({row['posterior_q025']:.4f}, {row['posterior_q975']:.4f})")
        print(f"  P(effect > 0 | data) = {row['probability_positive']:.4f}")
        print(f"  P(effect < 0 | data) = {row['probability_negative']:.4f}")

        if parameter == "aTT":
            print(
                "  NOTE: aTT is the coefficient on z_T^2 (quadratic term) -- its "
                "sign indicates whether the temperature response curves upward "
                "(aTT>0) or downward/saturates (aTT<0) away from the linear (aT) "
                "trend, not a simple 'more/less spillover' direction. Interpret "
                "jointly with aT, not in isolation."
            )


# =============================================================================
# Figures -- separate for each pathway (7 coefficients would be crowded
# and not biologically comparable on one axis)
# =============================================================================

def make_coefficient_figure(param_list, fig_path, title):
    plot_df = summary_df.set_index("parameter").loc[param_list].reset_index()

    y = np.arange(len(plot_df))
    median = plot_df["posterior_median"].to_numpy()
    lower = plot_df["posterior_q025"].to_numpy()
    upper = plot_df["posterior_q975"].to_numpy()
    xerr = np.vstack([median - lower, upper - median])

    fig, ax = plt.subplots(figsize=(6.3, 0.9 * len(plot_df) + 1.6))
    ax.errorbar(median, y, xerr=xerr, fmt="o", capsize=4, linewidth=1.7, markersize=7)
    ax.axvline(0, linestyle="--", linewidth=1.2, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Posterior coefficient")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_density_figure(param_list, fig_path, title):
    fig, axes = plt.subplots(nrows=len(param_list), ncols=1,
                               figsize=(6.4, 1.8 * len(param_list) + 0.6), sharex=False)
    if len(param_list) == 1:
        axes = [axes]

    for ax, parameter in zip(axes, param_list):
        values = posterior_df[parameter].to_numpy(dtype=float)
        row = summary_df.loc[summary_df["parameter"] == parameter].iloc[0]

        ax.hist(values, bins=55, density=True, alpha=0.75, edgecolor="none")
        ax.axvline(0, linestyle="--", linewidth=1.1, label="Zero effect")
        ax.axvline(row["posterior_median"], linestyle="-", linewidth=1.6, label="Posterior median")
        ax.axvline(row["posterior_q025"], linestyle=":", linewidth=1.2, label="95% CrI")
        ax.axvline(row["posterior_q975"], linestyle=":", linewidth=1.2)
        ax.set_ylabel("Posterior density")
        ax.set_title(PARAMETER_LABELS[parameter])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    axes[-1].set_xlabel("Coefficient value")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


make_coefficient_figure(HUMAN_PARAMETERS, COEFFICIENT_FIG_HUMAN,
                          "Human-transmission climate coefficients")
make_coefficient_figure(SPILLOVER_PARAMETERS, COEFFICIENT_FIG_SPILLOVER,
                          "Spillover climate coefficients")
make_density_figure(HUMAN_PARAMETERS, DENSITY_FIG_HUMAN,
                      "Human-transmission climate coefficients")
make_density_figure(SPILLOVER_PARAMETERS, DENSITY_FIG_SPILLOVER,
                      "Spillover climate coefficients")


# =============================================================================
# Final summary -- both pathways, side by side
# =============================================================================

print("\n" + "=" * 90)
print("FINAL CLIMATE-EFFECT SUMMARY -- BOTH PATHWAYS")
print("=" * 90)

for pathway_name, param_list in [("Human transmission", HUMAN_PARAMETERS),
                                   ("Spillover", SPILLOVER_PARAMETERS)]:
    print(f"\n--- {pathway_name} ---")
    for parameter in param_list:
        row = summary_df.loc[summary_df["parameter"] == parameter].iloc[0]
        q025, q975 = row["posterior_q025"], row["posterior_q975"]

        if q025 > 0:
            interpretation = "95% CrI entirely above zero (resolved positive)"
        elif q975 < 0:
            interpretation = "95% CrI entirely below zero (resolved negative)"
        else:
            interpretation = "95% CrI includes zero (not resolved)"

        print(f"  {parameter}: median={row['posterior_median']:.4f}, "
              f"95% CrI=({q025:.4f}, {q975:.4f}) -- {interpretation}")

print(f"\n[OK] Saved summary table:\n{SUMMARY_CSV}")
print(f"[OK] Saved human-pathway coefficient figure:\n{COEFFICIENT_FIG_HUMAN}")
print(f"[OK] Saved spillover-pathway coefficient figure:\n{COEFFICIENT_FIG_SPILLOVER}")
print(f"[OK] Saved human-pathway density figure:\n{DENSITY_FIG_HUMAN}")
print(f"[OK] Saved spillover-pathway density figure:\n{DENSITY_FIG_SPILLOVER}")

print("\nClimate-effect posterior analysis completed (final_v2, both pathways).")