# =============================================================================
# check_climate_covariate_correlations.py
# =============================================================================
"""
Check statistical dependence among the climate covariates used in the
Bombay plague model and posterior dependence among their fitted coefficients.

This script DOES NOT refit the model.

Analyses
--------
1. Pearson correlation among the fitted weekly climate covariates:
       zT, ze, zR

2. Spearman correlation among the same covariates.

3. Posterior Pearson and Spearman correlations among climate-effect
   coefficients in the final climate-driven beta_H model.

4. Scatterplot of temperature versus vapour pressure.

Outputs
-------
results/final_climate_correlation_analysis/

    climate_covariate_pearson.csv
    climate_covariate_spearman.csv
    posterior_climate_coefficients_pearson.csv
    posterior_climate_coefficients_spearman.csv
    temperature_vapour_pressure_scatter.pdf
    temperature_vapour_pressure_scatter.png
    climate_correlation_summary.txt

Usage
-----
    python scripts/check_climate_covariate_correlations.py
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

MODEL = "all_climate"

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_model_comparison_v2"
    / "nested_sampling"
    / f"{MODEL}__{FIT_TARGET}__final_v2.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_climate_correlation_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 12345


# =============================================================================
# Plot styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


# =============================================================================
# Helper functions
# =============================================================================

def get_result_array(results, name: str) -> np.ndarray:
    """
    Extract an array from dynesty Results regardless of whether it is
    exposed as an attribute or dictionary-like field.
    """

    if hasattr(results, name):
        return np.asarray(getattr(results, name))

    return np.asarray(results[name])


def get_equal_weight_posterior(
    samples: np.ndarray,
    logwt: np.ndarray,
    logz: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Convert dynesty's weighted posterior into an equal-weight posterior.
    """

    weights = np.exp(
        logwt - logz[-1]
    )

    rng = np.random.default_rng(seed)

    posterior = resample_equal(
        samples,
        weights,
        rstate=rng,
    )

    return np.asarray(
        posterior,
        dtype=float,
    )


def find_parameter(
    param_names: list[str],
    candidates: list[str],
) -> str | None:
    """
    Find the first candidate parameter name present in the fitted model.

    This allows for minor naming differences between versions of the code.
    """

    for candidate in candidates:
        if candidate in param_names:
            return candidate

    return None


# =============================================================================
# Main
# =============================================================================

def main():

    print("\n" + "=" * 90)
    print("CLIMATE COVARIATE AND POSTERIOR CORRELATION ANALYSIS")
    print("=" * 90)


    # =========================================================================
    # 1. Load the exact prepared data used for fitting
    # =========================================================================

    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Prepared climate/mortality data not found:\n"
            f"{paths.MERGED_CSV}"
        )

    df = pd.read_csv(
        paths.MERGED_CSV,
        parse_dates=[paths.DATE_COL],
    )

    df = (
        df
        .sort_values(paths.DATE_COL)
        .reset_index(drop=True)
    )

    required = [
        paths.DATE_COL,
        "zT",
        "ze",
        "zR",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


    # =========================================================================
    # 2. Climate covariate correlations
    # =========================================================================

    climate = df[
        [
            "zT",
            "ze",
            "zR",
        ]
    ].copy()

    climate = climate.rename(
        columns={
            "zT": "Temperature",
            "ze": "Vapour pressure",
            "zR": "Rainfall",
        }
    )

    print(
        f"\nNumber of fitted weeks: {len(climate)}"
    )

    print(
        f"Period: "
        f"{df[paths.DATE_COL].min().date()} "
        f"to "
        f"{df[paths.DATE_COL].max().date()}"
    )


    # -------------------------------------------------------------------------
    # Pearson
    # -------------------------------------------------------------------------

    pearson = climate.corr(
        method="pearson"
    )

    print("\n" + "-" * 90)
    print("PEARSON CORRELATION: MODEL CLIMATE COVARIATES")
    print("-" * 90)

    print(
        pearson.round(4).to_string()
    )


    # -------------------------------------------------------------------------
    # Spearman
    # -------------------------------------------------------------------------

    spearman = climate.corr(
        method="spearman"
    )

    print("\n" + "-" * 90)
    print("SPEARMAN CORRELATION: MODEL CLIMATE COVARIATES")
    print("-" * 90)

    print(
        spearman.round(4).to_string()
    )


    # -------------------------------------------------------------------------
    # Specifically extract temperature-vapour-pressure relationship
    # -------------------------------------------------------------------------

    temp_vp_pearson = pearson.loc[
        "Temperature",
        "Vapour pressure",
    ]

    temp_vp_spearman = spearman.loc[
        "Temperature",
        "Vapour pressure",
    ]

    print("\nTemperature vs vapour pressure:")
    print(
        f"  Pearson r  = {temp_vp_pearson:.4f}"
    )
    print(
        f"  Spearman rho = {temp_vp_spearman:.4f}"
    )


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    pearson.to_csv(
        OUTPUT_DIR
        / "climate_covariate_pearson.csv"
    )

    spearman.to_csv(
        OUTPUT_DIR
        / "climate_covariate_spearman.csv"
    )


    # =========================================================================
    # 3. Load final climate-driven posterior
    # =========================================================================

    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Final climate-driven posterior not found:\n"
            f"{RESULT_PATH}"
        )

    print("\n" + "=" * 90)
    print("POSTERIOR COEFFICIENT CORRELATIONS")
    print("=" * 90)

    print(
        f"\nLoading:\n{RESULT_PATH}"
    )

    with open(
        RESULT_PATH,
        "rb",
    ) as f:
        saved = pickle.load(f)

    results = saved["results"]

    param_names = list(
        saved["param_names"]
    )

    print("\nParameters in posterior:")
    print(param_names)


    # =========================================================================
    # 4. Construct equal-weight posterior
    # =========================================================================

    samples = get_result_array(
        results,
        "samples",
    )

    logwt = get_result_array(
        results,
        "logwt",
    )

    logz = get_result_array(
        results,
        "logz",
    )

    posterior = get_equal_weight_posterior(
        samples=samples,
        logwt=logwt,
        logz=logz,
        seed=RANDOM_SEED,
    )

    posterior_df = pd.DataFrame(
        posterior,
        columns=param_names,
    )

    print(
        f"\nEqual-weight posterior samples: "
        f"{len(posterior_df)}"
    )


    # =========================================================================
    # 5. Identify climate coefficient names
    # =========================================================================

    # Spillover coefficients
    aT = find_parameter(
        param_names,
        [
            "aT",
            "a_T",
        ],
    )

    aTT = find_parameter(
        param_names,
        [
            "aTT",
            "a_TT",
        ],
    )

    ae = find_parameter(
        param_names,
        [
            "ae",
            "a_e",
        ],
    )

    aR = find_parameter(
        param_names,
        [
            "aR",
            "a_R",
        ],
    )


    # Human/prevalence-associated amplification coefficients
    bT = find_parameter(
        param_names,
        [
            "bT_H",
            "b_T_H",
            "bT",
            "b_T",
        ],
    )

    be = find_parameter(
        param_names,
        [
            "be_H",
            "b_e_H",
            "be",
            "b_e",
        ],
    )

    bR = find_parameter(
        param_names,
        [
            "bR_H",
            "b_R_H",
            "bR",
            "b_R",
        ],
    )


    selected_parameters = [
        p
        for p in [
            aT,
            aTT,
            ae,
            aR,
            bT,
            be,
            bR,
        ]
        if p is not None
    ]

    if not selected_parameters:
        raise RuntimeError(
            "Could not identify any climate-effect parameters "
            "in the posterior."
        )

    print("\nClimate-effect parameters identified:")
    for p in selected_parameters:
        print(f"  {p}")


    # =========================================================================
    # 6. Posterior correlation matrices
    # =========================================================================

    posterior_pearson = posterior_df[
        selected_parameters
    ].corr(
        method="pearson"
    )

    posterior_spearman = posterior_df[
        selected_parameters
    ].corr(
        method="spearman"
    )


    print("\n" + "-" * 90)
    print("POSTERIOR PEARSON CORRELATION")
    print("-" * 90)

    print(
        posterior_pearson
        .round(4)
        .to_string()
    )


    print("\n" + "-" * 90)
    print("POSTERIOR SPEARMAN CORRELATION")
    print("-" * 90)

    print(
        posterior_spearman
        .round(4)
        .to_string()
    )


    posterior_pearson.to_csv(
        OUTPUT_DIR
        / "posterior_climate_coefficients_pearson.csv"
    )

    posterior_spearman.to_csv(
        OUTPUT_DIR
        / "posterior_climate_coefficients_spearman.csv"
    )


    # =========================================================================
    # 7. Extract the biologically important temperature-moisture posterior
    #    correlations
    # =========================================================================

    print("\n" + "-" * 90)
    print("TEMPERATURE-VAPOUR-PRESSURE COEFFICIENT DEPENDENCE")
    print("-" * 90)


    # -------------------------------------------------------------------------
    # Spillover
    # -------------------------------------------------------------------------

    spillover_corr = np.nan

    if (
        aT is not None
        and
        ae is not None
    ):

        spillover_corr = posterior_df[
            [aT, ae]
        ].corr(
            method="pearson"
        ).iloc[0, 1]

        spillover_spearman = posterior_df[
            [aT, ae]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        print(
            "\nSpillover:"
        )

        print(
            f"  corr({aT}, {ae})"
        )

        print(
            f"  Pearson r = "
            f"{spillover_corr:.4f}"
        )

        print(
            f"  Spearman rho = "
            f"{spillover_spearman:.4f}"
        )

    else:

        spillover_spearman = np.nan

        print(
            "\nCould not identify both spillover "
            "temperature and vapour-pressure coefficients."
        )


    # -------------------------------------------------------------------------
    # Amplification
    # -------------------------------------------------------------------------

    amplification_corr = np.nan

    if (
        bT is not None
        and
        be is not None
    ):

        amplification_corr = posterior_df[
            [bT, be]
        ].corr(
            method="pearson"
        ).iloc[0, 1]

        amplification_spearman = posterior_df[
            [bT, be]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        print(
            "\nPrevalence-associated amplification:"
        )

        print(
            f"  corr({bT}, {be})"
        )

        print(
            f"  Pearson r = "
            f"{amplification_corr:.4f}"
        )

        print(
            f"  Spearman rho = "
            f"{amplification_spearman:.4f}"
        )

    else:

        amplification_spearman = np.nan

        print(
            "\nCould not identify both amplification "
            "temperature and vapour-pressure coefficients."
        )


    # =========================================================================
    # 8. Temperature-vapour-pressure scatterplot
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5)
    )

    ax.scatter(
        climate["Temperature"],
        climate["Vapour pressure"],
        s=20,
        alpha=0.65,
    )


    # -------------------------------------------------------------------------
    # Descriptive least-squares line
    # -------------------------------------------------------------------------

    x = climate[
        "Temperature"
    ].to_numpy(
        dtype=float
    )

    y = climate[
        "Vapour pressure"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    x_line = np.linspace(
        np.min(x[valid]),
        np.max(x[valid]),
        200,
    )

    y_line = (
        intercept
        +
        slope * x_line
    )

    ax.plot(
        x_line,
        y_line,
        linewidth=1.4,
    )


    ax.set_xlabel(
        "Standardized temperature covariate ($z_T$)"
    )

    ax.set_ylabel(
        "Standardized vapour-pressure covariate ($z_e$)"
    )

    ax.set_title(
        "Temperature and vapour pressure in the fitted climate series"
    )


    ax.text(
        0.04,
        0.96,
        (
            f"Pearson $r$ = {temp_vp_pearson:.3f}\n"
            f"Spearman $\\rho$ = {temp_vp_spearman:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()


    scatter_pdf = (
        OUTPUT_DIR
        / "temperature_vapour_pressure_scatter.pdf"
    )

    scatter_png = (
        OUTPUT_DIR
        / "temperature_vapour_pressure_scatter.png"
    )


    fig.savefig(
        scatter_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        scatter_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


    # =========================================================================
    # 9. Save concise summary
    # =========================================================================

    summary_path = (
        OUTPUT_DIR
        / "climate_correlation_summary.txt"
    )

    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "CLIMATE COVARIATE CORRELATION ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Number of fitted weeks: {len(climate)}\n\n"
        )

        f.write(
            "Temperature vs vapour pressure in model input:\n"
        )

        f.write(
            f"Pearson r = {temp_vp_pearson:.6f}\n"
        )

        f.write(
            f"Spearman rho = {temp_vp_spearman:.6f}\n\n"
        )


        f.write(
            "Full Pearson climate correlation matrix:\n"
        )

        f.write(
            pearson.to_string()
        )

        f.write(
            "\n\n"
        )


        f.write(
            "Full Spearman climate correlation matrix:\n"
        )

        f.write(
            spearman.to_string()
        )

        f.write(
            "\n\n"
        )


        if np.isfinite(spillover_corr):

            f.write(
                "Posterior temperature-vapour-pressure "
                "coefficient dependence: spillover\n"
            )

            f.write(
                f"Pearson r = {spillover_corr:.6f}\n"
            )

            f.write(
                f"Spearman rho = "
                f"{spillover_spearman:.6f}\n\n"
            )


        if np.isfinite(amplification_corr):

            f.write(
                "Posterior temperature-vapour-pressure "
                "coefficient dependence: amplification\n"
            )

            f.write(
                f"Pearson r = {amplification_corr:.6f}\n"
            )

            f.write(
                f"Spearman rho = "
                f"{amplification_spearman:.6f}\n"
            )


    # =========================================================================
    # Done
    # =========================================================================

    print("\n" + "=" * 90)
    print("OUTPUTS")
    print("=" * 90)

    print(
        f"\nSaved all results to:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nMost important numbers for the reviewer:"
    )

    print(
        f"  Climate-series T-VP Pearson r: "
        f"{temp_vp_pearson:.4f}"
    )

    print(
        f"  Climate-series T-VP Spearman rho: "
        f"{temp_vp_spearman:.4f}"
    )

    if np.isfinite(spillover_corr):
        print(
            f"  Posterior spillover "
            f"T-VP coefficient correlation: "
            f"{spillover_corr:.4f}"
        )

    if np.isfinite(amplification_corr):
        print(
            f"  Posterior amplification "
            f"T-VP coefficient correlation: "
            f"{amplification_corr:.4f}"
        )

    print("\nDone.")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()