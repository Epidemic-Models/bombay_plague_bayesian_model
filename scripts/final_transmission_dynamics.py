# =============================================================================
# final_transmission_dynamics.py
# =============================================================================
"""
Posterior decomposition of the force of infection into reservoir-associated
spillover (beta_R) and human-associated amplification (beta_H * I / N)
for the FOUR final Bayesian model-comparison runs.

This script reads the final_v2 posterior results and does NOT refit anything.

For each beta_H formulation:

    1. Convert dynesty's weighted posterior into an equal-weight posterior
       reproducibly using a seeded random state.

    2. Select N_POSTERIOR_DRAWS posterior parameter vectors reproducibly.

    3. Save the exact selected posterior parameter vectors used in the
       downstream simulations.

    4. Re-simulate the mechanistic model once per posterior draw.

    5. Decompose the total force of infection into:

           reservoir-associated spillover:
               beta_R(t)

           human-associated amplification:
               beta_H(t) * I(t) / N

           total force of infection:
               lambda(t)

    6. Calculate the amplification reproduction number:

           R0_H(t) = beta_H(t) / gamma

    7. Aggregate rate-like quantities to weekly means.

    8. Compute weekly posterior medians and 95% credible intervals.

IMPORTANT
---------
The amplification contribution is recovered as

    amplification_force(t) = lambda(t) - beta_R(t)

rather than recomputing beta_H(t) * I(t) / N from the stored post-update
I(t). The internal Euler solver evaluates lambda using I from the previous
substep, so this subtraction preserves exact additivity:

    lambda(t)
        =
    beta_R(t)
        +
    amplification_force(t)

at the time step actually used by the simulator.

Reproducibility
---------------
Two stochastic steps occur downstream of the fitted posterior:

    1. equal-weight resampling of dynesty's weighted posterior;
    2. selection of N_POSTERIOR_DRAWS from that equal-weight posterior.

Both are explicitly seeded, with a model-specific offset so each of the
four beta_H formulations gets its own deterministic-but-distinct seed.

The exact selected parameter vectors are also written to CSV, so the
published analysis can later be reproduced without relying on the internal
implementation of dynesty.resample_equal.

Validation
----------
Every posterior trajectory array (spillover, amplification, total force of
infection, and R0_H) is checked for NaN/non-finite values AND for
materially negative values before any summary statistic is computed. All
four quantities are non-negative by construction: beta_R(t) and lambda(t)
are rates, amplification_force(t) = lambda(t) - beta_R(t) is non-negative
because lambda(t) >= beta_R(t) at every internal solver substep (see
IMPORTANT above), and R0_H(t) = beta_H(t) / gamma is a ratio of two
positive quantities. A materially negative value (beyond floating-point
tolerance) in any of these would indicate a real problem in the
decomposition, not a expected numerical artifact.

Outputs
--------------
results/final_figures/transmission_dynamics/

    posterior_transmission_dynamics_<betaH_mode>.csv

    selected_transmission_dynamics_draws_<betaH_mode>.csv

    posterior_human_R0_summary.csv

    posterior_transmission_dynamics_grid.png
    posterior_transmission_dynamics_grid.pdf

    posterior_transmission_dynamics_all_climate.png
    posterior_transmission_dynamics_all_climate.pdf

Usage
-----
    python scripts/final_transmission_dynamics.py
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"

MODELS = [
    "constant",
    "sigmoid",
    "humidity_only",
    "all_climate",
]

MODEL_LABELS = {
    "constant": r"Constant $\beta_H$",
    "sigmoid": r"Sigmoid $\beta_H$",
    "humidity_only": r"Vapour-pressure-dependent $\beta_H$",
    "all_climate": r"Climate-driven $\beta_H$",
}

FINAL_ROOT = PROJECT_ROOT / "results" / "final_model_comparison_v2"
FINAL_NESTED_DIR = FINAL_ROOT / "nested_sampling"

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Posterior settings
# -----------------------------------------------------------------------------

N_POSTERIOR_DRAWS = 1000
RANDOM_SEED = 12345


# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7

I0 = 0.0
E0 = 0.0
R0_INIT = 0.0

WEEK_RULE = "W-FRI"

# Tolerance for the non-negativity check below: values more negative than
# this are treated as a real problem, not floating-point noise.
NEGATIVE_TOL = -1e-10


# =============================================================================
# Plot styling
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

COLOR_SPILLOVER = "#1f77b4"
COLOR_AMPLIFICATION = "#E08214"
COLOR_TOTAL = "#2C7C2C"


# =============================================================================
# Helper functions
# =============================================================================

def load_final_result(betaH_mode: str) -> dict:
    """Load one final_v2 nested-sampling posterior result."""

    pkl_path = (
        FINAL_NESTED_DIR
        / f"{betaH_mode}__{FIT_TARGET}__final_v2.pkl"
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find final_v2 result for {betaH_mode!r}:\n"
            f"{pkl_path}\n\n"
            "Run scripts/run_final_all_models_v2.py first."
        )

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    return saved


def get_equal_weight_posterior(saved: dict, seed: int) -> np.ndarray:
    """
    Convert dynesty's weighted posterior sample cloud into an
    equal-weight posterior reproducibly.
    """

    results = saved["results"]

    weights = np.exp(results.logwt - results.logz[-1])

    rstate = np.random.default_rng(seed)

    posterior = resample_equal(
        results.samples,
        weights,
        rstate=rstate,
    )

    return posterior


def simulate_one_draw(
    theta: np.ndarray,
    param_names: list[str],
    betaH_mode: str,
    zT: np.ndarray,
    ze: np.ndarray,
    zR: np.ndarray,
    start_date: str,
) -> pd.DataFrame:
    """Run the mechanistic model once for one posterior parameter vector."""

    sample = dict(zip(param_names, np.asarray(theta, dtype=float)))

    sim_params = build_sim_params(sample, betaH_mode)

    sim = simulate_seir_dde(
        zT=zT,
        ze=ze,
        zR=zR,
        N=sample["N0"],
        params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0,
        E0=E0,
        R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=betaH_mode,
        start_date=start_date,
    )

    # -------------------------------------------------------------------------
    # Recover the amplification contribution EXACTLY as used by the simulator.
    #
    # lambda = beta_R + beta_H * I_previous / N
    #
    # Recomputing beta_H * current I / N would use the wrong substep state.
    # -------------------------------------------------------------------------

    sim["amplification_force"] = sim["lambda"] - sim["beta_R"]

    return sim


def weekly_mean_trajectory(
    sim: pd.DataFrame,
    week_ending_dates: pd.Series,
    col: str,
) -> np.ndarray:
    """
    Calculate the weekly mean of a rate-like internal-grid quantity and
    align it to the fitted epidemiological week grid.

    Appropriate for:
        beta_R
        amplification_force
        lambda
        R0_H
    """

    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])

    wk = (
        df
        .set_index("date")[col]
        .resample(WEEK_RULE)
        .mean()
        .reset_index()
        .rename(columns={col: f"{col}_week", "date": "week_ending"})
    )

    target = pd.DataFrame({
        "week_ending": pd.to_datetime(week_ending_dates)
    })

    aligned = pd.merge(target, wk, on="week_ending", how="left")

    values = aligned[f"{col}_week"].to_numpy(dtype=float)

    return values


def posterior_quantiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior 2.5%, 50%, and 97.5% quantiles across posterior draws."""

    return (
        np.nanquantile(arr, 0.025, axis=0),
        np.nanquantile(arr, 0.50, axis=0),
        np.nanquantile(arr, 0.975, axis=0),
    )


def validate_array(name: str, arr: np.ndarray, check_nonnegative: bool = False) -> None:
    """
    Fail explicitly if a posterior trajectory array contains invalid values.

    If check_nonnegative is True, also fail if any value is more negative
    than NEGATIVE_TOL -- appropriate for quantities that are non-negative
    by construction (beta_R, amplification_force, lambda, R0_H), so a
    materially negative value indicates a real problem rather than
    floating-point noise.
    """

    n_nan = int(np.isnan(arr).sum())

    if n_nan > 0:
        raise RuntimeError(f"{name} contains {n_nan} NaN values.")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains non-finite values.")

    if check_nonnegative and np.any(arr < NEGATIVE_TOL):
        n_negative = int(np.sum(arr < NEGATIVE_TOL))
        most_negative = float(np.min(arr))
        raise RuntimeError(
            f"{name} contains {n_negative} materially negative values "
            f"(most negative = {most_negative:.3e}, tolerance = "
            f"{NEGATIVE_TOL:.1e}) -- this quantity is non-negative by "
            "construction, so this indicates a real problem, not "
            "floating-point noise."
        )


# =============================================================================
# Main analysis
# =============================================================================

def main():

    # =========================================================================
    # 1. Load prepared data
    # =========================================================================

    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Prepared data not found:\n{paths.MERGED_CSV}\n\n"
            "Run scripts/prepare_data.py first."
        )

    merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
    merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

    required_columns = [paths.DATE_COL, "zT", "ze", "zR", FIT_TARGET]
    missing_columns = [c for c in required_columns if c not in merged_df.columns]

    if missing_columns:
        raise ValueError(
            "Prepared data are missing required columns: " + ", ".join(missing_columns)
        )

    week_ending = pd.to_datetime(merged_df[paths.DATE_COL])
    zT = merged_df["zT"].to_numpy(dtype=float)
    ze = merged_df["ze"].to_numpy(dtype=float)
    zR = merged_df["zR"].to_numpy(dtype=float)
    observed = merged_df[FIT_TARGET].to_numpy(dtype=float)
    start_date = str(pd.Timestamp(week_ending.iloc[0]).date())
    n_weeks = len(merged_df)

    # =========================================================================
    # 2. Main loop over models
    # =========================================================================

    print("\n" + "=" * 100)
    print("FINAL POSTERIOR TRANSMISSION-DYNAMICS DECOMPOSITION")
    print("=" * 100)

    print(f"\nFitted weeks: {n_weeks}")
    print(f"Study period: {week_ending.min().date()} to {week_ending.max().date()}")
    print(f"Posterior draws per model: {N_POSTERIOR_DRAWS}")

    all_model_summaries: dict[str, pd.DataFrame] = {}
    r0_summary_rows = []

    for model_index, betaH_mode in enumerate(MODELS):

        print("\n" + "-" * 80)
        print(f"MODEL: {betaH_mode}")
        print("-" * 80)

        # =====================================================================
        # 2a. Load posterior
        # =====================================================================

        saved = load_final_result(betaH_mode)
        param_names = list(saved["param_names"])

        # Model-specific deterministic seed for equal-weight resampling.
        resample_seed = RANDOM_SEED + 100 * model_index

        posterior_samples = get_equal_weight_posterior(saved, seed=resample_seed)

        # Sanity check for the final all-climate result.
        if betaH_mode == "all_climate":
            loaded_logz = float(saved["results"].logz[-1])
            expected_logz = -813.629

            if abs(loaded_logz - expected_logz) > 1.0:
                print(
                    "*** WARNING: all_climate logZ="
                    f"{loaded_logz:.3f}, expected approximately "
                    f"{expected_logz:.3f}. Check FINAL_NESTED_DIR. ***"
                )
            else:
                print(
                    f"[OK] all_climate logZ={loaded_logz:.3f} "
                    "matches the verified final_v2 result."
                )

        # =====================================================================
        # 2b. Select exact posterior draws reproducibly
        # =====================================================================

        n_available = len(posterior_samples)
        n_draws = min(N_POSTERIOR_DRAWS, n_available)

        subsample_seed = RANDOM_SEED + 10000 + model_index
        subsample_rng = np.random.default_rng(subsample_seed)

        draw_idx = subsample_rng.choice(n_available, size=n_draws, replace=False)
        draws = posterior_samples[draw_idx]

        print(f"Equal-weight posterior samples available: {n_available}")
        print(f"Posterior draws used: {n_draws}")

        # =====================================================================
        # 2c. Save exact parameter vectors used
        # =====================================================================

        selected_draws_df = pd.DataFrame(draws, columns=param_names)
        selected_draws_df.insert(0, "equal_weight_sample_index", draw_idx)
        selected_draws_df.insert(0, "posterior_draw", np.arange(1, n_draws + 1))

        selected_draws_path = (
            OUTPUT_DIR / f"selected_transmission_dynamics_draws_{betaH_mode}.csv"
        )
        selected_draws_df.to_csv(selected_draws_path, index=False)

        print(f"[OK] Saved exact selected posterior draws:\n  {selected_draws_path}")

        # =====================================================================
        # 2d. Allocate trajectory arrays
        # =====================================================================

        gamma_idx = param_names.index("gamma")

        spillover_draws = np.full((n_draws, n_weeks), np.nan, dtype=float)
        amplification_draws = np.full((n_draws, n_weeks), np.nan, dtype=float)
        total_draws = np.full((n_draws, n_weeks), np.nan, dtype=float)
        r0_draws = np.full((n_draws, n_weeks), np.nan, dtype=float)

        # =====================================================================
        # 2e. Simulate posterior draws
        # =====================================================================

        for i, theta in enumerate(draws):

            sim = simulate_one_draw(
                theta=theta,
                param_names=param_names,
                betaH_mode=betaH_mode,
                zT=zT,
                ze=ze,
                zR=zR,
                start_date=start_date,
            )

            spillover_draws[i, :] = weekly_mean_trajectory(sim, week_ending, "beta_R")
            amplification_draws[i, :] = weekly_mean_trajectory(
                sim, week_ending, "amplification_force"
            )
            total_draws[i, :] = weekly_mean_trajectory(sim, week_ending, "lambda")

            gamma = float(theta[gamma_idx])
            sim["R0_H"] = sim["beta_H"] / gamma

            r0_draws[i, :] = weekly_mean_trajectory(sim, week_ending, "R0_H")

            if (i + 1) % 50 == 0 or (i + 1) == n_draws:
                print(f"  processed {i + 1} / {n_draws}")

        # =====================================================================
        # 2f. Validate trajectories
        #
        # All four quantities are non-negative by construction (see the
        # module docstring's Validation section), so check_nonnegative=True
        # for all of them, not just amplification_draws.
        # =====================================================================

        validate_array("spillover_draws", spillover_draws, check_nonnegative=True)
        validate_array("amplification_draws", amplification_draws, check_nonnegative=True)
        validate_array("total_draws", total_draws, check_nonnegative=True)
        validate_array("r0_draws", r0_draws, check_nonnegative=True)

        # =====================================================================
        # 2g. Posterior weekly summaries
        # =====================================================================

        spill_lo, spill_med, spill_hi = posterior_quantiles(spillover_draws)
        amp_lo, amp_med, amp_hi = posterior_quantiles(amplification_draws)
        total_lo, total_med, total_hi = posterior_quantiles(total_draws)
        r0_lo, r0_med, r0_hi = posterior_quantiles(r0_draws)

        # =====================================================================
        # 2h. Save weekly posterior summary
        # =====================================================================

        model_df = pd.DataFrame({
            "week_ending": week_ending,
            "observed_deaths": observed,

            # Reservoir-associated spillover
            "spillover_q025": spill_lo,
            "spillover_median": spill_med,
            "spillover_q975": spill_hi,

            # Human-associated amplification
            # (kept as "human_*" column names for downstream compatibility)
            "human_q025": amp_lo,
            "human_median": amp_med,
            "human_q975": amp_hi,

            # Total force of infection
            "total_q025": total_lo,
            "total_median": total_med,
            "total_q975": total_hi,

            # Amplification reproduction number
            "R0_H_q025": r0_lo,
            "R0_H_median": r0_med,
            "R0_H_q975": r0_hi,
        })

        csv_path = OUTPUT_DIR / f"posterior_transmission_dynamics_{betaH_mode}.csv"
        model_df.to_csv(csv_path, index=False)

        print(f"\n[OK] Saved weekly posterior transmission dynamics:\n  {csv_path}")

        all_model_summaries[betaH_mode] = model_df

        # =====================================================================
        # 2i. R0_H diagnostics
        # =====================================================================

        r0_flat = r0_draws[np.isfinite(r0_draws)]

        if r0_flat.size:
            prop_above_1 = float(np.mean(r0_flat > 1.0))
        else:
            prop_above_1 = np.nan

        r0_summary_rows.append({
            "betaH_mode": betaH_mode,
            "R0_H_posterior_mean": float(np.mean(r0_flat)),
            "R0_H_posterior_median": float(np.median(r0_flat)),
            "R0_H_q025": float(np.quantile(r0_flat, 0.025)),
            "R0_H_q975": float(np.quantile(r0_flat, 0.975)),
            "R0_H_min": float(np.min(r0_flat)),
            "R0_H_max": float(np.max(r0_flat)),
            "pct_draw_week_combinations_R0_H_above_1": 100.0 * prop_above_1,
        })

        print(
            f"  R0_H pooled draw-week summary: "
            f"median={np.median(r0_flat):.3f}, "
            f"95% interval=({np.quantile(r0_flat, 0.025):.3f}, "
            f"{np.quantile(r0_flat, 0.975):.3f}), "
            f"{100.0 * prop_above_1:.1f}% of draw-week combinations exceed 1"
        )

    # =========================================================================
    # 3. Save R0_H summary
    # =========================================================================

    r0_summary_df = pd.DataFrame(r0_summary_rows)
    r0_csv = OUTPUT_DIR / "posterior_human_R0_summary.csv"
    r0_summary_df.to_csv(r0_csv, index=False)

    print("\n" + "=" * 100)
    print("R0_H SUMMARY — ALL MODELS")
    print("=" * 100)
    print(r0_summary_df.to_string(index=False))
    print(f"\n[OK] Saved:\n  {r0_csv}")

    # =========================================================================
    # 4. Figure 1: 2x2 posterior decomposition grid
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ax, betaH_mode in zip(axes.flat, MODELS):

        df = all_model_summaries[betaH_mode]

        # Reservoir-associated spillover
        ax.fill_between(
            df["week_ending"], df["spillover_q025"], df["spillover_q975"],
            color=COLOR_SPILLOVER, alpha=0.25, linewidth=0,
        )
        ax.plot(
            df["week_ending"], df["spillover_median"],
            color=COLOR_SPILLOVER, linewidth=1.6,
            label=r"Reservoir-associated spillover $\beta_R(t)$",
        )

        # Human-associated amplification
        ax.fill_between(
            df["week_ending"], df["human_q025"], df["human_q975"],
            color=COLOR_AMPLIFICATION, alpha=0.25, linewidth=0,
        )
        ax.plot(
            df["week_ending"], df["human_median"],
            color=COLOR_AMPLIFICATION, linewidth=1.6,
            label=r"Human-associated amplification $\beta_H(t)I(t)/N$",
        )

        # Total FOI
        ax.plot(
            df["week_ending"], df["total_median"],
            color=COLOR_TOTAL, linewidth=1.4, linestyle="--",
            label=r"Total $\lambda(t)$",
        )

        ax.set_title(MODEL_LABELS[betaH_mode])
        ax.set_ylabel(r"Force of infection (week$^{-1}$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout()

    grid_fig_path = OUTPUT_DIR / "posterior_transmission_dynamics_grid.png"
    grid_fig_path_pdf = OUTPUT_DIR / "posterior_transmission_dynamics_grid.pdf"

    fig.savefig(grid_fig_path, dpi=600, bbox_inches="tight")
    fig.savefig(grid_fig_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved grid figure:\n  {grid_fig_path}")
    print(f"[OK] Saved grid figure:\n  {grid_fig_path_pdf}")

    # =========================================================================
    # 5. Figure 2: detailed all-climate posterior decomposition
    # =========================================================================

    df = all_model_summaries["all_climate"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Reservoir-associated spillover
    ax1.fill_between(
        df["week_ending"], df["spillover_q025"], df["spillover_q975"],
        color=COLOR_SPILLOVER, alpha=0.25, linewidth=0,
    )
    ax1.plot(
        df["week_ending"], df["spillover_median"],
        color=COLOR_SPILLOVER, linewidth=2.0,
        label=r"Reservoir-associated spillover $\beta_R(t)$ (95% CrI)",
    )

    # Human-associated amplification
    ax1.fill_between(
        df["week_ending"], df["human_q025"], df["human_q975"],
        color=COLOR_AMPLIFICATION, alpha=0.25, linewidth=0,
    )
    ax1.plot(
        df["week_ending"], df["human_median"],
        color=COLOR_AMPLIFICATION, linewidth=2.0,
        label=r"Human-associated amplification $\beta_H(t)I(t)/N$ (95% CrI)",
    )

    # Total force of infection
    ax1.plot(
        df["week_ending"], df["total_median"],
        color=COLOR_TOTAL, linewidth=1.6, linestyle="--",
        label=r"Total $\lambda(t)$",
    )

    ax1.set_ylabel(r"Force of infection (week$^{-1}$)")
    ax1.set_xlabel("Week ending")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(False)

    # Observed mortality on secondary axis
    ax2 = ax1.twinx()
    ax2.scatter(
        df["week_ending"], df["observed_deaths"],
        s=10, color="black", alpha=0.35, zorder=1,
        label="Observed deaths (right axis)",
    )
    ax2.set_ylabel("Observed weekly plague deaths")
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    ax1.set_title(
        "Posterior decomposition of the force of infection "
        "— climate-driven amplification model"
    )

    fig.tight_layout()

    detail_fig_path = OUTPUT_DIR / "posterior_transmission_dynamics_all_climate.png"
    detail_fig_path_pdf = OUTPUT_DIR / "posterior_transmission_dynamics_all_climate.pdf"

    fig.savefig(detail_fig_path, dpi=600, bbox_inches="tight")
    fig.savefig(detail_fig_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Saved all-climate detail figure:\n  {detail_fig_path}")
    print(f"[OK] Saved all-climate detail figure:\n  {detail_fig_path_pdf}")

    # =========================================================================
    # Final report
    # =========================================================================

    print("\n" + "=" * 100)
    print("POSTERIOR TRANSMISSION-DYNAMICS DECOMPOSITION COMPLETE")
    print("=" * 100)
    print(f"\nAll outputs saved in:\n  {OUTPUT_DIR}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()