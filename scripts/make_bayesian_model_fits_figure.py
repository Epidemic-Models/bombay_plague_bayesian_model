# =============================================================================
# make_bayesian_model_fits_figure.py
# =============================================================================
"""
Bayesian posterior model-fit figures for the seven beta_H formulations.

This script reads the final nested-sampling posteriors and does NOT refit
the model. For each beta_H formulation it calculates:

    1. Posterior expected-death trajectory:
       - reproducibly resample dynesty's weighted posterior;
       - reproducibly select N_POSTERIOR_DRAWS parameter vectors;
       - simulate the expected-death trajectory for each draw;
       - calculate the weekly posterior median and 95% credible interval.

    2. Maximum-likelihood (MLE) trajectory:
       - use the sampled theta with the largest stored log-likelihood.

    3. Posterior-median-parameter trajectory (diagnostic only, saved to CSV,
       not plotted).

Figures (FIGURE_SETS)
---------------------
1. bayesian_model_fits_grid            main-text Fig. 1: six formulations
                                       (2x3), without the temperature-peak model
2. bayesian_model_fits_climate_models  SI figure: the two climate formulations
                                       (linear temperature, temperature peak)
                                       side by side

Each panel shows recorded deaths, the posterior median expected trajectory
and its 95% credible band, ordered by log evidence. Panel titles give the
formulation name and Delta log Z relative to the climate formulation with
linear temperature (REFERENCE_MODEL), matching main-text Table 1; negative
values mean higher evidence. All panels share the y-axis so
peak heights can be compared directly. Optional: the MLE trajectory
(SHOW_MLE) and dotted reference lines at the recorded wave peaks
(SHOW_PEAK_LINES).

IMPORTANT
---------
The 95% band is uncertainty in the latent expected-death trajectory mu(t).
It is NOT the full posterior-predictive interval, which would additionally
include negative-binomial observation noise and would therefore be wider.

Outputs
-------
results/final_figures/bayesian_model_fits/

    bayesian_model_fits_grid.png / .pdf
    bayesian_model_fits_climate_models.png / .pdf
    bayesian_model_fits_<model>.csv
    selected_model_fit_draws_<model>.csv
    mle_vs_median_divergence.csv
    model_fit_summary.csv          (log Z, Delta log Z, max logL, peak heights)

Usage
-----
    python scripts/make_bayesian_model_fits_figure.py
    python scripts/make_bayesian_model_fits_figure.py --replot

--replot rebuilds the figures from the saved CSVs of a previous run, without
re-simulating any trajectory. Use it for cosmetic changes (labels, colours,
panel layout). Drop the flag whenever the posteriors, the simulation settings
or the posterior-draw settings change.
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
from mechanistic_model import simulate_seir_dde, weekly_sum  # noqa: E402
from observation import make_delay_pmf_weeks, deaths_from_infections_fast  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
RAW_TARGET = "plague_deaths_reported"

OUTLIER_WEEK = "1897-02-19"

RESULTS_ROOT = PROJECT_ROOT / "results"

# -----------------------------------------------------------------------------
# One entry per formulation: where its posterior lives and how to label it.
# The panel order is NOT taken from this list; panels are sorted by log Z.
# Keep new formulations at the END of this dict: the random seeds for the
# posterior draws are keyed on position, so appending keeps the existing
# figures exactly reproducible.
# -----------------------------------------------------------------------------

MODELS = {
    "all_climate": {
        "label": "Climate, linear temperature",
        "path": RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"
                / f"all_climate__{FIT_TARGET}__final_v2.pkl",
    },
    "humidity_only": {
        "label": "Vapour pressure only",
        "path": RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"
                / f"humidity_only__{FIT_TARGET}__final_v2.pkl",
    },
    "rainfall_only": {
        "label": "Rainfall only",
        "path": RESULTS_ROOT / "rainfall_only_check" / "nested_sampling"
                / f"rainfall_only__{FIT_TARGET}__seeded.pkl",
    },
    "temperature_only": {
        "label": "Temperature only",
        "path": RESULTS_ROOT / "temperature_only_check" / "nested_sampling"
                / f"temperature_only__{FIT_TARGET}__seeded.pkl",
    },
    "constant": {
        "label": "Constant",
        "path": RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"
                / f"constant__{FIT_TARGET}__final_v2.pkl",
    },
    "sigmoid": {
        "label": "Single shift",
        "path": RESULTS_ROOT / "final_model_comparison_v2" / "nested_sampling"
                / f"sigmoid__{FIT_TARGET}__final_v2.pkl",
    },
    "all_climate_quadT": {
        "label": "Climate, temperature peak",
        "path": RESULTS_ROOT / "all_climate_quadT_check" / "nested_sampling"
                / f"all_climate_quadT__{FIT_TARGET}__seeded.pkl",
    },
}

# Delta log Z in panel titles and in the summary is measured from this model.
REFERENCE_MODEL = "all_climate"

# Which formulations go in which figure, and how many columns.
FIGURE_SETS = {
    "bayesian_model_fits_grid": {
        "models": ["all_climate", "humidity_only", "rainfall_only",
                   "temperature_only", "constant", "sigmoid"],
        "n_cols": 3,
    },
    "bayesian_model_fits_climate_models": {
        "models": ["all_climate", "all_climate_quadT"],
        "n_cols": 2,
    },
}

# If True, formulations whose posterior file is missing are skipped with a
# warning (useful while a run is still pending). Keep False for the final figure.
SKIP_MISSING_MODELS = False

# Rebuild the figures from the saved CSVs instead of re-simulating.
REPLOT = "--replot" in sys.argv

# Plot options
SHOW_MLE = False          # thin MLE line in every panel
SHOW_PEAK_LINES = True    # dotted lines at the three recorded wave peaks


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


# -----------------------------------------------------------------------------
# Infection-to-death delay settings
# -----------------------------------------------------------------------------

DELAY_KIND = "gamma"
DELAY_MAX_WEEKS = 12
DELAY_SHAPE = 2.0
DELAY_SCALE = 0.6


# -----------------------------------------------------------------------------
# Wave windows used for peak lines and the peak table
# -----------------------------------------------------------------------------

WAVE_WINDOWS = [
    ("Wave 1", "1896-08-01", "1897-07-31"),
    ("Wave 2", "1897-08-01", "1898-07-31"),
    ("Wave 3", "1898-08-01", "1899-12-31"),
]


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = RESULTS_ROOT / "final_figures" / "bayesian_model_fits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Plot styling
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLOR_OBSERVED = "#1f4e8c"
COLOR_MEDIAN = "#E08214"
COLOR_MLE = "#2C7C2C"
COLOR_OUTLIER = "#C0392B"
COLOR_PEAK = "#7f7f7f"

# Legend labels. Keep these identical to the wording of the figure captions.
LABEL_RECORDED = "Recorded deaths"
LABEL_MEDIAN = "Posterior median"
LABEL_BAND = "95% credible interval"
LABEL_MLE = "Maximum-likelihood fit"
LABEL_OUTLIER = "Originally reported value"
LABEL_PEAKS = "Recorded wave peaks"


# =============================================================================
# Helper functions
# =============================================================================

def load_result(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_result_array(results, name: str) -> np.ndarray:
    """Read an array from dynesty Results (attribute or dict-like)."""
    if hasattr(results, name):
        return np.asarray(getattr(results, name))
    return np.asarray(results[name])


def get_equal_weight_posterior(samples, logwt, logz, seed: int) -> np.ndarray:
    """Convert dynesty's weighted posterior into an equal-weight posterior."""
    weights = np.exp(logwt - logz[-1])
    rstate = np.random.default_rng(seed)
    return np.asarray(resample_equal(samples, weights, rstate=rstate), dtype=float)


def compute_mu_trajectory(theta, param_names, betaH_mode, merged_df,
                          zT, ze, zR, start_date, f_delay) -> np.ndarray:
    """
    Simulate one parameter vector and return the expected weekly
    plague-death trajectory mu(t).
    """
    sample = dict(zip(param_names, np.asarray(theta, dtype=float)))
    sim_params = build_sim_params(sample, betaH_mode)

    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR,
        N=sample["N0"],
        params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=betaH_mode,
        start_date=start_date,
    )

    weekly_inf = weekly_sum(sim, col="C", week_ending=WEEK_RULE)
    weekly_inf["week_ending"] = pd.to_datetime(weekly_inf["week_ending"])

    compare = pd.merge(merged_df[["week_ending"]], weekly_inf,
                       on="week_ending", how="left")

    if "C_week" in compare.columns:
        c_col = "C_week"
    else:
        candidates = [c for c in compare.columns if c.startswith("C_")]
        if not candidates:
            raise RuntimeError("Could not identify weekly infection-count column.")
        c_col = candidates[0]

    C_week = compare[c_col].to_numpy(dtype=float)

    if np.isnan(C_week).any():
        bad = np.where(np.isnan(C_week))[0]
        raise RuntimeError(f"Missing fitted weeks in infection trajectory: {bad.tolist()}")
    if not np.isfinite(C_week).all():
        raise RuntimeError("Weekly infection trajectory contains non-finite values.")
    if (C_week < -1e-10).any():
        raise RuntimeError("Weekly infection trajectory contains negative values.")

    mu = np.asarray(
        deaths_from_infections_fast(C_week=C_week, psi=sample["psi"], f_delay=f_delay),
        dtype=float,
    )

    if len(mu) != len(merged_df):
        raise RuntimeError(f"Expected-death trajectory has length {len(mu)}, "
                           f"expected {len(merged_df)}.")
    if not np.isfinite(mu).all():
        raise RuntimeError("Expected-death trajectory contains non-finite values.")

    return np.maximum(mu, 1e-9)


def sse(observed, predicted) -> float:
    """Sum of squared errors, a descriptive diagnostic only."""
    return float(np.sum((observed - predicted) ** 2))


def wave_peaks(week_ending: pd.Series, values: np.ndarray) -> list[tuple]:
    """Return (wave, peak value, peak date) for each wave window."""
    out = []
    for name, lo, hi in WAVE_WINDOWS:
        mask = ((week_ending >= pd.Timestamp(lo)) &
                (week_ending <= pd.Timestamp(hi))).to_numpy()
        if not mask.any():
            continue
        idx = np.where(mask)[0][np.argmax(values[mask])]
        out.append((name, float(values[idx]), week_ending.iloc[idx]))
    return out


# =============================================================================
# Main analysis
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Data
    # -------------------------------------------------------------------------
    if not paths.MERGED_CSV.exists():
        raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

    merged_df = (pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
                 .sort_values(paths.DATE_COL).reset_index(drop=True))

    required = [paths.DATE_COL, "zT", "ze", "zR", FIT_TARGET]
    missing = [c for c in required if c not in merged_df.columns]
    if missing:
        raise ValueError("Prepared data are missing columns: " + ", ".join(missing))

    week_ending = pd.to_datetime(merged_df[paths.DATE_COL])
    zT = merged_df["zT"].to_numpy(dtype=float)
    ze = merged_df["ze"].to_numpy(dtype=float)
    zR = merged_df["zR"].to_numpy(dtype=float)
    start_date = str(pd.Timestamp(week_ending.iloc[0]).date())
    n_weeks = len(merged_df)

    observed_fit = merged_df[FIT_TARGET].to_numpy(dtype=float)
    observed_raw = (merged_df[RAW_TARGET].to_numpy(dtype=float)
                    if RAW_TARGET in merged_df.columns else observed_fit.copy())

    outlier_mask = (week_ending == pd.Timestamp(OUTLIER_WEEK)).to_numpy()
    if not outlier_mask.any():
        print(f"[WARNING] Outlier week {OUTLIER_WEEK} not found in prepared data.")

    f_delay = make_delay_pmf_weeks(kind=DELAY_KIND, max_weeks=DELAY_MAX_WEEKS,
                                   shape=DELAY_SHAPE, scale=DELAY_SCALE)

    print("\n" + "=" * 90)
    print("BAYESIAN POSTERIOR MODEL-FIT TRAJECTORIES")
    if REPLOT:
        print("[--replot] Rebuilding figures from saved CSVs; nothing is re-simulated.")
    print("=" * 90)
    print(f"\nFitted weeks: {n_weeks}")
    print(f"Study period: {week_ending.min().date()} to {week_ending.max().date()}")
    print(f"Posterior draws per model: {N_POSTERIOR_DRAWS}")

    # -------------------------------------------------------------------------
    # 2. Check which posteriors exist
    # -------------------------------------------------------------------------
    needed = {m for fs in FIGURE_SETS.values() for m in fs["models"]} | {REFERENCE_MODEL}
    available = []
    for model, spec in MODELS.items():
        if model not in needed:
            continue
        if REPLOT or spec["path"].exists():
            available.append(model)
        elif SKIP_MISSING_MODELS:
            print(f"[WARNING] Skipping {model!r}: posterior not found at\n  {spec['path']}")
        else:
            raise FileNotFoundError(
                f"Posterior for {model!r} not found:\n  {spec['path']}\n"
                "Edit its path in MODELS, or set SKIP_MISSING_MODELS = True "
                "to draw the figure without it."
            )

    # -------------------------------------------------------------------------
    # 3. Model loop
    # -------------------------------------------------------------------------
    summaries: dict[str, pd.DataFrame] = {}
    logz_final: dict[str, float] = {}
    divergence_rows = []
    summary_rows = []

    prev_summary = None
    if REPLOT:
        prev_path = OUTPUT_DIR / "model_fit_summary.csv"
        if not prev_path.exists():
            raise FileNotFoundError(
                f"--replot needs the tables of a previous run:\n  {prev_path}\n"
                "Run the script once without --replot first."
            )
        prev_summary = pd.read_csv(prev_path).set_index("betaH_mode")

    for model_index, model in enumerate(available):
        print(f"\n--- {model} ---")

        if REPLOT:
            csv_path = OUTPUT_DIR / f"bayesian_model_fits_{model}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"--replot needs saved trajectories:\n  {csv_path}")
            if model not in prev_summary.index:
                raise KeyError(f"{model!r} is missing from model_fit_summary.csv; "
                               "rerun once without --replot.")
            summaries[model] = pd.read_csv(csv_path, parse_dates=["week_ending"])
            logz_final[model] = float(prev_summary.loc[model, "logZ"])
            print(f"  reused saved trajectories (log Z = {logz_final[model]:.2f})")
            continue

        saved = load_result(MODELS[model]["path"])
        results = saved["results"]
        param_names = list(saved["param_names"])

        logwt = get_result_array(results, "logwt")
        logz = get_result_array(results, "logz")
        logl = get_result_array(results, "logl")
        samples = get_result_array(results, "samples")
        logz_final[model] = float(logz[-1])

        # Equal-weight posterior and reproducible subsample. Seeds are keyed
        # on the model name so they do not change when models are added.
        model_key = list(MODELS).index(model)
        posterior = get_equal_weight_posterior(samples, logwt, logz,
                                               seed=RANDOM_SEED + 100 * model_key)
        n_available = len(posterior)
        n_draws = min(N_POSTERIOR_DRAWS, n_available)
        rng = np.random.default_rng(RANDOM_SEED + 10000 + model_key)
        draw_idx = rng.choice(n_available, size=n_draws, replace=False)
        draws = posterior[draw_idx]

        print(f"  log Z = {logz_final[model]:.2f}")
        print(f"  Equal-weight samples: {n_available}; draws used: {n_draws}")

        draws_df = pd.DataFrame(draws, columns=param_names)
        draws_df.insert(0, "equal_weight_sample_index", draw_idx)
        draws_df.insert(0, "posterior_draw", np.arange(1, n_draws + 1))
        draws_df.to_csv(OUTPUT_DIR / f"selected_model_fit_draws_{model}.csv", index=False)

        # Posterior expected-death trajectories
        mu_draws = np.full((n_draws, n_weeks), np.nan)
        for i, theta in enumerate(draws):
            mu_draws[i, :] = compute_mu_trajectory(
                theta, param_names, model, merged_df, zT, ze, zR, start_date, f_delay)
            if (i + 1) % 100 == 0 or (i + 1) == n_draws:
                print(f"  processed {i + 1} / {n_draws}")

        if not np.isfinite(mu_draws).all():
            raise RuntimeError(f"{model}: posterior trajectories contain non-finite values.")

        mu_q025, mu_q50, mu_q975 = np.quantile(mu_draws, [0.025, 0.5, 0.975], axis=0)

        # MLE trajectory
        best_idx = int(np.nanargmax(logl))
        theta_mle = np.asarray(samples[best_idx], dtype=float)
        max_loglik = float(logl[best_idx])
        mu_mle = compute_mu_trajectory(
            theta_mle, param_names, model, merged_df, zT, ze, zR, start_date, f_delay)

        # Posterior-median-parameter diagnostic (saved, not plotted)
        theta_med = np.median(posterior, axis=0)
        mu_med_params = compute_mu_trajectory(
            theta_med, param_names, model, merged_df, zT, ze, zR, start_date, f_delay)

        mle_sse = sse(observed_fit, mu_mle)
        med_sse = sse(observed_fit, mu_med_params)
        ratio = med_sse / mle_sse if mle_sse > 0 else np.nan
        print(f"  MLE: logL={max_loglik:.3f}, SSE={mle_sse:.1f}; "
              f"median-parameter SSE ratio={ratio:.2f}")

        divergence_rows.append({
            "betaH_mode": model, "max_loglik": max_loglik, "mle_sse": mle_sse,
            "median_param_sse": med_sse, "median_over_mle_sse_ratio": ratio,
        })

        model_df = pd.DataFrame({
            "week_ending": week_ending,
            "observed_fit": observed_fit,
            "observed_raw": observed_raw,
            "mu_q025": mu_q025,
            "mu_band_median": mu_q50,
            "mu_q975": mu_q975,
            "mu_mle": mu_mle,
            "mu_median_params": mu_med_params,
        })
        model_df.to_csv(OUTPUT_DIR / f"bayesian_model_fits_{model}.csv", index=False)
        summaries[model] = model_df

        row = {"betaH_mode": model, "label": MODELS[model]["label"],
               "logZ": logz_final[model], "max_loglik": max_loglik}
        for name, val, date in wave_peaks(week_ending, mu_q50):
            row[f"{name} median peak"] = round(val)
            row[f"{name} median peak date"] = date.date()
        for name, val, date in wave_peaks(week_ending, mu_mle):
            row[f"{name} MLE peak"] = round(val)
        summary_rows.append(row)

    # -------------------------------------------------------------------------
    # 4. Order by log Z and save tables
    # -------------------------------------------------------------------------
    if REFERENCE_MODEL not in logz_final:
        raise RuntimeError(f"The reference model {REFERENCE_MODEL!r} is needed.")
    ref_logz = logz_final[REFERENCE_MODEL]

    if REPLOT:
        summary_df = prev_summary.reset_index()
        print("\n[--replot] Saved tables were reused and not rewritten.")
    else:
        summary_df = pd.DataFrame(summary_rows)
        summary_df["delta_logZ_vs_reference"] = ref_logz - summary_df["logZ"]
        summary_df = summary_df.sort_values("logZ", ascending=False)
        obs_row = {"betaH_mode": "observed", "label": "Recorded"}
        for name, val, date in wave_peaks(week_ending, observed_fit):
            obs_row[f"{name} median peak"] = round(val)
            obs_row[f"{name} median peak date"] = date.date()
        summary_df = pd.concat([pd.DataFrame([obs_row]), summary_df], ignore_index=True)
        summary_df.to_csv(OUTPUT_DIR / "model_fit_summary.csv", index=False)

        divergence_df = pd.DataFrame(divergence_rows)
        divergence_df.to_csv(OUTPUT_DIR / "mle_vs_median_divergence.csv", index=False)

        print("\n" + "=" * 90)
        print("MODEL FIT SUMMARY (ordered by log Z)")
        print("=" * 90)
        print(summary_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Figures: one per entry in FIGURE_SETS, panels ordered by log Z
    # -------------------------------------------------------------------------
    obs_peaks = wave_peaks(week_ending, observed_fit)
    # Science requires capital panel letters on multi-panel figures.
    panel_letters = "ABCDEFGHIJKL"

    for fig_name, fs in FIGURE_SETS.items():
        order = sorted([m for m in fs["models"] if m in summaries],
                       key=lambda m: logz_final[m], reverse=True)
        if not order:
            print(f"[WARNING] {fig_name}: no formulations available, skipped.")
            continue
        n_cols = min(fs["n_cols"], len(order))
        n_panels = len(order)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)

        for k, (ax, model) in enumerate(zip(axes.flat, order)):
            df = summaries[model]

            if SHOW_PEAK_LINES:
                for _, val, _ in obs_peaks:
                    ax.axhline(val, color=COLOR_PEAK, linewidth=0.6, linestyle=":",
                               zorder=0)

            ax.fill_between(df["week_ending"], df["mu_q025"], df["mu_q975"],
                            color=COLOR_MEDIAN, alpha=0.25, linewidth=0,
                            label=LABEL_BAND)
            ax.plot(df["week_ending"], df["observed_fit"], marker="o", markersize=2.2,
                    color=COLOR_OBSERVED, linewidth=0.9, label=LABEL_RECORDED)
            ax.plot(df["week_ending"], df["mu_band_median"], color=COLOR_MEDIAN,
                    linewidth=1.6, label=LABEL_MEDIAN)
            if SHOW_MLE:
                ax.plot(df["week_ending"], df["mu_mle"], color=COLOR_MLE,
                        linewidth=1.0, label=LABEL_MLE)
            if outlier_mask.any():
                ax.scatter(week_ending[outlier_mask], observed_raw[outlier_mask],
                           color=COLOR_OUTLIER, s=18, zorder=5,
                           label=LABEL_OUTLIER)

            if model == REFERENCE_MODEL:
                delta_txt = "reference"
            else:
                delta = ref_logz - logz_final[model]
                delta_txt = f"$\\Delta\\log Z$ = {delta:+.1f}" if delta < 0 \
                    else f"$\\Delta\\log Z$ = {delta:.1f}"
            ax.set_title(f"({panel_letters[k]}) {MODELS[model]['label']} ({delta_txt})",
                         loc="left")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(axis="x", rotation=0)
            ax.grid(False)

        for ax in axes[:, 0]:
            ax.set_ylabel("Weekly plague deaths")
        for ax in axes.flat[n_panels:]:
            ax.set_visible(False)
        # A panel with an empty slot below it needs its own date labels.
        for k in range(n_panels):
            if k + n_cols >= n_panels:
                axes.flat[k].tick_params(axis="x", labelbottom=True)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if SHOW_PEAK_LINES:
            handles.append(plt.Line2D([], [], color=COLOR_PEAK, linewidth=0.6, linestyle=":"))
            labels.append(LABEL_PEAKS)
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(labels), 3 if n_cols < 3 else len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 1.03 if n_rows > 1 else 1.12))

        fig.tight_layout()

        for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
            out = OUTPUT_DIR / f"{fig_name}.{ext}"
            fig.savefig(out, bbox_inches="tight", **kw)
            print(f"[OK] Saved figure: {out}")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Peak comparison printed for the text
    # -------------------------------------------------------------------------
    print("\nPeak heights (posterior median / MLE) vs recorded:")
    obs = {name: val for name, val, _ in obs_peaks}
    for _, r in summary_df.iterrows():
        if r["betaH_mode"] == "observed":
            continue
        parts = []
        for w in ("Wave 1", "Wave 2", "Wave 3"):
            parts.append(f"{w}: {r[f'{w} median peak']:.0f}/{r[f'{w} MLE peak']:.0f} "
                         f"(recorded {obs[w]:.0f})")
        print(f"  {r['label']:<27} " + "; ".join(parts))

    print("\nDone.")


if __name__ == "__main__":
    main()