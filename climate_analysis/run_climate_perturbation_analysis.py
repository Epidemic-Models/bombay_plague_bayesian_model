# =============================================================================
# run_climate_perturbation_analysis.py
# =============================================================================
"""
Posterior climate-perturbation (scenario) analysis for the best-supported
all-climate model.

*** REPLACES the old LHS-based scenario table, which had two problems:
    1. It used a single best-fit parameter set per scenario (no
       uncertainty quantification).
    2. It borrowed beta_R's response from the "constant" model and
       beta_H's response from "humidity-only"/"climate-driven"
       separately -- a patchwork across three different, now-superseded
       models, none of which is the model this paper's conclusions
       actually rest on.

This version reports BOTH beta_R's and beta_H's response to every
perturbation, computed from POSTERIOR DRAWS of the single best-supported
all_climate model, so every percentage comes with a credible interval
instead of being a single point estimate, and everything is internally
consistent within one model rather than mixed across four. ***

Method
------
Following the original design: climate covariates are perturbed in their
RAW (pre-standardization) units, then re-standardized using the ORIGINAL
mean/SD (computed from the full climate-covered period, not just the
fitted weeks -- see verification in the accompanying notes), so
perturbed simulations remain on the same parameter scale used for
fitting. This is equivalent to a simple additive shift in z-space:

    z_perturbed(t) = z_baseline(t) + delta_raw / SD(raw covariate)

Two sets of perturbations are run:

  1. Raw units (matching the original manuscript):
       Temperature:     +/- 2 degC  (+/- 3.6 degF, since the historical
                        series is recorded in Fahrenheit)
       Vapour pressure: +/- 0.05 inHg
       Rainfall:        +/- 2 inches
     These are round numbers in the units of the historical record, but
     they are NOT the same size in standard deviations (about 1.0, 0.4 and
     0.2 SD respectively).

  2. One standard deviation (new):
       Each covariate shifted by +/- 1 SD of its own weekly series, i.e.
       delta_z = +/- 1. This puts the three covariates on the same
       relative scale, so their effects can be compared directly.

For each perturbation, N_POSTERIOR_DRAWS posterior parameter vectors are
used to simulate BOTH the baseline and perturbed climate series, and the
percentage change in beta_R(t) and beta_H(t) is computed per draw (mean,
peak, and median-weekly change, matching the original table's three
summary statistics). Reported values are the posterior median and 95%
credible interval across draws.

Outputs
-------
results/final_figures/scenario_analysis/
    climate_perturbation_effects_posterior.csv        (raw units, unchanged)
    climate_perturbation_effects_posterior.tex        (raw units, unchanged)
    climate_perturbation_effects_posterior_1sd.csv    (+/- 1 SD, new)
    climate_perturbation_effects_posterior_1sd.tex    (+/- 1 SD, new)
    climate_perturbation_selected_posterior_draws.csv

Reproducibility note
---------------------
Two distinct stochastic steps are involved: (1) converting dynesty's
weighted posterior sample cloud into an equal-weight posterior via
resample_equal, and (2) subsampling N_POSTERIOR_DRAWS draws from that
equal-weight posterior. These use two separate, independently specified
random seeds (RANDOM_SEED_RESAMPLE and RANDOM_SEED_DRAWS) rather than a
single shared seed, so that changing one step's sampling behaviour in the
future cannot silently change the other's.

Performance note
-----------------
The baseline (unperturbed) beta_R/beta_H trajectory is identical across
all six perturbation scenarios for a given posterior draw, so it is
simulated once per draw and cached, rather than recomputed for every
scenario -- this roughly halves the total number of mechanistic
simulations run (1000 baseline sims + 1000 x 6 perturbed sims, instead
of 1000 x 6 baseline sims + 1000 x 6 perturbed sims).

Usage
-----
    python scripts/run_climate_perturbation_analysis.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dynesty.utils import resample_equal

# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Same path configuration as plot_climate_perturbation_effects.py

if SCRIPT_DIR.name == "climate_analysis":

    PROJECT_ROOT = SCRIPT_DIR.parent
    CLIMATE_DIR = SCRIPT_DIR

elif SCRIPT_DIR.name == "scripts":

    PROJECT_ROOT = SCRIPT_DIR.parent
    CLIMATE_DIR = PROJECT_ROOT / "climate_analysis"

else:

    PROJECT_ROOT = SCRIPT_DIR.parent
    CLIMATE_DIR = PROJECT_ROOT / "climate_analysis"


# =============================================================================
# 2. IMPORT PROJECT MODULES
# =============================================================================

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import paths  # noqa: E402
from climate import build_weekly_climate  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402
FIT_TARGET = "plague_deaths_reported_fit"
BETA_H_MODE = "all_climate"

RESULT_PKL = (
    PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
    / f"{BETA_H_MODE}__{FIT_TARGET}__final_v2.pkl"
)

# Raw daily weather file, needed to recompute the TRUE standardization
# mean/SD (over the full 202-week climate-covered period, not just the
# 141 fitted weeks -- these differ, see accompanying verification notes).
RAW_WEATHER_CSV = PROJECT_ROOT / "data" / "bombay_weather_daily_1896_1899.csv"

N_POSTERIOR_DRAWS = 1000

# Two separate, independently specified seeds for the two distinct
# stochastic steps below (previously a single RANDOM_SEED was reused for
# both, which works but conflates two independent sampling decisions).
RANDOM_SEED_RESAMPLE = 12345  # for converting weighted -> equal-weight posterior
RANDOM_SEED_DRAWS = 54321     # for subsampling N_POSTERIOR_DRAWS from that posterior

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = E0 = R0_INIT = 0.0
WEEK_RULE = "W-FRI"

# Perturbations in RAW units (degF for temperature, since the historical
# series was recorded in Fahrenheit; inHg for vapour pressure; inches for
# rainfall).
PERTURBATIONS_RAW = {
    "temperature": 3.6,     # +/- 2 degC
    "vapour_pressure": 0.05,
    "rainfall": 2.0,
}

COVARIATE_COLS = {"temperature": "zT", "vapour_pressure": "ze", "rainfall": "zR"}

# =============================================================================
# OUTPUT DIRECTORY — PRESERVE ORIGINAL RESULTS
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_figures"
    / "scenario_analysis_updated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_SELECTED_DRAWS_CSV = OUTPUT_DIR / "climate_perturbation_selected_posterior_draws.csv"

# Tolerance below which a baseline value is treated as "zero" for the
# purposes of a percentage-change denominator (avoids inf/nan from
# division by a value that is technically nonzero but numerically tiny).
ZERO_TOL = 1e-12


# =============================================================================
# Load prepared data (fitted weeks) and full climate series (for correct SD)
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

week_ending = merged_df[paths.DATE_COL]
zT_base = merged_df["zT"].to_numpy(dtype=float)
ze_base = merged_df["ze"].to_numpy(dtype=float)
zR_base = merged_df["zR"].to_numpy(dtype=float)
start_date = str(pd.Timestamp(week_ending.iloc[0]).date())
n_weeks = len(merged_df)

if not RAW_WEATHER_CSV.exists():
    raise FileNotFoundError(
        f"Raw daily weather file not found:\n{RAW_WEATHER_CSV}\n"
        "This is needed to recompute the TRUE standardization SD used "
        "during fitting (over the full climate-covered period, not just "
        "the fitted weeks)."
    )

full_climate = build_weekly_climate(
    daily_csv_path=str(RAW_WEATHER_CSV),
    date_col="date", temp_col="Temperature_F", vapour_col="VapourPressure_inHg",
    rain_col="Rain_in", T_win=7, e_win=7, R_win=28,
    ell_T=14, ell_e=14, ell_R=28, week_rule=WEEK_RULE, standardize=True,
)

sd_xT = full_climate["xT"].std(ddof=0)
sd_xe = full_climate["xe"].std(ddof=0)
sd_xR = full_climate["xR"].std(ddof=0)

print(f"Standardization SDs (full {len(full_climate)}-week climate series):")
print(f"  temperature: {sd_xT:.4f} degF")
print(f"  vapour pressure: {sd_xe:.4f} inHg")
print(f"  rainfall: {sd_xR:.4f} in")

# Convert raw perturbations to z-space shifts
DELTA_Z = {
    "temperature": PERTURBATIONS_RAW["temperature"] / sd_xT,
    "vapour_pressure": PERTURBATIONS_RAW["vapour_pressure"] / sd_xe,
    "rainfall": PERTURBATIONS_RAW["rainfall"] / sd_xR,
}
print(f"\nEquivalent z-space shifts: {DELTA_Z}")

# SD of each raw covariate, used to report what "1 SD" means in real units
SD_RAW = {"temperature": sd_xT, "vapour_pressure": sd_xe, "rainfall": sd_xR}

# Two scenario sets: the original raw-unit shifts, and +/- 1 SD for every
# covariate (delta_z = 1 by definition of the standardization).
SCENARIO_SETS = {
    "raw_units": DELTA_Z,
    "one_sd": {covariate: 1.0 for covariate in DELTA_Z},
}

print("\n+/- 1 SD in real units:")
print(f"  temperature:     {sd_xT:.3f} degF = {sd_xT / 1.8:.3f} degC")
print(f"  vapour pressure: {sd_xe:.4f} inHg = {sd_xe * 33.8639:.3f} hPa")
print(f"  rainfall:        {sd_xR:.3f} in = {sd_xR * 25.4:.1f} mm")


# =============================================================================
# Load posterior
# =============================================================================

if not RESULT_PKL.exists():
    raise FileNotFoundError(f"Could not find final_v2 result:\n{RESULT_PKL}")

with open(RESULT_PKL, "rb") as f:
    saved = pickle.load(f)

results = saved["results"]
param_names = list(saved["param_names"])
weights = np.exp(results.logwt - results.logz[-1])
resample_rstate = np.random.default_rng(RANDOM_SEED_RESAMPLE)
posterior = resample_equal(results.samples, weights, rstate=resample_rstate)

print(f"\nlog(Z) = {results.logz[-1]:.3f}  (expect ~-813.6 for the verified final_v2 result)")

rng = np.random.default_rng(RANDOM_SEED_DRAWS)
n_draws = min(N_POSTERIOR_DRAWS, len(posterior))
draw_idx = rng.choice(len(posterior), size=n_draws, replace=False)
draws = posterior[draw_idx]

# -----------------------------------------------------------------------------
# Save the exact selected posterior parameter vectors, for full
# traceability -- these are the same n_draws parameter vectors used for
# every perturbation scenario below.
# -----------------------------------------------------------------------------

selected_draws_df = pd.DataFrame(draws, columns=param_names)
selected_draws_df.insert(0, "equal_weight_sample_index", draw_idx)
selected_draws_df.insert(0, "posterior_draw", np.arange(1, n_draws + 1))
selected_draws_df.to_csv(OUT_SELECTED_DRAWS_CSV, index=False)
print(f"[OK] Saved exact posterior draws: {OUT_SELECTED_DRAWS_CSV}")


# =============================================================================
# Simulation helper -- returns weekly-mean beta_R(t) and beta_H(t)
# =============================================================================

def simulate_betas(theta, zT, ze, zR):
    sample = dict(zip(param_names, np.asarray(theta, dtype=float)))
    sim_params = build_sim_params(sample, BETA_H_MODE)

    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR, N=sample["N0"], params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK, tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT, C_seed=sample["C_seed"],
        betaH_mode=BETA_H_MODE, start_date=start_date,
    )
    sim["date"] = pd.to_datetime(sim["date"])

    def weekly_mean(col):
        wk = sim.set_index("date")[col].resample(WEEK_RULE).mean().reindex(week_ending)
        return wk.to_numpy(dtype=float)

    return weekly_mean("beta_R"), weekly_mean("beta_H")


def safe_pct_changes(comp_base: np.ndarray, comp_pert: np.ndarray):
    """
    Compute percentage-change summaries for one component (beta_R or
    beta_H) for one posterior draw.

    Returns
    -------
    dict or None
        Dictionary containing "mean", "peak", and "median_weekly".
        Returns None only if the trajectories themselves are invalid
        (wrong shape, non-finite) or if the mean/peak denominator is
        effectively zero.

    Notes
    -----
    For the median weekly percentage change, weeks whose baseline
    coefficient is numerically indistinguishable from zero are excluded
    from that statistic only -- they do NOT cause the entire posterior
    draw's mean and peak statistics to be discarded, since those have
    independent denominators (the trajectory mean and the baseline-peak
    value respectively) unaffected by a single near-zero week elsewhere
    in the 141-week trajectory.
    """

    comp_base = np.asarray(comp_base, dtype=float)
    comp_pert = np.asarray(comp_pert, dtype=float)

    if comp_base.shape != comp_pert.shape:
        return None
    if not (np.all(np.isfinite(comp_base)) and np.all(np.isfinite(comp_pert))):
        return None

    # -------------------------------------------------------------------
    # Mean percentage change
    # -------------------------------------------------------------------
    mean_base = np.mean(comp_base)
    mean_pert = np.mean(comp_pert)
    if not np.isfinite(mean_base) or abs(mean_base) < ZERO_TOL:
        return None
    mean_change = 100.0 * (mean_pert - mean_base) / mean_base

    # -------------------------------------------------------------------
    # Peak percentage change: perturbed value at the week where the
    # BASELINE coefficient reaches its maximum (change at the baseline
    # peak, not a comparison of two independently-selected maxima).
    # -------------------------------------------------------------------
    peak_idx = int(np.argmax(comp_base))
    peak_base = float(comp_base[peak_idx])
    peak_pert = float(comp_pert[peak_idx])
    if not np.isfinite(peak_base) or abs(peak_base) < ZERO_TOL:
        return None
    peak_change = 100.0 * (peak_pert - peak_base) / peak_base

    # -------------------------------------------------------------------
    # Median weekly percentage change: exclude only weeks whose baseline
    # is numerically indistinguishable from zero, not the whole draw.
    # -------------------------------------------------------------------
    valid_week = (
        np.isfinite(comp_base)
        & np.isfinite(comp_pert)
        & (np.abs(comp_base) >= ZERO_TOL)
    )
    if not np.any(valid_week):
        return None
    weekly_pct = (
        100.0 * (comp_pert[valid_week] - comp_base[valid_week]) / comp_base[valid_week]
    )
    if not np.all(np.isfinite(weekly_pct)):
        return None
    median_weekly_change = float(np.median(weekly_pct))

    return {
        "mean": float(mean_change),
        "peak": float(peak_change),
        "median_weekly": median_weekly_change,
    }


# =============================================================================
# Run all perturbation scenarios
# =============================================================================

rows = []

print("\n" + "=" * 100)
print(f"POSTERIOR CLIMATE PERTURBATION ANALYSIS ({BETA_H_MODE}, {n_draws} draws per scenario)")
print("=" * 100)

# -----------------------------------------------------------------------------
# Cache the baseline (unperturbed) beta_R/beta_H trajectory for every posterior
# draw once, since it is identical across all six perturbation scenarios
# below -- avoids recomputing it 6x per draw (1000 draws x 6 scenarios x 2
# simulations/scenario previously; caching the baseline halves that to
# 1000 baseline sims + 1000 x 6 perturbed sims).
# -----------------------------------------------------------------------------

print("\nCaching baseline (unperturbed) simulations for all posterior draws...")
baseline_cache = []
for i, theta in enumerate(draws):
    baseline_cache.append(simulate_betas(theta, zT_base, ze_base, zR_base))
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{n_draws}")
print(f"[OK] Cached {len(baseline_cache)} baseline simulations.")

for scenario_set, deltas in SCENARIO_SETS.items():
  for covariate, delta_z in deltas.items():
    for sign, sign_label in [(+1, "+"), (-1, "-")]:

        z_arrays = {"zT": zT_base.copy(), "ze": ze_base.copy(), "zR": zR_base.copy()}
        z_arrays[COVARIATE_COLS[covariate]] = z_arrays[COVARIATE_COLS[covariate]] + sign * delta_z

        print(f"\n--- [{scenario_set}] {covariate} {sign_label} (delta_z = {sign * delta_z:+.3f}) ---")

        pct_change_R = {"mean": [], "peak": [], "median_weekly": []}
        pct_change_H = {"mean": [], "peak": [], "median_weekly": []}
        n_excluded = 0

        for i, theta in enumerate(draws):
            bR_base, bH_base = baseline_cache[i]
            bR_pert, bH_pert = simulate_betas(theta, z_arrays["zT"], z_arrays["ze"], z_arrays["zR"])

            r_result = safe_pct_changes(bR_base, bR_pert)
            h_result = safe_pct_changes(bH_base, bH_pert)

            if r_result is None or h_result is None:
                n_excluded += 1
                warnings.warn(
                    f"[{scenario_set}] {covariate} {sign_label}: draw {draw_idx[i]} excluded "
                    "(non-finite trajectory or near-zero denominator in "
                    f"{'beta_R' if r_result is None else 'beta_H'})."
                )
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{n_draws}")
                continue

            pct_change_R["mean"].append(r_result["mean"])
            pct_change_R["peak"].append(r_result["peak"])
            pct_change_R["median_weekly"].append(r_result["median_weekly"])

            pct_change_H["mean"].append(h_result["mean"])
            pct_change_H["peak"].append(h_result["peak"])
            pct_change_H["median_weekly"].append(h_result["median_weekly"])

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_draws}")

        n_valid = n_draws - n_excluded
        print(f"  valid draws: {n_valid}/{n_draws} (excluded: {n_excluded})")

        if n_valid == 0:
            raise RuntimeError(
                f"{covariate} {sign_label}: no valid draws remained -- "
                "cannot compute summary statistics."
            )

        for component, store in [("beta_R", pct_change_R), ("beta_H", pct_change_H)]:
            for stat_name, values in store.items():
                values = np.array(values)
                med = np.median(values)
                lo, hi = np.percentile(values, [2.5, 97.5])
                rows.append({
                    "scenario_set": scenario_set,
                    "delta_z": sign * delta_z,
                    "delta_raw": sign * delta_z * SD_RAW[covariate],
                    "covariate": covariate,
                    "direction": sign_label,
                    "component": component,
                    "statistic": stat_name,
                    "median_pct_change": med,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_posterior_draws": n_draws,
                    "n_valid_draws": n_valid,
                    "n_excluded_draws": n_excluded,
                })

        print(f"  beta_R mean change: median={np.median(pct_change_R['mean']):.1f}%  "
              f"(95% CrI {np.percentile(pct_change_R['mean'],2.5):.1f} to {np.percentile(pct_change_R['mean'],97.5):.1f})")
        print(f"  beta_H mean change: median={np.median(pct_change_H['mean']):.1f}%  "
              f"(95% CrI {np.percentile(pct_change_H['mean'],2.5):.1f} to {np.percentile(pct_change_H['mean'],97.5):.1f})")

all_df = pd.DataFrame(rows)

# Raw-unit results: same file and same columns as before, so existing
# downstream scripts (e.g. the forest plot) keep working unchanged.
NEW_COLS = ["scenario_set", "delta_z", "delta_raw"]
result_df = all_df[all_df.scenario_set == "raw_units"].drop(columns=NEW_COLS).reset_index(drop=True)
csv_path = OUTPUT_DIR / "climate_perturbation_effects_posterior.csv"
result_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")

# +/- 1 SD results: new file, with delta_z and delta_raw kept for clarity.
result_1sd_df = all_df[all_df.scenario_set == "one_sd"].reset_index(drop=True)
csv_1sd_path = OUTPUT_DIR / "climate_perturbation_effects_posterior_1sd.csv"
result_1sd_df.to_csv(csv_1sd_path, index=False)
print(f"[OK] Saved: {csv_1sd_path}")


# =============================================================================
# LaTeX table -- mean-change statistic only, for main-text brevity
# (full mean/peak/median-weekly breakdown is in the CSV for Supplement)
# =============================================================================

def write_latex_table(df, tex_path, scenario_labels, caption_lead, label):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption_lead + r" Values show the posterior median percentage change "
                 r"in the time-averaged coefficient, with 95\% credible intervals, relative to the unperturbed "
                 r"historical-climate baseline (" + str(n_draws) + r" posterior draws per scenario; excluded "
                 r"draws, if any, are reported per-scenario in the accompanying CSV).}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Direction & $\beta_R$ mean change (\%) & $\beta_H$ mean change (\%) \\")
    lines.append(r"\midrule")
    for covariate in DELTA_Z:
        for sign_label in ["+", "-"]:
            r_row = df[(df.covariate == covariate) & (df.direction == sign_label)
                       & (df.component == "beta_R") & (df.statistic == "mean")].iloc[0]
            h_row = df[(df.covariate == covariate) & (df.direction == sign_label)
                       & (df.component == "beta_H") & (df.statistic == "mean")].iloc[0]
            lines.append(
                f"{scenario_labels[covariate]} & {sign_label} & "
                f"{r_row.median_pct_change:.1f} ({r_row.ci_lo:.1f}, {r_row.ci_hi:.1f}) & "
                f"{h_row.median_pct_change:.1f} ({h_row.ci_lo:.1f}, {h_row.ci_hi:.1f}) \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved: {tex_path}")


# Raw-unit table (unchanged content)
write_latex_table(
    result_df,
    OUTPUT_DIR / "climate_perturbation_effects_posterior.tex",
    {"temperature": "Temperature ($\\pm 2^\\circ$C)",
     "vapour_pressure": "Vapour pressure ($\\pm 0.05$ inHg)",
     "rainfall": "Rainfall ($\\pm 2$ in)"},
    r"Posterior climate perturbation effects on transmission coefficients under the "
    r"best-supported climate-driven model.",
    "tab:direct_beta_effects_posterior",
)

# +/- 1 SD table (new)
write_latex_table(
    result_1sd_df,
    OUTPUT_DIR / "climate_perturbation_effects_posterior_1sd.tex",
    {"temperature": f"Temperature ($\\pm 1$ SD $= \\pm {sd_xT / 1.8:.2f}^\\circ$C)",
     "vapour_pressure": f"Vapour pressure ($\\pm 1$ SD $= \\pm {sd_xe * 33.8639:.2f}$ hPa)",
     "rainfall": f"Rainfall ($\\pm 1$ SD $= \\pm {sd_xR * 25.4:.0f}$ mm)"},
    r"Posterior climate perturbation effects on transmission coefficients under the "
    r"best-supported climate-driven model, with each climate covariate shifted by one standard "
    r"deviation of its weekly series, so that the three covariates are compared on the same relative scale.",
    "tab:direct_beta_effects_posterior_1sd",
)

print("\nDone.")