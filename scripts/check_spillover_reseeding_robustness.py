# =============================================================================
# check_spillover_reseeding_robustness.py
# =============================================================================
"""
Supporting robustness check: does prevalence-associated amplification stay
small at the two internal inter-wave troughs across the lower and upper
quartiles of the gamma posterior, not just at a single best-fit /
point-estimate parameter vector?

Background
----------
The all-climate model has a gamma-betaH0 degeneracy: weekly mortality
data constrain the ratio betaH0/gamma (the amplification reproduction
number, R_{0,H}) more strongly than the individual rate constants.
Because gamma controls the rate at which the infectious compartment
declines, uncertainty in gamma could affect the amount of
prevalence-associated amplification remaining between epidemic waves.
The current posterior distribution of gamma is therefore summarized
directly at runtime rather than hard-coded here (see the printed
"Gamma posterior" line, which reports the median, 95% credible
interval, IQR, and sampled range).

Because gamma controls how fast I(t) decays between epidemic waves, a
slow gamma (long infectious period) could in principle leave enough
residual infectious individuals in the population between waves for
prevalence-associated amplification (beta_H(t)*I(t)/N) to reignite a wave
on its own, WITHOUT a fresh spillover pulse from beta_R(t). If that
happened for plausible parameter draws, it would weaken the case that
renewed reservoir-associated input is required for later waves.

What this script does
----------------------
1. Loads the final all-climate posterior (equal-weight resampled,
   reproducibly).
2. Splits posterior draws into a LOW-gamma group (bottom quartile, i.e.
   slowest recovery / longest infectious period -- the scenario most
   favorable to amplification-only reignition) and a HIGH-gamma group
   (top quartile, fastest recovery). Each group is drawn using its own,
   independently seeded random generator.
3. For a random sample of draws from each group (reproducibly selected),
   re-simulates the full mechanistic model (mechanistic_model.simulate_seir_dde
   -- the same simulator used for fitting, not a re-implementation).
4. Decomposes lambda(t) = beta_R(t) + amplification_force(t) exactly (using
   amplification_force = lambda - beta_R, which is exact by construction --
   see final_transmission_dynamics.py for why this avoids a one-substep
   inconsistency that a naive beta_H*I/N recomputation would introduce).
5. At the two pre-specified INTERNAL inter-wave trough weeks only
   (1897-07-23, between Waves 1 and 2; 1898-06-24, between Waves 2 and 3 --
   a subset of analyse_epidemic_waves_and_amplification.py's full
   wave-boundary output, which also includes the series' start/end
   boundaries that are not interior troughs), computes how large the
   amplification force gets relative to its own peak value, for every
   draw in both groups, at EACH trough separately.
6. Compares the LOW-gamma and HIGH-gamma groups descriptively.

IMPORTANT -- how to interpret this script's output
---------------------------------------------------
This analysis establishes that prevalence-associated amplification is small
at the two internal troughs, across the lower and upper quartiles of the
gamma posterior (not the full posterior, and not a claim about covering the
literature's full range of biological readings -- see Background above).
That is necessary context, but it is NOT a standalone proof that renewed
spillover is required for later waves. The counterfactual spillover-removal
analysis (counterfactual_infections_all_climate.py / counterfactual_
spillover_cutoff_sensitivity.py), which directly re-simulates the model with
spillover switched off and shows what happens to cumulative infections, is
the primary evidence for that claim. This script is a robustness/supporting
analysis for that conclusion, not a substitute for it.

Requires
--------
1. The final all-climate nested-sampling result:
       results/final_model_comparison_v2/nested_sampling/
           all_climate__plague_deaths_reported_fit__final_v2.pkl

2. The pre-specified wave-boundary weeks, produced by the updated
   analyse_epidemic_waves_and_amplification.py:
       results/final_figures/transmission_dynamics/wave_analysis/
           wave_boundary_weeks.csv
   Run that script first if this file does not exist. This script uses
   only the two INTERNAL trough dates from that file (1897-07-23,
   1898-06-24), not the series' start/end boundaries.

Usage
-----
    python scripts/check_spillover_reseeding_robustness.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynesty.utils import resample_equal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

FIT_TARGET = "plague_deaths_reported_fit"
BETA_H_MODE = "all_climate"

RESULT_PKL = (
    PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
    / f"{BETA_H_MODE}__{FIT_TARGET}__final_v2.pkl"
)

# Pre-specified wave-boundary CSV, produced by
# analyse_epidemic_waves_and_amplification.py. We only use the two INTERNAL
# inter-wave troughs from this file, not the series' start/end boundaries.
BOUNDARY_CSV = (
    PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
    / "wave_analysis" / "wave_boundary_weeks.csv"
)

# The two internal inter-wave troughs, named explicitly rather than inferred
# from position in the boundary file.
TROUGH_1_DATE = pd.Timestamp("1897-07-23")  # between Wave 1 and Wave 2
TROUGH_2_DATE = pd.Timestamp("1898-06-24")  # between Wave 2 and Wave 3

N_DRAWS_PER_GROUP = 200

# Separate, independently reproducible seeds for the low- and high-gamma
# groups (previously both groups drew from one shared rng instance in a
# fixed call order, making each group's draws depend on the other's).
RANDOM_SEED_RESAMPLE = 42   # for the posterior equal-weight resample
RANDOM_SEED_LOW = 1001      # for selecting draws within the low-gamma group
RANDOM_SEED_HIGH = 2002     # for selecting draws within the high-gamma group

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = E0 = R0_INIT = 0.0
WEEK_RULE = "W-FRI"

# Reference marker only -- NOT a pass/fail threshold. See the descriptive
# summary at the end of the script for how results should be interpreted.
REFERENCE_MARKER_PCT = 10.0

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures" / "spillover_reseeding_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load prepared data
# =============================================================================

if not paths.MERGED_CSV.exists():
    raise FileNotFoundError(f"Prepared data not found:\n{paths.MERGED_CSV}")

merged_df = pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
merged_df = merged_df.sort_values(paths.DATE_COL).reset_index(drop=True)

week_ending = merged_df[paths.DATE_COL]
zT = merged_df["zT"].to_numpy(dtype=float)
ze = merged_df["ze"].to_numpy(dtype=float)
zR = merged_df["zR"].to_numpy(dtype=float)
start_date = str(pd.Timestamp(week_ending.iloc[0]).date())
n_weeks = len(merged_df)

if FIT_TARGET not in merged_df.columns:
    raise ValueError(f"{FIT_TARGET!r} not found in prepared data.")

observed = merged_df[FIT_TARGET].to_numpy(dtype=float)


# =============================================================================
# Load pre-specified trough weeks and restrict to the two internal troughs
# =============================================================================

if not BOUNDARY_CSV.exists():
    raise FileNotFoundError(
        f"Wave boundary weeks not found:\n{BOUNDARY_CSV}\n\n"
        "Run the updated analyse_epidemic_waves_and_amplification.py first "
        "(it exports wave_boundary_weeks.csv as part of its normal output)."
    )

boundary_dates = pd.read_csv(BOUNDARY_CSV, parse_dates=["week_ending"])["week_ending"]
boundary_dates_set = set(pd.to_datetime(boundary_dates))

for label, d in (("Trough 1 (1897-07-23)", TROUGH_1_DATE),
                  ("Trough 2 (1898-06-24)", TROUGH_2_DATE)):
    if d not in boundary_dates_set:
        raise ValueError(
            f"{label} not found among the wave-boundary dates in "
            f"{BOUNDARY_CSV}. Check that the hardcoded trough dates still "
            f"match the current wave-boundary detection output."
        )

week_ending_dt = pd.to_datetime(week_ending)
trough1_mask = (week_ending_dt == TROUGH_1_DATE).to_numpy()
trough2_mask = (week_ending_dt == TROUGH_2_DATE).to_numpy()

if trough1_mask.sum() != 1:
    raise ValueError(
        f"Expected exactly one fitted week matching Trough 1 "
        f"({TROUGH_1_DATE.date()}), found {trough1_mask.sum()}."
    )
if trough2_mask.sum() != 1:
    raise ValueError(
        f"Expected exactly one fitted week matching Trough 2 "
        f"({TROUGH_2_DATE.date()}), found {trough2_mask.sum()}."
    )

trough1_idx = int(np.where(trough1_mask)[0][0])
trough2_idx = int(np.where(trough2_mask)[0][0])

print(f"Fitted weeks: {n_weeks}")
print(
    "Using the two internal inter-wave troughs only:\n"
    f"  Trough 1 (Wave 1 -> Wave 2): {TROUGH_1_DATE.date()}\n"
    f"  Trough 2 (Wave 2 -> Wave 3): {TROUGH_2_DATE.date()}"
)


# =============================================================================
# Load posterior
# =============================================================================

if not RESULT_PKL.exists():
    raise FileNotFoundError(
        f"Could not find final all-climate result:\n{RESULT_PKL}\n"
        "Edit RESULT_PKL at the top of this script if your file is named "
        "differently or lives elsewhere."
    )

with open(RESULT_PKL, "rb") as f:
    saved = pickle.load(f)

results = saved["results"]
param_names = list(saved["param_names"])

weights = np.exp(results.logwt - results.logz[-1])
resample_rstate = np.random.default_rng(RANDOM_SEED_RESAMPLE)
posterior = resample_equal(results.samples, weights, rstate=resample_rstate)

gamma_idx = param_names.index("gamma")
gamma_vals = posterior[:, gamma_idx]

q025, q25, q75, q975 = np.percentile(gamma_vals, [2.5, 25, 75, 97.5])
print(
    f"\nGamma posterior: median={np.median(gamma_vals):.3f}, "
    f"95% CrI=({q025:.3f}, {q975:.3f}), "
    f"IQR=({q25:.3f}, {q75:.3f}), "
    f"sampled range=({gamma_vals.min():.3f}, {gamma_vals.max():.3f})"
)
print(
    "Note: this check compares the LOWER quartile (<= q25) and UPPER "
    "quartile (>= q75) of the gamma posterior, not its full range."
)


# =============================================================================
# Simulation helper
# =============================================================================

def simulate_one_draw(theta: np.ndarray) -> pd.DataFrame:
    sample = dict(zip(param_names, np.asarray(theta, dtype=float)))
    sim_params = build_sim_params(sample, BETA_H_MODE)

    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR,
        N=sample["N0"],
        params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=BETA_H_MODE,
        start_date=start_date,
    )
    # Exact decomposition: amplification_force = lambda - beta_R (not a
    # naive beta_H*I/N recomputation -- see final_transmission_dynamics.py
    # docstring for why that would introduce a one-substep inconsistency).
    sim["amplification_force"] = sim["lambda"] - sim["beta_R"]
    return sim


def weekly_mean(sim: pd.DataFrame, col: str) -> np.ndarray:
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    wk = df.set_index("date")[col].resample(WEEK_RULE).mean()
    wk = wk.reindex(week_ending)
    return wk.to_numpy(dtype=float)


# =============================================================================
# Run one stratified group
# =============================================================================

def run_group(mask: np.ndarray, n_draws: int, label: str, rng: np.random.Generator):
    idx_pool = np.where(mask)[0]
    n_draws = min(n_draws, len(idx_pool))
    draw_idx = rng.choice(idx_pool, size=n_draws, replace=False)

    amp_trajectories = np.full((n_draws, n_weeks), np.nan)
    gammas_used = np.empty(n_draws)
    valid = np.ones(n_draws, dtype=bool)
    n_nonfinite = 0
    n_zero_peak = 0

    for i, di in enumerate(draw_idx):
        theta = posterior[di]
        gammas_used[i] = theta[gamma_idx]

        try:
            sim = simulate_one_draw(theta)
            traj = weekly_mean(sim, "amplification_force")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"{label}: draw {di} raised {type(exc).__name__}: {exc} -- "
                "excluding this draw from the analysis."
            )
            valid[i] = False
            if (i + 1) % 20 == 0:
                print(f"    {label}: {i + 1}/{n_draws}")
            continue

        if len(traj) != n_weeks:
            warnings.warn(
                f"{label}: draw {di} returned {len(traj)} weekly values; "
                f"expected {n_weeks}. Excluding this draw."
            )
            valid[i] = False
            if (i + 1) % 20 == 0:
                print(f"    {label}: {i + 1}/{n_draws}")
            continue

        if not np.all(np.isfinite(traj)):
            n_nonfinite += 1
            warnings.warn(
                f"{label}: draw {di} (gamma={gammas_used[i]:.3f}) produced "
                "non-finite values in its amplification trajectory -- "
                "excluding this draw from the analysis."
            )
            valid[i] = False
            if (i + 1) % 20 == 0:
                print(f"    {label}: {i + 1}/{n_draws}")
            continue

        # By construction, amplification_force = lambda - beta_R should be
        # non-negative (lambda_A(t) = lambda(t) - beta_R(t) >= 0). Permit
        # only negligible floating-point subtraction error; a materially
        # negative value would indicate a real problem in the decomposition.
        if np.any(traj < -1e-12):
            warnings.warn(
                f"{label}: draw {di} (gamma={gammas_used[i]:.3f}) produced "
                "materially negative amplification-force values -- "
                "excluding this draw from the analysis."
            )
            valid[i] = False
            if (i + 1) % 20 == 0:
                print(f"    {label}: {i + 1}/{n_draws}")
            continue
        traj = np.maximum(traj, 0.0)

        amp_trajectories[i, :] = traj

        if (i + 1) % 20 == 0:
            print(f"    {label}: {i + 1}/{n_draws}")

    peak_amp = np.full(n_draws, np.nan)
    peak_amp[valid] = np.nanmax(amp_trajectories[valid, :], axis=1)

    zero_peak = valid & (~np.isfinite(peak_amp) | (peak_amp <= 0))
    if zero_peak.any():
        n_zero_peak = int(zero_peak.sum())
        warnings.warn(
            f"{label}: {n_zero_peak} draw(s) had a zero or non-finite peak "
            "amplification force -- excluding these draws from the "
            "trough/peak ratio (division would be undefined)."
        )
        valid = valid & ~zero_peak

    amp_trough1 = np.full(n_draws, np.nan)
    amp_trough2 = np.full(n_draws, np.nan)
    amp_trough1[valid] = amp_trajectories[valid, trough1_idx]
    amp_trough2[valid] = amp_trajectories[valid, trough2_idx]

    frac_trough1 = np.full(n_draws, np.nan)
    frac_trough2 = np.full(n_draws, np.nan)
    frac_trough1[valid] = amp_trough1[valid] / peak_amp[valid]
    frac_trough2[valid] = amp_trough2[valid] / peak_amp[valid]

    # Retained as a useful summary, but now derived from exactly the two
    # explicitly identified internal troughs (elementwise max of the two
    # per-trough fractions), not from an arbitrary multi-boundary mask.
    worst_frac_of_peak = np.full(n_draws, np.nan)
    worst_frac_of_peak[valid] = np.nanmax(
        np.vstack([frac_trough1[valid], frac_trough2[valid]]), axis=0
    )

    n_valid = int(valid.sum())
    n_excluded = n_draws - n_valid

    print(f"\n--- {label} ---")
    print(
        f"  n draws requested = {n_draws}, valid = {n_valid}, "
        f"excluded = {n_excluded} "
        f"(non-finite trajectory: {n_nonfinite}, zero/non-finite peak: {n_zero_peak})"
    )
    if n_valid > 0:
        print(
            f"  gamma range (valid draws) = "
            f"({gammas_used[valid].min():.3f}, {gammas_used[valid].max():.3f}), "
            f"median = {np.median(gammas_used[valid]):.3f}"
        )
        print(
            f"  Trough 1 ({TROUGH_1_DATE.date()}) amplification / peak: "
            f"median = {np.nanmedian(frac_trough1) * 100:.2f}%, "
            f"max = {np.nanmax(frac_trough1) * 100:.2f}%"
        )
        print(
            f"  Trough 2 ({TROUGH_2_DATE.date()}) amplification / peak: "
            f"median = {np.nanmedian(frac_trough2) * 100:.2f}%, "
            f"max = {np.nanmax(frac_trough2) * 100:.2f}%"
        )
        print(
            f"  Worst-of-two-troughs / peak: "
            f"median = {np.nanmedian(worst_frac_of_peak) * 100:.2f}%, "
            f"max = {np.nanmax(worst_frac_of_peak) * 100:.2f}%"
        )
    else:
        warnings.warn(f"{label}: no valid draws remained -- check inputs.")

    return {
        "label": label,
        "draw_idx": draw_idx,
        "valid": valid,
        "gammas": gammas_used,
        "amp_trajectories": amp_trajectories,
        "peak_amp": peak_amp,
        "amp_trough1": amp_trough1,
        "amp_trough2": amp_trough2,
        "frac_trough1": frac_trough1,
        "frac_trough2": frac_trough2,
        "worst_frac_of_peak": worst_frac_of_peak,
    }


print("\n" + "=" * 90)
print("RUNNING GAMMA-STRATIFIED TROUGH ROBUSTNESS CHECK")
print("=" * 90)

low_rng = np.random.default_rng(RANDOM_SEED_LOW)
high_rng = np.random.default_rng(RANDOM_SEED_HIGH)

low_group = run_group(
    gamma_vals <= q25, N_DRAWS_PER_GROUP,
    "LOW gamma (bottom quartile, slow recovery / long infectious period)",
    low_rng,
)
high_group = run_group(
    gamma_vals >= q75, N_DRAWS_PER_GROUP,
    "HIGH gamma (top quartile, fast recovery / short infectious period)",
    high_rng,
)


# =============================================================================
# Descriptive summary (NOT a standalone proof of reseeding)
# =============================================================================

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

low_valid = low_group["valid"]
high_valid = high_group["valid"]

low_max_pct = np.nanmax(low_group["worst_frac_of_peak"]) * 100 if low_valid.any() else float("nan")
high_max_pct = np.nanmax(high_group["worst_frac_of_peak"]) * 100 if high_valid.any() else float("nan")

print(
    f"\nLOW-gamma group (lower quartile of gamma posterior): "
    f"worst-of-two-troughs/peak ratio  "
    f"median={np.nanmedian(low_group['worst_frac_of_peak']) * 100:.2f}%  "
    f"max={low_max_pct:.2f}%"
)
print(
    f"HIGH-gamma group (upper quartile of gamma posterior): "
    f"worst-of-two-troughs/peak ratio  "
    f"median={np.nanmedian(high_group['worst_frac_of_peak']) * 100:.2f}%  "
    f"max={high_max_pct:.2f}%"
)

print("\nDESCRIPTIVE SUMMARY (not a standalone proof of reseeding):")
print(
    f"  At the two internal inter-wave troughs, prevalence-associated "
    f"amplification reaches at most\n"
    f"  {low_max_pct:.1f}% (lower-quartile-gamma group) and "
    f"{high_max_pct:.1f}% (upper-quartile-gamma group)\n"
    f"  of its own within-draw peak value.\n\n"
    f"  These values describe how strongly the amplification component "
    f"persists at the two\n"
    f"  inter-wave troughs under contrasting regions of the gamma "
    f"posterior. They provide\n"
    f"  supporting context for the reseeding interpretation, but are not a "
    f"standalone demonstration\n"
    f"  that renewed reservoir-associated input is required. The "
    f"spillover-removal counterfactual\n"
    f"  provides the direct test of that requirement."
)


# =============================================================================
# Save results
# =============================================================================

summary_rows = []
for group in (low_group, high_group):
    for i in range(len(group["gammas"])):
        if not group["valid"][i]:
            continue
        summary_rows.append({
            "group": group["label"],
            "posterior_index": int(group["draw_idx"][i]),
            "gamma": group["gammas"][i],
            "peak_amplification_force": group["peak_amp"][i],
            "amplification_force_trough1": group["amp_trough1"][i],
            "amplification_force_trough2": group["amp_trough2"][i],
            "trough1_pct_of_peak": group["frac_trough1"][i] * 100.0,
            "trough2_pct_of_peak": group["frac_trough2"][i] * 100.0,
            "worst_trough_pct_of_peak": group["worst_frac_of_peak"][i] * 100.0,
        })

summary_df = pd.DataFrame(summary_rows)
csv_path = OUTPUT_DIR / "gamma_stratified_trough_check.csv"
summary_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")

# Save the exact selected posterior parameter vectors used, keyed by group,
# so any specific result can be traced back to its full parameter vector.
# Includes excluded draws too (flagged), for full traceability.
draws_rows = []
for group in (low_group, high_group):
    for i, di in enumerate(group["draw_idx"]):
        row = {
            "group": group["label"],
            "posterior_index": int(di),
            "valid": bool(group["valid"][i]),
        }
        row.update(dict(zip(param_names, posterior[di])))
        draws_rows.append(row)

draws_df = pd.DataFrame(draws_rows)
draws_csv_path = OUTPUT_DIR / "gamma_stratified_selected_draws.csv"
draws_df.to_csv(draws_csv_path, index=False)
print(f"[OK] Saved: {draws_csv_path}")


# =============================================================================
# Figure: trough-week amplification fraction vs gamma, both groups
# (worst-of-two-troughs fraction, as before, now clearly labeled as such)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(low_group["gammas"][low_valid], low_group["worst_frac_of_peak"][low_valid] * 100,
           alpha=0.6, label="Low-gamma group (lower quartile)", color="#1f77b4")
ax.scatter(high_group["gammas"][high_valid], high_group["worst_frac_of_peak"][high_valid] * 100,
           alpha=0.6, label="High-gamma group (upper quartile)", color="#E08214")
ax.axhline(REFERENCE_MARKER_PCT, linestyle="--", color="gray",
           label=f"{REFERENCE_MARKER_PCT:.0f}% reference marker")
ax.set_xlabel(r"$\gamma$ (per week)")
ax.set_ylabel(
    "Worst-of-two-troughs prevalence-associated\namplification (% of that draw's own peak)"
)
ax.set_title(
    r"Does slow recovery ($\gamma$) let prevalence-associated amplification "
    "bridge the internal inter-wave troughs?"
)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

fig_path = OUTPUT_DIR / "gamma_stratified_trough_check.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved: {fig_path}")

print("\nDone.")