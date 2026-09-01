# =============================================================================
# sweep_gamma_reseeding_falsification.py
# =============================================================================
"""
Falsification check for the "every wave requires renewed spillover"
conclusion.

The earlier robustness check (check_spillover_reseeding_robustness.py)
showed that across the data-supported posterior range of gamma (lower and
upper quartiles), human transmission stays negligible (~0.5-1.1% of its
own peak) at BOTH of the two genuine internal inter-wave troughs
(1897-07-23 and 1898-06-24) -- meaning the model never lets human
transmission alone bridge a trough and reignite a wave, for any gamma
the data actually support.

By itself, that could be read two ways:

    (a) A genuine, falsifiable empirical result: the data specifically
        rule out the parameter regime where human transmission COULD
        bridge a trough, even though such a regime exists in principle.

    (b) A vacuous tautology: beta_H(t)*I(t)/N is just always going to be
        ~0 whenever I(t)~0, for ANY gamma, so the check never had a
        chance to fail.

This script distinguishes (a) from (b) directly, by deliberately pushing
gamma FAR below the data-supported range -- into territory no literature
source on plague infectious period supports -- and checking whether the
trough behavior eventually breaks down.

Grid extension (this version)
-------------------------------
The previous grid (0.05-3.00/week, i.e. infectious periods up to 140 days)
found NO breakdown anywhere: worst-trough leakage reached only 8.3% of
peak even at 140 days, well under the 10% reference threshold. This
version extends the grid further down (to gamma=0.02/week, a 350-day
infectious period) purely to determine whether a breakdown point exists
at all, before deciding how to word the manuscript claim. Note that
infectious periods in this extended range (175-350 days) have NO
biological basis for Y. pestis or any known human pathogen -- this is a
deliberately unrealistic probe of the model's mathematical structure, not
a claim about plausible parameter values. See the printed verdict for how
to interpret whatever this run finds.

Trough identification
----------------------
Uses the two explicitly identified, validated internal inter-wave troughs
(1897-07-23 and 1898-06-24), checked separately rather than pooled, same
as check_spillover_reseeding_robustness.py.

Method
------
All parameters except gamma are held FIXED at the all-climate MLE point
(from the final_v2 result). Gamma alone is swept across a wide grid. This
is a deliberate counterfactual sweep, NOT a refit -- it asks "what would
this exact parameterization do if gamma were different," holding
everything else constant, which is the right question for a falsification
check (the earlier robustness check already covered the question of what
the actual posterior supports).

For context, the sweep grid is annotated against the two candidate
biological readings from the literature (Gani & Leach 2004 for pneumonic
plague; Dean et al. 2018 PNAS for the human-ectoparasite bubonic model),
so the figure shows exactly how far outside literature-supported
territory you have to go before the conclusion could fail.

Outputs
-------
results/final_figures/spillover_reseeding_check/
    gamma_falsification_sweep.csv
    gamma_falsification_sweep.png
    gamma_falsification_sweep.pdf

Usage
-----
    python scripts/sweep_gamma_reseeding_falsification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
from likelihood import build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

BETA_H_MODE = "all_climate"
FIT_TARGET = "plague_deaths_reported_fit"

SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = E0 = R0_INIT = 0.0
WEEK_RULE = "W-FRI"

# The two genuine internal inter-wave troughs (validated against
# wave_boundary_weeks.csv; see check_spillover_reseeding_robustness.py).
TROUGH_1_DATE = pd.Timestamp("1897-07-23")  # between Wave 1 and Wave 2
TROUGH_2_DATE = pd.Timestamp("1898-06-24")  # between Wave 2 and Wave 3

# All parameters held fixed at this point EXCEPT gamma, which is swept below.
# This is the exact all-climate MLE theta from the final_v2 result (verified
# earlier: logL = -779.370198054828 at gamma = 2.7594435708006366).
FIXED_PARAMS = {
    "N0": 847147.1241052961,
    "C_seed": 45.46412824853358,
    "a0": -10.073045091632157,
    "aT": -0.6082436937953069,
    "aTT": -0.6844148082093953,
    "ae": 0.13827360046742365,
    "aR": -2.139813921992762,
    "betaH0": 1.6996953492852156,
    "psi": 0.993549838769308,
    "nb_kappa": 6.697431515981066,
    "bT_H": 0.23770891916478698,
    "be_H": -0.8540293425921641,
    "bR_H": 0.7735783778330729,
}

# Sweep grid: extended down to gamma=0.02/week (350-day infectious period)
# from the previous minimum of 0.05/week (140 days), which found no
# breakdown at all. New low-end points: 0.02, 0.03, 0.04 (350, 233, 175
# days respectively) -- purely to determine whether a breakdown point
# exists anywhere before finalizing how this result is worded.
GAMMA_GRID = np.array([
    0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65,
    0.75, 0.90, 1.10, 1.30, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00,
])

# Literature anchors (see conversation / Methods for citations):
#   Gani & Leach 2004 (pneumonic infectious period ~2.5 d)      -> ~2.8/week
#   Dean et al. 2018 (ectoparasite moribund phase ~2 d)         -> ~3.5/week
#   Dean et al. 2018 (ectoparasite total bacteremic period ~9d) -> ~0.8/week
LITERATURE_ANCHORS = {
    "Dean et al. 2018\n(total bacteremic period)": 0.78,
    "Gani & Leach 2004\n(pneumonic)": 2.80,
    "Dean et al. 2018\n(moribund phase)": 3.50,
}

ROBUSTNESS_THRESHOLD_PCT = 10.0

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_figures" / "spillover_reseeding_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_CSV = (
    PROJECT_ROOT / "results" / "final_figures" / "transmission_dynamics"
    / "wave_analysis" / "wave_boundary_weeks.csv"
)


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
# Identify the two genuine internal troughs (validated against the actual
# wave-boundary detection output, not a raw threshold)
# =============================================================================

if BOUNDARY_CSV.exists():
    boundary_dates = pd.read_csv(BOUNDARY_CSV, parse_dates=["week_ending"])["week_ending"]
    boundary_dates_set = set(pd.to_datetime(boundary_dates))
    for label, d in (("Trough 1 (1897-07-23)", TROUGH_1_DATE),
                      ("Trough 2 (1898-06-24)", TROUGH_2_DATE)):
        if d not in boundary_dates_set:
            raise ValueError(
                f"{label} not found among the wave-boundary dates in "
                f"{BOUNDARY_CSV}. Check that the hardcoded trough dates "
                f"still match the current wave-boundary detection output."
            )
else:
    print(
        f"WARNING: {BOUNDARY_CSV} not found -- proceeding with the "
        "hardcoded trough dates without cross-checking them against the "
        "wave-boundary detection output. Run "
        "analyse_epidemic_waves_and_amplification.py first if you want "
        "that validation."
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
    "Using the two genuine internal inter-wave troughs only:\n"
    f"  Trough 1 (Wave 1 -> Wave 2): {TROUGH_1_DATE.date()} "
    f"(observed deaths = {observed[trough1_idx]:.0f})\n"
    f"  Trough 2 (Wave 2 -> Wave 3): {TROUGH_2_DATE.date()} "
    f"(observed deaths = {observed[trough2_idx]:.0f})"
)
print(
    f"\nSweeping gamma over {len(GAMMA_GRID)} values (now extended down to "
    f"{GAMMA_GRID.min():.2f}/week, ~{7/GAMMA_GRID.min():.0f}-day infectious "
    f"period), holding all other parameters fixed at the all-climate MLE point."
)


# =============================================================================
# Simulate one gamma value
# =============================================================================

def simulate_for_gamma(gamma: float) -> pd.DataFrame:
    sample = dict(FIXED_PARAMS)
    sample["gamma"] = float(gamma)
    sim_params = build_sim_params(sample, BETA_H_MODE)

    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR, N=sample["N0"], params=sim_params,
        substeps_per_week=SUBSTEPS_PER_WEEK, tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT, C_seed=sample["C_seed"],
        betaH_mode=BETA_H_MODE, start_date=start_date,
    )
    sim["human_force"] = sim["lambda"] - sim["beta_R"]
    return sim


def weekly_mean(sim: pd.DataFrame, col: str) -> np.ndarray:
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    wk = df.set_index("date")[col].resample(WEEK_RULE).mean()
    wk = wk.reindex(week_ending)
    return wk.to_numpy(dtype=float)


# =============================================================================
# Run sweep
# =============================================================================

rows = []
for gamma in GAMMA_GRID:
    sim = simulate_for_gamma(gamma)
    human_traj = weekly_mean(sim, "human_force")

    peak_human = np.nanmax(human_traj)

    trough1_human = human_traj[trough1_idx]
    trough2_human = human_traj[trough2_idx]

    frac_trough1 = trough1_human / peak_human if peak_human > 0 else np.nan
    frac_trough2 = trough2_human / peak_human if peak_human > 0 else np.nan
    worst_frac = np.nanmax([frac_trough1, frac_trough2])

    infectious_period_days = 7.0 / gamma

    rows.append({
        "gamma": gamma,
        "infectious_period_days": infectious_period_days,
        "peak_human_force": peak_human,
        "trough1_human_force": trough1_human,
        "trough2_human_force": trough2_human,
        "trough1_pct_of_peak": frac_trough1 * 100.0,
        "trough2_pct_of_peak": frac_trough2 * 100.0,
        "worst_trough_pct_of_peak": worst_frac * 100.0,
    })

    flag = " <-- BREAKS DOWN" if worst_frac * 100.0 > ROBUSTNESS_THRESHOLD_PCT else ""
    print(
        f"  gamma={gamma:5.2f}/week (infectious period ~{infectious_period_days:5.1f} d): "
        f"trough1={frac_trough1*100:6.2f}%  trough2={frac_trough2*100:6.2f}%  "
        f"worst={worst_frac*100:6.2f}% of peak{flag}"
    )

sweep_df = pd.DataFrame(rows)

csv_path = OUTPUT_DIR / "gamma_falsification_sweep.csv"
sweep_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved: {csv_path}")


# =============================================================================
# Verdict
# =============================================================================

breakdown = sweep_df[sweep_df["worst_trough_pct_of_peak"] > ROBUSTNESS_THRESHOLD_PCT]

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)

if breakdown.empty:
    print(
        f"\n  The trough-leakage fraction (worst of the two genuine troughs) never "
        f"exceeded {ROBUSTNESS_THRESHOLD_PCT:.0f}% even at the most extreme gamma "
        f"tested ({GAMMA_GRID.min():.2f}/week, ~{7/GAMMA_GRID.min():.0f}-day "
        f"infectious period -- far outside any literature-supported range).\n\n"
        f"  This means the 'spillover reseeds every wave' result could NOT be falsified\n"
        f"  even by an implausibly slow-recovery counterfactual. That weakens the claim\n"
        f"  that this is a genuine, falsifiable empirical finding rather than a structural\n"
        f"  tautology -- consider extending the grid to even more extreme gamma, or\n"
        f"  reporting the result with that caveat."
    )
else:
    breakdown_gamma = breakdown["gamma"].max()
    breakdown_days = 7.0 / breakdown_gamma
    which_trough = (
        "Trough 1" if breakdown.iloc[breakdown["gamma"].argmax()]["trough1_pct_of_peak"]
        >= breakdown.iloc[breakdown["gamma"].argmax()]["trough2_pct_of_peak"]
        else "Trough 2"
    )
    print(
        f"\n  The trough-leakage fraction exceeds {ROBUSTNESS_THRESHOLD_PCT:.0f}% once gamma drops to "
        f"{breakdown_gamma:.2f}/week or below\n"
        f"  (infectious period >= ~{breakdown_days:.0f} days), first at {which_trough}.\n\n"
        f"  This CONFIRMS the spillover-reseeding result is a genuine, falsifiable finding:\n"
        f"  there exists a parameter regime (gamma <~ {breakdown_gamma:.2f}/week) where the model\n"
        f"  WOULD let human transmission alone bridge inter-wave troughs -- but the earlier\n"
        f"  robustness check showed the data-supported posterior (lower/upper quartiles) never\n"
        f"  enters that regime. The result is informative precisely because it could have\n"
        f"  failed, and didn't, for any gamma the data actually support."
    )
    print(
        f"\n  For context: {breakdown_days:.0f}+ days is well beyond both literature readings used\n"
        f"  earlier (Gani & Leach 2004: ~2.5 d; Dean et al. 2018: ~2-9 d), so the breakdown\n"
        f"  point itself sits outside biologically defensible territory for this disease."
    )


# =============================================================================
# Figure
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sweep_df["gamma"], sweep_df["trough1_pct_of_peak"], marker="o",
        color="#1f77b4", linewidth=1.5, markersize=5, zorder=3,
        label="Trough 1 (1897-07-23)")
ax.plot(sweep_df["gamma"], sweep_df["trough2_pct_of_peak"], marker="s",
        color="#B3222B", linewidth=1.5, markersize=5, zorder=3,
        label="Trough 2 (1898-06-24)")

ax.axhline(ROBUSTNESS_THRESHOLD_PCT, linestyle="--", color="gray",
           label=f"{ROBUSTNESS_THRESHOLD_PCT:.0f}% robustness threshold", zorder=2)

# Shade the data-supported posterior range (lower/upper quartiles from the
# earlier stratified check)
ax.axvspan(0.75, 3.00, color="#1f77b4", alpha=0.10, zorder=1,
           label="Data-supported posterior range\n(from stratified check)")

for label, gval in LITERATURE_ANCHORS.items():
    ax.axvline(gval, color="#2C7C2C", linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate(label, xy=(gval, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
                fontsize=7, rotation=90, va="top", ha="right", color="#2C7C2C")

ax.set_xlabel(r"$\gamma$ (per week) — counterfactual sweep, all other params fixed at MLE")
ax.set_ylabel("Human transmission at trough\n(% of that gamma's own peak)")
ax.set_title("Falsification check: does the spillover-reseeding result break down\n"
              "for biologically implausible gamma?")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

fig_path = OUTPUT_DIR / "gamma_falsification_sweep.png"
fig_path_pdf = OUTPUT_DIR / "gamma_falsification_sweep.pdf"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
fig.savefig(fig_path_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] Saved: {fig_path}")
print(f"[OK] Saved: {fig_path_pdf}")

print("\nDone.")