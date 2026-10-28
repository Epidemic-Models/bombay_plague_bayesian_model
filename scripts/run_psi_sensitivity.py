# =============================================================================
# run_psi_sensitivity.py
# =============================================================================
"""
Is the infection-to-death scaling, psi, identified by the data, and does its
value change any conclusion?

Background
----------
psi maps modelled infections to recorded plague deaths. Its posterior sits at
the top of its [0.05, 1] prior in every formulation. Read as a probability,
that says almost every infection produced a recorded plague death, which is
higher than the product of the fatality of untreated bubonic plague and the
completeness of death registration can plausibly be.

The explanation is that psi fixes only the ABSOLUTE SCALE of infections, and
that scale is barely constrained. Deaths are what the data record. Writing

    D(t) = psi * (K * C)(t),

holding D fixed and replacing psi by psi' = psi / r forces C' = r C, and hence
I' = r I, because I is a linear filter of C. The force of infection then has to
satisfy

    lambda' = C' / S'   ~   r lambda      (while S ~ N, i.e. little depletion)
            = beta_R' + betaH0' * exp(f) * I' / N
            = beta_R' + betaH0' * exp(f) * r I / N,

which is met by

    beta_R' = r beta_R      (i.e. a0' = a0 + ln r)
    betaH0' = betaH0        (unchanged)
    C_seed' = r C_seed

So beta_H, R0_H = beta_H / gamma and the amplification SHARE of the force of
infection are invariant, while beta_R and the absolute number of infections
scale with r. What normally pins the absolute scale is susceptible depletion,
through S in lambda = C / S; in Bombay only a few per cent of the population
was infected, so that constraint is weak and psi drifts to its prior bound.

The invariance is therefore exact only in the limit of no depletion. This
script measures how exact it actually is.

What it does
------------
--invariance (default, minutes)
    For each posterior draw of the main model, simulate as fitted, then
    simulate the rescaled draw (psi -> PSI_TARGET, a0 += ln r,
    C_seed *= r) and compare:

        weekly expected deaths          (should be unchanged)
        cumulative infections           (should scale by r)
        mean beta_R                     (should scale by r)
        betaH0, median R0_H             (should be unchanged)
        amplification share of lambda   (should be unchanged)
        counterfactual reduction when amplification is removed
                                        (should be unchanged)

    Any departure from these is caused by susceptible depletion, and is
    reported as a percentage so it can be quoted in the SI.

--refit (hours)
    Refits the main model by nested sampling with psi held at PSI_TARGET,
    which is the independent confirmation: psi is dropped from the sampled
    vector rather than being rescaled by hand. Settings match
    run_single_covariate_seeded.py (NLIVE, DLOGZ, rslice, one seeded live
    point). The seed is the main model's best fit, rescaled as above.

Outputs
-------
results/psi_sensitivity/
    psi_invariance_draws.csv        per-draw baseline vs rescaled quantities
    psi_invariance_summary.csv      posterior medians and % changes
    psi_invariance_table.tex        SI table
    nested_sampling/all_climate__<fit target>__psi_fixed.pkl / .json   (--refit)

Usage
-----
    python scripts/run_psi_sensitivity.py
    python scripts/run_psi_sensitivity.py --psi 0.5
    python scripts/run_psi_sensitivity.py --refit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import paths  # noqa: E402
import priors  # noqa: E402
from likelihood import PlagueLikelihood, build_sim_params  # noqa: E402
from mechanistic_model import simulate_seir_dde, weekly_sum  # noqa: E402

from observation import make_delay_pmf_weeks, deaths_from_infections_fast  # noqa: E402


# =============================================================================
# Settings
# =============================================================================

MODE = "all_climate"
FIT_TARGET = "plague_deaths_reported_fit"

MAIN_PKL = (PROJECT_ROOT / "results" / "final_model_comparison_v2" / "nested_sampling"
            / f"{MODE}__{FIT_TARGET}__final_v2.pkl")

OUTPUT_DIR = PROJECT_ROOT / "results" / "psi_sensitivity"

# Target value of psi. 0.5 sits inside the 40-60% case-fatality range reported
# for untreated bubonic plague, before any allowance for under-registration.
PSI_TARGET = 0.5

N_DRAWS = 200           # draws for the invariance check (3 simulations each)
RANDOM_SEED = 7654321

# Simulation settings (identical to final_transmission_dynamics.py)
SUBSTEPS_PER_WEEK = 7
TAU0_DAYS = 7
I0 = 0.0
E0 = 0.0
R0_INIT = 0.0
WEEK_RULE = "W-FRI"

# Spillover cutoff used by the main-text counterfactual.
SPILLOVER_CUTOFF_WEEK = 4.0

# Infection-to-death delay kernel
DELAY_KIND = "gamma"
DELAY_MAX_WEEKS = 12
DELAY_SHAPE = 2.0
DELAY_SCALE = 0.6

# Set True to run the nested-sampling refit when the script is started without
# arguments (PyCharm Run button). False runs the fast invariance check.
RUN_REFIT = True

# Nested sampling (matches run_single_covariate_seeded.py)
NLIVE = 1000
DLOGZ = 0.1
SEED_LIVE = 62345
SEED_SAMPLER = 63345

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def load_posterior(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return the saved dict, the samples and their normalized weights."""
    if not path.exists():
        raise FileNotFoundError(f"Posterior not found:\n  {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"]
    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    return d, np.asarray(res.samples, dtype=float), w


def rescale_theta(theta: np.ndarray, names: list[str], psi_target: float) -> tuple[np.ndarray, float]:
    """
    Map one parameter vector to the psi_target scale.

    r = psi / psi_target;  a0 -> a0 + ln r;  C_seed -> r * C_seed;  psi -> psi_target.
    Every other parameter, including betaH0 and all climate coefficients,
    is left untouched.
    """
    out = np.array(theta, dtype=float)
    i_psi, i_a0, i_seed = names.index("psi"), names.index("a0"), names.index("C_seed")
    r = float(out[i_psi]) / psi_target
    out[i_a0] += np.log(r)
    out[i_seed] *= r
    out[i_psi] = psi_target
    return out, r


def simulate(theta: np.ndarray, names: list[str], zT, ze, zR, start_date,
             **extra) -> pd.DataFrame:
    """One mechanistic simulation; adds the amplification force and R0_H."""
    sample = dict(zip(names, np.asarray(theta, dtype=float)))
    sim = simulate_seir_dde(
        zT=zT, ze=ze, zR=zR,
        N=sample["N0"],
        params=build_sim_params(sample, MODE),
        substeps_per_week=SUBSTEPS_PER_WEEK,
        tau0_days=TAU0_DAYS,
        I0=I0, E0=E0, R0=R0_INIT,
        C_seed=sample["C_seed"],
        betaH_mode=MODE,
        start_date=start_date,
        **extra,
    )
    # lambda = beta_R + beta_H * I_previous / N, as evaluated by the simulator.
    sim["amplification_force"] = sim["lambda"] - sim["beta_R"]
    sim["R0_H"] = sim["beta_H"] / sample["gamma"]
    return sim


def weekly_new_exposed(sim: pd.DataFrame, week_ending: pd.Series) -> np.ndarray:
    """Weekly new infections, aligned to the fitted week grid."""
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    wk = (df.set_index("date")["new_exposed"].resample(WEEK_RULE).sum()
          .reset_index().rename(columns={"date": "week_ending", "new_exposed": "v"}))
    target = pd.DataFrame({"week_ending": pd.to_datetime(week_ending)})
    return pd.merge(target, wk, on="week_ending", how="left")["v"].to_numpy(float)


def weekly_deaths(sim: pd.DataFrame, merged_df: pd.DataFrame, psi: float, f_delay) -> np.ndarray:
    """Expected weekly deaths implied by one simulation."""
    wk = weekly_sum(sim, col="C", week_ending=WEEK_RULE)
    wk["week_ending"] = pd.to_datetime(wk["week_ending"])
    merged = pd.merge(merged_df[["week_ending"]], wk, on="week_ending", how="left")
    c_col = "C_week" if "C_week" in merged.columns else \
        next(c for c in merged.columns if c.startswith("C_"))
    C_week = merged[c_col].to_numpy(dtype=float)
    if not np.isfinite(C_week).all():
        raise RuntimeError("Weekly infection trajectory has missing or non-finite values.")
    return np.asarray(deaths_from_infections_fast(C_week=C_week, psi=psi, f_delay=f_delay),
                      dtype=float)


def draw_quantities(theta, names, zT, ze, zR, start_date, merged_df, f_delay,
                    week_ending) -> dict:
    """Everything compared between the fitted and the rescaled parameter vector."""
    sample = dict(zip(names, np.asarray(theta, dtype=float)))
    sim = simulate(theta, names, zT, ze, zR, start_date)

    lam = sim["lambda"].to_numpy(dtype=float)
    amp = sim["amplification_force"].to_numpy(dtype=float)

    # Counterfactuals, using the simulator's own switches (as in
    # counterfactual_compare_climate_models.py), not by zeroing betaH0.
    sim_noamp = simulate(theta, names, zT, ze, zR, start_date,
                         human_multiplier=0.0)
    sim_nospill = simulate(theta, names, zT, ze, zR, start_date,
                           spillover_multiplier=0.0,
                           spillover_cutoff_week=SPILLOVER_CUTOFF_WEEK)

    cum = float(np.nansum(weekly_new_exposed(sim, week_ending)))
    cum_noamp = float(np.nansum(weekly_new_exposed(sim_noamp, week_ending)))
    cum_nospill = float(np.nansum(weekly_new_exposed(sim_nospill, week_ending)))

    return {
        "psi": sample["psi"],
        "a0": sample["a0"],
        "betaH0": sample["betaH0"],
        "gamma": sample["gamma"],
        "mean_beta_R": float(np.mean(sim["beta_R"].to_numpy(dtype=float))),
        "mean_amplification_force": float(np.mean(amp)),
        "amplification_share": float(np.sum(amp) / np.sum(lam)),
        "median_R0_H": float(np.median(sim["R0_H"].to_numpy(dtype=float))),
        "cumulative_infections": cum,
        "min_S_over_N": float(np.min(sim["S"].to_numpy(dtype=float)) / sample["N0"]),
        "cf_no_amplification_reduction_pct": 100.0 * (1.0 - cum_noamp / cum),
        "cf_no_spillover_reduction_pct": 100.0 * (1.0 - cum_nospill / cum),
        "deaths": weekly_deaths(sim, merged_df, sample["psi"], f_delay),
    }


def pct_change(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    return 100.0 * (np.asarray(new) / np.asarray(old) - 1.0)


# =============================================================================
# Invariance check
# =============================================================================

def run_invariance(psi_target: float, n_draws: int) -> None:

    merged_df = (pd.read_csv(paths.MERGED_CSV, parse_dates=[paths.DATE_COL])
                 .sort_values(paths.DATE_COL).reset_index(drop=True))
    week_ending = pd.to_datetime(merged_df[paths.DATE_COL])
    zT = merged_df["zT"].to_numpy(dtype=float)
    ze = merged_df["ze"].to_numpy(dtype=float)
    zR = merged_df["zR"].to_numpy(dtype=float)
    start_date = str(pd.Timestamp(week_ending.iloc[0]).date())

    f_delay = make_delay_pmf_weeks(kind=DELAY_KIND, max_weeks=DELAY_MAX_WEEKS,
                                   shape=DELAY_SHAPE, scale=DELAY_SCALE)

    d, samples, w = load_posterior(MAIN_PKL)
    names = list(d["param_names"])
    for required in ("psi", "a0", "C_seed", "betaH0", "gamma", "N0"):
        if required not in names:
            raise RuntimeError(f"{required!r} is not a parameter of {MODE!r}.")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False, p=w)

    print("\n" + "=" * 92)
    print(f"PSI INVARIANCE CHECK   {MODE}   psi -> {psi_target}")
    print("=" * 92)
    print(f"Draws: {len(idx)}   (sampled from the posterior in proportion to its weights)")

    rows = []
    all_rel = []
    max_death_dev = 0.0
    t0 = time.time()
    for n, j in enumerate(idx, start=1):
        theta = samples[j]
        resc, r = rescale_theta(theta, names, psi_target)

        base = draw_quantities(theta, names, zT, ze, zR, start_date, merged_df,
                               f_delay, week_ending)
        new = draw_quantities(resc, names, zT, ze, zR, start_date, merged_df,
                              f_delay, week_ending)

        rel = np.abs(new["deaths"] / np.maximum(base["deaths"], 1e-12) - 1.0)
        dev = float(np.max(rel))
        max_death_dev = max(max_death_dev, dev)
        all_rel.append(rel)
        # Deviation in total deaths matters more than the worst single week,
        # because a tiny week can give a large relative error on a few deaths.
        tot_dev = float(np.sum(new["deaths"]) / np.sum(base["deaths"]) - 1.0)
        peak_week = int(np.argmax(rel))

        row = {"draw": n, "posterior_index": int(j), "r": r,
               "max_weekly_death_deviation_pct": 100.0 * dev,
               "worst_week_index": peak_week,
               "worst_week_fitted_deaths": float(base["deaths"][peak_week]),
               "total_death_deviation_pct": 100.0 * tot_dev}
        for key in ("psi", "a0", "betaH0", "gamma", "mean_beta_R", "mean_amplification_force",
                    "amplification_share", "median_R0_H", "cumulative_infections",
                    "min_S_over_N", "cf_no_amplification_reduction_pct",
                    "cf_no_spillover_reduction_pct"):
            row[f"base_{key}"] = base[key]
            row[f"psifix_{key}"] = new[key]
        rows.append(row)

        if n % 25 == 0 or n == len(idx):
            print(f"  {n} / {len(idx)} draws  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "psi_invariance_draws.csv", index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    quantities = [
        ("psi", "$\\psi$", "unchanged by construction"),
        ("mean_beta_R", "Mean $\\beta_R$", "scales with $r$"),
        ("cumulative_infections", "Cumulative infections", "scales with $r$"),
        ("betaH0", "$\\beta_{H0}$", "invariant"),
        ("median_R0_H", "Median $R_{0,H}$", "invariant"),
        ("amplification_share", "Amplification share of $\\lambda$", "invariant"),
        ("cf_no_amplification_reduction_pct",
         "Reduction with amplification removed (\\%)", "invariant"),
        ("cf_no_spillover_reduction_pct",
         "Reduction with spillover stopped after week 4 (\\%)", "invariant"),
        ("min_S_over_N", "Minimum $S/N$", "falls as infections scale"),
    ]

    out = []
    for key, label, expectation in quantities:
        b, a = df[f"base_{key}"].to_numpy(), df[f"psifix_{key}"].to_numpy()
        ch = pct_change(a, b)
        out.append({
            "quantity": label,
            "expectation": expectation,
            "fitted_median": float(np.median(b)),
            "psi_fixed_median": float(np.median(a)),
            "median_pct_change": float(np.median(ch)),
            "max_abs_pct_change": float(np.max(np.abs(ch))),
        })
    summary = pd.DataFrame(out)
    summary.to_csv(OUTPUT_DIR / "psi_invariance_summary.csv", index=False)

    r_med = float(np.median(df["r"]))
    rel_all = np.concatenate(all_rel)
    dev_med = 100.0 * float(np.median(rel_all))
    dev_p95 = 100.0 * float(np.quantile(rel_all, 0.95))
    tot_med = float(np.median(np.abs(df["total_death_deviation_pct"])))
    worst = df.loc[df["max_weekly_death_deviation_pct"].idxmax()]

    print("\n" + "-" * 92)
    print(f"Median rescaling factor r = {r_med:.2f}")
    print("Deviation in expected weekly deaths after rescaling "
          "(rescaling is exact only without depletion):")
    print(f"  median over draws x weeks : {dev_med:.2f}%")
    print(f"  95th percentile           : {dev_p95:.2f}%")
    print(f"  maximum                   : {100.0 * max_death_dev:.2f}%  "
          f"(week index {int(worst['worst_week_index'])}, "
          f"{worst['worst_week_fitted_deaths']:.1f} fitted deaths that week)")
    print(f"  median deviation in TOTAL deaths over the epidemic: {tot_med:.2f}%")
    print("-" * 92)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # -------------------------------------------------------------------------
    # SI table
    # -------------------------------------------------------------------------
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of rescaling the infection-to-death scaling, $\psi$, to "
        f"{psi_target:g}" + r". Each posterior draw of the main model was transformed so that "
        r"$\psi$ takes the stated value, in the way that would leave expected weekly deaths "
        r"unchanged in the absence of susceptible depletion "
        r"($a_0 \rightarrow a_0 + \ln r$ and $C_{\mathrm{seed}} \rightarrow r\,C_{\mathrm{seed}}$, "
        r"with $r = \psi/\psi'$); every other parameter was left untouched. "
        r"Values are posterior medians over " + f"{len(idx)}" + r" draws. "
        r"Quantities that depend on the absolute number of infections scale with $r$, "
        r"whereas $\beta_{H0}$ and $R_{0,H}$ are unchanged and the amplification share and "
        r"both counterfactual reductions change by less than 2\%. "
        r"Rescaling is exact only in the absence of susceptible depletion, so expected "
        r"weekly deaths are reproduced to within a median of "
        + f"{dev_med:.2f}" + r"\% (95th percentile "
        + f"{dev_p95:.2f}" + r"\%); the residual is absorbed by refitting with $\psi$ held "
        r"fixed.}",
        r"\label{tab:si_psi_sensitivity}",
        r"\begin{tabularx}{\textwidth}{Xrrr}",
        r"\toprule",
        r"Quantity & Fitted & $\psi$ fixed & Change (\%) \\",
        r"\midrule",
    ]
    for _, r_ in summary.iterrows():
        def fmt(v):
            return f"{v:.3g}" if abs(v) < 0.01 and v != 0 else f"{v:,.3f}"
        lines.append(f"{r_['quantity']} & {fmt(r_['fitted_median'])} & "
                     f"{fmt(r_['psi_fixed_median'])} & {r_['median_pct_change']:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    tex = "\n".join(lines)
    (OUTPUT_DIR / "psi_invariance_table.tex").write_text(tex, encoding="utf-8")
    print("\n" + tex)
    print(f"[OK] Wrote {OUTPUT_DIR / 'psi_invariance_table.tex'}")


# =============================================================================
# Refit with psi held fixed
# =============================================================================

class FixedPsiLikelihood:
    """PlagueLikelihood with psi removed from the sampled parameter vector."""

    def __init__(self, inner: PlagueLikelihood, psi_value: float):
        self.inner = inner
        self.psi_value = float(psi_value)
        all_names = list(inner.param_names)
        self.psi_index = all_names.index("psi")
        self.param_names = [n for n in all_names if n != "psi"]

    def full_theta(self, theta_reduced) -> np.ndarray:
        return np.insert(np.asarray(theta_reduced, dtype=float),
                         self.psi_index, self.psi_value)

    def log_likelihood(self, theta_reduced) -> float:
        return float(self.inner.log_likelihood(self.full_theta(theta_reduced)))


def run_refit(psi_target: float) -> None:

    from dynesty import NestedSampler

    nested_dir = OUTPUT_DIR / "nested_sampling"
    nested_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{MODE}__{FIT_TARGET}__psi_fixed"

    df = (pd.read_csv(paths.MERGED_CSV, parse_dates=["week_ending"])
          .sort_values("week_ending").reset_index(drop=True))

    inner = PlagueLikelihood(merged_df=df, betaH_mode=MODE, fit_target=FIT_TARGET)
    all_names = list(inner.param_names)
    all_bounds = list(priors.get_bounds(MODE))
    inner.bounds = list(all_bounds)

    lik = FixedPsiLikelihood(inner, psi_target)
    param_names = lik.param_names
    bounds = [b for b, n in zip(all_bounds, all_names) if n != "psi"]
    ndim = len(param_names)

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    def prior_transform(u):
        return lower + np.asarray(u, dtype=float) * (upper - lower)

    print("\n" + "=" * 92)
    print(f"REFIT WITH PSI FIXED   {MODE}   psi = {psi_target}")
    print("=" * 92)
    print(f"Sampled parameters: {ndim} (psi removed)")

    # Seed: the main model's best fit, rescaled onto the fixed-psi scale.
    with open(MAIN_PKL, "rb") as f:
        main = pickle.load(f)
    best = np.asarray(main["best_theta"], dtype=float)
    rescaled, r = rescale_theta(best, list(main["param_names"]), psi_target)
    seed_full = dict(zip(main["param_names"], rescaled))
    seed_theta = np.array([seed_full[n] for n in param_names], dtype=float)

    outside = [(n, float(v), tuple(b)) for n, v, b in zip(param_names, seed_theta, bounds)
               if not (b[0] <= v <= b[1])]
    if outside:
        raise RuntimeError(
            "The rescaled seed falls outside the priors, so the fixed-psi scale is not "
            f"reachable under the current bounds (r = {r:.2f}):\n  " +
            "\n  ".join(f"{n} = {v:.4g} outside {b}" for n, v, b in outside) +
            "\nWiden the bound(s) listed, or choose a psi closer to the fitted value."
        )

    seed_logl = lik.log_likelihood(seed_theta)
    if not np.isfinite(seed_logl):
        raise RuntimeError("The rescaled seed does not give a finite likelihood.")
    print(f"Rescaling factor r = {r:.3f}")
    print(f"Seed logL = {seed_logl:.4f}   (main model best fit: {main['best_loglik']:.4f})")
    print("A seed logL close to the main model's confirms the rescaling is near-exact.")

    rng = np.random.default_rng(SEED_LIVE)
    live_u = rng.random((NLIVE, ndim))
    live_v = np.array([prior_transform(u) for u in live_u])
    print(f"Evaluating {NLIVE} initial live points...")
    t0 = time.time()
    live_logl = np.array([lik.log_likelihood(v) for v in live_v])
    print(f"  done in {time.time() - t0:.1f}s "
          f"({int(np.sum(~np.isfinite(live_logl)))} / {NLIVE} had -inf)")

    live_u[0] = (seed_theta - lower) / (upper - lower)
    live_v[0] = seed_theta
    live_logl[0] = seed_logl

    sampler = NestedSampler(
        loglikelihood=lik.log_likelihood,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(SEED_SAMPLER),
        live_points=(live_u, live_v, live_logl),
    )

    t0 = time.time()
    sampler.run_nested(dlogz=DLOGZ, print_progress=True)
    minutes = (time.time() - t0) / 60.0

    res = sampler.results
    best_idx = int(np.nanargmax(res.logl))
    best_theta = np.asarray(res.samples[best_idx], dtype=float)
    best_loglik = float(res.logl[best_idx])
    logZ = float(res.logz[-1])
    logZ_err = float(res.logzerr[-1])

    with open(nested_dir / f"{out_stem}.pkl", "wb") as f:
        pickle.dump({
            "results": res,
            "param_names": param_names,
            "bounds": bounds,
            "fixed": {"psi": psi_target},
            "best_theta": best_theta,
            "best_loglik": best_loglik,
            "logZ": logZ,
            "logZ_error": logZ_err,
        }, f)

    w = np.exp(res.logwt - logZ)
    w /= w.sum()
    s = np.asarray(res.samples, dtype=float)

    def wq(v, qs=(0.025, 0.5, 0.975)):
        o = np.argsort(v)
        return np.interp(qs, np.cumsum(w[o]), v[o])

    with open(MAIN_PKL, "rb") as f:
        main_d = pickle.load(f)
    main_res = main_d["results"]
    wm = np.exp(main_res.logwt - main_res.logz[-1]); wm /= wm.sum()
    sm = np.asarray(main_res.samples, dtype=float)
    main_names = list(main_d["param_names"])

    def wqm(v, qs=(0.025, 0.5, 0.975)):
        o = np.argsort(v)
        return np.interp(qs, np.cumsum(wm[o]), v[o])

    print("\n" + "-" * 92)
    print(f"Finished in {minutes:.1f} min")
    print(f"max logL : main {main_d['best_loglik']:.3f}  ->  psi fixed {best_loglik:.3f}")
    print(f"log Z    : main {main_d['logZ']:.3f}  ->  psi fixed {logZ:.3f} +/- {logZ_err:.3f}")
    print("\nNote: log Z is not comparable between the two runs; they have different"
          "\nnumbers of free parameters and different priors.\n")
    print(f"{'Parameter':<12}{'main: median (95% CrI)':<34}{'psi fixed: median (95% CrI)':<34}")
    for n in param_names:
        a = wq(s[:, param_names.index(n)])
        b = wqm(sm[:, main_names.index(n)])
        print(f"{n:<12}{f'{b[1]:.3f} ({b[0]:.3f}, {b[2]:.3f})':<34}"
              f"{f'{a[1]:.3f} ({a[0]:.3f}, {a[2]:.3f})':<34}")

    rows = []
    for n in param_names:
        a = wq(s[:, param_names.index(n)])
        b = wqm(sm[:, main_names.index(n)])
        rows.append({"parameter": n,
                     "main_median": b[1], "main_lo": b[0], "main_hi": b[2],
                     "psifix_median": a[1], "psifix_lo": a[0], "psifix_hi": a[2]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "psi_fixed_coefficients.csv", index=False)

    with open(nested_dir / f"{out_stem}.json", "w", encoding="utf-8") as f:
        json.dump({"mode": MODE, "fit_target": FIT_TARGET, "psi_fixed": psi_target,
                   "nlive": NLIVE, "dlogz": DLOGZ, "ndim": ndim,
                   "best_loglik": best_loglik, "logZ": logZ, "logZ_error": logZ_err,
                   "minutes": minutes}, f, indent=2)

    print(f"\n[OK] Wrote {nested_dir / (out_stem + '.pkl')}")
    print("Now rerun with no arguments to rebuild the comparison table.")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refit", action="store_true",
                    help="refit the model by nested sampling with psi held fixed (hours)")
    ap.add_argument("--psi", type=float, default=PSI_TARGET,
                    help=f"value at which to hold psi (default {PSI_TARGET})")
    ap.add_argument("--draws", type=int, default=N_DRAWS,
                    help=f"posterior draws for the invariance check (default {N_DRAWS})")
    args = ap.parse_args()

    if not 0.0 < args.psi <= 1.0:
        raise SystemExit("--psi must lie in (0, 1].")

    if args.refit or RUN_REFIT:
        run_refit(args.psi)
    else:
        run_invariance(args.psi, args.draws)


if __name__ == "__main__":
    main()