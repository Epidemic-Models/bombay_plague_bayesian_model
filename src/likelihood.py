# =============================================================================
# likelihood.py
# =============================================================================
"""
Bayesian likelihood layer: wires mechanistic_model.py (simulator),
observation.py (infection -> death mapping + NB likelihood), and
priors.py (bounds / prior_transform) into the two functions a sampler
actually needs:

    prior_transform(u)   -- for dynesty (nested sampling)
    log_likelihood(theta) -- for both dynesty and emcee
    log_posterior(theta)  -- log_prior + log_likelihood, for emcee

Nothing here touches the physics or the observation model -- this file
only handles bookkeeping: turning a flat parameter vector into the
`params` dict simulate_seir_dde expects, running the simulation,
converting infections to expected deaths, and returning one number.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import priors
from mechanistic_model import simulate_seir_dde, weekly_sum
from observation import make_delay_pmf_weeks, deaths_from_infections_fast, negbin_loglik_vec


SIM_PARAM_NAMES = ["gamma", "a0", "aT", "aTT", "ae", "aR", "betaH0"]


def build_sim_params(sample: dict, betaH_mode: str) -> dict:
    """
    Pull the params dict simulate_seir_dde needs out of a flat sample
    dict, including whichever extra beta_H covariate coefficients this
    formulation uses.
    """
    params = {name: float(sample[name]) for name in SIM_PARAM_NAMES}
    for extra in priors.MODEL_SPECS[betaH_mode]["extra_params"]:
        params[extra] = float(sample[extra])
    return params


class PlagueLikelihood:
    """
    Bundles a merged (weekly target + weekly climate) DataFrame and
    simulator settings, so a sampler only ever needs to pass theta.

    merged_df must have columns: week_ending, zT, ze, zR, and either
    plague_deaths_reported_fit or plague_deaths_proxy (whichever you
    pass as fit_target).

    Example
    -------
    >>> lik = PlagueLikelihood(merged_df, betaH_mode="humidity_only")
    >>> lik.log_likelihood(lik.example_theta())
    """
    def __init__(
        self,
        merged_df: pd.DataFrame,
        betaH_mode: str,
        substeps_per_week: int = 7,
        tau0_days: int = 7,
        I0: float = 0.0,
        E0: float = 0.0,
        R0: float = 0.0,
        delay_kind: str = "gamma",
        delay_tau_weeks: int = 2,
        delay_max_weeks: int = 12,
        delay_shape: float = 2.0,
        delay_scale: float = 0.6,
        delay_shift_weeks: float = 1.0,
        fit_target: str = "plague_deaths_reported_fit",
    ):
        if betaH_mode not in priors.MODEL_SPECS:
            raise ValueError(f"Unknown betaH_mode={betaH_mode!r}. Use one of {list(priors.MODEL_SPECS)}.")

        self.betaH_mode = betaH_mode
        self.param_names = priors.get_param_names(betaH_mode)
        self.bounds = priors.get_bounds(betaH_mode)
        self.ndim = len(self.param_names)

        self.merged_df = merged_df.reset_index(drop=True)
        self.substeps_per_week = substeps_per_week
        self.tau0_days = tau0_days
        self.I0, self.E0, self.R0 = I0, E0, R0

        self.f_delay = make_delay_pmf_weeks(
            kind=delay_kind, tau_weeks=delay_tau_weeks, max_weeks=delay_max_weeks,
            shape=delay_shape, scale=delay_scale, shift_weeks=delay_shift_weeks,
        )

        if fit_target not in self.merged_df.columns:
            raise ValueError(
                f"fit_target={fit_target!r} not found in merged_df columns: "
                f"{list(self.merged_df.columns)}"
            )
        self.fit_target = fit_target
        self.y_obs = self.merged_df[fit_target].to_numpy(dtype=float)

        self.zT = self.merged_df["zT"].to_numpy(dtype=float)
        self.ze = self.merged_df["ze"].to_numpy(dtype=float)
        self.zR = self.merged_df["zR"].to_numpy(dtype=float)
        self.start_date = str(pd.Timestamp(self.merged_df["week_ending"].iloc[0]).date())

    # -- convenience ---------------------------------------------------
    def example_theta(self) -> np.ndarray:
        """Midpoint of the prior box."""
        return np.array([0.5 * (lo + hi) for lo, hi in self.bounds])

    # -- prior -----------------------------------------------------------
    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        return priors.prior_transform(u, self.betaH_mode)

    def log_prior(self, theta: np.ndarray) -> float:
        return priors.log_prior(theta, self.betaH_mode)

    # -- likelihood --------------------------------------------------------
    def log_likelihood(self, theta: np.ndarray) -> float:
        for val, (lo, hi) in zip(theta, self.bounds):
            if not (lo <= val <= hi):
                return -np.inf

        sample = priors.params_dict_from_theta(theta, self.betaH_mode)
        sim_params = build_sim_params(sample, self.betaH_mode)

        try:
            sim = simulate_seir_dde(
                zT=self.zT, ze=self.ze, zR=self.zR,
                N=sample["N0"],
                params=sim_params,
                substeps_per_week=self.substeps_per_week,
                tau0_days=self.tau0_days,
                I0=self.I0, E0=self.E0, R0=self.R0,
                C_seed=sample["C_seed"],
                betaH_mode=self.betaH_mode,
                start_date=self.start_date,
            )
        except ValueError:
            return -np.inf

        weekly_inf = weekly_sum(sim, col="C", week_ending="W-FRI")
        weekly_inf["week_ending"] = pd.to_datetime(weekly_inf["week_ending"])

        compare = pd.merge(
            self.merged_df[["week_ending"]], weekly_inf, on="week_ending", how="left",
        )
        c_week_col = "C_week" if "C_week" in compare.columns else [
            c for c in compare.columns if c.startswith("C_")
        ][0]
        C_week = compare[c_week_col].fillna(0.0).to_numpy(dtype=float)

        if len(C_week) != len(self.y_obs):
            return -np.inf

        mu_deaths = deaths_from_infections_fast(C_week=C_week, psi=sample["psi"], f_delay=self.f_delay)
        mu_deaths = np.maximum(mu_deaths, 1e-9)

        ll = negbin_loglik_vec(y=self.y_obs, mu=mu_deaths, k=sample["nb_kappa"])
        return ll if np.isfinite(ll) else -np.inf

    def log_posterior(self, theta: np.ndarray) -> float:
        """log_prior + log_likelihood, for emcee-style samplers."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    # -- posterior predictive -----------------------------------------------
    def predict_weekly_deaths(self, theta: np.ndarray) -> np.ndarray:
        """Deterministic mu_deaths(t) for one parameter vector (no NB noise)."""
        sample = priors.params_dict_from_theta(theta, self.betaH_mode)
        sim_params = build_sim_params(sample, self.betaH_mode)

        sim = simulate_seir_dde(
            zT=self.zT, ze=self.ze, zR=self.zR,
            N=sample["N0"], params=sim_params,
            substeps_per_week=self.substeps_per_week, tau0_days=self.tau0_days,
            I0=self.I0, E0=self.E0, R0=self.R0,
            C_seed=sample["C_seed"], betaH_mode=self.betaH_mode,
            start_date=self.start_date,
        )
        weekly_inf = weekly_sum(sim, col="C", week_ending="W-FRI")
        weekly_inf["week_ending"] = pd.to_datetime(weekly_inf["week_ending"])
        compare = pd.merge(self.merged_df[["week_ending"]], weekly_inf, on="week_ending", how="left")
        c_week_col = "C_week" if "C_week" in compare.columns else [
            c for c in compare.columns if c.startswith("C_")
        ][0]
        C_week = compare[c_week_col].fillna(0.0).to_numpy(dtype=float)
        mu_deaths = deaths_from_infections_fast(C_week=C_week, psi=sample["psi"], f_delay=self.f_delay)
        return np.maximum(mu_deaths, 1e-9)