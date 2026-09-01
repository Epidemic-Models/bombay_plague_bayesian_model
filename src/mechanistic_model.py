# =============================================================================
# mechanistic_model.py
# =============================================================================
"""
Mechanistic SEIR delay model for the Bombay plague analysis.

This module defines the deterministic transmission model used by the Bayesian
inference framework. It separates plague transmission into two components:

    1. A climate-dependent reservoir/spillover component, beta_R(t)
    2. A human-associated amplification component, beta_H(t)

The total force of infection is

    lambda(t) = beta_R(t) + beta_H(t) * I(t) / N

and new infections occur at rate

    C(t) = lambda(t) * S(t).

The epidemic dynamics are represented by a delay-SEIR system:

    dS/dt = -C(t)
    dE/dt =  C(t) - C(t - tau)
    dI/dt =  C(t - tau) - gamma * I(t)
    dR/dt =  gamma * I(t)

where tau is a fixed incubation delay and gamma is the removal/recovery rate.
All internal model times are expressed in weeks.


RESERVOIR / SPILLOVER COMPONENT
-------------------------------

The reservoir-associated contribution is climate dependent:

    beta_R(t) = exp(eta_R(t))

with

    eta_R(t) =
        a0
        + aT  * zT(t)
        + aTT * zT(t)^2
        + ae  * ze(t)
        + aR  * zR(t),

where

    zT = standardized temperature covariate
    ze = standardized vapour-pressure covariate
    zR = standardized rainfall covariate.

The climate covariates supplied to this module are already processed,
lagged, aggregated to weekly resolution, and standardized by climate.py.
The mechanistic model therefore does not itself construct or estimate
the climate lags.


HUMAN-ASSOCIATED AMPLIFICATION COMPONENT
-----------------------------------------

Four alternative formulations of beta_H(t) are implemented. These represent
competing hypotheses about temporal variation in human-associated
transmission.

1. Constant

       beta_H(t) = betaH0

   Human-associated transmission is assumed not to vary through time.


2. Sigmoid

       beta_H(t) =
           betaH0 * exp(theta * sigmoid(t; t_star, kappa))

   where

       sigmoid(t; t_star, kappa)
           = 1 / (1 + exp[-kappa * (t - t_star)]).

   This formulation allows human-associated transmission to change
   systematically through epidemic time independently of the climate
   covariates.


3. Humidity-only

       beta_H(t) =
           betaH0 * exp(be_H * ze(t))

   Human-associated transmission varies with the standardized
   vapour-pressure covariate only.


4. All-climate

       beta_H(t) =
           betaH0 * exp(
               bT_H * zT(t)
               + be_H * ze(t)
               + bR_H * zR(t)
           )

   Human-associated transmission varies with temperature,
   vapour pressure, and rainfall.


MODEL IMPLEMENTATION
--------------------

The delay-SEIR equations are solved using explicit Euler integration with
multiple internal time steps per week.

The climate covariates are weekly quantities. Consequently, their values are
held piecewise constant over the internal integration steps belonging to a
given week.

A constant pre-epidemic incidence history, C_seed, can be supplied for
times t < 0. This provides the incidence history required to evaluate the
delayed term C(t - tau) near the beginning of the simulation.

The simulator returns the complete internal-time trajectory, including

    S
    E
    I
    R
    C
    lambda
    beta_R
    beta_H

as well as the corresponding time variables.


WEEKLY AGGREGATION
------------------

The function weekly_sum() converts internal model output to weekly totals.

Because C is stored as an incidence rate per week, weekly totals are obtained
by integrating over each internal time step:

    new_exposed = C * dt.


REPORTING
---------

The helper function apply_reporting() optionally applies a stepwise reporting
fraction to weekly expected outcomes. It is retained here as a model utility
but is separate from the transmission dynamics themselves.


IMPORTANT
---------

This module defines the deterministic mechanistic model only.

It does NOT:

    - define Bayesian priors,
    - calculate the observation likelihood,
    - perform Bayesian sampling,
    - estimate climate lags,
    - perform model comparison,
    - or generate posterior summaries.

Those components are implemented separately so that the same mechanistic
model can be used by each Bayesian model formulation.

Supported beta_H modes:

    "constant"
    "sigmoid"
    "humidity_only"
    "all_climate"
"""

import numpy as np
import pandas as pd


def eta_H(t, zT, ze, zR, betaH0, betaH_mode, params):
    """
    Human-associated transmission contribution on the log scale.

    Modes
    -----
    constant:
        beta_H(t) = betaH0

    sigmoid:
        beta_H(t) = betaH0 * exp(theta * sigmoid(t, t_star, kappa))

    humidity_only:
        beta_H(t) = betaH0 * exp(be_H * ze)

    all_climate:
        beta_H(t) = betaH0 * exp(
            bT_H * zT + be_H * ze + bR_H * zR
        )
    """

    if betaH_mode == "constant":
        return np.log(betaH0)

    if betaH_mode == "sigmoid":
        theta = float(params["theta"])
        t_star = float(params["t_star"])
        kappa = float(params["kappa"])

        s = 1.0 / (1.0 + np.exp(-kappa * (t - t_star)))
        return np.log(betaH0) + theta * s

    if betaH_mode == "humidity_only":
        be_H = float(params["be_H"])
        return np.log(betaH0) + be_H * ze

    if betaH_mode == "rainfall_only":
        bR_H = float(params["bR_H"])
        return np.log(betaH0) + bR_H * zR

    if betaH_mode == "all_climate":
        bT_H = float(params["bT_H"])
        be_H = float(params["be_H"])
        bR_H = float(params["bR_H"])
        return np.log(betaH0) + bT_H * zT + be_H * ze + bR_H * zR

    raise ValueError(
        f"Unknown betaH_mode={betaH_mode!r}. "
        "Use 'constant', 'sigmoid', 'humidity_only', "
        "or 'all_climate'."
    )


def beta_H_from_mode(
    t,
    zT,
    ze,
    zR,
    betaH0,
    betaH_mode,
    params,
):
    return np.exp(
        eta_H(
            t=t,
            zT=zT,
            ze=ze,
            zR=zR,
            betaH0=betaH0,
            betaH_mode=betaH_mode,
            params=params,
        )
    )


def eta_R(zT, ze, zR, a0, aT, aTT, ae, aR):
    """
    Linear predictor for reservoir/spillover transmission on log scale.
    """
    return a0 + aT * zT + aTT * (zT ** 2) + ae * ze + aR * zR


def beta_R_from_climate(zT, ze, zR, a0, aT, aTT, ae, aR):
    """
    Reservoir/spillover transmission contribution:
        beta_R(t) = exp(eta_R(...))
    """
    return np.exp(eta_R(zT, ze, zR, a0, aT, aTT, ae, aR))


def simulate_seir_dde(
    zT: np.ndarray,
    ze: np.ndarray,
    zR: np.ndarray,
    N: float,
    params: dict,
    substeps_per_week: int,
    tau0_days: int,
    I0: float = 0.0,
    E0: float = 0.0,
    R0: float = 0.0,
    C_seed: float = 0.0,
    betaH_mode: str = "constant",
    start_date=None,
    # -----------------------------------------------------------------
    # Counterfactual controls (default keep original behaviour)
    # -----------------------------------------------------------------
    human_multiplier: float = 1.0,
    spillover_multiplier: float = 1.0,
    spillover_cutoff_week: float | None = None,
):
    """
    Simulate the SEIR model with a fixed incubation delay using explicit Euler
    on a fine internal time grid.

    Model:
        dS/dt = -C(t)
        dE/dt =  C(t) - C(t - tau)
        dI/dt =  C(t - tau) - gamma I
        dR/dt =  gamma I

    with
        C(t) = lambda(t) * S(t)
        lambda(t) = beta_R(t) + beta_H(t) * I(t) / N

    Time units inside the model are WEEKS.

    IMPORTANT:
    zT, ze, zR are assumed to be WEEKLY covariates aligned to the observation weeks.
    If the solver uses substeps within a week, the climate is held piecewise constant
    within each week.

    Counterfactual controls
    -----------------------
    human_multiplier : float
        Multiplies beta_H(t) at all times.
        - 1.0 : baseline (fitted model)
        - 0.0 : no human transmission
        - 0.5 : half-strength human transmission, etc.

    spillover_multiplier : float
        Multiplies beta_R(t) AFTER spillover_cutoff_week (if provided).
        Often used with 0.0 to extinguish reservoir/spillover after an initial
        seeding period.

    spillover_cutoff_week : float | None
        Time in model weeks at which to start applying spillover_multiplier.
        If None, spillover_multiplier is not used and beta_R(t) is baseline.
    """
    zT = np.asarray(zT, dtype=float)
    ze = np.asarray(ze, dtype=float)
    zR = np.asarray(zR, dtype=float)

    if not (len(zT) == len(ze) == len(zR)):
        raise ValueError("zT, ze, and zR must have the same length.")

    if len(zT) < 1:
        raise ValueError("Climate arrays must have at least one value.")

    if N <= 0:
        raise ValueError("N must be positive.")

    if substeps_per_week <= 0:
        raise ValueError("substeps_per_week must be a positive integer.")

    if tau0_days < 0:
        raise ValueError("tau0_days must be nonnegative.")

    if I0 < 0 or E0 < 0 or R0 < 0 or C_seed < 0:
        raise ValueError("Initial conditions and C_seed must be nonnegative.")

    if I0 + E0 + R0 > N:
        raise ValueError("Initial conditions exceed total population N.")

    required = ["gamma", "a0", "aT", "aTT", "ae", "aR", "betaH0"]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    # beta_H mode-specific parameter checks
    if betaH_mode == "sigmoid":
        needed = ["theta", "t_star", "kappa"]
        missing = [key for key in needed if key not in params]
        if missing:
            raise ValueError(
                f"Missing parameters {missing} for betaH_mode='sigmoid'."
            )

    if betaH_mode == "humidity_only":
        if "be_H" not in params:
            raise ValueError(
                "Missing parameter 'be_H' for betaH_mode='humidity_only'."
            )

    if betaH_mode == "rainfall_only":
        if "bR_H" not in params:
            raise ValueError(
                "Missing parameter 'bR_H' for betaH_mode='rainfall_only'."
            )

    if betaH_mode == "all_climate":
        needed = ["bT_H", "be_H", "bR_H"]
        missing = [key for key in needed if key not in params]
        if missing:
            raise ValueError(
                f"Missing parameters {missing} for betaH_mode='all_climate'."
            )

    gamma = float(params["gamma"])

    a0 = float(params["a0"])
    aT = float(params["aT"])
    aTT = float(params["aTT"])
    ae = float(params["ae"])
    aR = float(params["aR"])

    betaH0 = float(params["betaH0"])

    dt = 1.0 / substeps_per_week
    tau = tau0_days / 7.0

    history_steps = int(np.ceil(tau / dt))

    # zT etc are WEEKLY, so the simulation horizon is based on number of weeks
    n_weeks = len(zT)
    t_max_weeks = n_weeks - 1
    n_main_steps = int(round(t_max_weeks / dt))

    total_steps = history_steps + n_main_steps + 1
    time_weeks = (np.arange(total_steps) - history_steps) * dt
    time_days = 7.0 * time_weeks

    i0 = history_steps

    S = np.zeros(total_steps, dtype=float)
    E = np.zeros(total_steps, dtype=float)
    I = np.zeros(total_steps, dtype=float)
    R = np.zeros(total_steps, dtype=float)

    C = np.zeros(total_steps, dtype=float)          # incidence rate per week
    lam = np.zeros(total_steps, dtype=float)
    bR = np.zeros(total_steps, dtype=float)
    bH = np.zeros(total_steps, dtype=float)

    S[i0] = N - I0 - E0 - R0
    E[i0] = E0
    I[i0] = I0
    R[i0] = R0

    # Constant state history for t < 0
    S[:i0] = S[i0]
    E[:i0] = E[i0]
    I[:i0] = I[i0]
    R[:i0] = R[i0]

    # Nonzero incidence history for t < 0 if desired
    C[:i0] = C_seed

    for i in range(i0 + 1, total_steps):
        t_prev = time_weeks[i - 1]

        # Weekly climate held constant within each week
        week_idx = int(np.floor(max(0.0, t_prev)))
        week_idx = min(week_idx, n_weeks - 1)

        bR_now = beta_R_from_climate(
            zT[week_idx], ze[week_idx], zR[week_idx],
            a0, aT, aTT, ae, aR
        )

        bH_now = beta_H_from_mode(
            t=t_prev,
            zT=zT[week_idx],
            ze=ze[week_idx],
            zR=zR[week_idx],
            betaH0=betaH0,
            betaH_mode=betaH_mode,
            params=params,
        )

        # ------------------------------------------------------------
        # Counterfactual switches
        # ------------------------------------------------------------

        # 1) Modify human transmission
        #    human_multiplier = 0 → no human transmission
        bH_now *= human_multiplier

        # 2) Modify spillover after a cutoff time (if provided)
        #    e.g. spillover_multiplier = 0, spillover_cutoff_week = 4
        #    → no reservoir transmission after week 4.
        if spillover_cutoff_week is not None and t_prev >= spillover_cutoff_week:
            bR_now *= spillover_multiplier

        lam_now = bR_now + bH_now * I[i - 1] / N

        # incidence rate at current step
        C_now = lam_now * S[i - 1]

        # positivity safeguard: cannot move more than S in one Euler step
        C_now = min(C_now, S[i - 1] / dt)

        # delayed incidence C(t - tau), interpolated on the internal grid
        lag_pos = (i - 1) - tau / dt
        lag_lo = int(np.floor(lag_pos))
        lag_hi = int(np.ceil(lag_pos))

        if lag_lo < 0:
            C_lag = C_seed
        elif lag_lo == lag_hi:
            C_lag = C[lag_lo]
        else:
            w_hi = lag_pos - lag_lo
            w_lo = 1.0 - w_hi
            C_lag = w_lo * C[lag_lo] + w_hi * C[lag_hi]

        dS = -C_now
        dE = C_now - C_lag
        dI = C_lag - gamma * I[i - 1]
        dR = gamma * I[i - 1]

        S[i] = S[i - 1] + dt * dS
        E[i] = E[i - 1] + dt * dE
        I[i] = I[i - 1] + dt * dI
        R[i] = R[i - 1] + dt * dR

        S[i] = max(S[i], 0.0)
        E[i] = max(E[i], 0.0)
        I[i] = max(I[i], 0.0)
        R[i] = max(R[i], 0.0)

        C[i] = C_now
        lam[i] = lam_now
        bR[i] = bR_now
        bH[i] = bH_now

    out = pd.DataFrame({
        "time_weeks": time_weeks,
        "time_days": time_days,
        "dt_weeks": np.full(total_steps, dt, dtype=float),
        "S": S,
        "E": E,
        "I": I,
        "R": R,
        "C": C,                         # rate per week
        "new_exposed": C * dt,          # actual flow over one internal step
        "lambda": lam,
        "beta_R": bR,
        "beta_H": bH,
    })

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        out["date"] = start_date + pd.to_timedelta(out["time_days"], unit="D")

    return out


def weekly_sum(sim_df: pd.DataFrame, col: str = "C", week_ending: str = "W-FRI"):
    """
    Aggregate a simulated quantity to weekly sums.

    If `col` is a rate-like flow stored on the internal grid (for example C),
    use `new_exposed` instead when you want weekly totals. This function will
    automatically integrate `C` over dt if `dt_weeks` is available.
    """
    if col not in sim_df.columns:
        raise ValueError(f"Column '{col}' not found in simulation output.")

    df = sim_df.copy()

    # If we are aggregating C, convert from rate to step flow
    if col == "C" and "dt_weeks" in df.columns:
        agg_values = df["C"] * df["dt_weeks"]
        out_col = "C_week"
    else:
        agg_values = df[col]
        out_col = f"{col}_week"

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["_agg"] = agg_values
        wk = (
            df.set_index("date")["_agg"]
            .resample(week_ending)
            .sum()
            .reset_index()
            .rename(columns={"_agg": out_col, "date": "week_ending"})
        )
        return wk

    if "time_days" not in df.columns:
        raise ValueError("Simulation output must contain either 'date' or 'time_days'.")

    df = df[df["time_days"] >= 0].copy()
    df["_agg"] = agg_values[df["time_days"] >= 0]
    df["week_index"] = (df["time_days"] // 7).astype(int)

    wk = (
        df.groupby("week_index", as_index=False)["_agg"]
        .sum()
        .rename(columns={"_agg": out_col})
    )
    return wk


def apply_reporting(
    mu_week: np.ndarray,
    week_dates,
    rho1: float,
    rho2: float,
    break_date,
):
    """
    Apply a stepwise multiplicative reporting fraction:
        rho_w = rho1 before break_date
        rho_w = rho2 on/after break_date
    """
    mu_week = np.asarray(mu_week, dtype=float)
    week_dates = pd.to_datetime(week_dates)
    break_date = pd.Timestamp(break_date)

    rho = np.where(week_dates < break_date, rho1, rho2)
    return rho * mu_week, rho
