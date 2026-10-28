# observation.py
# ------------------------------------------------------------
# Observation / measurement model for weekly death data.
#
# Inputs:
#   C_week[w]  = model-implied infections (incidence) in week w
#
# Outputs:
#   muD[w]     = expected deaths in week w
#
# We map infections -> deaths via:
#
#   muD[w] = psi * sum_{j=0..Jmax} C_week[w-j] * f_delay[j]
#
# where f_delay[j] is a discrete delay PMF for the
# infection-to-death lag.
#
# Then we evaluate a Negative Binomial log-likelihood:
#
#   y_w ~ NB(mean=muD[w], dispersion=kappa)
#
# IMPORTANT
# ---------
# Three delay-kernel options are supported:
#
#   1. "fixed"
#      All probability mass is placed at one discrete lag tau_weeks.
#
#   2. "gamma"
#      ORIGINAL Bayesian implementation used in the final_v2 fits.
#      The Gamma PDF is evaluated at weekly midpoints j + 0.5 and
#      normalized across j = 0,...,max_weeks.
#
#   3. "gamma_cdf"
#      Sensitivity-analysis implementation.
#      Gamma probability mass is integrated within each weekly interval:
#
#          P(j <= T < j+1)
#          = F(j+1) - F(j)
#
#      and then normalized across the represented lag range.
#
# The existing "gamma" behaviour is deliberately preserved so that the
# original final_v2 Bayesian analyses remain exactly reproducible.
# ------------------------------------------------------------

from __future__ import annotations

from math import lgamma

import numpy as np
from scipy.stats import gamma as gamma_dist


# =============================================================================
# Delay distribution (weeks)
# =============================================================================

def make_delay_pmf_weeks(
        kind: str = "fixed",
        tau_weeks: int = 2,
        max_weeks: int = 12,
        shape: float = 2.0,
        scale: float = 1.0,
        shift_weeks: float = 1.0,
) -> np.ndarray:
    """
    Build a discrete infection-to-death delay PMF
    f_delay[j] for j = 0,...,max_weeks.

    Parameters
    ----------
    kind : str
        Delay-kernel construction method.

        "fixed"
            All probability mass is placed at discrete lag
            ``tau_weeks``.

        "gamma"
            ORIGINAL midpoint-based Gamma discretization used by the
            final_v2 Bayesian fits.

            The Gamma probability density is evaluated at the weekly
            midpoints

                j + 0.5

            for j = 0,...,max_weeks, and the resulting values are
            normalized to sum to one.

        "gamma_cdf"
            Alternative Gamma discretization for sensitivity analysis.

            The probability assigned to lag j is the integrated Gamma
            probability over the weekly interval [j, j+1):

                F(j+1) - F(j),

            where F is the Gamma cumulative distribution function.
            The resulting probabilities are normalized over
            j = 0,...,max_weeks.

        "gamma_shifted"
            Gamma CDF-bin discretization with a hard lower floor at
            ``shift_weeks``.

            Intended for sensitivity analysis on the "no death before
            the person exits the incubation/exposed compartment"
            assumption: the delay distribution is a Gamma shifted to
            the right by ``shift_weeks``, so it is impossible to
            observe a death in any bin before that floor is reached.
            For bin j, the integrated probability is

                F(j + 1 - shift_weeks) - F(j - shift_weeks),

            where F is the Gamma CDF, clipped at 0 and normalized
            over j = 0,...,max_weeks.

    tau_weeks : int
        Discrete lag used only when ``kind="fixed"``.

        This argument is ignored for ``kind="gamma"``,
        ``kind="gamma_cdf"``, and ``kind="gamma_shifted"``.

    max_weeks : int
        Largest discrete lag represented in the delay kernel.

        The returned PMF therefore has length ``max_weeks + 1``.

    shape : float
        Gamma shape parameter.

        Used only for ``kind="gamma"``, ``kind="gamma_cdf"``, and
        ``kind="gamma_shifted"``.

    scale : float
        Gamma scale parameter, in weeks.

        Used only for ``kind="gamma"``, ``kind="gamma_cdf"``, and
        ``kind="gamma_shifted"``.

    shift_weeks : float
        Rightward shift applied to the Gamma distribution before
        CDF-binning, in weeks.

        Used only for ``kind="gamma_shifted"``. Ignored otherwise.

    Returns
    -------
    np.ndarray
        One-dimensional probability mass function of length
        ``max_weeks + 1`` whose elements sum to one.

    Notes
    -----
    The ``"gamma"`` option must not be changed if exact reproduction
    of the existing final_v2 Bayesian fits is required.

    The ``"gamma_cdf"`` option is provided specifically for assessing
    sensitivity to the numerical discretization of the same underlying
    continuous Gamma delay distribution.

    The ``"gamma_shifted"`` option is provided specifically for
    assessing sensitivity to allowing nonzero death probability before
    the incubation period has elapsed.
    """

    # -------------------------------------------------------------------------
    # General validation
    # -------------------------------------------------------------------------

    if max_weeks < 0:
        raise ValueError(
            "max_weeks must be >= 0"
        )

    max_weeks = int(
        max_weeks
    )

    j = np.arange(
        max_weeks + 1,
        dtype=float,
    )

    # =========================================================================
    # 1. Fixed discrete delay
    # =========================================================================

    if kind == "fixed":

        tau_weeks = int(
            tau_weeks
        )

        if not (
                0
                <= tau_weeks
                <= max_weeks
        ):
            raise ValueError(
                "tau_weeks must be within "
                "0..max_weeks"
            )

        f = np.zeros(
            max_weeks + 1,
            dtype=float,
        )

        f[
            tau_weeks
        ] = 1.0

        return f

    # =========================================================================
    # Gamma parameter validation
    # =========================================================================

    if kind in {
        "gamma",
        "gamma_cdf",
        "gamma_shifted",
    }:

        if (
                shape <= 0
                or
                scale <= 0
        ):
            raise ValueError(
                "shape and scale must be > 0 "
                "for Gamma delay kernels"
            )

    # =========================================================================
    # 2. ORIGINAL Gamma midpoint discretization
    #
    # IMPORTANT:
    # This is the implementation used in the final_v2 Bayesian fits.
    # Do not modify this branch if exact reproducibility is required.
    # =========================================================================

    if kind == "gamma":

        # Evaluate the continuous Gamma PDF at the midpoint of each
        # represented weekly lag interval.
        #
        # j = 0 -> x = 0.5 weeks
        # j = 1 -> x = 1.5 weeks
        # j = 2 -> x = 2.5 weeks
        # etc.
        x = (
                j
                + 0.5
        )

        # Gamma log-PDF:
        #
        # log f(x)
        # =
        # (shape - 1) log(x)
        # - x / scale
        # - log Gamma(shape)
        # - shape log(scale)
        #
        # This is deliberately kept algebraically identical to the
        # implementation used for the original final_v2 fits.
        logpdf = (
                (shape - 1.0)
                * np.log(x)
                -
                (x / scale)
                -
                (
                        lgamma(shape)
                        +
                        shape
                        * np.log(scale)
                )
        )

        pdf = np.exp(
            logpdf
        )

        total = float(
            np.sum(pdf)
        )

        if (
                total <= 0
                or
                not np.isfinite(total)
        ):
            raise RuntimeError(
                "Gamma midpoint PMF normalization "
                "failed. Check shape and scale."
            )

        f = (
                pdf
                / total
        )

        return f

    # =========================================================================
    # 3. Gamma CDF-bin discretization
    #
    # Sensitivity-analysis alternative.
    #
    # Rather than representing the probability in each one-week interval
    # by the PDF value at its midpoint, calculate the ACTUAL continuous
    # Gamma probability contained in that interval:
    #
    #     P(j <= T < j+1)
    #       =
    #     F(j+1) - F(j)
    #
    # where F is the Gamma CDF.
    # =========================================================================

    if kind == "gamma_cdf":

        lower_edges = j

        upper_edges = (
                j
                + 1.0
        )

        cdf_lower = gamma_dist.cdf(
            lower_edges,
            a=shape,
            scale=scale,
        )

        cdf_upper = gamma_dist.cdf(
            upper_edges,
            a=shape,
            scale=scale,
        )

        f = (
                cdf_upper
                -
                cdf_lower
        )

        # Numerical protection against extremely small negative values
        # caused by floating-point arithmetic.
        f = np.maximum(
            f,
            0.0,
        )

        total = float(
            np.sum(f)
        )

        if (
                total <= 0
                or
                not np.isfinite(total)
        ):
            raise RuntimeError(
                "Gamma CDF-bin PMF normalization "
                "failed. Check shape and scale."
            )

        # Normalize over the represented lag range.
        #
        # This mirrors the treatment of the original midpoint kernel:
        # probability beyond max_weeks + 1 is excluded and the retained
        # probability is renormalized to sum to one.
        f = (
                f
                / total
        )

        return f

    # =========================================================================
    # 4. Gamma CDF-bin discretization, shifted right by shift_weeks
    #
    # Sensitivity-analysis alternative.
    #
    # Same CDF-bin construction as above, but the underlying Gamma is
    # shifted right by shift_weeks before binning, so bin j captures
    #
    #     P(j <= T + shift_weeks < j+1)
    #       =
    #     F(j + 1 - shift_weeks) - F(j - shift_weeks)
    #
    # where F is the Gamma CDF. This enforces a hard floor: no mass is
    # placed at any lag below shift_weeks, by construction.
    # =========================================================================

    if kind == "gamma_shifted":

        lower_edges = (
                j
                - shift_weeks
        )

        upper_edges = (
                j
                + 1.0
                - shift_weeks
        )

        cdf_lower = gamma_dist.cdf(
            lower_edges,
            a=shape,
            scale=scale,
        )

        cdf_upper = gamma_dist.cdf(
            upper_edges,
            a=shape,
            scale=scale,
        )

        f = (
                cdf_upper
                -
                cdf_lower
        )

        # Numerical protection against extremely small negative values
        # caused by floating-point arithmetic.
        f = np.maximum(
            f,
            0.0,
        )

        total = float(
            np.sum(f)
        )

        if (
                total <= 0
                or
                not np.isfinite(total)
        ):
            raise RuntimeError(
                "Gamma shifted-CDF-bin PMF normalization "
                "failed. Check shape, scale, and shift_weeks."
            )

        # Normalize over the represented lag range.
        f = (
                f
                / total
        )

        return f

    # =========================================================================
    # Unknown kernel
    # =========================================================================

    raise ValueError(
        f"Unknown kind={kind!r}. "
        "Use 'fixed', 'gamma', 'gamma_cdf', or 'gamma_shifted'."
    )

# =============================================================================
# Infections -> expected deaths
# =============================================================================

def deaths_from_infections_fast(
    C_week: np.ndarray,
    psi: float,
    f_delay: np.ndarray,
) -> np.ndarray:
    """
    Convert weekly infections into expected weekly deaths.

    The observation model is

        muD[w]
        =
        psi
        *
        sum_j C_week[w-j] * f_delay[j]

    where ``f_delay`` is a causal discrete infection-to-death delay PMF.

    Parameters
    ----------
    C_week : np.ndarray
        Weekly model-implied infection incidence.

    psi : float
        Infection-to-mortality scaling parameter.

    f_delay : np.ndarray
        Discrete delay probability mass function.

    Returns
    -------
    np.ndarray
        Expected weekly mortality.
    """

    C_week = np.asarray(
        C_week,
        dtype=float,
    )

    f_delay = np.asarray(
        f_delay,
        dtype=float,
    )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if C_week.ndim != 1:
        raise ValueError(
            "C_week must be 1D"
        )

    if f_delay.ndim != 1:
        raise ValueError(
            "f_delay must be 1D"
        )

    if psi <= 0:
        raise ValueError(
            "psi must be > 0"
        )

    if not np.all(
        np.isfinite(C_week)
    ):
        raise ValueError(
            "C_week contains non-finite values"
        )

    if not np.all(
        np.isfinite(f_delay)
    ):
        raise ValueError(
            "f_delay contains non-finite values"
        )

    if np.any(
        f_delay < 0
    ):
        raise ValueError(
            "f_delay must be nonnegative"
        )


    # -------------------------------------------------------------------------
    # Normalize kernel if it is slightly off one due only to numerical error
    # -------------------------------------------------------------------------

    total = float(
        np.sum(f_delay)
    )

    if (
        total <= 0
        or
        not np.isfinite(total)
    ):
        raise ValueError(
            "f_delay must have a positive finite sum"
        )

    if not np.isclose(
        total,
        1.0,
    ):
        f_delay = (
            f_delay
            / total
        )


    # -------------------------------------------------------------------------
    # Causal convolution
    #
    # np.convolve gives:
    #
    #   full[w] = sum_j C_week[w-j] * f_delay[j]
    #
    # Keeping the first len(C_week) entries gives expected mortality over
    # exactly the observed model weeks.
    # -------------------------------------------------------------------------

    mu = np.convolve(
        C_week,
        f_delay,
        mode="full",
    )[
        :len(C_week)
    ]

    return (
        psi
        * mu
    )


# =============================================================================
# Negative Binomial likelihood
# =============================================================================

def negbin_loglik_vec(
    y: np.ndarray,
    mu: np.ndarray,
    k: float,
) -> float:
    """
    Negative Binomial log-likelihood with mean ``mu`` and
    dispersion parameter ``k``.

    Parameterization
    ----------------

        Var(Y)
        =
        mu
        +
        mu^2 / k

    Equivalently,

        r = k

        p = r / (r + mu)

    and

        P(Y=y)
        =
        Gamma(y+r)
        -------------------------  p^r (1-p)^y.
        Gamma(r) Gamma(y+1)

    Parameters
    ----------
    y : np.ndarray
        Observed weekly counts.

    mu : np.ndarray
        Expected weekly counts.

    k : float
        Negative Binomial dispersion parameter.

    Returns
    -------
    float
        Total log-likelihood across weeks.
    """

    y = np.asarray(
        y,
        dtype=float,
    )

    mu = np.asarray(
        mu,
        dtype=float,
    )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if y.shape != mu.shape:
        raise ValueError(
            "y and mu must have the same shape"
        )

    if (
        k <= 0
        or
        not np.isfinite(k)
    ):
        return -np.inf

    if np.any(
        mu <= 0
    ):
        return -np.inf

    if np.any(
        ~np.isfinite(mu)
    ):
        return -np.inf

    if np.any(
        y < 0
    ):
        return -np.inf

    if np.any(
        ~np.isfinite(y)
    ):
        return -np.inf


    # -------------------------------------------------------------------------
    # Negative Binomial parameterization
    # -------------------------------------------------------------------------

    r = float(
        k
    )

    p = (
        r
        /
        (
            r
            + mu
        )
    )


    # -------------------------------------------------------------------------
    # Log likelihood
    #
    # log P(Y=y)
    # =
    # log Gamma(y+r)
    # - log Gamma(r)
    # - log Gamma(y+1)
    # + r log(p)
    # + y log(1-p)
    # -------------------------------------------------------------------------

    ll = (
        np.vectorize(
            lgamma
        )(
            y
            + r
        )
        -
        lgamma(
            r
        )
        -
        np.vectorize(
            lgamma
        )(
            y
            + 1.0
        )
        +
        r
        * np.log(
            p
        )
        +
        y
        * np.log(
            1.0
            - p
        )
    )


    total_ll = float(
        np.sum(
            ll
        )
    )

    if not np.isfinite(
        total_ll
    ):
        return -np.inf

    return total_ll