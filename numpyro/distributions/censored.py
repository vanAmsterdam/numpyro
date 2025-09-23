# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

from numpyro._typing import DistributionT, ConstraintT
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)


class LeftCensoredDistribution(Distribution):
    r"""
    Distribution wrapper for left-censored survival outcomes.

    This distribution augments an event-time distribution with left-censoring,
    so that the likelihood contribution depends on the censoring indicator.

    Parameters
    ----------
    base_dist : numpyro.distributions.Distribution
        Parametric distribution for the *uncensored* event times
        (e.g., Exponential, Weibull, LogNormal, etc.).
        This distribution must have non-negative support and implement a `cdf` method.
    censored : array-like of {0,1}
        Censoring indicator per observation:
        - 0 → event time is observed exactly
        - 1 → observation is left-censored at the reported time
        (true event time occurred *on or before* the reported time)

    Notes
    -----
    - The `log_prob(value)` method expects `value` to be the observed follow-up
    time (upper bound) for each subject. The contribution to the log-likelihood is:

        log f(time)    if censored == 0
        log F(time)    if censored == 1

    where f is the density and F the cumulative distribution function of `base_dist`.

    - In R's **survival** package notation, this corresponds to
    `Surv(time, event, type = "left")`.

        Example:
        `Surv(time = c(2, 4, 6), event = c(0, 1, 0), type="left")`
        means:
        * subject 1 had an event exactly at t=2
        * subject 2 had an event before or at t=4 (left-censored)
        * subject 3 had an event exactly at t=6

    Examples
    --------
    >>> base = dist.LogNormal(0., 1.)
    >>> surv_dist = LeftCensoredDistribution(base, censored=jnp.array([0, 1, 1]))
    >>> loglik = surv_dist.log_prob(jnp.array([2., 4., 6.]))
    # loglik[0] uses density at 2
    # loglik[1] uses CDF at 4
    # loglik[2] uses CDF at 6
    """

    arg_constraints = {"censored": constraints.boolean}
    reparametrized_params = ["censored"]
    pytree_data_fields = ("base_dist", "censored")

    def __init__(
        self,
        base_dist: DistributionT,
        censored: ArrayLike = True,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        assert hasattr(base_dist, "cdf")
        assert base_dist.support is constraints.positive, (
            "The base distribution should be univariate and have positive support."
        )
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(censored))
        self.base_dist: DistributionT = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.censored,) = promote_shapes(censored, shape=batch_shape)
        self._support = base_dist.support
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.sample(key, sample_shape)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> ConstraintT:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        minval = jnp.finfo(value).tiny

        def logF(x):
            # log(F(x)) with stability
            return jnp.log(jnp.clip(self.base_dist.cdf(x), minval, 1.0))

        return jnp.where(
            self.censored,
            logF(value),  # left-censored observations: log F(t)
            self.base_dist.log_prob(value),  # observed values: log f(t)
        )

class RightCensoredDistribution(Distribution):
    r"""
    Distribution wrapper for right-censored survival outcomes.

    This distribution augments an event-time distribution with right-censoring,
    so that the likelihood contribution depends on the censoring indicator.

    Parameters
    ----------
    base_dist : numpyro.distributions.Distribution
        Parametric distribution for the *uncensored* event times
        (e.g., Exponential, Weibull, LogNormal, etc.).
        This distribution must have non-negative support and implement a `cdf` method.
    censored : array-like of {0,1}
        Censoring indicator per observation:
        - 0 → event occurred at the observed time
        - 1 → observation is right-censored at the observed time

    Notes
    -----
    - The `log_prob(value)` method expects `value` to be the observed follow-up
      time for each subject. The contribution to the log-likelihood is:

          log f(time)    if censored == 0
          log S(time)    if censored == 1

      where f is the density and S (i.e. 1 - F(x) = `1 - base_dist.cdf(x)`) the survival function of `base_dist`.

    - In R's **survival** package notation, this corresponds to
      `Surv(time, event)` with `type = "right"`.

        Example:
        `Surv(time = c(5, 8, 10), event = c(1, 0, 1))`
        means:
          * subject 1 had an event at t=5
          * subject 2 was censored at t=8
          * subject 3 had an event at t=10

    Examples
    --------
    >>> base = dist.Exponential(rate=0.1)
    >>> surv_dist = RightCensoredDistribution(base, censored=jnp.array([0, 1, 0]))
    >>> loglik = surv_dist.log_prob(jnp.array([5., 8., 10.]))
    # loglik[0] uses density at 5
    # loglik[1] uses survival at 8
    # loglik[2] uses density at 10
    """

    arg_constraints = {"censored": constraints.boolean}
    reparametrized_params = ["censored"]
    pytree_data_fields = ("base_dist", "censored")

    def __init__(
        self,
        base_dist: DistributionT,
        censored: ArrayLike = True,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        assert hasattr(base_dist, "cdf")
        assert base_dist.support is constraints.positive, (
            "The base distribution should be univariate and have positive support."
        )
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(censored))
        self.base_dist: DistributionT = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.censored,) = promote_shapes(censored, shape=batch_shape)
        self._support = base_dist.support
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.sample(key, sample_shape)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> ConstraintT:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        def logS(x):
            # log(1 - F(x)) with stability
            return jnp.log1p(-self.base_dist.cdf(x))

        return jnp.where(
            self.censored,
            logS(value),  # censored observations: log S(t)
            self.base_dist.log_prob(value),  # observed values: log f(t)
        )


class IntervalCensoredDistribution(Distribution):
    r"""
    Distribution wrapper for interval-censored survival outcomes.

    This distribution augments an event-time distribution with interval censoring,
    so that the likelihood contribution depends on whether the observation is
    left-censored, right-censored, or truly interval-censored.

    Parameters
    ----------
    base_dist : numpyro.distributions.Distribution
        Parametric distribution for the *uncensored* event times
        (e.g., Exponential, Weibull, LogNormal, etc.).
        This distribution must have non-negative support and implement a `cdf` method.

    Notes
    -----
    - The `log_prob(value)` method expects `value` to be a two-dimensional array
    of shape `(batch_size, 2)`, where each row is `(x1, x2)`:

        * If `x1` is NaN and `x2` is finite:
                Interval = (-inf, x2] → left-censored at `x2`
                Contribution = log F(x2)

        * If `x1` is finite and `x2` is NaN:
                Interval = (x1, inf) → right-censored at `x1`
                Contribution = log S(x1) = log(1 - F(x1))

        * If both `x1` and `x2` are finite:
                Interval = (x1, x2] → event occurred within the interval
                Contribution = log(F(x2) - F(x1))

    where F is the cumulative distribution function of `base_dist` and
    S is its survival function.

    - This matches the semantics of R’s **survival** package with
    `Surv(l, r, type = "interval2")`.

        Example:
        `Surv(l = c(2, 4, 6), r = c(5, Inf, 9), type="interval2")`
        means:
        * subject 1: event occurred in (2, 5]
        * subject 2: event right-censored at 4
        * subject 3: event occurred in (6, 9]

    Examples
    --------
    >>> base = dist.Weibull(concentration=2.0, scale=3.0)
    >>> surv_dist = IntervalCensoredDistribution(base)
    >>> # Three observations: left-, right-, and interval-censored
    >>> values = jnp.array([
    ...     [jnp.nan, 4.0],   # left-censored at 4
    ...     [5.0,     jnp.nan], # right-censored at 5
    ...     [2.0,     6.0],   # interval (2,6]
    ... ])
    >>> loglik = surv_dist.log_prob(values)
    # loglik[0] = log F(4)
    # loglik[1] = log (1 - F(5))
    # loglik[2] = log (F(6) - F(2))
    """

    arg_constraints = {"censored": constraints.boolean}
    reparametrized_params = ["censored"]
    pytree_data_fields = ("base_dist", "censored")

    def __init__(
        self,
        base_dist: DistributionT,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        assert hasattr(base_dist, "cdf")
        assert base_dist.support is constraints.positive, (
            "The base distribution should be univariate and have positive support."
        )
        self.base_dist = base_dist
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        return self.base_dist.sample(key, sample_shape)


    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        eps = jnp.finfo(value).eps

        x1 = jnp.take(value, 0, axis=-1)
        x2 = jnp.take(value, 1, axis=-1)

        # Masks
        m_left = jnp.isnan(x1) & jnp.isfinite(x2)  # (-inf, x2]
        m_right = jnp.isfinite(x1) & jnp.isnan(x2)  # (x1,  inf)
        m_int = jnp.isfinite(x1) & jnp.isfinite(x2)  # (x1,  x2]

        # Replace NaNs with safe numbers BEFORE passing to cdf
        # Choose a safe point inside the support, which is positive
        lb = 0.0  # for positive support
        x1_safe = jnp.where(m_right | m_int, jnp.maximum(x1, lb + eps), lb + eps)
        x2_safe = jnp.where(m_left | m_int, jnp.maximum(x2, lb + eps), lb + eps)

        # CDFs on safe inputs
        F1 = jnp.clip(self.base_dist.cdf(x1_safe), eps, 1.0 - eps)
        F2 = jnp.clip(self.base_dist.cdf(x2_safe), eps, 1.0 - eps)

        # Likelihood pieces (no NaNs)
        # right-censored: log S(x1) = log(1 - F1)
        lp_right = jnp.log1p(-F1)

        # left-censored: log F(x2)
        lp_left = jnp.log(F2)

        # interval: log(F2 - F1)
        diff = jnp.clip(F2 - F1, eps, 1.0)  # ensure strictly positive
        lp_int = jnp.log(diff)

        # Combine with masks; default 0 so we don't carry NaNs around
        logp = jnp.where(m_right, lp_right, 0.0)
        logp = logp + jnp.where(m_left, lp_left, 0.0)
        logp = logp + jnp.where(m_int, lp_int, 0.0)

        return logp
