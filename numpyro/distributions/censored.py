# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

from numpyro._typing import DistributionT
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)

class LeftCensoredDistribution(Distribution):
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
        super().__init__(batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        minval = jnp.finfo(value).tiny
        def logF(x):
            # log(F(x)) with stability
            return jnp.log(jnp.clip(self.base_dist.cdf(x), minval, 1.))
        return jnp.where(
            self.censored,
            logF(value), # left-censored observations: log F(t)
            self.base_dist.log_prob(value) # observed values: log f(t)
        )

class RightCensoredDistribution(Distribution):
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
        super().__init__(batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        def logS(x):
            # log(1 - F(x)) with stability
            return jnp.log1p(-self.base_dist.cdf(x))

        return jnp.where(
            self.censored,
            logS(value), # censored observations: log S(t)
            self.base_dist.log_prob(value) # observed values: log f(t)
