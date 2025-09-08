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


class RightCensoredDistribution(Distribution):
    arg_constraints = {"event": constraints.boolean}
    reparametrized_params = ["event"]
    pytree_data_fields = ("base_dist", "event")

    def __init__(
        self,
        base_dist: DistributionT,
        event: ArrayLike = True,
        *,
        validate_args: Optional[bool] = None,
    ):
        # test if base_dist has an implemented cdf method
        assert hasattr(base_dist, "cdf")
        # TODO: check what support needed
        assert base_dist.support is constraints.positive, (
            "The base distribution should be univariate and have positive support."
        )
        batch_shape = lax.broadcast_shapes(base_dist.batch_shape, jnp.shape(event))
        self.base_dist: DistributionT = jax.tree.map(
            lambda p: promote_shapes(p, shape=batch_shape)[0], base_dist
        )
        (self.event,) = promote_shapes(event, shape=batch_shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        # Helper
        def logS(x):
            # log(1 - F(x)) with stability
            return jnp.log1p(-self.base_dist.cdf(x))

        out = jnp.zeros(self.batch_shape)

        # observed event times: log f(t)
        if self.event.any():
            out = out + jnp.where(self.event, self.base_dist.log_prob(value), 0.0)

        # right censored observations: log S(t)
        if (1 - self.event).any():
            out = out + jnp.where(1 - self.event, logS(value), 0.0)

        return out
