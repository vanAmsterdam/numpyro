from functools import partial
import warnings

import jax
from jax import device_get, lax, random, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_flatten

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import ComposeTransform, biject_to
from numpyro.handlers import block, condition, seed, substitute, trace
from numpyro.util import not_jax_tracer, while_loop

__all__ = [
    'find_valid_initial_params',
    'log_density',
    'log_likelihood',
    'init_to_feasible',
    'init_to_median',
    'init_to_prior',
    'init_to_uniform',
    'init_to_value',
    'potential_energy',
    'initialize_model',
    'predictive',
    'transformed_potential_energy',
]


def log_density(model, model_args, model_kwargs, params, skip_dist_transforms=False):
    """
    Computes log of joint density for the model given latent values ``params``.

    :param model: Python callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict params: dictionary of current parameter values keyed by site
        name.
    :param bool skip_dist_transforms: whether to compute log probability of a site
        (if its prior is a transformed distribution) in its base distribution
        domain.
    :return: log of joint density and a corresponding model trace
    """
    # We skip transforms in
    #   + autoguide's model
    #   + hmc's model
    # We apply transforms in
    #   + autoguide's guide
    #   + svi's model + guide
    if skip_dist_transforms:
        model = substitute(model, base_param_map=params)
    else:
        model = substitute(model, param_map=params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    log_joint = 0.
    for site in model_trace.values():
        if site['type'] == 'sample':
            value = site['value']
            intermediates = site['intermediates']
            if intermediates:
                if skip_dist_transforms:
                    log_prob = site['fn'].base_dist.log_prob(intermediates[0][0])
                else:
                    log_prob = site['fn'].log_prob(value, intermediates)
            else:
                log_prob = site['fn'].log_prob(value)
            log_prob = np.sum(log_prob)
            if 'scale' in site:
                log_prob = site['scale'] * log_prob
            log_joint = log_joint + log_prob
    return log_joint, model_trace


def transform_fn(transforms, params, invert=False):
    """
    Callable that applies a transformation from the `transforms` dict to values in the
    `params` dict and returns the transformed values keyed on the same names.

    :param transforms: Dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param params: Dictionary of arrays keyed by names.
    :param invert: Whether to apply the inverse of the transforms.
    :return: `dict` of transformed params.
    """
    if invert:
        transforms = {k: v.inv for k, v in transforms.items()}
    return {k: transforms[k](v) if k in transforms else v
            for k, v in params.items()}


def constrain_fn(model, model_args, model_kwargs, transforms, params):
    """
    Gets value at each latent site in `model` given unconstrained parameters `params`.
    The `transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict transforms: dictionary of transforms keyed by names. Names in
        `transforms` and `params` should align.
    :param dict params: dictionary of unconstrained values keyed by site
        names.
    :return: `dict` of transformed params.
    """
    params_constrained = transform_fn(transforms, params)
    substituted_model = substitute(model, base_param_map=params_constrained)
    model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
    return {k: model_trace[k]['value'] for k, v in params.items() if k in model_trace}


def potential_energy(model, model_args, model_kwargs, inv_transforms, params):
    """
    Computes potential energy of a model given unconstrained params.
    The `inv_transforms` is used to transform these unconstrained parameters to base values
    of the corresponding priors in `model`. If a prior is a transformed distribution,
    the corresponding base value lies in the support of base distribution. Otherwise,
    the base value lies in the support of the distribution.

    :param model: a callable containing NumPyro primitives.
    :param tuple model_args: args provided to the model.
    :param dict model_kwargs`: kwargs provided to the model.
    :param dict inv_transforms: dictionary of transforms keyed by names.
    :param dict params: unconstrained parameters of `model`.
    :return: potential energy given unconstrained parameters.
    """
    params_constrained = transform_fn(inv_transforms, params)
    log_joint, model_trace = log_density(model, model_args, model_kwargs, params_constrained,
                                         skip_dist_transforms=True)
    for name, t in inv_transforms.items():
        t_log_det = np.sum(t.log_abs_det_jacobian(params[name], params_constrained[name]))
        if 'scale' in model_trace[name]:
            t_log_det = model_trace[name]['scale'] * t_log_det
        log_joint = log_joint + t_log_det
    return - log_joint


def transformed_potential_energy(potential_energy, inv_transform, z):
    """
    Given a potential energy `p(x)`, compute potential energy of `p(z)`
    with `z = transform(x)` (i.e. `x = inv_transform(z)`).

    :param potential_energy: a callable to compute potential energy of original
        variable `x`.
    :param ~numpyro.distributions.constraints.Transform inv_transform: a
        transform from the new variable `z` to `x`.
    :param z: new variable to compute potential energy
    :return: potential energy of `z`.
    """
    x, intermediates = inv_transform.call_with_intermediates(z)
    logdet = inv_transform.log_abs_det_jacobian(z, x, intermediates=intermediates)
    return potential_energy(x) - logdet


def _init_to_median(site, num_samples=15, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        samples = numpyro.sample('_init', fn,
                                 sample_shape=(num_samples,) + site['kwargs']['sample_shape'])
        return np.median(samples, axis=0)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_median(num_samples=15):
    """
    Initialize to the prior median.

    :param int num_samples: number of prior points to calculate median.
    """
    return partial(_init_to_median, num_samples=num_samples)


def init_to_prior():
    """
    Initialize to a prior sample.
    """
    return init_to_median(num_samples=1)


def _init_to_uniform(site, radius=2, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if isinstance(site['fn'], dist.TransformedDistribution):
            fn = site['fn'].base_dist
        else:
            fn = site['fn']
        value = numpyro.sample('_init', fn, sample_shape=site['kwargs']['sample_shape'])
        base_transform = biject_to(fn.support)
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(base_transform.inv(value)))
        return base_transform(unconstrained_value)

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        unconstrained_value = numpyro.sample('_unconstrained_init', dist.Uniform(-radius, radius),
                                             sample_shape=np.shape(transform.inv(value)))
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
        else:
            base_transform = transform
        return base_transform(unconstrained_value)


def init_to_uniform(radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    return partial(_init_to_uniform, radius=radius)


def init_to_feasible():
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    return init_to_uniform(radius=0)


def _init_to_value(site, values={}, skip_param=False):
    if site['type'] == 'sample' and not site['is_observed']:
        if site['name'] not in values:
            return _init_to_uniform(site, skip_param=skip_param)

        value = values[site['name']]
        if isinstance(site['fn'], dist.TransformedDistribution):
            value = ComposeTransform(site['fn'].transforms).inv(value)
        return value

    if site['type'] == 'param' and not skip_param:
        # return base value of param site
        constraint = site['kwargs'].pop('constraint', real)
        transform = biject_to(constraint)
        value = site['args'][0]
        if isinstance(transform, ComposeTransform):
            base_transform = transform.parts[0]
            value = base_transform(transform.inv(value))
        return value


def init_to_value(values):
    """
    Initialize to the value specified in `values`. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in `values`.

    :param dict values: dictionary of initial values keyed by site name.
    """
    return partial(_init_to_value, values=values)


def find_valid_initial_params(rng_key, model, *model_args, init_strategy=init_to_uniform(),
                              param_as_improper=False, prototype_params=None, **model_kwargs):
    """
    Given a model with Pyro primitives, returns an initial valid unconstrained
    parameters. This function also returns an `is_valid` flag to say whether the
    initial parameters are valid.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param `*model_args`: args provided to the model.
    :param callable init_strategy: a per-site initialization function.
    :param bool param_as_improper: a flag to decide whether to consider sites with
        `param` statement as sites with improper priors.
    :param `**model_kwargs`: kwargs provided to the model.
    :return: tuple of (`init_params`, `is_valid`).
    """
    init_strategy = jax.partial(init_strategy, skip_param=not param_as_improper)

    def cond_fn(state):
        i, _, _, is_valid = state
        return (i < 100) & (~is_valid)

    def body_fn(state):
        i, key, _, _ = state
        key, subkey = random.split(key)

        # Wrap model in a `substitute` handler to initialize from `init_loc_fn`.
        # Use `block` to not record sample primitives in `init_loc_fn`.
        seeded_model = substitute(model, substitute_fn=block(seed(init_strategy, subkey)))
        model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
        constrained_values, inv_transforms = {}, {}
        for k, v in model_trace.items():
            if v['type'] == 'sample' and not v['is_observed']:
                if v['intermediates']:
                    constrained_values[k] = v['intermediates'][0][0]
                    inv_transforms[k] = biject_to(v['fn'].base_dist.support)
                else:
                    constrained_values[k] = v['value']
                    inv_transforms[k] = biject_to(v['fn'].support)
            elif v['type'] == 'param' and param_as_improper:
                constraint = v['kwargs'].pop('constraint', real)
                transform = biject_to(constraint)
                if isinstance(transform, ComposeTransform):
                    base_transform = transform.parts[0]
                    inv_transforms[k] = base_transform
                    constrained_values[k] = base_transform(transform.inv(v['value']))
                else:
                    inv_transforms[k] = transform
                    constrained_values[k] = v['value']
        params = transform_fn(inv_transforms,
                              {k: v for k, v in constrained_values.items()},
                              invert=True)
        potential_fn = jax.partial(potential_energy, model, model_args, model_kwargs, inv_transforms)
        pe, param_grads = value_and_grad(potential_fn)(params)
        z_grad = ravel_pytree(param_grads)[0]
        is_valid = np.isfinite(pe) & np.all(np.isfinite(z_grad))
        return i + 1, key, params, is_valid

    if prototype_params is not None:
        init_state = (0, rng_key, prototype_params, False)
    else:
        _, _, prototype_params, is_valid = init_state = body_fn((0, rng_key, None, None))
        if not_jax_tracer(is_valid):
            if device_get(is_valid):
                return prototype_params, is_valid

    _, _, init_params, is_valid = while_loop(cond_fn, body_fn, init_state)
    return init_params, is_valid


def initialize_model(rng_key, model, *model_args, init_strategy=init_to_uniform(), **model_kwargs):
    """
    Given a model with Pyro primitives, returns a function which, given
    unconstrained parameters, evaluates the potential energy (negative
    joint density). In addition, this also returns initial parameters
    sampled from the prior to initiate MCMC sampling and functions to
    transform unconstrained values at sample sites to constrained values
    within their respective support.

    :param jax.random.PRNGKey rng_key: random number generator seed to
        sample from the prior. The returned `init_params` will have the
        batch shape ``rng_key.shape[:-1]``.
    :param model: Python callable containing Pyro primitives.
    :param `*model_args`: args provided to the model.
    :param callable init_strategy: a per-site initialization function.
    :param `**model_kwargs`: kwargs provided to the model.
    :return: tuple of (`init_params`, `potential_fn`, `constrain_fn`),
        `init_params` are values from the prior used to initiate MCMC,
        `constrain_fn` is a callable that uses inverse transforms
        to convert unconstrained HMC samples to constrained values that
        lie within the site's support.
    """
    seeded_model = seed(model, rng_key if rng_key.ndim == 1 else rng_key[0])
    model_trace = trace(seeded_model).get_trace(*model_args, **model_kwargs)
    constrained_values, inv_transforms = {}, {}
    has_transformed_dist = False
    for k, v in model_trace.items():
        if v['type'] == 'sample' and not v['is_observed']:
            if v['intermediates']:
                constrained_values[k] = v['intermediates'][0][0]
                inv_transforms[k] = biject_to(v['fn'].base_dist.support)
                has_transformed_dist = True
            else:
                constrained_values[k] = v['value']
                inv_transforms[k] = biject_to(v['fn'].support)
        elif v['type'] == 'param':
            constraint = v['kwargs'].pop('constraint', real)
            transform = biject_to(constraint)
            if isinstance(transform, ComposeTransform):
                base_transform = transform.parts[0]
                constrained_values[k] = base_transform(transform.inv(v['value']))
                inv_transforms[k] = base_transform
                has_transformed_dist = True
            else:
                inv_transforms[k] = transform
                constrained_values[k] = v['value']

    prototype_params = transform_fn(inv_transforms,
                                    {k: v for k, v in constrained_values.items()},
                                    invert=True)

    # NB: we use model instead of seeded_model to prevent unexpected behaviours (if any)
    potential_fn = jax.partial(potential_energy, model, model_args, model_kwargs, inv_transforms)
    if has_transformed_dist:
        # FIXME: why using seeded_model here triggers an error for funnel reparam example
        # if we use MCMC class (mcmc function works fine)
        constrain_fun = jax.partial(constrain_fn, model, model_args, model_kwargs, inv_transforms)
    else:
        constrain_fun = jax.partial(transform_fn, inv_transforms)

    def single_chain_init(key):
        return find_valid_initial_params(key, model, *model_args, init_strategy=init_strategy,
                                         param_as_improper=True, prototype_params=prototype_params,
                                         **model_kwargs)

    if rng_key.ndim == 1:
        init_params, is_valid = single_chain_init(rng_key)
    else:
        init_params, is_valid = lax.map(single_chain_init, rng_key)

    if not_jax_tracer(is_valid):
        if device_get(~np.all(is_valid)):
            raise RuntimeError("Cannot find valid initial parameters. Please check your model again.")
    return init_params, potential_fn, constrain_fun


def predictive(rng_key, model, posterior_samples, *args, num_samples=None, return_sites=None, **kwargs):
    """
    Run model by sampling latent parameters from `posterior_samples`, and return
    values at sample sites from the forward run. By default, only sample sites not contained in
    `posterior_samples` are returned. This can be modified by changing the `return_sites`
    keyword argument.

    .. warning::
        The interface for the `predictive` function is experimental, and
        might change in the future.

    :param jax.random.PRNGKey rng_key: seed to draw samples
    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param list return_sites: sites to return; by default only sample sites not present
        in `posterior_samples` are returned.
    :param int num_samples: number of samples
    :param kwargs: model kwargs.
    :return: dict of samples from the predictive distribution.
    """
    def single_prediction(rng_key, samples):
        model_trace = trace(seed(condition(model, samples), rng_key)).get_trace(*args, **kwargs)
        if return_sites is not None:
            sites = return_sites
        else:
            sites = {k for k, site in model_trace.items()
                     if site['type'] != 'plate' and k not in samples}
        return {name: site['value'] for name, site in model_trace.items() if name in sites}

    if posterior_samples:
        batch_size = tree_flatten(posterior_samples)[0][0].shape[0]
        if num_samples is not None and num_samples != batch_size:
            warnings.warn("Sample's leading dimension size {} is different from the "
                          "provided {} num_samples argument. Defaulting to {}."
                          .format(batch_size, num_samples, batch_size), UserWarning)
        num_samples = batch_size

    if num_samples is None:
        raise ValueError("No sample sites in model to infer `num_samples`.")

    rng_keys = random.split(rng_key, num_samples)
    return vmap(single_prediction)(rng_keys, posterior_samples)


def log_likelihood(model, posterior_samples, *args, **kwargs):
    """
    Returns log likelihood at observation nodes of model, given samples of all latent variables.

    .. warning::
        The interface for the `log_likelihood` function is experimental, and
        might change in the future.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param kwargs: model kwargs.
    :return: dict of log likelihoods at observation sites.
    """
    def single_loglik(samples):
        model_trace = trace(substitute(model, samples)).get_trace(*args, **kwargs)
        return {name: site['fn'].log_prob(site['value']) for name, site in model_trace.items()
                if site['type'] == 'sample' and site['is_observed']}

    return vmap(single_loglik)(posterior_samples)
