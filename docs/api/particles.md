# Particles

The `Particles` class manages the collection of samples and their associated metadata throughout the PS run.

## Overview

The `Particles` class:

- Stores all sampled particles across iterations
- Tracks log-likelihoods, weights, and other metadata
- Computes importance weights for posterior estimation
- Provides access to the full sampling history

---

## Class Reference

::: tempest.particles.Particles
    options:
      members:
        - __init__
        - update
        - get
        - pop
        - compute_logw_and_logz
        - compute_results
      show_root_heading: true
      show_source: true
      heading_level: 3

---

## Usage

The `Particles` class is primarily used internally by the `Sampler`, but you can access it for detailed analysis:

```python
import tempest as pc

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
sampler.run()

# Access particles object
particles = sampler.particles

# Get all samples in unit hypercube
u_all = particles.get("u", flat=True)

# Get all samples in physical space
x_all = particles.get("x", flat=True)

# Get all log-likelihoods
logl_all = particles.get("logl", flat=True)

# Get data from specific iteration
x_iter_5 = particles.get("x", index=5)
```

---

## Stored Data

The `Particles` object stores the following quantities for each iteration:

| Key | Shape | Description |
| --- | --- | --- |
| `u` | (n_active, n_dim) | Samples in unit hypercube |
| `x` | (n_active, n_dim) | Samples in physical space |
| `logl` | (n_active,) | Log-likelihood values |
| `logw` | (n_active,) | Log importance weights |
| `blobs` | varies | Additional data from likelihood |
| `iter` | scalar | Iteration index |
| `logz` | scalar | Evidence estimate at iteration |
| `calls` | scalar | Cumulative likelihood evaluations |
| `steps` | scalar | MCMC steps taken |
| `efficiency` | scalar | MCMC efficiency |
| `ess` | scalar | Effective sample size |
| `accept` | scalar | MCMC acceptance rate |
| `beta` | scalar | Inverse temperature |

---

## Computing Weights

The `compute_logw_and_logz` method computes importance weights targeting a specific temperature:

```python
# Compute weights for the posterior (beta=1)
log_weights, log_evidence = particles.compute_logw_and_logz(beta_final=1.0)

# Convert to normalized weights
weights = np.exp(log_weights)
weights /= np.sum(weights)
```

### Weight Computation Details

The importance weights account for:

1. The target temperature (`beta_final`)
2. The temperature at which each sample was drawn
3. The number of samples at each iteration (balance heuristic)

This allows combining samples from all iterations into a single weighted posterior.

---

## Full Results

Get a complete results dictionary:

```python
results = particles.compute_results()

# Contains all stored data plus final weights
print(results.keys())
# dict_keys(['u', 'x', 'logl', 'logw', 'blobs', 'iter', 'logz', ...])
```
