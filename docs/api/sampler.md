# Sampler

The `Sampler` class is the main interface to Tempest. It implements the Persistent Sampling algorithm for Bayesian inference.

## Overview

The sampler manages the entire PS workflow:

1. Initialization from prior samples
2. Iterative tempering towards the posterior
3. MCMC mutation with persistent proposals
4. Evidence estimation

---

## Class Reference

::: tempest.sampler.Sampler
    options:
      members:
        - __init__
        - run
        - sample
        - posterior
        - evidence
        - results
        - save_state
        - load_state
      show_root_heading: true
      show_source: true
      heading_level: 3

---

## Quick Reference

### Creating a Sampler

```python
import tempest as tp
import numpy as np

n_dim = 5

def prior_transform(u):
    return 20 * u - 10  # U(-10, 10)

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=512,
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_transform` | callable | - | Prior distribution or transform function |
| `log_likelihood` | callable | - | Log-likelihood function |
| `n_dim` | int | - | Number of dimensions |
| `n_particles` | Optional[int] | None | Number of particles per iteration. None (default) computes as 2 * n_dim. |
| `ess_ratio` | float | 2.0 | Target ESS ratio (ESS / n_particles). Target ESS = ess_ratio * n_particles. |
| `volume_variation` | Optional[float] | None | Target coefficient of variation for volume. None for ESS-only mode. |
| `vectorize` | bool | False | Vectorized likelihood evaluation |
| `pool` | Pool/int | None | Parallelization pool |
| `clustering` | bool | True | Enable hierarchical clustering |

### Running the Sampler

```python
sampler.run(
    n_total=4096,      # Target independent samples
    progress=True,      # Show progress bar
    save_every=10,      # Checkpoint frequency
)
```

### Extracting Results

```python
# Weighted posterior samples
samples, weights, logl = sampler.posterior()

# Evidence estimate
logz, logz_err = sampler.evidence()

# Full results dictionary
results = sampler.results
```

---

## Examples

### Basic Usage

```python
import numpy as np
import tempest as tp

n_dim = 5

def prior_transform(u):
    return 20 * u - 10

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
sampler.run(n_total=4096)

samples, weights, logl = sampler.posterior()
```

### With Parallelization

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=8,  # 8 processes
    # n_active is optional - automatically set to n_effective // 2 = 256
    # For optimal load balancing: n_active=256 (evenly divisible by 8)
)
```

### Resuming from Checkpoint

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
sampler.run(
    n_total=8192,
    resume_state_path="states/ps_100.state",
)
```
