# Rosenbrock Distribution

This example demonstrates sampling from the challenging Rosenbrock distribution, a common benchmark for optimization and sampling algorithms.

## Problem Description

The Rosenbrock function is defined as:

$$
f(x) = \sum_{i=0}^{n/2-1} \left[ 100(x_{2i+1} - x_{2i}^2)^2 + (1 - x_{2i})^2 \right]
$$

This creates a narrow, curved valley that is difficult to sample from. The minimum is at $(1, 1, \ldots, 1)$.

---

## Implementation

```python
import numpy as np
import tempest as tp
from scipy.stats import uniform
import matplotlib.pyplot as plt

# Number of dimensions
n_dim = 10

# Define prior transform: U(-10, 10) for each parameter
def prior_transform(u):
    return 20 * u - 10

# Define log-likelihood (negative Rosenbrock function)
def log_likelihood(x):
    """
    Log-likelihood from Rosenbrock function.
    
    Parameters
    ----------
    x : ndarray, shape (n_dim,) or (n_samples, n_dim)
        Parameter values
    
    Returns
    -------
    logl : float or ndarray
        Log-likelihood value(s)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    result = -np.sum(
        100.0 * (x[:, 1::2] - x[:, ::2]**2)**2 
        + (1.0 - x[:, ::2])**2, 
        axis=1
    )
    
    return result.squeeze() if result.size == 1 else result

# Create sampler
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=1024,
    n_active=512,
    vectorize=True,
    random_state=42,
)

# Run sampler
sampler.run(n_total=8192, progress=True)

# Get results
samples, weights, logl = sampler.posterior()
logz, logz_err = sampler.evidence()

print(f"Number of samples: {len(samples)}")
print(f"Log-evidence: {logz:.2f}")
```

---

## Analysis

### Parameter Estimates

```python
# Weighted mean and standard deviation
mean = np.average(samples, weights=weights, axis=0)
std = np.sqrt(np.average((samples - mean)**2, weights=weights, axis=0))

print("Parameter estimates:")
for i in range(n_dim):
    print(f"  x[{i}] = {mean[i]:.3f} ± {std[i]:.3f}")
```

Expected: All parameters should be close to 1.0.

### Visualization

```python
import corner

# Resample for unweighted corner plot
idx = np.random.choice(len(samples), size=5000, p=weights, replace=True)
samples_resampled = samples[idx]

# Corner plot
fig = corner.corner(
    samples_resampled[:, :4],  # First 4 dimensions
    labels=[f"$x_{{{i}}}$" for i in range(4)],
    truths=[1.0, 1.0, 1.0, 1.0],
    quantiles=[0.16, 0.5, 0.84],
)
plt.savefig("rosenbrock_corner.png")
```

---

## High-Dimensional Challenge

The Rosenbrock becomes increasingly challenging with more dimensions:

```python
for n_dim in [2, 10, 20, 50]:
    def prior_transform(u):
        return 20 * u - 10
    
    sampler = tp.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        n_effective=max(512, n_dim * 20),
        n_active=max(256, n_dim * 10),
        vectorize=True,
    )
    
    sampler.run(n_total=4096)
    
    samples, weights, _ = sampler.posterior()
    mean = np.average(samples, weights=weights, axis=0)
    
    print(f"n_dim={n_dim}: mean distance from (1,1,...) = {np.linalg.norm(mean - 1):.4f}")
```

---

## Parallelized Version

For expensive evaluations, use parallelization:

```python
from multiprocess import Pool

def log_likelihood_serial(x):
    """Serial version for parallel evaluation."""
    return -np.sum(
        100.0 * (x[1::2] - x[::2]**2)**2 
        + (1.0 - x[::2])**2
    )

n_dim = 10

def prior_transform(u):
    return 20 * u - 10

with Pool(4) as pool:
    sampler = tp.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood_serial,
        n_dim=n_dim,
        pool=pool,
        n_active=256,  # Multiple of pool size
    )
    sampler.run(n_total=4096)
```

---

## Performance Tips

!!! tip "Particle Count"
    For the Rosenbrock, increase `n_effective` with dimension:
    - 2D: 256 effective particles
    - 10D: 512-1024 effective particles
    - 20D+: 1024-2048 effective particles

!!! tip "Vectorization"
    Always use `vectorize=True` for the Rosenbrock since it's cheap to evaluate.

!!! tip "Clustering"
    Enable clustering for dimensions > 4, as the banana-shaped marginals benefit from adaptive proposals.
