# Quick Start

This guide will help you get started with Tempest in just a few minutes.

## Basic Workflow

Using Tempest involves four main steps:

1. **Define the prior distribution** over your parameters
2. **Define the log-likelihood function** that evaluates how well parameters fit your data
3. **Create a Sampler** with your prior and likelihood
4. **Run the sampler** and analyze the results

---

## Step 1: Define the Prior

Tempest works with priors defined as transformations from the unit hypercube. The simplest way is to use scipy distributions:

```python
import numpy as np
from scipy.stats import uniform, norm

n_dim = 3  # Number of parameters

# Define prior transform: U(-5, 5) for each dimension
def prior_transform(u):
    return 10 * u - 5  # Transform [0,1] to [-5, 5]
```

For more complex priors, you can define a prior transform function:

```python
def prior_transform(u):
    """Transform unit cube samples to prior samples."""
    x = np.zeros_like(u)
    x[0] = 10 * u[0] - 5      # U(-5, 5)
    x[1] = norm.ppf(u[1], 0, 1)  # N(0, 1)
    x[2] = np.exp(u[2] * 4)   # log-uniform(0, 4)
    return x
```

---

## Step 2: Define the Log-Likelihood

The log-likelihood function evaluates the probability of observing your data given parameter values:

```python
def log_likelihood(x):
    """
    Example: Multivariate Gaussian centered at origin.
    
    Parameters
    ----------
    x : ndarray
        Parameter values, shape (n_dim,) or (n_samples, n_dim) if vectorized.
    
    Returns
    -------
    float or ndarray
        Log-likelihood value(s).
    """
    return -0.5 * np.sum(x**2)
```

For vectorized evaluation (faster with many particles):

```python
def log_likelihood_vectorized(x):
    """Vectorized log-likelihood for batch evaluation."""
    return -0.5 * np.sum(x**2, axis=1)
```

---

## Step 3: Create the Sampler

```python
import tempest as pc

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=512,    # Number of effective samples
    n_active=256,       # Number of active particles
)
```

For vectorized likelihood:

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_vectorized,
    n_dim=n_dim,
    vectorize=True,
)
```

---

## Step 4: Run and Analyze

```python
# Run the sampler
sampler.run(n_total=4096)  # Target number of independent samples

# Get posterior samples
samples, weights, logl = sampler.posterior()

# Get evidence estimate
logz, logz_err = sampler.evidence()
print(f"log(Z) = {logz:.2f}")
```

---

## Complete Example

Here's a complete example sampling from a 2D Gaussian:

```python
import numpy as np
import tempest as pc
from scipy.stats import uniform

# Setup
n_dim = 2

def prior_transform(u):
    """U(-10, 10) for each dimension."""
    return 20 * u - 10

def log_likelihood(x):
    """2D Gaussian centered at (2, 3) with unit variance."""
    mu = np.array([2.0, 3.0])
    return -0.5 * np.sum((x - mu)**2)

# Create and run sampler
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
sampler.run(n_total=2048)

# Get results
samples, weights, logl = sampler.posterior()
logz, _ = sampler.evidence()

# Print summary
print(f"Mean: {np.average(samples, weights=weights, axis=0)}")
print(f"Std:  {np.sqrt(np.average((samples - np.average(samples, weights=weights, axis=0))**2, weights=weights, axis=0))}")
print(f"log(Z) = {logz:.2f}")
```

---

## Next Steps

- Learn about [Prior Distributions](user_guide/priors.md)
- Explore [Parallelization](user_guide/parallelization.md) for large problems
- Check out more [Examples](examples/rosenbrock.md)
- Read the [API Reference](api/sampler.md) for all options
