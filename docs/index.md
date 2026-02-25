# Tempest

**Tempest** is a Python implementation of the **Persistent Sampling** method for accelerated Bayesian inference.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/minaskar/tempest/blob/master/LICENCE)
[![Documentation Status](https://readthedocs.org/projects/tempest-sampler/badge/?version=latest)](https://tempest-sampler.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/tempest.svg)](https://badge.fury.io/py/tempest)

---

## What is Tempest?

Tempest is a Python package for fast Bayesian posterior and model evidence estimation. It leverages the **Persistent Sampling (PS)** algorithm, offering significant speed improvements over traditional methods like MCMC and Nested Sampling.

### Key Features

- **Fast Posterior Sampling**: Accelerated Bayesian inference using persistent proposals
- **Evidence Estimation**: Reliable estimation of the Bayesian model evidence (marginal likelihood)
- **Multimodal Support**: Effective handling of multimodal distributions with automatic clustering
- **Parallelization**: Support for multiprocessing and MPI for large-scale problems
- **Flexible Priors**: Easy-to-use prior distribution specification
- **Robust MCMC**: Uses t-preconditioned Crank-Nicolson for efficient parameter space exploration

### When to Use Tempest

Tempest is ideal for:

- Large-scale scientific problems with expensive likelihood evaluations
- Problems with non-linear correlations between parameters
- Multimodal posterior distributions
- Applications in cosmology, astronomy, and other scientific domains

---

## Quick Example

Here's a simple example showing how to sample from a 10-dimensional Rosenbrock distribution:

```python
import tempest as tp
import numpy as np
from scipy.stats import uniform

n_dim = 10  # Number of dimensions

# Define the prior transform: U(-10, 10) for each dimension
def prior_transform(u):
    return 20 * u - 10  # Transform [0,1] to [-10, 10]

# Define the log-likelihood function
def log_likelihood(x):
    return -np.sum(10.0 * (x[:, ::2]**2.0 - x[:, 1::2])**2.0 
                   + (x[:, ::2] - 1.0)**2.0, axis=1)

# Create the sampler
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    vectorize=True,
)

# Run the sampler
sampler.run()

# Get weighted posterior samples
samples, weights, logl = sampler.posterior()

# Get Bayesian evidence estimate
logz, logz_err = sampler.evidence()
```

---

## Examples

Explore more applications of Tempest:

- [Rosenbrock Distribution](examples/rosenbrock.md) - Sampling from a challenging banana-shaped distribution
- [Gaussian Mixture](examples/gaussian_mixture.md) - Handling multimodal posteriors with clustering
- [Bayesian Model Comparison](examples/model_comparison.md) - Using Bayes factors to compare competing models
- [Working with Blobs](examples/blobs.md) - Storing auxiliary quantities from likelihood evaluations

---

## Installation

Install Tempest using pip:

```bash
pip install tempest-sampler
```

Or from source:

```bash
git clone https://github.com/minaskar/tempest.git
cd tempest
pip install .
```

See the [Installation Guide](installation.md) for more details.

---

## License

Tempest is free software made available under the MIT License. For details see the [LICENCE](https://github.com/minaskar/tempest/blob/master/LICENCE) file.

---

## Acknowledgements

If you use Tempest in your research, please cite the relevant papers. See the [Citation](citation.md) page for details.
