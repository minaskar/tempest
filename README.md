# **TEMPEST**

**Tempest is a Python implementation of the Persistent Sampling method for accelerated Bayesian inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/minaskar/tempest/blob/master/LICENCE)
[![Documentation Status](https://readthedocs.org/projects/tempest-sampler/badge/?version=latest)](https://tempest-sampler.readthedocs.io/en/latest/?badge=latest)


# Getting started

## Brief introduction

``Tempest`` is a Python package for fast Bayesian posterior and model evidence estimation. It leverages 
the Persistent Sampling (PS) algorithm, offering significant speed improvements over 
traditional methods like MCMC and Nested Sampling. Ideal for large-scale scientific problems 
with expensive likelihood evaluations, non-linear correlations, and multimodality, ``Tempest`` 
provides efficient and scalable posterior sampling and model evidence estimation. Widely used 
in cosmology and astronomy, ``Tempest`` is user-friendly, flexible, and actively maintained.

## Documentation

Read the docs at [tempest-sampler.readthedocs.io](https://tempest-sampler.readthedocs.io) for more information, examples and tutorials. For a detailed list of changes, see the [CHANGELOG.md](https://github.com/minaskar/tempest/blob/main/CHANGELOG.md).

## Installation

To install ``tempest`` using ``pip`` run:

```bash
pip install tempest
```

or, to install from source:

```bash
git clone https://github.com/minaskar/tempest.git
cd tempest
python setup.py install
```

## Basic example

For instance, if you wanted to draw samples from a 10-dimensional Rosenbrock distribution with a uniform prior, you would do something like:

```python
import tempest as tp
import numpy as np

n_dim = 10  # Number of dimensions

# Define prior transform: U(-10, 10) for each dimension
def prior_transform(u):
    return 20 * u - 10

# Define log-likelihood
def log_likelihood(x):
    return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
            + (x[:,::2] - 1.0)**2.0, axis=1)

# Create and run sampler
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    vectorize=True,
)
sampler.run()

samples, weights, logl = sampler.posterior() # Weighted posterior samples

logz, logz_err = sampler.evidence() # Bayesian model evidence estimate and uncertainty
```


# Attribution & Citation

Please cite the following papers if you found this code useful in your research:

```bash
@article{karamanis2025persistent,
  title={Persistent Sampling: Enhancing the Efficiency of Sequential Monte Carlo},
  author={Karamanis, Minas and Seljak, Uro{\v{s}}},
  journal={Statistics and Computing},
  volume={35},
  number={5},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```

# Licence

Copyright 2026-Present Minas Karamanis and contributors.

``Tempest`` is free software made available under the MIT License. For details see the `LICENCE` file.
