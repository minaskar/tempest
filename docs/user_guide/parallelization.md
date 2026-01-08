# Parallelization

Tempest supports multiple parallelization strategies to accelerate sampling for expensive likelihood functions.

## Overview

| Method | Best For | Setup Complexity |
|--------|----------|------------------|
| Multiprocessing | Single machine, multiple cores | Easy |
| MPI | HPC clusters, distributed computing | Medium |
| Vectorization | GPU-compatible likelihoods | Easy |

---

## Multiprocessing

Use Python's multiprocessing to parallelize likelihood evaluations across CPU cores.

### Basic Usage

```python
import tempest as pc
import numpy as np

n_dim = 10

def prior_transform(u):
    return 20 * u - 10  # U(-10, 10)

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=8,  # Use 8 processes
)

sampler.run()
```

### Using multiprocess Pool

For more control, create a pool explicitly:

```python
from multiprocess import Pool

with Pool(8) as pool:
    sampler = pc.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        pool=pool,
    )
    sampler.run()
```

### Considerations

!!! warning "Pickling Requirements"
    The likelihood function must be picklable. Avoid:
    - Lambda functions
    - Functions defined inside other functions
    - Functions with unpicklable objects in closure

```python
# ❌ Won't work - lambda not picklable
sampler = pc.Sampler(
    prior=prior,
    likelihood=lambda x: -0.5 * np.sum(x**2),
    pool=4,
)

# ✅ Works - module-level function
def log_likelihood(x):
    return -0.5 * np.sum(x**2)

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=4,
)
```

### Optimal Number of Processes

- Set `n_active` as a multiple of the number of processes for efficient load balancing
- Generally, use the number of physical cores (not hyperthreads)

```python
import os

n_cores = os.cpu_count() // 2  # Physical cores
n_active = n_cores * 32  # 32 particles per core

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=n_cores,
    n_active=n_active,
)
```

---

## MPI Parallelization

For HPC clusters and distributed computing, use MPI via `mpi4py`.

### Setup

First, install mpi4py:

```bash
pip install mpi4py
```

### Using MPIPool

```python
import tempest as pc
from tempest import MPIPool

# Create MPI pool
pool = MPIPool()

if pool.is_master():
    sampler = pc.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        pool=pool,
    )
    sampler.run()

pool.close()
```

### Running with MPI

```bash
mpiexec -n 8 python my_script.py
```

### Complete MPI Example

```python
# mpi_example.py
import numpy as np
import tempest as pc
from tempest import MPIPool

def log_likelihood(x):
    """Expensive likelihood calculation."""
    return -0.5 * np.sum(x**2)

n_dim = 10

def prior_transform(u):
    return 20 * u - 10  # U(-10, 10)

# Initialize MPI pool
pool = MPIPool()

if pool.is_master():
    # Only master process runs the sampler
    sampler = pc.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        pool=pool,
        n_active=256,
    )
    
    sampler.run(n_total=4096)
    
    # Get results on master
    samples, weights, logl = sampler.posterior()
    print(f"Collected {len(samples)} samples")

pool.close()
```

Run with:
```bash
mpiexec -n 16 python mpi_example.py
```

### MPI Tips

!!! tip "Worker Scaling"
    - MPI uses N-1 workers (one process is the master)
    - Set `n_active` divisible by the number of workers

!!! warning "Cluster Job Scripts"
    On HPC systems, ensure your job script requests the correct resources:
    ```bash
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=32
    mpiexec -n 64 python my_script.py
    ```

---

## Vectorization

If your likelihood can evaluate multiple samples at once, vectorization is the most efficient approach.

### Basic Vectorization

```python
def log_likelihood_vectorized(x):
    """
    Vectorized likelihood.
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_dim)
        Batch of parameter samples
    
    Returns
    -------
    logl : ndarray, shape (n_samples,)
        Log-likelihood for each sample
    """
    return -0.5 * np.sum(x**2, axis=1)

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_vectorized,
    n_dim=n_dim,
    vectorize=True,
)
```

### GPU Acceleration with JAX

```python
import jax.numpy as jnp
from jax import jit

@jit
def log_likelihood_jax(x):
    """JAX-accelerated likelihood."""
    return -0.5 * jnp.sum(x**2, axis=1)

def log_likelihood(x):
    """Wrapper to convert to numpy."""
    return np.array(log_likelihood_jax(jnp.array(x)))

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    vectorize=True,
)
```

### Vectorization Requirements

- Input: `(n_samples, n_dim)` array
- Output: `(n_samples,)` array of log-likelihoods
- Not compatible with blobs (auxiliary return values)

---

## Combining Strategies

### Multiprocessing + Vectorization

For complex models, combine both:

```python
# Each process evaluates a batch
def log_likelihood_batch(x):
    """Process a batch on one core."""
    return -0.5 * np.sum(x**2, axis=1)

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_batch,
    n_dim=n_dim,
    vectorize=True,  # Within each process
    pool=4,          # Across processes
)
```

!!! warning "Overhead Consideration"
    For very fast likelihoods, parallelization overhead may outweigh benefits. Benchmark before committing to a strategy.

---

## Performance Tips

### Profiling Likelihood Time

```python
import time

def log_likelihood_timed(x):
    start = time.time()
    result = expensive_calculation(x)
    elapsed = time.time() - start
    print(f"Likelihood evaluation: {elapsed:.3f}s")
    return result
```

### Guidelines

| Likelihood Time | Recommended Strategy |
|-----------------|---------------------|
| < 1 ms | Vectorization only |
| 1-100 ms | Multiprocessing or vectorization |
| > 100 ms | MPI for clusters |

### Efficient n_active Selection

```python
# For p processes, choose n_active = k * p for some integer k
n_processes = 8
n_active = 32 * n_processes  # 256 particles

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=n_processes,
    n_active=n_active,
)
```
