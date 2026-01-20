# Parallel (MPI) - ezmpi Integration

Tempest integrates with [ezmpi](https://github.com/minaskar/ezmpi) for MPI-based parallelization.

## Overview

[ezmpi](https://github.com/minaskar/ezmpi) is a simple MPI-based processing pool that distributes tasks across multiple processes using MPI (Message Passing Interface). It provides the same interface as Tempest's former `MPIPool` with additional improvements and maintenance.

Use it for:
- High-performance computing (HPC) clusters
- Large-scale distributed computing
- Multi-node parallelization

## Installation

Install ezmpi separately:

```bash
pip install ezmpi
```

You also need an MPI implementation installed on your system:
- OpenMPI
- MPICH
- Intel MPI

## Usage

Use ezmpi's `MPIPool` class with Tempest:

```python
import tempest as tp
from ezmpi import MPIPool

def log_likelihood(x):
    """Expensive likelihood calculation."""
    return -0.5 * np.sum(x**2)

n_dim = 10

def prior_transform(u):
    return 20 * u - 10  # U(-10, 10)

# Create MPI pool
pool = MPIPool()

if pool.is_master():
    sampler = tp.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        pool=pool,
        n_active=256,
    )
    sampler.run(n_total=4096)
    
    samples, weights, logl = sampler.posterior()
    print(f"Collected {len(samples)} samples")

pool.close()
```

## Running MPI Scripts

Save your script and run with `mpiexec`:

```bash
mpiexec -n 8 python my_sampling_script.py
```

### On HPC Clusters

Example SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=tempest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00

module load mpi4py

mpiexec -n 64 python my_sampling_script.py
```

## API Reference

The ezmpi package provides the `MPIPool` class with the following key methods:

- `__init__(comm=None, use_dill=True)`: Initialize pool
- `map(worker, tasks)`: Execute worker function on each task
- `close()`: Shutdown worker processes
- `is_master()`: Check if current process is master (rank 0)
- `is_worker()`: Check if current process is worker (rank > 0)

For detailed documentation, visit [ezmpi.readthedocs.io](https://ezmpi.readthedocs.io).
