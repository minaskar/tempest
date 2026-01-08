# Parallel (MPI)

The `parallel` module provides MPI-based parallelization through the `MPIPool` class.

## Overview

`MPIPool` allows distributing likelihood evaluations across multiple processes using MPI (Message Passing Interface). This is essential for:

- High-performance computing (HPC) clusters
- Large-scale distributed computing
- Multi-node parallelization

---

## Class Reference

::: tempest.parallel.MPIPool
    options:
      members:
        - __init__
        - map
        - wait
        - close
        - is_master
        - is_worker
      show_root_heading: true
      show_source: true
      heading_level: 3

---

## Requirements

Install `mpi4py`:

```bash
pip install mpi4py
```

You also need an MPI implementation installed on your system:
- OpenMPI
- MPICH
- Intel MPI

---

## Basic Usage

```python
import numpy as np
import tempest as pc
from tempest import MPIPool

def log_likelihood(x):
    """Expensive likelihood calculation."""
    return -0.5 * np.sum(x**2)

n_dim = 10

def prior_transform(u):
    return 20 * u - 10  # U(-10, 10)

# Create MPI pool
pool = MPIPool()

if pool.is_master():
    sampler = pc.Sampler(
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

---

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

---

## Master-Worker Pattern

MPIPool uses a master-worker pattern:

- **Master** (rank 0): Runs the sampler, distributes tasks
- **Workers** (rank > 0): Receive tasks, compute likelihoods, return results

```python
pool = MPIPool()

if pool.is_master():
    # Only the master runs the sampler
    sampler = pc.Sampler(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=n_dim,
        pool=pool,
    )
    sampler.run()
    # Only the master has results
    samples = sampler.posterior()[0]
else:
    # Workers automatically wait for tasks
    pass

pool.close()
```

---

## Configuration

### Using dill for Pickling

By default, `MPIPool` uses `dill` for serialization, which can pickle more complex objects:

```python
pool = MPIPool(use_dill=True)  # Default
```

Disable if you have issues with specific objects:

```python
pool = MPIPool(use_dill=False)
```

### Custom Communicator

For advanced MPI usage, provide a custom communicator:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD.Split(color, key)
pool = MPIPool(comm=comm)
```

---

## Best Practices

### Load Balancing

Set `n_active` as a multiple of the number of workers:

```python
n_workers = pool.size  # Number of worker processes
n_active = 32 * n_workers

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=pool,
    n_active=n_active,
)
```

### Resource Sizing

- Use 1 master + N workers = N+1 total MPI ranks
- Each worker gets one likelihood evaluation at a time
- For very fast likelihoods, multiprocessing may be faster

### Error Handling

Wrap your code to handle MPI errors gracefully:

```python
try:
    pool = MPIPool()
    if pool.is_master():
        sampler = pc.Sampler(
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=n_dim,
            pool=pool,
        )
        sampler.run()
finally:
    pool.close()
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
| --- | --- |
| "Tried to create MPI pool with only one process" | Use `mpiexec -n N` with N > 1 |
| Hanging on large clusters | Check MPI environment and network |
| Pickling errors | Ensure likelihood is module-level function |

### Debugging

Run with verbose MPI output:

```bash
mpiexec -n 4 --verbose python script.py
```
