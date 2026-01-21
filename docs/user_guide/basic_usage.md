# Basic Usage

This guide covers the core functionality of Tempest and how to use it effectively.

## The Sampler Class

The `Sampler` class is the main interface to Tempest. It manages the entire sampling process, from initialization through posterior estimation.

### Creating a Sampler

```python
import tempest as tp
import numpy as np

# Define prior and likelihood
n_dim = 5

def prior_transform(u):
    """U(-10, 10) for each dimension."""
    return 20 * u - 10

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

# Create sampler with default settings
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_effective` | 512 | Number of effective particles |
| `n_active` | 256 | Number of active particles per iteration |
| `n_boost` | `None` | Target number of effective particles to boost towards (absolute value, not a multiplier) |
| `vectorize` | False | Whether likelihood accepts batched inputs |
| `pool` | None | Process pool for parallelization |

---

## Running the Sampler

### Basic Run

```python
sampler.run(n_total=4096)  # Target ~4096 independent samples
```

### With Progress Bar Control

```python
sampler.run(
    n_total=4096,
    progress=True,  # Show progress bar
)
```

### Saving Checkpoints

Save intermediate states for long runs:

```python
sampler.run(
    n_total=4096,
    save_every=10,  # Save every 10 iterations
)
```

State files are saved to the `states/` directory by default.

### Resuming from Checkpoint

```python
sampler.run(
    n_total=4096,
    resume_state_path="states/ps_50.state",
)
```

---

## Extracting Results

### Posterior Samples

```python
# Get weighted samples
samples, weights, logl = sampler.posterior()

# Get resampled (unweighted) samples
samples, logl = sampler.posterior(resample=True)
```

### Evidence Estimate

```python
logz, logz_err = sampler.evidence()
print(f"log(Z) = {logz:.2f} ± {logz_err:.2f}")
```

### Full Results Dictionary

```python
results = sampler.results

# The results dictionary contains iteration history and samples
# Access samples and statistics as needed for analysis
```

---

## Working with Blobs

"Blobs" allow you to store additional quantities computed during likelihood evaluation:

```python
def log_likelihood_with_blobs(x):
    chi2 = np.sum((x - data)**2 / sigma**2)
    logl = -0.5 * chi2
    # Return tuple: (logl, blob1, blob2, ...)
    return logl, chi2

sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_with_blobs,
    n_dim=n_dim,
)

sampler.run()

# Access blobs
samples, weights, logl, blobs = sampler.posterior()
chi2_values = blobs  # Your stored quantities
```

!!! warning "Blobs and Vectorization"
    Blobs are not compatible with `vectorize=True`. Use scalar likelihood evaluation when you need blobs.

---

## Controlling MCMC Behavior

### MCMC Sampler Type

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    sample='tpcn',  # 't-preconditioned Crank-Nicolson' (default)
    # sample='rwm',  # Random Walk Metropolis
)
```

### MCMC Steps

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_steps=10,      # Steps after plateau detection
    n_max_steps=100, # Maximum steps per iteration
)
```

---

## Boundary Conditions

For parameters with special boundaries:

### Periodic Boundaries

Useful for phase-like parameters:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    periodic=[0, 2],  # Parameters 0 and 2 wrap around
)
```

### Reflective Boundaries

Useful for ratio parameters where a/b ≡ b/a:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    reflective=[1],  # Parameter 1 reflects at boundaries
)
```

---

## ESS Metrics

Control how the sampler determines temperature steps:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    metric='ess',  # Effective Sample Size (default)
    # metric='uss',  # Unique Sample Size
)
```

The `'uss'` metric can be more robust for multimodal distributions.

---

## Clustering

Enable hierarchical clustering for multimodal distributions:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    clustering=True,      # Enable clustering (default)
    normalize=True,       # Normalize before clustering
    split_threshold=1.0,  # BIC threshold modifier
)
```

Disable clustering for unimodal targets:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    clustering=False,
)
```

---

## Reproducibility

Set a random seed for reproducible results:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    random_state=42,
)
```

---

## Example: Full Configuration

```python
import numpy as np
import tempest as tp

n_dim = 10

def prior_transform(u):
    return 20 * u - 10

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

# Configure everything
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_effective=1024,
    n_active=512,
    n_boost=2048,
    vectorize=False,
    sample='tpcn',
    n_steps=n_dim,
    n_max_steps=10 * n_dim,
    clustering=True,
    normalize=True,
    metric='ess',
    resample='mult',
    output_dir='results',
    output_label='gaussian',
    random_state=42,
)

# Run with checkpointing
sampler.run(n_total=8192, save_every=20, progress=True)

# Get results
samples, weights, logl = sampler.posterior()
logz, logz_err = sampler.evidence()
```

---

## Next Steps

- **Understand the algorithm**: Read [How Persistent Sampling Works](../how_it_works.md) to understand the four-step pipeline
- **Configure priors**: Learn about [Prior Distributions](priors.md)
- **Scale up**: Explore [Parallelization](parallelization.md) for large problems
- **See examples**: Check out [Rosenbrock](../examples/rosenbrock.md) and other examples
- **Fine-tune**: See [Advanced Features](advanced.md) for optimization tips

State files will be saved as `my_results/analysis_1_*.state`.

---

## Reproducibility

Set a random seed for reproducible results:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    random_state=42,
)
```

---

## Example: Full Configuration

```python
import numpy as np
import tempest as tp

n_dim = 10

def prior_transform(u):
    return 20 * u - 10

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

# Configure everything
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_effective=1024,
    n_active=512,
    n_boost=2048,
    vectorize=False,
    sample='tpcn',
    n_steps=n_dim,
    n_max_steps=10 * n_dim,
    clustering=True,
    normalize=True,
    metric='ess',
    resample='mult',
    output_dir='results',
    output_label='gaussian',
    random_state=42,
)

# Run with checkpointing
sampler.run(n_total=8192, save_every=20, progress=True)

# Get results
samples, weights, logl = sampler.posterior()
logz, logz_err = sampler.evidence()
```
