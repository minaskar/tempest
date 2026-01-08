# Advanced Features

This guide covers advanced features of Tempest for complex sampling problems.

## Dynamic Sampling

Tempest can dynamically adjust the number of effective particles based on the sampling progress.

### Enabling Dynamic Mode

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    dynamic=True,  # Default: True
)
```

With `dynamic=True`, Tempest automatically adjusts `n_effective` based on the effective sample size (ESS) of importance weights. This is particularly useful when:

- The target distribution is far from the prior
- There are strong correlations between parameters
- The distribution is multimodal

### Boosting Particles

Use `n_boost` to gradually increase particle count as sampling approaches the posterior:

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=512,
    n_boost=4,  # Up to 4x increase
)
```

This starts with fewer particles and increases them as the effective sample size improves, balancing computational cost with accuracy.

---

## Clustering for Multimodal Distributions

Tempest uses hierarchical Gaussian mixture clustering to handle multimodal distributions.

### Configuration

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    clustering=True,        # Enable clustering
    normalize=True,         # Normalize data before clustering
    split_threshold=1.0,    # BIC threshold for splits
    n_max_clusters=None,    # Maximum clusters (None = auto)
    cluster_every=1,        # Update clusters every N iterations
)
```

### When Clustering Helps

- **Multimodal posteriors**: Multiple well-separated peaks
- **Non-elliptical distributions**: Complex shapes requiring local proposals
- **Phase transitions**: When modes appear/disappear during tempering

### Viewing Cluster Information

The progress bar shows `K=` indicating the number of active clusters:

```
Iter: 50 [beta=0.85, K=3, ESS=512, ...]
```

---

## Step-by-Step Sampling

For fine-grained control, use the `sample()` method instead of `run()`:

```python
sampler = pc.Sampler(prior=prior, likelihood=log_likelihood)

# Initialize
sampler.iter = 0
sampler.calls = 0
sampler.beta = 0.0
sampler.logz = 0.0

# Custom sampling loop
while sampler.beta < 1.0:
    state = sampler.sample()
    
    # Access current state
    print(f"Iteration {state['iter']}")
    print(f"Beta: {state['beta']:.4f}")
    print(f"ESS: {state['ess']:.1f}")
    print(f"Log-evidence: {state['logz']:.2f}")
    
    # Custom stopping criteria
    if state['ess'] > 10000:
        break
    
    # Custom actions between iterations
    save_intermediate_results(state)
```

---

## Saving and Loading States

### Automatic Checkpointing

```python
sampler.run(
    n_total=4096,
    save_every=20,  # Save every 20 iterations
)
```

Files are saved to `{output_dir}/{output_label}_{iter}.state`.

### Manual Save/Load

```python
# Save current state
sampler.save_state("my_checkpoint.state")

# Later, in a new session
sampler = pc.Sampler(prior=prior, likelihood=log_likelihood)
sampler.load_state("my_checkpoint.state")
sampler.run()  # Continue from checkpoint
```

### State Contents

The state file contains:
- All particle positions and likelihoods
- Current temperature (beta)
- Evidence estimate
- Clustering information
- RNG state (for reproducibility)

---

## Importance Weight Trimming

Control how importance weights are processed:

```python
samples, weights, logl = sampler.posterior(
    trim_importance_weights=True,  # Remove high-weight outliers
    ess_trim=0.99,                 # Target ESS fraction
    bins_trim=1000,                # Binning for trimming
)
```

Trimming removes samples with extremely high weights that can dominate averages.

---

## Resampling Schemes

Control how particles are resampled between iterations:

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    resample='mult',  # Multinomial resampling (default)
    # resample='syst',  # Systematic resampling
)
```

- **Multinomial** (`'mult'`): Simple random resampling
- **Systematic** (`'syst'`): Lower variance, more uniform coverage

---

## Temperature Ladder Control

### Metric Selection

```python
sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    metric='ess',  # Effective Sample Size (default)
    # metric='uss',  # Unique Sample Size
)
```

The metric determines how temperature steps are chosen:
- **ESS**: Based on importance weight variance
- **USS**: Based on number of unique particles after resampling (more conservative)

---

## Handling Numerical Issues

### Likelihood Returns

Ensure your likelihood handles edge cases:

```python
def log_likelihood(x):
    if not is_physical(x):
        return -np.inf  # Reject unphysical parameters
    
    try:
        result = compute_likelihood(x)
    except Exception:
        return -np.inf  # Reject on numerical failures
    
    if not np.isfinite(result):
        return -np.inf
    
    return result
```

### Prior Bounds

Tempest works in the unit hypercube. Parameters outside [0, 1] after transformation indicate prior issues.

---

## Results Analysis

### Full Results Object

```python
results = sampler.results

# Iteration-wise data
beta_history = results['beta']
ess_history = results['ess']
acceptance_history = results['accept']
efficiency_history = results['efficiency']
logz_history = results['logz']

# Sample data
all_samples = results['x']  # Shape: (n_iter, n_active, n_dim)
all_logl = results['logl']
log_weights = results['logw']
```

### Computing Weighted Statistics

```python
samples, weights, logl = sampler.posterior()

# Weighted mean
mean = np.average(samples, weights=weights, axis=0)

# Weighted covariance
centered = samples - mean
cov = np.average(centered[:, :, np.newaxis] * centered[:, np.newaxis, :], 
                 weights=weights, axis=0)

# Weighted quantiles
from scipy.stats import median_abs_deviation
# Note: for weighted quantiles, use specialized functions
```

---

## Memory Management

For very long runs, manage memory by periodically saving and clearing:

```python
sampler = pc.Sampler(prior=prior, likelihood=log_likelihood)

for batch in range(10):
    sampler.run(n_total=1000, save_every=50)
    
    # Save results
    samples, weights, logl = sampler.posterior()
    np.save(f"samples_batch_{batch}.npy", samples)
    
    # Reset for next batch (if needed)
    # Note: This resets the sampler state
```

---

## Debugging

### Verbose Progress

The progress bar shows key diagnostics:
- `beta`: Current inverse temperature (0→1)
- `calls`: Total likelihood evaluations
- `ESS`: Effective sample size
- `logZ`: Evidence estimate
- `logL`: Mean log-likelihood
- `acc`: MCMC acceptance rate
- `steps`: MCMC steps per iteration
- `eff`: MCMC efficiency
- `K`: Number of clusters

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Very low ESS | Too few particles | Increase `n_effective` |
| Beta stuck near 0 | Poor prior/likelihood ratio | Check prior bounds |
| Low acceptance | Poor proposal | Enable clustering, check boundaries |
| K=1 always | Unimodal or clustering disabled | Normal for simple problems |
