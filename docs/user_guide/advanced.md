# Advanced Features

This guide covers advanced features of Tempest for complex sampling problems.

## Boosting Particles

Use `n_boost` to gradually increase particle count as sampling approaches the posterior:

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=512,
    n_boost=2048,  # Target 2048 effective particles
)
```

This starts with 512 effective particles and gradually increases to 2048 as sampling converges, balancing computational cost with accuracy. Set `n_boost=n_effective` to disable boosting, or `n_boost=None` to use the default behavior (no boosting).

---

## Clustering for Multimodal Distributions

Tempest uses hierarchical Gaussian mixture clustering to handle multimodal distributions.

### Configuration

```python
sampler = tp.Sampler(
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

For fine-grained control with checkpointing:

```python
sampler = tp.Sampler(prior=prior, likelihood=log_likelihood)

# Run with save_every to create checkpoints
sampler.run(n_total=1000, save_every=10)

# Process results in batches
for i in range(0, sampler.results['niter'], 10):
    samples, weights, logl = sampler.posterior()
    # Analyze intermediate results
    save_intermediate_results(samples, weights, logl)
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
sampler = tp.Sampler(prior=prior, likelihood=log_likelihood)
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
sampler = tp.Sampler(
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
sampler = tp.Sampler(
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

### Computing Weighted Statistics

```python
samples, weights, logl = sampler.posterior()

# Weighted mean
mean = np.average(samples, weights=weights, axis=0)

# Weighted covariance
centered = samples - mean
cov = np.average(centered[:, :, np.newaxis] * centered[:, np.newaxis, :], 
                 weights=weights, axis=0)

# Weighted quantiles - use specialized functions for weighted data
```

---

## Memory Management

For very long runs, manage memory by periodically saving and clearing:

```python
sampler = tp.Sampler(prior=prior, likelihood=log_likelihood)

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
