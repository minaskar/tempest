# Working with Blobs

This example demonstrates how to use "blobs" in Tempest to store auxiliary quantities computed during likelihood evaluation. Blobs allow you to cache expensive calculations, store diagnostic information, and compute derived quantities without re-evaluating the likelihood.

---

## What Are Blobs?

Blobs are additional return values from your likelihood function beyond the log-likelihood itself. When your likelihood computes quantities of interest (e.g., chi-squared, model predictions, physical parameters), you can return them as blobs and retrieve them later alongside your posterior samples.

**Key Benefits:**
- Avoid recomputing expensive quantities during analysis
- Store diagnostic information for debugging
- Compute derived quantities that depend on the likelihood evaluation

**Important Limitation:** Blobs are not compatible with `vectorize=True`. Use scalar likelihood evaluation when you need blobs.

---

## Problem Setup: 2D Gaussian Distribution

We'll demonstrate blobs using a simple 2D Gaussian likelihood where we can compute various auxiliary quantities.

**Mathematical Formulation:**
- Likelihood: $\mathcal{N}(\mu=[0, 0], \Sigma=I)$
- Prior: Uniform $[-10, 10]$ for both dimensions
- True parameters: $\mu = (0, 0)$

```python
import numpy as np
import tempest as tp

n_dim = 2

def prior_transform(u):
    """Uniform prior from unit hypercube to physical parameters."""
    return 20 * u - 10
```

---

## Case 1: Single Blob

Start with a simple likelihood that returns chi-squared as a single blob value.

```python
def log_likelihood_single_blob(x):
    """Likelihood returning chi-squared as a single blob."""
    chi2 = np.sum(x**2)
    logl = -0.5 * chi2
    return logl, chi2

# Create sampler with single blob
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_single_blob,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=float,  # Each blob is a scalar float
)

# Run the sampler
sampler.run(n_total=2048, progress=False)

# Retrieve posterior samples and blobs
samples, weights, logl, blobs = sampler.posterior(return_blobs=True)

# Expected shapes:
# samples.shape = (n_samples, n_dim) = (2048, 2)
# blobs.shape = (n_samples,) = (2048,)

# Compute weighted statistics from blobs
mean_chi2 = np.average(blobs, weights=weights)
std_chi2 = np.sqrt(np.average((blobs - mean_chi2)**2, weights=weights))

print(f"Mean chi-squared: {mean_chi2:.2f} ± {std_chi2:.2f}")
```

---

## Case 2: Multiple Blobs

Return multiple auxiliary quantities. We'll demonstrate both unnamed and named field approaches.

```python
def log_likelihood_multiple_blobs(x):
    """Likelihood returning multiple blob quantities."""
    chi2 = np.sum(x**2)
    radius = np.sqrt(chi2)
    max_abs = np.max(np.abs(x))
    logl = -0.5 * chi2
    return logl, chi2, radius, max_abs

# Approach 1: Unnamed fields (tuple unpacking)
# Blobs are returned as a numeric array
sampler_unnamed = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_multiple_blobs,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=(float, 3),  # 3 float values per sample
)

sampler_unnamed.run(n_total=2048, progress=False)
samples, weights, logl, blobs = sampler_unnamed.posterior(return_blobs=True)

# Expected: blobs.shape = (n_samples, 3) = (2048, 3)
chi2_values = blobs[:, 0]      # First field
radius_values = blobs[:, 1]    # Second field
max_abs_values = blobs[:, 2]   # Third field

# Approach 2: Named fields (recommended)
# Use numpy structured arrays for clarity
sampler_named = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_multiple_blobs,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=[
        ("chi2", float),
        ("radius", float),
        ("max_abs", float)
    ],
)

sampler_named.run(n_total=2048, progress=False)
samples, weights, logl, blobs = sampler_named.posterior(return_blobs=True)

# Expected: blobs.shape = (n_samples,) = (2048,)
# Each element is a structured numpy array

# Access fields by name
chi2_values = blobs["chi2"]      # Shape: (n_samples,)
radius_values = blobs["radius"]   # Shape: (n_samples,)
max_abs_values = blobs["max_abs"] # Shape: (n_samples,)

# Compute correlations with parameters
corr_x0_radius = np.corrcoef(samples[:, 0], radius_values)[0, 1]
print(f"Correlation between x[0] and radius: {corr_x0_radius:.3f}")
```

---

## Case 3: Higher-Dimensional Blob Arrays

Blobs can be multi-dimensional arrays, useful for storing per-parameter or per-data-point quantities.

```python
def log_likelihood_array_blob(x):
    """Likelihood returning array-valued blobs."""
    chi2_per_dim = x**2  # chi2 for each dimension
    chi2_total = np.sum(chi2_per_dim)
    logl = -0.5 * chi2_total
    return logl, chi2_per_dim

# Single array blob
sampler_array = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_array_blob,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=(float, n_dim),  # Each blob is length-n_dim array
)

sampler_array.run(n_total=2048, progress=False)
samples, weights, logl, blobs = sampler_array.posterior(return_blobs=True)

# Expected: blobs.shape = (n_samples, n_dim) = (2048, 2)
chi2_dim0 = blobs[:, 0]  # chi2 from first dimension
chi2_dim1 = blobs[:, 1]  # chi2 from second dimension

# Structured with mixed scalar and array fields
sampler_mixed = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_array_blob,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=[
        ("chi2_total", float),
        ("chi2_per_dim", float, n_dim)  # Array field
    ],
)

sampler_mixed.run(n_total=2048, progress=False)
samples, weights, logl, blobs = sampler_mixed.posterior(return_blobs=True)

# Expected: blobs.shape = (n_samples,) = (2048,)
# blobs["chi2_total"].shape = (n_samples,) = (2048,)
# blobs["chi2_per_dim"].shape = (n_samples, n_dim) = (2048, 2)

total_chi2 = blobs["chi2_total"]
chi2_per_dim = blobs["chi2_per_dim"]
```

---

## Case 4: Using Blob Data for Analysis

Practical examples of working with retrieved blobs.

```python
# Run the sampler with named blobs
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_multiple_blobs,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=[("chi2", float), ("radius", float), ("max_abs", float)],
)

sampler.run(n_total=2048, progress=False)
samples, weights, logl, blobs = sampler.posterior(return_blobs=True)

# Weighted statistics
mean_radius = np.average(blobs["radius"], weights=weights)
std_radius = np.sqrt(np.average((blobs["radius"] - mean_radius)**2, weights=weights))

print(f"Mean radius: {mean_radius:.2f} ± {std_radius:.2f}")

# Conditional analysis: examine parameters for high-radius samples
high_radius_threshold = np.percentile(blobs["radius"], 90)
high_radius_mask = blobs["radius"] > high_radius_threshold

samples_high_r = samples[high_radius_mask]
weights_high_r = weights[high_radius_mask]

mean_x0_high_r = np.average(samples_high_r[:, 0], weights=weights_high_r)
print(f"Mean x[0] for high-radius samples: {mean_x0_high_r:.3f}")

# Blob-only posterior (ignore parameters)
logz, logz_err = sampler.evidence()
chi2_mean, chi2_cov = tp.utils.weighted_avg_and_cov(blobs["chi2"], weights)
```

---

## Important Considerations

### Memory Usage
Blobs are stored for all active particles at every iteration. Memory scales as:
```
memory ≈ n_iterations × n_active × blob_size × dtype_bytes
```
For large blobs or long runs, consider periodic saving and clearing.

### Common Errors

**Wrong blob return shape:**
```python
# Wrong: return logl, [chi2, radius]  # List instead of tuple
# Wrong: return logl, chi2, radius    # Missing tuple wrapping if single value
# Correct: return logl, (chi2, radius)  # Always return tuple
```

**Forgetting `return_blobs=True`:**
```python
# Wrong: samples, weights, logl, blobs = sampler.posterior()
# Raises: ValueError: not enough values to unpack

# Correct: samples, weights, logl, blobs = sampler.posterior(return_blobs=True)
```

**Blob dtype mismatch:**
```python
# If likelihood returns 2 values but blobs_dtype expects 3
# Result: Runtime error or incorrect array shapes
```

### Best Practices

1. **Use structured dtypes**: Named fields (`blobs["chi2"]`) are clearer than indexing (`blobs[:, 0]`)

2. **Specify dtypes explicitly**: Prevents ambiguous array creation
```python
# Good: blobs_dtype=[("chi2", float)]
# Avoid: blobs_dtype=None  (infers from first result, may cause issues)
```

3. **Return numpy scalars/arrays**: Not Python lists
```python
# Good: return logl, np.array([chi2, radius])
# Avoid: return logl, [chi2, radius]
```

4. **Profile memory usage** for large blob arrays

5. **Consider post-processing trade-offs**: Blobs are ideal for expensive calculations, but simple quantities may be faster to recompute

### Compatibility Notes

- **Clustering**: Blobs work correctly with clustering enabled
- **MCMC moves**: Blobs are properly handled during mutation steps
- **Resampling**: Blobs are automatically resampled along with particles
- **Saving/Loading**: Blobs are preserved in sampler state files
- **Parallelization**: Works correctly with `pool` parameter

### Performance Trade-offs

**When to use blobs:**
- Likelihood evaluations are expensive
- Quantities require significant computation
- Need to analyze quantities at posterior parameter values
- Debugging or diagnostic information

**When to compute post-hoc:**
- Blob arrays would be extremely large
- Quantities are trivial to compute
- Only need quantities for subset of samples
- Memory-constrained environments

---

## Complete Working Example

Here is a complete, runnable example combining all blob concepts:

```python
import numpy as np
import tempest as tp

# Setup
n_dim = 2

def prior_transform(u):
    return 20 * u - 10

def log_likelihood_with_blobs(x):
    """Example likelihood returning structured blobs."""
    chi2_total = np.sum(x**2)
    chi2_per_dim = x**2
    radius = np.sqrt(chi2_total)
    max_deviation = np.max(np.abs(x))
    logl = -0.5 * chi2_total
    
    return logl, (chi2_total, radius, max_deviation, chi2_per_dim)

# Create sampler with multiple blob types
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood_with_blobs,
    n_dim=n_dim,
    n_effective=512,
    blobs_dtype=[
        ("chi2_total", float),
        ("radius", float),
        ("max_deviation", float),
        ("chi2_per_dim", float, n_dim)
    ],
    random_state=42,
)

# Run sampling
sampler.run(n_total=2048, progress=False)

# Retrieve results with blobs
samples, weights, logl, blobs = sampler.posterior(return_blobs=True)

# Expected shapes:
# samples.shape = (2048, 2)
# blobs.shape = (2048,)
# blobs["chi2_total"].shape = (2048,)
# blobs["chi2_per_dim"].shape = (2048, 2)

# Weighted statistics
print("=== Blob Statistics ===")
print(f"Mean chi2_total: {np.average(blobs['chi2_total'], weights=weights):.2f}")
print(f"Mean radius: {np.average(blobs['radius'], weights=weights):.2f}")
print(f"Mean max_deviation: {np.average(blobs['max_deviation'], weights=weights):.2f}")

# Extract per-dimension chi2 values
chi2_dim0 = blobs["chi2_per_dim"][:, 0]
chi2_dim1 = blobs["chi2_per_dim"][:, 1]

print(f"Mean chi2_dim0: {np.average(chi2_dim0, weights=weights):.2f}")
print(f"Mean chi2_dim1: {np.average(chi2_dim1, weights=weights):.2f}")

# Evidence and effective sample size
logz, logz_err = sampler.evidence()
print(f"\nLog-evidence: {logz:.2f} ± {logz_err:.2f}")
```

---

## Summary

Blobs provide a flexible mechanism to store auxiliary data from your likelihood function:

- **Simple blobs**: Single values (`blobs_dtype=float`)
- **Multiple blobs**: Structured arrays with named fields
- **Array blobs**: Multi-dimensional data per sample
- **Automatic handling**: Preserved during resampling and MCMC steps

The key is to define appropriate `blobs_dtype` that matches what your likelihood returns, and remember to use `return_blobs=True` when retrieving results.
