# Clustering

The `cluster` module provides hierarchical Gaussian mixture clustering for handling multimodal distributions.

## Overview

Tempest uses clustering to:

- Identify multiple modes in the posterior
- Construct local proposals for each cluster
- Improve MCMC efficiency for multimodal targets

The clustering is based on a hierarchical Gaussian mixture model that adaptively splits clusters based on the Bayesian Information Criterion (BIC).

---

## Classes Reference

### GaussianMixture

::: tempest.cluster.GaussianMixture
    options:
      members:
        - __init__
        - fit
      show_root_heading: true
      show_source: true
      heading_level: 4

### HierarchicalGaussianMixture

::: tempest.cluster.HierarchicalGaussianMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## How Clustering Works

### Hierarchical Splitting

1. Start with all samples in one cluster
2. Fit a 2-component GMM to each cluster
3. Accept split if BIC improves (adjusted by `split_threshold`)
4. Repeat until no beneficial splits remain

### Usage in PS

During sampling:

1. Samples are clustered based on their positions
2. Each cluster gets its own proposal distribution (multivariate Student-t)
3. MCMC proposals are made within each cluster
4. Cluster assignments are updated periodically

---

## Configuration

Control clustering through `Sampler` parameters:

```python
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    clustering=True,          # Enable clustering
    normalize=True,           # Normalize data before clustering
    split_threshold=1.0,      # BIC threshold modifier
    n_max_clusters=None,      # Maximum clusters (None = auto)
    cluster_every=1,          # Update frequency
)
```

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `clustering` | True | Enable/disable clustering |
| `normalize` | True | Normalize to [0,1]^D before clustering |
| `split_threshold` | 1.0 | Higher = fewer clusters, lower = more |
| `n_max_clusters` | None | Maximum number of clusters |
| `cluster_every` | 1 | Update clusters every N iterations |

---

## Monitoring Clusters

The progress bar shows `K=` indicating active clusters:

```
Iter: 50 [beta=0.85, K=3, ESS=512, ...]
```

Access cluster information programmatically:

```python
# After running
n_clusters = sampler.K
cluster_means = sampler.means
cluster_covs = sampler.covariances
```

---

## When to Use Clustering

### Enable Clustering When:

- Target has multiple modes
- Posterior has complex, non-elliptical shape
- Marginal distributions are multimodal or banana-shaped

### Disable Clustering When:

- Target is unimodal and well-approximated by a Gaussian
- Low-dimensional problems (< 3D)
- Speed is critical and target is simple

```python
# Disable for simple problems
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    clustering=False,
)
```

---

## Tips

### Adjusting Split Sensitivity

- `split_threshold > 1.0`: Harder to split, fewer clusters
- `split_threshold < 1.0`: Easier to split, more clusters
- `split_threshold = 0.5`: Aggressive splitting

### High-Dimensional Clustering

For high dimensions, clustering becomes harder:

```python
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    clustering=True,
    normalize=True,      # Important for scaling
    split_threshold=0.8, # More permissive
)
```
