# Advanced Parameter Guide

⚠️ **Advanced Users Only**

This guide covers advanced parameters for sophisticated sampling problems and power users. If you're just getting started with Tempest, you probably don't need to read this.

**Most users should only adjust two parameters: `n_particles` and `n_steps`. All other parameters can be left at their default values for almost all problems.**

If you've already tried standard parameters and need more control, or you're developing new sampling methods, this guide is for you.

---

## Core Sampling Parameters (Advanced)

### `n_active` (active particles)

**Purpose**: Number of particles updated in each iteration. Controls iteration granularity.

**Default**: This parameter has been removed. Use `n_particles` directly instead.

**Note**: The current API uses `n_particles` to control the number of particles. For parallelization, `n_particles` should be set to an integer multiple of the number of CPUs. Example: for 8 CPUs, use n_particles=256 (32 per CPU) or n_particles=224 (28 per CPU).

### `n_total` (total samples to collect)

**Purpose**: Controls how many samples you get after reaching the posterior. This is a parameter to `run()`, not `Sampler()`.

**How it works**: When beta reaches 1, sampling continues at beta=1 to collect more samples. This is what `n_total` controls.

**Typical range**: 4000-10000 is more than enough for publication-quality plots, even for complex posteriors.

**Rule**: Use 4096-8192 for most problems. Only increase if you need very smooth posterior estimates.

---

## MCMC Proposal Parameters

Control how particles explore parameter space each iteration.

### `sample` (sampler type)

**Default**: `"tpcn"` (t-preconditioned Crank-Nicolson)

**Options**:
- `"tpcn"`: t-preconditioned Crank-Nicolson (recommended)
- `"rwm"`: Random-Walk Metropolis (for testing only)

**Why tpcn?**: Exhibits superior scaling to high dimensions. Much more efficient than alternatives.

**Rule**: Set to `"tpcn"` unless you have a specific reason not to.

### `n_max_steps` (maximum MCMC steps per dimension)

**Purpose**: Upper bound on steps per iteration. The actual maximum is `n_max_steps * n_dim`.

**Default**: `20 * n_steps` (where `n_steps` is the per-dimension base value)

**When to modify**: Only if you need to hard-limit the adaptive step calculation for very expensive likelihoods.

**Rule**: Leave at default unless you know what you're doing.

---

## Clustering Parameters

Handle multimodal distributions automatically.

### `split_threshold` (BIC threshold)

**Purpose**: Controls how easily clusters split (how easy to form new clusters).

**Default**: `1.0`

**When to change**: If the automatic clustering is too aggressive or not aggressive enough. Lower values (0.5-0.8) create more clusters. Higher values require stronger evidence to split.

**Rule**: Leave at default `1.0` unless you have specific clustering behavior issues.

### `n_max_clusters` (maximum clusters)

**Purpose**: Limits the maximum number of clusters to prevent runaway cluster creation.

**Default**: `None` (automatic)

**When to set**: For very complex posteriors where you want to enforce a hard limit.

**Rule**: Leave as `None` unless you need to enforce a hard limit.

### `cluster_every`

**Purpose**: Update cluster assignments every N iterations.

**Default**: `1` (every iteration)

**When to increase**: For cheap posteriors where clustering overhead matters.

**Rule**: Leave at `1` for most cases.

---



## Boundary Conditions

Handle special parameter boundaries.

### `periodic`

**When to use**: Phase-like parameters that wrap around (e.g., angles, phases).

**How**: Provide list of parameter indices `[0, 2]` for parameters that are periodic.

**Rule**: Only set if you have truly periodic parameters.

**Example**:
```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=6,
    periodic=[0, 3],  # Parameters 0 and 3 are periodic
)
```

### `reflective`

**When to use**: Ratio parameters where symmetry exists (e.g., `a/b` treated same as `b/a`).

**How**: Provide list of parameter indices `[1]` for parameters that reflect at boundaries.

**Rule**: Only set if you have parameters with reflective symmetry.

**Example**:
```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=5,
    reflective=[2],  # Parameter 2 has reflective symmetry
)
```

---

## Performance Parameters

Control computational efficiency.

### `pool` (parallelization)

**Purpose**: Enable parallel likelihood evaluation.

**How to use**:
- Set to integer: `pool=4` creates 4 worker processes
- Set to pool object: Any object with `.map()` method (multiprocessing, MPI, Ray, etc.)

**When to use**: When likelihood evaluation is expensive (simulations, complex calculations).

**Rule**: Use when likelihood evaluation is expensive. For cheap likelihoods, overhead may not be worth it.

**Example**:
```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    pool=8,  # Use 8 parallel workers
)
```

### `vectorize` (batch evaluation)

**Purpose**: Indicates whether likelihood accepts batched inputs (faster).

**When to enable**: `True` for cheap analytical likelihoods (vectorized numpy operations)

**When to disable**: `False` for expensive simulations or when using blobs

**Rule**: Enable if you can write a vectorized likelihood function (accepts 2D array, returns 1D array).

**Example**:
```python
# Vectorized likelihood (fast)
def log_likelihood(x):
    # x shape: (n_samples, n_dim)
    return -0.5 * np.sum(x**2, axis=1)  # Returns (n_samples,)

sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    vectorize=True,  # Enable batch evaluation
)
```

**Note**: Using `blobs` (auxiliary data) automatically disables vectorization.

---

## Diagnostics and Troubleshooting

### Reading the Progress Bar

The progress bar shows key diagnostics:
- `beta`: Current inverse temperature (0→1)
- `calls`: Total likelihood evaluations
- `ESS`: Effective sample size
- `logZ`: Evidence estimate
- `logL`: Mean log-likelihood
- `acc`: MCMC acceptance rate
- `steps`: MCMC steps performed
- `eff`: MCMC efficiency (most useful diagnostic)
- `K`: Number of clusters

### Key Diagnostic: `eff` (MCMC efficiency)

**What it means**: Measures how effectively MCMC is exploring parameter space.

**When to worry**: If `eff < 0.1`, this may indicate problems.

**What to do**:
- If biased results: **Increase `n_steps`**
- If high variance: **Increase `n_effective`**
- If both: Adjust both parameters

### Common Parameter-Related Issues

**Problem**: Results vary between runs
- **Solution**: Increase `n_effective` (reduces variance)

**Problem**: Posterior looks too constrained
- **Solution**: Increase `n_steps` (reduces bias)

**Problem**: Slow progress, many iterations
- **Solution**: Reduce `n_particles` (as long as it's large enough to capture posterior geometry)

**Problem**: Missing modes in multimodal posterior
- **Solution**: Ensure `clustering=True`, increase `n_particles`, use wider priors

---

## Verifying Convergence in Critical Analyses

For publication-quality or critical results, always verify convergence:

**Method**: Run two independent samplers with different settings:
- **Run 1**: Standard settings (`n_particles=512`, `n_steps=5`)
- **Run 2**: More conservative settings (`n_particles=1024`, `n_steps=10`)

**Verification**: Compare the resulting posteriors:
- If they match closely: Results are converged and reliable
- If they differ significantly: Neither has converged, increase both parameters further

**Rule**: When in doubt, run with doubled `n_particles` and `n_steps` and verify they give consistent results.

---

## Return to Standard Parameters

If you find yourself overwhelmed by these advanced parameters, **go back to the standard Parameter Selection Guide**. For 99% of problems, you only need:

- **`n_particles`**: Controls variance (scales with complexity)
- **`n_steps`**: Controls bias (increase if biased)

Start simple, add complexity only when needed.

[Back to Standard Parameter Selection Guide](parameter_selection.md)
