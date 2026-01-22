# Parameter Selection Guide

This guide explains Tempest's parameters and provides heuristics for choosing them effectively.

## Overview & Philosophy

### Three Fundamental Rules

1. **Increasing `n_effective` reduces variance in the results**
   - Higher values capture more features but cost more computation

2. **Increasing `n_steps` reduces bias in the results**
   - More steps give better exploration but cost more computation

3. **Total computational cost scales linearly with `n_effective` and `n_steps`**
   - Doubling either parameter approximately doubles runtime

### Trade-offs

Tempest's parameters control three fundamental trade-offs:

- **Accuracy vs Speed**: More particles and steps give better results but cost more compute
- **Variance vs Bias**: `n_effective` reduces variance, `n_steps` reduces bias
- **Memory vs Performance**: Parallelization and vectorization speed up evaluation

Parameters fall into four categories:
- **Core sampling**: Control particle counts and boosting
- **MCMC proposal**: Control exploration per iteration
- **Clustering**: Handle multimodal distributions
- **Performance**: Parallelization and efficiency

## Core Sampling Parameters

These are the most important parameters for controlling Tempest's behavior.

### `n_effective` (target effective sample size)

**Purpose**: Determines the resolution capabilities of the sampler. Controls how many independent samples you aim to obtain.

**Why it matters**: This is one of the two most important parameters. Higher values better capture non-Gaussian features, multimodality, and high-dimensional structure. It directly affects result quality and variance.

**Scaling heuristics**:
- Low-dim (2-4D): 256-512
- Medium-dim (5-15D): 512-1024
- High-dim (20D+): 1024-2048 or more (scales roughly with D² - dimensionality squared)
- Complex posteriors (multimodal, skewed): Use upper end of range or higher

**Rule**: Start with 512 for 5-10D problems. For high dimensions (>20D), scale roughly with D².

### `n_active` (active particles)

**Purpose**: Number of particles updated in each iteration. Controls iteration granularity.

**Optimal value**: `n_effective // 2` is optimal for all problems. Not recommended to change this.

**Trade-off**: Lower values → more iterations but shorter; higher values → fewer iterations but longer. Must be less than `n_effective`.

### `n_total` (total samples to collect)

**Purpose**: Controls how many samples you get after reaching the posterior.

**How it works**: When beta reaches 1, you typically have only `n_effective` samples from the tempered sequence. Sampling continues at beta=1 to collect more samples. This is what `n_total` controls.

**Typical range**: 4000-10000 is more than enough for publication-quality plots, even for complex posteriors.

**Rule**: Use 4096-8192 for most problems. Only increase if you need very smooth posterior estimates.

### `n_boost` (dynamic particle boosting)

**Purpose**: Focus computational budget where it matters by increasing particles near the posterior.

**When to use**: When the prior is much wider than the posterior (common in practice). Start with low `n_effective`, boost near posterior.

**Impact**: Makes a huge difference for wide priors. Controls cost during early iterations when likelihood evaluations are cheap but many particles are needed for exploration.

**CRITICAL WARNING**: Do NOT use if you need evidence estimates (`logZ`). Only for posterior estimation.

**Rule**: Set to 2-4× `n_effective` for posterior-only analysis with wide priors. Leave as `None` otherwise.

## MCMC Proposal Parameters

Control how particles explore parameter space each iteration.

### `sample` (sampler type)

**Always use**: `"tpcn"` (t-preconditioned Crank-Nicolson)

**Why**: Exhibits superior scaling to high dimensions. Much more efficient than alternatives.

**Alternative**: `"rwm"` (Random-Walk Metropolis) is included only for testing. Scales as O(D²) and is not recommended for production use.

**Rule**: Set to `"tpcn"` and never change it.

### `n_steps` (MCMC steps per iteration)

**Purpose**: Determines how much particles explore in each iteration. One of the two most important parameters.

**Effect**: Controls bias in results. More steps = better exploration but slower runtime.

**Default**: `n_dim // 2` (half the dimensionality)

**When to increase**: If you suspect biased results. Signs include:
- Posterior looks constrained when it shouldn't be
- Traces show poor mixing
- Multiple runs give different results

**Rule**: Increase if results appear biased. Start with `n_dim // 2`, increase to `n_dim` or `2*n_dim` if needed.

**Relationship with efficiency**: The actual number of MCMC steps performed is `n_steps` scaled by the inverse of the efficiency (`eff`). If the efficiency drops, more steps are automatically performed to counter that. Nothing beyond that.

### `n_max_steps` (maximum MCMC steps)

**Purpose**: Upper bound on steps per iteration. Safety mechanism for plateau detection.

**Default**: `10 * n_steps`

**When to modify**: Only if you're an advanced user who understands the plateau detection algorithm.

**Rule**: Leave at default unless you know what you're doing.

## Clustering Parameters

Handle multimodal distributions automatically.

### `clustering`

**Always use**: `True`

**Why**: Highly automated routine that won't cluster unless necessary. No performance penalty for unimodal distributions. Essential for multimodal problems.

**Behavior**: Automatically detects modes and adapts proposals accordingly.

**Rule**: Always set to `True`. There's no downside.

### `split_threshold` (BIC threshold)

**Purpose**: Controls how easily clusters split (how easy to form new clusters).

**Default**: `1.0`

**When to change**: Only if you know what you're doing. Lower values (0.5-0.8) create more clusters. Higher values require stronger evidence to split.

**Rule**: Leave at default `1.0`.

### `n_max_clusters` (maximum clusters)

**Purpose**: Limits maximum number of clusters to prevent runaway cluster creation.

**Default**: `None` (automatic)

**When to set**: For very complex posteriors where you want to prevent excessive clustering.

**Rule**: Leave as `None` unless you need to enforce a hard limit.

### `cluster_every`

**Purpose**: Update cluster assignments every N iterations.

**Default**: `1` (every iteration)

**When to increase**: For cheap posteriors where clustering overhead matters.

**Rule**: Leave at `1` for most cases.

## Temperature Ladder Parameters

Control how tempering progresses from prior to posterior.

### `metric` (temperature step criterion)

**Always use**: `"ess"` (Effective Sample Size)

**Why**: Works well in all cases. Determines temperature steps based on importance weight variance.

**Alternative**: `"uss"` (Unique Sample Size) is more conservative but unnecessary in practice.

**Rule**: Set to `"ess"` and don't change it.

## Boundary Conditions

Handle special parameter boundaries.

### `periodic`

**When to use**: Phase-like parameters that wrap around (e.g., angles, phases).

**How**: Provide list of parameter indices `[0, 2]` for parameters that are periodic.

**Rule**: Only set if you have truly periodic parameters.

### `reflective`

**When to use**: Ratio parameters where symmetry exists (e.g., `a/b` treated same as `b/a`).

**How**: Provide list of parameter indices `[1]` for parameters that reflect at boundaries.

**Rule**: Only set if you have parameters with reflective symmetry.

## Performance Parameters

Control computational efficiency.

### `pool` (parallelization)

**Purpose**: Enable parallel likelihood evaluation.

**How to use**:
- Set to integer: `pool=4` creates 4 worker processes
- Set to pool object: Any object with `.map()` method (multiprocessing, MPI, Ray, etc.)

**Rule**: Use when likelihood evaluation is expensive. For cheap likelihoods, overhead may not be worth it.

### `vectorize` (batch evaluation)

**Purpose**: Indicates whether likelihood accepts batched inputs (faster).

**When to enable**: `True` for cheap analytical likelihoods

**When to disable**: `False` for expensive simulations or when using blobs

**Rule**: Enable if you can write a vectorized likelihood function (accepts 2D array, returns 1D array).

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
- **Solution**: Reduce `n_effective` (as long as it's large enough to capture posterior geometry), or enable `n_boost`

**Problem**: Missing modes in multimodal posterior
- **Solution**: Ensure `clustering=True`, increase `n_effective`, use wider priors

### The Two Most Important Parameters

Remember:
1. **`n_effective`**: Controls variance and resolution (scales with complexity)
2. **`n_steps`**: Controls bias and exploration (increase if biased)

Most issues can be solved by adjusting these two parameters.

### Verifying Convergence in Critical Analyses

For publication-quality or critical results, always verify convergence:

**Method**: Run two independent samplers with different settings:
- **Run 1**: Standard settings (`n_effective=512`, `n_steps=n_dim/2`)
- **Run 2**: More conservative settings (`n_effective=1024`, `n_steps=n_dim`)

**Verification**: Compare the resulting posteriors:
- If they match closely: Results are converged and reliable
- If they differ significantly: Neither has converged, increase both parameters further

**Rule**: When in doubt, run with doubled `n_effective` and `n_steps` and verify they give consistent results.

## Minimal Parameter Selection by Problem Type

| Problem Type | n_effective | n_steps | clustering | Other Notes |
|-------------|-------------|---------|------------|-------------|
| Simple, low-dim (2-4D) | 256-512 | n_dim/2 | True (safe) | n_total=4096 |
| Medium-dim (5-15D) | 512-1024 | n_dim/2 | True | Standard case |
| High-dim (20D+) | 1024-2048+ | n_dim | True | Scales ~D² |
| Multimodal | 1024+ | n_dim | True | May need more |
| Posterior-only | 256-512 | n_dim/2 | True | Use n_boost=4× |
| With evidence | 512-1024 | n_dim/2 | True | No n_boost |
| Expensive likelihood | Lower end | Higher | True | Use pool |
| Cheap likelihood | Higher | n_dim/2 | True | Use vectorize=True |

## Quick Reference

| Parameter | Default | When to Increase | When to Decrease | Critical? |
|-----------|---------|------------------|------------------|-----------|
| `n_effective` | 512 | High dim, complex, high variance | Cheap likelihood | Yes |
| `n_active` | 256 | Not recommended | Not recommended | No |
| `n_total` | 4096 | Need more samples | Quick testing | No |
| `n_boost` | None | Wide prior for posterior-only | N/A (Don't use) | No* |
| `n_steps` | n_dim/2 | Biased results | Speed critical | Yes |
| `n_max_steps` | 10×n_steps | Advanced users only | Advanced users only | No |
| `clustering` | True | Always True | Never | No |
| `split_threshold` | 1.0 | Don't change | Don't change | No |
| `metric` | 'ess' | Always 'ess' | Always 'ess' | No |
| `sample` | 'tpcn' | Always 'tpcn' | Always 'tpcn' | No |
| `pool` | None | Expensive likelihood | Cheap likelihood | No |
| `vectorize` | False | Cheap likelihood | Expensive or blobs | No |

*Do not use n_boost if you need evidence estimates