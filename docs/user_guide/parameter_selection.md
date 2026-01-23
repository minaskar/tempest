# Parameter Selection Guide

This guide explains Tempest's parameters and provides heuristics for choosing them effectively.

**For most users, you only need to worry about TWO parameters: `n_effective` and `n_steps`. All other parameters can (and should) be left at their default values for 99% of problems.**

If you're just getting started or have a standard inference problem, focus on this guide. For advanced features and more control, see the [Advanced Parameter Guide](parameter_selection_advanced.md).

## Overview & Philosophy

### Three Fundamental Rules

1. **Increasing `n_effective` reduces variance in the results**
   - Higher values capture more features but cost more computation

2. **Increasing `n_steps` reduces bias in the results**
   - More steps give better exploration but cost more computation

3. **Total computational cost scales linearly with `n_effective` and `n_steps`**
   - Doubling either parameter approximately doubles runtime

### The Two Most Important Parameters

Remember:
1. **`n_effective`**: Controls variance and resolution (scales with complexity)
2. **`n_steps`**: Controls bias and exploration (increase if biased)

Most issues can be solved by adjusting these two parameters.

---

## Core Parameters Everyone Should Know

### `n_effective` (target effective sample size) [MOST IMPORTANT]

**Purpose**: Determines the resolution capabilities of the sampler. Controls how many independent samples you aim to obtain. This is one of the two most important parameters.

**Why it matters**: Higher values better capture non-Gaussian features, multimodality, and high-dimensional structure. It directly affects result quality and variance.

**Scaling heuristics**:
- Low-dim (2-4D): 256-512
- Medium-dim (5-15D): 512-1024
- High-dim (20D+): 1024-2048 or more (scales roughly with D² - dimensionality squared)
- Complex posteriors (multimodal, skewed): Use upper end of range or higher

**Rule**: Start with 512 for 5-10D problems. For high dimensions (>20D), scale roughly with D².

**When to increase**: If you see high variance between runs or need to capture more detailed features.

**When to decrease**: For quick testing or when likelihood evaluations are extremely expensive and you can tolerate more uncertainty.

### `n_steps` (MCMC steps per dimension) [SECOND MOST IMPORTANT]

**Purpose**: Determines how much particles explore in each iteration. This is the second most important parameter.

**Why it matters**: Controls bias in results. More steps = better exploration but slower runtime.

**New behavior** (as of recent versions):
- `n_steps` now represents **steps per dimension** (not absolute steps)
- The actual number adapts automatically based on sampler performance
- Adaptive formula: `n_steps_0 * n_dim * (0.234/acceptance_rate) * (sigma_0/sigma)**2`
- Minimum: `n_steps_0 * n_dim`
- Maximum: `n_max_steps_0 * n_dim` (where `n_max_steps_0 = 20 * n_steps_0`)

**Default**: `5` (per dimension)

**Rule**: For most problems, the default `5` works well. If you suspect biased results, increase to `10` or `20`.

**When to increase** (signs of bias):
- Posterior looks constrained when it shouldn't be
- Traces show poor mixing
- Multiple runs give different results
- Efficiency (`eff`) in progress bar is very low (< 0.1)

**When to decrease**: For quick testing when you're confident results are converged.

**Note on efficiency**: If MCMC efficiency drops, the adaptive mechanism automatically increases steps to compensate. You typically don't need to worry about this.

---

## Quick Start by Problem Type

| Problem Type | n_effective | n_steps | Notes |
|-------------|-------------|---------|-------|
| **Simple, low-dim (2-4D)** | 256-512 | 5 | Standard defaults work well |
| **Medium-dim (5-15D)** | 512-1024 | 5 | Most common case |
| **High-dim (20D+)** | 1024-2048+ | 5-10 | Start with 5, increase if needed |
| **Multimodal** | 1024+ | 5-10 | May need more sampling |
| **Posterior-only** | 256-512 | 5 | Consider n_boost (see Advanced) |
| **With evidence** | 512-1024 | 5 | No n_boost (see Advanced) |
| **Expensive likelihood** | Lower end | 5 | Use pool=4-8 for parallelization |
| **Cheap likelihood** | Higher | 5 | May help with speed |

**Starting point for most problems**: `n_effective=512`, `n_steps=5` (the defaults)

---

## Verifying Convergence in Critical Analyses

For publication-quality or critical results, always verify convergence:

**Method**: Run two independent samplers with different settings:
- **Run 1**: Standard settings (`n_effective=512`, `n_steps=5`)
- **Run 2**: More conservative settings (`n_effective=1024`, `n_steps=10`)

**Verification**: Compare the resulting posteriors:
- If they match closely: Results are converged and reliable
- If they differ significantly: Neither has converged, increase both parameters further

**Rule**: When in doubt, run with doubled `n_effective` and `n_steps` and verify they give consistent results.

---

## Common Issues and Solutions

### Problem: Results vary between runs
- **Symptom**: Each time you run, you get different posterior estimates
- **Solution**: Increase `n_effective` to reduce variance
- **Try**: Double `n_effective` (e.g., from 512 to 1024)

### Problem: Posterior looks too constrained or biased
- **Symptom**: The posterior seems narrower than expected, or traces show poor mixing
- **Solution**: Increase `n_steps` to reduce bias and improve exploration
- **Try**: Increase `n_steps` from 5 to 10 or 20

### Problem: Slow runtime
- **Symptom**: Sampling takes too long
- **Solution**: If results look converged, reduce `n_effective` or use parallelization with `pool=4-8`
- **Try**: Temporarily reduce `n_effective` to 256 for testing

### Problem: Unsure if converged
- **Solution**: Run two independent samplers with different random seeds and compare
- If they match closely, you're likely converged
- If different, increase parameters

---

## When to Use Advanced Parameters

**The standard parameters above (n_effective and n_steps) are sufficient for 99% of problems.**

You **only** need the [Advanced Parameter Guide](parameter_selection_advanced.md) if:

1. You're optimizing for extreme performance (expensive likelihood evaluations)
2. You need parallelization with specific CPU configurations
3. You're doing posterior-only analysis with `n_boost`
4. You're dealing with special boundary conditions (periodic or reflective)
5. You want to experiment with clustering control
6. You suspect very specific technical issues

**Rule**: Always try standard parameters first. Only explore advanced settings after you've ruled out parameter tuning issues.

---

## Quick Reference (Essential Only)

| Parameter | Default | When to Increase | When to Decrease | Critical? |
|-----------|---------|------------------|------------------|-----------|
| `n_effective` | 512 | High dimension, complex, variance issues | Testing only | **Yes** |
| `n_steps` | 5 (per dim) | Bias, poor mixing, efficiency < 0.1 | Testing, speed | **Yes** |

For other parameters, see the [Advanced Parameter Guide](parameter_selection_advanced.md).

---

## That's It!

Most users will never need more than what's in this guide. Start with the defaults:
- `n_effective=512`
- `n_steps=5`

Adjust only if needed based on the guidance above. For more control and advanced features, see the [Advanced Parameter Guide](parameter_selection_advanced.md).
