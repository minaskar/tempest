# How Persistent Sampling Works

**Understanding the Tempest Algorithm Inside and Out**

This guide explains how Persistent Sampling (PS) works, from the high-level concepts to the mathematical details. Each section is written at two levels: a beginner-friendly explanation and an advanced treatment for researchers. Look for the info boxes for deeper technical content.

---

## Table of Contents

- [1. Introduction & Motivation](#1-introduction-motivation)
- [2. Core Algorithm Overview](#2-core-algorithm-overview)
- [3. The Four-Step Pipeline](#3-the-four-step-pipeline)
  - [3.1 Reweighting: Finding the Next Temperature](#31-reweighting-finding-the-next-temperature)
  - [3.2 Training: Learning the Distribution Structure](#32-training-learning-the-distribution-structure)
  - [3.3 Resampling: Selecting the Active Set](#33-resampling-selecting-the-active-set)
  - [3.4 Mutation: Evolving Particles with MCMC](#34-mutation-evolving-particles-with-mcmc)
- [4. Advanced Features Deep Dive](#4-advanced-features-deep-dive)
  - [4.1 Boosting: Smart Resource Allocation](#41-boosting-smart-resource-allocation)
  - [4.2 Evidence Estimation](#42-evidence-estimation)
  - [4.3 Clustering for Multimodal Distributions](#43-clustering-for-multimodal-distributions)
- [5. Comparison with Other Methods](#5-comparison-with-other-methods)
- [6. Understanding Diagnostics](#6-understanding-diagnostics)

---

## 1. Introduction & Motivation

### Why Bayesian Inference is Hard

Bayesian inference lets us update our beliefs about parameters after seeing data. The goal is to find the posterior distribution - the probability of parameters given observations. This is computationally challenging for two main reasons:

1. **The Curse of Dimensionality**: As you add parameters, parameter space volume grows exponentially. Most of that space contains terrible solutions. Finding good ones is like searching for a needle in a high-dimensional haystack.

2. **Multimodality**: The posterior often has multiple separate peaks (modes). Standard methods explore one mode and get stuck, missing other plausible solutions.

### Traditional Methods and Their Limitations

**MCMC (Markov Chain Monte Carlo)** explores parameter space via random-walk proposals, accepting moves that improve posterior probability. It efficiently samples unimodal distributions but struggles with multiple modes and strongly correlated parameters.

**Nested Sampling** builds from the prior but doesn't reuse information efficiently, wasting computational work.

### The Persistent Sampling Innovation

Persistent Sampling is a Sequential Monte Carlo method that maintains a population of particles evolving via persistent proposals - starting from previous positions rather than drawing fresh samples. This combines the best of both worlds:

- Particle system with tempering for multimodal exploration
- Persistent proposals for efficient local exploration  
- Adaptive proposals that learn posterior geometry

The result is faster convergence, better multimodal handling, and built-in evidence estimation.

!!! info "Advanced: Mathematical Formulation"
    
    The fundamental problem is computing expectations under the posterior: E[θ] = ∫ θπ(θ|D)dθ
    
    Standard MCMC constructs a Markov chain with spectral gap that deteriorates exponentially with dimension for simple random-walk proposals.
    
    Persistent Sampling uses a tempering path: π_β(θ) ∝ L(θ)^β p(θ)^(1-β) for β in [0,1]. The particle approximation converges to π_β with O(N^(-1/2)) error regardless of dimension, achieving polynomial complexity.

---

## 2. Tempering: Building the Distribution Sequence

### Why Tempering is Necessary

Directly sampling from complex posterior distributions is extremely difficult. The posterior often has:

- **Multiple modes** (separate peaks)
- **Narrow valleys** (strong correlations)
- **High curvature** (geometry changes dramatically)

Trying to sample directly is like parachuting into a foggy mountain range at night—you're likely to land in a poor location and get stuck.

**Tempering** solves this by creating a smooth path from a simple distribution (the prior) to the complex posterior, allowing gradual adaptation.

### How Tempering Works

The key idea is to introduce a **temperature parameter β** (beta) that smoothly interpolates between distributions:

- **β = 0**: Prior distribution π(θ) - easy to sample from
- **β = 1**: Posterior distribution π(θ|D) = L(θ)π(θ)/Z - our target
- **0 < β < 1**: Intermediate distributions that gradually increase likelihood influence

This creates a sequence of distributions: π₀, π₀.₁, π₀.₂, ..., π₀.₉, π₁.₀

**Why this helps**:
- Start with simple, well-understood prior
- Slowly discover posterior structure
- Never overwhelmed by sudden complexity
- Particles naturally flow toward high-probability regions

### Temperature Schedule Adaptation

Persistent Sampling automatically determines the temperature schedule using the Effective Sample Size (ESS):

1. At each iteration, find β_next such that ESS ≈ target (default: 512)
2. If temperature step is too large → ESS drops drastically
3. If temperature step is too small → waste computation
4. Use bisection to find optimal β_next

This ensures smooth, stable progression from prior to posterior.

!!! info "Advanced: Mathematical Formulation of Tempering"
    
    **Tempering Path**: The sequence of distributions is defined by:
    
    $$\pi_\beta(\theta) = \frac{\mathcal{L}(\theta)^\beta \, \pi(\theta)^{1-\beta}}{\mathcal{Z}_\beta}$$
    
    where:
    - $\mathcal{L}(\theta)$ is the likelihood
    - $\pi(\theta)$ is the prior  
    - $\mathcal{Z}_\beta$ is the normalizing constant
    - $\beta \in [0,1]$ is the inverse temperature
    
    **Properties**:
    - At $\beta=0$: $\pi_0(\theta) = \pi(\theta)$ (pure prior)
    - At $\beta=1$: $\pi_1(\theta) = \frac{\mathcal{L}(\theta)\pi(\theta)}{\mathcal{Z}}$ (posterior)
    - For intermediate β: Smooth interpolation between prior and posterior
    
    **Temperature Adaptation**: The next temperature $\beta_{t+1}$ is chosen to satisfy:
    
    $$\beta_{t+1} = \inf\{\beta \in [\beta_t, 1] : \text{ESS}(\beta) \geq \alpha N\}$$
    
    where ESS is computed from importance weights $w_i \propto \mathcal{L}(\theta_i)^{\beta - \beta_t}$, and α is the target fraction (typically 0.5-0.9).
    
    **Why Tempering Mathematically**: The KL divergence between successive distributions is:
    
    $$D_{KL}(\pi_{\beta_t} \|\ \pi_{\beta_{t+1}}) = (\beta_{t+1} - \beta_t) \cdot \text{Var}_{\pi_{\beta_t}}[\log \mathcal{L}(\theta)] + O((\beta_{t+1} - \beta_t)^2)$$
    
    This shows that small temperature steps → small KL divergence → stable importance weights.

### The Persistent Sampling Advantage

Standard Sequential Monte Carlo (SMC) also uses tempering, but PS improves upon it by:

- **Retaining all historical particles** rather than discarding them
- **Using multiple importance sampling** across the entire history
- **Resampling from a growing pool** of (t-1)×N particles instead of just N

This means PS has more information at each temperature level, leading to better proposals and lower variance estimates.

---

## 3. Core Algorithm Overview

### The Four-Step Cycle

Persistent Sampling iterates through four steps that transform samples from prior to posterior:

```
Prior (β=0) → [Reweight → Train → Resample → Mutate] → Posterior (β=1)
```

**Key Concepts**:

- **Tempering**: Gradually increase β from 0 to 1, letting the algorithm adapt slowly (see Section 2 for details)
- **Importance Sampling**: Particles have weights bridging successive temperatures
- **Persistent Proposals**: Start from previous positions, reusing computational work
- **Particle System**: Maintain dozens/hundreds of particles exploring simultaneously

### Multiple Importance Sampling (The "Persistence" Innovation)

**What makes PS different from standard SMC?**

Standard SMC only uses particles from the **previous iteration**. Persistent Sampling uses **all historical particles** through **multiple importance sampling (MIS)**.

**How it works**:

1. At iteration t, we have particles from iterations 1 through t-1
2. Each particle gets a weight comparing its target density with the mixture of all historical distributions
3. The weight formula uses the **balance heuristic** for optimal combination:

   $$ \text{logw}_s = \beta_{\text{final}} \cdot \text{logl}_s - \log\left( \sum_{t} \frac{n_t}{N} \cdot e^{\beta_t \cdot \text{logl}_s - \text{logZ}_t} \right) $$

   where:
   - $\beta_{\text{final}}$ is the target temperature
   - $\text{logl}_s$ is the particle's log-likelihood
   - $n_t/N$ weights each iteration by its particle count
   - The denominator is the **mixture density** of all historical distributions

   This optimally combines samples from multiple distributions without additional likelihood evaluations.

**Why this helps**:

- **More particles**: Instead of N particles, we have (t-1)×N particles
- **Better coverage**: Historical particles explored different regions
- **Lower variance**: More particles → more accurate estimates
- **No extra cost**: Likelihood values are cached from previous iterations

**The key insight**: Treating all historical particles as a **mixture distribution** gives us a rich, diverse sample pool that standard SMC throws away.

### Pseudocode for the Main Algorithm

```python
# Initialize from prior
for i in range(n_particles):
    u[i] = uniform(0, 1, size=n_dim)
    x[i] = prior_transform(u[i])
    logl[i] = log_likelihood(x[i])

beta = 0.0
while beta < 1:
    beta_next = find_next_beta(beta, particles, target_ESS=512)
    proposal = fit_gaussian_mixture(all_historical_particles)
    active = resample_with_weights(particles, weights)
    for particle in active:
        particle = mcmc_mutation(particle, proposal, beta_next)
    beta = beta_next
    save_to_history(active)

return samples, log_evidence
```

!!! info "Advanced: SMCS and Persistent Proposals"
    
    Persistent Sampling fits into the Sequential Monte Carlo Sampler framework. Standard SMCS uses independent proposals: q_t(θ_t) = q(θ_t)
    
    PS uses conditional proposals: q_t(θ_t|θ_(t-1)) = K_βt(θ_t|θ_(t-1))
    
    where K_βt is a Markov kernel with invariant distribution π_βt. This simple change addresses particle degeneracy by ensuring particles remain in high-probability regions.

---

## 3. The Four-Step Pipeline

### 3.1 Reweighting: Finding the Next Temperature

**Goal**: Determine β_(next) such that current and new distributions aren't too far apart.

**Why it matters**: Jump too far → near-zero weights. Jump too little → waste iterations.

**How it works**:

1. Start with current β
2. Propose candidate β > current
3. Compute importance weights: w_i ∝ L(θ_i)^(β_candidate - β)
4. Calculate Effective Sample Size (ESS)
5. Adjust β using bisection until ESS ≈ target (default: 512)

**Output**: β_(next) and importance weights for resampling

**Progress bar indicators**: beta (0→1), ESS (target: 512)

**When it helps**: Controls temperature progression to maintain particle diversity

**When it struggles**: With extreme prior-likelihood scale mismatches

!!! info "Advanced: Effective Sample Size and Bisection"
    
    **Effective Sample Size**: ESS = 1/Σw_i^2. Maximum N (all equal), minimum 1 (one dominates).
    
    **Bisection Algorithm**: Guarantees O(log tolerance) convergence. Adaptive boosting increases target ESS near posterior via sigmoid curve, allocating particles where most valuable.

---

### 3.2 Training: Learning the Distribution Structure

**Goal**: Build a model of the current distribution for intelligent MCMC proposals.

**Why it matters**: Blind random-walk MCMC is inefficient. Knowing where probability mass is lets us propose high-probability moves.

**How it works**:

1. **Collect historical particles**: All accumulated so far
2. **Assign importance weights**: Weight by current β
3. **Fit hierarchical Gaussian mixture**: Start with one component, iteratively split where data justify it, stop when BIC indicates no improvement
4. **Build mode statistics**: For each component, compute mean, covariance, weight, and degrees of freedom

**Output**: ModeStatistics object with proposal distributions for each cluster

**Progress bar indicator**: K (number of active clusters discovered)

**When it helps**: Multimodal posteriors, non-elliptical shapes, phase transitions

**When to disable**: Unimodal distributions (unnecessary overhead), very high dimensions

!!! info "Advanced: Hierarchical Clustering and Student-t Fitting"
    
    **Hierarchical Gaussian Mixture**: Agglomerative clustering with BIC stopping criterion. Splitting threshold controls aggressiveness.
    
    **Student-t Fitting**: Each component uses multivariate Student-t: p(θ|μ,Σ,ν) for heavy tails. Degrees of freedom ν fit via maximum likelihood.
    
    **Mode Statistics Structure**: Contains n_modes, means[K,d], covariances[K,d,d], weights[K], dfs[K], assignments[N]. Enables per-cluster MCMC with adaptive step sizes.

---

### 3.3 Resampling: Selecting the Active Set

**Goal**: Select which particles to evolve via MCMC in the next iteration.

**Why it matters**: Focus computation on particles that represent the current distribution well.

**How it works**:

**If β = 0 (warmup)**:
- Skip resampling
- Draw fresh prior samples (ensures diverse initial exploration)

**If β > 0**:
1. Compute importance weights for all historical particles
2. Normalize weights to sum to 1
3. Resample n_active particles with replacement according to weights
   - **Multinomial**: Simple random (higher variance)
   - **Systematic**: Deterministic stratified (lower variance, default)
4. Resample in u-space: Unit hypercube coordinates first, then transform

**Output**: New active particles ready for MCMC mutation

**Progress bar indicator**: N/A (happens every iteration)

**When it helps**: Focuses computation on high-probability regions

**When to tune**: Can adjust resample='syst' vs 'mult' if seeing issues

!!! info "Advanced: Resampling Theory and Numerical Stability"
    
    **Systematic Resampling**: Stratified uniforms u_k = u_0 + (k-1)/N. Lower variance than multinomial.
    
    **Resampling in u-space**: More numerically stable than x-space. Unit hypercube [0,1]^d resampling avoids boundary effects from non-linear prior_transform.
    
    **Constant Active Set**: Tempest resamples every iteration for stable memory usage and simplified parallelization.

---

### 3.4 Mutation: Evolving Particles with MCMC

**Goal**: Explore parameter space around each particle using Markov Chain Monte Carlo.

**Why it matters**: Resampling only copies particles. MCMC explores new nearby regions, discovering local structure.

**How it works**:

**If β = 0 (warmup)**:
- Skip MCMC mutation
- Draw fresh prior samples (diverse initial positions)

**If β > 0**:
1. **Per-cluster proposals**: Use proposal from assigned cluster
2. **t-pCN (default) or RWM**:
   - **t-pCN**: t-preconditioned Crank-Nicolson, gradient-free preconditioning for high dimensions
   - **RWM**: Random Walk Metropolis, simpler alternative
   
   **Note**: The original Persistent Sampling paper (Karamanis & Seljak, 2025) uses **Random Walk Metropolis (RWM)** with Robbins-Monro adaptation targeting 23.4% acceptance. Tempest extends this by offering **t-pCN** as the default, which provides better mixing in high dimensions (O(d) vs O(d²) scaling).
   
3. **Multiple steps**: Run for n_steps iterations (default: 10) or until convergence
4. **Boundary handling**:
   - **Periodic**: Wrap around (for angles, phases)
   - **Reflective**: Bounce off (for ratio parameters)
5. **Persistent start**: Start from current position (not from scratch)

**Output**: Evolved active particles representing distribution at new temperature

**Progress bar indicators**:
- `acc`: Acceptance rate (healthy: 0.2-0.8)
- `steps`: Number of MCMC steps taken
- `eff`: Efficiency (fraction of accepted steps, inverse of autocorrelation time)

**When it helps**: All phases after warmup. Critical for exploring posterior structure

**When to tune**: Can adjust n_steps, try RWM if t-pCN problems

!!! info "Advanced: t-pCN Derivation and Convergence"
    
    **t-pCN Proposal**: θ' = μ + sqrt(1-α^2)(θ-μ) + αν where ν ∼ t_ν(0,Σ)
    
    **Convergence Properties**: Geometrically ergodic under mild conditions. Mixing time scales O(d) vs O(d^2) for RWM. Invariant by construction.
    
    **Adaptive n_steps**: Plateau detection heuristic stops early if log-likelihood stabilizes (max: n_max_steps)
    
    **Boundary Conditions**: Periodic and reflective maintain detailed balance while respecting constraints

---

## 4. Advanced Features Deep Dive

### 4.1 Boosting: Smart Resource Allocation

**What it is**: Gradually increase particle count as sampling progresses from prior to posterior.

**Why it's useful**: Save computation early when fewer particles needed, increase accuracy near posterior.

**How it works**:

When n_boost is set:
- Start with n_effective particles (e.g., 512)
- As β increases, smoothly transition toward n_boost (e.g., 2048)
- Use sigmoid curve for smooth increase
- Adjust both n_active and n_effective targets

**When to use**: Expensive likelihoods, high dimensions, need high accuracy near posterior

**When NOT to use**: Cheap likelihoods, small problems

**Performance impact**: 30-50% computational savings vs static allocation

!!! info "Advanced: Boosting Curve Mathematics"
    
    **Sigmoid Function**: n_eff(β) = n_eff^0 + (n_boost - n_eff^0) / (1 + exp(-s(β-β_0)))
    
    **Savings**: Relative cost = ∫_0^1 n_eff(β)dβ / n_boost ≈ 0.5-0.7
    
    **Optimal Design**: Allocates particles where information gain per iteration is highest (near posterior)

---

### 4.2 Evidence Estimation

**What it is**: Built-in estimation of Bayesian model evidence (marginal likelihood, logZ).

**Why it matters**: logZ is crucial for Bayesian model comparison, automatically implementing Occam's razor.

**How it works**:

1. At each iteration, record average log-likelihood at current β
2. Integrate along β path: logZ = ∫_0^1 E_πβ[logL(θ)]dβ
3. Use trapezoidal quadrature for numerical integration
4. Estimate uncertainty via bootstrap

**Interpretation**: Higher logZ = better model. ΔlogZ > 5 is strong evidence.

**When reliable**: n_effective ≥ 512, smooth β progression, well-explored posterior

**Accuracy**: Typical error O(N^(-1/2)) where N is number of particles

**Comparison to nested sampling**: PS directly integrates likelihood expectation, can be more accurate for prior-likelihood mismatches

!!! info "Advanced: Thermodynamic Integration Theory"
    
    **Thermodynamic Identity**: For π_β(θ) = L(θ)^β p(θ)/Z(β), the derivative d(logZ)/dβ = E_πβ[logL(θ)]
    
    **Numerical Integration**: Trapezoidal quadrature: logZ ≈ Σ (β_(t+1) - β_t)/2 * (⟨logL⟩_(t+1) + ⟨logL⟩_t)
    
    **Uncertainty**: σ_(logZ)^2 ≈ Var_bootstrap(logZ*)

---

### 4.3 Clustering for Multimodal Distributions

**What it is**: Automatic detection and modeling of multiple posterior modes.

**Why it matters**: Single Gaussian proposals inefficient for multimodal distributions. Clustering enables mode-specific proposals.

**How it works**:

1. Collect weighted particles from all iterations
2. Fit hierarchical Gaussian mixture, starting with one component
3. Iteratively split components based on data
4. Stop when BIC indicates no improvement
5. Assign particles to clusters by highest posterior probability
6. Build per-cluster proposals for MCMC
7. Refit clustering periodically (default: every iteration)

**When it helps**: Multiple well-separated peaks, non-elliptical shapes, phase transitions

**When to disable**: Unimodal distributions, very high dimensions (>50D)

**Parameters**: split_threshold controls aggressiveness, n_max_clusters limits K

**Progress bar indicator**: K (number of active clusters)

!!! info "Advanced: Hierarchical Clustering Algorithm"
    
    **Agglomerative Splitting**: Test split along principal component, run EM, compute BIC(K+1) - BIC(K)
    
    **BIC**: logL(Θ_K|X) - (p_K/2)logN where p_K = K-1 + Kd + Kd(d+1)/2
    
    **Complexity**: O(KN_iter d²) per iteration. For K ≪ N, overhead minimal vs likelihood evaluations.
    
    **High-Dimensional**: In d>50, O(d²) covariance parameters require ≥10d samples per component. Use n_max_clusters to prevent overfitting.

---

## 5. Comparison with Other Methods

### Comparison Table

| Feature | Persistent Sampling | MCMC (emcee) | Nested Sampling | Variational Inference |
|---------|-------------------|--------------|-----------------|---------------------|
| Multimodality | Excellent (clustering) | Poor (single chain) | Good | Needs multiple runs |
| Evidence (logZ) | Built-in | Requires extra steps | Built-in | Approximate |
| Parallelization | Excellent | Limited | Moderate | Good |
| Setup Difficulty | Moderate | Easy | Easy | Moderate |
| High Dimensions | Good (t-pCN) | Poor (RWM) | Moderate | Excellent |
| Memory Usage | High | Low | High | Low |

### When to Use

**Best choice when**: Complex geometry, multimodality, need evidence, expensive likelihood, high dimensions with structure

**Consider alternatives**: Simple unimodal posteriors, severe memory constraints, >100D without structure, extremely cheap likelihoods

### Migration from Other Methods

**From emcee**:
- `n_walkers` → Set `n_effective` to similar value (e.g., 32 walkers → n_effective=512, ~16× multiplier)
- `sampler.run_mcmc()` → `sampler.run()`
- Samples: `sampler.posterior()` returns `(x, weights, logl)` instead of `sampler.flatchain`
- Enable clustering for multimodality: `clustering=True`

**From dynesty**:
- `nlive` → Set `n_effective` to similar value (direct mapping)
- `NestedSampler` → Use `Sampler` class
- Enable clustering for multimodal problems: `clustering=True`
- PS provides similar evidence estimation with persistence advantage

---

## 6. Understanding Diagnostics

### Progress Bar Breakdown

```
Iter: 50 [beta=0.85, ESS=512, logZ=-15.3, logL=-12.1, acc=0.42, steps=10, eff=0.85, K=3]
```

| Indicator | Healthy Range | What It Means |
|-----------|---------------|---------------|
| beta | 0→1 gradual | Current inverse temperature |
| ESS | Close to n_effective until β=1, then increases to n_total | Effective sample size |
| logZ | Stabilizing (not necessarily increasing) | Log evidence estimate |
| logL | Generally increasing with β | Mean log-likelihood |
| acc | >0.15 healthy | MCMC acceptance rate |
| steps | d-10×d okay | MCMC steps taken |
| eff | >0.1 healthy | MCMC efficiency |
| K | ≥1 healthy | Number of clusters |

**Healthy sampling**: Smooth β progression, ESS tracking target, logZ converging, acc>0.15, eff>0.1, K reflects modes

**Common patterns**:
- Beta stuck near 0: prior/likelihood scale mismatch
- Very low ESS: increase n_effective
- Low acceptance: enable clustering or try RWM
- K=1 for multimodal: decrease split_threshold

---

## Next Steps

- Try the [Quick Start Guide](quickstart.md) for hands-on practice
- Explore [Basic Usage](user_guide/basic_usage.md) for configuration
- Read about [Parallelization](user_guide/parallelization.md) for large problems
- See [Examples](examples/rosenbrock.md) for real applications

---

## References

### Primary Persistent Sampling Paper

**Persistent Sampling: Enhancing the Efficiency of Sequential Monte Carlo**
- Authors: Minas Karamanis, Uroš Seljak
- Journal: Statistics and Computing, 2025, 35(5):144
- DOI: [https://doi.org/10.1007/s11222-025-10682-y](https://doi.org/10.1007/s11222-025-10682-y)
- arXiv: [2407.20722](https://arxiv.org/abs/2407.20722)

This paper introduces the Persistent Sampling algorithm, which systematically retains and reuses particles from all prior iterations to construct a growing, weighted ensemble, achieving significant variance reduction without additional likelihood evaluations.

### Foundational SMC Literature

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269-342.

- Chopin, N. (2002). A sequential particle filter method for static models. *Biometrika*, 89(3), 539-552.

- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo samplers. *Journal of the Royal Statistical Society: Series B*, 68(3), 411-436.

- Dai, C., Heng, J., Jacob, P. E., & Whiteley, N. (2022). An invitation to sequential Monte Carlo samplers. *Journal of the American Statistical Association*, 117(539), 1587-1600.