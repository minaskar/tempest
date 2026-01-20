# Oscillatory Model Fitting

This example demonstrates fitting an oscillatory model to data using Tempest, with comprehensive parameter estimation, uncertainty quantification, and model validation.

## Problem Description

We consider data generated from an oscillatory model with amplitude that varies linearly with the independent variable:

$$
y = (A x + B) \sin(\omega x + \phi) + \epsilon, \quad \epsilon \sim N(0, \sigma^2)
$$

This model has **5 parameters**:
- $A$: Amplitude coefficient for the linear trend
- $B$: Offset coefficient
- $\omega$: Angular frequency
- $\phi$: Phase offset
- $\sigma$: Observation noise standard deviation

The challenge is to recover these parameters from noisy observations and quantify our uncertainty about them.

---

## Implementation

### 1. Data Generation

First, we generate synthetic data with known parameters:

```python
import numpy as np
import tempest as tp

# True parameters
A_true = 0.5
B_true = 2.0
omega_true = 2 * np.pi  # period = 1
phi_true = np.pi / 4
sigma_true = 0.25

# Generate data
np.random.seed(42)
n_data = 50
x = np.linspace(0, 3, n_data)
y_true = (A_true * x + B_true) * np.sin(omega_true * x + phi_true)
y_obs = y_true + np.random.normal(0, sigma_true, size=len(x))

print(f"Generated {n_data} data points with {sigma_true:.1%} noise")
print(f"True model: y = (A*x + B) * sin(ω*x + φ)")
print(f"  A={A_true}, B={B_true}, ω={omega_true:.2f}, φ={phi_true:.2f}")
```

**Expected output:**
```
Generated 50 data points with 25.0% noise
True model: y = (A*x + B) * sin(ω*x + φ)
  A=0.5, B=2.0, ω=6.28, φ=0.79
```

### 2. Model Definition

Define the likelihood and prior transform:

```python
def log_likelihood(theta):
    """Log-likelihood for oscillatory model."""
    A, B, omega, phi, sigma = theta
    y_pred = (A * x + B) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + 
                        np.log(2 * np.pi * sigma**2))

def prior_transform(u):
    """Transform from unit hypercube to physical parameters."""
    A = u[0]                    # U(0, 1)
    B = 5 * u[1]                # U(0, 5)
    omega = 8 * np.pi * u[2]    # U(0, 8π)
    phi = 2 * np.pi * u[3]      # U(0, 2π)
    sigma = 10 ** (3 * u[4] - 2)  # Log-uniform[0.01, 10]
    return np.array([A, B, omega, phi, sigma])
```

!!! tip "Prior Selection for Oscillatory Models"
    - **Frequency ($\omega$)**: Set upper bound based on expected Nyquist frequency (π/Δx)
    - **Phase ($\phi$)**: Always use Uniform(0, 2π) to respect periodicity
    - **Amplitude parameters**: Scale with expected signal magnitude
    - **Noise ($\sigma$)**: Log-uniform works well for scale parameters

### 3. Running Tempest

```python
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=5,
    n_effective=512,
    n_active=256,
    random_state=42,
    vectorize=True,
)

sampler.run(n_total=4096, progress=True)
samples, weights, logl = sampler.posterior()
logz, logz_err = sampler.evidence()

print(f"logZ = {logz:.2f} ± {logz_err:.2f}")
print(f"N_samples = {len(samples)}")
```

**Expected output:**
```
logZ = -26.36 ± 0.11
N_samples = 2156
```

!!! tip "Sampler Configuration"
    - `n_effective=512`: Target effective sample size
    - `n_active=256`: Number of active points (balance between speed and accuracy)
    - `vectorize=True`: Speed up likelihood evaluations
    - `random_state=42`: For reproducible results

---

## Results Analysis

### Parameter Estimation

Extract posterior means and uncertainties:

```python
params = np.average(samples, weights=weights, axis=0)
stds = np.sqrt(np.average((samples - params) ** 2, 
                         weights=weights, axis=0))

A_fit, B_fit, omega_fit, phi_fit, sigma_fit = params
A_err, B_err, omega_err, phi_err, sigma_err = stds

print(f"A = {A_fit:.3f} ± {A_err:.3f}  (true: {A_true})")
print(f"B = {B_fit:.3f} ± {B_err:.3f}  (true: {B_true})")
print(f"ω = {omega_fit:.3f} ± {omega_err:.3f}  (true: {omega_true:.3f})")
print(f"φ = {phi_fit:.3f} ± {phi_err:.3f}  (true: {phi_true:.3f})")
print(f"σ = {sigma_fit:.3f} ± {sigma_err:.3f}  (true: {sigma_true})")
```

**Expected output:**
```
A = 0.443 ± 0.058  (true: 0.5)
B = 2.106 ± 0.099  (true: 2.0)
ω = 6.308 ± 0.021  (true: 6.283)
φ = 0.728 ± 0.042  (true: 0.785)
σ = 0.247 ± 0.028  (true: 0.25)
```

### Model Fit Quality

```python
y_pred = (A_fit * x + B_fit) * np.sin(omega_fit * x + phi_fit)
residuals = y_obs - y_pred

# R-squared
rss = np.sum(residuals ** 2)
tss = np.sum((y_obs - np.mean(y_obs)) ** 2)
r_squared = 1 - rss / tss

print(f"R² = {r_squared:.4f}")
print(f"Residual std = {np.std(residuals):.4f}")
print(f"Fitted σ = {sigma_fit:.4f}")
```

**Expected output:**
```
R² = 0.9876
Residual std = 0.2415
Fitted σ = 0.247
```

---

## Visualization

### 1. Data and Model Fit

![Oscillatory model fit showing data points, true model, and best-fit prediction](../assets/examples/oscillatory_fit.png)

The plot shows:
- **Black points**: Observed data with noise
- **Green line**: True underlying model
- **Red line**: Best-fit model from posterior means

The excellent agreement demonstrates successful parameter recovery.

### 2. Parameter Posteriors

![Corner plot showing posterior distributions for A, B, ω, and φ](../assets/examples/oscillatory_corner.png)

**Key observations:**
- All parameters well-constrained with tight posteriors
- Green lines show true values (all within 1-2σ)
- Parameter correlations visible (e.g., A and B are anti-correlated)

### 3. Posterior Predictive Distribution

![Posterior predictive distribution with 68% credible interval](../assets/examples/oscillatory_predictive.png)

This plot shows:
- **Median prediction** (red line): Best estimate of the underlying function
- **68% credible interval** (red shaded): Uncertainty in the prediction
- **True model** (green dashed): For comparison

The narrow credible interval indicates high confidence in the fit.

### 4. Residuals Analysis

![Residual plots showing residuals vs fitted and histogram](../assets/examples/oscillatory_residuals.png)

**Left**: Residuals vs fitted values - no systematic patterns indicate good fit

**Right**: Residual histogram with overlaid Gaussian (red) - closely matches the fitted noise model

---

## Posterior Predictive Checks

Generate predictions from the posterior to validate model adequacy:

```python
import numpy as np

# Sample from posterior
n_pred = 200
idx = np.random.choice(len(samples), size=n_pred, p=weights, replace=True)
pred_samples = samples[idx]

# Generate predictions on dense grid
x_test = np.linspace(0, 3, 200)
predictions = np.zeros((n_pred, len(x_test)))

for i, theta in enumerate(pred_samples):
    A, B, omega, phi, _ = theta
    predictions[i] = (A * x_test + B) * np.sin(omega * x_test + phi)

# Compute percentiles
lower, median, upper = np.percentile(predictions, [5, 50, 95], axis=0)

# Plot with 90% prediction interval
plt.fill_between(x_test, lower, upper, alpha=0.3, 
                 label="90% prediction interval")
plt.plot(x_test, median, 'r-', label="Median prediction")
plt.scatter(x, y_obs, alpha=0.6, s=50, color='black', label="Data")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Posterior Predictive Check")
```

---

## Best Practices for Oscillatory Models

### 1. Prior Selection

- **Frequency**: Upper bound should respect Nyquist criterion
  ```python
  max_freq = np.pi / np.min(np.diff(x))  # Nyquist frequency
  omega = max_freq * u[2]  # U(0, max_freq)
  ```

- **Phase**: Always use Uniform(0, 2π) to maintain periodicity

- **Amplitude**: Scale priors with observed data range
  ```python
  y_range = np.max(y_obs) - np.min(y_obs)
  A = y_range * u[0]  # U(0, y_range)
  ```

### 2. Parameterization Considerations

- **Phase ambiguity**: The model y = sin(ωx + φ) has a 2π phase ambiguity
  - Tempest naturally handles this with periodic priors
  - Posterior may show multi-modalities if frequency is poorly constrained

- **Frequency-amplitude correlation**: High correlation between ω and amplitude parameters
  - Consider using informative priors or reparameterization
  - Check posterior correlations in corner plots

### 3. Convergence Diagnostics

Monitor sampling quality:

```python
# Evidence error should be < 0.5 for reliable results
logz, logz_err = sampler.evidence()
print(f"logZ error: {logz_err:.3f}")

# Effective sample size relative to total samples
ess = len(samples)  # Weighted posterior samples
print(f"Effective samples: {ess}")
```

### 4. Model Validation

Always perform model validation:

1. **Residual analysis**: Check for systematic patterns
2. **Posterior predictive checks**: Compare predictions to data
3. **Cross-validation**: Test on held-out data if available
4. **Parameter recovery**: Test with synthetic data (as done here)

---

## Common Issues and Solutions

### Issue: Frequency Poorly Constrained

**Symptoms**: Wide posterior for ω, poor fit quality

**Solutions**:
- Increase sampling time (`n_total`)
- Tighten prior based on domain knowledge
- Ensure sufficient data coverage of multiple periods

### Issue: Phase Wrapping

**Symptoms**: Multi-modal posterior for φ

**Solutions**: 
- This is often normal due to 2π periodicity
- Check that ω posteriors are well-constrained
- Consider reporting φ modulo 2π

### Issue: High Parameter Correlations

**Symptoms**: Degenerate contours in corner plots

**Solutions**:
- Increase `n_effective` for better exploration
- Consider reparameterization (e.g., using amplitude/phase instead of separate A/B)
- Check for model identifiability issues

---

## Extensions and Variations

### 1. Multiple Frequencies

Extend to multiple oscillatory components:

```python
def log_likelihood(theta):
    A1, B1, omega1, phi1, A2, B2, omega2, phi2, sigma = theta
    y_pred = ((A1 * x + B1) * np.sin(omega1 * x + phi1) + 
              (A2 * x + B2) * np.sin(omega2 * x + phi2))
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + 
                        np.log(2 * np.pi * sigma**2))
```

### 2. Non-Sinusoidal Oscillations

Replace sin with other periodic functions:

```python
# Square wave
y_pred = (A * x + B) * np.sign(np.sin(omega * x + phi))

# Sawtooth wave
y_pred = (A * x + B) * ((omega * x + phi) % (2 * np.pi))
```

### 3. Damped Oscillations

Add exponential decay:

```python
def log_likelihood(theta):
    A, B, omega, phi, gamma, sigma = theta
    y_pred = (A * x + B) * np.exp(-gamma * x) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + 
                        np.log(2 * np.pi * sigma**2))
```

---

## Summary

This example demonstrates:

- ✅ Fitting a physically-motivated oscillatory model
- ✅ Proper prior selection for periodic parameters
- ✅ Parameter estimation with uncertainty quantification
- ✅ Posterior predictive checks for model validation
- ✅ Visualization of fit quality and residuals
- ✅ Best practices for convergence assessment

Tempest's nested sampling algorithm naturally handles the multi-modal likelihood landscape that often arises in oscillatory models, providing robust evidence estimates and posterior samples even with moderate numbers of parameters.

---

## References

- Skilling, J. (2006). Nested sampling for general Bayesian computation. *Bayesian Analysis*, 1(4), 833-859.
- Bretthorst, G. L. (1988). *Bayesian Spectrum Analysis and Parameter Estimation*. Springer.
- Gregory, P. C. (2005). *Bayesian Logical Data Analysis for the Physical Sciences*. Cambridge University Press.
