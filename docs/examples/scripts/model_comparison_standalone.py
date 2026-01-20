#!/usr/bin/env python3
"""
Bayesian Model Comparison Example: Linear vs Oscillatory

This script demonstrates how to:
1. Fit two competing models to the same data
2. Compute Bayesian evidences using Tempest
3. Calculate and interpret Bayes factors
4. Extract parameter estimates and uncertainties

The data is synthetically generated from an oscillatory model, so we expect
the oscillatory model to be strongly favored despite having more parameters.
"""

import numpy as np
import os
import tempest as tp
import matplotlib.pyplot as plt
import corner

print("=" * 60)
print("Bayesian Model Comparison: Linear vs Oscillatory")
print("=" * 60)

# Configuration
np.random.seed(42)
n_data = 50
x = np.linspace(0, 3, n_data)

# Generate synthetic data from oscillatory model
A_true, B_true = 0.5, 2.0
omega_true = 2 * np.pi  # period = 1
phi_true = np.pi / 4
sigma_true = 0.25

y_true = (A_true * x + B_true) * np.sin(omega_true * x + phi_true)
y_obs = y_true + np.random.normal(0, sigma_true, size=len(x))

print(f"\nGenerated {n_data} data points with {sigma_true:.1%} noise")
print(f"True model: y = (A*x + B) * sin(ω*x + φ)")
print(f"  A={A_true}, B={B_true}, ω={omega_true:.2f}, φ={phi_true:.2f}")

# ============================================================================
# Visualize data generation
# ============================================================================
print("\n" + "=" * 60)
print("Creating data generation visualization...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y_true, "g-", linewidth=2, label="True model")
ax.scatter(x, y_obs, alpha=0.6, s=50, color="black", label="Synthetic data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Synthetic Data Generation")
ax.legend()
ax.grid(True, alpha=0.3)

output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "data_generation.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}\n")

# ============================================================================
# Model 1: Linear Regression (3 parameters)
# ============================================================================
print("\n" + "=" * 60)
print("Fitting Model 1: Linear (y = a*x + b + noise)")
print("=" * 60)


def log_likelihood_linear(theta):
    """Log-likelihood for linear model."""
    a, b, sigma = theta
    y_pred = a * x + b
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_linear(u):
    """Prior transform for linear model."""
    a = 10 * u[0] - 5  # U(-5, 5)
    b = 20 * u[1] - 10  # U(-10, 10)
    sigma = 10 ** (3 * u[2] - 2)  # Log-uniform [0.01, 10]
    return np.array([a, b, sigma])


sampler_linear = tp.Sampler(
    prior_transform=prior_transform_linear,
    log_likelihood=log_likelihood_linear,
    n_dim=3,
    n_effective=512,
    n_active=256,
    random_state=42,
)

sampler_linear.run(n_total=4096, progress=False)
samples_lin, weights_lin, logl_lin = sampler_linear.posterior()
logz_lin, logz_err_lin = sampler_linear.evidence()

print(f"\nLinear model results:")
if logz_err_lin is not None:
    print(f"  log Z = {logz_lin:.2f} ± {logz_err_lin:.2f}")
else:
    print(f"  log Z = {logz_lin:.2f}")
print(f"  N_samples = {len(samples_lin)}")

# ============================================================================
# Model 2: Oscillatory Model (5 parameters)
# ============================================================================
print("\n" + "=" * 60)
print("Fitting Model 2: Oscillatory (y = (A*x + B) * sin(ω*x + φ) + noise)")
print("=" * 60)


def log_likelihood_oscillatory(theta):
    """Log-likelihood for oscillatory model."""
    A, B, omega, phi, sigma = theta
    y_pred = (A * x + B) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_oscillatory(u):
    """Prior transform for oscillatory model."""
    A = u[0]  # U(0, 1)
    B = 5 * u[1]  # U(0, 5)
    omega = 8 * np.pi * u[2]  # U(0, 8π)
    phi = 2 * np.pi * u[3]  # U(0, 2π)
    sigma = 10 ** (3 * u[4] - 2)  # Log-uniform [0.01, 10]
    return np.array([A, B, omega, phi, sigma])


sampler_osc = tp.Sampler(
    prior_transform=prior_transform_oscillatory,
    log_likelihood=log_likelihood_oscillatory,
    n_dim=5,
    n_effective=512,
    n_active=256,
    random_state=42,
)

sampler_osc.run(n_total=4096, progress=False)
samples_osc, weights_osc, logl_osc = sampler_osc.posterior()
logz_osc, logz_err_osc = sampler_osc.evidence()

print(f"\nOscillatory model results:")
if logz_err_osc is not None:
    print(f"  log Z = {logz_osc:.2f} ± {logz_err_osc:.2f}")
else:
    print(f"  log Z = {logz_osc:.2f}")
print(f"  N_samples = {len(samples_osc)}")

# ============================================================================
# Bayes Factor Calculation
# ============================================================================
print("\n" + "=" * 60)
print("Model Comparison Results")
print("=" * 60)

bayes_factor = np.exp(logz_osc - logz_lin)
log10_bayes_factor = (logz_osc - logz_lin) / np.log(10)

print(f"\nBayes Factor (Oscillatory/Linear) = {bayes_factor:.2e}")
print(f"log₁₀(Bayes Factor) = {log10_bayes_factor:.2f}")

# Interpretation
if log10_bayes_factor < 0:
    strength = "Negative (supports Linear)"
elif log10_bayes_factor < 0.5:
    strength = "Weak"
elif log10_bayes_factor < 1.0:
    strength = "Substantial"
elif log10_bayes_factor < 2.0:
    strength = "Strong"
else:
    strength = "Decisive"

print(f"\nInterpretation: {strength} evidence for Oscillatory model")

# ============================================================================
# Parameter Estimation
# ============================================================================
print("\n" + "=" * 60)
print("Parameter Estimates")
print("=" * 60)

# Linear model parameters
params_lin = np.average(samples_lin, weights=weights_lin, axis=0)
stds_lin = np.sqrt(
    np.average((samples_lin - params_lin) ** 2, weights=weights_lin, axis=0)
)

a_fit, b_fit, sigma_fit_lin = params_lin
a_err, b_err, sigma_err_lin = stds_lin

print(f"\nLinear Model (3 parameters):")
print(f"  a (slope)    = {a_fit:7.3f} ± {a_err:.3f}")
print(f"  b (intercept)= {b_fit:7.3f} ± {b_err:.3f}")
print(f"  σ (noise)    = {sigma_fit_lin:7.3f} ± {sigma_err_lin:.3f}")

# Oscillatory model parameters
params_osc = np.average(samples_osc, weights=weights_osc, axis=0)
stds_osc = np.sqrt(
    np.average((samples_osc - params_osc) ** 2, weights=weights_osc, axis=0)
)

A_fit, B_fit, omega_fit, phi_fit, sigma_fit_osc = params_osc
A_err, B_err, omega_err, phi_err, sigma_err_osc = stds_osc

print(f"\nOscillatory Model (5 parameters):")
print(f"  A (amplitude coeff) = {A_fit:7.3f} ± {A_err:.3f}  (true: {A_true})")
print(f"  B (offset)          = {B_fit:7.3f} ± {B_err:.3f}  (true: {B_true})")
print(
    f"  ω (frequency)       = {omega_fit:7.3f} ± {omega_err:.3f}  (true: {omega_true:.3f})"
)
print(f"  φ (phase)           = {phi_fit:7.3f} ± {phi_err:.3f}  (true: {phi_true:.3f})")
print(
    f"  σ (noise)           = {sigma_fit_osc:7.3f} ± {sigma_err_osc:.3f}  (true: {sigma_true})"
)

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

# Generate predictions
y_pred_lin = a_fit * x + b_fit
y_pred_osc = (A_fit * x + B_fit) * np.sin(omega_fit * x + phi_fit)

# Create 4-panel plot
fig = plt.figure(figsize=(14, 10))

# Panel 1: Data and model predictions
ax1 = plt.subplot(2, 2, (1, 2))
ax1.scatter(x, y_obs, alpha=0.6, s=50, label="Data", color="black", zorder=3)
ax1.plot(x, y_pred_osc, "r-", linewidth=2, label=f"Oscillatory (logZ={logz_osc:.1f})")
ax1.plot(x, y_pred_lin, "b--", linewidth=2, label=f"Linear (logZ={logz_lin:.1f})")
ax1.plot(x, y_true, "g-", alpha=0.7, linewidth=1, label="True model")
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title("Model Comparison: Data and Fits", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Panel 2: Oscillatory model posteriors (corner plot)
fig_corner = corner.corner(
    samples_osc[:, :4],  # 4 physical parameters
    labels=["A", "B", r"$\omega$", r"$\phi$"],
    truths=[A_true, B_true, omega_true, phi_true],
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
)
plt.close(fig_corner)  # Don't display separately

# Load corner plot as image
from matplotlib import image
import os

# Save and load corner plot
fig_corner.savefig("/tmp/corner_temp.png", dpi=150, bbox_inches="tight")
corner_img = image.imread("/tmp/corner_temp.png")

ax2 = plt.subplot(2, 2, 3)
ax2.imshow(corner_img)
ax2.axis("off")
ax2.set_title("Oscillatory Model Posteriors", fontsize=12, fontweight="bold")

# Panel 3: Evidence comparison
ax3 = plt.subplot(2, 2, 4)
ax3.axis("off")
ax3.text(
    0.5,
    0.5,
    f"Evidence Comparison:\n\n"
    f"Linear: logZ = {logz_lin:.2f}\n"
    f"Oscillatory: logZ = {logz_osc:.2f}\n\n"
    f"Bayes Factor = {bayes_factor:.1e}\n"
    f"log₁₀(BF) = {log10_bayes_factor:.1f}\n\n"
    f"Interpretation: {strength}\n"
    f"evidence for Oscillatory model",
    transform=ax3.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
)

plt.tight_layout()
plt.savefig("model_comparison_results.png", dpi=150, bbox_inches="tight")
print("Saved: model_comparison_results.png\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 60)
print("Summary")
print("=" * 60)
print("\nThis example demonstrates:")
print("• Bayesian evidence computation with Tempest")
print("• Bayes factor calculation for model comparison")
print("• Parameter estimation with uncertainty quantification")
print(f"• {strength.lower()} preference for the more complex oscillatory model")
print("\nThe oscillatory model wins despite having more parameters")
print("because it explains the data much better, and")
print("Bayesian evidence naturally penalizes unnecessary complexity.")
print("=" * 60)
