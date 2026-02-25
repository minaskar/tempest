#!/usr/bin/env python3
"""
Generate plots for Oscillatory Model Fitting example.

This script demonstrates fitting an oscillatory model to synthetic data using
tempest, with comprehensive visualization of results.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import tempest as tp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend
import corner

# Configuration
# Determine project root and correct output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
output_dir = os.path.join(project_root, "docs", "examples", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)

print("Generating synthetic oscillatory data...")

# True parameters for oscillatory model
A_true = 0.5  # Amplitude coefficient for linear trend
B_true = 2.0  # Offset coefficient
omega_true = 2 * np.pi  # Frequency (period = 1)
phi_true = np.pi / 4  # Phase offset
sigma_true = 0.25  # 25% noise level

# Generate data
np.random.seed(42)
n_data = 50
x = np.linspace(0, 3, n_data)
y_true = (A_true * x + B_true) * np.sin(omega_true * x + phi_true)
y_obs = y_true + np.random.normal(0, sigma_true, size=len(x))

print(f"Generated {n_data} data points with {sigma_true:.1%} noise")
print(f"True parameters: A={A_true}, B={B_true}, ω={omega_true:.2f}, φ={phi_true:.2f}")


# Model definition: Oscillatory Model (5 parameters: A, B, omega, phi, sigma)
def log_likelihood_oscillatory(theta):
    """Log-likelihood for oscillatory model."""
    A, B, omega, phi, sigma = theta
    y_pred = (A * x + B) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_oscillatory(u):
    """Prior transform for oscillatory model."""
    A = u[0]  # U(0, 1)
    B = 5 * u[1]  # U(0, 5)
    omega = 8 * np.pi * u[2]  # U(0, 8π), wide enough for the problem
    phi = 2 * np.pi * u[3]  # U(0, 2π)
    sigma = 10 ** (3 * u[4] - 2)  # Log-uniform from 0.01 to 10
    return np.array([A, B, omega, phi, sigma])


print("\nRunning Tempest sampler for oscillatory model...")
sampler = tp.Sampler(
    prior_transform=prior_transform_oscillatory,
    log_likelihood=log_likelihood_oscillatory,
    n_dim=5,
    n_particles=512,
    random_state=42,
)

sampler.run(n_total=4096, progress=False)
samples, weights, logl = sampler.posterior()
logz, logz_err = sampler.evidence()

print(f"Sampling completed: logZ = {logz:.2f}")
if logz_err is not None:
    print(f"  logZ error = {logz_err:.2f}")
print(f"Number of posterior samples: {len(samples)}")

# Get best-fit parameters (weighted posterior mean)
params = np.average(samples, weights=weights, axis=0)
stds = np.sqrt(np.average((samples - params) ** 2, weights=weights, axis=0))

A_fit, B_fit, omega_fit, phi_fit, sigma_fit = params
A_err, B_err, omega_err, phi_err, sigma_err = stds

print(f"\nParameter estimates:")
print(f"  A = {A_fit:.3f} ± {A_err:.3f} (true: {A_true})")
print(f"  B = {B_fit:.3f} ± {B_err:.3f} (true: {B_true})")
print(f"  ω = {omega_fit:.3f} ± {omega_err:.3f} (true: {omega_true:.3f})")
print(f"  φ = {phi_fit:.3f} ± {phi_err:.3f} (true: {phi_true:.3f})")
print(f"  σ = {sigma_fit:.3f} ± {sigma_err:.3f} (true: {sigma_true})")

# Generate predictions
y_pred = (A_fit * x + B_fit) * np.sin(omega_fit * x + phi_fit)

print("\nGenerating visualizations...")

# Figure 1: Data, true model, and best-fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y_obs, alpha=0.6, s=50, color="black", label="Observed data", zorder=3)
ax.plot(x, y_true, "g-", linewidth=2, label="True model", alpha=0.7, zorder=1)
ax.plot(x, y_pred, "r-", linewidth=2, label="Best-fit model", zorder=2)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Oscillatory Model Fit", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

output_path = os.path.join(output_dir, "oscillatory_fit.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")

# Figure 2: Corner plot of posterior distributions
fig_corner = corner.corner(
    samples[:, :4],  # Exclude sigma for cleaner visualization
    labels=["A", "B", r"$\omega$", r"$\phi$"],
    truths=[A_true, B_true, omega_true, phi_true],
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 12},
)

output_path = os.path.join(output_dir, "oscillatory_corner.png")
fig_corner.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig_corner)
print(f"Saved: {output_path}")

# Figure 3: Posterior predictive distribution with uncertainty bands
print("\nGenerating posterior predictive samples...")
n_predictive = 200
idx = np.random.choice(len(samples), size=n_predictive, p=weights, replace=True)
predictive_samples = samples[idx]

# Generate predictions for each sample
x_dense = np.linspace(0, 3, 200)
predictions = np.zeros((n_predictive, len(x_dense)))
for i, theta in enumerate(predictive_samples):
    A, B, omega, phi, _ = theta
    predictions[i] = (A * x_dense + B) * np.sin(omega * x_dense + phi)

# Compute percentiles for credible intervals
q16, q50, q84 = np.percentile(predictions, [16, 50, 84], axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
# Plot 68% credible interval
ax.fill_between(
    x_dense, q16, q84, alpha=0.3, color="red", label="68% credible interval"
)
# Plot median prediction
ax.plot(x_dense, q50, "r-", linewidth=2, label="Median prediction")
# Plot observed data
ax.scatter(x, y_obs, alpha=0.6, s=50, color="black", label="Observed data", zorder=3)
# Plot true model
ax.plot(
    x_dense,
    (A_true * x_dense + B_true) * np.sin(omega_true * x_dense + phi_true),
    "g--",
    linewidth=1,
    alpha=0.7,
    label="True model",
)

ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Posterior Predictive Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

output_path = os.path.join(output_dir, "oscillatory_predictive.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")

# Figure 4: Residuals analysis
residuals = y_obs - y_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs fitted
ax1.scatter(y_pred, residuals, alpha=0.6, s=50, color="black")
ax1.axhline(y=0, color="r", linestyle="--", alpha=0.7)
ax1.set_xlabel("Fitted values", fontsize=12)
ax1.set_ylabel("Residuals", fontsize=12)
ax1.set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Histogram of residuals
ax2.hist(residuals, bins=15, alpha=0.7, color="gray", edgecolor="black", density=True)
ax2.axvline(x=0, color="r", linestyle="--", alpha=0.7)
# Overlay expected normal distribution
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
y_norm = np.exp(-0.5 * (x_norm / sigma_fit) ** 2) / (sigma_fit * np.sqrt(2 * np.pi))
ax2.plot(x_norm, y_norm, "r-", linewidth=2, label=f"N(0, {sigma_fit:.3f})")
ax2.set_xlabel("Residuals", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, "oscillatory_residuals.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")

print("\nAll visualizations generated successfully!")
