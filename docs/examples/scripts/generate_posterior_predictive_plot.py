#!/usr/bin/env python3
"""
Generate posterior predictive visualization for Bayesian model comparison.

This script creates a square plot showing:
- Observed data points
- True underlying model
- Posterior predictive distributions (median and 95% CI) for both models
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

# Configuration
output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
n_data = 50
x = np.linspace(0, 3, n_data)

# True model parameters
A_true, B_true = 0.5, 2.0
omega_true = 2 * np.pi  # period = 1
phi_true = np.pi / 4
sigma_true = 0.25

# Generate synthetic data
y_true = (A_true * x + B_true) * np.sin(omega_true * x + phi_true)
y_obs = y_true + np.random.normal(0, sigma_true, size=len(x))

print("Generating posterior predictive distributions...")


# ============================================================================
# Model 1: Linear Regression
# ============================================================================
def log_likelihood_linear(theta):
    a, b, sigma = theta
    y_pred = a * x + b
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_linear(u):
    a = 10 * u[0] - 5
    b = 20 * u[1] - 10
    sigma = 10 ** (3 * u[2] - 2)
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
samples_lin, weights_lin, _ = sampler_linear.posterior()


# ============================================================================
# Model 2: Oscillatory Model
# ============================================================================
def log_likelihood_oscillatory(theta):
    A, B, omega, phi, sigma = theta
    y_pred = (A * x + B) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_oscillatory(u):
    A = u[0]
    B = 5 * u[1]
    omega = 8 * np.pi * u[2]
    phi = 2 * np.pi * u[3]
    sigma = 10 ** (3 * u[4] - 2)
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
samples_osc, weights_osc, _ = sampler_osc.posterior()

# ============================================================================
# Generate Posterior Predictive Samples
# ============================================================================
n_pred_samples = 200

# Linear model predictions
print("Generating linear model predictive samples...")
lin_preds = np.zeros((n_pred_samples, len(x)))
for i in range(n_pred_samples):
    # Sample from posterior (weighted resampling)
    idx = np.random.choice(len(samples_lin), p=weights_lin)
    a, b, _ = samples_lin[idx]
    lin_preds[i] = a * x + b

# Oscillatory model predictions
print("Generating oscillatory model predictive samples...")
osc_preds = np.zeros((n_pred_samples, len(x)))
for i in range(n_pred_samples):
    # Sample from posterior (weighted resampling)
    idx = np.random.choice(len(samples_osc), p=weights_osc)
    A, B, omega, phi, _ = samples_osc[idx]
    osc_preds[i] = (A * x + B) * np.sin(omega * x + phi)

# Compute quantiles (16th, 50th, 84th percentile for 68% CI, or use 2.5th, 50th, 97.5th for 95%)
lin_q025, lin_q50, lin_q975 = np.percentile(lin_preds, [2.5, 50, 97.5], axis=0)
osc_q025, osc_q50, osc_q975 = np.percentile(osc_preds, [2.5, 50, 97.5], axis=0)

# ============================================================================
# Create Visualization
# ============================================================================
print("Creating posterior predictive plot...")

fig, ax = plt.subplots(figsize=(8, 8))  # Square plot

# Plot data and true model
ax.scatter(x, y_obs, color="black", alpha=0.7, s=60, label="Observed data", zorder=5)
ax.plot(x, y_true, "g-", linewidth=3, label="True model", zorder=4)

# Linear model predictions (95% credible interval)
ax.fill_between(x, lin_q025, lin_q975, alpha=0.3, color="blue", label="Linear 95% CI")
ax.plot(x, lin_q50, "b--", linewidth=2, label="Linear median")

# Oscillatory model predictions (95% credible interval)
ax.fill_between(
    x, osc_q025, osc_q975, alpha=0.3, color="red", label="Oscillatory 95% CI"
)
ax.plot(x, osc_q50, "r-", linewidth=2, label="Oscillatory median")

ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Posterior Predictive Distributions", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

# Ensure square aspect ratio
ax.set_aspect("auto")

plt.tight_layout()

# Save plot
output_path = os.path.join(output_dir, "posterior_predictive.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
print(f"Linear model: {n_pred_samples} predictive samples, 95% CI")
print(f"Oscillatory model: {n_pred_samples} predictive samples, 95% CI")
