#!/usr/bin/env python3
"""
Generate corner plots for Bayesian model comparison example.
Creates separate corner plots for each model's parameter posteriors.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import tempest as tp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

# Configuration
output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
n_data = 50
x = np.linspace(0, 3, n_data)

A_true, B_true = 0.5, 2.0
omega_true = 2 * np.pi
phi_true = np.pi / 4
sigma_true = 0.25

y_true = (A_true * x + B_true) * np.sin(omega_true * x + phi_true)
y_obs = y_true + np.random.normal(0, sigma_true, size=len(x))

# ============================================================================
# Linear Model Corner Plot
# ============================================================================
print("Generating linear model corner plot...")


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

# Generate resampled unweighted samples for corner plot
idx = np.random.choice(len(samples_lin), size=3000, p=weights_lin, replace=True)
samples_lin_unweighted = samples_lin[idx]

# Create corner plot
fig = corner.corner(
    samples_lin_unweighted,
    labels=["a", "b", r"$\sigma$"],
    truths=[None, None, None],
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
    plot_datapoints=False,
    plot_density=False,
    plot_contours=True,
    fill_contours=True,
    figsize=(8, 8),
)

output_path = os.path.join(output_dir, "linear_corner.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")

# ============================================================================
# Oscillatory Model Corner Plot
# ============================================================================
print("Generating oscillatory model corner plot...")


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

# Generate resampled unweighted samples for corner plot
idx = np.random.choice(len(samples_osc), size=3000, p=weights_osc, replace=True)
samples_osc_unweighted = samples_osc[idx]

# Create corner plot (excluding sigma for clarity, plotting only physical parameters)
fig = corner.corner(
    samples_osc_unweighted[:, :4],  # A, B, omega, phi only
    labels=["A", "B", r"$\omega$", r"$\phi$"],
    truths=[A_true, B_true, omega_true, phi_true],
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
    plot_datapoints=False,
    plot_density=False,
    plot_contours=True,
    fill_contours=True,
    figsize=(8, 8),
)

output_path = os.path.join(output_dir, "oscillatory_corner.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
print("\nAll corner plots generated successfully!")
