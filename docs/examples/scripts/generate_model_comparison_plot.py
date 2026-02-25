#!/usr/bin/env python3
"""
Generate plots for Model Comparison example (Linear vs Oscillatory).
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
output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)

print("Generating synthetic data...")

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


# Model 1: Linear Regression (3 parameters: a, b, sigma)
def log_likelihood_linear(theta):
    a, b, sigma = theta
    y_pred = a * x + b
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_linear(u):
    # Broad priors
    a = 10 * u[0] - 5  # U(-5, 5)
    b = 20 * u[1] - 10  # U(-10, 10)
    sigma = 10 ** (3 * u[2] - 2)  # Log-uniform from 0.01 to 10
    return np.array([a, b, sigma])


# Model 2: Oscillatory Model (5 parameters: A, B, omega, phi, sigma)
def log_likelihood_oscillatory(theta):
    A, B, omega, phi, sigma = theta
    y_pred = (A * x + B) * np.sin(omega * x + phi)
    return -0.5 * np.sum(((y_obs - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


def prior_transform_oscillatory(u):
    # Relatively broad priors for this simple problem
    A = u[0]  # U(0, 1)
    B = 5 * u[1]  # U(0, 5)
    omega = 8 * np.pi * u[2]  # U(0, 8π), wide enough for the problem
    phi = 2 * np.pi * u[3]  # U(0, 2π)
    sigma = 10 ** (3 * u[4] - 2)  # Log-uniform from 0.01 to 10
    return np.array([A, B, omega, phi, sigma])


print("\nRunning Model 1 (Linear) sampler...")
sampler_linear = tp.Sampler(
    prior_transform=prior_transform_linear,
    log_likelihood=log_likelihood_linear,
    n_dim=3,
    n_particles=512,
    random_state=42,
)

sampler_linear.run(n_total=4096, progress=False)
samples_lin, weights_lin, _ = sampler_linear.posterior()
logz_lin, _ = sampler_linear.evidence()

print(f"Linear model: logZ = {logz_lin:.2f}")

print("\nRunning Model 2 (Oscillatory) sampler...")
sampler_osc = tp.Sampler(
    prior_transform=prior_transform_oscillatory,
    log_likelihood=log_likelihood_oscillatory,
    n_dim=5,
    n_particles=512,
    random_state=42,
)

sampler_osc.run(n_total=4096, progress=False)
samples_osc, weights_osc, _ = sampler_osc.posterior()
logz_osc, _ = sampler_osc.evidence()

print(f"Oscillatory model: logZ = {logz_osc:.2f}")

# Compute Bayes factor
bayes_factor = np.exp(logz_osc - logz_lin)
log10_bayes_factor = (logz_osc - logz_lin) / np.log(10)

print(f"\nBayes Factor (Oscillatory/Linear) = {bayes_factor:.2e}")
print(f"log₁₀(BF) = {log10_bayes_factor:.2f}")

# Get best-fit parameters
params_lin = np.average(samples_lin, weights=weights_lin, axis=0)
a_fit, b_fit, sigma_fit_lin = params_lin

params_osc = np.average(samples_osc, weights=weights_osc, axis=0)
A_fit, B_fit, omega_fit, phi_fit, sigma_fit_osc = params_osc

print(f"\nBest-fit linear: a={a_fit:.3f}, b={b_fit:.3f}")
print(
    f"Best-fit oscillatory: A={A_fit:.3f}, B={B_fit:.3f}, ω={omega_fit:.3f}, φ={phi_fit:.3f}"
)

# Generate predictions
y_pred_osc = (A_fit * x + B_fit) * np.sin(omega_fit * x + phi_fit)
y_pred_lin = a_fit * x + b_fit

print("\nGenerating visualization plot...")

fig = plt.figure(figsize=(14, 10))

# Top panel: Data and model fits
ax1 = plt.subplot(2, 2, (1, 2))
ax1.scatter(x, y_obs, alpha=0.6, s=50, label="Data", color="black", zorder=3)
ax1.plot(
    x,
    y_pred_osc,
    "r-",
    label=f"Oscillatory (logZ={logz_osc:.1f})",
    linewidth=2,
    zorder=2,
)
ax1.plot(
    x, y_pred_lin, "b--", label=f"Linear (logZ={logz_lin:.1f})", linewidth=2, zorder=2
)
ax1.plot(x, y_true, "g-", label="True model", alpha=0.7, linewidth=1, zorder=1)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title("Model Comparison: Data and Fits", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Bottom-left: Corner plot for oscillatory model
# corner creates its own figure, don't pass fig parameter
fig_corner = corner.corner(
    samples_osc[:, :4],  # Exclude sigma
    labels=["A", "B", r"$\omega$", r"$\phi$"],
    truths=[A_true, B_true, omega_true, phi_true],
    show_titles=True,
    title_fmt=".2f",
    max_n_ticks=3,
)
# Now we need to embed this into the main figure
# One approach: save corner plot, reopen and embed
fig_corner.savefig("/tmp/corner_temp.png", dpi=150, bbox_inches="tight")
plt.close(fig_corner)

# Load the corner plot as an image and display in ax2
from matplotlib import image

corner_img = image.imread("/tmp/corner_temp.png")
ax2 = plt.subplot(2, 2, 3)
ax2.imshow(corner_img)
ax2.axis("off")
ax2.set_title("Oscillatory Model Posteriors", fontsize=12, fontweight="bold")
# Adjust position
fig_corner.axes[0].figure = fig

# Bottom-right: Evidence convergence plot (mock for now)
ax3 = plt.subplot(2, 2, 4)
ax3.axis("off")  # Will add proper convergence plot later
ax3.text(
    0.5,
    0.5,
    f"Evidence Comparison:\n\n"
    f"Linear: logZ = {logz_lin:.2f}\n"
    f"Oscillatory: logZ = {logz_osc:.2f}\n\n"
    f"Bayes Factor (Osc/Linear) = {bayes_factor:.1e}\n"
    f"log₁₀(BF) = {log10_bayes_factor:.1f}\n\n"
    f"Interpretation: Very strong\\nevidence for oscillatory model",
    transform=ax3.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
)

plt.tight_layout()

# Save plot
output_path = os.path.join(output_dir, "model_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved plot: {output_path}")
print("\nModel comparison complete!")
