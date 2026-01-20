#!/usr/bin/env python3
"""
Generate plots for Gaussian Mixture example documentation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import tempest as tp
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Configuration
output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)

# Define functions (from docs/examples/gaussian_mixture.md)
n_dim = 2
means = [
    np.array([-2.0, 0.0]),
    np.array([2.0, 0.0]),
]
covs = [
    np.array([[0.5, 0.3], [0.3, 0.5]]),
    np.array([[0.5, -0.3], [-0.3, 0.5]]),
]
weights = [0.5, 0.5]


def prior_transform(u):
    return 20 * u - 10


def log_likelihood(x):
    """Log-likelihood of Gaussian mixture."""
    x = np.atleast_2d(x)
    log_probs = []
    for w, mu, cov in zip(weights, means, covs):
        log_probs.append(np.log(w) + multivariate_normal.logpdf(x, mean=mu, cov=cov))
    log_probs = np.array(log_probs)
    return np.logaddexp.reduce(log_probs, axis=0).squeeze()


# Classify samples function
def classify_samples(samples, means):
    """Classify samples by nearest mean."""
    distances = np.array([np.linalg.norm(samples - mean, axis=1) for mean in means])
    return np.argmin(distances, axis=0)


print("Running Gaussian Mixture sampler...")
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_effective=1024,
    n_active=512,
    clustering=True,
    vectorize=True,
    random_state=42,
)

sampler.run(n_total=4096, progress=False)
samples, weights_post, _ = sampler.posterior()

print(f"Generated {len(samples)} samples")

# Generate scatter plot
print("Generating scatter plot...")
plt.figure(figsize=(8, 8))

# Plot samples
idx = np.random.choice(len(samples), size=2000, p=weights_post, replace=True)
samples_plot = samples[idx]

# Classify by nearest mean
labels = classify_samples(samples_plot, means)

# Plot each mode with different color
colors = ["C0", "C1"]
for i, (mean, cov) in enumerate(zip(means, covs)):
    mask = labels == i
    plt.scatter(
        samples_plot[mask, 0], samples_plot[mask, 1], alpha=0.6, s=10, label=f"Mode {i}"
    )

# Plot true means
for i, mean in enumerate(means):
    plt.plot(
        mean[0],
        mean[1],
        "k*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=1,
    )

plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.legend()
plt.title("Gaussian Mixture: Recovered Posterior")
plt.axis("equal")
plt.grid(True, alpha=0.3)

output_path = os.path.join(output_dir, "gaussian_mixture.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
