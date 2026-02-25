#!/usr/bin/env python3
"""
Generate plots for Rosenbrock example documentation.
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
import corner

# Configuration
n_dim = 10
output_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
os.makedirs(output_dir, exist_ok=True)


# Define functions (from docs/examples/rosenbrock.md)
def prior_transform(u):
    return 20 * u - 10


def log_likelihood(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    result = -np.sum(
        100.0 * (x[:, 1::2] - x[:, ::2] ** 2) ** 2 + (1.0 - x[:, ::2]) ** 2, axis=1
    )
    return result.squeeze() if result.size == 1 else result


print("Running Rosenbrock sampler...")
# Run sampler with smaller n_total for faster execution
sampler = tp.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
    n_particles=1024,
    vectorize=True,
    random_state=42,
)

sampler.run(n_total=4096, progress=False)  # Reduced from 8192 for speed
samples, weights, logl = sampler.posterior()

print(f"Generated {len(samples)} samples")

# Generate corner plot
print("Generating corner plot...")
idx = np.random.choice(
    len(samples), size=3000, p=weights, replace=True
)  # Reduced from 5000
samples_resampled = samples[idx]

fig = corner.corner(
    samples_resampled[:, :4],
    labels=[f"$x_{{{i}}}$" for i in range(4)],
    truths=[1.0, 1.0, 1.0, 1.0],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
)

output_path = os.path.join(output_dir, "rosenbrock_corner.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
