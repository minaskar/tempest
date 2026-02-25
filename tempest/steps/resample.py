"""Resampling step for Persistent Sampling algorithm."""

from typing import Optional

import numpy as np
from matplotlib.axes import Subplot

from tempest.cluster import HierarchicalGaussianMixture
from tempest.state_manager import StateManager
from tempest.tools import systematic_resample


class Resampler:
    """
    Resampling step for particle selection using importance weights.

    Resamples particles from the historical pool to create the active set
    for the next MCMC mutation step.

    Parameters
    ----------
    state : StateManager
        State manager for reading/writing particle state.
    n_particles : int
        Number of particles to resample (fixed during run).
    resample : str
        Resampling scheme: "syst" (systematic) or "mult" (multinomial).
    clusterer : HierarchicalGaussianMixture, optional
        Clustering model for assigning cluster labels (shared with Trainer).
    clustering : bool
        Whether clustering is enabled (default: True).
    have_blobs : bool
        Whether likelihood returns auxiliary data (default: False).
    """

    def __init__(
        self,
        state: StateManager,
        n_particles: int,
        resample: str = "syst",
        clusterer: Optional[HierarchicalGaussianMixture] = None,
        clustering: bool = True,
        have_blobs: bool = False,
    ):
        """Initialize Resampler."""
        self.state = state
        self.n_particles = n_particles
        self.resample = resample
        self.clusterer = clusterer
        self.clustering = clustering
        self.have_blobs = have_blobs

    def run(self, weights: np.ndarray) -> None:
        """
        Resample particles from historical pool.

        Parameters
        ----------
        weights : np.ndarray
            Normalized importance weights for all historical particles.

        Updates current state:
            - u: unit hypercube coordinates (shape: [n_particles, n_dim])
            - x: physical coordinates (shape: [n_particles, n_dim])
            - logl: log-likelihoods (shape: [n_particles])
            - blobs: auxiliary data (shape: [n_particles, ...] if enabled)
            - assignments: cluster labels (shape: [n_particles])
        """
        # Skip resampling during warmup (beta=0) - will draw fresh prior samples
        beta = self.state.get_current("beta")
        if beta == 0.0:
            self.state.set_current("assignments", np.zeros(self.n_particles, dtype=int))
            return

        u = self.state.get_history("u", flat=True)
        x = self.state.get_history("x", flat=True)
        logl = self.state.get_history("logl", flat=True)
        blobs = self.state.get_history("blobs", flat=True) if self.have_blobs else None

        if self.resample == "mult":
            idx_resampled = np.random.choice(
                np.arange(len(weights)), size=self.n_particles, replace=True, p=weights
            )
        elif self.resample == "syst":
            idx_resampled = systematic_resample(self.n_particles, weights=weights)

        u_resampled = u[idx_resampled]
        self.state.update_current(
            {
                "u": u_resampled,
                "x": x[idx_resampled],
                "logl": logl[idx_resampled],
                "assignments": self.clusterer.predict(u_resampled)
                if self.clustering
                else np.zeros(self.n_particles, dtype=int),
            }
        )

        if self.have_blobs:
            self.state.set_current("blobs", blobs[idx_resampled])
