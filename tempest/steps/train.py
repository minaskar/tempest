"""Training step for Persistent Sampling algorithm."""

import numpy as np
from typing import Optional

from tempest.state_manager import StateManager
from tempest.tools import ProgressBar, trim_weights
from tempest.cluster import HierarchicalGaussianMixture
from tempest.modes import ModeStatistics


class Trainer:
    """
    Training step for fitting clustering model and building mode statistics.

    Fits a hierarchical Gaussian mixture model to weighted historical particles
    and constructs ModeStatistics objects for MCMC proposal distributions.

    Parameters
    ----------
    state : StateManager
        State manager for reading particle history.
    pbar : ProgressBar, optional
        Progress bar to update with stats.
    clusterer : HierarchicalGaussianMixture
        Clustering model to fit (shared with Resampler).
    cluster_every : int
        Fit clustering every N iterations (default: 1).
    clustering : bool
        Whether clustering is enabled (default: True).
    TRIM_ESS : float
        Target ESS for weight trimming (default: 512).
    TRIM_BINS : int
        Number of bins for histogram-based trimming (default: 10).
    DOF_FALLBACK : float
        Fallback degrees of freedom for Student-t fitting (default: 1.0).

    Attributes
    ----------
    clusterer : HierarchicalGaussianMixture
        Clustering model instance (mutable - gets fitted during run).
    """

    def __init__(
        self,
        state: StateManager,
        pbar: Optional[ProgressBar] = None,
        clusterer: Optional[HierarchicalGaussianMixture] = None,
        cluster_every: int = 1,
        clustering: bool = True,
        TRIM_ESS: float = 512,
        TRIM_BINS: int = 10,
        DOF_FALLBACK: float = 1.0,
    ):
        """Initialize Trainer."""
        self.state = state
        self.pbar = pbar
        self.clusterer = clusterer
        self.cluster_every = cluster_every
        self.clustering = clustering
        self.TRIM_ESS = TRIM_ESS
        self.TRIM_BINS = TRIM_BINS
        self.DOF_FALLBACK = DOF_FALLBACK

    def run(self, weights: np.ndarray) -> ModeStatistics:
        """
        Fit clustering model and build mode statistics.

        Parameters
        ----------
        weights : np.ndarray
            Importance weights for all historical particles.

        Returns
        -------
        mode_stats : ModeStatistics
            Fitted mode statistics for MCMC proposal kernel.
        """
        # Skip training at beta=0 (mutation draws fresh prior samples)
        beta_val = self.state.get_current("beta")
        if beta_val == 0.0:
            # Return dummy ModeStatistics (won't be used at beta=0)
            n_dim = self.state.n_dim
            return ModeStatistics(
                means=np.zeros((1, n_dim)),
                covariances=np.eye(n_dim).reshape(1, n_dim, n_dim),
                degrees_of_freedom=np.array([self.DOF_FALLBACK]),
            )

        iter_val = self.state.get_current("iter")

        # Trim weights for clustering robustness
        trim_idx, weights_trimmed = trim_weights(
            np.arange(len(weights)), weights, ess=self.TRIM_ESS, bins=self.TRIM_BINS
        )

        if self.clustering and (iter_val % self.cluster_every == 0 or iter_val == 0):
            # Fit clustering model and mode statistics
            u = self.state.get_history("u", flat=True)[trim_idx]
            self.clusterer.fit(u, weights_trimmed)
            labels = self.clusterer.predict(u)
            mode_stats = ModeStatistics.from_particles(
                u, weights_trimmed, labels, dof_fallback=self.DOF_FALLBACK
            )
        elif self.clustering and not (
            iter_val % self.cluster_every == 0 or iter_val == 0
        ):
            # Use previous clustering - return existing mode_stats
            # This requires the caller to keep track of previous mode_stats
            # For now, refit (inefficient but correct)
            u = self.state.get_history("u", flat=True)[trim_idx]
            labels = self.clusterer.predict(u)
            mode_stats = ModeStatistics.from_particles(
                u, weights_trimmed, labels, dof_fallback=self.DOF_FALLBACK
            )
        else:
            # No clustering - fit global Student-t distribution
            u = self.state.get_history("u", flat=True)[trim_idx]
            mode_stats = ModeStatistics.from_global(
                u, weights_trimmed, dof_fallback=self.DOF_FALLBACK
            )

        # Update progress bar with number of clusters (K)
        if self.pbar is not None:
            self.pbar.update_stats(dict(K=mode_stats.K))

        return mode_stats
