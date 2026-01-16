"""Mode statistics for t-preconditioned Crank-Nicolson MCMC."""

import numpy as np
from typing import Union, Optional
from tempest.student import fit_mvstud


class ModeStatistics:
    """
    Encapsulates mode statistics for t-preconditioned Crank-Nicolson MCMC.

    Stores the mean, covariance, and degrees-of-freedom for each mode,
    along with precomputed Cholesky decompositions and inverse covariances.

    These statistics define multivariate Student-t distributions used by
    the tpCN proposal kernel to efficiently sample from multimodal posteriors.

    Parameters
    ----------
    means : np.ndarray
        Mode means (shape: [K, n_dim])
    covariances : np.ndarray
        Mode covariances (shape: [K, n_dim, n_dim])
    degrees_of_freedom : np.ndarray
        Degrees of freedom for each mode (shape: [K])

    Attributes
    ----------
    means : np.ndarray
        Mode means (shape: [K, n_dim])
    covariances : np.ndarray
        Mode covariances (shape: [K, n_dim, n_dim])
    degrees_of_freedom : np.ndarray
        Degrees of freedom for each mode (shape: [K])
    inv_covariances : np.ndarray
        Precomputed inverse covariances (shape: [K, n_dim, n_dim])
    chol_covariances : np.ndarray
        Precomputed Cholesky decompositions (shape: [K, n_dim, n_dim])
    K : int
        Number of modes
    n_dim : int
        Dimensionality of parameter space

    Notes
    -----
    The inverse covariances and Cholesky decompositions are precomputed
    during construction to avoid redundant computation during MCMC mutation.

    Examples
    --------
    >>> # Fit from clustered particles
    >>> mode_stats = ModeStatistics.from_particles(u, weights, labels)
    >>> # Fit from global particle set (no clustering)
    >>> mode_stats = ModeStatistics.from_global(u, weights)
    """

    DOF_FALLBACK = 1.0

    def __init__(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        degrees_of_freedom: np.ndarray,
    ):
        """
        Initialize ModeStatistics from precomputed parameters.

        Parameters
        ----------
        means : np.ndarray
            Mode means (shape: [K, n_dim])
        covariances : np.ndarray
            Mode covariances (shape: [K, n_dim, n_dim])
        degrees_of_freedom : np.ndarray
            Degrees of freedom for each mode (shape: [K])
        """
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)
        self.degrees_of_freedom = np.asarray(degrees_of_freedom)

        # Ensure 2D shape for single mode case
        if self.means.ndim == 1:
            self.means = self.means.reshape(1, -1)
        if self.covariances.ndim == 2:
            self.covariances = self.covariances.reshape(1, *self.covariances.shape)
        if self.degrees_of_freedom.ndim == 0:
            self.degrees_of_freedom = np.array([self.degrees_of_freedom])

        # Validate shapes
        K, n_dim = self.means.shape
        if self.covariances.shape != (K, n_dim, n_dim):
            raise ValueError(
                f"Covariances shape {self.covariances.shape} incompatible with "
                f"means shape {self.means.shape}"
            )
        if self.degrees_of_freedom.shape != (K,):
            raise ValueError(
                f"Degrees of freedom shape {self.degrees_of_freedom.shape} "
                f"incompatible with K={K}"
            )

        # Precompute derived quantities for efficient MCMC
        self.inv_covariances = np.linalg.inv(self.covariances)
        self.chol_covariances = np.linalg.cholesky(self.covariances)

    @property
    def K(self) -> int:
        """Number of modes."""
        return self.means.shape[0]

    @property
    def n_dim(self) -> int:
        """Dimensionality of parameter space."""
        return self.means.shape[1]

    @classmethod
    def from_particles(
        cls,
        u: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray,
        dof_fallback: float = DOF_FALLBACK,
        resample_factor: int = 4,
    ) -> "ModeStatistics":
        """
        Fit Student-t distributions to weighted particles per cluster.

        For each cluster identified by unique labels, fits a multivariate
        Student-t distribution using the EM algorithm. Particles are
        resampled according to weights for robust fitting.

        Parameters
        ----------
        u : np.ndarray
            Particle positions in unit hypercube (shape: [N, n_dim])
        weights : np.ndarray
            Particle weights (shape: [N]). Will be normalized to sum to 1.
        labels : np.ndarray
            Cluster assignments for each particle (shape: [N])
        dof_fallback : float, optional
            Fallback degrees of freedom if fitting returns non-finite value.
            Default is 1.0.
        resample_factor : int, optional
            Multiplier for resampling particles for robust fitting.
            Each cluster is resampled to `n_cluster * resample_factor` particles.
            Default is 4.

        Returns
        -------
        ModeStatistics
            Fitted mode statistics with K modes where K = number of unique labels.

        Notes
        -----
        Particles are resampled with replacement according to their weights
        before fitting to ensure robust parameter estimation even with
        heavily weighted particles.
        """
        u = np.asarray(u)
        weights = np.asarray(weights)
        labels = np.asarray(labels)

        if u.shape[0] != weights.shape[0] or u.shape[0] != labels.shape[0]:
            raise ValueError("u, weights, and labels must have compatible shapes")

        # Normalize weights
        weights = weights / np.sum(weights)

        means = []
        covariances = []
        degrees_of_freedom = []

        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Extract particles for this cluster
            idx_cluster = np.where(labels == label)[0]
            u_cluster = u[idx_cluster]
            weights_cluster = weights[idx_cluster]
            weights_cluster = weights_cluster / np.sum(weights_cluster)

            # Resample weighted particles for robust fitting
            n_cluster = len(u_cluster)
            n_resample = n_cluster * resample_factor
            idx_resample = np.random.choice(
                n_cluster, size=n_resample, replace=True, p=weights_cluster
            )
            u_resampled = u_cluster[idx_resample]

            # Fit multivariate Student-t distribution
            mean, covariance, dof = fit_mvstud(u_resampled)

            # Apply fallback for non-finite DOF
            if ~np.isfinite(dof):
                dof = dof_fallback

            means.append(mean)
            covariances.append(covariance)
            degrees_of_freedom.append(dof)

        return cls(
            means=np.array(means),
            covariances=np.array(covariances),
            degrees_of_freedom=np.array(degrees_of_freedom),
        )

    @classmethod
    def from_global(
        cls,
        u: np.ndarray,
        weights: np.ndarray,
        dof_fallback: float = DOF_FALLBACK,
        resample_factor: int = 4,
    ) -> "ModeStatistics":
        """
        Fit a single global Student-t distribution to weighted particles.

        Used when clustering is disabled or for the initial iteration.
        Fits one multivariate Student-t distribution to the entire particle set.

        Parameters
        ----------
        u : np.ndarray
            Particle positions in unit hypercube (shape: [N, n_dim])
        weights : np.ndarray
            Particle weights (shape: [N]). Will be normalized to sum to 1.
        dof_fallback : float, optional
            Fallback degrees of freedom if fitting returns non-finite value.
            Default is 1.0.
        resample_factor : int, optional
            Multiplier for resampling particles for robust fitting.
            Particles are resampled to `N * resample_factor`.
            Default is 4.

        Returns
        -------
        ModeStatistics
            Fitted mode statistics with K=1 (single global mode).

        Notes
        -----
        Particles are resampled with replacement according to their weights
        before fitting to ensure robust parameter estimation even with
        heavily weighted particles.
        """
        u = np.asarray(u)
        weights = np.asarray(weights)

        if u.shape[0] != weights.shape[0]:
            raise ValueError("u and weights must have same length")

        # Normalize weights
        weights = weights / np.sum(weights)

        # Resample weighted particles for robust fitting
        n_particles = u.shape[0]
        n_resample = n_particles * resample_factor
        idx_resample = np.random.choice(
            n_particles, size=n_resample, replace=True, p=weights
        )
        u_resampled = u[idx_resample]

        # Fit multivariate Student-t distribution
        mean, covariance, dof = fit_mvstud(u_resampled)

        # Apply fallback for non-finite DOF
        if ~np.isfinite(dof):
            dof = dof_fallback

        return cls(
            means=mean.reshape(1, -1),
            covariances=covariance.reshape(1, *covariance.shape),
            degrees_of_freedom=np.array([dof]),
        )

    def __repr__(self) -> str:
        """String representation of ModeStatistics."""
        return (
            f"ModeStatistics(K={self.K}, n_dim={self.n_dim}, "
            f"dof={self.degrees_of_freedom})"
        )
