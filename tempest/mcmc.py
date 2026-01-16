import numpy as np
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod

from .modes import ModeStatistics


class _BaseMCMCRunner(ABC):
    """Base class for MCMC runners with common infrastructure."""

    def __init__(
        self,
        u: np.ndarray,
        x: np.ndarray,
        logl: np.ndarray,
        blobs: Optional[np.ndarray],
        assignments: np.ndarray,
        beta: float,
        mode_stats: ModeStatistics,
        log_likelihood: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
        prior_transform: Callable[[np.ndarray], np.ndarray],
        progress_bar: Optional[Callable] = None,
        n_steps: int = 100,
        n_max: int = 1000,
        periodic: Optional[np.ndarray] = None,
        reflective: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        self.beta = beta
        self.mode_stats = mode_stats
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.progress_bar = progress_bar
        self.n_steps = n_steps
        self.n_max = n_max
        self.periodic = periodic
        self.reflective = reflective
        self.verbose = verbose

        # Clone state variables to avoid modifying inputs
        self.u = u.copy()
        self.x = x.copy()
        self.logl = logl.copy()
        self.blobs = blobs.copy() if blobs is not None else None
        self.assignments = assignments.copy()

        self.n_walkers, self.n_dim = x.shape
        self.n_clusters = mode_stats.K
        self.n_calls = 0

        # Initialize sigma
        self.sigma_0 = 2.38 / np.sqrt(self.n_dim)
        self.sigmas = self._initialize_sigmas()

        # Convergence tracking
        self.best_average_logl = np.mean(logl)
        self.cnt = 0
        self.iteration = 0

    @abstractmethod
    def _initialize_sigmas(self) -> np.ndarray:
        """Initialize step sizes for each cluster."""
        pass

    @abstractmethod
    def _propose(self, k: int) -> np.ndarray:
        """Generate proposal for walker k."""
        pass

    @abstractmethod
    def _compute_acceptance_factor(
        self, u_prime: np.ndarray, logl_prime: np.ndarray
    ) -> np.ndarray:
        """Compute acceptance probability factor (beyond likelihood ratio)."""
        pass

    @abstractmethod
    def _adapt_sigma(self, c: int, mean_accept: float):
        """Adapt step size for cluster c."""
        pass

    def _evaluate_likelihood(
        self, x_prime: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate log-likelihood for proposals."""
        if self.blobs is not None:
            logl_prime, blobs_prime = self.log_likelihood(x_prime)
        else:
            logl_prime, _ = self.log_likelihood(x_prime)
            blobs_prime = None
        self.n_calls += self.n_walkers
        return logl_prime, blobs_prime

    def _update_progress_bar(self, alpha: np.ndarray):
        """Update progress bar with current statistics."""
        if self.progress_bar is not None and self.verbose:
            progress_info = {
                "calls": self.progress_bar.info.get("calls", 0) + self.n_walkers,
                "acc": alpha.mean(),
                "steps": self.iteration,
                "logL": self.logl.mean(),
                "eff": self.sigmas.mean() / self.sigma_0,
            }
            self.progress_bar.update_stats(progress_info)

    def _check_convergence(self) -> bool:
        """Check if MCMC has converged."""
        average_logl = self.logl.mean()
        if average_logl > self.best_average_logl:
            self.cnt = 0
            self.best_average_logl = average_logl
        else:
            self.cnt += 1
            threshold = self.n_steps * (self.sigma_0 / np.median(self.sigmas)) ** 2.0
            if self.cnt >= threshold:
                return True

        if self.iteration >= self.n_max:
            return True

        return False

    def run(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float, float, int, int
    ]:
        """Run MCMC sampling."""
        while True:
            self.iteration += 1

            # Generate proposals for all walkers
            u_prime = np.empty_like(self.u)
            for k in range(self.n_walkers):
                u_prime[k] = self._propose(k)

            # Transform to x space
            x_prime = np.array([self.prior_transform(u_p) for u_p in u_prime])

            # Evaluate log-likelihood
            logl_prime, blobs_prime = self._evaluate_likelihood(x_prime)

            # Compute acceptance probability
            alpha = self._compute_acceptance_factor(u_prime, logl_prime)
            alpha = np.exp(self.beta * (logl_prime - self.logl) + alpha)
            alpha = np.minimum(1.0, alpha)
            alpha = np.nan_to_num(alpha, nan=0.0)

            # Metropolis criterion
            u_rand = np.random.rand(self.n_walkers)
            mask_accept = u_rand < alpha

            # Update accepted walkers
            self.u[mask_accept] = u_prime[mask_accept]
            self.x[mask_accept] = x_prime[mask_accept]
            self.logl[mask_accept] = logl_prime[mask_accept]
            if self.blobs is not None:
                self.blobs[mask_accept] = blobs_prime[mask_accept]

            # Adapt sigmas for each cluster
            for c in range(self.n_clusters):
                mask_cluster = self.assignments == c
                if not np.any(mask_cluster):
                    continue

                mean_accept = alpha[mask_cluster].mean()
                self._adapt_sigma(c, mean_accept)

            # Update progress bar
            self._update_progress_bar(alpha)

            # Check convergence
            if self._check_convergence():
                break

        average_efficiency = self.sigmas.mean() / self.sigma_0
        average_acceptance = alpha.mean()

        return (
            self.u,
            self.x,
            self.logl,
            self.blobs,
            average_efficiency,
            average_acceptance,
            self.iteration,
            self.n_calls,
        )


class _TPCNRunner(_BaseMCMCRunner):
    """t-preconditioned Crank-Nicolson MCMC runner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract mode statistics
        self.means = self.mode_stats.means
        self.degrees_of_freedom = self.mode_stats.degrees_of_freedom
        self.inv_covs = self.mode_stats.inv_covariances
        self.chol_covs = self.mode_stats.chol_covariances

    def _initialize_sigmas(self) -> np.ndarray:
        return np.ones(self.n_clusters) * np.minimum(self.sigma_0, 0.99)

    def _propose(self, k: int) -> np.ndarray:
        """Generate t-preconditioned Crank-Nicolson proposal for walker k."""
        mu = self.means[self.assignments[k]]
        diff = self.u[k] - mu
        chol_cov = self.chol_covs[self.assignments[k]]
        sigma = self.sigmas[self.assignments[k]]

        # Compute scaling factor s
        dot_product = diff @ self.inv_covs[self.assignments[k]] @ diff
        gamma_shape = (self.n_dim + self.degrees_of_freedom[self.assignments[k]]) / 2
        gamma_scale = 2.0 / (self.degrees_of_freedom[self.assignments[k]] + dot_product)
        s = 1.0 / np.random.gamma(shape=gamma_shape, scale=gamma_scale)

        # Generate proposal with boundary checking
        while True:
            proposal = (
                mu
                + np.sqrt(1.0 - sigma**2.0) * diff
                + sigma * np.sqrt(s) * chol_cov @ np.random.randn(self.n_dim)
            )
            proposal = apply_boundary_conditions(
                proposal, self.periodic, self.reflective
            )
            if check_bounds(proposal, self.periodic, self.reflective):
                return proposal

    def _compute_acceptance_factor(
        self, u_prime: np.ndarray, logl_prime: np.ndarray
    ) -> np.ndarray:
        """Compute Student-t density ratio for TPCN."""
        means_assigned = self.means[self.assignments]

        # Current state
        diff = self.u - means_assigned
        dot_products = np.einsum(
            "ij,ijk,ik->i", diff, self.inv_covs[self.assignments], diff
        )
        B = (
            -0.5
            * (self.n_dim + self.degrees_of_freedom[self.assignments])
            * np.log(1 + dot_products / self.degrees_of_freedom[self.assignments])
        )

        # Proposed state
        diff_prime = u_prime - means_assigned
        dot_prime = np.einsum(
            "ij,ijk,ik->i", diff_prime, self.inv_covs[self.assignments], diff_prime
        )
        A = (
            -0.5
            * (self.n_dim + self.degrees_of_freedom[self.assignments])
            * np.log(1 + dot_prime / self.degrees_of_freedom[self.assignments])
        )

        return -A + B

    def _adapt_sigma(self, c: int, mean_accept: float):
        """Adapt step size for cluster c with clipping."""
        adaptation_rate = 1.0 / (self.iteration + 1) ** 1.0
        self.sigmas[c] = np.clip(
            self.sigmas[c] + adaptation_rate * (mean_accept - 0.234),
            0,
            min(self.sigma_0, 0.99),
        )


class _RWMRunner(_BaseMCMCRunner):
    """Random Walk Metropolis MCMC runner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chol_covs = self.mode_stats.chol_covariances

    def _initialize_sigmas(self) -> np.ndarray:
        return np.ones(self.n_clusters) * self.sigma_0

    def _propose(self, k: int) -> np.ndarray:
        """Generate random walk Metropolis proposal for walker k."""
        chol_cov = self.chol_covs[self.assignments[k]]
        sigma = self.sigmas[self.assignments[k]]

        while True:
            proposal = self.u[k] + sigma * chol_cov @ np.random.randn(self.n_dim)
            proposal = apply_boundary_conditions(
                proposal, self.periodic, self.reflective
            )
            if check_bounds(proposal, self.periodic, self.reflective):
                return proposal

    def _compute_acceptance_factor(
        self, u_prime: np.ndarray, logl_prime: np.ndarray
    ) -> np.ndarray:
        """RWM has symmetric proposal, so no correction factor."""
        return np.zeros(self.n_walkers)

    def _adapt_sigma(self, c: int, mean_accept: float):
        """Adapt step size for cluster c."""
        adaptation_rate = 1.0 / (self.iteration + 1)
        self.sigmas[c] = self.sigmas[c] + adaptation_rate * (mean_accept - 0.234)


def apply_boundary_conditions(
    u: np.ndarray,
    periodic: Optional[np.ndarray] = None,
    reflective: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply periodic and reflective boundary conditions to proposals.

    Parameters
    ----------
    u : np.ndarray
        Array of transformed parameters (shape: [n_walkers, n_dim] or [n_dim]).
    periodic : Optional[np.ndarray]
        Array of parameter indices with periodic boundary conditions.
    reflective : Optional[np.ndarray]
        Array of parameter indices with reflective boundary conditions.

    Returns
    -------
    np.ndarray
        Array with boundary conditions applied.
    """
    u = u.copy()

    # Apply periodic boundary conditions (wrap around)
    if periodic is not None:
        for idx in periodic:
            u[..., idx] = u[..., idx] % 1.0

    # Apply reflective boundary conditions
    if reflective is not None:
        for idx in reflective:
            # Reflect values outside [0, 1] back into the domain
            val = u[..., idx]
            # Use floor division to determine number of reflections
            n_reflect = np.floor(val).astype(int)
            remainder = val - n_reflect
            # Odd number of reflections means we need to flip
            u[..., idx] = np.where(n_reflect % 2 == 0, remainder, 1.0 - remainder)

    return u


def check_bounds(
    u: np.ndarray,
    periodic: Optional[np.ndarray] = None,
    reflective: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Check if all components are within [0, 1], considering boundary conditions.

    Parameters
    ----------
    u : np.ndarray
        Array of transformed parameters (shape: [n_walkers, n_dim] or [n_dim]).
    periodic : Optional[np.ndarray]
        Array of parameter indices with periodic boundary conditions.
    reflective : Optional[np.ndarray]
        Array of parameter indices with reflective boundary conditions.

    Returns
    -------
    np.ndarray
        Boolean array indicating if each sample is within bounds.
    """
    # Indices that need strict bounds checking (not periodic or reflective)
    n_dim = u.shape[-1]
    all_indices = set(range(n_dim))
    special_indices = set()
    if periodic is not None:
        special_indices.update(periodic)
    if reflective is not None:
        special_indices.update(reflective)
    strict_indices = list(all_indices - special_indices)

    if len(strict_indices) == 0:
        # All indices have boundary conditions, always valid
        if u.ndim == 1:
            return True
        return np.ones(u.shape[0], dtype=bool)

    # Check only strict indices
    u_strict = u[..., strict_indices]
    if u.ndim == 1:
        return np.all(u_strict >= 0) and np.all(u_strict <= 1)
    return np.all(u_strict >= 0, axis=-1) & np.all(u_strict <= 1, axis=-1)


def parallel_mcmc(
    u: np.ndarray,
    x: np.ndarray,
    logl: np.ndarray,
    blobs: Optional[np.ndarray],
    assignments: np.ndarray,
    beta: float,
    mode_stats: ModeStatistics,
    log_likelihood: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    progress_bar: Optional[Callable] = None,
    n_steps: int = 100,
    n_max: int = 1000,
    sample: str = "tpcn",
    periodic: Optional[np.ndarray] = None,
    reflective: Optional[np.ndarray] = None,
    verbose: bool = True,
):
    """
    Perform parallel MCMC sampling with t-preconditioned Crank-Nicolson or Random Walk Metropolis.

    Parameters
    ----------
    u : np.ndarray
        Array of transformed parameters (shape: [n_walkers, n_dim]).
    x : np.ndarray
        Array of parameters in original space (shape: [n_walkers, n_dim]).
    logl : np.ndarray
        Array of log-likelihoods (shape: [n_walkers]).
    blobs : Optional[np.ndarray]
        Array of blobs or auxiliary information (shape: [n_walkers, ...]).
    assignments : np.ndarray
        Array of cluster assignments for each walker (shape: [n_walkers]).
    beta : float
        Inverse temperature parameter.
    mode_stats : ModeStatistics
        Mode statistics object containing means, covariances, degrees of freedom,
        and precomputed inverse covariances and Cholesky decompositions.
    log_likelihood : Callable
        Function to compute log-likelihood given parameters in x space.
    prior_transform : Callable
        Function to transform parameters from u space to x space.
    progress_bar : Optional[Callable], optional
        Function to update progress, by default None.
    n_steps : int, optional
        Number of steps for termination based on adaptation, by default 1000.
    periodic : Optional[np.ndarray], optional
        Array of parameter indices with periodic boundary conditions, by default None.
        Periodic parameters wrap around when they exceed [0, 1].
    reflective : Optional[np.ndarray], optional
        Array of parameter indices with reflective boundary conditions, by default None.
        Reflective parameters bounce back when they exceed [0, 1].

    Returns
    -------
    Tuple containing updated u, x, logl, blobs, average efficiency, average acceptance rate,
    number of iterations, and number of likelihood calls.
    """

    if sample == "rwm":
        return parallel_random_walk_metropolis(
            u,
            x,
            logl,
            blobs,
            assignments,
            beta,
            mode_stats,
            log_likelihood,
            prior_transform,
            progress_bar,
            n_steps,
            n_max,
            periodic,
            reflective,
            verbose,
        )
    else:
        return parallel_t_preconditioned_crank_nicolson(
            u,
            x,
            logl,
            blobs,
            assignments,
            beta,
            mode_stats,
            log_likelihood,
            prior_transform,
            progress_bar,
            n_steps,
            n_max,
            periodic,
            reflective,
            verbose,
        )


def parallel_t_preconditioned_crank_nicolson(
    u: np.ndarray,
    x: np.ndarray,
    logl: np.ndarray,
    blobs: Optional[np.ndarray],
    assignments: np.ndarray,
    beta: float,
    mode_stats: ModeStatistics,
    log_likelihood: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    progress_bar: Optional[Callable] = None,
    n_steps: int = 100,
    n_max: int = 1000,
    periodic: Optional[np.ndarray] = None,
    reflective: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float, float, int, int
]:
    """
    Perform parallel t-preconditioned Crank-Nicolson updates for MCMC sampling.

    Parameters
    ----------
    u : np.ndarray
        Array of transformed parameters (shape: [n_walkers, n_dim]).
    x : np.ndarray
        Array of parameters in original space (shape: [n_walkers, n_dim]).
    logl : np.ndarray
        Array of log-likelihoods (shape: [n_walkers]).
    blobs : Optional[np.ndarray]
        Array of blobs or auxiliary information (shape: [n_walkers, ...]).
    assignments : np.ndarray
        Array of cluster assignments for each walker (shape: [n_walkers]).
    beta : float
        Inverse temperature parameter.
    mode_stats : ModeStatistics
        Mode statistics object containing means, covariances, degrees of freedom,
        and precomputed inverse covariances and Cholesky decompositions.
    log_likelihood : Callable
        Function to compute log-likelihood given parameters in x space.
    prior_transform : Callable
        Function to transform parameters from u space to x space.
    progress_bar : Optional[Callable], optional
        Function to update progress, by default None.
    n_steps : int, optional
        Number of steps for termination based on adaptation, by default 1000.
    n_max : int, optional
        Maximum number of iterations, by default 10000.
    periodic : Optional[np.ndarray], optional
        Array of parameter indices with periodic boundary conditions, by default None.
        Periodic parameters wrap around when they exceed [0, 1].
    reflective : Optional[np.ndarray], optional
        Array of parameter indices with reflective boundary conditions, by default None.
        Reflective parameters bounce back when they exceed [0, 1].

    Returns
    -------
    Tuple containing updated u, x, logl, blobs, average efficiency, average acceptance rate,
    number of iterations, and number of likelihood calls.
    """
    runner = _TPCNRunner(
        u,
        x,
        logl,
        blobs,
        assignments,
        beta,
        mode_stats,
        log_likelihood,
        prior_transform,
        progress_bar,
        n_steps,
        n_max,
        periodic,
        reflective,
        verbose,
    )
    return runner.run()


def parallel_random_walk_metropolis(
    u: np.ndarray,
    x: np.ndarray,
    logl: np.ndarray,
    blobs: Optional[np.ndarray],
    assignments: np.ndarray,
    beta: float,
    mode_stats: ModeStatistics,
    log_likelihood: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    progress_bar: Optional[Callable] = None,
    n_steps: int = 1000,
    n_max: int = 10000,
    periodic: Optional[np.ndarray] = None,
    reflective: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    float,
    float,
    int,
    int,
]:
    """
    Perform parallel Random Walk Metropolis updates for MCMC sampling.

    Parameters
    ----------
    u : np.ndarray
        Array of transformed parameters (shape: [n_walkers, n_dim]).
    x : np.ndarray
        Array of parameters in original space (shape: [n_walkers, n_dim]).
    logl : np.ndarray
        Array of log-likelihoods (shape: [n_walkers]).
    blobs : Optional[np.ndarray]
        Array of blobs or auxiliary information (shape: [n_walkers, ...]).
    assignments : np.ndarray
        Array of cluster assignments for each walker (shape: [n_walkers]).
    beta : float
        Inverse temperature parameter.
    mode_stats : ModeStatistics
        Mode statistics object containing covariances and precomputed Cholesky decompositions.
    log_likelihood : Callable
        Function to compute log-likelihood given parameters in x space.
    prior_transform : Callable
        Function to transform parameters from u space to x space.
    progress_bar : Optional[Callable], optional
        Function to update progress, by default None.
    n_steps : int, optional
        Number of steps for termination based on adaptation, by default 1000.
    n_max : int, optional
        Maximum number of iterations, by default 10000.
    periodic : Optional[np.ndarray], optional
        Array of parameter indices with periodic boundary conditions, by default None.
        Periodic parameters wrap around when they exceed [0, 1].
    reflective : Optional[np.ndarray], optional
        Array of parameter indices with reflective boundary conditions, by default None.
        Reflective parameters bounce back when they exceed [0, 1].

    Returns
    -------
    Tuple containing updated u, x, logl, blobs, average efficiency, average acceptance rate,
    number of iterations, and number of likelihood calls.
    """
    runner = _RWMRunner(
        u,
        x,
        logl,
        blobs,
        assignments,
        beta,
        mode_stats,
        log_likelihood,
        prior_transform,
        progress_bar,
        n_steps,
        n_max,
        periodic,
        reflective,
        verbose,
    )
    return runner.run()
