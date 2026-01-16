import numpy as np
from typing import Callable, Optional, Tuple

from .modes import ModeStatistics


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
    n_calls = 0
    n_walkers, n_dim = x.shape
    n_clusters = mode_stats.K

    # Clone state variables to avoid modifying inputs
    u = u.copy()
    x = x.copy()
    logl = logl.copy()
    if blobs is not None:
        blobs = blobs.copy()
    assignments = assignments.copy()

    # Extract mode statistics (already precomputed in ModeStatistics)
    means = mode_stats.means
    degrees_of_freedom = mode_stats.degrees_of_freedom
    inv_covs = mode_stats.inv_covariances
    chol_covs = mode_stats.chol_covariances

    # Precompute sigmas
    sigma_0 = 2.38 / np.sqrt(n_dim)
    sigmas = np.ones(n_clusters) * np.minimum(sigma_0, 0.99)

    best_average_logl = np.mean(logl)
    cnt = 0
    iteration = 0

    while True:
        iteration += 1

        # Compute differences for all walkers
        means_assigned = means[assignments]  # Shape: [n_walkers, n_dim]
        diff = u - means_assigned  # Shape: [n_walkers, n_dim]

        # Compute scaling factors s for all walkers
        dot_products = np.einsum("ij,ijk,ik->i", diff, inv_covs[assignments], diff)
        gamma_shape = (n_dim + degrees_of_freedom[assignments]) / 2
        gamma_scale = 2.0 / (degrees_of_freedom[assignments] + dot_products)
        s = 1.0 / np.random.gamma(shape=gamma_shape, scale=gamma_scale)

        # Initialize u_prime
        u_prime = np.empty_like(u)

        # Propose new u_prime for each walker, ensuring all components are within [0, 1]
        for k in range(n_walkers):
            mu = means[assignments[k]]
            chol_cov = chol_covs[assignments[k]]
            sigma = sigmas[assignments[k]]
            while True:
                proposal = (
                    mu
                    + np.sqrt(1.0 - sigma**2.0) * diff[k]
                    + sigma * np.sqrt(s[k]) * chol_cov @ np.random.randn(n_dim)
                )
                # Apply boundary conditions for periodic/reflective parameters
                proposal = apply_boundary_conditions(proposal, periodic, reflective)
                if check_bounds(proposal, periodic, reflective):
                    u_prime[k] = proposal
                    break

        # Transform to x space
        x_prime = np.array([prior_transform(u_p) for u_p in u_prime])

        # Evaluate log-likelihood
        if blobs is not None:
            logl_prime, blobs_prime = log_likelihood(x_prime)
        else:
            logl_prime, _ = log_likelihood(x_prime)
            blobs_prime = None

        n_calls += n_walkers

        # Compute Metropolis acceptance factors
        diff_prime = u_prime - means_assigned  # Shape: [n_walkers, n_dim]
        dot_prime = np.einsum(
            "ij,ijk,ik->i", diff_prime, inv_covs[assignments], diff_prime
        )
        A = (
            -0.5
            * (n_dim + degrees_of_freedom[assignments])
            * np.log(1 + dot_prime / degrees_of_freedom[assignments])
        )
        B = (
            -0.5
            * (n_dim + degrees_of_freedom[assignments])
            * np.log(1 + dot_products / degrees_of_freedom[assignments])
        )

        # Calculate acceptance probability
        alpha = np.exp(beta * (logl_prime - logl) - A + B)
        alpha = np.minimum(1.0, alpha)
        alpha = np.nan_to_num(alpha, nan=0.0)

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask_accept = u_rand < alpha

        # Update accepted walkers
        u[mask_accept] = u_prime[mask_accept]
        x[mask_accept] = x_prime[mask_accept]
        logl[mask_accept] = logl_prime[mask_accept]
        if blobs is not None:
            blobs[mask_accept] = blobs_prime[mask_accept]

        # Adapt sigmas and means
        for c in range(n_clusters):
            mask_cluster = assignments == c
            if not np.any(mask_cluster):
                continue

            mean_accept = alpha[mask_cluster].mean()
            adaptation_rate = 1.0 / (iteration + 1) ** 1.0  # r = 1.0

            # Update sigma with diminishing adaptation
            sigmas[c] = np.clip(
                sigmas[c] + adaptation_rate * (mean_accept - 0.234),
                0,
                min(sigma_0, 0.99),
            )

            # Update mean with moving average
            # mean_update = u[mask_cluster].mean(axis=0)
            # means[c] += adaptation_rate * (mean_update - means[c])

        # Update progress bar if provided
        if progress_bar is not None and verbose:
            progress_info = {
                "calls": progress_bar.info.get("calls", 0) + n_walkers,
                "acc": alpha.mean(),
                "steps": iteration,
                "logL": logl.mean(),
                "eff": sigmas.mean() / sigma_0,
            }
            progress_bar.update_stats(progress_info)

        # Check for convergence based on log-likelihood improvement
        average_logl = logl.mean()
        if average_logl > best_average_logl:
            cnt = 0
            best_average_logl = average_logl
        else:
            cnt += 1
            threshold = n_steps * (sigma_0 / np.median(sigmas)) ** 2.0
            if cnt >= threshold:
                break

        # Check maximum iterations
        if iteration >= n_max:
            break

    average_efficiency = sigmas.mean() / sigma_0
    average_acceptance = alpha.mean()

    return u, x, logl, blobs, average_efficiency, average_acceptance, iteration, n_calls


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
    n_calls = 0
    n_walkers, n_dim = x.shape
    n_clusters = mode_stats.K

    # Clone state variables to avoid modifying inputs
    u = u.copy()
    x = x.copy()
    logl = logl.copy()
    if blobs is not None:
        blobs = blobs.copy()
    assignments = assignments.copy()

    # Extract precomputed Cholesky decompositions from mode_stats
    chol_covs = mode_stats.chol_covariances

    # Precompute sigmas
    sigma_0 = 2.38 / np.sqrt(n_dim)
    sigmas = np.ones(n_clusters) * sigma_0

    best_average_logl = np.mean(logl)
    cnt = 0
    iteration = 0

    while True:
        iteration += 1

        # Propose new u_prime for each walker, ensuring all components are within [0, 1]
        u_prime = np.empty_like(u)
        for k in range(n_walkers):
            chol_cov = chol_covs[assignments[k]]
            sigma = sigmas[assignments[k]]
            while True:
                proposal = u[k] + sigma * chol_cov @ np.random.randn(n_dim)
                # Apply boundary conditions for periodic/reflective parameters
                proposal = apply_boundary_conditions(proposal, periodic, reflective)
                if check_bounds(proposal, periodic, reflective):
                    u_prime[k] = proposal
                    break

        # Transform to x space
        x_prime = np.array([prior_transform(u_p) for u_p in u_prime])

        # Evaluate log-likelihood
        if blobs is not None:
            logl_prime, blobs_prime = log_likelihood(x_prime)
        else:
            logl_prime, _ = log_likelihood(x_prime)
            blobs_prime = None

        n_calls += n_walkers

        # Calculate acceptance probability
        alpha = np.exp(beta * (logl_prime - logl))
        alpha = np.minimum(1.0, alpha)
        alpha = np.nan_to_num(alpha, nan=0.0)

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask_accept = u_rand < alpha

        # Update accepted walkers
        u[mask_accept] = u_prime[mask_accept]
        x[mask_accept] = x_prime[mask_accept]
        logl[mask_accept] = logl_prime[mask_accept]
        if blobs is not None:
            blobs[mask_accept] = blobs_prime[mask_accept]

        # Adapt sigmas
        for c in range(n_clusters):
            mask_cluster = assignments == c
            if not np.any(mask_cluster):
                continue

            mean_accept = alpha[mask_cluster].mean()
            adaptation_rate = 1.0 / (iteration + 1)  # r = 1.0

            # Update sigma with diminishing adaptation
            sigmas[c] = (sigmas[c] + adaptation_rate * (mean_accept - 0.234),)

        # Update progress bar if provided
        if progress_bar is not None and verbose:
            progress_info = {
                "calls": getattr(progress_bar, "info", {}).get("calls", 0) + n_walkers,
                "acc": alpha.mean(),
                "steps": iteration,
                "logL": logl.mean(),
                "eff": sigmas.mean() / sigma_0,
            }
            progress_bar.update_stats(progress_info)

        # Check for convergence based on log-likelihood improvement
        average_logl = logl.mean()
        if average_logl > best_average_logl:
            cnt = 0
            best_average_logl = average_logl
        else:
            cnt += 1
            threshold = n_steps * (sigma_0 / np.median(sigmas)) ** 2.0
            if cnt >= threshold:
                break

        # Check maximum iterations
        if iteration >= n_max:
            break

    average_efficiency = sigmas.mean() / sigma_0
    average_acceptance = alpha.mean()

    return u, x, logl, blobs, average_efficiency, average_acceptance, iteration, n_calls
