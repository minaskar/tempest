"""Reweighting step for Persistent Sampling algorithm."""

from typing import Optional

import numpy as np

from tempest.state_manager import StateManager
from tempest.tools import ProgressBar, effective_sample_size, volume_variation


class Reweighter:
    """
    Reweighting step for determining next temperature level.

    Uses bisection to find the next inverse temperature (beta) that achieves
    a target metric value from importance weights. Supports two modes:
    - ESS mode (volume_variation=None): Target effective sample size (ESS = ess_ratio * n_particles)
    - Dynamic mode (volume_variation=float): ESS + dynamic adjustment based on volume variation

    Parameters
    ----------
    state : StateManager
        State manager for reading/writing particle state.
    pbar : ProgressBar, optional
        Progress bar to update with stats.
    n_particles : int
        Number of particles (fixed during run).
    ess_ratio : float
        Target ESS ratio (ESS / n_particles) for ESS mode.
    volume_variation : float, optional
        Target coefficient of variation for volume. None disables dynamic mode.
    ESS_TOLERANCE : float
        Relative tolerance for metric target (default: 0.01).
    BETA_TOLERANCE : float
        Tolerance for beta convergence (default: 1e-4).

    Attributes
    ----------
    n_particles : int
        Number of particles (fixed during run).
    target_metric : float
        Target value for the metric.
    volume_variation : float, optional
        Target coefficient of variation for volume. None for ESS mode.
    """

    def __init__(
        self,
        state: StateManager,
        pbar: Optional[ProgressBar] = None,
        n_particles: int = 256,
        ess_ratio: float = 2.0,
        volume_variation: Optional[float] = None,
        ESS_TOLERANCE: float = 0.001,
        BETA_TOLERANCE: float = 1e-5,
    ):
        """Initialize Reweighter."""
        self.state = state
        self.pbar = pbar

        # Store configuration
        self.n_particles = n_particles
        self.ess_ratio = ess_ratio
        self.volume_variation = volume_variation

        # Compute target metric based on mode
        if volume_variation is not None:
            self.target_metric = volume_variation
        else:
            self.target_metric = ess_ratio * n_particles

        # Algorithm constants
        self.ESS_TOLERANCE = ESS_TOLERANCE
        self.BETA_TOLERANCE = BETA_TOLERANCE

    def _compute_metric_and_weights(self, beta: float) -> tuple:
        """Compute weights, ESS, and metric value for a given beta.

        Parameters
        ----------
        beta : float
            Inverse temperature value.

        Returns
        -------
        weights : np.ndarray
            Unnormalized importance weights.
        ess_est : float
            Effective sample size estimate.
        metric_val : float
            Metric value (ESS or volume_variation depending on self.metric).
        """
        logw, _ = self.state.compute_logw_and_logz(beta)
        weights = np.exp(logw - np.max(logw))
        ess_est = effective_sample_size(weights)

        if self.volume_variation is not None:
            # dynamic mode - compute volume variation metric
            u = self.state.get_history("u", flat=True)
            weights_norm = weights / np.sum(weights)
            metric_val = volume_variation(u, weights_norm)
        else:
            # ESS mode - metric is just ESS
            metric_val = ess_est

        return weights, ess_est, metric_val

    # Maximum number of bisection iterations before forced exit
    _MAX_BISECTION_ITERATIONS = 200

    def _find_beta_bisection(
        self,
        beta_min: float,
        beta_max: float,
        target: float,
        metric_fn: callable,
    ) -> tuple:
        """Find beta via bisection to achieve target metric value.

        Continues iterating until the metric converges within
        ``ESS_TOLERANCE * target``, using ``BETA_TOLERANCE`` only as a
        sanity bound (with ``_MAX_BISECTION_ITERATIONS`` as a hard limit)
        to avoid runaway loops.

        Parameters
        ----------
        beta_min : float
            Lower bound for beta search.
        beta_max : float
            Upper bound for beta search.
        target : float
            Target metric value to achieve.
        metric_fn : callable
            Function taking beta and returning (metric_value, aux_data).
            aux_data is passed through and returned.

        Returns
        -------
        beta : float
            Found beta value.
        aux_data : any
            Auxiliary data returned by metric_fn at the final beta.
        """
        for _ in range(self._MAX_BISECTION_ITERATIONS):
            beta = (beta_max + beta_min) * 0.5

            metric_val, aux_data = metric_fn(beta)

            # Handle non-finite metric values (e.g., from rank-deficient covariance)
            if not np.isfinite(metric_val):
                if self.volume_variation is not None:
                    metric_val = (
                        1e10  # Treat as very large (bad) value for volume_variation
                    )
                else:
                    metric_val = 1e10  # Treat as very large (ESS too high)

            # Check convergence: metric must be within tolerance of target
            metric_converged = np.abs(metric_val - target) < self.ESS_TOLERANCE * target

            if metric_converged or beta == 1.0:
                return beta, aux_data

            # Update bisection bounds based on metric direction
            if self.volume_variation is None:
                # For ESS: as beta increases, ESS decreases
                # metric_val < target: ESS is too low, need to decrease beta
                if metric_val < target:
                    beta_max = beta
                else:
                    beta_min = beta
            else:  # dynamic mode with volume_variation
                # For volume_variation: as beta increases, CV increases
                # When CV is above target (too high), we need lower beta
                if metric_val < target:
                    # CV is below target (good), can try higher beta
                    beta_min = beta
                else:
                    # CV is above target (too high), need lower beta
                    beta_max = beta

        # Safety exit: max iterations reached.  Return the best estimate.
        return beta, aux_data

    def _find_ess_bracket(
        self,
        beta_current: float,
        ess_target: float,
    ) -> tuple:
        """Find bracket [beta_low, beta_high] where ESS crosses the target.

        Finds two beta values such that ESS(beta_low) >= ess_target and
        ESS(beta_high) < ess_target. This bracket can then be used for
        bisection to find the exact beta where ESS = ess_target.

        Parameters
        ----------
        beta_current : float
            Current beta value (lower bound for search).
        ess_target : float
            Target ESS value.

        Returns
        -------
        beta_low : float
            Beta where ESS >= ess_target. Equals beta_current if ESS
            is already below target at beta_current (can't advance).
        beta_high : float
            Beta where ESS < ess_target. Equals beta_current if ESS
            is already below target (both endpoints equal). Equals 1.0
            if ESS >= target throughout [beta_current, 1.0] (both
            endpoints equal, meaning no crossing exists).
        """
        beta_low = beta_current
        beta_high = 1.0

        # Check if ESS already below (or essentially equal to) target at
        # beta_current.  When ESS == target at the current beta, ANY higher
        # beta will give ESS < target, so the "first beta > 0 where ESS =
        # target" is infinitesimally close to beta_current.  Advancing to
        # such a tiny step is wasteful (the MCMC move would be equivalent
        # to prior sampling).  Using <= instead of < ensures we stay and
        # accumulate more particles when ESS exactly equals the target.
        _, ess_at_current, _ = self._compute_metric_and_weights(beta_current)
        if ess_at_current <= ess_target:
            return beta_current, beta_current  # Can't advance, stay at current beta

        # Check if ESS >= target throughout [beta_current, 1.0]
        _, ess_at_one, _ = self._compute_metric_and_weights(1.0)
        if ess_at_one >= ess_target:
            return 1.0, 1.0  # Valid throughout, can go all the way to 1.0

        # Bisection to find where ESS drops below target
        while beta_high - beta_low > self.BETA_TOLERANCE:
            beta_mid = (beta_high + beta_low) * 0.5
            _, ess_mid, _ = self._compute_metric_and_weights(beta_mid)

            if ess_mid >= ess_target:
                # ESS still sufficient, can try higher beta
                beta_low = beta_mid
            else:
                # ESS too low, need lower beta
                beta_high = beta_mid

        return beta_low, beta_high

    def _finalize_iteration(
        self,
        beta: float,
        weights: np.ndarray,
        ess_est: float,
        logz: float,
        cv: Optional[float] = None,
    ) -> np.ndarray:
        """Finalize iteration by updating state and returning normalized weights.

        Parameters
        ----------
        beta : float
            Current beta value.
        weights : np.ndarray
            Unnormalized importance weights.
        ess_est : float
            Effective sample size.
        logz : float
            Log evidence estimate.
        cv : float, optional
            Volume variation (coefficient of variation) at this iteration.

        Returns
        -------
        weights : np.ndarray
            Normalized importance weights.
        """
        # Normalize weights before returning
        weights = weights / np.sum(weights)

        update = {
            "logz": logz,
            "beta": beta,
            "ess": ess_est,
        }
        if cv is not None:
            update["cv"] = cv

        self.state.update_current(update)
        return weights

    def run(self) -> np.ndarray:
        """
        Determine next temperature level and compute importance weights.

        Updates state with:
            - iter: incremented iteration number
            - beta: new inverse temperature
            - logz: new log-evidence estimate
            - ess: new effective sample size
            - cv: volume variation at determined beta

        Returns
        -------
        weights : np.ndarray
            Normalized importance weights for all historical particles.
            Shape: (n_total,) where n_total is sum of all historical samples.
        """
        # Update iteration index
        iter_val = self.state.get_current("iter") + 1
        self.state.set_current("iter", iter_val)
        if self.pbar is not None:
            self.pbar.update_iter()

        # Handle first iteration (no particles yet)
        if self.state.get_history_length() == 0:
            self.state.update_current(
                {
                    "beta": 0.0,
                    "logz": 0.0,
                    "ess": self.ess_ratio * self.n_particles,
                    "cv": 0.0,
                }
            )
            if self.pbar is not None:
                self.pbar.update_stats(
                    dict(
                        beta=0.0,
                        ESS=int(self.ess_ratio * self.n_particles),
                        logZ=0.0,
                        CV=0.0,
                    )
                )
            return np.ones(self.n_particles) / self.n_particles

        beta_prev = self.state.get_current("beta")

        # Step 1: Find bracket where ESS crosses the target
        # Returns (beta_low, beta_high) where ESS(beta_low) >= target and
        # ESS(beta_high) < target. If both equal, no crossing exists.
        ess_target = self.ess_ratio * self.n_particles
        beta_low, beta_high = self._find_ess_bracket(beta_prev, ess_target)

        if self.volume_variation is None:
            # For ESS mode: find beta where ESS = ess_ratio * n_particles

            def ess_fn(beta):
                weights, ess_est, _ = self._compute_metric_and_weights(beta)
                return ess_est, (weights, ess_est)

            if beta_low == beta_high:
                # No crossing exists: either can't advance (ESS < target at
                # beta_prev) or ESS >= target all the way to 1.0
                beta = beta_low
                weights, ess_est, _ = self._compute_metric_and_weights(beta)
            else:
                # Crossing exists: bisect in [beta_prev, beta_high] to find
                # where ESS = target. ESS(beta_prev) >= target (guaranteed
                # since beta_low != beta_high) and ESS(beta_high) < target.
                beta, (weights, ess_est) = self._find_beta_bisection(
                    beta_prev,
                    beta_high,
                    ess_target,
                    ess_fn,
                )

            # Compute volume variation at determined beta
            u = self.state.get_history("u", flat=True)
            weights_norm = weights / np.sum(weights)
            cv = volume_variation(u, weights_norm)

            _, logz = self.state.compute_logw_and_logz(beta)
            if self.pbar is not None:
                self.pbar.update_stats(
                    dict(beta=beta, ESS=int(ess_est), logZ=logz, CV=cv)
                )
            return self._finalize_iteration(beta, weights, ess_est, logz, cv=cv)
        else:
            # For dynamic mode: use beta_upper as upper limit
            # If beta_upper == beta_prev, we stay at current beta
            # Otherwise, search in [beta_prev, beta_upper] for target CV

            if beta_low == beta_high:
                # No crossing: stay at current beta or go to 1.0
                beta = beta_low
                weights, ess_est, _ = self._compute_metric_and_weights(beta)
            else:
                # Compute volume_variation at boundaries of ESS-safe range
                _, ess_at_prev, vol_var_prev = self._compute_metric_and_weights(
                    beta_prev
                )
                # beta_low is the maximum beta where ESS >= target
                # (same role as the old beta_upper)
                _, ess_at_limit, vol_var_limit = self._compute_metric_and_weights(
                    beta_low
                )

                # Determine beta based on volume_variation at boundaries
                # As beta increases, volume_variation increases (unlike ESS which decreases)
                # At beta_prev: vol_var is lower (less constrained)
                # At beta_low: vol_var is higher (more constrained, ESS ~ target)
                if self.volume_variation >= vol_var_limit:
                    # Target is above what we can achieve at beta_low (too conservative)
                    # Use beta_low (best we can do while maintaining ESS >= target)
                    beta = beta_low
                    weights = None
                    ess_est = ess_at_limit
                elif self.volume_variation <= vol_var_prev:
                    # Target is below current vol_var (too ambitious)
                    # Stay at beta_prev to collect more samples and potentially improve coverage
                    beta = beta_prev
                    weights = None
                    ess_est = ess_at_prev
                else:
                    # Crossing exists: use bisection to find where volume_variation = target
                    # Search within ESS-safe range [beta_prev, beta_low]
                    def volume_variation_fn(beta):
                        weights, ess_est, metric_val = self._compute_metric_and_weights(
                            beta
                        )
                        return metric_val, (weights, ess_est)

                    beta, (weights, ess_est) = self._find_beta_bisection(
                        beta_prev,
                        beta_low,
                        self.volume_variation,
                        volume_variation_fn,
                    )

                if weights is None:
                    weights, ess_est, _ = self._compute_metric_and_weights(beta)

            # Compute volume variation at determined beta
            u = self.state.get_history("u", flat=True)
            weights_norm = weights / np.sum(weights)
            cv = volume_variation(u, weights_norm)

            _, logz = self.state.compute_logw_and_logz(beta)

            if self.pbar is not None:
                self.pbar.update_stats(
                    dict(beta=beta, ESS=int(ess_est), logZ=logz, CV=cv)
                )
            return self._finalize_iteration(beta, weights, ess_est, logz, cv=cv)
