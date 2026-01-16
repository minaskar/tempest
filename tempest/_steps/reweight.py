"""Reweighting step for Persistent Sampling algorithm."""

import numpy as np
from typing import Optional

from tempest.state_manager import StateManager
from tempest.tools import ProgressBar, effective_sample_size, unique_sample_size


class Reweighter:
    """
    Reweighting step for determining next temperature level.

    Uses bisection to find the next inverse temperature (beta) that achieves
    a target effective sample size (ESS) from importance weights.

    Parameters
    ----------
    state : StateManager
        State manager for reading/writing particle state.
    pbar : ProgressBar, optional
        Progress bar to update with stats.
    n_effective : int
        Target number of effective particles.
    n_active : int
        Current number of active particles.
    metric : str
        Metric for effective sample size: "ess" or "uss" (default: "ess").
    ESS_TOLERANCE : float
        Relative tolerance for ESS target (default: 0.01).
    BETA_TOLERANCE : float
        Tolerance for beta convergence (default: 1e-4).
    dynamic : bool
        Whether to dynamically adjust n_effective based on unique samples (default: False).
    DYNAMIC_RATIO_LOWER : float
        Lower ratio bound for dynamic adjustment (default: 0.95).
    DYNAMIC_RATIO_UPPER : float
        Upper ratio bound for dynamic adjustment (default: 1.05).
    n_boost : int, optional
        Target n_effective for boosting towards posterior (default: None).
    n_effective_init : int
        Initial n_effective before boosting (only used if n_boost is set).
    n_active_init : int
        Initial n_active before boosting (only used if n_boost is set).
    BOOST_STEEPNESS : float
        Steepness parameter for boosting curve (default: 2.0).

    Attributes
    ----------
    n_effective : int
        Current target number of effective particles (mutable).
    n_active : int
        Current number of active particles (mutable).
    """

    def __init__(
        self,
        state: StateManager,
        pbar: Optional[ProgressBar] = None,
        n_effective: int = 512,
        n_active: int = 256,
        metric: str = "ess",
        ESS_TOLERANCE: float = 0.01,
        BETA_TOLERANCE: float = 1e-4,
        dynamic: bool = False,
        DYNAMIC_RATIO_LOWER: float = 0.95,
        DYNAMIC_RATIO_UPPER: float = 1.05,
        n_boost: Optional[int] = None,
        n_effective_init: int = 512,
        n_active_init: int = 256,
        BOOST_STEEPNESS: float = 2.0,
    ):
        """Initialize Reweighter."""
        self.state = state
        self.pbar = pbar

        # Mutable parameters (modified during dynamic/boost)
        self.n_effective = n_effective
        self.n_active = n_active

        # Configuration
        self.metric = metric
        self.ESS_TOLERANCE = ESS_TOLERANCE
        self.BETA_TOLERANCE = BETA_TOLERANCE
        self.dynamic = dynamic
        self.DYNAMIC_RATIO_LOWER = DYNAMIC_RATIO_LOWER
        self.DYNAMIC_RATIO_UPPER = DYNAMIC_RATIO_UPPER
        self.n_boost = n_boost
        self.n_effective_init = n_effective_init
        self.n_active_init = n_active_init
        self.BOOST_STEEPNESS = BOOST_STEEPNESS

        # Compute dynamic ratio for reference
        if self.dynamic and self.n_active > 0:
            self.dynamic_ratio = self.n_effective / self.n_active
        else:
            self.dynamic_ratio = 1.0

    def run(self) -> np.ndarray:
        """
        Determine next temperature level and compute importance weights.

        Updates state with:
            - iter: incremented iteration number
            - beta: new inverse temperature
            - logz: new log-evidence estimate
            - ess: new effective sample size

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
                    "ess": self.n_effective,
                }
            )
            if self.pbar is not None:
                self.pbar.update_stats(
                    dict(beta=0.0, ESS=int(self.n_effective), logZ=0.0)
                )
            return np.ones(self.n_active) / self.n_active

        beta_prev = self.state.get_current("beta")
        beta_max = 1.0
        beta_min = np.copy(beta_prev)

        def get_weights_and_ess(beta):
            logw, _ = self.state.compute_logw_and_logz(beta)
            weights = np.exp(logw - np.max(logw))
            if self.metric == "ess":
                ess_est = effective_sample_size(weights)
            elif self.metric == "uss":
                ess_est = unique_sample_size(weights)
            return weights, ess_est

        weights_prev, ess_est_prev = get_weights_and_ess(beta_prev)
        weights_max, ess_est_max = get_weights_and_ess(beta_max)

        if ess_est_prev <= self.n_effective:
            beta = beta_prev
            weights = weights_prev
            logz = self.state.get_current("logz")
            ess_est = ess_est_prev
            if self.pbar is not None:
                self.pbar.update_stats(
                    dict(beta=beta, ESS=int(ess_est_prev), logZ=logz)
                )
        elif ess_est_max >= self.n_effective:
            beta = beta_max
            weights = weights_max
            _, logz = self.state.compute_logw_and_logz(beta)
            ess_est = ess_est_max
            if self.pbar is not None:
                self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est_max), logZ=logz))
        else:
            while True:
                beta = (beta_max + beta_min) * 0.5

                weights, ess_est = get_weights_and_ess(beta)

                if (
                    np.abs(ess_est - self.n_effective)
                    < self.ESS_TOLERANCE * self.n_effective
                    or beta == 1.0
                ):
                    _, logz = self.state.compute_logw_and_logz(beta)
                    if self.pbar is not None:
                        self.pbar.update_stats(
                            dict(beta=beta, ESS=int(ess_est), logZ=logz)
                        )
                    break
                elif ess_est < self.n_effective:
                    beta_max = beta
                else:
                    beta_min = beta

        logw, _ = self.state.compute_logw_and_logz(beta)
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)

        if self.dynamic:
            # Adjust the number of effective particles based on the expected number of unique particles
            n_unique_active = unique_sample_size(weights, k=self.n_active)
            # Maintain the original ratio of unique active to effective particles
            if n_unique_active < self.n_active * (
                self.DYNAMIC_RATIO_LOWER * self.dynamic_ratio
            ):
                self.n_effective = int(
                    self.n_active / n_unique_active * self.n_effective
                )
            elif n_unique_active > self.n_active * np.minimum(
                self.DYNAMIC_RATIO_UPPER * self.dynamic_ratio, 1.0
            ):
                self.n_effective = int(
                    n_unique_active / self.n_active * self.n_effective
                )

        if self.n_boost is not None:
            _, posterior_ess = get_weights_and_ess(1.0)

            r = (posterior_ess - 1.0) / self.n_effective
            new_n_effective = int((1 - r) * self.n_effective_init + r * self.n_boost)
            new_n_effective = min(new_n_effective, self.n_boost)
            if new_n_effective > self.n_effective:
                self.n_effective = new_n_effective
                target_n_active = self.n_boost // 2
                self.n_active = int(
                    self.n_active_init
                    + (target_n_active - self.n_active_init) * r**self.BOOST_STEEPNESS
                )

            # Cap n_effective and n_active at n_boost and n_boost // 2
            if self.n_effective > self.n_boost:
                self.n_effective = self.n_boost
            max_n_active = self.n_boost // 2
            if self.n_active > max_n_active:
                self.n_active = max_n_active

        self.state.update_current(
            {
                "logz": logz,
                "beta": beta,
                "ess": ess_est,
            }
        )
        return weights
