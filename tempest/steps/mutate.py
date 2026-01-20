"""Mutation step for Persistent Sampling algorithm."""

import numpy as np
from typing import Optional, Callable, List

from tempest.state_manager import StateManager
from tempest.tools import ProgressBar
from tempest.modes import ModeStatistics
from tempest.mcmc import parallel_mcmc


class Mutator:
    """
    Mutation step for evolving particles using MCMC.

    Applies t-preconditioned Crank-Nicolson or random walk Metropolis
    MCMC to evolve the active particle set.

    Parameters
    ----------
    state : StateManager
        State manager for reading/writing particle state.
    prior_transform : callable
        Function mapping unit hypercube to physical parameters.
    log_likelihood : callable
        Likelihood function (already wrapped with _log_like).
    pbar : ProgressBar, optional
        Progress bar to update with stats.
    n_active_fn : callable
        Function returning current n_active (from Reweighter).
    n_dim : int
        Dimensionality of parameter space.
    n_steps : int
        Number of MCMC steps per particle (default: 10).
    n_max_steps : int
        Maximum number of MCMC steps (default: 1000).
    sampler : str
        MCMC sampler: "tpcn" or "rwm" (default: "tpcn").
    periodic : list, optional
        Indices of periodic parameters.
    reflective : list, optional
        Indices of reflective parameters.
    have_blobs : bool
        Whether likelihood returns auxiliary data (default: False).
    """

    def __init__(
        self,
        state: StateManager,
        prior_transform: Callable,
        log_likelihood: Callable,
        pbar: Optional[ProgressBar] = None,
        n_active_fn: Optional[Callable[[], int]] = None,
        n_dim: int = 1,
        n_steps: int = 10,
        n_max_steps: int = 1000,
        sampler: str = "tpcn",
        periodic: Optional[List[int]] = None,
        reflective: Optional[List[int]] = None,
        have_blobs: bool = False,
    ):
        """Initialize Mutator."""
        self.state = state
        self.prior_transform = prior_transform
        self.log_likelihood = log_likelihood
        self.pbar = pbar
        self.n_active_fn = n_active_fn
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.n_max_steps = n_max_steps
        self.sampler = sampler
        self.periodic = periodic
        self.reflective = reflective
        self.have_blobs = have_blobs

    def run(self, mode_stats: ModeStatistics) -> None:
        """
        Evolve particles using MCMC.

        At beta=0 (warmup phase), draws fresh samples from the prior instead
        of running MCMC, since prior samples are independent.

        Parameters
        ----------
        mode_stats : ModeStatistics
            Mode statistics for MCMC proposal kernel.

        Updates current state:
            - u: evolved unit hypercube coordinates
            - x: evolved physical coordinates
            - logl: evolved log-likelihoods
            - blobs: evolved auxiliary data (if enabled)
            - calls: total likelihood evaluations
            - steps: number of MCMC steps taken
            - acceptance: acceptance rate
            - efficiency: MCMC efficiency
            - logz: corrected log-evidence (if infinite likelihoods)
        """
        # During warmup (beta=0), draw fresh prior samples instead of MCMC
        beta = self.state.get_current("beta")
        if beta == 0.0:
            n_active = self.n_active_fn()
            u = np.random.rand(n_active, self.n_dim)
            x = np.array([self.prior_transform(u[i]) for i in range(n_active)])
            logl, blobs = self.log_likelihood(x)
            assignments = np.zeros(n_active, dtype=int)
            calls = self.state.get_current("calls") + n_active

            self.state.update_current(
                {
                    "u": u,
                    "x": x,
                    "logl": logl,
                    "blobs": blobs,
                    "assignments": assignments,
                    "calls": calls,
                    "steps": 1,
                    "acceptance": 1.0,
                    "efficiency": 1.0,
                }
            )

            # Resample prior particles with infinite likelihoods
            inf_logl_mask = np.isinf(logl)
            if np.any(inf_logl_mask):
                all_idx = np.arange(len(x))
                infinite_idx = all_idx[inf_logl_mask]
                finite_idx = all_idx[~inf_logl_mask]
                if len(finite_idx) > 0:
                    idx = np.random.choice(
                        finite_idx, size=len(infinite_idx), replace=True
                    )
                    x[infinite_idx] = x[idx]
                    u[infinite_idx] = u[idx]
                    logl[infinite_idx] = logl[idx]
                    if self.have_blobs:
                        blobs[infinite_idx] = blobs[idx]

                    self.state.set_current("x", x)
                    self.state.set_current("u", u)
                    self.state.set_current("logl", logl)
                    if self.have_blobs:
                        self.state.set_current("blobs", blobs)

                # Correct logZ for fraction of prior with finite likelihood support
                n_finite = len(finite_idx)
                n_total = len(logl)
                logz = self.state.get_current("logz") + np.log(n_finite / n_total)
                self.state.set_current("logz", logz)
            return

        blobs = (
            self.state.get_current("blobs")
            if self.have_blobs and self.state.get_current("blobs") is not None
            else None
        )

        (
            u,
            x,
            logl,
            blobs,
            efficiency,
            acceptance,
            steps,
            mcmc_calls,
        ) = parallel_mcmc(
            u=self.state.get_current("u"),
            x=self.state.get_current("x"),
            logl=self.state.get_current("logl"),
            blobs=blobs,
            assignments=self.state.get_current("assignments"),
            beta=self.state.get_current("beta"),
            mode_stats=mode_stats,
            log_likelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            progress_bar=self.pbar,
            n_steps=self.n_steps,
            n_max=self.n_max_steps,
            sample=self.sampler,
            periodic=self.periodic,
            reflective=self.reflective,
            verbose=True,
        )

        self.state.update_current(
            {
                "u": u,
                "x": x,
                "logl": logl,
                "efficiency": efficiency,
                "acceptance": acceptance,
                "steps": steps,
            }
        )

        if self.have_blobs:
            self.state.set_current("blobs", blobs.copy())

        calls = self.state.get_current("calls") + mcmc_calls
        self.state.set_current("calls", calls)
