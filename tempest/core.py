from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import (
    SamplerConfig,
    BETA_TOLERANCE,
    ESS_TOLERANCE,
    TRIM_ESS,
    TRIM_BINS,
    DOF_FALLBACK,
    BOOST_STEEPNESS,
)
from .state_manager import StateManager


class SamplerCore:
    """
    Internal coordinator that handles the sampling algorithm.
    Not part of public API - Sampler delegates to this.
    """

    def __init__(
        self,
        config: SamplerConfig,
        state: StateManager,
    ):
        """Initialize sampler core with validated configuration."""
        self.config = config
        self.state = state

        # Initialize components (moved from Sampler._initialize_steps)
        from .steps.reweight import Reweighter
        from .steps.train import Trainer
        from .steps.resample import Resampler
        from .steps.mutate import Mutator

        self.reweighter = Reweighter(
            state=self.state,
            pbar=None,  # Will be set in run_sampling
            n_effective=config.n_effective,
            n_active=config.n_active,
            metric=config.metric,
            ESS_TOLERANCE=ESS_TOLERANCE,  # From centralized config
            BETA_TOLERANCE=BETA_TOLERANCE,  # From centralized config
            n_boost=config.n_boost,
            n_effective_init=config.n_effective,
            n_active_init=config.n_active,
            BOOST_STEEPNESS=BOOST_STEEPNESS,  # From centralized config
        )

        # Initialize clusterer if clustering is enabled
        clusterer = None
        if config.clustering:
            from .cluster import HierarchicalGaussianMixture

            clusterer = HierarchicalGaussianMixture(
                n_init=1,
                max_iterations=1000
                if config.n_max_clusters is None
                else config.n_max_clusters - 1,
                min_points=None if config.n_max_clusters is None else 4 * config.n_dim,
                threshold_modifier=config.split_threshold,
                covariance_type="full",
                verbose=False,
                normalize=config.normalize,
            )

        self.trainer = Trainer(
            state=self.state,
            pbar=None,
            clusterer=clusterer,  # Pass initialized clusterer
            cluster_every=config.cluster_every,
            clustering=config.clustering,
            TRIM_ESS=TRIM_ESS,
            TRIM_BINS=TRIM_BINS,
            DOF_FALLBACK=DOF_FALLBACK,  # Standardized value
        )

        self.resampler = Resampler(
            state=self.state,
            n_active_fn=lambda: self.reweighter.n_active,
            resample=config.resample,
            clusterer=clusterer,  # Pass same clusterer instance
            clustering=config.clustering,
            have_blobs=config.blobs_dtype is not None,
        )

        self.trainer = Trainer(
            state=self.state,
            pbar=None,
            clusterer=clusterer,  # Pass initialized clusterer
            cluster_every=config.cluster_every,
            clustering=config.clustering,
            TRIM_ESS=TRIM_ESS,
            TRIM_BINS=TRIM_BINS,
            DOF_FALLBACK=DOF_FALLBACK,  # Standardized value
        )

        self.resampler = Resampler(
            state=self.state,
            n_active_fn=lambda: self.reweighter.n_active,
            resample=config.resample,
            clusterer=clusterer,  # Pass same clusterer instance
            clustering=config.clustering,
            have_blobs=config.blobs_dtype is not None,
        )

        self.mutator = Mutator(
            state=self.state,
            prior_transform=config.prior_transform,
            log_likelihood=self._log_like,
            pbar=None,
            n_active_fn=lambda: self.reweighter.n_active,
            n_dim=config.n_dim,
            n_steps=config.n_steps,
            n_max_steps=config.n_max_steps,
            sampler=config.sample,
            periodic=config.periodic,
            reflective=config.reflective,
            have_blobs=config.blobs_dtype is not None,
        )

        # Progress bar (initialized in run_sampling)
        self.pbar = None
        self.t0 = 0

    def run_sampling(
        self,
        n_total: int = 4096,
        progress: bool = True,
        resume_state_path: Optional[Union[str, Path]] = None,
        save_every: Optional[int] = None,
    ) -> None:
        """Execute full sampling run (replaces Sampler.run logic)."""
        if resume_state_path is not None:
            self._initialize_from_resume(resume_state_path)
            # Get iteration count, default to 0 if not found (backward compatibility)
            iter_val = self.state.get_current("iter")
            t0 = int(iter_val) if iter_val is not None else 0
            # Also explicitly set iter in state if missing
            if iter_val is None:
                self.state.set_current("iter", t0)
        else:
            t0 = 0
            self._initialize_fresh()

        self.n_total = int(n_total)
        self.t0 = t0

        # Initialize progress bar for this run
        from .tools import ProgressBar

        self.pbar = ProgressBar(progress, initial=t0)
        self._update_progress_bar_initial()

        # Assign pbar to components
        self.reweighter.pbar = self.pbar
        self.trainer.pbar = self.pbar
        self.mutator.pbar = self.pbar

        # Run PS loop (adaptive warmup and annealing)
        while self._not_termination():
            self.execute_iteration(save_every=save_every, t0=t0)

        # Compute final evidence
        _, logz = self.state.compute_logw_and_logz(1.0)
        self.state.set_current("logz", logz)
        self.logz_err = None

        # Save final state
        if save_every is not None:
            self.save_sampler_state(
                self.config.output_dir / f"{self.config.output_label}_final.state"
            )

        # Close progress bar
        self.pbar.close()

    def execute_iteration(self, save_every: Optional[int], t0: int) -> dict:
        """Execute one iteration (replaces Sampler.sample)."""
        # Save state if requested
        if save_every is not None:
            iter_val = self.state.get_current("iter")
            if (iter_val - t0) % int(save_every) == 0 and iter_val != t0:
                self.save_sampler_state(
                    self.config.output_dir
                    / f"{self.config.output_label}_{iter_val}.state"
                )

        # Execute pipeline: reweight → train → resample → mutate
        weights = self.reweighter.run()
        mode_stats = self.trainer.run(weights)
        self.resampler.run(weights)
        self.mutator.run(mode_stats)

        # Update progress bar
        self._update_progress_bar()

        # Save particles to history
        self.state.commit_current_to_history()

        return self.state.get_current()

    def compute_posterior(
        self,
        resample=False,
        return_blobs=False,
        trim_importance_weights=True,
        return_logw=False,
        ess_trim=0.99,
        bins_trim=1000,
    ):
        """Compute posterior (replaces Sampler.posterior - 80 lines)."""
        logw, logz = self.state.compute_logw_and_logz(1.0)
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)

        u = self.state.get_history("u", flat=True)
        x = self.state.get_history("x", flat=True)
        logl = self.state.get_history("logl", flat=True)

        if self.config.blobs_dtype is not None:
            blobs = self.state.get_history("blobs", flat=True)
        else:
            blobs = None

        if trim_importance_weights:
            from .tools import trim_weights

            idx, weights = trim_weights(
                np.arange(len(weights)), weights, ess=ess_trim, bins=bins_trim
            )
            u = u[idx]
            x = x[idx]
            logl = logl[idx]
            if blobs is not None:
                blobs = blobs[idx]

        if resample:
            from .tools import systematic_resample

            idx = systematic_resample(len(weights), weights)
            u = u[idx]
            x = x[idx]
            logl = logl[idx]
            if blobs is not None:
                blobs = blobs[idx]
            weights = np.ones(len(idx)) / len(idx)

        if return_blobs and blobs is not None:
            if return_logw:
                return x, weights, logl, blobs, logw
            else:
                return x, weights, logl, blobs
        else:
            if return_logw:
                return x, weights, logl, logw
            else:
                return x, weights, logl

    def compute_evidence(self):
        """Compute logZ (replaces Sampler.evidence - 5 lines)."""
        logz = self.state.get_current("logz")
        return logz, getattr(self, "logz_err", None)

    def save_sampler_state(self, path: Union[str, Path]):
        """Save state (replaces Sampler.save_state - 41 lines)."""
        import dill

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get state dict
        d = self.state.to_dict()

        # Add sampler metadata
        d["random_state"] = self.config.random_state
        d["n_total"] = getattr(self, "n_total", None)
        d["logz_err"] = getattr(self, "logz_err", None)

        try:
            # Remove pool-related attributes that can't be pickled
            if hasattr(self.config, "pool") and self.config.pool is not None:
                pool_state = self.config.pool
                self.config.pool = None
                d["sampler"] = dill.dumps(self)
                self.config.pool = pool_state
            else:
                d["sampler"] = dill.dumps(self)
        except Exception as e:
            print(f"Error while saving state: {e}")
            raise

        # Save to file
        with open(path, "wb") as f:
            dill.dump(d, f)

    def load_sampler_state(self, path: Union[str, Path]):
        """Load state (replaces Sampler.load_state - 28 lines)."""
        import dill

        with open(Path(path), "rb") as f:
            d = dill.load(f)

        # Restore state manager
        self.state.from_dict(d)

        # Ensure all required keys exist with valid types (backward compatibility)
        # Some older state files may be missing certain keys
        required_keys = {
            "iter": 0,
            "calls": 0,
            "beta": 0.0,
            "logz": 0.0,
            "steps": 0,
            "acceptance": 0.0,
            "efficiency": 0.0,
        }
        for key, default_val in required_keys.items():
            if self.state.get_current(key) is None:
                self.state.set_current(key, default_val)

        # Restore sampler attributes
        if "n_total" in d:
            self.n_total = d["n_total"]
        if "logz_err" in d:
            self.logz_err = d["logz_err"]

        # Set random seed
        if "random_state" in d and d["random_state"] is not None:
            np.random.seed(d["random_state"])

    def _log_like(self, x):
        """Compute log likelihood (replaces Sampler._log_like - 54 lines)."""
        import numpy as np

        if self.config.vectorize:
            return self.config.log_likelihood(x), None
        elif self.config.pool is not None:
            results = list(self._get_distribute_func()(self.config.log_likelihood, x))
        else:
            results = list(map(self.config.log_likelihood, x))

        # Check if results have blobs by testing the first result
        if results and isinstance(results[0], (tuple, list)) and len(results[0]) > 1:
            # Results have blobs
            blob = [l[1:] for l in results]
            logl = np.array([float(l[0]) for l in results])

            # Get the blobs dtype
            if self.config.blobs_dtype is not None:
                dt = self.config.blobs_dtype
            else:
                try:
                    dt = np.atleast_1d(blob[0]).dtype
                except ValueError:
                    dt = np.dtype("object")
                if dt.kind in "US":
                    # Strings need to be object arrays or we risk truncation
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

            return logl, blob
        else:
            # No blobs - single values
            logl = np.array([float(l) for l in results])
            return logl, None

    def _not_termination(self):
        """Check termination (replaces Sampler._not_termination - 27 lines)."""
        from .tools import effective_sample_size, unique_sample_size

        logw, _ = self.state.compute_logw_and_logz(1.0)

        # If no particles yet (first iteration), continue
        if len(logw) == 0:
            return True

        weights = np.exp(logw - np.max(logw))
        if self.config.metric == "ess":
            ess = effective_sample_size(weights)
        elif self.config.metric == "uss":
            ess = unique_sample_size(weights)

        beta = self.state.get_current("beta")
        return 1.0 - beta >= 1e-4 or ess < getattr(self, "n_total", 0)

    def _initialize_fresh(self):
        """Initialize fresh run (replaces part of Sampler.run)."""
        self.state.set_current("iter", 0)
        self.state.set_current("calls", 0)
        self.state.set_current("beta", 0.0)
        self.state.set_current("logz", 0.0)

    def _initialize_from_resume(self, resume_state_path):
        """Initialize from resume (replaces part of Sampler.run)."""
        self.load_sampler_state(resume_state_path)
        t0 = (
            int(self.state.get_current("iter"))
            if self.state.get_current("iter") is not None
            else 0
        )
        self.t0 = t0

    def _update_progress_bar_initial(self):
        """Update progress bar for fresh start."""
        if self.pbar is not None:
            self.pbar.update_stats(
                dict(
                    beta=0.0,
                    calls=0,
                    ESS=self.config.n_effective,
                    logZ=0.0,
                    logL=0.0,
                    acc=0.0,
                    steps=0,
                    eff=0.0,
                    K=1,
                )
            )

    def _update_progress_bar(self):
        """Update progress bar after iteration."""
        if self.pbar is not None:
            current = self.state.get_current()
            self.pbar.update_stats(
                dict(
                    calls=current["calls"],
                    beta=current["beta"],
                    ESS=int(current["ess"]),
                    logZ=current["logz"],
                    logL=np.mean(current["logl"])
                    if current["logl"] is not None
                    else 0.0,
                    acc=current["acceptance"],
                    steps=current["steps"],
                    eff=current["efficiency"],
                )
            )

    def _get_distribute_func(self):
        """Get distribution function (map or pool.map)."""
        if self.config.pool is None:
            return map
        elif isinstance(self.config.pool, int) and self.config.pool > 1:
            from multiprocess import Pool

            pool = Pool(self.config.pool)
            return pool.map
        else:
            return self.config.pool.map


if __name__ == "__main__":
    pass
