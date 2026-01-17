from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, List

from .config import SamplerConfig
from ._core import SamplerCore
from .state_manager import StateManager
from .tools import FunctionWrapper


class Sampler:
    """
    Public API facade for Tempest sampler - maintains backward compatibility.

    All configuration and algorithm logic is delegated to internal components:
    - SamplerConfig: validates and stores configuration
    - SamplerCore: executes the sampling algorithm
    - StateManager: manages current and historical state
    """

    def __init__(
        self,
        prior_transform: callable,
        log_likelihood: callable,
        n_dim: int,
        n_effective: int = 512,
        n_active: int = 256,
        n_boost: int = None,
        log_likelihood_args: Optional[list] = None,
        log_likelihood_kwargs: Optional[dict] = None,
        vectorize: bool = False,
        blobs_dtype: Optional[str] = None,
        periodic: Optional[list] = None,
        reflective: Optional[list] = None,
        pool: Optional[Union[int, object]] = None,
        clustering: bool = True,
        normalize: bool = True,
        cluster_every: int = 1,
        split_threshold: float = 1.0,
        n_max_clusters: Optional[int] = None,
        metric: str = "ess",
        sample: str = "tpcn",
        n_steps: Optional[int] = None,
        n_max_steps: Optional[int] = None,
        resample: str = "mult",
        output_dir: Optional[str] = None,
        output_label: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Tempest sampler.

        Parameters are validated and stored in SamplerConfig, then delegated
        to SamplerCore for execution.
        """
        # Wrap likelihood function
        wrapped_likelihood = FunctionWrapper(
            log_likelihood, log_likelihood_args, log_likelihood_kwargs
        )

        # Create validated configuration
        config = SamplerConfig(
            prior_transform=prior_transform,
            log_likelihood=wrapped_likelihood,
            n_dim=n_dim,
            n_effective=n_effective,
            n_active=n_active,
            n_boost=n_boost,
            log_likelihood_args=log_likelihood_args,
            log_likelihood_kwargs=log_likelihood_kwargs,
            vectorize=vectorize,
            blobs_dtype=blobs_dtype,
            periodic=periodic,
            reflective=reflective,
            pool=pool,
            clustering=clustering,
            normalize=normalize,
            cluster_every=cluster_every,
            split_threshold=split_threshold,
            n_max_clusters=n_max_clusters,
            metric=metric,
            sample=sample,
            n_steps=n_steps,
            n_max_steps=n_max_steps,
            resample=resample,
            output_dir=output_dir,
            output_label=output_label,
            random_state=random_state,
        )

        # Create state manager
        state = StateManager(n_dim)

        # Create internal coordinator
        self._core = SamplerCore(config, state)

        # Expose state for backward compatibility (tests access sampler.state)
        self.state = state

    def run(
        self,
        n_total: int = 4096,
        progress: bool = True,
        resume_state_path: Union[str, Path, None] = None,
        save_every: Optional[int] = None,
    ):
        """
        Run Persistent Sampling.

        Parameters
        ----------
        n_total : int
            The total number of effectively independent samples to be
            collected (default is ``n_total=4096``).
        progress : bool
            If True, print progress bar (default is ``progress=True``).
        resume_state_path : str or Path or None
            Path of state file used to resume a run. Default is ``None`` in which case
            the sampler does not load any previously saved states.
        save_every : int or None
            Argument which determines how often (i.e. every how many iterations) ``Tempest`` saves
            state files to the ``output_dir`` directory. Default is ``None`` in which case no state
            files are stored during the run.
        """
        return self._core.run_sampling(
            n_total=n_total,
            progress=progress,
            resume_state_path=resume_state_path,
            save_every=save_every,
        )

    def sample(self, save_every: Optional[int] = None, t0: int = 0) -> dict:
        """
        Perform a single iteration of the PS algorithm.

        Parameters
        ----------
        save_every : int or None
            Argument which determines how often (i.e. every how many iterations) ``Tempest`` saves
            state files to the ``output_dir`` directory. Default is ``None`` in which case no state
            files are stored during the run.
        t0 : int
            The starting iteration index, used for determining when to save states.
            Default is ``0``.

        Returns
        -------
        state : dict
            Dictionary containing the current state of the particles.
        """
        return self._core.execute_iteration(save_every=save_every, t0=t0)

    def posterior(
        self,
        resample: bool = False,
        return_blobs: bool = False,
        trim_importance_weights: bool = True,
        return_logw: bool = False,
        ess_trim: float = 0.99,
        bins_trim: int = 1000,
    ) -> tuple:
        """
        Return posterior samples.

        Parameters
        ----------
        resample : bool
            If True, resample particles (default is ``resample=False``).
        return_blobs : bool
            If True, return auxiliary data from likelihood (default is ``return_blobs=False``).
        trim_importance_weights : bool
            If True, trim importance weights (default is ``trim_importance_weights=True``).
        return_logw : bool
            If True, return log importance weights (default is ``return_logw=False``).
        ess_trim : float
            Effective sample size threshold for trimming (default is ``ess_trim=0.99``).
        bins_trim : int
            Number of bins for trimming (default is ``bins_trim=1000``).

        Returns
        -------
        x : np.ndarray
            Physical coordinates of posterior samples.
        weights : np.ndarray
            Importance weights.
        logl : np.ndarray
            Log-likelihood values.
        blobs : np.ndarray (optional)
            Auxiliary data if return_blobs=True.
        logw : np.ndarray (optional)
            Log importance weights if return_logw=True.
        """
        return self._core.compute_posterior(
            resample=resample,
            return_blobs=return_blobs,
            trim_importance_weights=trim_importance_weights,
            return_logw=return_logw,
            ess_trim=ess_trim,
            bins_trim=bins_trim,
        )

    def evidence(self) -> tuple[float, Optional[float]]:
        """
        Return log evidence estimate and error.

        Returns
        -------
        logz : float
            Log evidence estimate.
        logz_err : float or None
            Error estimate (currently None, for future use).
        """
        return self._core.compute_evidence()

    def save_state(self, path: Union[str, Path]):
        """
        Save sampler state to file.

        Parameters
        ----------
        path : str or Path
            Path where state will be saved.
        """
        self._core.save_sampler_state(Path(path))

    def load_state(self, path: Union[str, Path]):
        """
        Load sampler state from file.

        Parameters
        ----------
        path : str or Path
            Path to state file.
        """
        self._core.load_sampler_state(Path(path))

    def __getstate__(self):
        """Get state for pickling (for backward compatibility)."""
        state = self.__dict__.copy()
        # Remove pool-related attributes that can't be pickled
        if "_core" in state and hasattr(state["_core"], "pool"):
            del state["_core"]
        return state

    def results(self):
        """Return results (backward compatibility)."""
        return self.state.compute_results()

    # Backward compatible property accessors
    @property
    def n_dim(self) -> int:
        """Number of dimensions."""
        return self._core.config.n_dim

    @property
    def n_effective(self) -> int:
        """Number of effective particles."""
        return self._core.config.n_effective

    @property
    def n_active(self) -> int:
        """Number of active particles."""
        return self._core.config.n_active

    @property
    def n_steps(self) -> int:
        """Number of MCMC steps."""
        return self._core.config.n_steps

    @property
    def n_max_steps(self) -> int:
        """Maximum number of MCMC steps."""
        return self._core.config.n_max_steps

    @property
    def n_boost(self) -> Optional[int]:
        """Boost target for particle count."""
        return self._core.config.n_boost

    @property
    def n_total(self) -> Optional[int]:
        """Total effective samples target."""
        return getattr(self._core, "n_total", None)

    @property
    def metric(self) -> str:
        """Metric used (ess or uss)."""
        return self._core.config.metric

    @property
    def resample(self) -> str:
        """Resampling method (mult or syst)."""
        return self._core.config.resample

    @property
    def clustering(self) -> bool:
        """Whether clustering is enabled."""
        return self._core.config.clustering

    @property
    def vectorize(self) -> bool:
        """Whether likelihood is vectorized."""
        return self._core.config.vectorize

    @property
    def output_dir(self) -> Path:
        """Output directory for state files."""
        return self._core.config.output_dir

    @property
    def output_label(self) -> str:
        """Label for output files."""
        return self._core.config.output_label

    @property
    def random_state(self) -> Optional[int]:
        """Random seed."""
        return self._core.config.random_state

    @property
    def periodic(self) -> Optional[list]:
        """Periodic boundary condition indices."""
        return self._core.config.periodic

    @property
    def reflective(self) -> Optional[list]:
        """Reflective boundary condition indices."""
        return self._core.config.reflective

    @property
    def beta(self) -> float:
        """Current inverse temperature."""
        return self.state.get_current("beta")

    @property
    def logz(self) -> float:
        """Current log evidence estimate."""
        return self.state.get_current("logz")

    @property
    def ess(self) -> float:
        """Current effective sample size."""
        return self.state.get_current("ess")

    @property
    def n_effective_init(self) -> int:
        """Initial number of effective particles."""
        return self._core.config.n_effective
