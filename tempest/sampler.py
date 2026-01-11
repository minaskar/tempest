from pathlib import Path
from typing import Union, Optional

import os
import dill
import numpy as np
from multiprocess import Pool

from .mcmc import parallel_mcmc
from .tools import (
    systematic_resample,
    FunctionWrapper,
    trim_weights,
    ProgressBar,
    effective_sample_size,
    unique_sample_size,
)
from .particles import Particles
from .cluster import HierarchicalGaussianMixture
from .student import fit_mvstud


class Sampler:
    r"""Persistent Sampling class.

    Parameters
    ----------
    prior_transform : callable
        Class implementing the prior distribution.
    log_likelihood : callable
        Function returning the log likelihood of a set of parameters.
    n_dim : int
        The total number of parameters/dimensions (Optional as it can be infered from the prior class).
    n_effective : int
        The number of effective particles (default is ``n_effective=512``). Higher values
        lead to more accurate results but also increase the computational cost.  This should be
        set to a value that is large enough to ensure that the target distribution is well
        represented by the particles. The number of effective particles should be greater than
        the number of active particles. If ``n_effective=None``, the default value is ``n_effective=2*n_active``.
    n_active : int
        The number of active particles (default is ``n_active=256``). It must be smaller than ``n_effective``.
        For best results, the number of active particles should be no more than half the number of effective particles.
        This is the number of particles that are evolved using MCMC at each iteration. If a pool is provided,
        the number of active particles should be a multiple of the number of processes in the pool to ensure
        efficient parallelisation. If ``n_active=None``, the default value is ``n_active=n_effective//2``.
    n_boost : int or ``None``
        Target number of effective particles to boost towards as the sampler approaches the posterior
        (default is ``n_boost=None``). When provided and greater than ``n_effective``, the number of effective
        and active particles will gradually increase from their initial values to ``n_boost`` based on the posterior
        ESS, improving sampling efficiency in the later stages. If ``n_boost == n_effective``, no boosting occurs.
    log_likelihood_args : list
        Extra arguments to be passed to log_likelihood (default is ``log_likelihood_args=None``). Example:
        ``log_likelihood_args=[data]``.
    log_likelihood_kwargs : dict
        Extra arguments to be passed to log_likelihood (default is ``log_likelihood_kwargs=None``). Example:
        ``log_likelihood_kwargs={"data": data}``.
    vectorize : bool
        If True, vectorize ``likelihood`` calculation (default is ``vectorize=False``). If False,
        the likelihood is calculated for each particle individually. If ``vectorize=True``, the likelihood
        is calculated for all particles simultaneously. This can lead to a significant speed-up if the likelihood
        function is computationally expensive. However, it requires that the likelihood function can handle
        arrays of shape ``(n_active, n_dim)`` as input and return an array of shape ``(n_active,)`` as output.
    blobs_dtype : list
        Data type of the blobs returned by the likelihood function (default is ``blobs_dtype=None``). If ``blobs_dtype``
        is not provided, the data type is inferred from the blobs returned by the likelihood function. If the blobs
        are not of the same data type, they are converted to an object array. If the blobs are strings, the data type
        is set to ``object``. If the blobs ``dtype`` is known in advance, it can be provided as a list of data types
        (e.g., ``blobs_dtype=[("blob_1", float), ("blob_2", int)]``). Blobs can be used to store additional data
        returned by the likelihood function (e.g., chi-squared values, residuals, etc.). Blobs are stored as a
        structured array with named fields when the data type is provided. Currently, the blobs feature is not
        compatible with vectorized likelihood calculations.
    periodic : list or ``None``
        List of parameter indeces that should be wrapped around the domain (default is ``periodic=None``).
        This can be useful for phase parameters that might be periodic e.g. on a range ``[0,2*np.pi]``. For example,
        ``periodic=[0,1]`` will wrap around the first and second parameters.
    reflective : list or ``None``
        List of parameter indeces that should be reflected around the domain (default is ``reflective=None``).
        This can arise in cases where parameters are ratios where ``a/b`` and  ``b/a`` are equivalent. For example,
        ``reflective=[0,1]`` will reflect the first and second parameters.
    pool : pool or int
        Number of processes to use for parallelisation (default is ``pool=None``). If ``pool`` is an integer
        greater than 1, a ``multiprocessing`` pool is created with the specified number of processes (e.g., ``pool=8``).
        If ``pool`` is an instance of ``mpi4py.futures.MPIPoolExecutor``, the code runs in parallel using MPI.
        If a pool is provided, the number of active particles should be a multiple of the number of processes in
        the pool to ensure efficient parallelisation. If ``pool=None``, the code runs in serial mode. When a pool
        is provided, please ensure that the likelihood function is picklable.
    train_frequency : int or None
        Frequency of training the normalizing flow (default is ``train_frequency=None``).
        If ``train_frequency=None``, the normalizing flow is trained every ``n_effective//n_active``
        iterations. If ``train_frequency=1``, the normalizing flow is trained at every iteration.
        If ``train_frequency>1``, the normalizing flow is trained every ``train_frequency`` iterations.
    dynamic : bool
        If True, dynamically adjust the effective sample size (ESS) threshold based on the
        number of unique particles (default is ``dynamic=True``). This can be useful for
        targets with a large number of modes or strong non-linear correlations between parameters.
    clustering : bool
        If True, enable hierarchical Gaussian mixture clustering of particles (default is
        ``clustering=True``).
    normalize : bool
        If True, normalize input data to [0,1]^D before clustering (default is ``normalize=True``).
        This can improve clustering performance when parameters have very different scales.
    split_threshold : float
        Multiplicative factor applied to the automatically computed BIC threshold when considering
        cluster splits (default is ``split_threshold=1.0``). Values larger than one make splits harder;
        values between zero and one make them easier.
    metric : str
        Metric used for determining the next temperature (``beta``) level (default is ``metric="ess"``).
        Options are ``"ess"`` (Effective Sample Size) or ``"uss"`` (Unique Sample Size). The metric
        is used to determine the next temperature level based on the ESS or USS of the importance
        weights. If the ESS or USS of the importance weights is below the target threshold, the temperature
        is increased. If the ESS or USS is above the target threshold, the temperature is decreased. The
        target threshold is set by the ``n_effective`` parameter.
    n_prior : int
        Number of prior samples to draw (default is ``n_prior=2*(n_effective//n_active)*n_active``). This
        is used to initialise the particles at the beginning of the run. The prior samples are used to
        warm-up the sampler and ensure that the particles are well distributed across the prior volume.
    sample : ``str``
        Type of MCMC sampler to use (default is ``sample="tpcn"``). Options are
        ``"pcn"`` (t-preconditioned Crank-Nicolson) or ``"rwm"`` (Random-walk Metropolis).
        t-preconditioned Crank-Nicolson is the default and recommended sampler for PS as it
        is more efficient and scales better with the number of parameters.
    n_steps : int
        Number of MCMC steps after logP plateau (default is ``n_steps=n_dim``). This is used
        for early stopping of MCMC. Higher values can lead to better exploration but also
        increase the computational cost. If ``n_steps=None``, the default value is ``n_steps=n_dim``.
    n_max_steps : int
        Maximum number of MCMC steps (default is ``n_max_steps=10*n_dim``).
    resample : ``str``
        Resampling scheme to use (default is ``resample="mult"``). Options are
        ``"syst"`` (systematic resampling) or ``"mult"`` (multinomial resampling).
    output_dir : ``str`` or ``None``
        Output directory for storing the state files of the
        sampler. Default is ``None`` which creates a ``states``
        directory. Output files can be used to resume a run.
    output_label : ``str`` or ``None``
        Label used in state files. Defaullt is ``None`` which
        corresponds to ``"ps"``. The saved states are named
        as ``"{output_dir}/{output_label}_{i}.state"`` where
        ``i`` is the iteration index.  Output files can be
        used to resume a run.
    random_state : int or ``None``
        Initial random seed.
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
        dynamic: bool = True,
        pool: Optional[Union[int, Pool]] = None,
        clustering: bool = True,
        normalize: bool = True,
        cluster_every: int = 1,
        split_threshold: float = 1.0,
        n_max_clusters: Optional[int] = None,
        metric: str = "ess",
        n_prior: Optional[int] = None,
        sample: str = "tpcn",
        n_steps: Optional[int] = None,
        n_max_steps: Optional[int] = None,
        resample: str = "mult",
        output_dir: Optional[str] = None,
        output_label: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        # Constants
        self.BETA_TOLERANCE = 1e-4  # Tolerance for beta termination check
        self.ESS_TOLERANCE = 0.01  # Relative tolerance for ESS in beta bisection
        self.DYNAMIC_RATIO_LOWER = 0.95  # Lower threshold for dynamic ESS adjustment
        self.DYNAMIC_RATIO_UPPER = 1.05  # Upper threshold for dynamic ESS adjustment
        self.DOF_FALLBACK = 1e6  # Fallback value for non-finite degrees of freedom
        self.TRIM_ESS = 0.99  # ESS threshold for trimming importance weights
        self.TRIM_BINS = 1000  # Number of bins for weight trimming
        self.BOOST_STEEPNESS = 0.125  # Steepness of the transition for n_active boost

        # Random seed
        if random_state is not None:
            np.random.seed(random_state)
        self.random_state = random_state

        # Prior distribution
        self.prior_transform = prior_transform

        # Log likelihood function
        self.log_likelihood = FunctionWrapper(
            log_likelihood, log_likelihood_args, log_likelihood_kwargs
        )

        # Blobs data type
        self.blobs_dtype = blobs_dtype
        self.have_blobs = blobs_dtype is not None

        # Number of parameters
        self.n_dim = int(n_dim)

        # Periodic and reflective boundary conditions
        self.periodic = np.array(periodic) if periodic is not None else None
        self.reflective = np.array(reflective) if reflective is not None else None

        # Check that at least one parameter is provided
        if n_active is None and n_effective is None:
            raise ValueError(
                "At least one of n_active or n_effective must be provided."
            )

        # Number of active particles
        if n_active is None:
            self.n_active = int(n_effective / 2)
        else:
            self.n_active = int(n_active)
        self.n_active_init = self.n_active

        # Effective Sample Size
        if n_effective is None:
            self.n_effective = int(2 * n_active)
        else:
            self.n_effective = int(n_effective)
        self.n_effective_init = self.n_effective

        # Boost factor
        if n_boost is None:
            self.n_boost = None
        else:
            self.n_boost = int(n_boost)
            if self.n_boost < self.n_effective:
                raise ValueError(
                    f"n_boost ({self.n_boost}) must be >= n_effective ({self.n_effective})"
                )

        # Number of MCMC steps after logP plateau
        if n_steps is None:
            self.n_steps = int(self.n_dim // 2)
        else:
            self.n_steps = int(n_steps)

        # Maximum number of MCMC steps
        if n_max_steps is None:
            self.n_max_steps = 10 * self.n_steps
        else:
            self.n_max_steps = int(n_max_steps)

        # Total ESS for termination
        self.n_total = None

        # Particle manager
        self.particles = Particles(n_active, n_dim)

        # Parallelism
        self.pool = pool
        if pool is None:
            self.distribute = map
        elif isinstance(pool, int) and pool > 1:
            self.pool = Pool(pool)
            self.distribute = self.pool.map
        else:
            self.distribute = pool.map

        # Likelihood vectorization
        self.vectorize = vectorize
        if self.vectorize and self.have_blobs:
            raise ValueError("Cannot vectorize likelihood with blobs.")

        # Output
        if output_dir is None:
            self.output_dir = Path("states")
        else:
            self.output_dir = Path(output_dir)
        if output_label is None:
            self.output_label = "ps"
        else:
            self.output_label = output_label

        # Effective vs Unique Sample Size
        if metric not in ["ess", "uss"]:
            raise ValueError(f"Invalid metric {metric}. Options are 'ess' or 'uss'.")
        else:
            self.metric = metric

        # Dynamic ESS
        self.dynamic = dynamic
        self.dynamic_ratio = (
            unique_sample_size(np.ones(self.n_effective), k=self.n_active)
            / self.n_active
        )

        # Sampling algorithm
        if sample not in ["tpcn", "rwm"]:
            raise ValueError(f"Invalid sample {sample}. Options are 'tpcn' or 'rwm'.")
        else:
            self.sampler = sample

        # Clusterer
        self.clustering = clustering
        self.normalize = normalize
        if self.clustering:
            self.clusterer = HierarchicalGaussianMixture(
                n_init=1,
                max_iterations=n_max_clusters - 1,
                min_points=None if n_max_clusters is None else 4 * self.n_dim,
                threshold_modifier=split_threshold,
                covariance_type="full",
                normalize=normalize,
                verbose=False,
            )
        else:
            self.clusterer = None
        self.cluster_every = cluster_every
        self.cluster_every_init = cluster_every
        self.K = 1

        # Resampling algorithm
        if resample not in ["mult", "syst"]:
            raise ValueError(
                f"Invalid resample {resample}. Options are 'mult' or 'syst'."
            )
        else:
            self.resample = resample

        # Prior samples to draw
        if n_prior is None:
            self.n_prior = int(
                2 * np.maximum(self.n_effective // self.n_active, 1) * self.n_active
            )
        else:
            self.n_prior = int(np.maximum(n_prior / self.n_active, 1) * self.n_active)
        self.prior_samples = None

        self.n_warmup_iters = None  # Set in run()

        self.progress = None
        self.pbar = None

        # Particle Ensemble State
        self.u = None
        self.x = None
        self.logl = None
        self.assignments = None
        self.weights = None
        self.blobs = None
        self.acceptance = None
        self.steps = None
        self.efficiency = None
        self.ess = None
        self.beta = None
        self.logz = None
        self.calls = None
        self.iter = None

    def run(
        self,
        n_total: int = 4096,
        progress: bool = True,
        resume_state_path: Union[str, Path] = None,
        save_every: int = None,
    ):
        r"""Run Persistent Sampling.

        Parameters
        ----------
        n_total : int
            The total number of effectively independent samples to be
            collected (default is ``n_total=2048``).
        n_evidence : int
            The number of importance samples used to estimate the
            evidence (default is ``n_evidence=4096``). If ``n_evidence=0``,
            the evidence is not estimated using importance sampling and the
            SMC estimate is used instead. If ``preconditioned=False``,
            the evidence is estimated using SMC and ``n_evidence`` is ignored.
        progress : bool
            If True, print progress bar (default is ``progress=True``).
        resume_state_path : ``Union[str, Path]``
            Path of state file used to resume a run. Default is ``None`` in which case
            the sampler does not load any previously saved states. An example of using
            this option to resume or continue a run is e.g. ``resume_state_path = "states/ps_1.state"``.
        save_every : ``int`` or ``None``
            Argument which determines how often (i.e. every how many iterations) ``Tempest`` saves
            state files to the ``output_dir`` directory. Default is ``None`` in which case no state
            files are stored during the run.
        """
        if resume_state_path is not None:
            self.load_state(resume_state_path)
            t0 = self.iter
            # Initialise progress bar
            self.pbar = ProgressBar(self.progress, initial=t0)
            self.pbar.update_stats(
                dict(
                    beta=self.particles.get("beta", -1),
                    calls=self.particles.get("calls", -1),
                    ESS=self.particles.get("ess", -1),
                    logZ=self.particles.get("logz", -1),
                    logL=np.mean(self.particles.get("logl", -1)),
                    acc=self.particles.get("accept", -1),
                    steps=self.particles.get("steps", -1),
                    eff=self.particles.get("efficiency", -1),
                    K=getattr(
                        self, "K", len(getattr(self, "means", np.atleast_2d([0])))
                    ),
                )
            )
        else:
            t0 = 0
            self.iter = 0
            self.calls = 0
            self.beta = 0.0  # Initialize beta for first iteration
            self.logz = 0.0
            # Run parameters
            self.progress = progress

            # Initialise progress bar
            self.pbar = ProgressBar(self.progress)
            self.pbar.update_stats(
                dict(
                    beta=0.0,
                    calls=self.calls,
                    ESS=self.n_effective,
                    logZ=0.0,
                    logL=0.0,
                    acc=0.0,
                    steps=0,
                    eff=0.0,
                    K=1,
                )
            )

        self.n_total = int(n_total)

        # Number of warmup iterations (prior sampling at beta=0)
        self.n_warmup_iters = self.n_prior // self.n_active

        # Run PS loop (includes warmup iterations at beta=0)
        while self._not_termination():
            self.sample(save_every=save_every, t0=t0)

        # Compute evidence
        _, self.logz = self.particles.compute_logw_and_logz(1.0)
        self.logz_err = None

        # Save final state
        if save_every is not None:
            self.save_state(self.output_dir / f"{self.output_label}_final.state")

        # Close progress bar
        self.pbar.close()

    def sample(self, save_every: int = None, t0: int = 0):
        r"""Perform a single iteration of the PS algorithm.

        This method performs one iteration of the Persistent Sampling
        algorithm, including reweighting, training, resampling, and mutation steps.
        It can be used to run the sampler step-by-step for more fine-grained control.

        Parameters
        ----------
        save_every : ``int`` or ``None``
            Argument which determines how often (i.e. every how many iterations) ``Tempest`` saves
            state files to the ``output_dir`` directory. Default is ``None`` in which case no state
            files are stored during the run.
        t0 : ``int``
            The starting iteration index, used for determining when to save states.
            Default is ``0``.

        Returns
        -------
        state : dict
            Dictionary containing the current state of the particles with keys:
            - "u": Unit hypercube coordinates of particles
            - "x": Physical coordinates of particles
            - "logl": Log-likelihood values
            - "assignments": Cluster assignments
            - "blobs": Additional data from likelihood (if available)
            - "iter": Current iteration index
            - "calls": Total number of likelihood evaluations
            - "steps": Number of MCMC steps taken
            - "efficiency": MCMC efficiency
            - "ess": Effective sample size
            - "accept": Acceptance rate
            - "beta": Current inverse temperature
            - "logz": Current log-evidence estimate
        """
        # Save state if requested
        if save_every is not None:
            if (self.iter - t0) % int(save_every) == 0 and self.iter != t0:
                self.save_state(
                    self.output_dir / f"{self.output_label}_{self.iter}.state"
                )

        # Choose next beta based on ESS of weights
        self._reweight()

        # Train clustering
        self._train()

        # Resample particles
        self._resample()

        # Evolve particles using MCMC
        self._mutate()

        # Update progress bar
        self.pbar.update_stats(
            dict(
                calls=self.calls,
                beta=self.beta,
                ESS=int(self.ess),
                logZ=self.logz,
                logL=np.mean(self.logl),
                acc=self.acceptance,
                steps=self.steps,
                eff=self.efficiency,
            )
        )

        # Save particles
        self.particles.update(
            {
                "u": self.u,
                "x": self.x,
                "logl": self.logl,
                "assignments": self.assignments,
                "blobs": self.blobs,
                "iter": self.iter,
                "calls": self.calls,
                "steps": self.steps,
                "efficiency": self.efficiency,
                "ess": self.ess,
                "accept": self.acceptance,
                "beta": self.beta,
                "logz": self.logz,
            }
        )

        # Return current state
        return {
            "u": self.u.copy(),
            "x": self.x.copy(),
            "logl": self.logl.copy(),
            "assignments": self.assignments.copy(),
            "blobs": self.blobs.copy() if self.have_blobs else None,
            "iter": self.iter,
            "calls": self.calls,
            "steps": self.steps,
            "efficiency": self.efficiency,
            "ess": self.ess,
            "accept": self.acceptance,
            "beta": self.beta,
            "logz": self.logz,
        }

    def _not_termination(self):
        """
        Check if termination criterion is satisfied.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        termination : bool
            True if termination criterion is not satisfied.
        """
        log_weights, _ = self.particles.compute_logw_and_logz(1.0)

        # If no particles yet (first iteration), continue
        if len(log_weights) == 0:
            return True

        weights = np.exp(log_weights - np.max(log_weights))
        if self.metric == "ess":
            ess = effective_sample_size(weights)
        elif self.metric == "uss":
            ess = unique_sample_size(weights)

        return 1.0 - self.beta >= self.BETA_TOLERANCE or ess < self.n_total

    def _mutate(self):
        """
        Evolve particles using MCMC.

        At beta=0 (warmup phase), draws fresh samples from the prior instead
        of running MCMC, since prior samples are independent.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        # During warmup (beta=0), draw fresh prior samples instead of MCMC
        if self.beta == 0.0:
            self.u = np.random.rand(self.n_active, self.n_dim)
            self.x = np.array(
                [self.prior_transform(self.u[i]) for i in range(self.n_active)]
            )
            self.logl, self.blobs = self._log_like(self.x)
            self.assignments = np.zeros(self.n_active, dtype=int)
            self.calls += self.n_active
            self.steps = 1
            self.acceptance = 1.0
            self.efficiency = 1.0

            # Resample prior particles with infinite likelihoods
            inf_logl_mask = np.isinf(self.logl)
            if np.any(inf_logl_mask):
                all_idx = np.arange(len(self.x))
                infinite_idx = all_idx[inf_logl_mask]
                finite_idx = all_idx[~inf_logl_mask]
                if len(finite_idx) > 0:
                    idx = np.random.choice(
                        finite_idx, size=len(infinite_idx), replace=True
                    )
                    self.x[infinite_idx] = self.x[idx]
                    self.u[infinite_idx] = self.u[idx]
                    self.logl[infinite_idx] = self.logl[idx]
                    if self.have_blobs:
                        self.blobs[infinite_idx] = self.blobs[idx]

                # Correct logZ for fraction of prior with finite likelihood support
                n_finite = len(finite_idx)
                n_total = len(self.logl)
                self.logz += np.log(n_finite / n_total)
            return

        if self.have_blobs:
            blobs = self.blobs.copy()
        else:
            blobs = None

        (
            self.u,
            self.x,
            self.logl,
            blobs,
            self.efficiency,
            self.acceptance,
            self.steps,
            calls,
        ) = parallel_mcmc(
            u=self.u,
            x=self.x,
            logl=self.logl,
            blobs=blobs,
            assignments=self.assignments,
            beta=self.beta,
            means=self.means,
            covariances=self.covariances,
            degrees_of_freedom=self.degrees_of_freedom,
            log_likelihood=self._log_like,
            prior_transform=self.prior_transform,
            progress_bar=self.pbar,
            n_steps=self.n_steps,
            n_max=self.n_max_steps,
            sample=self.sampler,
            periodic=self.periodic,
            reflective=self.reflective,
            verbose=True,
        )

        if self.have_blobs:
            self.blobs = blobs.copy()
        self.calls += calls

    def _train(self):
        """
        Train normalizing flow.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        # Skip training during warmup (beta=0) - use simple isotropic proposal
        if self.beta == 0.0:
            self.means = np.array([[0.5] * self.n_dim])
            self.covariances = np.array([np.eye(self.n_dim) * 0.1])
            self.degrees_of_freedom = np.array([self.DOF_FALLBACK])
            self.K = 1
            return

        if self.clustering and (self.iter % self.cluster_every == 0 or self.iter == 0):
            # Fit clustering model
            self.clusterer.fit(self.u, self.weights)
            labels = self.clusterer.predict(self.u)
            means = []
            covariances = []
            degrees_of_freedom = []
            for label in range(np.unique(labels).shape[0]):
                idx = np.where(labels == label)[0]
                u_idx = self.u[idx]
                weights_idx = self.weights[idx]
                weights_idx /= np.sum(weights_idx)
                u_idx_resampled = u_idx[
                    np.random.choice(
                        np.arange(len(u_idx)),
                        size=len(u_idx) * 4,
                        replace=True,
                        p=weights_idx,
                    )
                ]
                mean, covariance, dof = fit_mvstud(u_idx_resampled)
                if ~np.isfinite(dof):
                    dof = self.DOF_FALLBACK
                means.append(mean)
                covariances.append(covariance)
                degrees_of_freedom.append(dof)

            self.means = np.array(means)
            self.covariances = np.array(covariances)
            self.degrees_of_freedom = np.array(degrees_of_freedom)
            self.K = self.means.shape[0]
        elif self.clustering and not (
            self.iter % self.cluster_every == 0 or self.iter == 0
        ):
            # Use previous clustering
            pass
        else:
            u_resampled = self.u[
                np.random.choice(
                    np.arange(len(self.weights)),
                    size=self.n_effective * 4,
                    replace=True,
                    p=self.weights,
                )
            ]
            mean, covariance, dof = fit_mvstud(u_resampled)
            self.means = mean.reshape(1, self.n_dim)
            self.covariances = covariance.reshape(1, self.n_dim, self.n_dim)
            if ~np.isfinite(dof):
                dof = self.DOF_FALLBACK
            self.degrees_of_freedom = np.array([dof])
            self.K = 1

        # Update progress bar with number of clusters (K)
        if self.pbar is not None:
            self.pbar.update_stats(dict(K=self.K))

    def _resample(self):
        """
        Resample particles.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        # Skip resampling during warmup (beta=0) - will draw fresh prior samples
        if self.beta == 0.0:
            self.assignments = np.zeros(self.n_active, dtype=int)
            return

        u = self.u
        x = self.x
        logl = self.logl
        weights = self.weights
        blobs = self.blobs

        if self.resample == "mult":
            idx_resampled = np.random.choice(
                np.arange(len(weights)), size=self.n_active, replace=True, p=weights
            )
        elif self.resample == "syst":
            idx_resampled = systematic_resample(self.n_active, weights=weights)

        self.u = u[idx_resampled]
        self.x = x[idx_resampled]
        self.logl = logl[idx_resampled]
        if self.have_blobs:
            self.blobs = blobs[idx_resampled]

        if self.clustering:
            self.assignments = self.clusterer.predict(self.u)
        else:
            self.assignments = np.zeros(self.n_active, dtype=int)

    def _reweight(self):
        """
        Reweight particles.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        # Update iteration index
        self.iter += 1
        self.pbar.update_iter()

        # During warmup phase, keep beta=0 to accumulate prior samples
        if self.iter <= self.n_warmup_iters:
            self.beta = 0.0
            self.logz = 0.0
            self.ess = self.n_effective
            # Uniform weights during warmup
            n_particles = (
                len(self.particles.get("logl", flat=True))
                if self.iter > 1
                else self.n_active
            )
            self.weights = np.ones(n_particles) / n_particles
            self.pbar.update_stats(
                dict(beta=self.beta, ESS=int(self.ess), logZ=self.logz)
            )
            return

        beta_prev = self.beta
        beta_max = 1.0
        beta_min = np.copy(beta_prev)

        def get_weights_and_ess(beta):
            logw, _ = self.particles.compute_logw_and_logz(beta)
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
            logz = self.logz
            ess_est = ess_est_prev
            self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est_prev), logZ=logz))
        elif ess_est_max >= self.n_effective:
            beta = beta_max
            weights = weights_max
            _, logz = self.particles.compute_logw_and_logz(beta)
            ess_est = ess_est_max
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
                    _, logz = self.particles.compute_logw_and_logz(beta)
                    self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est), logZ=logz))
                    break
                elif ess_est < self.n_effective:
                    beta_max = beta
                else:
                    beta_min = beta

        logw, _ = self.particles.compute_logw_and_logz(beta)
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

        idx, weights = trim_weights(
            np.arange(len(weights)), weights, ess=self.TRIM_ESS, bins=self.TRIM_BINS
        )
        self.u = self.particles.get("u", index=None, flat=True)[idx]
        self.x = self.particles.get("x", index=None, flat=True)[idx]
        self.logl = self.particles.get("logl", index=None, flat=True)[idx]
        if self.have_blobs:
            self.blobs = self.particles.get("blobs", index=None, flat=True)[idx]
        self.logz = logz
        self.beta = beta
        self.weights = weights
        self.ess = ess_est

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

    def _log_like(self, x):
        """
        Compute log likelihood.

        Parameters
        ----------
        x : array_like
            Array of parameter values.

        Returns
        -------
        logl : float
            Log likelihood.
        blob : array_like
            Additional data (default is ``None``).
        """
        if self.vectorize:
            return self.log_likelihood(x), None
        elif self.pool is not None:
            results = list(self.distribute(self.log_likelihood, x))
        else:
            results = list(map(self.log_likelihood, x))

        try:
            blob = [l[1:] for l in results if len(l) > 1]
            if not len(blob):
                raise IndexError
            logl = np.array([float(l[0]) for l in results])
            self.have_blobs = True
        except (IndexError, TypeError):
            logl = np.array([float(l) for l in results])
            blob = None
        else:
            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
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

    def evidence(self):
        """
        Return the log evidence estimate and error.
        """
        return self.logz, self.logz_err

    def __getstate__(self):
        """
        Get state information for pickling.
        """
        state = self.__dict__.copy()

        try:
            # deal with pool
            if state["pool"] is not None:
                del state["pool"]  # remove pool
                del state["distribute"]  # remove `pool.map` function hook
        except:  # TODO use specific exception type
            pass

        return state

    def posterior(
        self,
        resample=False,
        return_blobs=False,
        trim_importance_weights=True,
        return_logw=False,
        ess_trim=0.99,
        bins_trim=1_000,
    ):
        """
        Return posterior samples.

        Parameters
        ----------
        resample : bool
            If True, resample particles (default is ``resample=False``).
        trim_importance_weights : bool
            If True, trim importance weights (default is ``trim_importance_weights=True``).
        return_logw : bool
            If True, return log importance weights (default is ``return_logw=False``).
        ess_trim : float
            Effective sample size threshold for trimming (default is ``ess_trim=0.99``).
        bins_trim : int
            Number of bins for trimming (default is ``bins_trim=1_000``).

        Returns
        -------
        samples : ``np.ndarray``
            Samples from the posterior.
        weights : ``np.ndarray``
            Importance weights.
        logl : ``np.ndarray``
            Log likelihoods.
        logp : ``np.ndarray``
            Log priors.
        """
        if return_blobs and not self.have_blobs:
            raise ValueError("No blobs available.")

        samples = self.particles.get("x", flat=True)
        logl = self.particles.get("logl", flat=True)
        if return_blobs:
            blobs = self.particles.get("blobs", flat=True)
        logw, _ = self.particles.compute_logw_and_logz(1.0)
        weights = np.exp(logw)

        if trim_importance_weights:
            idx, weights = trim_weights(
                np.arange(len(samples)), weights, ess=ess_trim, bins=bins_trim
            )
            samples = samples[idx]
            logl = logl[idx]
            logw = logw[idx]
            if return_blobs:
                blobs = blobs[idx]

        if resample:
            if self.resample == "mult":
                idx_resampled = np.random.choice(
                    np.arange(len(weights)), size=len(samples), replace=True, p=weights
                )
            elif self.resample == "syst":
                idx_resampled = systematic_resample(len(weights), weights=weights)
            if return_blobs:
                return samples[idx_resampled], logl[idx_resampled], blobs[idx_resampled]
            else:
                return samples[idx_resampled], logl[idx_resampled]

        else:
            if return_logw:
                if return_blobs:
                    return samples, logw, logl, blobs
                else:
                    return samples, logw, logl
            else:
                if return_blobs:
                    return samples, weights, logl, blobs
                else:
                    return samples, weights, logl

    @property
    def results(self):
        """
        Return results.

        Returns
        -------
        results : dict
            Dictionary containing the results.
        """
        return self.particles.compute_results()

    def save_state(self, path: Union[str, Path]):
        """Save current state of sampler to file.

        Parameters
        ----------
        path : ``Union[str, Path]``
            Path to save state.
        """
        print(f"Saving PS state to {path}")
        Path(path).parent.mkdir(exist_ok=True)
        temp_path = Path(path).with_suffix(".temp")
        with open(temp_path, "wb") as f:
            state = self.__dict__.copy()
            del state["pbar"]  # Cannot be pickled
            try:
                # deal with pool
                if state["pool"] is not None:
                    del state["pool"]  # remove pool
                    del state["distribute"]  # remove `pool.map` function hook
            except BaseException as e:
                print(e)

            dill.dump(file=f, obj=state)
            f.flush()
            os.fsync(f.fileno())

        os.rename(temp_path, path)

    def load_state(self, path: Union[str, Path]):
        """Load state of sampler from file.

        Parameters
        ----------
        path : ``Union[str, Path]``
            Path from which to load state.
        """
        with open(path, "rb") as f:
            state = dill.load(file=f)
        self.__dict__ = {**self.__dict__, **state}
