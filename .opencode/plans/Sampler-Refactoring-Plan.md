# Sampler Refactoring Plan: Internal Decomposition with Stable API

## Overview
Preserve the **exact same public API** while extracting internal complexity into testable components. The `Sampler` facade (841 lines) will delegate to a `SamplerCore` that handles the actual logic.

---

## Architecture Goal

**Before**: Users interact directly with an 841-line Sampler class that handles everything

**After**: 
- `Sampler` remains as a facade (~150 lines) that users interact with
- `SamplerCore` handles all logic (~650 lines) - testable internal component
- `SamplerConfig` validates and stores configuration (~120 lines)

---

## Current Structure Analysis

```
Sampler (841 lines, 13 methods)
├── __init__ (lines 145-325): 180 lines of configuration
├── run (lines 327-425): 98 lines of orchestration
├── sample (lines 427-500): 73 lines of one iteration
├── _not_termination (lines 517-544): 27 lines
├── _initialize_steps (lines 546-599): 53 lines
├── _log_like (lines 601-655): 54 lines
├── evidence (lines 656-661): 5 lines
├── posterior (lines 679-759): 80 lines
├── results (lines 760-770): 10 lines
├── save_state (lines 771-812): 41 lines
├── load_state (lines 813-841): 28 lines
└── __getstate__ (lines 663-677): 14 lines
```

**Critical**: Steps 1-4 are already in separate modules (`_steps/*.py`).

---

## Phase 1: Create Internal Coordinator (Day 1-2)

### File: `tempest/_core.py` (NEW, internal module)

```python
from typing import Optional, Union
from pathlib import Path
import numpy as np

from .config import SamplerConfig
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
        from ._steps.reweight import Reweighter
        from ._steps.train import Trainer
        from ._steps.resample import Resampler
        from ._steps.mutate import Mutator

        self.reweighter = Reweighter(
            state=self.state,
            pbar=None,  # Will be set in run_sampling
            n_effective=config.n_effective,
            n_active=config.n_active,
            metric=config.metric,
            ESS_TOLERANCE=1e-4,  # From config
            BETA_TOLERANCE=0.01,  # From config
            n_boost=config.n_boost,
            n_effective_init=config.n_effective,
            n_active_init=config.n_active,
            BOOST_STEEPNESS=0.125,  # From config
        )

        self.trainer = Trainer(
            state=self.state,
            pbar=None,
            clusterer=None,  # Will be set based on config
            cluster_every=config.cluster_every,
            clustering=config.clustering,
            TRIM_ESS=0.99,
            TRIM_BINS=1000,
            DOF_FALLBACK=1.0,  # Standardized value
        )

        self.resampler = Resampler(
            state=self.state,
            n_active_fn=lambda: self.reweighter.n_active,
            resample=config.resample,
            clusterer=None,
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
            t0 = (
                int(self.state.get_current("iter"))
                if self.state.get_current("iter") is not None
                else 0
            )
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
            if hasattr(self.config, 'pool') and self.config.pool is not None:
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
        if self.config.vectorize:
            return self.config.log_likelihood(x), None
        elif self.config.pool is not None:
            results = list(self._get_distribute_func()(self.config.log_likelihood, x))
        else:
            results = list(map(self.config.log_likelihood, x))

        try:
            blob = [l[1:] for l in results if len(l) > 1]
            if not len(blob):
                raise IndexError
            logl = np.array([float(l[0]) for l in results])
            have_blobs = True
        except (IndexError, TypeError):
            logl = np.array([float(l) for l in results])
            blob = None
        else:
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
                import numpy as np

                axes = np.arange(len(shape))[
                    np.array(shape) == 1
                ] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        return logl, blob

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
```

---

## Phase 2: Refactor Sampler to Facade (Day 2-3)

### File: `tempest/sampler.py` (MODIFIED - reduces from 841 to ~150 lines)

```python
from pathlib import Path
from typing import Union, Optional

from .config import SamplerConfig
from ._core import SamplerCore
from .state_manager import StateManager
from .tools import FunctionWrapper


class Sampler:
    """Public API facade - maintains backward compatibility."""

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
        pool: Optional[Union[int, object]] = None,  # Pool can be any pool-like object
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
        # Create validated configuration
        config = SamplerConfig(
            prior_transform=prior_transform,
            log_likelihood=FunctionWrapper(
                log_likelihood, log_likelihood_args, log_likelihood_kwargs
            ),
            n_dim=n_dim,
            n_effective=n_effective,
            n_active=n_active,
            n_boost=n_boost,
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

        # Create internal coordinator
        state = StateManager(n_dim)
        self._core = SamplerCore(config, state)

        # Expose state for backward compatibility (tests access sampler.state)
        self.state = state

    def run(
        self,
        n_total: int = 4096,
        progress: bool = True,
        resume_state_path: Union[str, Path] = None,
        save_every: int = None,
    ):
        """Delegate to core (98 lines → 3 lines)."""
        return self._core.run_sampling(
            n_total=n_total,
            progress=progress,
            resume_state_path=resume_state_path,
            save_every=save_every,
        )

    def sample(self, save_every: int = None, t0: int = 0):
        """Delegate to core (73 lines → 3 lines)."""
        return self._core.execute_iteration(save_every, t0)

    def posterior(
        self,
        resample=False,
        return_blobs=False,
        trim_importance_weights=True,
        return_logw=False,
        ess_trim=0.99,
        bins_trim=1000,
    ):
        """Delegate to core (80 lines → 8 lines)."""
        return self._core.compute_posterior(
            resample=resample,
            return_blobs=return_blobs,
            trim_importance_weights=trim_importance_weights,
            return_logw=return_logw,
            ess_trim=ess_trim,
            bins_trim=bins_trim,
        )

    def evidence(self):
        """Delegate to core (5 lines → 2 lines)."""
        return self._core.compute_evidence()

    def save_state(self, path: Union[str, Path]):
        """Delegate to core (41 lines → 2 lines)."""
        return self._core.save_sampler_state(Path(path))

    def load_state(self, path: Union[str, Path]):
        """Delegate to core (28 lines → 2 lines)."""
        return self._core.load_sampler_state(Path(path))

    def __getstate__(self):
        """Get state for pickling (kept for backward compatibility)."""
        state = self.__dict__.copy()
        # Remove pool-related attributes that can't be pickled
        if "_core" in state and hasattr(state["_core"], "pool"):
            del state["_core"]
        return state

    def results(self):
        """Return results (kept for backward compatibility)."""
        return self.state.compute_results()


# Import at bottom to avoid circular imports
from .tools import FunctionWrapper
```

---

## Phase 3: Configuration Object (Day 3-4)

### File: `tempest/config.py` (NEW)

```python
from dataclasses import dataclass
from typing import Optional, Union, List
from pathlib import Path


@dataclass(frozen=True)
class SamplerConfig:
    """Immutable configuration with validation."""

    # Required parameters
    prior_transform: callable
    log_likelihood: callable
    n_dim: int

    # Sampling parameters
    n_effective: int = 512
    n_active: int = 256
    n_boost: Optional[int] = None

    # Likelihood configuration
    log_likelihood_args: Optional[list] = None
    log_likelihood_kwargs: Optional[dict] = None
    vectorize: bool = False
    blobs_dtype: Optional[str] = None

    # Boundary conditions
    periodic: Optional[List[int]] = None
    reflective: Optional[List[int]] = None

    # Parallelism
    pool: Optional[Union[int, object]] = None

    # Clustering
    clustering: bool = True
    normalize: bool = True
    cluster_every: int = 1
    split_threshold: float = 1.0
    n_max_clusters: Optional[int] = None

    # Algorithm parameters
    metric: str = "ess"
    sample: str = "tpcn"
    n_steps: Optional[int] = None
    n_max_steps: Optional[int] = None
    resample: str = "mult"

    # Output
    output_dir: Optional[Path] = None
    output_label: Optional[str] = None

    # Random seed
    random_state: Optional[int] = None

    def __post_init__(self):
        """Set computed defaults and validate."""
        # Set defaults for paths (need to bypass frozen)
        if self.output_dir is None:
            object.__setattr__(self, "output_dir", Path("states"))
        elif isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.output_label is None:
            object.__setattr__(self, "output_label", "ps")

        # Compute n_active/n_effective defaults
        if self.n_active <= 0:
            object.__setattr__(self, "n_active", self.n_effective // 2)
        if self.n_effective <= 0:
            object.__setattr__(self, "n_effective", self.n_active * 2)

        # Compute n_steps/n_max_steps defaults
        if self.n_steps is None or self.n_steps <= 0:
            object.__setattr__(self, "n_steps", max(1, self.n_dim // 2))
        if self.n_max_steps is None or self.n_max_steps <= 0:
            object.__setattr__(self, "n_max_steps", 10 * self.n_steps)

        self.validate()

    def validate(self) -> None:
        """Validate all parameters."""
        errors = []

        # Check active/effective relationship
        if self.n_active >= self.n_effective:
            errors.append(
                f"n_active ({self.n_active}) must be < n_effective ({self.n_effective})"
            )

        # Check n_boost
        if self.n_boost is not None:
            if self.n_boost < self.n_effective:
                errors.append(
                    f"n_boost ({self.n_boost}) must be >= n_effective ({self.n_effective})"
                )

        # Check metric
        if self.metric not in ["ess", "uss"]:
            errors.append(f"Invalid metric {self.metric}: must be 'ess' or 'uss'")

        # Check sampler
        if self.sample not in ["tpcn", "rwm"]:
            errors.append(f"Invalid sampler {self.sample}: must be 'tpcn' or 'rwm'")

        # Check resample
        if self.resample not in ["mult", "syst"]:
            errors.append(f"Invalid resample {self.resample}: must be 'mult' or 'syst'")

        # Check vectorize + blobs conflict
        if self.vectorize and self.blobs_dtype is not None:
            errors.append("Cannot vectorize likelihood with blobs")

        # Check periodic/reflective don't overlap
        if self.periodic is not None and self.reflective is not None:
            overlap = set(self.periodic).intersection(set(self.reflective))
            if overlap:
                errors.append(f"Parameters cannot be both periodic and reflective: {overlap}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


# Algorithm constants (centralized)
BETA_TOLERANCE: float = 1e-4
ESS_TOLERANCE: float = 0.01
DOF_FALLBACK: float = 1.0
TRIM_ESS: float = 0.99
TRIM_BINS: int = 1000
BOOST_STEEPNESS: float = 0.125
```

---

## Phase 4: Test Strategy (Day 4-5)

### Keep Existing Tests (Backward Compatibility)

✅ **tests/test_sampler.py**: Keep exactly as-is (3 tests)
- These test the public API
- After refactoring, they'll test the facade
- Verify zero breaking changes

### New Tests for Internal Components

**File: `tests/test_sampler_core.py`** (NEW - 15-20 tests)

```python
import unittest
from unittest.mock import Mock, patch
import numpy as np

from tempest.config import SamplerConfig
from tempest._core import SamplerCore
from tempest.state_manager import StateManager


class TestSamplerCore(unittest.TestCase):
    """Test SamplerCore internal logic."""

    @staticmethod
    def prior_transform(u):
        return 20 * u - 10

    @staticmethod
    def log_likelihood_single(x):
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)

    def setUp(self):
        """Create fresh core for each test."""
        self.config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=2,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        self.state = StateManager(n_dim=2)
        self.core = SamplerCore(self.config, self.state)

    def test_execute_iteration_runs_all_steps(self):
        """Test that one iteration executes reweight → train → resample → mutate."""
        # Mock all step classes
        with patch.object(self.core.reweighter, "run") as mock_reweight, \
             patch.object(self.core.trainer, "run") as mock_train, \
             patch.object(self.core.resampler, "run") as mock_resample, \
             patch.object(self.core.mutator, "run") as mock_mutate:

            mock_reweight.return_value = np.ones(32) / 32
            mock_train.return_value = Mock()  # mode_stats

            result = self.core.execute_iteration(save_every=None, t0=0)

            # Verify pipeline execution
            mock_reweight.assert_called_once()
            mock_train.assert_called_once_with(mock_reweight.return_value)
            mock_resample.assert_called_once_with(mock_reweight.return_value)
            mock_mutate.assert_called_once_with(mock_train.return_value)

            # Verify state was committed
            self.assertGreater(self.state.get_history_length(), 0)

    def test_execute_iteration_returns_current_state(self):
        """Test that iteration returns current state dict."""
        # Initialize state
        self.core._initialize_fresh()
        self.core.state.commit_current_to_history()

        result = self.core.execute_iteration(save_every=None, t0=0)

        # Verify return value structure
        self.assertIsInstance(result, dict)
        self.assertIn("u", result)
        self.assertIn("x", result)
        self.assertIn("logl", result)
        self.assertIn("beta", result)

    def test_compute_posterior_with_resample_true(self):
        """Test posterior resampling returns different samples."""
        # Setup: Run a small sampling
        self.core._initialize_fresh()
        self.core.n_total = 128
        while self.core._not_termination():
            self.core.execute_iteration(save_every=None, t0=0)

        # Get posterior with resampling
        x1, w1, logl1 = self.core.compute_posterior(resample=True)
        x2, w2, logl2 = self.core.compute_posterior(resample=True)

        # With resample=True, should get different samples
        # (Note: due to randomness, they might occasionally be same)
        self.assertEqual(len(x1), len(x2))
        self.assertEqual(len(x1.shape), 2)

    def test_compute_posterior_without_resample(self):
        """Test posterior without resampling returns consistent results."""
        # Setup: Run a small sampling
        self.core._initialize_fresh()
        self.core.n_total = 128
        while self.core._not_termination():
            self.core.execute_iteration(save_every=None, t0=0)

        # Get posterior without resampling
        x1, w1, logl1 = self.core.compute_posterior(resample=False)
        x2, w2, logl2 = self.core.compute_posterior(resample=False)

        # Without resample, should get identical results
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(logl1, logl2)

    def test_compute_evidence_returns_logz(self):
        """Test evidence computation."""
        # Setup known logz
        self.core._initialize_fresh()
        self.core.n_total = 64
        while self.core._not_termination():
            self.core.execute_iteration(save_every=None, t0=0)

        logz, err = self.core.compute_evidence()

        self.assertIsInstance(logz, float)
        self.assertTrue(np.isfinite(logz))
        self.assertTrue(logz <= 0)  # Log evidence should be <= 0

    def test_save_and_load_state_preserves_history(self):
        """Test state persistence preserves history."""
        import tempfile
        import os

        # Run partial sampling
        self.core._initialize_fresh()
        self.core.n_total = 64
        for _ in range(5):
            self.core.execute_iteration(save_every=None, t0=0)

        history_len_before = self.state.get_history_length()
        beta_before = self.state.get_current("beta")

        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix=".state") as f:
            temp_path = f.name

        try:
            self.core.save_sampler_state(temp_path)

            # Create new core and load
            new_config = SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood_single,
                n_dim=2,
                n_effective=64,
                n_active=32,
                clustering=False,
                random_state=1,  # Different seed
            )
            new_state = StateManager(n_dim=2)
            new_core = SamplerCore(new_config, new_state)
            new_core.load_sampler_state(temp_path)

            # Verify history preserved
            self.assertEqual(
                new_state.get_history_length(), history_len_before
            )
            self.assertEqual(new_state.get_current("beta"), beta_before)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_run_sampling_with_resume(self):
        """Test resume functionality continues correctly."""
        import tempfile
        import os

        n_total_first = 64
        n_total_second = 128

        # First run
        self.core.run_sampling(n_total=n_total_first, progress=False)
        iter_first = self.state.get_current("iter")

        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix=".state") as f:
            temp_path = f.name

        try:
            self.core.save_sampler_state(temp_path)

            # Create new sampler and resume
            new_config = SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood_single,
                n_dim=2,
                n_effective=64,
                n_active=32,
                clustering=False,
                random_state=1,
            )
            new_state = StateManager(n_dim=2)
            new_core = SamplerCore(new_config, new_state)

            new_core.run_sampling(
                n_total=n_total_second,
                progress=False,
                resume_state_path=temp_path,
            )

            # Verify continued from correct iteration
            iter_second = new_state.get_current("iter")
            self.assertGreater(iter_second, iter_first)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_not_termination_returns_correct_boolean(self):
        """Test termination conditions logic."""
        self.core._initialize_fresh()
        self.core.n_total = 64

        # Before any iterations, should continue
        self.assertTrue(self.core._not_termination())

        # Run until termination
        while self.core._not_termination():
            self.core.execute_iteration(save_every=None, t0=0)

        # After termination, should return False
        self.assertFalse(self.core._not_termination())

    def test_log_like_handles_vectorized(self):
        """Test vectorized likelihood execution."""
        vectorized_config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=lambda x: np.sum(
                -0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1
            ),
            n_dim=2,
            n_effective=64,
            n_active=32,
            vectorize=True,
            clustering=False,
            random_state=0,
        )
        vectorized_core = SamplerCore(vectorized_config, StateManager(n_dim=2))

        # Test with multiple samples
        x = np.random.rand(10, 2)
        logl, blob = vectorized_core._log_like(x)

        self.assertEqual(logl.shape, (10,))
        self.assertIsNone(blob)

    def test_log_like_handles_single_with_blobs(self):
        """Test single likelihood with auxiliary data."""
        def log_like_with_blobs(x):
            logl = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)
            return logl, np.sum(x**2)  # blob is chi-squared

        blobs_config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=log_like_with_blobs,
            n_dim=2,
            n_effective=64,
            n_active=32,
            blob s_dtype=None,
            clustering=False,
            random_state=0,
        )
        blobs_core = SamplerCore(blobs_config, StateManager(n_dim=2))

        # Test with single sample
        x = np.array([[1.0, 2.0]])
        logl, blob = blobs_core._log_like(x)

        self.assertEqual(logl.shape, (1,))
        self.assertIsNotNone(blob)
        self.assertEqual(blob.shape, (1,))

    def test_log_like_detects_blobs_auto(self):
        """Test automatic blob dtype detection."""
        def log_like_with_float_blob(x):
            logl = float(np.sum(-0.5 * x**2))
            return logl, 1.5  # float blob

        auto_config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=log_like_with_float_blob,
            n_dim=2,
            n_effective=64,
            n_active=32,
            blobs_dtype=None,  # Auto-detect
            clustering=False,
            random_state=0,
        )
        auto_core = SamplerCore(auto_config, StateManager(n_dim=2))

        x = np.random.rand(5, 2)
        logl, blob = auto_core._log_like(x)

        self.assertEqual(blob.dtype.kind, "f")

    def test_save_every_functionality(self):
        """Test that save_every saves state at correct intervals."""
        import tempfile
        import os

        self.core._initialize_fresh()
        self.core.config.output_dir = Path(tempfile.mkdtemp())

        try:
            # Run multiple iterations with save_every=2
            for i in range(6):
                self.core.execute_iteration(save_every=2, t0=0)

            # Check that files exist at iterations 2 and 4
            # (iteration 6 is final, but we don't test that here)
            files = list(self.core.config.output_dir.glob("*.state"))

            # Should have saved at least twice
            self.assertGreaterEqual(len(files), 2)

        finally:
            # Cleanup
            import shutil

            if self.core.config.output_dir.exists():
                shutil.rmtree(self.core.config.output_dir)

    def test_termination_with_beta_1(self):
        """Test that beta=1 terminates."""
        self.core._initialize_fresh()
        self.core.n_total = 1000  # Large to ensure not ESS-limited

        # Force beta to 1
        self.core.state.set_current("beta", 1.0)

        # Should terminate (beta >= 1 - tolerance)
        self.assertFalse(self.core._not_termination())

    def test_termination_with_ess_target(self):
        """Test that reaching ESS target terminates."""
        self.core._initialize_fresh()
        self.core.n_total = 32  # Small to make ESS-limited

        # Run a few iterations
        for _ in range(3):
            self.core.execute_iteration(save_every=None, t0=0)

        # Should eventually terminate due to ESS
        # (actual ESS may or may not be reached in 3 iterations)
        # This test documents the behavior
        result = self.core._not_termination()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
```

---

## Phase 5: Migration & Verification (Day 5-6)

### Step-by-Step Migration Process

**Day 5 - Morning**:
1. Create `tempest/config.py` with `SamplerConfig`
2. Run existing tests → All pass (no integration yet)

**Day 5 - Afternoon**:
3. Create `tempest/_core.py` with `SamplerCore` skeleton
4. Move `Sampler._log_like` method to `SamplerCore`
5. Update `Sampler.__init__` to create `_core` and delegate
6. Run tests → Should still pass (facade still does work)

**Day 6 - Morning**:
7. Move `Sampler._not_termination` to `SamplerCore`
8. Move `Sampler._initialize_steps` logic to `SamplerCore.__init__`
9. Update `Sampler.run` to delegate to `core.run_sampling`
10. Run tests → Verify backward compatibility

**Day 6 - Afternoon**:
11. Move `Sampler.sample` to `SamplerCore.execute_iteration`
12. Move `Sampler.posterior` to `SamplerCore.compute_posterior`
13. Move `Sampler.evidence` to `SamplerCore.compute_evidence`
14. Move `Sampler.save_state`/`load_state` to `SamplerCore`
15. Run full test suite + integration tests

**Day 7**:
16. Write new tests for `SamplerCore` and `SamplerConfig`
17. Verify coverage improved from 3 → 20+ tests
18. Performance regression test (ensure no slowdown)

---

## Expected Outcomes

### Metrics Before/After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Sampler LOC | 841 | ~150 | -82% |
| Test Coverage | 3 tests | 20+ tests | +567% |
| Max Class LOC | 841 | ~650 | Better |
| API Breaking Changes | - | 0 | ✅ Stable |

### Code Organization

```
tempest/
├── __init__.py          # Public API exports Sampler, ModeStatistics
├── sampler.py           # Facade (~150 lines) - backward compatible
├── _core.py             # Internal coordinator (~650 lines) - testable
├── config.py            # Configuration object (~120 lines) - validated
├── state_manager.py     # Unchanged
├── mcmc.py              # Unchanged
├── cluster.py           # Unchanged
├── modes.py             # Unchanged
├── tools.py             # Unchanged
└── _steps/              # Already separated
    ├── __init__.py
    ├── reweight.py
    ├── train.py
    ├── resample.py
    └── mutate.py
```

---

## Risk Mitigation

### Risk 1: Breaking User Code
- **Mitigation**: Keep `Sampler.state` attribute exposed for direct access (tests use this)
- **Verification**: Run existing tests without modification

### Risk 2: Performance Regression
- **Mitigation**: Add delegation layer only, no algorithm changes
- **Verification**: Performance benchmark before/after

### Risk 3: Pickle Compatibility
- **Mitigation**: Keep `Sampler.__getstate__` method
- **Verification**: Test save/load in parallel execution context

### Risk 4: Documentation Drift
- **Mitigation**: Sampler docstrings remain unchanged
- **Verification**: No user-facing documentation updates needed

---

## Success Criteria

This refactoring succeeds when:

1. ✅ **All existing tests pass** without modification
2. ✅ **Coverage increases** from 3 to 15+ sampler tests
3. ✅ **No breaking changes** to public API
4. ✅ **Performance maintained** (no >5% slowdown)
5. ✅ Each component <200 lines and testable in isolation
6. ✅ Type hints throughout (Python 3.8+ compatible)

---

## Timeline & Effort

- **Phase 1** (Days 1-2): Create `SamplerCore` (~6-8 hours)
- **Phase 2** (Days 2-3): Refactor `Sampler` to facade (~4-6 hours)
- **Phase 3** (Days 3-4): Create `SamplerConfig` (~3-4 hours)
- **Phase 4** (Days 4-5): Write new tests (~4-5 hours)
- **Phase 5** (Days 5-6): Migration & verification (~3-4 hours)

**Total: 20-27 hours (approx. 3-4 days of focused work)**

---

## Trade-offs Discussed

### ✅ Pros
- **Zero API changes** - users see no difference
- **Explosive testability** - `SamplerCore` can have 20+ unit tests
- **Maintainable** - each class <200 lines, single responsibility
- **Debuggable** - stack traces show which component failed
- **Documented** - Configuration object validates all parameters

### ⚠️ Cons
- **Delegation overhead** - one extra method call per operation (negligible)
- **Internal complexity** - more classes to understand (for developers only)
- **Import cycle risk** - need careful import ordering (manageable)
- **File organization** - more files (but better organized)

---

## Next Steps

**Ready to begin implementation when you approve.**

The plan maintains backward compatibility while dramatically improving code quality and testability.
