from __future__ import annotations

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Any
from pathlib import Path


@dataclass(frozen=True)
class SamplerConfig:
    """Immutable configuration with validation for Sampler."""

    # Required parameters
    prior_transform: Callable[[np.ndarray], np.ndarray]
    log_likelihood: Callable[[np.ndarray], Union[np.ndarray, tuple]]
    n_dim: int

    # Sampling parameters
    n_particles: Optional[int] = None  # Default: 2 * n_dim
    ess_ratio: float = 2.0  # Target ESS ratio (ESS / n_particles)
    volume_variation: Optional[float] = (
        None  # Target coefficient of variation of volume. None disables dynamic mode.
    )

    # Likelihood configuration
    log_likelihood_args: Optional[list] = None
    log_likelihood_kwargs: Optional[dict] = None
    vectorize: bool = False
    blobs_dtype: Optional[str] = None

    # Boundary conditions
    periodic: Optional[List[int]] = None
    reflective: Optional[List[int]] = None

    # Parallelism
    pool: Optional[Union[int, Any]] = None

    # Clustering
    clustering: bool = True
    normalize: bool = True
    cluster_every: int = 1
    split_threshold: float = 1.0
    n_max_clusters: Optional[int] = None

    # Algorithm parameters
    sample: str = "tpcn"
    n_steps: Optional[int] = None
    n_max_steps: Optional[int] = None
    resample: str = "mult"

    # Output
    output_dir: Optional[Path] = None
    output_label: Optional[str] = None

    # Random seed
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Set computed defaults and validate."""
        # Basic type validation before computations
        if not isinstance(self.n_dim, int):
            raise ValueError(f"n_dim must be int, got {type(self.n_dim).__name__}")

        # Set defaults for paths (need to bypass frozen)
        if self.output_dir is None:
            object.__setattr__(self, "output_dir", Path("states"))
        elif isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.output_label is None:
            object.__setattr__(self, "output_label", "ps")

        # Set default n_particles if not provided
        if self.n_particles is None:
            object.__setattr__(self, "n_particles", 2 * self.n_dim)

        # Compute n_steps/n_max_steps defaults
        # n_steps now represents n_steps_0 (base steps per dimension at optimal acceptance rate of 23.4%)
        if self.n_steps is None or self.n_steps <= 0:
            object.__setattr__(self, "n_steps", 1)
        # n_max_steps now represents n_max_steps_0 (maximum steps per dimension)
        if self.n_max_steps is None or self.n_max_steps <= 0:
            object.__setattr__(self, "n_max_steps", 20 * self.n_steps)

        self.validate()

        # Warn if using dynamic mode with insufficient particles
        if self.volume_variation is not None and self.n_particles < self.n_dim + 1:
            warnings.warn(
                f"For dynamic mode, n_particles ({self.n_particles}) "
                f"should be >= n_dim + 1 ({self.n_dim + 1}) for reliable results. "
                f"Volume variation calculation may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )

    def validate(self) -> None:
        """Validate all parameters and raise ValueError if invalid."""
        errors = []

        # Check basic types
        if not callable(self.prior_transform):
            errors.append("prior_transform must be callable")
        if not callable(self.log_likelihood):
            errors.append("log_likelihood must be callable")
        if not isinstance(self.n_dim, int) or self.n_dim <= 0:
            errors.append(f"n_dim must be positive int, got {self.n_dim}")

        # Check n_particles
        if not isinstance(self.n_particles, int):
            errors.append(f"n_particles must be int, got {type(self.n_particles)}")
        if self.n_particles <= 0:
            errors.append(
                f"n_particles must be positive integer, got {self.n_particles}"
            )

        # Check ess_ratio
        if not isinstance(self.ess_ratio, (int, float)):
            errors.append(f"ess_ratio must be numeric, got {type(self.ess_ratio)}")
        if self.ess_ratio <= 0:
            errors.append(f"ess_ratio must be positive, got {self.ess_ratio}")

        # Check volume_variation
        if self.volume_variation is not None:
            if not isinstance(self.volume_variation, (int, float)):
                errors.append(
                    f"volume_variation must be numeric or None, got {type(self.volume_variation)}"
                )
            elif self.volume_variation <= 0:
                errors.append(
                    f"volume_variation ({self.volume_variation}) must be positive"
                )

        # Check sampler
        if self.sample not in ["tpcn", "rwm"]:
            errors.append(f"Invalid sampler '{self.sample}': must be 'tpcn' or 'rwm'")

        # Check resample
        if self.resample not in ["mult", "syst"]:
            errors.append(
                f"Invalid resample '{self.resample}': must be 'mult' or 'syst'"
            )

        # Check vectorize + blobs conflict
        if self.vectorize and self.blobs_dtype is not None:
            errors.append("Cannot vectorize likelihood with blobs")

        # Check periodic/reflective don't overlap
        if self.periodic is not None and self.reflective is not None:
            overlap = set(self.periodic).intersection(set(self.reflective))
            if overlap:
                errors.append(
                    f"Parameters cannot be both periodic and reflective: {overlap}"
                )

        # Check list parameters
        if self.periodic is not None:
            if not all(
                isinstance(i, int) and 0 <= i < self.n_dim for i in self.periodic
            ):
                errors.append(
                    f"periodic indices must be integers in [0, {self.n_dim - 1}], got {self.periodic}"
                )
        if self.reflective is not None:
            if not all(
                isinstance(i, int) and 0 <= i < self.n_dim for i in self.reflective
            ):
                errors.append(
                    f"reflective indices must be integers in [0, {self.n_dim - 1}], got {self.reflective}"
                )

        # Check paths
        if not isinstance(self.output_dir, Path):
            errors.append(f"output_dir must be Path, got {type(self.output_dir)}")
        if self.output_label is not None and not isinstance(self.output_label, str):
            errors.append(
                f"output_label must be str or None, got {type(self.output_label)}"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

    def get_target_metric(self) -> float:
        """Get the target metric value based on mode.

        Returns
        -------
        target : float
            For ESS mode (volume_variation=None): ess_ratio * n_particles
            For dynamic mode: volume_variation
        """
        if self.volume_variation is not None:
            return self.volume_variation
        else:
            return self.ess_ratio * self.n_particles

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "prior_transform": self.prior_transform,
            "log_likelihood": self.log_likelihood,
            "n_dim": self.n_dim,
            "n_particles": self.n_particles,
            "ess_ratio": self.ess_ratio,
            "volume_variation": self.volume_variation,
            "log_likelihood_args": self.log_likelihood_args,
            "log_likelihood_kwargs": self.log_likelihood_kwargs,
            "vectorize": self.vectorize,
            "blobs_dtype": self.blobs_dtype,
            "periodic": self.periodic,
            "reflective": self.reflective,
            "pool": self.pool,
            "clustering": self.clustering,
            "normalize": self.normalize,
            "cluster_every": self.cluster_every,
            "split_threshold": self.split_threshold,
            "n_max_clusters": self.n_max_clusters,
            "sample": self.sample,
            "n_steps": self.n_steps,
            "n_max_steps": self.n_max_steps,
            "resample": self.resample,
            "output_dir": str(self.output_dir),
            "output_label": self.output_label,
            "random_state": self.random_state,
        }


# Algorithm constants (centralized to avoid inconsistency)
BETA_TOLERANCE: float = 1e-4
ESS_TOLERANCE: float = 0.01
DOF_FALLBACK: float = 1e6
TRIM_ESS: float = 0.99
TRIM_BINS: int = 1000
