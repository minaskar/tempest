from __future__ import annotations

import numpy as np
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
    n_effective: Optional[int] = 512
    n_active: Optional[int] = None
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
    pool: Optional[Union[int, Any]] = None

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

        # Get the current values (may be None)
        n_active = self.n_active
        n_effective = self.n_effective

        # Default logic based on which values are provided
        if n_active is None and n_effective is None:
            # Both None: use standard defaults
            object.__setattr__(self, "n_effective", 512)
            object.__setattr__(self, "n_active", 256)
        elif n_active is None:
            # Only n_active is None, compute it from n_effective
            if n_effective is None:
                # This should not happen (handled above), but be defensive
                object.__setattr__(self, "n_effective", 512)
                object.__setattr__(self, "n_active", 256)
            else:
                object.__setattr__(self, "n_active", max(1, n_effective // 2))
        elif n_effective is None:
            # Only n_effective is None, compute it from n_active
            object.__setattr__(self, "n_effective", n_active * 2)
        # else: both provided explicitly, use as-is

        # Compute n_steps/n_max_steps defaults
        # n_steps now represents n_steps_0 (base steps per dimension at optimal acceptance rate of 23.4%)
        if self.n_steps is None or self.n_steps <= 0:
            object.__setattr__(self, "n_steps", 1)
        # n_max_steps now represents n_max_steps_0 (maximum steps per dimension)
        if self.n_max_steps is None or self.n_max_steps <= 0:
            object.__setattr__(self, "n_max_steps", 20 * self.n_steps)

        self.validate()

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

        # Check active/effective are positive integers
        if not isinstance(self.n_active, int) or not isinstance(self.n_effective, int):
            errors.append("n_active and n_effective must be integers")
        if self.n_active is not None and self.n_active <= 0:
            errors.append(f"n_active must be positive integer, got {self.n_active}")
        if self.n_effective is not None and self.n_effective <= 0:
            errors.append(
                f"n_effective must be positive integer, got {self.n_effective}"
            )

        # Check active/effective relationship
        if self.n_active >= self.n_effective:
            errors.append(
                f"n_active ({self.n_active}) must be < n_effective ({self.n_effective})"
            )

        # Check n_boost
        if self.n_boost is not None:
            if not isinstance(self.n_boost, int):
                errors.append(f"n_boost must be int or None, got {type(self.n_boost)}")
            elif self.n_boost < self.n_effective:
                errors.append(
                    f"n_boost ({self.n_boost}) must be >= n_effective ({self.n_effective})"
                )

        # Check metric
        if self.metric not in ["ess", "uss"]:
            errors.append(f"Invalid metric '{self.metric}': must be 'ess' or 'uss'")

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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "prior_transform": self.prior_transform,
            "log_likelihood": self.log_likelihood,
            "n_dim": self.n_dim,
            "n_effective": self.n_effective,
            "n_active": self.n_active,
            "n_boost": self.n_boost,
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
            "metric": self.metric,
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
BOOST_STEEPNESS: float = 0.125
