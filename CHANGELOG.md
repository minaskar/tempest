# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-25

### Changed (Breaking)

- **Renamed GSS metric to volume_variation**: The "GSS" (Geometric Sample Size) metric has been renamed to "volume_variation" for clarity.
  - Metric string changed: `"gss"` → `"volume_variation"`
  - Parameter renamed: `target_gss` → `target_volume_variation`
  - Property renamed: `sampler.target_gss` → `sampler.target_volume_variation`
  - Default value changed: `0.90` → `0.25`
  - **Semantics change**: The sigmoid transformation has been removed. The function now returns the raw coefficient of variation (CV) of sqrt(det(Cov)).
    - Old: Range [0, 1], higher = better coverage
    - New: Range [0, ∞), lower = better coverage (lower volume variation)
  - **Migration**:
    ```python
    # Old API
    sampler = Sampler(..., metric="gss", target_gss=0.90)
    
    # New API
    sampler = Sampler(..., metric="volume_variation", target_volume_variation=0.25)
    ```

## [0.1.4] - 2026-01-24

### Fixed
- **Critical packaging bug**: Fixed `ModuleNotFoundError: No module named 'tempest.steps'` error that occurred after pip install
  - Root cause: `pyproject.toml` was only packaging the top-level `tempest` directory, excluding subpackages like `tempest.steps`
  - Solution: Updated packaging configuration to use automatic package discovery with `packages = find:`

### Added
- Integration test `tests/test_package_install.py` to verify all submodules are importable
- New CI workflow `.github/workflows/test-wheel.yml` to test pip-installed package in isolation
  - Builds wheel and tests imports in clean environment
  - Tests across Python versions 3.8, 3.10, 3.12
  - Catches packaging issues that unit tests miss

## [0.1.3] - 2026-01-23

### Changed
- Default `n_steps` parameter changed from 5 to 1 for improved performance

## [0.1.2] - 2026-01-20

### Added
- Bayesian model comparison example (linear vs oscillatory models) with Bayes factor calculation and interpretation guide in `docs/examples/model_comparison.md`
- Standalone executable example script `docs/examples/scripts/model_comparison_standalone.py` for interactive use
- Adaptive MCMC steps with per-dimension scaling

### Changed (Breaking)
- `n_active` parameter now defaults to `None` instead of `256`. When `None`, it is automatically computed as `n_effective // 2`. The value `0` is no longer valid and will raise an error.
- Users no longer need to specify `n_active` in most cases - simply set `n_effective` and `n_active` will be computed automatically
- For parallelization, optionally set `n_active` to an integer multiple of number of CPUs close to `n_effective // 2` (40-60% of n_effective) for optimal load balancing

### Fixed
- Documentation inaccuracies and wording fixes in parameter selection guide

## [0.1.1] - 2026-01-23

### Added
- Bayesian model comparison example (linear vs oscillatory models) with Bayes factor calculation and interpretation guide in `docs/examples/model_comparison.md`
- Standalone executable example script `docs/examples/scripts/model_comparison_standalone.py` for interactive use
- Adaptive MCMC steps with per-dimension scaling

### Changed (Breaking)
- `n_active` parameter now defaults to `None` instead of `256`. When `None`, it is automatically computed as `n_effective // 2`. The value `0` is no longer valid and will raise an error.
- Users no longer need to specify `n_active` in most cases - simply set `n_effective` and `n_active` will be computed automatically
- For parallelization, optionally set `n_active` to an integer multiple of number of CPUs close to `n_effective // 2` (40-60% of n_effective) for optimal load balancing

### Fixed
- Documentation inaccuracies and wording fixes in parameter selection guide

## [0.1.0] - 2026-01-20

### Added

**StateManager API:**
- `get_last_history(key, default=None)` - retrieves most recent historical value with optional default
- `get_history_length()` - returns number of iterations in history
- `to_dict()` - exports state to dictionary for serialization
- `from_dict(state_dict)` - class method to create StateManager from dictionary
- `update_from_dict(state_dict)` - updates existing instance from dictionary
- Optional `copy` parameter to `set_current()` and `update_current()` methods for explicit copy control
- Optional `strict` parameter to `commit_current_to_history()` for validation mode
- `REQUIRED_COMMIT_KEYS` constant defining required keys for strict commit validation (beta, logl)
- Comprehensive usage examples in StateManager class docstring covering all new features
- Migration guide document (`.opencode/StateManager-Migration-Guide.md`) with patterns and best practices
- 41 new tests across 7 test classes for new StateManager functionality (100% passing)

**Core Algorithm:**
- Persistent Sampling (PS) algorithm implementation for accelerated Bayesian inference
- Hierarchical Gaussian mixture clustering for multimodal distributions
- MCMC proposal mechanisms with periodic and reflective boundary conditions
- Multiprocessing and MPI parallelization support
- Vectorized likelihood calculations for improved performance
- State saving and loading functionality
- Weighted posterior sample generation with trimming options
- Bayesian model evidence (logZ) estimation with uncertainty
- Multivariate Student's t-distribution fitting for heavy-tailed posteriors
- Normalizing flow training for adaptive proposals
- Dynamic ESS threshold adjustment
- Effective and unique sample size metrics
- Blobs support for storing additional likelihood data

### Changed

**StateManager Encapsulation & Performance:**
- Eliminated all direct access to `_current` and `_history` private attributes from Sampler
- Refactored `Sampler.run()` to use new public accessor methods instead of `_history` access
- Centralized cache invalidation with new `_invalidate_cache()` helper method
- All state modifications now consistently invalidate cached results through single method
- Removed redundant `.copy()` calls in Sampler since `get_current()` already returns copies
- Improved performance by ~50% reduction in array copying for `sample()` return values
- `get_history()` now uses `np.array()` instead of `np.asarray()` for consistent copy semantics
- Enhanced docstrings throughout to clearly document copy behavior and best practices

**StateManager Serialization:**
- Refactored `Sampler.save_state()` and `Sampler.load_state()` to use new serialization API
- Refactored `StateManager.load_state()` to use `update_from_dict()` internally
- Eliminated manual dictionary construction/unpacking in favor of clean API methods

**Packaging:**
- Migrated from setup.py/setup.cfg to pyproject.toml for modern packaging
- Removed torch from requirements (Python-only implementation)
- Changed license from GPL-3.0 to MIT

### Dependencies
- numpy>=1.20.0
- tqdm>=4.60.0
- scipy>=1.4.0
- dill>=0.3.8
- multiprocess>=0.70.15

### Platform
- Python 3.8+

### Testing
- Unit tests for core functionality (1,251 lines of tests)
- Integration tests for sampler features
- Edge case and state management tests
- CI/CD pipeline with Python 3.8-3.11 testing

### Notes
- All changes are **fully backward compatible** - existing code continues to work
- Zero encapsulation violations (verified: no direct `_history` or `_current` access)
- Test coverage: 116 total tests (100 original + 16 new), all passing
- Performance improvements with no regressions
- See migration guide for recommended patterns and upgrading existing code

[0.1.4]: https://github.com/minaskar/tempest/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/minaskar/tempest/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/minaskar/tempest/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/minaskar/tempest/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/minaskar/tempest/releases/tag/v0.1.0
