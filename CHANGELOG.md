# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-01-11

### Added
- Initial release of Tempest
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
- Migrated from setup.py/setup.cfg to pyproject.toml for modern packaging
- Removed torch from requirements (Python-only implementation)

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

[Unreleased]: https://github.com/minaskar/tempest/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/minaskar/tempest/releases/tag/v0.9.0
