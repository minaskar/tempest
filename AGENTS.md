# Tempest - AGENTS.md

This guide is for agentic coding assistants working on the Tempest codebase.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Create virtual environment with uv
uv sync

# Activate venv (manual)
source .venv/bin/activate
```

### Running Tests
```bash
# Run all tests
python -m unittest discover tests

# Run single test file
python -m unittest tests.test_sampler

# Run specific test method
python -m unittest tests.test_sampler.SamplerTestCase.test_run

# Run with pytest (if installed)
pytest tests/
```

### Linting
```bash
# Check for syntax errors and undefined names (stops on error)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check for style issues (warnings only, doesn't stop)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Building
```bash
# Install from source
python setup.py install

# Or using uv
uv pip install -e .
```

## Code Style Guidelines

### Import Order
Standard library → third-party → local tempest imports. Group with blank lines between.

```python
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from multiprocess import Pool
import dill

from tempest.mcmc import parallel_mcmc
from tempest.tools import systematic_resample, FunctionWrapper
```

### Type Hints
Use Python 3.8+ compatible type hints for function signatures, especially for complex types. Keep it lightweight - don't over-annotate.

```python
def load_state(self, path: Union[str, Path]):
    """Load state of sampler from file."""
    ...
```

### Docstrings
Use NumPy style docstrings. Include Parameters, Returns, and Notes sections where applicable.

```python
def prior_transform(u):
    """Transform from uniform [0,1] to standard normal.
    
    Parameters
    ----------
    u : np.ndarray
        Unit hypercube coordinates.
    
    Returns
    -------
    x : np.ndarray
        Transformed physical parameters.
    """
```

### Naming Conventions
- **Classes**: PascalCase (`Sampler`, `Particles`, `HierarchicalGaussianMixture`)
- **Functions/Methods**: snake_case (`run`, `prior_transform`, `log_likelihood`)
- **Constants**: UPPER_SNAKE_CASE (`BETA_TOLERANCE`, `ESS_TOLERANCE`)
- **Private methods**: leading underscore (`_reweight`, `_mutate`, `_train`)
- **Public attributes**: lowercase (`self.u`, `self.x`, `self.logl`)

### Formatting
- PEP 8 compliant
- Max line length: 127 characters
- Max complexity: 10 (per flake8 config)
- Use meaningful variable names, avoid abbreviations unless widely known (`n_dim`, `ess`, `logl`)

### Error Handling
Use descriptive error messages with context. Prefer `ValueError` for invalid inputs.

```python
if metric not in ["ess", "uss"]:
    raise ValueError(f"Invalid metric {metric}. Options are 'ess' or 'uss'.")
```

### Constants
Define constants at class level or module level in UPPER_SNAKE_CASE with descriptive comments.

```python
# Constants
self.BETA_TOLERANCE = 1e-4
self.ESS_TOLERANCE = 0.01
self.DYNAMIC_RATIO_LOWER = 0.95
```

### Function Wrappers
Use `FunctionWrapper` (from `tempest.tools`) for wrapping likelihood functions with args/kwargs.

### Parallel Processing
Use `multiprocess.Pool` for parallelization. The sampler accepts `pool` parameter as int (creates pool) or Pool instance.

### Testing
- Use `unittest` framework
- Test files in `tests/` directory named `test_*.py`
- Test classes inherit from `unittest.TestCase`
- Use `random_state` parameter for reproducibility in tests

### Important Patterns
- The main class is `Sampler` in `tempest/sampler.py`
- Core algorithm: `_reweight()` → `_train()` → `_resample()` → `_mutate()` (in `sample()` method)
- Particles managed by `Particles` class in `tempest/particles.py`
- State persistence via `save_state()`/`load_state()` using `dill`

### Dependencies
Core dependencies: `numpy`, `scipy`, `tqdm`, `dill`, `multiprocess`. All defined in `pyproject.toml` and `requirements.txt`.

### Before Submitting Changes
1. Run all tests: `python -m unittest discover tests`
2. Run linter: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
3. Update `CHANGELOG.md` with relevant entries
4. Update tests if functionality changes
