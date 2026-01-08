# Tools

The `tools` module provides utility functions used throughout Tempest.

## Overview

This module contains:

- Resampling algorithms
- Effective sample size calculations
- Progress bar management
- Function wrappers for parallelization

---

## Functions Reference

### Weight Manipulation

::: tempest.tools.trim_weights
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

::: tempest.tools.effective_sample_size
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

::: tempest.tools.unique_sample_size
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Resampling

::: tempest.tools.systematic_resample
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Utility Classes

::: tempest.tools.ProgressBar
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

::: tempest.tools.FunctionWrapper
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## Usage Examples

### Effective Sample Size

```python
from tempest.tools import effective_sample_size
import numpy as np

# Compute ESS from weights
weights = np.array([0.4, 0.3, 0.2, 0.1])
ess = effective_sample_size(weights)
print(f"ESS: {ess:.1f}")  # ~3.3 for these weights
```

### Weight Trimming

```python
from tempest.tools import trim_weights
import numpy as np

# Remove high-weight outliers
samples = np.random.randn(1000, 2)
weights = np.random.exponential(size=1000)
weights /= weights.sum()

samples_trimmed, weights_trimmed = trim_weights(
    samples, weights, 
    ess=0.99,  # Keep 99% of original ESS
    bins=1000
)
```

### Systematic Resampling

```python
from tempest.tools import systematic_resample
import numpy as np

# Resample with weights
weights = np.array([0.5, 0.3, 0.15, 0.05])
indices = systematic_resample(size=10, weights=weights)
print(indices)  # e.g., [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
```

---

## Implementation Details

### ESS Calculation

The effective sample size is computed as:

$$
\text{ESS} = \frac{1}{\sum_i w_i^2}
$$

where $w_i$ are normalized weights summing to 1.

### Unique Sample Size

The unique sample size estimates how many unique samples would result from resampling $k$ times:

$$
\text{USS} = \sum_i \left(1 - (1 - w_i)^k\right)
$$

This is useful for multimodal distributions where ESS can be misleading.

### Systematic Resampling

Systematic resampling is a low-variance alternative to multinomial resampling. It places evenly-spaced "pointers" across the cumulative distribution, ensuring more uniform coverage of the weight distribution.
