# Prior Distributions

This guide covers how to define prior distributions for Tempest.

## Prior Concepts

Tempest uses a **prior transform** approach, where samples are drawn from a unit hypercube $[0, 1]^n$ and transformed to the prior space. This is efficient for importance sampling and allows flexible prior definitions.

---

## Using scipy.stats Distributions

The simplest way to define priors is using scipy distributions in a list:

```python
from scipy.stats import uniform, norm, expon

n_dim = 3

# List of scipy.stats distributions
prior_dists = [
    uniform(-10, 20),    # U(-10, 10)
    norm(0, 1),          # N(0, 1)
    expon(scale=1),      # Exp(λ=1)
]

# Create transform from distributions
def prior_transform(u):
    return np.array([dist.ppf(u[i]) for i, dist in enumerate(prior_dists)])

sampler = pc.Sampler(
    prior_transform=prior_transform,
    log_likelihood=log_likelihood,
    n_dim=n_dim,
)
```

Or more simply, use a list comprehension:

```python
def prior_transform(u):
    return np.array([prior_dists[i].ppf(u[i]) for i in range(len(u))])

---

## Common Prior Distributions

### Uniform Prior

```python
from scipy.stats import uniform

# uniform(loc, scale) gives U(loc, loc+scale)
prior_param = uniform(0, 10)  # U(0, 10)
prior_param = uniform(-5, 10)  # U(-5, 5)
```

### Gaussian Prior

```python
from scipy.stats import norm

# norm(loc, scale) gives N(loc, scale²)
prior_param = norm(0, 1)     # Standard normal
prior_param = norm(5, 2)     # N(5, 4)
```

### Log-Uniform Prior

```python
from scipy.stats import loguniform

# loguniform(a, b) gives log-uniform on [a, b]
prior_param = loguniform(1e-3, 1e3)
```

### Truncated Normal

```python
from scipy.stats import truncnorm

# truncnorm(a, b, loc, scale)
# a, b are the lower/upper bounds in standard units
a, b = (0 - 5) / 1, (10 - 5) / 1  # Truncate N(5, 1) to [0, 10]
prior_param = truncnorm(a, b, loc=5, scale=1)
```

### Beta Distribution

```python
from scipy.stats import beta

# Useful for bounded parameters
prior_param = beta(2, 5)  # Skewed towards 0
```

---

## Custom Prior Transform

For complex priors, define a transform function directly:

```python
import numpy as np
from scipy.stats import norm

def prior_transform(u):
    """
    Transform samples from unit hypercube to prior space.
    
    Parameters
    ----------
    u : ndarray, shape (n_dim,)
        Sample from [0, 1]^n_dim
    
    Returns
    -------
    x : ndarray, shape (n_dim,)
        Sample from prior distribution
    """
    x = np.zeros_like(u)
    
    # Parameter 0: Uniform(-10, 10)
    x[0] = 20 * u[0] - 10
    
    # Parameter 1: Gaussian N(0, 1)
    x[1] = norm.ppf(u[1])
    
    # Parameter 2: Log-uniform(1e-3, 1e3)
    x[2] = 10**(6 * u[2] - 3)
    
    return x

# Use directly in sampler
sampler = pc.Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
)
```

---

## Correlated Priors

For correlated priors (e.g., multivariate Gaussian), use Cholesky decomposition:

```python
import numpy as np
from scipy.stats import norm

# Prior: N(mu, Sigma)
mu = np.array([0, 0])
Sigma = np.array([[1.0, 0.5],
                  [0.5, 1.0]])

# Cholesky decomposition
L = np.linalg.cholesky(Sigma)

def prior_transform(u):
    # Transform to standard normal
    z = norm.ppf(u)
    # Apply correlation
    return mu + L @ z
```

---

## Conditional Priors

For hierarchical or conditional priors:

```python
def prior_transform(u):
    x = np.zeros(3)
    
    # Parameter 0: U(0, 1)
    x[0] = u[0]
    
    # Parameter 1 depends on parameter 0
    # x[1] ~ U(0, x[0])
    x[1] = u[1] * x[0]
    
    # Parameter 2: Normal with std depending on x[0]
    x[2] = norm.ppf(u[2], loc=0, scale=x[0])
    
    return x
```

---

## Physical Parameter Bounds

Ensure physical bounds are respected:

```python
from scipy.stats import truncnorm

def prior_transform(u):
    x = np.zeros(2)
    
    # Mass: positive, centered at 1, std 0.5
    a = (0 - 1) / 0.5  # Lower bound in std units
    x[0] = truncnorm.ppf(u[0], a=a, b=np.inf, loc=1, scale=0.5)
    
    # Probability: must be in [0, 1]
    x[1] = u[1]  # Already uniform on [0, 1]
    
    return x
```

---

## Tips for Prior Selection

!!! tip "Prior Predictive Checks"
    Always check your prior by sampling from it and visualizing:
    ```python
    u_samples = np.random.rand(1000, n_dim)
    x_samples = np.array([prior_transform(u) for u in u_samples])
    # Plot histograms, corner plots, etc.
    ```

!!! tip "Informative vs. Uninformative"
    - Use informative priors when you have prior knowledge
    - Use wide, uninformative priors when exploring
    - Log-uniform priors are useful for scale parameters

!!! warning "Improper Priors"
    Tempest requires proper (normalizable) priors. Improper priors like $p(\theta) \propto 1$ over infinite support will cause issues.

---

## Example: Astronomical Prior

```python
import numpy as np
from scipy.stats import uniform, norm, loguniform

def prior_transform(u):
    """Prior for an exoplanet transit model."""
    x = np.zeros(5)
    
    # Period: log-uniform from 1 to 100 days
    x[0] = loguniform.ppf(u[0], 1, 100)
    
    # Transit epoch: uniform within one period
    x[1] = x[0] * u[1]
    
    # Planet-to-star radius ratio: U(0, 0.3)
    x[2] = 0.3 * u[2]
    
    # Impact parameter: U(0, 1)
    x[3] = u[3]
    
    # Orbital inclination (degrees): 90 ± 10
    x[4] = norm.ppf(u[4], 90, 10)
    
    return x
```
