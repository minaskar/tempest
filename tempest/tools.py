import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def trim_weights(
    samples: np.ndarray, weights: np.ndarray, ess: float = 0.99, bins: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Trim samples and weights to a given effective sample size.

    Parameters
    ----------
    samples : ``np.ndarray``
        Samples.
    weights : ``np.ndarray``
        Weights.
    ess : ``float``
        Effective sample size threshold.
    bins : ``int``
        Number of bins to use for trimming.

    Returns
    -------
    samples_trimmed : ``np.ndarray``
        Trimmed samples.
    weights_trimmed : ``np.ndarray``
        Trimmed weights.
    """

    # normalize weights
    weights /= np.sum(weights)
    # compute untrimmed ess
    ess_total = 1.0 / np.sum(weights**2.0)
    # define percentile grid
    percentiles = np.linspace(0, 99, bins)

    i = bins - 1
    while True:
        p = percentiles[i]
        # compute weight threshold
        threshold = np.percentile(weights, p)
        mask = weights >= threshold
        weights_trimmed = weights[mask]
        weights_trimmed /= np.sum(weights_trimmed)
        ess_trimmed = 1.0 / np.sum(weights_trimmed**2.0)
        if ess_trimmed / ess_total >= ess:
            break
        i -= 1

    return samples[mask], weights_trimmed


def volume_variation(x, w=None):
    """
    Compute volume variation metric.

    Metric that determines how well the samples capture the covariance structure
    of the target distribution using influence function approach (no bootstrap).

    Computes CV(sqrt(det(Cov))) using the influence function formula:
    CV = (1/2) * sqrt(sum_i w_i^2 * (d_i^2 - n_dim)^2)
    where d_i are Mahalanobis distances and w_i are normalized weights.

    The sqrt(det(Cov)) represents the volume of the confidence ellipsoid,
    so this metric measures the coefficient of variation of the ellipsoid volume.

    Parameters
    ----------
    x : np.ndarray
        Samples array with shape (n_samples, n_dim).
    w : np.ndarray, optional
        Weights array with shape (n_samples,). If None, assumes uniform weights.

    Returns
    -------
    volume_variation : float
        Coefficient of variation of sqrt(det(Cov)) (range [0, infinity))
        where lower is better. With perfect coverage, volume_variation = 0.0.
    """
    x = np.asarray(x)
    n_samples, n_dim = x.shape
    if n_samples < n_dim + 1:
        return 1e10  # Large finite value instead of np.inf

    if w is None:
        w = np.ones(n_samples)

    w = np.asarray(w)
    w = w / np.sum(w)

    weighted_mean = np.sum(x * w[:, np.newaxis], axis=0)
    xc = x - weighted_mean

    cov = np.dot(xc.T, xc * w[:, np.newaxis])

    if np.linalg.matrix_rank(cov) < n_dim:
        # Add regularization to make covariance matrix invertible
        reg = 1e-6 * np.trace(cov)
        cov = cov + np.eye(n_dim) * reg

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If inversion fails, return a large finite value instead of np.inf
        return 1e10
    d2 = np.sum(xc @ cov_inv * xc, axis=1)

    # Clip to prevent overflow in early iterations when particles are far from center
    deviation = np.clip(d2 - n_dim, -1e6, 1e6)
    cv = 0.5 * np.sqrt(np.sum(w**2 * deviation**2))

    return cv


def effective_sample_size(weights: np.ndarray) -> float:
    """
        Compute effective sample size (ESS).

    Parameters
    ----------
    weights : ``np.ndarray``
        Weights.

    Returns
    -------
    ess : ``float``
        Effective sample size.
    """
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights**2.0)


def compute_ess(logw: np.ndarray):
    r"""
        Compute effective sample size (per centage).

    Parameters
    ----------
    logw : ``np.ndarray``
        Log-weights.
    Returns
    -------
    ess : float
        Effective sample size divided by actual number
        of samples (between 0 and 1)
    """
    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    weights = np.exp(logw_normed) / np.sum(np.exp(logw_normed))
    return 1.0 / np.sum(weights * weights) / len(weights)


def increment_logz(logw: np.ndarray):
    r"""
        Compute log evidence increment factor.

    Parameters
    ----------
    logw : ``np.ndarray``
        Log-weights.
    Returns
    -------
    ess : float
        logZ increment.
    """
    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    return logw_max + np.logaddexp.reduce(logw_normed)


def systematic_resample(
    size: int, weights: np.ndarray, random_state: Optional[int] = None
) -> np.ndarray:
    """
        Resample a new set of points from the weighted set of inputs
        such that they all have equal weight.

    Parameters
    ----------
    size : `int`
        Number of samples to draw.
    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.
    random_state : `int`, optional
        Random seed.

    Returns
    -------
    indeces : `~numpy.ndarray` with shape (nsamples,)
        Indices of the resampled array.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> systematic_resample(4, w)
    array([0, 0, 0, 2])

    Notes
    -----
    Implements the systematic resampling method.
    """

    if random_state is not None:
        np.random.seed(random_state)

    if abs(np.sum(weights) - 1.0) > SQRTEPS:
        weights = np.array(weights) / np.sum(weights)

    positions = (np.random.random() + np.arange(size)) / size

    j = 0
    cumulative_sum = weights[0]
    indeces = np.empty(size, dtype=int)
    for i in range(size):
        while positions[i] > cumulative_sum:
            j += 1
            cumulative_sum += weights[j]
        indeces[i] = j

    return indeces


class ProgressBar:
    """
        Progress bar class.

    Parameters
    ----------
    show : `bool`
        Whether or not to show a progress bar. Default is ``True``.
    """

    def __init__(self, show: bool = True, initial: int = 0):
        self.progress_bar = tqdm(desc="Iter", disable=not show, initial=initial)
        self.info: Dict[str, Any] = dict()

    def update_stats(self, info: Dict[str, Any]) -> None:
        """
            Update shown stats.

        Parameters
        ----------
        info : dict
            Dictionary with stats to show.
        """
        self.info = {**self.info, **info}
        self.progress_bar.set_postfix(ordered_dict=self.info)

    def update_iter(self) -> None:
        """
        Update iteration counter.
        """
        self.progress_bar.update(1)

    def close(self) -> None:
        """
        Close progress bar.
        """
        self.progress_bar.close()


class FunctionWrapper(object):
    r"""
        Make the log-likelihood or log-prior function pickleable
        when ``args`` or ``kwargs`` are also included.

    Parameters
    ----------
    f : callable
        Log probability function.
    args : list
        Extra positional arguments to be passed to f.
    kwargs : dict
        Extra keyword arguments to be passed to f.
    """

    def __init__(
        self,
        f: Callable,
        args: Optional[List[Any]],
        kwargs: Optional[Dict[str, Any]],
    ):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x: np.ndarray) -> Any:
        """
            Evaluate log-likelihood or log-prior function.

        Parameters
        ----------
        x : ``np.ndarray``
            Input position array.

        Returns
        -------
        f : float or ``np.ndarray``
            f(x)
        """
        return self.f(x, *self.args, **self.kwargs)
