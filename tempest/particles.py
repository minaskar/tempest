from typing import Any
import numpy as np

class Particles:
    """
    Class to store the particles and their associated weights.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    
    Attributes
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    u : numpy.ndarray
        Array of shape (n_particles, n_dim) containing the particles.
    logdetj : numpy.ndarray
        Array of shape (n_particles,) containing the log-determinant of the
        Jacobian of the transformation from the unit hypercube to the
        parameter space.
    logl : numpy.ndarray
        Array of shape (n_particles,) containing the log-likelihoods.
    logp : numpy.ndarray
        Array of shape (n_particles,) containing the log-priors.
    logw : numpy.ndarray
        Array of shape (n_particles,) containing the log-weights.
    iter : numpy.ndarray
        Array of shape (n_particles,) containing the iteration number of each
        particle.
    logz : numpy.ndarray
        Array of shape (n_particles,) containing the log-evidence of each
        particle.
    calls : numpy.ndarray
        Array of shape (n_particles,) containing the number of likelihood
        evaluations of each particle.
    steps : numpy.ndarray
        Array of shape (n_particles,) containing the number of steps of each
        particle.
    efficiency : numpy.ndarray
        Array of shape (n_particles,) containing the efficiency of each
        particle.
    ess : numpy.ndarray
        Array of shape (n_particles,) containing the effective sample size of
        each particle.
    accept : numpy.ndarray
        Array of shape (n_particles,) containing the acceptance rate of each
        particle.
    beta : numpy.ndarray
        Array of shape (n_particles,) containing the inverse temperature of
        each particle.
    """

    def __init__(self, n_particles, n_dim):
        self.n_particles = n_particles
        self.n_dim = n_dim

        self.past = dict(
            u = [],
            x = [],
            logl = [],
            logw = [],
            blobs = [],
            iter = [],
            logz = [],
            calls = [],
            steps = [],
            efficiency = [],
            ess = [],
            accept = [],
            beta = [],
        )

        self.results_dict = None

    def update(self, data):
        """
        Update the particles with the given data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to be added to the particles.
        
        Notes
        -----
        The dictionary must contain the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.
        """
        for key in data.keys():
            if key in self.past.keys():
                value = data.get(key)
                # Save to past states
                self.past.get(key).append(value)

    def pop(self, key):
        """
        Remove the last element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be removed.
        
        Notes
        -----
        This method is useful to remove the last element of the particles
        after the resampling step.
        """
        _ = self.past.get(key).pop()

    def get(self, key, index=None, flat=False):
        """
        Get the element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be returned.
        index : int, optional
            Index of the element to be returned. If None, all elements are
            returned.
        flat : bool, optional
            If True, the elements are returned as a flattened array. Otherwise,
            the elements are returned as a numpy.ndarray.
        
        Returns
        -------
        element : numpy.ndarray
            Array of shape (n_particles,) or (n_particles, n_dim) containing
            the elements of the given key.
        
        Notes
        -----
        If index is None, the elements are returned as a numpy.ndarray. If
        index is not None, the elements are returned as a numpy.ndarray with
        shape (n_dim,). If flat is True, the elements are returned as a
        flattened array.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.get("u").shape
        (10, 2)
        >>> particles.get("u", index=0).shape
        (2,)
        >>> particles.get("u", index=0, flat=True).shape
        (2,)
        >>> particles.get("u", index=None, flat=True).shape
        (20,)
        """
        if index is None:
            if flat:
                return np.concatenate(self.past.get(key))
            else:
                return np.asarray(self.past.get(key))
        else:
            return self.past.get(key)[index]
        
    def compute_logw_and_logz(self, beta_final=1.0, normalize=True):
        """
        Compute importance log-weights for all collected samples targeting a
        final inverse temperature ``beta_final`` and the corresponding log-evidence
        estimate.

        This implementation is robust to variable numbers of active particles
        across iterations (adaptive n_active). It treats the overall proposal as
        a mixture over the per-iteration tempered distributions, weighted by the
        number of particles drawn from each (balance heuristic for MIS).

        Let T be the number of iterations for which particles were stored, with
        per-iteration parameters (beta_t, logZ_t, n_t) where n_t is the number
        of particles in iteration t and N = sum_t n_t. For each collected sample
        s with log-likelihood logl_s, the (unnormalized) importance weight is

            logw_s = beta_final * logl_s - log( sum_t (n_t/N) * exp(beta_t * logl_s - logZ_t) ).

        Parameters
        ----------
        beta_final : float, optional
            Target inverse temperature. Defaults to 1.0 (posterior).
        normalize : bool, optional
            If True, return log-weights normalized to sum to 1 in linear space.

        Returns
        -------
        logw : numpy.ndarray, shape (N,)
            Per-sample log-weights, optionally normalized so that exp(logw).sum() == 1.
        logz_new : float
            Log-evidence estimate for the target at ``beta_final`` based on
            the (unnormalized) log-weights.
        """

        # Per-iteration parameters (length T)
        beta = np.asarray(self.get("beta"))  # shape (T,)
        logz_iter = np.asarray(self.get("logz"))  # shape (T,)

        if beta.size == 0:
            # No stored states yet; return empty arrays
            return np.array([]), -np.inf

        # Per-sample log-likelihoods across all iterations, flattened (length N)
        logl_all = self.get("logl", flat=True)  # shape (N,)

        # A = beta_final * logl_s for each sample s
        A = logl_all * beta_final  # shape (N,)

        # Compute B_s = log( sum_t (n_t/N) * exp(beta_t * logl_s - logZ_t) ) robustly.
        # This uses the balance heuristic for MIS: weight each iteration by its
        # particle count n_t, so iterations with more particles contribute more.

        # Get number of particles per iteration
        logl_per_iter = self.past.get("logl")  # list of arrays, one per iteration
        n_per_iter = np.array([len(logl_per_iter[t]) for t in range(len(beta))])  # shape (T,)
        N_total = n_per_iter.sum()

        # Form a matrix b_{s,t} = beta_t * logl_s - logZ_t with broadcasting.
        # logl_all[:, None] -> (N,1); beta[None, :] -> (1,T); logz_iter[None, :] -> (1,T)
        b = logl_all[:, None] * beta[None, :] - logz_iter[None, :]  # shape (N, T)

        # Add log mixture weights: log(n_t / N_total) for each iteration
        log_mixture_weights = np.log(n_per_iter) - np.log(N_total)  # shape (T,)
        b_weighted = b + log_mixture_weights[None, :]  # shape (N, T)

        # Stable log-sum-exp over t for each sample s
        B = np.logaddexp.reduce(b_weighted, axis=1)  # shape (N,)

        # Unnormalized per-sample log-weights for the target at beta_final
        logw = A - B  # shape (N,)

        # Evidence estimate uses unnormalized log-weights
        logz_new = np.logaddexp.reduce(logw) - np.log(logw.size)

        if normalize and logw.size:
            logw = logw - np.logaddexp.reduce(logw)

        return logw, logz_new
    
    def compute_results(self):
        """
        Compute the results of the particles.

        Returns
        -------
        results_dict : dict
            Dictionary containing the results of the particles.

        Notes
        -----
        The dictionary contains the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.compute_results().keys()
        dict_keys(['u', 'logdetj', 'logl', 'logp', 'logw', 'blobs', 'iter', 'logz', 'calls', 'steps', 'efficiency', 'ess', 'accept', 'beta'])
        """
        if self.results_dict is None:
            self.results_dict = dict()
            for key in self.past.keys():
                self.results_dict[key] = self.get(key)

            logw, _ = self.compute_logw_and_logz(1.0)

            self.results_dict["logw"] = logw
            #self.results_dict["ess"] = np.exp(log_ess)

        return self.results_dict

