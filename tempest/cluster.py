import numpy as np
from typing import Optional


class GaussianMixture:
    """
    Gaussian Mixture Model with support for weighted samples.

    This class provides similar functionality to sklearn's GaussianMixture
    but optionally accepts sample weights during fitting.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    covariance_type : str
        Type of covariance parameters. Options: 'full', 'tied', 'diag', 'spherical'.
    max_iter : int
        Maximum number of EM iterations.
    n_init : int
        Number of initializations to perform.
    tol : float
        Convergence threshold.
    reg_covar : float
        Regularization added to covariance diagonal.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        max_iter=1000,
        n_init=1,
        tol=1e-3,
        reg_covar=1e-6,
        random_state=None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        # Fitted parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = None

    def fit(
        self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "GaussianMixture":
        """
        Fit the Gaussian Mixture Model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. If None, all samples have equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples, n_features = X.shape

        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight must have the same length as X")

        # Normalize weights
        sample_weight = sample_weight / np.sum(sample_weight)

        # Weighted EM algorithm
        best_params = None
        best_lower_bound = -np.inf

        if self.random_state is not None:
            np.random.seed(self.random_state)

        for init in range(self.n_init):
            # Initialize parameters
            weights, means, covariances = self._initialize_parameters(X, sample_weight)

            # EM iterations
            lower_bound = -np.inf
            for iteration in range(self.max_iter):
                # E-step: compute responsibilities
                responsibilities = self._e_step(X, weights, means, covariances)

                # M-step: update parameters
                weights, means, covariances = self._m_step(
                    X, responsibilities, sample_weight
                )

                # Compute log-likelihood
                new_lower_bound = self._compute_lower_bound(
                    X, weights, means, covariances, sample_weight
                )

                # Check convergence
                if new_lower_bound - lower_bound < self.tol:
                    break

                lower_bound = new_lower_bound

            # Keep best initialization
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = (weights, means, covariances, iteration + 1)

        # Store best parameters
        self.weights_, self.means_, self.covariances_, self.n_iter_ = best_params
        self.converged_ = self.n_iter_ < self.max_iter
        self.lower_bound_ = best_lower_bound

        return self

    def _initialize_parameters(self, X, sample_weight):
        """Initialize GMM parameters using weighted k-means++."""
        n_samples, n_features = X.shape

        # Initialize means using weighted k-means++
        means = np.zeros((self.n_components, n_features))

        # First center: weighted random sample
        cumsum = np.cumsum(sample_weight)
        r = np.random.rand() * cumsum[-1]
        means[0] = X[np.searchsorted(cumsum, r)]

        # Remaining centers
        for k in range(1, self.n_components):
            # Compute weighted distances to nearest center
            distances = np.min(
                [np.sum((X - means[j]) ** 2, axis=1) for j in range(k)], axis=0
            )
            probabilities = distances * sample_weight
            probabilities /= np.sum(probabilities)

            cumsum = np.cumsum(probabilities)
            r = np.random.rand() * cumsum[-1]
            means[k] = X[np.searchsorted(cumsum, r)]

        # Initialize responsibilities and compute initial parameters
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            distances = np.sum((X - means[k]) ** 2, axis=1)
            responsibilities[:, k] = np.exp(-0.5 * distances)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # Compute initial weights and covariances
        weights, means, covariances = self._m_step(X, responsibilities, sample_weight)

        return weights, means, covariances

    def _e_step(self, X, weights, means, covariances):
        """E-step: compute responsibilities."""
        from scipy.stats import multivariate_normal

        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            cov = self._get_covariance(covariances, k)
            try:
                responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
                    X, mean=means[k], cov=cov + np.eye(cov.shape[0]) * self.reg_covar
                )
            except (np.linalg.LinAlgError, ValueError):
                responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
                    X, mean=means[k], cov=np.eye(len(means[k])) * self.reg_covar
                )

        # Normalize
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True) + 1e-10

        return responsibilities

    def _m_step(self, X, responsibilities, sample_weight):
        """M-step: update parameters using weighted samples."""
        n_samples, n_features = X.shape

        # Combine responsibilities with sample weights
        weighted_resp = responsibilities * sample_weight[:, np.newaxis]

        # Update weights
        weights = np.sum(weighted_resp, axis=0)
        weights /= np.sum(weights)

        # Update means
        means = np.dot(weighted_resp.T, X) / (
            np.sum(weighted_resp, axis=0)[:, np.newaxis] + 1e-10
        )

        # Update covariances
        covariances = self._compute_covariances(X, means, weighted_resp)

        return weights, means, covariances

    def _compute_covariances(self, X, means, weighted_resp):
        """Compute covariances based on covariance_type."""
        n_samples, n_features = X.shape

        if self.covariance_type == "full":
            covariances = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.dot(weighted_resp[:, k] * diff.T, diff)
                covariances[k] /= np.sum(weighted_resp[:, k]) + 1e-10

        elif self.covariance_type == "tied":
            covariances = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances += np.dot(weighted_resp[:, k] * diff.T, diff)
            covariances /= n_samples

        elif self.covariance_type == "diag":
            covariances = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.sum(
                    weighted_resp[:, k, np.newaxis] * diff**2, axis=0
                )
                covariances[k] /= np.sum(weighted_resp[:, k]) + 1e-10

        elif self.covariance_type == "spherical":
            covariances = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.sum(weighted_resp[:, k, np.newaxis] * diff**2)
                covariances[k] /= np.sum(weighted_resp[:, k]) * n_features + 1e-10

        return covariances

    def _get_covariance(self, covariances, k):
        """Get covariance matrix for component k."""
        if self.covariance_type == "full":
            return covariances[k]
        elif self.covariance_type == "tied":
            return covariances
        elif self.covariance_type == "diag":
            return np.diag(covariances[k])
        elif self.covariance_type == "spherical":
            n_features = self.means_.shape[1]
            return np.eye(n_features) * covariances[k]

    def _compute_lower_bound(self, X, weights, means, covariances, sample_weight):
        """Compute weighted log-likelihood lower bound."""
        from scipy.stats import multivariate_normal

        n_samples = X.shape[0]
        log_likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            cov = self._get_covariance(covariances, k)
            try:
                log_prob = multivariate_normal.logpdf(
                    X, mean=means[k], cov=cov + np.eye(cov.shape[0]) * self.reg_covar
                )
                log_likelihood += weights[k] * np.exp(log_prob)
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Weight by sample weights
        log_likelihood = np.log(log_likelihood + 1e-10)
        return np.sum(sample_weight * log_likelihood)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        from scipy.stats import multivariate_normal

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples = X.shape[0]
        log_probabilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            cov = self._get_covariance(self.covariances_, k)
            try:
                log_probabilities[:, k] = np.log(
                    self.weights_[k] + 1e-10
                ) + multivariate_normal.logpdf(
                    X,
                    mean=self.means_[k],
                    cov=cov + np.eye(cov.shape[0]) * self.reg_covar,
                )
            except (np.linalg.LinAlgError, ValueError):
                log_probabilities[:, k] = -np.inf

        return np.argmax(log_probabilities, axis=1)

    def bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion."""
        n_samples, n_features = X.shape

        # Number of free parameters
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "spherical":
            cov_params = self.n_components

        n_parameters = (
            (self.n_components - 1) + self.n_components * n_features + cov_params
        )

        # Log-likelihood
        log_likelihood = (
            self._compute_lower_bound(
                X,
                self.weights_,
                self.means_,
                self.covariances_,
                np.ones(n_samples) / n_samples,
            )
            * n_samples
        )

        return -2 * log_likelihood + n_parameters * np.log(n_samples)


class HierarchicalGaussianMixture:
    def __init__(
        self,
        n_init=1,
        max_iterations=1000,
        min_points=None,
        threshold_modifier=1.0,
        covariance_type="full",
        verbose=False,
        normalize=False,
    ):
        self.n_init = n_init
        self.max_iterations = max_iterations
        self.min_points = min_points
        self.covariance_type = covariance_type
        self.verbose = verbose
        self.normalize = normalize
        modifier = float(threshold_modifier)
        if modifier <= 0:
            raise ValueError("threshold_modifier must be positive.")
        self.threshold_modifier = modifier

        # Attributes to be populated during fitting
        self.labels_ = None
        self.cluster_centers_ = []
        self.cluster_covariances_ = []
        self.cluster_weights_ = []
        self.n_clusters_ = 0
        self._gmm_ready = False

        # Normalization bounds
        self._data_min = None
        self._data_max = None

    def _normalize_data(self, X):
        """Normalize data to [0, 1]^D range."""
        if self._data_min is None or self._data_max is None:
            raise ValueError("Normalization bounds not set. Call fit first.")

        X_norm = (X - self._data_min) / (self._data_max - self._data_min + 1e-10)
        return X_norm

    def _denormalize_data(self, X_norm):
        """Denormalize data from [0, 1]^D back to original range."""
        if self._data_min is None or self._data_max is None:
            raise ValueError("Normalization bounds not set. Call fit first.")

        X = X_norm * (self._data_max - self._data_min) + self._data_min
        return X

    def _denormalize_covariance(self, cov_norm):
        """Denormalize covariance matrix."""
        if self._data_min is None or self._data_max is None:
            raise ValueError("Normalization bounds not set. Call fit first.")

        scale = self._data_max - self._data_min
        if len(cov_norm.shape) == 2:  # full or tied covariance
            cov = cov_norm * np.outer(scale, scale)
        elif len(cov_norm.shape) == 1:  # diagonal covariance
            cov = cov_norm * (scale**2)
        else:  # spherical covariance
            cov = cov_norm * np.mean(scale**2)
        return cov

    def _compute_effective_sample_size(self, weights):
        """Compute the effective sample size (ESS) for weighted samples."""
        weights = np.asarray(weights)
        normalized_weights = weights / np.sum(weights)
        return 1.0 / np.sum(normalized_weights**2)

    def _compute_bic_tolerance(self, n_features, weights):
        """Compute automatic BIC tolerance based on dimensionality and effective sample size."""
        D = n_features
        N_eff = self._compute_effective_sample_size(weights)
        n_params = D + D * (D + 1) / 2 + 1
        return n_params * np.log(N_eff)

    def fit(
        self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "HierarchicalGaussianMixture":
        """Fit the hierarchical Gaussian mixture model."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight must have the same length as X")

        # Store normalization bounds and normalize if requested
        if self.normalize:
            self._data_min = np.min(X, axis=0)
            self._data_max = np.max(X, axis=0)
            X = self._normalize_data(X)

        min_points = self.min_points if self.min_points is not None else 2 * n_features
        clusters = [[i for i in range(n_samples)]]
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            best_improvement = -np.inf
            best_split = None
            best_parent_idx = None
            best_bic_threshold = None

            for idx, indices in enumerate(clusters):
                if len(indices) < min_points:
                    continue

                data = X[indices]
                weights = sample_weight[indices]

                base_threshold = self._compute_bic_tolerance(n_features, weights)
                bic_threshold = self.threshold_modifier * base_threshold

                parent_gmm = GaussianMixture(
                    n_components=1,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    random_state=42,
                )
                parent_gmm.fit(data, sample_weight=weights)
                parent_bic = parent_gmm.bic(data)

                child_gmm = GaussianMixture(
                    n_components=2,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    random_state=42,
                )
                child_gmm.fit(data, sample_weight=weights)
                child_bic = child_gmm.bic(data)
                improvement = parent_bic - child_bic

                if self.verbose:
                    print(
                        "Cluster {idx}: parent BIC={parent_bic:.2f}, children BIC={child_bic:.2f}, "
                        "improvement={improvement:.2f}, threshold={bic_threshold:.2f}".format(
                            idx=idx,
                            parent_bic=parent_bic,
                            child_bic=child_bic,
                            improvement=improvement,
                            bic_threshold=bic_threshold,
                        )
                    )

                if improvement > bic_threshold and improvement > best_improvement:
                    labels = child_gmm.predict(data)
                    child1 = [indices[i] for i in range(len(indices)) if labels[i] == 0]
                    child2 = [indices[i] for i in range(len(indices)) if labels[i] == 1]
                    if len(child1) >= min_points and len(child2) >= min_points:
                        best_improvement = improvement
                        best_split = (child1, child2)
                        best_parent_idx = idx
                        best_bic_threshold = bic_threshold

            if best_split is None:
                if self.verbose:
                    print(f"No further splits accepted after {iteration} iterations.")
                break

            clusters.pop(best_parent_idx)
            clusters.extend(best_split)
            if self.verbose:
                print(
                    "Iteration {iteration}: Split cluster {parent} into {child_a} and {child_b} "
                    "(BIC improvement {improvement:.2f}, threshold {threshold:.2f})".format(
                        iteration=iteration,
                        parent=best_parent_idx,
                        child_a=len(clusters) - 2,
                        child_b=len(clusters) - 1,
                        improvement=best_improvement,
                        threshold=best_bic_threshold,
                    )
                )

        labels = np.full(n_samples, -1, dtype=int)
        cluster_centers = []
        cluster_covariances = []

        for cluster_idx, indices in enumerate(clusters):
            data = X[indices]
            weights = sample_weight[indices]

            if len(data) >= n_features:
                gmm = GaussianMixture(
                    n_components=1,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    random_state=42,
                )
                gmm.fit(data, sample_weight=weights)
                center = gmm.means_[0]
                cov = (
                    gmm.covariances_[0]
                    if self.covariance_type == "full"
                    else gmm.covariances_
                )
            else:
                center = np.mean(data, axis=0)
                cov = np.eye(n_features)

            # Denormalize centers and covariances if normalization was used
            if self.normalize:
                center = self._denormalize_data(center)
                cov = self._denormalize_covariance(cov)

            cluster_centers.append(center)
            cluster_covariances.append(cov)
            labels[indices] = cluster_idx

        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        self.cluster_covariances_ = cluster_covariances
        self.n_clusters_ = len(clusters)

        total_weight = np.sum(sample_weight)
        self.cluster_weights_ = np.array(
            [
                np.sum(sample_weight[labels == i]) / total_weight
                for i in range(self.n_clusters_)
            ]
        )

        self._gmm_ready = self.n_clusters_ > 0 and len(self.cluster_centers_) > 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data points."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Normalize input data if normalization is enabled
        if self.normalize:
            X = self._normalize_data(X)
            # Normalize centers for comparison
            centers = np.array([self._normalize_data(c) for c in self.cluster_centers_])
        else:
            centers = np.array(self.cluster_centers_)

        if self._gmm_ready:
            try:
                probabilities = self._compute_gaussian_probabilities(X)
                return np.argmax(probabilities, axis=1)
            except Exception as exc:
                if self.verbose:
                    print(
                        f"Warning: GMM prediction failed: {exc}. Using nearest center fallback."
                    )

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        return np.argmin(distances, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for new data points."""
        if self.cluster_centers_ is None or not self.cluster_centers_:
            raise ValueError("The model has not been fitted yet.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Normalize input data if normalization is enabled
        if self.normalize:
            X = self._normalize_data(X)
            # Normalize centers for comparison
            centers = np.array([self._normalize_data(c) for c in self.cluster_centers_])
        else:
            centers = np.array(self.cluster_centers_)

        if self._gmm_ready:
            try:
                return self._compute_gaussian_probabilities(X)
            except Exception as exc:
                if self.verbose:
                    print(
                        f"Warning: GMM probability prediction failed: {exc}. Using distance fallback."
                    )

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        inv_distances = 1.0 / (distances + 1e-8)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    def _compute_gaussian_probabilities(self, X):
        """Compute Gaussian mixture probabilities manually.

        Note: X should already be normalized if normalization is enabled.
        """
        from scipy.stats import multivariate_normal

        n_samples = X.shape[0]
        log_probabilities = np.zeros((n_samples, self.n_clusters_))

        for k in range(self.n_clusters_):
            # Normalize mean and covariance if normalization is enabled
            if self.normalize:
                mean = self._normalize_data(self.cluster_centers_[k])
                # Normalize covariance
                scale = self._data_max - self._data_min
                cov_orig = self.cluster_covariances_[k]
                if len(cov_orig.shape) == 2:  # full or tied covariance
                    cov = cov_orig / np.outer(scale, scale)
                elif len(cov_orig.shape) == 1:  # diagonal covariance
                    cov = cov_orig / (scale**2)
                else:  # spherical covariance
                    cov = cov_orig / np.mean(scale**2)
            else:
                mean = self.cluster_centers_[k]
                cov = self.cluster_covariances_[k]
            weight = self.cluster_weights_[k]

            if self.covariance_type == "full":
                try:
                    regularized_cov = cov + np.eye(cov.shape[0]) * 1e-6
                    log_prob = multivariate_normal.logpdf(
                        X, mean=mean, cov=regularized_cov
                    )
                except Exception:
                    log_prob = multivariate_normal.logpdf(
                        X, mean=mean, cov=np.eye(len(mean))
                    )
            else:
                try:
                    if self.covariance_type == "tied":
                        full_cov = cov
                    elif self.covariance_type == "diag":
                        full_cov = np.diag(cov)
                    elif self.covariance_type == "spherical":
                        full_cov = np.eye(len(mean)) * cov
                    else:
                        full_cov = np.eye(len(mean))

                    regularized_cov = full_cov + np.eye(full_cov.shape[0]) * 1e-6
                    log_prob = multivariate_normal.logpdf(
                        X, mean=mean, cov=regularized_cov
                    )
                except Exception:
                    log_prob = multivariate_normal.logpdf(
                        X, mean=mean, cov=np.eye(len(mean))
                    )

            log_probabilities[:, k] = log_prob + np.log(weight + 1e-10)

        from scipy.special import logsumexp

        log_prob_norm = logsumexp(log_probabilities, axis=1, keepdims=True)
        return np.exp(log_probabilities - log_prob_norm)
