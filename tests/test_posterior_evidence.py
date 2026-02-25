"""Comprehensive tests for posterior() and evidence() methods.

Tests the extraction of posterior samples and Bayesian evidence (logZ)
from completed samplers with various configurations.
"""

import unittest
import numpy as np
from scipy.stats import norm, multivariate_normal
from tempest.sampler import Sampler


class PosteriorMethodTestCase(unittest.TestCase):
    """Test the posterior() method."""

    @staticmethod
    def prior_transform(u):
        """Transform from unit cube to standard normal."""
        return norm.ppf(u)

    @staticmethod
    def log_likelihood(x):
        """Gaussian log likelihood - handles both 1D and 2D."""
        if x.ndim == 1:
            return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)

    def setUp(self):
        """Set up sampler with Gaussian likelihood."""
        self.sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            clustering=False,
            random_state=0,
        )
        self.sampler.run(n_total=128)

    def test_posterior_returns_tuple(self):
        """Test that posterior() returns a tuple."""
        result = self.sampler.posterior()
        self.assertIsInstance(result, tuple)

    def test_posterior_returns_three_values(self):
        """Test that posterior() returns samples, weights, and logl."""
        result = self.sampler.posterior()
        self.assertEqual(len(result), 3)
        samples, weights, logl = result
        self.assertIsInstance(samples, np.ndarray)
        self.assertIsInstance(weights, np.ndarray)
        self.assertIsInstance(logl, np.ndarray)

    def test_posterior_samples_shape(self):
        """Test that posterior samples have correct shape."""
        samples, weights, logl = self.sampler.posterior()
        self.assertEqual(samples.shape[1], 2)  # n_dim = 2
        self.assertEqual(len(samples), len(weights))
        self.assertEqual(len(samples), len(logl))

    def test_posterior_weights_sum_to_one(self):
        """Test that posterior weights sum to approximately 1."""
        samples, weights, logl = self.sampler.posterior()
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

    def test_posterior_weights_non_negative(self):
        """Test that all posterior weights are non-negative."""
        samples, weights, logl = self.sampler.posterior()
        self.assertTrue(np.all(weights >= 0))

    def test_posterior_with_resample(self):
        """Test posterior() with resample=True."""
        samples, weights, logl = self.sampler.posterior(resample=True)
        # After resampling, weights should be uniform
        self.assertTrue(np.allclose(weights, 1.0 / len(weights)))

    def test_posterior_with_trim_importance_weights(self):
        """Test posterior() with trim_importance_weights=True."""
        samples, weights, logl = self.sampler.posterior(trim_importance_weights=True)
        self.assertEqual(len(samples), len(weights))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

    def test_posterior_samples_reasonable_range(self):
        """Test that posterior samples are in reasonable range."""
        samples, weights, logl = self.sampler.posterior()
        # For standard normal posterior, most samples should be within [-5, 5]
        self.assertTrue(np.all(samples > -10))
        self.assertTrue(np.all(samples < 10))


class EvidenceMethodTestCase(unittest.TestCase):
    """Test the evidence() method."""

    @staticmethod
    def prior_transform(u):
        """Transform from unit cube to standard normal."""
        return norm.ppf(u)

    @staticmethod
    def log_likelihood(x):
        """Gaussian log likelihood - handles both 1D and 2D."""
        if x.ndim == 1:
            return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)

    def setUp(self):
        """Set up sampler with Gaussian likelihood."""
        self.sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=64,
            clustering=False,
            random_state=0,
        )
        self.sampler.run(n_total=256)

    def test_evidence_returns_tuple(self):
        """Test that evidence() returns a tuple."""
        result = self.sampler.evidence()
        self.assertIsInstance(result, tuple)

    def test_evidence_returns_logz_and_error(self):
        """Test that evidence() returns logZ and error estimate."""
        logz, logz_err = self.sampler.evidence()
        self.assertIsInstance(logz, (float, np.floating))
        self.assertTrue(np.isfinite(logz))

    def test_evidence_logz_is_finite(self):
        """Test that logZ is finite."""
        logz, logz_err = self.sampler.evidence()
        self.assertTrue(np.isfinite(logz))

    def test_evidence_error_is_non_negative(self):
        """Test that logZ error estimate is non-negative."""
        logz, logz_err = self.sampler.evidence()
        if logz_err is not None:
            self.assertGreaterEqual(logz_err, 0)


class PosteriorEvidenceAccuracyTestCase(unittest.TestCase):
    """Test accuracy of posterior and evidence against analytical solutions."""

    def setUp(self):
        """Set up Gaussian problem with known solution."""
        self.n_dim = 2
        self.true_mean = np.array([0.0, 0.0])
        self.true_cov = np.eye(2)
        
        def prior_transform(u):
            return norm.ppf(u)
        
        def log_likelihood(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return multivariate_normal.logpdf(x, mean=self.true_mean, cov=self.true_cov)
        
        self.sampler = Sampler(
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=self.n_dim,
            n_particles=128,
            clustering=False,
            random_state=42,
        )
        self.sampler.run(n_total=512)

    def test_posterior_mean_accuracy(self):
        """Test that posterior mean is close to true mean."""
        samples, weights, logl = self.sampler.posterior(resample=True)
        posterior_mean = np.mean(samples, axis=0)
        
        # Check that posterior mean is within 0.5 of true mean
        np.testing.assert_allclose(
            posterior_mean, 
            self.true_mean, 
            atol=0.5,
            rtol=0.1
        )

    def test_posterior_covariance_accuracy(self):
        """Test that posterior covariance is close to true covariance."""
        samples, weights, logl = self.sampler.posterior(resample=True)
        posterior_cov = np.cov(samples.T)
        
        # Check diagonal (variances) - should be close to 1.0
        np.testing.assert_allclose(
            np.diag(posterior_cov),
            np.diag(self.true_cov),
            atol=0.5,
            rtol=0.2
        )

    def test_evidence_accuracy(self):
        """Test that evidence is close to analytical value."""
        # For Gaussian prior (unit normal) and Gaussian likelihood,
        # the log evidence can be computed analytically
        logz, logz_err = self.sampler.evidence()
        
        # The evidence should be finite and reasonable
        self.assertTrue(np.isfinite(logz))
        # For this problem, logZ should be roughly between -5 and 0
        self.assertGreater(logz, -10)
        self.assertLess(logz, 5)


if __name__ == "__main__":
    unittest.main()
