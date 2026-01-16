import unittest
import numpy as np

from tempest.student import fit_mvstud


class FitMvstudTestCase(unittest.TestCase):
    """Test cases for fit_mvstud function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_fit_gaussian_data(self):
        """Test fitting Student-t to Gaussian data."""
        # Generate Gaussian data
        n_samples = 200
        n_dim = 3
        true_mean = np.array([1.0, 2.0, 3.0])
        true_cov = np.eye(n_dim) * 0.5

        data = np.random.multivariate_normal(true_mean, true_cov, n_samples)

        # Fit Student-t distribution
        mu, Sigma, nu = fit_mvstud(data)

        # Check shapes
        self.assertEqual(mu.shape, (n_dim,))
        self.assertEqual(Sigma.shape, (n_dim, n_dim))

        # Check that results are finite or nu can be infinite for Gaussian-like data
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(Sigma)))
        self.assertTrue(np.isfinite(nu) or np.isinf(nu))

        # Mean should be close to true mean
        np.testing.assert_array_almost_equal(mu, true_mean, decimal=1)

        # Covariance should be close to true covariance
        np.testing.assert_array_almost_equal(Sigma, true_cov, decimal=1)

        # DOF should be high for Gaussian data (> 10) or infinite
        if np.isfinite(nu):
            self.assertGreater(nu, 10.0)

    def test_fit_2d_data(self):
        """Test fitting to 2D data."""
        n_samples = 150
        mean = np.array([0.5, -0.5])
        cov = np.array([[1.0, 0.3], [0.3, 0.8]])

        data = np.random.multivariate_normal(mean, cov, n_samples)

        mu, Sigma, nu = fit_mvstud(data)

        # Check shapes
        self.assertEqual(mu.shape, (2,))
        self.assertEqual(Sigma.shape, (2, 2))

        # Covariance should be symmetric
        np.testing.assert_array_almost_equal(Sigma, Sigma.T)

        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(Sigma)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_fit_1d_data(self):
        """Test fitting to 1D data."""
        n_samples = 100
        data = np.random.randn(n_samples, 1)

        mu, Sigma, nu = fit_mvstud(data)

        # Check shapes
        self.assertEqual(mu.shape, (1,))
        self.assertEqual(Sigma.shape, (1, 1))

        # Check that variance is positive
        self.assertGreater(Sigma[0, 0], 0)

    def test_convergence(self):
        """Test that algorithm converges within max iterations."""
        n_samples = 100
        n_dim = 2
        data = np.random.randn(n_samples, n_dim)

        # Fit with default tolerance
        mu, Sigma, nu = fit_mvstud(data, tolerance=1e-6, max_iter=100)

        # Should produce valid results
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(Sigma)))
        self.assertTrue(np.isfinite(nu) or np.isinf(nu))  # Can be infinite

    def test_tolerance_parameter(self):
        """Test effect of tolerance parameter."""
        n_samples = 100
        n_dim = 2
        data = np.random.randn(n_samples, n_dim)

        # Fit with loose tolerance (should converge faster)
        mu1, Sigma1, nu1 = fit_mvstud(data, tolerance=1e-2, max_iter=100)

        # Fit with tight tolerance
        mu2, Sigma2, nu2 = fit_mvstud(data, tolerance=1e-8, max_iter=100)

        # Both should produce valid results
        self.assertTrue(np.all(np.isfinite(mu1)))
        self.assertTrue(np.all(np.isfinite(mu2)))

    def test_max_iter_parameter(self):
        """Test max_iter parameter limits iterations."""
        n_samples = 100
        n_dim = 2
        data = np.random.randn(n_samples, n_dim)

        # Very few iterations
        mu, Sigma, nu = fit_mvstud(data, tolerance=1e-10, max_iter=5)

        # Should still produce valid results (even if not fully converged)
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(Sigma)))

    def test_heavy_tailed_data(self):
        """Test fitting to heavy-tailed data."""
        # Generate Student-t data with low DOF (heavy tails)
        n_samples = 200
        n_dim = 2
        dof = 3.0

        # Generate Student-t samples
        data = np.random.standard_t(dof, (n_samples, n_dim))

        mu, Sigma, nu = fit_mvstud(data)

        # Mean should be close to zero
        np.testing.assert_array_almost_equal(mu, np.zeros(n_dim), decimal=0)

        # DOF should be relatively low (< 20) for heavy-tailed data
        if np.isfinite(nu):
            self.assertLess(nu, 20.0)

    def test_constant_data(self):
        """Test behavior with constant (degenerate) data."""
        n_samples = 50
        n_dim = 2
        # All samples are the same
        data = np.ones((n_samples, n_dim)) * 5.0

        # This may raise LinAlgError due to singular covariance
        # which is expected behavior for degenerate data
        try:
            mu, Sigma, nu = fit_mvstud(data)

            # If it succeeds, mean should equal the constant value
            np.testing.assert_array_almost_equal(mu, np.array([5.0, 5.0]), decimal=1)

            # Results should be finite or handle gracefully
            self.assertTrue(np.all(np.isfinite(mu)))
        except np.linalg.LinAlgError:
            # Singular matrix is expected for constant data
            pass

    def test_outliers(self):
        """Test robustness to outliers."""
        n_samples = 100
        n_dim = 2

        # Generate mostly Gaussian data
        data = np.random.randn(n_samples, n_dim) * 0.5

        # Add a few outliers
        data[-5:] = np.random.randn(5, n_dim) * 10.0

        mu, Sigma, nu = fit_mvstud(data)

        # Should still fit reasonably well
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(Sigma)))

        # Mean should be roughly near zero (not pulled too much by outliers)
        self.assertTrue(np.all(np.abs(mu) < 2.0))

    def test_positive_definite_covariance(self):
        """Test that returned covariance is positive definite."""
        n_samples = 100
        n_dim = 3
        data = np.random.randn(n_samples, n_dim)

        mu, Sigma, nu = fit_mvstud(data)

        # Check symmetry
        np.testing.assert_array_almost_equal(Sigma, Sigma.T)

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(Sigma)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_correlated_data(self):
        """Test fitting to correlated data."""
        n_samples = 200
        n_dim = 3

        # Create correlation structure
        mean = np.zeros(n_dim)
        cov = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]])

        data = np.random.multivariate_normal(mean, cov, n_samples)

        mu, Sigma, nu = fit_mvstud(data)

        # Check that fitted covariance captures correlations
        # (off-diagonal elements should be non-zero)
        self.assertGreater(np.abs(Sigma[0, 1]), 0.3)
        self.assertGreater(np.abs(Sigma[1, 2]), 0.2)

    def test_different_scales(self):
        """Test with data on different scales."""
        n_samples = 150
        data = np.random.randn(n_samples, 3)
        # Scale each dimension differently
        data[:, 0] *= 0.1
        data[:, 1] *= 1.0
        data[:, 2] *= 10.0

        mu, Sigma, nu = fit_mvstud(data)

        # Diagonal elements should reflect different scales
        self.assertLess(Sigma[0, 0], Sigma[1, 1])
        self.assertLess(Sigma[1, 1], Sigma[2, 2])

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        n_samples = 100
        n_dim = 2

        np.random.seed(123)
        data = np.random.randn(n_samples, n_dim)

        # First fit
        mu1, Sigma1, nu1 = fit_mvstud(data)

        # Second fit with same data
        mu2, Sigma2, nu2 = fit_mvstud(data)

        # Results should be identical
        np.testing.assert_array_equal(mu1, mu2)
        np.testing.assert_array_equal(Sigma1, Sigma2)
        self.assertEqual(nu1, nu2)


if __name__ == "__main__":
    unittest.main()
