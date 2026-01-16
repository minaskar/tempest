import unittest
import numpy as np

from tempest.modes import ModeStatistics


class ModeStatisticsTestCase(unittest.TestCase):
    """Test cases for ModeStatistics class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_dim = 3

        # Create simple mode statistics
        self.mean = np.array([0.5, 0.5, 0.5])
        self.cov = np.eye(self.n_dim) * 0.1
        self.dof = 5.0

    def test_init_single_mode(self):
        """Test initialization with a single mode."""
        stats = ModeStatistics(
            means=self.mean, covariances=self.cov, degrees_of_freedom=self.dof
        )

        # Check properties
        self.assertEqual(stats.K, 1)
        self.assertEqual(stats.n_dim, self.n_dim)

        # Check shapes
        self.assertEqual(stats.means.shape, (1, self.n_dim))
        self.assertEqual(stats.covariances.shape, (1, self.n_dim, self.n_dim))
        self.assertEqual(stats.degrees_of_freedom.shape, (1,))

        # Check precomputed quantities
        self.assertEqual(stats.inv_covariances.shape, (1, self.n_dim, self.n_dim))
        self.assertEqual(stats.chol_covariances.shape, (1, self.n_dim, self.n_dim))

    def test_init_multiple_modes(self):
        """Test initialization with multiple modes."""
        K = 3
        means = np.random.rand(K, self.n_dim)
        covs = np.array([np.eye(self.n_dim) * (i + 1) * 0.1 for i in range(K)])
        dofs = np.array([5.0, 10.0, 20.0])

        stats = ModeStatistics(means=means, covariances=covs, degrees_of_freedom=dofs)

        # Check properties
        self.assertEqual(stats.K, K)
        self.assertEqual(stats.n_dim, self.n_dim)

        # Check shapes
        self.assertEqual(stats.means.shape, (K, self.n_dim))
        self.assertEqual(stats.covariances.shape, (K, self.n_dim, self.n_dim))
        self.assertEqual(stats.degrees_of_freedom.shape, (K,))

    def test_init_shape_validation(self):
        """Test that initialization validates shapes correctly."""
        # Incompatible covariance shape
        with self.assertRaises(ValueError):
            ModeStatistics(
                means=np.array([[0.5, 0.5]]),
                covariances=np.eye(3),  # Wrong dimension
                degrees_of_freedom=np.array([5.0]),
            )

        # Incompatible DOF shape
        with self.assertRaises(ValueError):
            ModeStatistics(
                means=np.array([[0.5, 0.5], [0.3, 0.3]]),
                covariances=np.array([np.eye(2), np.eye(2)]),
                degrees_of_freedom=np.array([5.0]),  # Should be length 2
            )

    def test_precomputed_quantities(self):
        """Test that inverse and Cholesky decomposition are computed correctly."""
        stats = ModeStatistics(
            means=self.mean, covariances=self.cov, degrees_of_freedom=self.dof
        )

        # Verify inverse covariance
        inv_cov = stats.inv_covariances[0]
        product = stats.covariances[0] @ inv_cov
        np.testing.assert_array_almost_equal(product, np.eye(self.n_dim))

        # Verify Cholesky decomposition
        chol = stats.chol_covariances[0]
        reconstructed = chol @ chol.T
        np.testing.assert_array_almost_equal(reconstructed, stats.covariances[0])

    def test_from_global(self):
        """Test creating ModeStatistics from global particles."""
        n_particles = 100
        u = np.random.rand(n_particles, self.n_dim)
        weights = np.ones(n_particles)

        stats = ModeStatistics.from_global(u, weights, resample_factor=4)

        # Should create single mode
        self.assertEqual(stats.K, 1)
        self.assertEqual(stats.n_dim, self.n_dim)

        # DOF should be finite and positive
        self.assertTrue(np.isfinite(stats.degrees_of_freedom[0]))
        self.assertGreater(stats.degrees_of_freedom[0], 0)

        # Mean should be roughly centered
        self.assertTrue(np.all(stats.means[0] >= 0))
        self.assertTrue(np.all(stats.means[0] <= 1))

    def test_from_global_with_weights(self):
        """Test from_global with non-uniform weights."""
        n_particles = 100
        u = np.random.rand(n_particles, self.n_dim)
        weights = np.random.rand(n_particles)

        stats = ModeStatistics.from_global(u, weights, resample_factor=4)

        # Should still create single mode
        self.assertEqual(stats.K, 1)
        self.assertEqual(stats.n_dim, self.n_dim)

    def test_from_global_mismatched_shapes(self):
        """Test that from_global validates input shapes."""
        u = np.random.rand(100, self.n_dim)
        weights = np.ones(50)  # Wrong size

        with self.assertRaises(ValueError):
            ModeStatistics.from_global(u, weights)

    def test_from_particles(self):
        """Test creating ModeStatistics from clustered particles."""
        n_particles = 200

        # Create two clusters
        u1 = np.random.rand(n_particles // 2, self.n_dim) * 0.5
        u2 = np.random.rand(n_particles // 2, self.n_dim) * 0.5 + 0.5
        u = np.vstack([u1, u2])

        weights = np.ones(n_particles)
        labels = np.array([0] * (n_particles // 2) + [1] * (n_particles // 2))

        stats = ModeStatistics.from_particles(u, weights, labels, resample_factor=4)

        # Should create two modes
        self.assertEqual(stats.K, 2)
        self.assertEqual(stats.n_dim, self.n_dim)

        # Both DOFs should be finite and positive
        self.assertTrue(np.all(np.isfinite(stats.degrees_of_freedom)))
        self.assertTrue(np.all(stats.degrees_of_freedom > 0))

    def test_from_particles_with_weights(self):
        """Test from_particles with non-uniform weights."""
        n_particles = 200
        u = np.random.rand(n_particles, self.n_dim)
        weights = np.random.rand(n_particles)
        labels = np.random.randint(0, 3, n_particles)

        stats = ModeStatistics.from_particles(u, weights, labels, resample_factor=4)

        # Should create 3 modes
        self.assertEqual(stats.K, 3)
        self.assertEqual(stats.n_dim, self.n_dim)

    def test_from_particles_mismatched_shapes(self):
        """Test that from_particles validates input shapes."""
        u = np.random.rand(100, self.n_dim)
        weights = np.ones(100)
        labels = np.zeros(50)  # Wrong size

        with self.assertRaises(ValueError):
            ModeStatistics.from_particles(u, weights, labels)

    def test_dof_fallback(self):
        """Test DOF fallback for non-finite values."""
        # Create degenerate data that might cause infinite DOF
        n_particles = 100
        u = np.ones((n_particles, self.n_dim)) * 0.5  # All same point
        u += np.random.randn(n_particles, self.n_dim) * 1e-10  # Tiny noise
        weights = np.ones(n_particles)

        fallback_dof = 2.5
        stats = ModeStatistics.from_global(
            u, weights, dof_fallback=fallback_dof, resample_factor=4
        )

        # DOF should be finite (either fitted or fallback)
        self.assertTrue(np.isfinite(stats.degrees_of_freedom[0]))
        self.assertGreater(stats.degrees_of_freedom[0], 0)

    def test_repr(self):
        """Test string representation."""
        stats = ModeStatistics(
            means=self.mean, covariances=self.cov, degrees_of_freedom=self.dof
        )

        repr_str = repr(stats)

        # Should contain key information
        self.assertIn("ModeStatistics", repr_str)
        self.assertIn("K=1", repr_str)
        self.assertIn(f"n_dim={self.n_dim}", repr_str)

    def test_auto_reshape_single_mode(self):
        """Test automatic reshaping for single mode case."""
        # Input as 1D arrays (should be reshaped to 2D)
        mean_1d = np.array([0.5, 0.5, 0.5])
        cov_2d = np.eye(3) * 0.1
        dof_scalar = 5.0

        stats = ModeStatistics(
            means=mean_1d, covariances=cov_2d, degrees_of_freedom=dof_scalar
        )

        # Should be reshaped to proper dimensions
        self.assertEqual(stats.means.shape, (1, 3))
        self.assertEqual(stats.covariances.shape, (1, 3, 3))
        self.assertEqual(stats.degrees_of_freedom.shape, (1,))


if __name__ == "__main__":
    unittest.main()
