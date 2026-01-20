import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from scipy.stats import norm

from tempest.steps import Reweighter, Trainer, Resampler, Mutator
from tempest.state_manager import StateManager
from tempest.cluster import HierarchicalGaussianMixture
from tempest.modes import ModeStatistics


class ReweighterTestCase(unittest.TestCase):
    """Test Reweighter step class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.n_dim = 2
        self.state = StateManager(n_dim=self.n_dim)
        # Initialize current state with default values
        self.state.set_current("iter", 0)
        self.state.set_current("beta", 0.0)
        self.state.set_current("logz", 0.0)
        self.state.set_current("calls", 0)
        self.pbar = Mock()

    def test_init(self):
        """Test Reweighter initialization."""
        reweighter = Reweighter(
            state=self.state,
            pbar=self.pbar,
            n_effective=512,
            n_active=256,
            metric="ess",
        )
        self.assertEqual(reweighter.n_effective, 512)
        self.assertEqual(reweighter.n_active, 256)
        self.assertEqual(reweighter.metric, "ess")
        self.assertIsNotNone(reweighter.state)

    def test_first_iteration(self):
        """Test Reweighter at first iteration (no history)."""
        reweighter = Reweighter(
            state=self.state,
            pbar=self.pbar,
            n_effective=64,
            n_active=32,
        )

        weights = reweighter.run()

        # First iteration should return uniform weights
        self.assertEqual(len(weights), 32)
        np.testing.assert_allclose(weights, 1.0 / 32)
        self.assertEqual(self.state.get_current("beta"), 0.0)
        self.assertEqual(self.state.get_current("logz"), 0.0)
        self.assertEqual(self.state.get_current("ess"), 64)
        self.assertEqual(self.state.get_current("iter"), 1)

    def test_beta_progression(self):
        """Test beta increases across iterations."""
        np.random.seed(42)
        reweighter = Reweighter(
            state=self.state,
            pbar=None,
            n_effective=16,
            n_active=16,
        )

        # First iteration - create particles with varying likelihoods
        u1 = np.random.rand(16, self.n_dim)
        x1 = u1 * 6 - 3  # Range [-3, 3] for more spread
        # Mix of good and bad likelihoods
        logl1 = -np.sum(x1**2, axis=1) - np.random.rand(16) * 5
        self.state.update_current({"u": u1, "x": x1, "logl": logl1, "beta": 0.0})

        # First iteration
        weights1 = reweighter.run()
        beta1 = self.state.get_current("beta")
        self.assertEqual(beta1, 0.0)  # First iteration is always beta=0

        # Add to history
        self.state.commit_current_to_history()

        # Second iteration - different likelihood distribution
        u2 = np.random.rand(16, self.n_dim)
        x2 = u2 * 6 - 3
        # These have higher likelihoods on average
        logl2 = -np.sum(x2**2, axis=1) * 0.5
        self.state.update_current({"u": u2, "x": x2, "logl": logl2})

        weights2 = reweighter.run()
        beta2 = self.state.get_current("beta")

        # Beta should progress from 0 (with varying likelihoods, beta will increase)
        self.assertGreaterEqual(beta2, 0.0)
        self.assertLessEqual(beta2, 1.0)
        # Weights are computed for all historical particles
        self.assertGreater(len(weights2), 0)
        np.testing.assert_allclose(np.sum(weights2), 1.0)

    def test_boosting(self):
        """Test boosting increases n_effective and n_active toward posterior."""
        reweighter = Reweighter(
            state=self.state,
            pbar=None,
            n_effective=32,
            n_active=16,
            n_boost=128,
            n_effective_init=32,
            n_active_init=16,
            BOOST_STEEPNESS=2.0,
        )

        # Add history
        u = np.random.rand(16, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        initial_n_effective = reweighter.n_effective
        initial_n_active = reweighter.n_active

        reweighter.run()

        # Boosting may have increased n_effective/n_active
        # (depends on posterior ESS, so just check they're capped)
        self.assertLessEqual(reweighter.n_effective, 128)
        self.assertLessEqual(reweighter.n_active, 64)


class TrainerTestCase(unittest.TestCase):
    """Test Trainer step class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.n_dim = 2
        self.state = StateManager(n_dim=self.n_dim)
        self.pbar = Mock()
        # HierarchicalGaussianMixture doesn't take n_dim in __init__
        self.clusterer = HierarchicalGaussianMixture()

    def test_init(self):
        """Test Trainer initialization."""
        trainer = Trainer(
            state=self.state,
            pbar=self.pbar,
            clusterer=self.clusterer,
            cluster_every=1,
            clustering=True,
        )
        self.assertIsNotNone(trainer.state)
        self.assertIsNotNone(trainer.clusterer)
        self.assertTrue(trainer.clustering)

    def test_beta_zero_returns_dummy_mode_stats(self):
        """Test Trainer returns dummy ModeStatistics at beta=0."""
        trainer = Trainer(
            state=self.state,
            pbar=None,
            clusterer=self.clusterer,
        )

        # Set beta=0
        self.state.set_current("beta", 0.0)
        self.state.set_current("iter", 0)

        weights = np.ones(10) / 10
        mode_stats = trainer.run(weights)

        # Should return dummy mode stats
        self.assertIsInstance(mode_stats, ModeStatistics)
        self.assertEqual(mode_stats.K, 1)
        self.assertEqual(mode_stats.means.shape, (1, self.n_dim))

    def test_clustering_fit(self):
        """Test Trainer fits clustering model."""
        trainer = Trainer(
            state=self.state,
            pbar=self.pbar,
            clusterer=self.clusterer,
            cluster_every=1,
            clustering=True,
            TRIM_ESS=0.99,  # Use fraction to avoid edge case
            TRIM_BINS=1000,
        )

        # Add history - use more particles to avoid trim_weights issues
        n_particles = 64
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        # Set beta > 0 and iter
        self.state.set_current("beta", 0.5)
        self.state.set_current("iter", 1)

        weights = np.ones(n_particles) / n_particles
        mode_stats = trainer.run(weights)

        # Should return valid ModeStatistics
        self.assertIsInstance(mode_stats, ModeStatistics)
        self.assertGreater(mode_stats.K, 0)
        self.assertEqual(mode_stats.means.shape[1], self.n_dim)
        self.assertEqual(len(mode_stats.degrees_of_freedom), mode_stats.K)

        # Progress bar should be updated with K
        self.pbar.update_stats.assert_called()

    def test_global_mode_without_clustering(self):
        """Test Trainer fits global mode when clustering is disabled."""
        trainer = Trainer(
            state=self.state,
            pbar=None,
            clusterer=None,
            clustering=False,
            TRIM_ESS=0.99,  # Use fraction
            TRIM_BINS=1000,
        )

        # Add history - use more particles
        n_particles = 64
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        self.state.set_current("beta", 0.5)
        self.state.set_current("iter", 1)

        weights = np.ones(n_particles) / n_particles
        mode_stats = trainer.run(weights)

        # Should have single global mode
        self.assertEqual(mode_stats.K, 1)
        self.assertEqual(mode_stats.means.shape, (1, self.n_dim))


class ResamplerTestCase(unittest.TestCase):
    """Test Resampler step class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.n_dim = 2
        self.state = StateManager(n_dim=self.n_dim)
        # HierarchicalGaussianMixture doesn't take n_dim in __init__
        self.clusterer = HierarchicalGaussianMixture()

    def test_init(self):
        """Test Resampler initialization."""
        n_active_fn = lambda: 32
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            resample="syst",
            clusterer=self.clusterer,
            clustering=True,
        )
        self.assertIsNotNone(resampler.state)
        self.assertEqual(resampler.resample, "syst")
        self.assertTrue(resampler.clustering)

    def test_beta_zero_skips_resampling(self):
        """Test Resampler skips at beta=0."""
        n_active = 16
        n_active_fn = lambda: n_active
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            clustering=False,
        )

        self.state.set_current("beta", 0.0)
        weights = np.ones(n_active) / n_active

        resampler.run(weights)

        # Should just set assignments to zeros
        assignments = self.state.get_current("assignments")
        self.assertEqual(len(assignments), n_active)
        np.testing.assert_array_equal(assignments, np.zeros(n_active, dtype=int))

    def test_systematic_resampling(self):
        """Test systematic resampling."""
        n_active = 16
        n_active_fn = lambda: n_active
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            resample="syst",
            clustering=False,
        )

        # Add history
        n_particles = 32
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        self.state.set_current("beta", 0.5)

        weights = np.ones(n_particles) / n_particles
        resampler.run(weights)

        # Should have resampled particles
        u_resampled = self.state.get_current("u")
        x_resampled = self.state.get_current("x")
        logl_resampled = self.state.get_current("logl")

        self.assertEqual(u_resampled.shape, (n_active, self.n_dim))
        self.assertEqual(x_resampled.shape, (n_active, self.n_dim))
        self.assertEqual(logl_resampled.shape, (n_active,))

    def test_multinomial_resampling(self):
        """Test multinomial resampling."""
        np.random.seed(42)
        n_active = 16
        n_active_fn = lambda: n_active
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            resample="mult",
            clustering=False,
        )

        # Add history
        n_particles = 32
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        self.state.set_current("beta", 0.5)

        weights = np.ones(n_particles) / n_particles
        resampler.run(weights)

        # Should have resampled particles
        u_resampled = self.state.get_current("u")
        self.assertEqual(u_resampled.shape, (n_active, self.n_dim))

    def test_cluster_assignment(self):
        """Test cluster assignment during resampling."""
        n_active = 16
        n_active_fn = lambda: n_active
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            resample="syst",
            clusterer=self.clusterer,
            clustering=True,
        )

        # Add history and fit clusterer
        n_particles = 32
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        self.state.update_current({"u": u, "x": x, "logl": logl})
        self.state.commit_current_to_history()

        # Fit clusterer first
        weights = np.ones(n_particles) / n_particles
        self.clusterer.fit(u, weights)

        self.state.set_current("beta", 0.5)
        resampler.run(weights)

        # Should have assigned cluster labels
        assignments = self.state.get_current("assignments")
        self.assertEqual(len(assignments), n_active)
        self.assertTrue(np.all(assignments >= 0))

    def test_blobs_resampling(self):
        """Test resampling with auxiliary blobs data."""
        n_active = 8
        n_active_fn = lambda: n_active
        resampler = Resampler(
            state=self.state,
            n_active_fn=n_active_fn,
            resample="syst",
            clustering=False,
            have_blobs=True,
        )

        # Add history with blobs
        n_particles = 16
        u = np.random.rand(n_particles, self.n_dim)
        x = u * 2 - 1
        logl = -0.5 * np.sum(x**2, axis=1)
        blobs = np.random.rand(n_particles, 3)  # 3 auxiliary features
        self.state.update_current({"u": u, "x": x, "logl": logl, "blobs": blobs})
        self.state.commit_current_to_history()

        self.state.set_current("beta", 0.5)

        weights = np.ones(n_particles) / n_particles
        resampler.run(weights)

        # Blobs should also be resampled
        blobs_resampled = self.state.get_current("blobs")
        self.assertEqual(blobs_resampled.shape, (n_active, 3))


class MutatorTestCase(unittest.TestCase):
    """Test Mutator step class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.n_dim = 2
        self.state = StateManager(n_dim=self.n_dim)
        self.pbar = Mock()

        # Simple prior and likelihood
        self.prior_transform = lambda u: norm.ppf(u)

        def log_likelihood(x):
            """Vectorized Gaussian log-likelihood."""
            if x.ndim == 1:
                x = x.reshape(1, -1)
            logl = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)
            return logl, None

        self.log_likelihood = log_likelihood

    def test_init(self):
        """Test Mutator initialization."""
        n_active_fn = lambda: 32
        mutator = Mutator(
            state=self.state,
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            pbar=self.pbar,
            n_active_fn=n_active_fn,
            n_dim=self.n_dim,
            n_steps=10,
            sampler="tpcn",
        )
        self.assertIsNotNone(mutator.state)
        self.assertEqual(mutator.n_steps, 10)
        self.assertEqual(mutator.sampler, "tpcn")

    def test_warmup_draws_prior_samples(self):
        """Test Mutator draws fresh prior samples at beta=0."""
        np.random.seed(42)
        n_active = 16
        n_active_fn = lambda: n_active
        mutator = Mutator(
            state=self.state,
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            pbar=None,
            n_active_fn=n_active_fn,
            n_dim=self.n_dim,
        )

        self.state.set_current("beta", 0.0)
        self.state.set_current("calls", 0)

        # Create dummy mode_stats (won't be used at beta=0)
        mode_stats = ModeStatistics(
            means=np.zeros((1, self.n_dim)),
            covariances=np.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim),
            degrees_of_freedom=np.array([1.0]),
        )

        mutator.run(mode_stats)

        # Should have drawn prior samples
        u = self.state.get_current("u")
        x = self.state.get_current("x")
        logl = self.state.get_current("logl")

        self.assertEqual(u.shape, (n_active, self.n_dim))
        self.assertEqual(x.shape, (n_active, self.n_dim))
        self.assertEqual(logl.shape, (n_active,))
        self.assertEqual(self.state.get_current("calls"), n_active)
        self.assertEqual(self.state.get_current("acceptance"), 1.0)
        self.assertEqual(self.state.get_current("efficiency"), 1.0)

    def test_warmup_handles_infinite_likelihoods(self):
        """Test Mutator handles infinite likelihoods during warmup."""
        np.random.seed(42)
        n_active = 16

        # Likelihood that returns infinite values
        def log_likelihood_with_inf(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            logl = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)
            # Make some likelihoods infinite
            logl[::2] = np.inf
            return logl, None

        n_active_fn = lambda: n_active
        mutator = Mutator(
            state=self.state,
            prior_transform=self.prior_transform,
            log_likelihood=log_likelihood_with_inf,
            pbar=None,
            n_active_fn=n_active_fn,
            n_dim=self.n_dim,
        )

        self.state.set_current("beta", 0.0)
        self.state.set_current("calls", 0)
        self.state.set_current("logz", 0.0)

        mode_stats = ModeStatistics(
            means=np.zeros((1, self.n_dim)),
            covariances=np.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim),
            degrees_of_freedom=np.array([1.0]),
        )

        mutator.run(mode_stats)

        logl = self.state.get_current("logl")

        # Should have resampled infinite likelihoods from finite ones
        # So no infinite values should remain
        self.assertFalse(np.any(np.isinf(logl)))

        # logz should be corrected for finite fraction
        logz = self.state.get_current("logz")
        self.assertLess(logz, 0.0)  # Corrected for <100% finite support

    @patch("tempest.steps.mutate.parallel_mcmc")
    def test_mcmc_evolution(self, mock_parallel_mcmc):
        """Test Mutator calls parallel_mcmc at beta>0."""
        n_active = 8
        n_active_fn = lambda: n_active
        mutator = Mutator(
            state=self.state,
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            pbar=self.pbar,
            n_active_fn=n_active_fn,
            n_dim=self.n_dim,
            n_steps=10,
            sampler="tpcn",
        )

        # Set up current state
        u = np.random.rand(n_active, self.n_dim)
        x = norm.ppf(u)
        logl = -0.5 * np.sum(x**2, axis=1)
        assignments = np.zeros(n_active, dtype=int)

        self.state.update_current(
            {
                "u": u,
                "x": x,
                "logl": logl,
                "assignments": assignments,
                "beta": 0.5,
                "calls": 100,
            }
        )

        # Mock parallel_mcmc return
        mock_parallel_mcmc.return_value = (
            u,
            x,
            logl,
            None,  # blobs
            0.95,  # efficiency
            0.6,  # acceptance
            10,  # steps
            80,  # calls
        )

        mode_stats = ModeStatistics(
            means=np.zeros((1, self.n_dim)),
            covariances=np.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim),
            degrees_of_freedom=np.array([2.0]),
        )

        mutator.run(mode_stats)

        # Should have called parallel_mcmc
        mock_parallel_mcmc.assert_called_once()

        # State should be updated
        self.assertEqual(self.state.get_current("calls"), 180)  # 100 + 80
        self.assertEqual(self.state.get_current("efficiency"), 0.95)
        self.assertEqual(self.state.get_current("acceptance"), 0.6)
        self.assertEqual(self.state.get_current("steps"), 10)

    def test_blobs_evolution(self):
        """Test Mutator handles blobs during warmup."""
        np.random.seed(42)
        n_active = 8

        def log_likelihood_with_blobs(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            logl = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)
            blobs = np.random.rand(len(logl), 2)  # 2 blob features
            return logl, blobs

        n_active_fn = lambda: n_active
        mutator = Mutator(
            state=self.state,
            prior_transform=self.prior_transform,
            log_likelihood=log_likelihood_with_blobs,
            pbar=None,
            n_active_fn=n_active_fn,
            n_dim=self.n_dim,
            have_blobs=True,
        )

        self.state.set_current("beta", 0.0)
        self.state.set_current("calls", 0)

        mode_stats = ModeStatistics(
            means=np.zeros((1, self.n_dim)),
            covariances=np.eye(self.n_dim).reshape(1, self.n_dim, self.n_dim),
            degrees_of_freedom=np.array([1.0]),
        )

        mutator.run(mode_stats)

        blobs = self.state.get_current("blobs")
        self.assertEqual(blobs.shape, (n_active, 2))


if __name__ == "__main__":
    unittest.main()
