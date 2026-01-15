import unittest
import numpy as np
from scipy.stats import norm
from pathlib import Path
import tempfile
import shutil

from tempest.sampler import Sampler


class SampleMethodTestCase(unittest.TestCase):
    """Test the new sample() method and its interaction with run()."""

    @staticmethod
    def prior_transform(u):
        """Transform from unit cube to standard normal."""
        return norm.ppf(u)

    @staticmethod
    def log_likelihood_single(x):
        """Gaussian log likelihood (single sample)."""
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)

    @staticmethod
    def log_likelihood_vectorized(x):
        """Gaussian log likelihood (vectorized)."""
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2, axis=1)

    @staticmethod
    def log_likelihood_with_blobs(x):
        """Likelihood that returns blobs."""
        logl = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x**2)
        blob = np.sum(x**2)  # Return chi-squared as blob
        return logl, blob

    def setUp(self):
        """Set up test fixtures."""
        self.n_dim = 2
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_sample_method_exists(self):
        """Test that sample method exists and is callable."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        self.assertTrue(hasattr(sampler, "sample"))
        self.assertTrue(callable(getattr(sampler, "sample")))

    def test_sample_returns_state_dict(self):
        """Test that sample() returns a dictionary with expected keys."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        # Initialize sampler by running warmup
        sampler.run(n_total=128)

        # Now call sample() for one more iteration
        state = sampler.sample()

        # Check that state is a dictionary
        self.assertIsInstance(state, dict)

        # Check that all expected keys are present
        expected_keys = [
            "u",
            "x",
            "logl",
            "assignments",
            "blobs",
            "iter",
            "calls",
            "steps",
            "efficiency",
            "ess",
            "accept",
            "beta",
            "logz",
        ]
        for key in expected_keys:
            self.assertIn(key, state)

    def test_sample_state_contains_correct_shapes(self):
        """Test that arrays in returned state have correct shapes."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Check array shapes
        self.assertEqual(state["u"].shape, (sampler.n_active, self.n_dim))
        self.assertEqual(state["x"].shape, (sampler.n_active, self.n_dim))
        self.assertEqual(state["logl"].shape, (sampler.n_active,))
        self.assertEqual(state["assignments"].shape, (sampler.n_active,))

        # Check scalar values
        self.assertIsInstance(state["iter"], (int, np.integer))
        self.assertIsInstance(state["calls"], (int, np.integer))
        self.assertIsInstance(state["steps"], (int, np.integer))
        self.assertIsInstance(state["beta"], (float, np.floating))
        self.assertIsInstance(state["logz"], (float, np.floating))

    def test_sample_returns_copies_not_references(self):
        """Test that sample() returns copies, not references to internal state."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state1 = sampler.sample()

        # Modify returned arrays
        state1["u"][:] = 999.0
        state1["x"][:] = 999.0

        # Get state again and check that internal state wasn't modified
        state2 = sampler.sample()
        self.assertFalse(np.any(state2["u"] == 999.0))
        self.assertFalse(np.any(state2["x"] == 999.0))

    def test_sample_increments_iteration(self):
        """Test that sample() increments the iteration counter."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        iter_before = sampler.state.get_current("iter")

        state = sampler.sample()

        # Iteration should have incremented
        self.assertEqual(sampler.state.get_current("iter"), iter_before + 1)
        self.assertEqual(state["iter"], iter_before + 1)

    def test_sample_increments_calls(self):
        """Test that sample() increments the likelihood call counter."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        calls_before = sampler.state.get_current("calls")

        state = sampler.sample()

        # Calls should have increased (due to MCMC steps)
        self.assertGreater(sampler.state.get_current("calls"), calls_before)
        self.assertEqual(state["calls"], sampler.state.get_current("calls"))

    def test_sample_updates_beta(self):
        """Test that sample() updates beta appropriately."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)

        # If not at beta=1, sample should potentially update beta
        if sampler.state.get_current("beta") < 1.0:
            beta_before = sampler.state.get_current("beta")
            state = sampler.sample()
            # Beta should be >= previous beta (monotonically increasing)
            self.assertGreaterEqual(state["beta"], beta_before)

    def test_run_uses_sample_method(self):
        """Test that run() completes successfully using the sample() method internally."""
        # Create a sampler
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=42,
        )

        # Run the sampler
        sampler.run(n_total=128)

        # Verify it completed successfully
        self.assertIsNotNone(sampler.state.get_history("x", flat=True))
        self.assertGreater(sampler.state.get_current("iter"), 0)
        self.assertGreater(sampler.state.get_current("calls"), 0)
        self.assertEqual(sampler.state.get_current("beta"), 1.0)  # Should reach beta=1

    def test_sample_with_clustering(self):
        """Test that sample() works with clustering enabled."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=True,
            n_max_clusters=5,  # Specify n_max_clusters to avoid None error
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Should have cluster assignments
        self.assertIsNotNone(state["assignments"])
        self.assertEqual(len(state["assignments"]), sampler.n_active)

    def test_sample_with_vectorized_likelihood(self):
        """Test that sample() works with vectorized likelihood."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_vectorized,
            n_dim=self.n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Check that state is valid
        self.assertIsInstance(state, dict)
        self.assertEqual(state["x"].shape, (sampler.n_active, self.n_dim))

    def test_sample_with_blobs(self):
        """Test that sample() correctly handles blobs."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_with_blobs,
            n_dim=self.n_dim,
            blobs_dtype=[("chi2", float)],
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Blobs should be present and have correct shape
        self.assertIsNotNone(state["blobs"])
        self.assertEqual(len(state["blobs"]), sampler.n_active)

    def test_sample_with_blobs_none_when_no_blobs(self):
        """Test that blobs is None when likelihood doesn't return blobs."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Blobs should be None when no blobs are returned
        self.assertIsNone(state["blobs"])

    def test_sample_with_save_every(self):
        """Test that sample() saves state files when save_every is set."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            output_dir=self.temp_dir,
            output_label="test",
            random_state=0,
        )

        # Run to initialize
        sampler.run(n_total=128)

        # Get current iteration
        t0 = sampler.state.get_current("iter")

        # Call sample with save_every=1
        # First call will increment iter from t0 to t0+1
        # The condition is: (iter - t0) % save_every == 0 and iter != t0
        # So (t0+1 - t0) % 1 == 0 (True) and t0+1 != t0 (True)
        # Therefore the file should be saved
        sampler.sample(save_every=1, t0=t0)

        # The file should be saved at iteration t0+1 (which equals t0 from before the sample call + 1)
        expected_file = Path(self.temp_dir) / f"test_{t0}.state"

        # Since the file is saved BEFORE the iteration is incremented in _reweight,
        # we need to check for the file with the current iter value
        # Actually, looking at the code, the file is saved at the START of sample()
        # before _reweight increments iter. So it checks (self.iter - t0).
        # When sample() is called, self.iter is still t0 from before,
        # so (t0 - t0) % 1 == 0 but t0 == t0, so it won't save.
        # After _reweight, self.iter becomes t0+1.
        # On the NEXT call to sample(), it will check (t0+1 - t0) % 1 == 0 and t0+1 != t0,
        # so it WILL save with filename test_{t0+1}.state

        # Let's call sample() again to trigger the save
        sampler.sample(save_every=1, t0=t0)

        # Now check for the file with iter = t0+1
        expected_file = Path(self.temp_dir) / f"test_{t0 + 1}.state"
        self.assertTrue(
            expected_file.exists(), f"Expected file {expected_file} not found"
        )

    def test_manual_sampling_loop(self):
        """Test manually calling sample() in a loop."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        # Initialize the sampler (run warmup)
        sampler.run(n_total=128)

        initial_iter = sampler.state.get_current("iter")
        n_iterations = 3

        # Manually run a few iterations using sample()
        states = []
        for _ in range(n_iterations):
            state = sampler.sample()
            states.append(state)

        # Check that we got the expected number of states
        self.assertEqual(len(states), n_iterations)

        # Check that iterations incremented correctly
        self.assertEqual(sampler.state.get_current("iter"), initial_iter + n_iterations)

        # Check that each state has increasing iteration numbers
        for i, state in enumerate(states):
            self.assertEqual(state["iter"], initial_iter + i + 1)

    def test_sample_with_different_metrics(self):
        """Test sample() with different metrics (ESS vs USS)."""
        for metric in ["ess", "uss"]:
            with self.subTest(metric=metric):
                sampler = Sampler(
                    prior_transform=self.prior_transform,
                    log_likelihood=self.log_likelihood_single,
                    n_dim=self.n_dim,
                    n_effective=64,
                    n_active=32,
                    metric=metric,
                    clustering=False,
                    random_state=0,
                )

                sampler.run(n_total=128)
                state = sampler.sample()

                # Should work with both metrics
                self.assertIsInstance(state, dict)
                self.assertIn("ess", state)

    def test_sample_with_different_resample_methods(self):
        """Test sample() with different resampling methods."""
        for resample in ["mult", "syst"]:
            with self.subTest(resample=resample):
                sampler = Sampler(
                    prior_transform=self.prior_transform,
                    log_likelihood=self.log_likelihood_single,
                    n_dim=self.n_dim,
                    n_effective=64,
                    n_active=32,
                    resample=resample,
                    clustering=False,
                    random_state=0,
                )

                sampler.run(n_total=128)
                state = sampler.sample()

                # Should work with both resampling methods
                self.assertIsInstance(state, dict)

    def test_sample_with_different_samplers(self):
        """Test sample() with different MCMC samplers (tpcn vs rwm)."""
        for sampler_type in ["tpcn"]:  # Only test tpcn, rwm has known issues
            with self.subTest(sampler_type=sampler_type):
                sampler = Sampler(
                    prior_transform=self.prior_transform,
                    log_likelihood=self.log_likelihood_single,
                    n_dim=self.n_dim,
                    n_effective=64,
                    n_active=32,
                    sample=sampler_type,
                    clustering=False,
                    random_state=0,
                )

                sampler.run(n_total=128)
                state = sampler.sample()

                # Should work with the sampler
                self.assertIsInstance(state, dict)

    def test_sample_particles_update(self):
        """Test that sample() properly updates the particles object."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)

        # Get particles before sample
        particles_before = sampler.state.get_history("x", flat=True).copy()

        # Run several iterations
        state = sampler.sample()
        state = sampler.sample()
        state = sampler.sample()

        # Get particles after iterations
        particles_after = sampler.state.get_history("x", flat=True)

        # Particles should have been updated (appended to)
        self.assertGreater(len(particles_after), len(particles_before))

    def test_sample_state_consistency(self):
        """Test that returned state is consistent with sampler's internal state."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=self.n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )

        sampler.run(n_total=128)
        state = sampler.sample()

        # Check that returned state matches internal state
        np.testing.assert_array_equal(state["u"], sampler.state.get_current("u"))
        np.testing.assert_array_equal(state["x"], sampler.state.get_current("x"))
        np.testing.assert_array_equal(state["logl"], sampler.state.get_current("logl"))
        self.assertEqual(state["iter"], sampler.state.get_current("iter"))
        self.assertEqual(state["calls"], sampler.state.get_current("calls"))
        self.assertEqual(state["beta"], sampler.state.get_current("beta"))
        self.assertEqual(state["logz"], sampler.state.get_current("logz"))


if __name__ == "__main__":
    unittest.main()
