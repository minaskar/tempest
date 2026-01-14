import unittest
from pathlib import Path
import numpy as np

from tempest.state_manager import StateManager, CURRENT_STATE_KEYS, HISTORY_STATE_KEYS
from tempest.particles import Particles


class StateManagerBasicTestCase(unittest.TestCase):
    """Test basic StateManager operations."""

    def setUp(self):
        self.n_dim = 3
        self.n_particles = 10
        self.state = StateManager(self.n_dim)

    def test_initialization(self):
        """Test StateManager initialization."""
        self.assertEqual(self.state.n_dim, self.n_dim)
        self.assertIsNotNone(self.state._current)
        self.assertIsNotNone(self.state._history)
        self.assertIsNone(self.state._results_dict)

        # Check all current keys are None
        for key in CURRENT_STATE_KEYS:
            self.assertIsNone(self.state._current[key])

        # Check all history lists are empty
        for key in HISTORY_STATE_KEYS:
            self.assertEqual(len(self.state._history[key]), 0)

    def test_get_current_all(self):
        """Test getting all current state."""
        current = self.state.get_current()
        self.assertIsInstance(current, dict)
        self.assertEqual(len(current), len(CURRENT_STATE_KEYS))
        for key in CURRENT_STATE_KEYS:
            self.assertIsNone(current[key])

    def test_get_current_single(self):
        """Test getting single current state value."""
        self.assertIsNone(self.state.get_current("beta"))
        self.assertIsNone(self.state.get_current("iter"))

    def test_set_current(self):
        """Test setting single current state value."""
        self.state.set_current("beta", 0.5)
        self.assertAlmostEqual(self.state.get_current("beta"), 0.5)
        self.state.set_current("iter", 5)
        self.assertEqual(self.state.get_current("iter"), 5)

    def test_update_current(self):
        """Test updating multiple current state values."""
        data = {"beta": 0.5, "iter": 10, "logz": -100.0}
        self.state.update_current(data)
        for key, value in data.items():
            retrieved = self.state.get_current(key)
            if isinstance(value, float):
                self.assertAlmostEqual(retrieved, value)
            else:
                self.assertEqual(retrieved, value)

    def test_set_current_invalid_key(self):
        """Test setting invalid key raises error."""
        with self.assertRaises(ValueError):
            self.state.set_current("invalid_key", 0.5)

    def test_update_current_invalid_key(self):
        """Test updating with invalid key raises error."""
        with self.assertRaises(ValueError):
            self.state.update_current({"beta": 0.5, "invalid": 1.0})

    def test_get_current_returns_copy(self):
        """Test that get_current returns a copy."""
        u = np.random.randn(5, self.n_dim)
        self.state.set_current("u", u)
        retrieved = self.state.get_current("u")
        retrieved[0, 0] = 999
        self.assertNotEqual(self.state.get_current("u")[0, 0], 999)


class StateManagerHistoryTestCase(unittest.TestCase):
    """Test history management."""

    def setUp(self):
        self.n_dim = 2
        self.n_particles = 8
        self.state = StateManager(self.n_dim)

    def test_commit_current_to_history(self):
        """Test committing current state to history."""
        u = np.random.randn(self.n_particles, self.n_dim)
        logl = np.random.randn(self.n_particles)
        beta = 0.5
        iter_val = 3

        self.state.set_current("u", u)
        self.state.set_current("logl", logl)
        self.state.set_current("beta", beta)
        self.state.set_current("iter", iter_val)
        self.state.set_current("logz", -50.0)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        # Check that values were committed
        self.assertEqual(len(self.state._history["u"]), 1)
        self.assertEqual(len(self.state._history["logl"]), 1)
        np.testing.assert_array_equal(self.state._history["u"][0], u)
        np.testing.assert_array_equal(self.state._history["logl"][0], logl)

    def test_get_history_index(self):
        """Test getting history at specific index."""
        u = np.random.randn(self.n_particles, self.n_dim)
        self.state.set_current("u", u)
        self.state.set_current("beta", 0.5)
        self.state.set_current("logz", -50.0)
        self.state.set_current("iter", 1)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        retrieved = self.state.get_history("u", index=0)
        np.testing.assert_array_equal(retrieved, u)

    def test_get_history_all(self):
        """Test getting all history."""
        for i in range(3):
            u = np.random.randn(self.n_particles, self.n_dim) + i
            self.state.set_current("u", u)
            self.state.set_current("beta", i * 0.1)
            self.state.set_current("logz", -50.0 - i)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)
            self.state.commit_current_to_history()

        history = self.state.get_history("u")
        self.assertEqual(len(history), 3)
        self.assertEqual(history.shape, (3, self.n_particles, self.n_dim))

    def test_get_history_flat(self):
        """Test getting flattened history."""
        for i in range(3):
            u = np.random.randn(self.n_particles, self.n_dim) + i
            self.state.set_current("u", u)
            self.state.set_current("beta", i * 0.1)
            self.state.set_current("logz", -50.0 - i)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)
            self.state.commit_current_to_history()

        flat_history = self.state.get_history("u", flat=True)
        self.assertEqual(flat_history.shape[0], 3 * self.n_particles)

    def test_get_history_invalid_key(self):
        """Test getting invalid history key raises error."""
        with self.assertRaises(ValueError):
            self.state.get_history("invalid")

    def test_get_history_invalid_index(self):
        """Test getting history with invalid index raises error."""
        self.state.set_current("u", np.random.randn(5, 2))
        self.state.set_current("beta", 0.5)
        self.state.set_current("logz", -50.0)
        self.state.set_current("iter", 1)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        with self.assertRaises(IndexError):
            self.state.get_history("u", index=5)


class StateManagerWeightsTestCase(unittest.TestCase):
    """Test weight computation."""

    def setUp(self):
        self.n_dim = 2
        self.state = StateManager(self.n_dim)

    def test_compute_logw_empty_history(self):
        """Test weight computation with empty history."""
        logw, logz = self.state.compute_logw_and_logz()
        self.assertEqual(len(logw), 0)
        self.assertEqual(logz, -np.inf)

    def test_compute_logw_single_iteration(self):
        """Test weight computation with single iteration."""
        n_particles = 10
        logl = np.random.randn(n_particles)
        beta = 0.5
        logz = -50.0

        self.state.set_current("logl", logl)
        self.state.set_current("beta", beta)
        self.state.set_current("logz", logz)
        self.state.set_current("iter", 1)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        logw, logz_new = self.state.compute_logw_and_logz(beta_final=1.0)
        self.assertEqual(len(logw), n_particles)
        self.assertIsInstance(logz_new, float)

    def test_compute_logw_multiple_iterations(self):
        """Test weight computation with multiple iterations."""
        n_particles = 10

        for i in range(3):
            logl = np.random.randn(n_particles)
            beta = i * 0.3
            logz = -50.0 - i * 5

            self.state.set_current("logl", logl)
            self.state.set_current("beta", beta)
            self.state.set_current("logz", logz)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)

            self.state.commit_current_to_history()

        logw, logz_new = self.state.compute_logw_and_logz(beta_final=1.0)
        self.assertEqual(len(logw), 3 * n_particles)
        self.assertIsInstance(logz_new, float)

    def test_compute_logw_normalization(self):
        """Test weight normalization."""
        n_particles = 10
        logl = np.random.randn(n_particles)

        self.state.set_current("logl", logl)
        self.state.set_current("beta", 0.5)
        self.state.set_current("logz", -50.0)
        self.state.set_current("iter", 1)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        logw_norm, _ = self.state.compute_logw_and_logz(beta_final=1.0, normalize=True)
        logw_unnorm, _ = self.state.compute_logw_and_logz(
            beta_final=1.0, normalize=False
        )

        self.assertNotEqual(logw_norm[0], logw_unnorm[0])
        self.assertAlmostEqual(np.exp(logw_norm).sum(), 1.0, places=5)


class StateManagerCompareWithParticlesTestCase(unittest.TestCase):
    """Compare StateManager output with Particles output."""

    def setUp(self):
        self.n_dim = 2
        self.n_particles = 10
        self.state = StateManager(self.n_dim)
        self.particles = Particles(self.n_particles, self.n_dim)

    def test_get_history_matches_particles(self):
        """Test that get_history matches Particles.get."""
        for i in range(3):
            u = np.random.randn(self.n_particles, self.n_dim) + i
            logl = np.random.randn(self.n_particles)
            beta = i * 0.2
            logz = -50.0 - i * 5

            # Update StateManager
            self.state.set_current("u", u)
            self.state.set_current("logl", logl)
            self.state.set_current("beta", beta)
            self.state.set_current("logz", logz)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)
            self.state.commit_current_to_history()

            # Update Particles (with scalars for per-iteration values)
            blobs = np.zeros(self.n_particles)
            self.particles.update(
                {
                    "u": u,
                    "logl": logl,
                    "logw": np.zeros(self.n_particles),
                    "blobs": blobs,
                    "iter": i,
                    "logz": logz,
                    "calls": 100,
                    "steps": 10,
                    "efficiency": 0.8,
                    "ess": 0.9,
                    "accept": 0.7,
                    "beta": beta,
                }
            )

        # Compare outputs
        for flat in [False, True]:
            sm_u = self.state.get_history("u", flat=flat)
            p_u = self.particles.get("u", flat=flat)
            np.testing.assert_array_equal(sm_u, p_u)

            sm_logl = self.state.get_history("logl", flat=flat)
            p_logl = self.particles.get("logl", flat=flat)
            np.testing.assert_array_equal(sm_logl, p_logl)

    def test_compute_logw_matches_particles(self):
        """Test that compute_logw_and_logz matches Particles."""
        for i in range(3):
            logl = np.random.randn(self.n_particles)
            beta = i * 0.2
            logz = -50.0 - i * 5

            # Update StateManager
            self.state.set_current("logl", logl)
            self.state.set_current("beta", beta)
            self.state.set_current("logz", logz)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)
            self.state.commit_current_to_history()

            # Update Particles (with scalars for per-iteration values)
            blobs = np.zeros(self.n_particles)
            self.particles.update(
                {
                    "u": np.random.randn(self.n_particles, self.n_dim),
                    "logl": logl,
                    "logw": np.zeros(self.n_particles),
                    "blobs": blobs,
                    "iter": i,
                    "logz": logz,
                    "calls": 100,
                    "steps": 10,
                    "efficiency": 0.8,
                    "ess": 0.9,
                    "accept": 0.7,
                    "beta": beta,
                }
            )

        for normalize in [True, False]:
            sm_logw, sm_logz = self.state.compute_logw_and_logz(
                1.0, normalize=normalize
            )
            p_logw, p_logz = self.particles.compute_logw_and_logz(
                1.0, normalize=normalize
            )

            np.testing.assert_array_almost_equal(sm_logw, p_logw, decimal=10)
            self.assertAlmostEqual(sm_logz, p_logz, places=10)


class StateManagerPersistenceTestCase(unittest.TestCase):
    """Test save/load functionality."""

    def setUp(self):
        self.n_dim = 2
        self.state = StateManager(self.n_dim)
        self.test_path = Path("test_state.state")

    def tearDown(self):
        if self.test_path.exists():
            self.test_path.unlink()

    def test_save_load_roundtrip(self):
        """Test that save and load preserve state."""
        u = np.random.randn(10, self.n_dim)
        logl = np.random.randn(10)
        beta = 0.5
        logz = -50.0
        iter_val = 5

        self.state.set_current("u", u)
        self.state.set_current("logl", logl)
        self.state.set_current("beta", beta)
        self.state.set_current("logz", logz)
        self.state.set_current("iter", iter_val)
        self.state.set_current("calls", 100)
        self.state.set_current("steps", 10)
        self.state.set_current("efficiency", 0.8)
        self.state.set_current("ess", 0.9)
        self.state.set_current("acceptance", 0.7)

        self.state.commit_current_to_history()

        self.state.save_state(self.test_path)
        self.assertTrue(self.test_path.exists())

        new_state = StateManager(self.n_dim)
        new_state.load_state(self.test_path)

        self.assertEqual(new_state.n_dim, self.n_dim)
        np.testing.assert_array_equal(new_state.get_history("u", index=0), u)
        np.testing.assert_array_equal(new_state.get_history("logl", index=0), logl)
        self.assertEqual(new_state.get_history("beta", index=0), beta)

    def test_save_creates_directory(self):
        """Test that save creates parent directory."""
        path = Path("test_dir/test_state.state")
        try:
            self.state.save_state(path)
            self.assertTrue(path.exists())
            self.assertTrue(path.parent.exists())
        finally:
            if path.exists():
                path.unlink()
            if path.parent.exists():
                path.parent.rmdir()

    def test_compute_results(self):
        """Test compute_results method."""
        for i in range(2):
            u = np.random.randn(10, self.n_dim)
            logl = np.random.randn(10)
            beta = i * 0.2

            self.state.set_current("u", u)
            self.state.set_current("logl", logl)
            self.state.set_current("beta", beta)
            self.state.set_current("logz", -50.0 - i)
            self.state.set_current("iter", i)
            self.state.set_current("calls", 100)
            self.state.set_current("steps", 10)
            self.state.set_current("efficiency", 0.8)
            self.state.set_current("ess", 0.9)
            self.state.set_current("acceptance", 0.7)

            self.state.commit_current_to_history()

        results = self.state.compute_results()
        self.assertIsInstance(results, dict)
        self.assertIn("logw", results)
        self.assertIn("u", results)
        self.assertIn("logl", results)
        self.assertEqual(len(results["logw"]), 20)


if __name__ == "__main__":
    unittest.main()
