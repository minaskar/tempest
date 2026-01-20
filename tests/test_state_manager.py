import unittest
from pathlib import Path
import numpy as np

from tempest.state_manager import (
    StateManager,
    CURRENT_STATE_KEYS,
    HISTORY_STATE_KEYS,
)


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


class StateManagerAccessorTestCase(unittest.TestCase):
    """Test new accessor methods for encapsulation."""

    def setUp(self):
        self.n_dim = 3
        self.n_particles = 10
        self.state = StateManager(self.n_dim)

    def test_get_last_history_empty(self):
        """Test get_last_history returns default when empty."""
        result = self.state.get_last_history("beta", default=-1)
        self.assertEqual(result, -1)

        result = self.state.get_last_history("beta", default=None)
        self.assertIsNone(result)

    def test_get_last_history_with_data(self):
        """Test get_last_history returns last entry."""
        # Add multiple values
        self.state.set_current("beta", 0.1)
        self.state.commit_current_to_history()
        self.state.set_current("beta", 0.5)
        self.state.commit_current_to_history()
        self.state.set_current("beta", 0.9)
        self.state.commit_current_to_history()

        result = self.state.get_last_history("beta")
        self.assertAlmostEqual(result, 0.9)

    def test_get_last_history_returns_copy(self):
        """Test get_last_history returns a copy for arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        self.state.set_current("logl", arr)
        self.state.commit_current_to_history()

        result = self.state.get_last_history("logl")
        result[0] = 999.0

        # Original should be unchanged
        original = self.state.get_last_history("logl")
        self.assertAlmostEqual(original[0], 1.0)

    def test_get_last_history_invalid_key(self):
        """Test get_last_history raises error for invalid key."""
        with self.assertRaises(ValueError):
            self.state.get_last_history("invalid_key")

    def test_get_last_history_array_values(self):
        """Test get_last_history works with array values."""
        u = np.random.randn(self.n_particles, self.n_dim)
        self.state.set_current("u", u)
        self.state.commit_current_to_history()

        result = self.state.get_last_history("u")
        np.testing.assert_array_equal(result, u)

    def test_get_history_length_empty(self):
        """Test get_history_length returns 0 for new StateManager."""
        self.assertEqual(self.state.get_history_length(), 0)

    def test_get_history_length_after_commits(self):
        """Test get_history_length returns correct count."""
        self.assertEqual(self.state.get_history_length(), 0)

        # Add first iteration
        self.state.set_current("beta", 0.1)
        self.state.commit_current_to_history()
        self.assertEqual(self.state.get_history_length(), 1)

        # Add second iteration
        self.state.set_current("beta", 0.5)
        self.state.commit_current_to_history()
        self.assertEqual(self.state.get_history_length(), 2)

        # Add third iteration
        self.state.set_current("beta", 0.9)
        self.state.commit_current_to_history()
        self.assertEqual(self.state.get_history_length(), 3)

    def test_get_history_length_consistent(self):
        """Test get_history_length is consistent across multiple commits."""
        # Commit multiple iterations with different values
        for i in range(5):
            self.state.set_current("beta", i * 0.2)
            self.state.set_current("logl", np.random.randn(self.n_particles))
            self.state.set_current("u", np.random.randn(self.n_particles, self.n_dim))
            self.state.commit_current_to_history()

        self.assertEqual(self.state.get_history_length(), 5)

        # Verify history length matches actual history
        self.assertEqual(len(self.state.get_history("beta")), 5)
        self.assertEqual(len(self.state.get_history("logl")), 5)


class StateManagerSerializationTestCase(unittest.TestCase):
    """Test serialization methods (to_dict, from_dict, update_from_dict)."""

    def setUp(self):
        self.n_dim = 3
        self.n_particles = 10
        self.state = StateManager(self.n_dim)

    def test_to_dict_structure(self):
        """Test to_dict returns correct keys."""
        result = self.state.to_dict()
        self.assertIsInstance(result, dict)
        self.assertIn("_current", result)
        self.assertIn("_history", result)
        self.assertIn("n_dim", result)
        self.assertEqual(result["n_dim"], self.n_dim)

    def test_to_dict_returns_copies(self):
        """Test to_dict returns copies, not references."""
        self.state.set_current("beta", 0.5)
        result = self.state.to_dict()

        # Modify returned dict
        result["_current"]["beta"] = 999

        # Original should be unchanged
        self.assertEqual(self.state.get_current("beta"), 0.5)

    def test_to_dict_with_data(self):
        """Test to_dict captures current state correctly."""
        u = np.random.randn(self.n_particles, self.n_dim)
        logl = np.random.randn(self.n_particles)
        beta = 0.75

        self.state.set_current("u", u)
        self.state.set_current("logl", logl)
        self.state.set_current("beta", beta)
        self.state.commit_current_to_history()

        result = self.state.to_dict()

        # Verify structure
        self.assertIn("_current", result)
        self.assertIn("_history", result)

        # Verify current data
        self.assertEqual(result["_current"]["beta"], beta)

        # Verify history data
        self.assertEqual(len(result["_history"]["beta"]), 1)
        self.assertEqual(result["_history"]["beta"][0], beta)

    def test_from_dict_creates_valid_instance(self):
        """Test from_dict creates working instance."""
        # Set up source state
        self.state.set_current("beta", 0.5)
        self.state.set_current("iter", 3)
        self.state.commit_current_to_history()

        # Export and create new instance
        state_dict = self.state.to_dict()
        new_state = StateManager.from_dict(state_dict)

        # Verify it's a valid StateManager
        self.assertIsInstance(new_state, StateManager)
        self.assertEqual(new_state.n_dim, self.n_dim)

        # Verify history was transferred
        self.assertEqual(new_state.get_history_length(), 1)
        self.assertEqual(new_state.get_history("beta", index=0), 0.5)

    def test_from_dict_with_partial_dict(self):
        """Test from_dict handles missing keys gracefully."""
        # Minimal dict with only n_dim
        minimal_dict = {"n_dim": 5}
        state = StateManager.from_dict(minimal_dict)

        self.assertEqual(state.n_dim, 5)
        self.assertEqual(state.get_history_length(), 0)

    def test_to_dict_from_dict_roundtrip(self):
        """Test full roundtrip preserves data."""
        # Set up state with data
        u = np.random.randn(self.n_particles, self.n_dim)
        logl = np.random.randn(self.n_particles)

        self.state.set_current("u", u)
        self.state.set_current("logl", logl)
        self.state.set_current("beta", 0.5)
        self.state.commit_current_to_history()

        self.state.set_current("u", u * 2)
        self.state.set_current("logl", logl * 2)
        self.state.set_current("beta", 0.8)
        self.state.commit_current_to_history()

        # Roundtrip
        state_dict = self.state.to_dict()
        new_state = StateManager.from_dict(state_dict)

        # Verify all data preserved
        self.assertEqual(new_state.get_history_length(), 2)
        np.testing.assert_array_equal(new_state.get_history("u", index=0), u)
        np.testing.assert_array_equal(new_state.get_history("logl", index=0), logl)
        self.assertEqual(new_state.get_history("beta", index=0), 0.5)
        self.assertEqual(new_state.get_history("beta", index=1), 0.8)

    def test_update_from_dict_merges_state(self):
        """Test update_from_dict merges correctly."""
        # Initial state
        self.state.set_current("beta", 0.1)
        self.state.commit_current_to_history()

        # Create update dict
        update_dict = {
            "_current": {"beta": 0.5, "iter": 5},
            "_history": {"beta": [0.2, 0.3, 0.4]},
        }

        # Update
        self.state.update_from_dict(update_dict)

        # Verify merge (history should have all values from update)
        self.assertEqual(self.state.get_current("beta"), 0.5)
        self.assertEqual(self.state.get_current("iter"), 5)

        # History should contain the updated values
        # (dict.update() adds/replaces keys, so we get the update values)
        self.assertGreater(self.state.get_history_length(), 0)

    def test_update_from_dict_partial(self):
        """Test update_from_dict with partial dict."""
        self.state.set_current("beta", 0.1)
        self.state.set_current("iter", 1)

        # Update only beta
        update_dict = {"_current": {"beta": 0.9}}
        self.state.update_from_dict(update_dict)

        # Beta should be updated, iter unchanged
        self.assertEqual(self.state.get_current("beta"), 0.9)
        self.assertEqual(self.state.get_current("iter"), 1)

    def test_update_from_dict_invalidates_cache(self):
        """Test update_from_dict invalidates results cache."""
        # Set up state and compute results (populates cache)
        self.state.set_current("u", np.random.randn(10, self.n_dim))
        self.state.set_current("logl", np.random.randn(10))
        self.state.set_current("beta", 0.5)
        self.state.commit_current_to_history()

        # This should populate _results_dict cache
        _ = self.state.compute_results()
        self.assertIsNotNone(self.state._results_dict)

        # Update from dict should invalidate cache
        update_dict = {"_current": {"beta": 0.9}}
        self.state.update_from_dict(update_dict)

        # Cache should be invalidated
        self.assertIsNone(self.state._results_dict)


class StateManagerCopySemanticsTestCase(unittest.TestCase):
    """Test copy semantics for StateManager methods."""

    def setUp(self):
        self.n_dim = 2
        self.state = StateManager(self.n_dim)

    def test_set_current_with_copy_true(self):
        """Test set_current with copy=True (default) prevents external mutations."""
        original = np.array([1.0, 2.0, 3.0])
        self.state.set_current("logl", original, copy=True)

        # Modify original
        original[0] = 999.0

        # Internal state should be unchanged
        stored = self.state.get_current("logl")
        self.assertEqual(stored[0], 1.0)

    def test_set_current_with_copy_false(self):
        """Test set_current with copy=False allows external mutations (use with caution)."""
        original = np.array([1.0, 2.0, 3.0])
        self.state.set_current("logl", original, copy=False)

        # Modify original
        original[0] = 999.0

        # Internal state WILL be changed (because copy=False)
        stored = self.state._current["logl"]  # Access internal state directly
        self.assertEqual(stored[0], 999.0)

    def test_set_current_default_is_copy_true(self):
        """Test set_current defaults to copy=True."""
        original = np.array([1.0, 2.0, 3.0])
        self.state.set_current("logl", original)

        # Modify original
        original[0] = 999.0

        # Internal state should be unchanged (default is copy=True)
        stored = self.state.get_current("logl")
        self.assertEqual(stored[0], 1.0)

    def test_update_current_with_copy_true(self):
        """Test update_current with copy=True prevents external mutations."""
        u_original = np.random.randn(10, self.n_dim)
        logl_original = np.random.randn(10)

        self.state.update_current({"u": u_original, "logl": logl_original}, copy=True)

        # Modify originals
        u_original[0, 0] = 999.0
        logl_original[0] = 888.0

        # Internal state should be unchanged
        stored_u = self.state.get_current("u")
        stored_logl = self.state.get_current("logl")
        self.assertNotEqual(stored_u[0, 0], 999.0)
        self.assertNotEqual(stored_logl[0], 888.0)

    def test_update_current_with_copy_false(self):
        """Test update_current with copy=False allows external mutations."""
        u_original = np.random.randn(10, self.n_dim)
        logl_original = np.random.randn(10)

        self.state.update_current({"u": u_original, "logl": logl_original}, copy=False)

        # Modify originals
        u_original[0, 0] = 999.0
        logl_original[0] = 888.0

        # Internal state WILL be changed
        stored_u = self.state._current["u"]
        stored_logl = self.state._current["logl"]
        self.assertEqual(stored_u[0, 0], 999.0)
        self.assertEqual(stored_logl[0], 888.0)

    def test_get_history_returns_copy(self):
        """Test get_history returns a copy, not a view."""
        u = np.random.randn(10, self.n_dim)
        self.state.set_current("u", u)
        self.state.commit_current_to_history()

        # Get history
        history = self.state.get_history("u")

        # Modify returned value
        history[0][0, 0] = 999.0

        # Internal history should be unchanged
        internal_history = self.state._history["u"]
        self.assertNotEqual(internal_history[0][0, 0], 999.0)

    def test_get_history_with_index_returns_copy(self):
        """Test get_history with index returns a copy."""
        u = np.random.randn(10, self.n_dim)
        self.state.set_current("u", u)
        self.state.commit_current_to_history()

        # Get specific iteration
        iteration_0 = self.state.get_history("u", index=0)

        # Modify returned value
        iteration_0[0, 0] = 999.0

        # Internal history should be unchanged
        internal_history = self.state._history["u"]
        self.assertNotEqual(internal_history[0][0, 0], 999.0)

    def test_get_current_returns_copy(self):
        """Test get_current returns a copy."""
        u = np.random.randn(10, self.n_dim)
        self.state.set_current("u", u)

        # Get current value
        retrieved = self.state.get_current("u")

        # Modify returned value
        retrieved[0, 0] = 999.0

        # Internal state should be unchanged
        internal_u = self.state._current["u"]
        self.assertNotEqual(internal_u[0, 0], 999.0)


class StateManagerStrictValidationTestCase(unittest.TestCase):
    """Test strict validation mode for commit_current_to_history."""

    def setUp(self):
        self.n_dim = 2
        self.state = StateManager(self.n_dim)

    def test_commit_without_strict_allows_missing_keys(self):
        """Test commit without strict mode allows missing keys (backward compatibility)."""
        # Set only beta, not logl
        self.state.set_current("beta", 0.5)

        # Should not raise in non-strict mode (default)
        try:
            self.state.commit_current_to_history()
        except ValueError:
            self.fail(
                "commit_current_to_history() raised ValueError unexpectedly in non-strict mode"
            )

    def test_commit_strict_with_all_required_keys_succeeds(self):
        """Test strict commit succeeds when all required keys are present."""
        # Set all required keys
        self.state.set_current("beta", 0.5)
        self.state.set_current("logl", np.random.randn(10))

        # Should succeed in strict mode
        try:
            self.state.commit_current_to_history(strict=True)
        except ValueError:
            self.fail(
                "commit_current_to_history(strict=True) raised ValueError unexpectedly with all required keys"
            )

    def test_commit_strict_missing_beta_raises(self):
        """Test strict commit raises ValueError when beta is missing."""
        # Set logl but not beta
        self.state.set_current("logl", np.random.randn(10))

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            self.state.commit_current_to_history(strict=True)

        self.assertIn("beta", str(context.exception))
        self.assertIn("required keys are missing", str(context.exception).lower())

    def test_commit_strict_missing_logl_raises(self):
        """Test strict commit raises ValueError when logl is missing."""
        # Set beta but not logl
        self.state.set_current("beta", 0.5)

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            self.state.commit_current_to_history(strict=True)

        self.assertIn("logl", str(context.exception))
        self.assertIn("required keys are missing", str(context.exception).lower())

    def test_commit_strict_missing_all_required_keys_raises(self):
        """Test strict commit raises ValueError when all required keys are missing."""
        # Don't set any keys

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as context:
            self.state.commit_current_to_history(strict=True)

        exception_str = str(context.exception)
        self.assertIn("beta", exception_str)
        self.assertIn("logl", exception_str)
        self.assertIn("required keys are missing", exception_str.lower())

    def test_commit_strict_with_optional_keys_succeeds(self):
        """Test strict commit allows optional keys to be missing."""
        # Set required keys
        self.state.set_current("beta", 0.5)
        self.state.set_current("logl", np.random.randn(10))

        # Don't set optional keys like u, x, etc.

        # Should still succeed (only required keys are checked)
        try:
            self.state.commit_current_to_history(strict=True)
        except ValueError:
            self.fail(
                "commit_current_to_history(strict=True) raised ValueError with optional keys missing"
            )

    def test_commit_strict_false_is_default(self):
        """Test that strict=False is the default behavior."""
        # Set only beta
        self.state.set_current("beta", 0.5)

        # These should be equivalent (both should succeed)
        try:
            self.state.commit_current_to_history()
            self.state.commit_current_to_history(strict=False)
        except ValueError:
            self.fail("Default behavior should be strict=False")

    def test_commit_strict_with_none_values_raises(self):
        """Test strict commit raises when required keys are explicitly None."""
        # Set keys to None explicitly
        self.state.set_current("beta", None)
        self.state.set_current("logl", None)

        # Should raise ValueError in strict mode (None is not allowed)
        with self.assertRaises(ValueError) as context:
            self.state.commit_current_to_history(strict=True)

        exception_str = str(context.exception)
        self.assertIn("required keys are missing", exception_str.lower())


if __name__ == "__main__":
    unittest.main()
