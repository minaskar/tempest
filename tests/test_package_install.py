import unittest
import numpy as np


class TestPackageInstall(unittest.TestCase):
    """Test that the package structure is correct and all imports work.

    This test verifies that the packaging configuration includes all necessary
    subpackages like tempest.steps. These tests run in the development
    environment; the actual pip install behavior is tested in CI via
    test-wheel.yml.
    """

    def test_steps_module_exists(self):
        """Verify tempest.steps module can be imported."""
        import tempest.steps

        self.assertIn("steps", tempest.steps.__name__)

    def test_steps_classes_importable(self):
        """Verify step classes can be imported from tempest.steps."""
        from tempest.steps import Reweighter, Trainer, Resampler, Mutator

        # Just verify they're the correct classes
        self.assertTrue(hasattr(Reweighter, "__init__"))
        self.assertTrue(hasattr(Trainer, "__init__"))
        self.assertTrue(hasattr(Resampler, "__init__"))
        self.assertTrue(hasattr(Mutator, "__init__"))

    def test_sampler_instantiation_triggers_steps_import(self):
        """Verify Sampler can be instantiated (triggers steps import)."""
        from tempest import Sampler

        def prior_transform(u):
            """Simple prior transform: [0,1] -> [-10,10]."""
            return 20.0 * u - 10.0

        def log_likelihood(x):
            """Simple Gaussian likelihood."""
            return -0.5 * np.sum(x**2)

        # This should work without ModuleNotFoundError for tempest.steps
        sampler = Sampler(
            prior_transform=prior_transform,
            log_likelihood=log_likelihood,
            n_dim=2,
            n_particles=32,
        )

        self.assertEqual(sampler.n_dim, 2)
        self.assertEqual(sampler.n_particles, 32)
