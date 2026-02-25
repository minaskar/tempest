import unittest
import numpy as np
from pathlib import Path

from tempest.config import SamplerConfig


class TestSamplerConfig(unittest.TestCase):
    """Test SamplerConfig validation and defaults."""

    @staticmethod
    def prior_transform(u):
        return 20 * u - 10

    @staticmethod
    def log_likelihood(x):
        return np.sum(-0.5 * x**2)

    def test_minimal_valid_config(self):
        """Test that minimal valid configuration works."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
        )
        self.assertEqual(config.n_dim, 2)
        self.assertEqual(config.n_particles, 4)  # 2 * n_dim
        self.assertEqual(config.ess_ratio, 2.0)
        self.assertIsNone(config.volume_variation)

    def test_defaults_set_correctly(self):
        """Test that computed defaults are set correctly."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=10,
            n_particles=100,
        )
        # n_particles should be as specified
        self.assertEqual(config.n_particles, 100)
        # n_steps should default to 1
        self.assertEqual(config.n_steps, 1)
        # n_max_steps should default to 20 * n_steps
        self.assertEqual(config.n_max_steps, 20)
        # output_dir should default to Path("states")
        self.assertEqual(config.output_dir, Path("states"))
        # output_label should default to "ps"
        self.assertEqual(config.output_label, "ps")

    def test_n_particles_override_default(self):
        """Test that explicitly set n_particles is respected."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_particles=50,
        )
        self.assertEqual(config.n_particles, 50)

    def test_invalid_resample_raises_error(self):
        """Test that invalid resample raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                resample="invalid",  # Invalid resample
            )
        self.assertIn(
            "Invalid resample 'invalid': must be 'mult' or 'syst'", str(cm.exception)
        )

    def test_vectorize_blobs_conflict_raises_error(self):
        """Test that vectorize=True with blobs raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                vectorize=True,
                blobs_dtype="float",  # Conflict with vectorize
            )
        self.assertIn("Cannot vectorize likelihood with blobs", str(cm.exception))

    def test_periodic_reflective_overlap_raises_error(self):
        """Test that overlapping periodic/reflective raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                periodic=[0, 1, 2],
                reflective=[2, 3],  # Overlap at index 2
            )
        self.assertIn(
            "Parameters cannot be both periodic and reflective", str(cm.exception)
        )
        self.assertIn("2", str(cm.exception))

    def test_invalid_periodic_index_raises_error(self):
        """Test that invalid periodic indices raise error."""
        # Test index too high
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=3,
                periodic=[0, 1, 5],  # 5 >= n_dim
            )
        self.assertIn("periodic indices must be integers in [0, 2]", str(cm.exception))

        # Test negative index
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=3,
                periodic=[-1],  # Negative
            )
        self.assertIn("periodic indices must be integers in [0, 2]", str(cm.exception))

    def test_path_string_converted_to_path(self):
        """Test that string paths are converted to Path objects."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            output_dir="custom_output",
            output_label="custom_label",
        )
        self.assertIsInstance(config.output_dir, Path)
        self.assertEqual(config.output_dir, Path("custom_output"))
        self.assertEqual(config.output_label, "custom_label")

    def test_n_dim_validation(self):
        """Test that invalid n_dim raises error."""
        # Test non-integer
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim="invalid",  # type: ignore
            )
        self.assertIn("n_dim must be int", str(cm.exception))

        # Test negative
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=-1,
            )
        self.assertIn("n_dim must be positive int", str(cm.exception))

    def test_immutability(self):
        """Test that config is frozen/immutable."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=100,
        )

        # Try to modify (should fail)
        with self.assertRaises(AttributeError):
            config.n_particles = 200

        # Original value should be unchanged
        self.assertEqual(config.n_particles, 100)

    def test_to_dict_serialization(self):
        """Test that to_dict produces valid serialization dict."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_particles=64,
            clustering=True,
        )
        d = config.to_dict()
        self.assertEqual(d["n_dim"], 5)
        self.assertEqual(d["n_particles"], 64)
        self.assertIsNone(d["volume_variation"])

    def test_ess_ratio_validation(self):
        """Test that invalid ess_ratio raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                ess_ratio=-1.0,  # Negative
            )
        self.assertIn("ess_ratio must be positive", str(cm.exception))

    def test_volume_variation_validation(self):
        """Test that invalid volume_variation raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                volume_variation=-0.1,  # Negative
            )
        self.assertIn("volume_variation", str(cm.exception))
        self.assertIn("must be positive", str(cm.exception))

    def test_zero_n_particles_raises_error(self):
        """Test that n_particles=0 raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_particles=0,  # Zero should be invalid
            )
        self.assertIn("n_particles must be positive integer, got 0", str(cm.exception))

    def test_negative_n_particles_raises_error(self):
        """Test that negative n_particles raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_particles=-10,
            )
        self.assertIn(
            "n_particles must be positive integer, got -10", str(cm.exception)
        )

    def test_get_target_metric_ess_mode(self):
        """Test get_target_metric for ESS mode (volume_variation=None)."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            volume_variation=None,
            n_particles=50,
            ess_ratio=2.0,
        )
        # Should return ess_ratio * n_particles
        self.assertEqual(config.get_target_metric(), 100.0)

    def test_get_target_metric_dynamic_mode(self):
        """Test get_target_metric for dynamic mode (volume_variation set)."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            volume_variation=0.25,
        )
        # Should return volume_variation
        self.assertEqual(config.get_target_metric(), 0.25)

    def test_dynamic_mode_warning_insufficient_particles(self):
        """Test that dynamic mode warns with insufficient particles."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=10,
                volume_variation=0.25,  # Dynamic mode enabled
                n_particles=5,  # Less than n_dim + 1 = 11
            )
            # Check that a warning was raised
            self.assertEqual(len(w), 1)
            self.assertIn("dynamic mode", str(w[0].message))
            self.assertIn("n_particles", str(w[0].message))
            self.assertIsNotNone(config.volume_variation)


if __name__ == "__main__":
    unittest.main()
