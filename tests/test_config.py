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
        self.assertEqual(config.n_effective, 512)
        self.assertEqual(config.n_active, 256)

    def test_defaults_set_correctly(self):
        """Test that computed defaults are set correctly."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=10,
            n_effective=512,
        )
        # n_active should default to n_effective // 2
        self.assertEqual(config.n_active, 256)
        # n_steps should default to n_dim // 2
        self.assertEqual(config.n_steps, 5)
        # n_max_steps should default to 10 * n_steps
        self.assertEqual(config.n_max_steps, 50)
        # output_dir should default to Path("states")
        self.assertEqual(config.output_dir, Path("states"))
        # output_label should default to "ps"
        self.assertEqual(config.output_label, "ps")

    def test_n_active_overrides_default(self):
        """Test that explicitly set n_active is respected."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_effective=100,
            n_active=30,
        )
        self.assertEqual(config.n_active, 30)
        # n_effective should not change
        self.assertEqual(config.n_effective, 100)

    def test_n_effective_computed_from_n_active(self):
        """Test that n_effective defaults to 2*n_active when n_effective=None."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_effective=None,  # Explicitly set to None to trigger computation
            n_active=50,  # This should trigger n_effective = 2*n_active = 100
        )
        # n_active was explicitly set
        self.assertEqual(config.n_active, 50)
        # n_effective should be computed as 2*n_active = 100
        self.assertEqual(config.n_effective, 100)

    def test_n_boost_validation_valid(self):
        """Test that valid n_boost passes validation."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_effective=64,
            n_active=32,
            n_boost=128,
        )
        self.assertEqual(config.n_boost, 128)

    def test_n_boost_validation_invalid_too_small(self):
        """Test that n_boost < n_effective raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                n_effective=64,
                n_active=32,
                n_boost=32,  # Too small
            )
        self.assertIn("n_boost (32) must be >= n_effective (64)", str(cm.exception))

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                metric="invalid",  # Invalid metric
            )
        self.assertIn(
            "Invalid metric 'invalid': must be 'ess' or 'uss'", str(cm.exception)
        )

    def test_invalid_sampler_raises_error(self):
        """Test that invalid sampler raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                sample="invalid",  # Invalid sampler
            )
        self.assertIn(
            "Invalid sampler 'invalid': must be 'tpcn' or 'rwm'", str(cm.exception)
        )

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
            n_effective=512,
            n_active=256,
        )

        # Try to modify (should fail)
        with self.assertRaises(AttributeError):
            config.n_effective = 200

        # Original value should be unchanged
        self.assertEqual(config.n_effective, 512)

    def test_to_dict_serialization(self):
        """Test that to_dict produces valid serialization dict."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_effective=128,
            n_active=64,
            clustering=True,
            metric="uss",
        )

        d = config.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["n_dim"], 5)
        self.assertEqual(d["n_effective"], 128)
        self.assertEqual(d["n_active"], 64)
        self.assertEqual(d["clustering"], True)
        self.assertEqual(d["metric"], "uss")
        self.assertEqual(d["output_dir"], "states")  # Stringified Path
        self.assertEqual(d["output_label"], "ps")

    def test_custom_n_steps(self):
        """Test that custom n_steps is respected."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=10,
            n_steps=20,  # Custom value
        )
        # Custom value should be kept
        self.assertEqual(config.n_steps, 20)
        # n_max_steps should be 10 * n_steps
        self.assertEqual(config.n_max_steps, 200)

    def test_n_active_less_than_n_effective(self):
        """Test that n_active must be less than n_effective."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                n_effective=100,
                n_active=100,  # Equal, not less
            )
        self.assertIn("must be < n_effective", str(cm.exception))

    def test_none_n_effective_computed(self):
        """Test that n_effective=None triggers default computation."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_effective=None,
            n_active=30,  # This should be kept
        )
        # n_effective should be computed as 2*n_active = 60
        self.assertEqual(config.n_effective, 60)
        self.assertEqual(config.n_active, 30)

    def test_none_n_active_computed(self):
        """Test that n_active=None triggers default computation."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_effective=100,
            n_active=None,  # This should trigger computation
        )
        # n_active should be computed as n_effective // 2 = 50
        self.assertEqual(config.n_active, 50)
        self.assertEqual(config.n_effective, 100)

    def test_zero_n_active_raises_error(self):
        """Test that n_active=0 raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_effective=100,
                n_active=0,  # Zero should be invalid
            )
        self.assertIn("n_active must be positive integer, got 0", str(cm.exception))

    def test_zero_n_effective_raises_error(self):
        """Test that n_effective=0 raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_effective=0,  # Zero should be invalid
                n_active=50,
            )
        self.assertIn("n_effective must be positive integer, got 0", str(cm.exception))

    def test_negative_n_active_raises_error(self):
        """Test that negative n_active raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_effective=100,
                n_active=-10,
            )
        self.assertIn("n_active must be positive integer, got -10", str(cm.exception))

    def test_negative_n_effective_raises_error(self):
        """Test that negative n_effective raises error."""
        with self.assertRaises(ValueError) as cm:
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=5,
                n_effective=-50,
                n_active=25,
            )
        self.assertIn(
            "n_effective must be positive integer, got -50", str(cm.exception)
        )


if __name__ == "__main__":
    unittest.main()
