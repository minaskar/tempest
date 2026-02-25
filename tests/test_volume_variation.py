"""Comprehensive tests for the volume_variation parameter.

The volume_variation parameter enables dynamic mode sampling,
which uses both ESS and volume variation metrics to determine
when to increment beta.
"""

import unittest
import numpy as np
from scipy.stats import norm
from tempest.sampler import Sampler
from tempest.config import SamplerConfig


class VolumeVariationConfigTestCase(unittest.TestCase):
    """Test volume_variation configuration and validation."""

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

    def test_volume_variation_none_uses_ess_mode(self):
        """Test that volume_variation=None uses ESS-only mode."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            volume_variation=None,
        )
        # Should use ESS mode
        self.assertIsNone(config.volume_variation)

    def test_volume_variation_positive_enables_dynamic_mode(self):
        """Test that positive volume_variation enables dynamic mode."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            volume_variation=0.25,
        )
        self.assertEqual(config.volume_variation, 0.25)

    def test_volume_variation_zero_raises_error(self):
        """Test that volume_variation=0 raises ValueError."""
        with self.assertRaises(ValueError):
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                n_particles=32,
                volume_variation=0.0,
            )

    def test_volume_variation_negative_raises_error(self):
        """Test that negative volume_variation raises ValueError."""
        with self.assertRaises(ValueError):
            SamplerConfig(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=2,
                n_particles=32,
                volume_variation=-0.1,
            )

    def test_volume_variation_with_ess_ratio(self):
        """Test volume_variation works with ess_ratio parameter."""
        config = SamplerConfig(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            ess_ratio=2.0,
            volume_variation=0.25,
        )
        self.assertEqual(config.ess_ratio, 2.0)
        self.assertEqual(config.volume_variation, 0.25)


class VolumeVariationSamplingTestCase(unittest.TestCase):
    """Test sampling with volume_variation enabled."""

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

    def test_ess_mode_sampling(self):
        """Test that ESS-only mode (volume_variation=None) works."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            ess_ratio=2.0,
            volume_variation=None,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        
        # Should complete successfully
        self.assertEqual(sampler.state.get_current("beta"), 1.0)
        self.assertGreater(sampler.state.get_current("iter"), 0)

    def test_dynamic_mode_sampling(self):
        """Test that dynamic mode (volume_variation set) works."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            ess_ratio=2.0,
            volume_variation=0.25,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        
        # Should complete successfully
        self.assertEqual(sampler.state.get_current("beta"), 1.0)
        self.assertGreater(sampler.state.get_current("iter"), 0)

    def test_different_volume_variation_values(self):
        """Test sampling with different volume_variation values."""
        for vol_var in [0.1, 0.25, 0.5]:
            with self.subTest(volume_variation=vol_var):
                sampler = Sampler(
                    prior_transform=self.prior_transform,
                    log_likelihood=self.log_likelihood,
                    n_dim=2,
                    n_particles=32,
                    ess_ratio=2.0,
                    volume_variation=vol_var,
                    clustering=False,
                    random_state=42,
                )
                sampler.run(n_total=128)
                
                # Should complete and reach beta=1
                self.assertEqual(sampler.state.get_current("beta"), 1.0)

    def test_dynamic_mode_with_clustering(self):
        """Test dynamic mode with clustering enabled."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=64,
            ess_ratio=2.0,
            volume_variation=0.25,
            clustering=True,
            random_state=0,
        )
        sampler.run(n_total=128)
        
        self.assertEqual(sampler.state.get_current("beta"), 1.0)

    def test_ess_vs_dynamic_mode_comparison(self):
        """Compare ESS-only and dynamic mode behavior."""
        # ESS-only mode
        sampler_ess = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            ess_ratio=2.0,
            volume_variation=None,
            clustering=False,
            random_state=42,
        )
        sampler_ess.run(n_total=128)
        
        # Dynamic mode
        sampler_dynamic = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            ess_ratio=2.0,
            volume_variation=0.25,
            clustering=False,
            random_state=42,
        )
        sampler_dynamic.run(n_total=128)
        
        # Both should complete successfully
        self.assertEqual(sampler_ess.state.get_current("beta"), 1.0)
        self.assertEqual(sampler_dynamic.state.get_current("beta"), 1.0)


class VolumeVariationEdgeCasesTestCase(unittest.TestCase):
    """Test edge cases for volume_variation."""

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

    def test_very_small_volume_variation(self):
        """Test with very small volume_variation value."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            volume_variation=0.05,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertEqual(sampler.state.get_current("beta"), 1.0)

    def test_large_volume_variation(self):
        """Test with large volume_variation value."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=2,
            n_particles=32,
            volume_variation=1.0,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertEqual(sampler.state.get_current("beta"), 1.0)

    def test_high_dimensional_with_volume_variation(self):
        """Test volume_variation with higher dimensions."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=5,
            n_particles=64,
            volume_variation=0.25,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertEqual(sampler.state.get_current("beta"), 1.0)

    def test_multimodal_with_volume_variation(self):
        """Test volume_variation with multimodal likelihood."""
        def log_likelihood_bimodal(x):
            """Bimodal Gaussian mixture."""
            if x.ndim == 1:
                x = x.reshape(1, -1)
            dist1 = np.sum((x - 2)**2, axis=1)
            dist2 = np.sum((x + 2)**2, axis=1)
            logp1 = -0.5 * dist1 - x.shape[1] * 0.5 * np.log(2 * np.pi)
            logp2 = -0.5 * dist2 - x.shape[1] * 0.5 * np.log(2 * np.pi)
            max_logp = np.maximum(logp1, logp2)
            result = max_logp + np.log(np.exp(logp1 - max_logp) + 
                                       np.exp(logp2 - max_logp)) - np.log(2)
            if x.shape[0] == 1:
                return float(result[0])
            return result
        
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=log_likelihood_bimodal,
            n_dim=2,
            n_particles=64,
            volume_variation=0.25,
            clustering=True,
            random_state=0,
        )
        sampler.run(n_total=256)
        self.assertEqual(sampler.state.get_current("beta"), 1.0)


if __name__ == "__main__":
    unittest.main()
