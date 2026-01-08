import unittest
import numpy as np
from scipy.stats import norm
from tempest.sampler import Sampler


class EdgeCasesTestCase(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    @staticmethod
    def prior_transform_gaussian(u):
        """Transform from unit cube to standard normal."""
        return norm.ppf(u)
    
    @staticmethod
    def prior_transform_periodic(u):
        """Transform for periodic parameters [0, 2*pi]."""
        return 2 * np.pi * u
    
    @staticmethod
    def log_likelihood_gaussian(x):
        """Gaussian log likelihood."""
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)
    
    @staticmethod
    def log_likelihood_bimodal(x):
        """Bimodal Gaussian mixture."""
        # Two Gaussians separated by distance
        dist1 = np.sum((x - 2)**2, axis=1)
        dist2 = np.sum((x + 2)**2, axis=1)
        logp1 = -0.5 * dist1 - len(x[0]) * 0.5 * np.log(2 * np.pi)
        logp2 = -0.5 * dist2 - len(x[0]) * 0.5 * np.log(2 * np.pi)
        # Log-sum-exp trick
        max_logp = np.maximum(logp1, logp2)
        return max_logp + np.log(np.exp(logp1 - max_logp) + np.exp(logp2 - max_logp)) - np.log(2)
    
    def test_periodic_boundaries(self):
        """Test periodic boundary conditions."""
        n_dim = 2
        sampler = Sampler(
            prior_transform=self.prior_transform_periodic,
            log_likelihood=self.log_likelihood_gaussian,
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            periodic=[0, 1],  # Both parameters periodic
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertIsNotNone(sampler.periodic)
        self.assertEqual(len(sampler.periodic), 2)
    
    def test_reflective_boundaries(self):
        """Test reflective boundary conditions."""
        n_dim = 2
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=self.log_likelihood_gaussian,
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            reflective=[0],  # First parameter reflective
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertIsNotNone(sampler.reflective)
        self.assertEqual(len(sampler.reflective), 1)
    
    def test_single_dimension(self):
        """Test 1D case."""
        n_dim = 1
        
        def prior_transform_1d(u):
            # Return scalar for 1D
            if np.isscalar(u):
                return norm.ppf(u)
            return norm.ppf(u)
        
        def log_likelihood_1d(x):
            # Handle both scalar and array inputs
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return -0.5 * np.log(2 * np.pi) - 0.5 * np.sum(x ** 2, axis=1)
        
        sampler = Sampler(
            prior_transform=prior_transform_1d,
            log_likelihood=log_likelihood_1d,
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertEqual(sampler.n_dim, 1)
    
    def test_bimodal_distribution(self):
        """Test bimodal distribution (challenging case)."""
        n_dim = 2
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=self.log_likelihood_bimodal,
            n_dim=n_dim,
            vectorize=True,
            n_effective=128,
            n_active=64,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=256)
        self.assertIsNotNone(sampler.logz)
    
    def test_small_n_effective(self):
        """Test with very small n_effective."""
        n_dim = 2
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=self.log_likelihood_gaussian,
            n_dim=n_dim,
            vectorize=True,
            n_effective=16,
            n_active=8,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=32)
        self.assertEqual(sampler.n_effective_init, 16)
    
    def test_likelihood_with_args(self):
        """Test likelihood with additional arguments."""
        n_dim = 2
        
        def log_likelihood_with_args(x, scale=1.0):
            return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * (x / scale) ** 2, axis=1)
        
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=log_likelihood_with_args,
            log_likelihood_kwargs={'scale': 2.0},
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertIsNotNone(sampler.logz)
    
    def test_no_vectorization(self):
        """Test non-vectorized likelihood (processed one sample at a time)."""
        n_dim = 2
        
        def log_likelihood_single(x):
            return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2)
        
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=log_likelihood_single,
            n_dim=n_dim,
            vectorize=False,
            n_effective=32,
            n_active=16,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=64)
        self.assertFalse(sampler.vectorize)
    
    def test_very_narrow_likelihood(self):
        """Test very peaked/narrow likelihood."""
        n_dim = 2
        
        def log_likelihood_narrow(x):
            # Very narrow Gaussian (sigma = 0.1)
            return np.sum(-0.5 * np.log(2 * np.pi * 0.01) - 0.5 * x**2 / 0.01, axis=1)
        
        sampler = Sampler(
            prior_transform=self.prior_transform_gaussian,
            log_likelihood=log_likelihood_narrow,
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)
        self.assertIsNotNone(sampler.logz)


if __name__ == '__main__':
    unittest.main()
