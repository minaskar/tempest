import unittest
import numpy as np

from scipy.stats import norm

from tempest.sampler import Sampler

class SamplerTestCase(unittest.TestCase):
    @staticmethod
    def prior_transform(u):
        # Transform from uniform [0,1] to standard normal
        return norm.ppf(u)
    
    @staticmethod
    def log_likelihood_single(x):
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2)

    @staticmethod
    def log_likelihood_vectorized(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    def test_run(self):
        
        n_dim = 2

        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_single,
            n_dim=n_dim,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)

    def test_run_vectorized(self):
        
        n_dim = 2

        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood_vectorized,
            n_dim=n_dim,
            vectorize=True,
            n_effective=64,
            n_active=32,
            clustering=False,
            random_state=0,
        )
        sampler.run(n_total=128)


if __name__ == '__main__':
    unittest.main()
