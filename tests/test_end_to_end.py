import unittest
import numpy as np
from scipy.stats import multivariate_normal
from tempest.sampler import Sampler


class EndToEndGaussianTestCase(unittest.TestCase):
    """End-to-end test: recover Gaussian posterior from uniform prior with clustering enabled."""

    def setUp(self):
        """Set up 10-dimensional Gaussian problem."""
        self.n_dim = 10

        self.true_mean = np.array(
            [2.0, -1.5, 0.5, 3.2, -2.8, 1.1, -0.7, 2.5, -1.2, 0.9]
        )

        self.true_cov = np.diag([1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15, 0.95])

        self.true_logz = -29.96

    @staticmethod
    def prior_transform(u):
        """Transform uniform[0,1] to uniform[-10, 10]."""
        return 20 * u - 10

    def log_likelihood(self, x):
        """Gaussian log-likelihood with pre-specified parameters."""
        return multivariate_normal.logpdf(x, mean=self.true_mean, cov=self.true_cov)

    def test_end_to_end_gaussian_posterior(self):
        """Test that sampler recovers Gaussian posterior with clustering enabled."""
        sampler = Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=self.n_dim,
            vectorize=True,
            n_effective=256,
            n_active=128,
            clustering=False,
            random_state=42,
            n_steps=1,
        )

        sampler.run(n_total=2048, progress=False)

        samples, weights, _ = sampler.posterior(
            trim_importance_weights=True, resample=False
        )

        posterior_mean = np.average(samples, weights=weights, axis=0)
        posterior_cov = np.cov(samples, rowvar=False, aweights=weights)

        np.testing.assert_allclose(
            posterior_mean,
            self.true_mean,
            atol=0.25,
            rtol=0,
            err_msg="Posterior mean deviates from true mean",
        )

        np.testing.assert_allclose(
            np.diag(posterior_cov),
            np.diag(self.true_cov),
            atol=0.5,
            rtol=0,
            err_msg="Posterior covariance deviates from true covariance",
        )

        logz, _ = sampler.evidence()
        print(f"Evidence: {logz:.3f} (analytical: {self.true_logz:.3f})")
        self.assertAlmostEqual(
            logz, self.true_logz, delta=0.5, msg="Evidence tolerance check"
        )

        self.assertGreater(sampler.state.get_current("beta"), 0.99)
        self.assertGreater(sampler.state.get_current("acceptance"), 0.1)


if __name__ == "__main__":
    unittest.main()
