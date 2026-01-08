import unittest

import numpy as np

from tempest.tools import (compute_ess, effective_sample_size, 
                          unique_sample_size, trim_weights, 
                          systematic_resample, increment_logz,
                          FunctionWrapper)


class ESSTestCase(unittest.TestCase):
    def test_ess_single_particle(self):
        self.assertEqual(compute_ess(np.array([1.0])), 1.0)
        self.assertEqual(compute_ess(np.array([251.0])), 1.0)
        self.assertEqual(compute_ess(np.array([-421.0])), 1.0)
        self.assertEqual(compute_ess(np.array([-421.125251])), 1.0)
        self.assertEqual(compute_ess(np.array([0.0])), 1.0)
    
    def test_ess_uniform_weights(self):
        """Test ESS with uniform weights."""
        n = 100
        logw = np.zeros(n)  # Equal log weights
        ess = compute_ess(logw)
        # For uniform weights, ESS percentage should be 1.0
        self.assertAlmostEqual(ess, 1.0, places=5)
    
    def test_ess_non_uniform_weights(self):
        """Test ESS with non-uniform weights."""
        logw = np.array([0.0, -1.0, -2.0, -3.0])
        ess = compute_ess(logw)
        # ESS percentage should be less than 1.0 for non-uniform weights
        self.assertLess(ess, 1.0)
        self.assertGreater(ess, 0.0)
    
    def test_effective_sample_size(self):
        """Test effective_sample_size function."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        ess = effective_sample_size(weights)
        self.assertAlmostEqual(ess, 4.0, places=5)
    
    def test_effective_sample_size_skewed(self):
        """Test ESS with skewed weights."""
        weights = np.array([0.9, 0.05, 0.03, 0.02])
        ess = effective_sample_size(weights)
        self.assertLess(ess, 4.0)
        self.assertGreater(ess, 1.0)
    
    def test_unique_sample_size(self):
        """Test unique_sample_size function."""
        weights = np.ones(100)
        uss = unique_sample_size(weights)
        self.assertGreater(uss, 0)
        self.assertLessEqual(uss, 100)
    
    def test_unique_sample_size_with_k(self):
        """Test unique_sample_size with k parameter."""
        weights = np.ones(100)
        k = 10
        uss = unique_sample_size(weights, k=k)
        self.assertGreater(uss, 0)
        self.assertLessEqual(uss, 100)


class ResamplingTestCase(unittest.TestCase):
    def test_systematic_resample(self):
        """Test systematic resampling."""
        n = 100
        weights = np.random.rand(n)
        weights /= weights.sum()
        
        indices = systematic_resample(n, weights)
        
        self.assertEqual(len(indices), n)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < n))
    
    def test_systematic_resample_uniform(self):
        """Test resampling with uniform weights."""
        n = 100
        weights = np.ones(n) / n
        
        indices = systematic_resample(n, weights)
        
        self.assertEqual(len(indices), n)
        # With uniform weights, all indices should appear roughly equally
        unique_counts = np.bincount(indices, minlength=n)
        # Most indices should appear at least once
        self.assertGreater(np.sum(unique_counts > 0), n // 2)


class UtilityTestCase(unittest.TestCase):
    def test_increment_logz(self):
        """Test log evidence increment calculation."""
        logw = np.array([0.0, -1.0, -2.0])
        logz_inc = increment_logz(logw)
        # Should return a finite value
        self.assertTrue(np.isfinite(logz_inc))
    
    def test_increment_logz_uniform(self):
        """Test log evidence increment with uniform weights."""
        n = 100
        logw = np.zeros(n)
        logz_inc = increment_logz(logw)
        # log(n) for uniform weights
        self.assertAlmostEqual(logz_inc, np.log(n), places=5)
    
    def test_trim_weights(self):
        """Test weight trimming."""
        n = 1000
        samples = np.random.randn(n, 2)
        weights = np.random.rand(n)
        weights /= weights.sum()
        
        samples_trimmed, weights_trimmed = trim_weights(samples, weights, ess=0.9)
        
        # Trimmed should have fewer samples
        self.assertLessEqual(len(samples_trimmed), n)
        self.assertLessEqual(len(weights_trimmed), n)
        # Weights should sum to 1
        self.assertAlmostEqual(weights_trimmed.sum(), 1.0, places=5)
    
    def test_function_wrapper(self):
        """Test FunctionWrapper utility."""
        def func(x, a=1, b=2):
            return x + a + b
        
        wrapper = FunctionWrapper(func, args=None, kwargs={'a': 5, 'b': 3})
        result = wrapper(10)
        self.assertEqual(result, 18)
    
    def test_function_wrapper_with_args(self):
        """Test FunctionWrapper with positional args."""
        def func(x, a, b):
            return x * a + b
        
        wrapper = FunctionWrapper(func, args=[2, 3], kwargs=None)
        result = wrapper(5)
        self.assertEqual(result, 13)


if __name__ == '__main__':
    unittest.main()
