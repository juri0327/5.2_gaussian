# -*- coding: utf-8 -*-
from main import Gaussian
import unittest
import numpy as np

class TestCase(unittest.TestCase):

    def assertAllClose(self, a, b, rtol=1e-5, atol=1e-8):
        self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol))

    def test_fit(self):
        model = Gaussian(dim=2)
        np.random.seed(43)
        X = np.random.multivariate_normal(
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.9], [0.9, 4.0]]),
            size=10000)
        model.fit(X)
        self.assertAllClose(model.mean,
                            np.array([1.0, 2.0]),
                            rtol=1e-1,
                            atol=1e-2)
        self.assertAllClose(model.cov,
                            np.array([[1.0, 0.9],
                                      [0.9, 4.0]]),
                            rtol=1e-1,
                            atol=1e-2)

    def test_sample(self):
        model = Gaussian(dim=2)
        model.mean = np.zeros(2)
        model.cov = np.identity(2)
        X = model.sample(10000)
        self.assertAllClose(np.mean(X, axis=0),
                            np.array([0.0, 0.0]),
                            rtol=1e-1,
                            atol=1e-1)
        self.assertAllClose(np.cov(X.T, bias=True),
                            np.identity(2),
                            rtol=1e-1,
                            atol=1e-1)

if __name__ == "__main__":
    unittest.main()
