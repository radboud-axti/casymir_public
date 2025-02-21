import unittest
import numpy as np
from casymir.casymir import Signal

freq = np.linspace(0, 4, 10)
magnitude = np.array([500] * 10)
wiener = np.array([100] * 10)


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.signal_obj = Signal(freq, magnitude, wiener)

    def test_stochastic_gain(self):
        expected_signal = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
        expected_wiener = np.array([525, 525, 525, 525, 525, 525, 525, 525, 525, 525])

        self.signal_obj.stochastic_gain(2, 0.5)

        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)

    def test_stochastic_blur(self):
        expected_signal = np.array([500, 450, 400, 350, 300, 250, 200, 150, 100, 50])
        expected_wiener = np.array([100, 176, 244, 304, 356, 400, 436, 464, 484, 496])
        t = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

        self.signal_obj.stochastic_blur(t)

        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)

    def test_deterministic_blur(self):
        expected_signal = np.array([500, 450, 400, 350, 300, 250, 200, 150, 100, 50])
        expected_wiener = np.array([100, 81, 64, 49, 36, 25, 16, 9, 4, 1])
        t = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

        self.signal_obj.deterministic_blur(t)

        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)

    def test_resample(self):
        t = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.signal_obj.deterministic_blur(t)
        self.signal_obj.resample()
        expected_signal = np.array([500, 450, 400, 350, 300, 250, 200, 150, 100, 50])
        expected_wiener = np.array([101, 85, 73, 65, 61, 61, 65, 73, 85, 101])
        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
