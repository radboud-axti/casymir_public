import unittest
import numpy as np
from casymir.casymir import Signal

freq = np.linspace(0, 4, 10)
mean_quanta = 500
magnitude = np.array([500] * 10)
wiener = np.array([100] * 10)


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.freq = freq
        self.mean_quanta = mean_quanta
        self.magnitude = magnitude
        self.wiener = wiener
        self.signal_obj = Signal(freq, mean_quanta, magnitude, wiener)

    def test_stochastic_gain(self):
        mean_gain = 2
        gain_std = 0.5
        self.signal_obj.stochastic_gain(mean_gain, gain_std)

        expected_mean_quanta = self.mean_quanta * mean_gain
        expected_signal = self.magnitude * mean_gain
        expected_wiener = (mean_gain ** 2) * self.wiener + (gain_std ** 2) * self.mean_quanta

        self.assertAlmostEqual(self.signal_obj.mean_quanta, expected_mean_quanta, places=6)
        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)

    def test_stochastic_blur(self):
        t = np.linspace(1, 0.1, 10)
        self.signal_obj.stochastic_blur(t)

        expected_signal = t * self.magnitude
        expected_wiener = (t ** 2) * self.wiener + (1 - t ** 2) * self.mean_quanta

        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)

    def test_deterministic_blur(self):
        a = np.linspace(1, 0.1, 10)
        self.signal_obj.deterministic_blur(a)

        expected_mean_quanta = self.mean_quanta * a[0]
        expected_signal = a * self.magnitude
        expected_wiener = (a ** 2) * self.wiener

        self.assertAlmostEqual(self.signal_obj.mean_quanta, expected_mean_quanta, places=6)
        np.testing.assert_allclose(self.signal_obj.signal, expected_signal, atol=1e-6)
        np.testing.assert_allclose(self.signal_obj.wiener, expected_wiener, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
