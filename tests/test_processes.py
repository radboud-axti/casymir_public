import unittest

import numpy as np

from casymir.processes import initial_signal, initial_signal_cbct_position


class TestProcesses(unittest.TestCase):

    def setUp(self):
        self.detector = type("DetectorStub", (), {"px_size": 0.2, "elems": 8})()
        self.spectrum = type("SpectrumStub", (), {"get_fluence": lambda self: 100.0})()

    def test_initial_signal_without_position_uses_central_fluence(self):
        signal, gain, _ = initial_signal(self.detector, self.spectrum)

        self.assertEqual(gain, 100.0)
        self.assertEqual(signal.mean_quanta, 100.0)
        np.testing.assert_allclose(signal.signal, 100.0)

    def test_initial_signal_position_uses_flat_detector_cosine_cubed_correction(self):
        position_uv_mm = (300.0, 400.0)
        sdd_mm = 1000.0
        signal, gain, _ = initial_signal(
            self.detector,
            self.spectrum,
            detector_position_uv_mm=position_uv_mm,
            sdd_mm=sdd_mm,
        )

        ray_distance_mm = np.sqrt(sdd_mm**2 + position_uv_mm[0]**2 + position_uv_mm[1]**2)
        expected = 100.0 * (sdd_mm / ray_distance_mm) ** 3
        self.assertAlmostEqual(gain, expected)
        self.assertAlmostEqual(signal.mean_quanta, expected)
        np.testing.assert_allclose(signal.signal, expected)

    def test_initial_signal_requires_position_and_distance_together(self):
        with self.assertRaises(ValueError):
            initial_signal(self.detector, self.spectrum, detector_position_uv_mm=(0.0, 0.0))

        with self.assertRaises(ValueError):
            initial_signal(self.detector, self.spectrum, sdd_mm=1000.0)

    def test_cbct_position_wrapper_preserves_legacy_pixel_coordinates(self):
        detector_position_uv = (6.0, 4.0)
        detector_shape_uv = (10, 12)
        sdd_mm = 950.0
        wrapped = initial_signal_cbct_position(
            self.detector,
            self.spectrum,
            detector_position_uv,
            detector_shape_uv,
            sdd_mm,
        )

        u_mm = (detector_position_uv[0] + 0.5 - detector_shape_uv[0] / 2) * self.detector.px_size
        v_mm = (detector_position_uv[1] + 0.5) * self.detector.px_size
        direct = initial_signal(
            self.detector,
            self.spectrum,
            detector_position_uv_mm=(u_mm, v_mm),
            sdd_mm=sdd_mm,
        )

        np.testing.assert_allclose(wrapped[0].signal, direct[0].signal)
        self.assertEqual(wrapped[1], direct[1])


if __name__ == "__main__":
    unittest.main()
