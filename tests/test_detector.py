import unittest
from importlib import resources
import numpy as np
from casymir.casymir import Detector
from casymir.casymir import Signal


class TestDetector(unittest.TestCase):

    def setUp(self):
        with resources.path("casymir.data.detectors", "Se.yaml") as material_path:
            self.material_file = str(material_path)

        with resources.path("casymir.data.detectors", "CsI.yaml") as material_path:
            self.material_file_indirect = str(material_path)

        self.detector = Detector("direct", self.material_file, {
            "type": "direct",
            "active_layer": "Se",
            "px_size": 0.085,
            "ff": 1,
            "thickness": 200,
            "trapping_depth": 7,
            "elems": 2816,
            "add_noise": 0
        })

        self.detector_indirect = Detector("indirect", self.material_file_indirect, {
            "type": "indirect",
            "active_layer": "CsI",
            "px_size": 0.1518,
            "ff": 0.83,
            "thickness": 700,
            "trapping_depth": 7,
            "elems": 256,
            "add_noise": 100
        })

        freq = np.linspace(0, 1 / (2 * 0.085), 2816)
        magnitude = np.ones_like(freq)
        signal = np.ones_like(freq)
        wiener = np.ones_like(freq)
        self.signal = Signal(freq, magnitude[0], signal, wiener)

    def test_initialization(self):
        self.assertEqual(self.detector.type, "direct")
        self.assertEqual(self.detector.active_layer, "Se")
        self.assertEqual(self.detector.px_size, 0.085)
        self.assertEqual(self.detector.ff, 1)
        self.assertEqual(self.detector.thickness, 200)
        self.assertEqual(self.detector.trapping_depth, 7)
        self.assertEqual(self.detector.elems, 2816)
        self.assertEqual(self.detector.add_noise, 0)
        self.assertAlmostEqual(self.detector.pxa, 0.007225, places=6)

    def test_calculate_fk(self):
        fk_value = self.detector.calculate_fk()
        expected_fk = 0.8037975690266177
        self.assertAlmostEqual(fk_value, expected_fk, places=6)

    def test_calculate_Tk(self):
        tk_value = self.detector.calculate_Tk(self.signal)

        # Reference values for second, middle, and last elements
        reference_tk_second = 0.9999996799903333
        reference_tk_middle = 0.6766989967662854
        reference_tk_last = 0.42138829234554287

        self.assertAlmostEqual(tk_value[1], reference_tk_second, places=6)
        self.assertAlmostEqual(tk_value[len(tk_value) // 2], reference_tk_middle, places=6)
        self.assertAlmostEqual(tk_value[-1], reference_tk_last, places=6)

    def test_get_mu(self):
        energy = np.linspace(1.05, 27.95, 270, endpoint=True)
        self.detector.get_mu(energy)
        mu = self.detector.mu

        ref_mu_min = 18.945780501755618
        ref_mu_max = 4969.8482118864395

        self.assertAlmostEqual(np.min(mu), ref_mu_min, places=6)
        self.assertAlmostEqual(np.max(mu), ref_mu_max, places=6)

    def test_get_QE(self):
        energy = np.linspace(1.05, 27.95, 270, endpoint=True)
        qe = self.detector.get_QE(energy)

        ref_qe_min = 0.8365448463657389
        ref_qe_max = 1.0

        self.assertAlmostEqual(np.min(qe), ref_qe_min, places=6)
        self.assertAlmostEqual(np.max(qe), ref_qe_max, places=6)

    def test_com_term_z(self):
        self.detector_indirect.thickness = 5
        energy = np.array([20, 60])
        self.detector_indirect.get_mu(energy)
        com_term_value = self.detector_indirect.com_term_z(energy)

        # Reference values
        reference_com_term = np.array([[0.00624447, 0.00193421],
                                       [0.00628118, 0.00193432],
                                       [0.00631812, 0.00193444],
                                       [0.00635528, 0.00193457],
                                       [0.00639268, 0.00193469]])

        np.testing.assert_allclose(com_term_value, reference_com_term, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
