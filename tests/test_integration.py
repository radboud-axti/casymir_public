import unittest
from importlib import resources
import numpy as np

import casymir.casymir
import casymir.processes
from casymir.casymir import Spectrum, Tube, Detector


class TestIntegration(unittest.TestCase):

    def setUp(self) -> None:
        with resources.path("casymir.data.detectors", "CsI.yaml") as material_path:
            self.material_file_indirect = str(material_path)

        self.detector_data = {
            "type": "indirect",
            "active_layer": "CsI",
            "px_size": 0.1518,
            "ff": 0.85,
            "thickness": 700,
            "trapping_depth": 7,
            "elems": 256,
            "add_noise": 100,
            "extra_materials": [('Al', 0.05), ('Carbon Fiber', 2.5), ('Silicon Dioxide', 1)]
        }
        self.detector = Detector("indirect", self.material_file_indirect, self.detector_data)

        self.tube_data = {
            "target_angle": 10,
            "target": "W",
            "SID": 95,
            "filter": [('Be', 1.4), ('Al', 1.42), ('Al', 0.044)],
            "external_filter": [('Al', 10), ('Air', 950)]
        }

        self.tube = Tube(self.tube_data)
        self.spectrum = Spectrum('test_spec', 51.1, 0.52364, self.detector, self.tube)

    def test_initialization(self):
        # Checks the inputs to the model are correctly generated.
        hvl = self.spectrum.get_hvl()
        k_air = self.spectrum.get_k_air()
        fluence = self.spectrum.get_fluence()
        energy, incident_spectrum = self.spectrum.get_incident_spec()

        self.assertAlmostEqual(hvl, 3.8185861481105765, places=6)
        self.assertAlmostEqual(k_air, 1.254536044675497, places=6)
        self.assertAlmostEqual(int(fluence), 26513, delta=10)
        self.assertIsInstance(energy, np.ndarray)
        self.assertEqual(np.shape(energy)[0], 501)

