import unittest
from importlib import resources
import numpy as np
from unittest.mock import MagicMock
from casymir.casymir import Spectrum, Tube, Detector


class TestSpectrum(unittest.TestCase):

    def setUp(self) -> None:

        with resources.path("casymir.data.detectors", "CsI.yaml") as material_path:
            self.material_file_indirect = str(material_path)

        self.mock_tube = Tube({
            'target_angle': 10,
            'target': 'W',
            'SID': 95,
            'filter': [('Be', 1.4), ('Al', 1.42),  ('Al', 0.044)],
            'external_filter': [('Al', 1.0), ('Air', 950)]
        })

        self.mock_detector = Detector("indirect", self.material_file_indirect, {
            "type": "indirect",
            "active_layer": "CsI",
            "px_size": 0.1518,
            "ff": 0.83,
            "thickness": 700,
            "trapping_depth": 7,
            "elems": 256,
            "add_noise": 100,
            "extra_materials": [('Al', 0.05), ('Carbon Fiber', 2.5), ('Silicon Dioxide', 1)]
        })

        self.spectrum = Spectrum('test_spec', 51.1, 0.52364, self.mock_detector, self.mock_tube)
        self.spectrum.spec = MagicMock()

    def test_add_filtration(self):
        self.spectrum.add_filtration([('Al', 2), ('Cu', 1.5)])
        self.spectrum.spec.multi_filter.assert_called_once_with([('Al', 2), ('Cu', 1.5)])

    def test_get_incident_spec(self):
        self.spectrum.spec.get_spectrum = MagicMock(return_value=(np.array([10, 20, 30]), np.array([100, 200, 300])))
        energy, incident_spectrum = self.spectrum.get_incident_spec([0, 0, 95])

        np.testing.assert_array_equal(energy, np.array([10, 20, 30]))
        np.testing.assert_array_equal(incident_spectrum, np.array([100, 200, 300]))
        self.spectrum.spec.get_spectrum.assert_called_once_with(edges=False, flu=True, diff=False, x=0, y=0, z=95)

    def test_get_hvl(self):
        self.spectrum.spec.get_hvl1 = MagicMock(return_value=1.4)
        hvl = self.spectrum.get_hvl([0, 0, 95])
        self.assertEqual(hvl, 1.4)
        self.spectrum.spec.get_hvl1.assert_called_once_with(x=0, y=0, z=95)

    def test_get_k_air(self):
        self.spectrum.spec.get_kerma = MagicMock(return_value=120.5)
        k_air = self.spectrum.get_k_air([0, 0, 65])
        self.assertEqual(k_air, 120.5)
        self.spectrum.spec.get_kerma.assert_called_once_with(norm=False, x=0, y=0, z=65)

    def test_get_fluence_custom_coordinates(self):
        spectrum = Spectrum('test_spec', 51.1, 0.52364, self.mock_detector, self.mock_tube)
        spectrum.spec.get_flu = MagicMock(return_value=26513)
        fluence = spectrum.get_fluence(coordinates=(0, 0, 95))
        self.assertEqual(fluence, 26513 * 1e-2)
        spectrum.spec.get_flu.assert_called_once_with(x=0, y=0, z=95)
