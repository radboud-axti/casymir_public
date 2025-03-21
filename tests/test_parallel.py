import unittest
import numpy as np
import casymir.casymir
import casymir.processes
import casymir.parallel
from importlib import resources
import copy


class TestParallel(unittest.TestCase):

    def setUp(self):
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
        self.detector = casymir.casymir.Detector("indirect", self.material_file_indirect, self.detector_data)

        self.tube = casymir.casymir.Tube({
            'target_angle': 10,
            'target': 'W',
            'SID': 95,
            'filter': [("Be", 1.4), ("Al", 1.514)],
            'external_filter': [("Al", 10), ("Air", 950)]
        })

        self.detector.extra_materials = [("Carbon Fiber", 2.5), ("Silicon Dioxide", 1)]
        self.spec = casymir.casymir.Spectrum(
            name="spectrum",
            kV=51.1,
            mAs=80.56 * 0.0065,
            detector=self.detector,
            tube=self.tube
        )

        self.sig, _, _ = casymir.processes.initial_signal(self.detector, self.spec)
        self.sig, _, _ = casymir.processes.quantum_selection(self.detector, self.spec, self.sig)

    def generic_gain(self, detector_o, spectrum, signal):
        ga = 10
        signal_2 = copy.copy(signal)
        signal_2.stochastic_gain(ga, np.sqrt(ga))
        return signal_2, ga, np.ones(np.size(signal.freq))

    def generic_selection(self, detector_o, spectrum, signal):
        g1 = 0.7
        signal_2 = copy.copy(signal)
        signal_2.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)))
        return signal_2, g1, np.ones(np.size(signal.freq))

    def generic_blur(self, detector_o, spectrum, signal):
        ta = np.abs(np.sinc(detector_o.px_size * detector_o.ff * signal.freq))
        signal_2 = copy.copy(signal)
        signal_2.deterministic_blur(ta)
        return signal_2, 1, ta

    def test_fork_node(self):
        # Testing fork node with two paths
        Path_1 = casymir.parallel.Path([self.generic_gain], name="Path 1")
        Path_2 = casymir.parallel.Path([self.generic_selection, self.generic_gain, self.generic_blur], name="Path 2")

        Node_F = casymir.parallel.Node(node_type="Fork")
        Node_F.add(Path_1)
        Node_F.add(Path_2)

        combined_signal = casymir.parallel.apply_parallel_process(Node_F, self.sig, self.detector, self.spec)

        # Individual path calculations
        sig1, g1, t1 = self.generic_gain(self.detector, self.spec, self.sig)
        sig2, g2, t2 = self.generic_selection(self.detector, self.spec, self.sig)
        sig2, g3, t3 = self.generic_gain(self.detector, self.spec, sig2)
        sig2, g4, t4 = self.generic_blur(self.detector, self.spec, sig2)

        # Analytic Wiener cross-term
        analytic_wiener = sig1.wiener + sig2.wiener + 2 * self.sig.wiener * g1 * g2 * g3 * t1 * t4

        np.testing.assert_allclose(combined_signal.wiener, analytic_wiener, atol=1e-6)
        np.testing.assert_allclose(combined_signal.signal, sig1.signal + sig2.signal, atol=1e-6)

    def test_bernoulli_node(self):
        # Testing Bernoulli node with two mutually exclusive paths
        p_path1 = 0.6
        Path_1 = casymir.parallel.Path([self.generic_gain], name="Path 1")
        Path_2 = casymir.parallel.Path([self.generic_selection, self.generic_gain, self.generic_blur], name="Path 2")

        Node_B = casymir.parallel.Node(node_type="Bernoulli")
        Node_B.add(Path_1, probability=p_path1)
        Node_B.add(Path_2, probability=1 - p_path1)

        combined_signal = casymir.parallel.apply_parallel_process(Node_B, self.sig, self.detector, self.spec)

        # Individual path calculations
        sig1, g1, t1 = self.generic_gain(self.detector, self.spec, self.sig)
        sig2, g2, t2 = self.generic_selection(self.detector, self.spec, self.sig)
        sig2, g3, t3 = self.generic_gain(self.detector, self.spec, sig2)
        sig2, g4, t4 = self.generic_blur(self.detector, self.spec, sig2)

        # Scaled individual path signals
        scaled_signal = p_path1 * sig1.signal + (1 - p_path1) * sig2.signal
        scaled_wiener = p_path1 * sig1.wiener + (1 - p_path1) * sig2.wiener

        # Cross-term calculation for Bernoulli split
        cross_covariance = -p_path1 * (1 - p_path1)
        cross_term = 2 * cross_covariance * g1 * g2 * g3 * t1 * t4 * (self.sig.wiener - self.sig.signal)

        analytic_wiener = scaled_wiener + cross_term

        np.testing.assert_allclose(combined_signal.signal, scaled_signal, atol=1e-6)
        np.testing.assert_allclose(combined_signal.wiener, analytic_wiener, atol=1e-6)

    def test_complex_cascade(self):
        # Create paths
        PathA = casymir.parallel.Path([self.generic_gain], name="Path A")
        PathB = casymir.parallel.Path([self.generic_selection, self.generic_gain], name="Path B")
        PathC = casymir.parallel.Path([self.generic_gain, self.generic_blur], name="Path C")

        # Nodes
        Node_Fluorescence = casymir.parallel.Node(node_type="Fork", name="Split 2")
        Node_Fluorescence.add(PathB)
        Node_Fluorescence.add(PathC)

        p = 0.3  # probability
        Node_Abs_vs_Fluor = casymir.parallel.Node(node_type="Bernoulli", name="Split 1")
        Node_Abs_vs_Fluor.add(PathA, probability=1 - p)
        Node_Abs_vs_Fluor.add(Node_Fluorescence, probability=p)

        combined_signal = casymir.parallel.apply_parallel_process(Node_Abs_vs_Fluor, self.sig, self.detector, self.spec)

        # Individual path calculations (analytic)
        sigA, gA, tA = self.generic_gain(self.detector, self.spec, self.sig)

        sigB, gB1, tB1 = self.generic_selection(self.detector, self.spec, self.sig)
        sigB, gB2, tB2 = self.generic_gain(self.detector, self.spec, sigB)

        sigC, gC1, tC1 = self.generic_gain(self.detector, self.spec, self.sig)
        sigC, gC2, tC2 = self.generic_blur(self.detector, self.spec, sigC)

        # Combined analytical signals and Wiener spectra
        signal_combined = (1 - p) * sigA.signal + p * (sigB.signal + sigC.signal)

        wiener_combined = ((1 - p) * sigA.wiener +
                           p * (sigB.wiener + sigC.wiener +
                                2 * self.sig.wiener * gB1 * gB2 * gC1 * gC2 * tB1 * tB2 * tC2))

        cross_covariance = -p * (1 - p)
        wiener_combined += 2 * cross_covariance * gA * (gB1 * gB2 * tB2 + gC1 * gC2 * tC2) * (
                self.sig.wiener - self.sig.signal)

        np.testing.assert_allclose(combined_signal.signal, signal_combined, atol=1e-6)
        np.testing.assert_allclose(combined_signal.wiener, wiener_combined, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
