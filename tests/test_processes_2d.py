import unittest

import numpy as np

from casymir.casymir import SignalND
from casymir.processes_2d import (
    add_electronic_noise_2d,
    cbct_ambr_ramp_filter_2d,
    cbct_discrete_ram_lak_filter_2d,
    dbt_discrete_ram_lak_filter_2d,
    detector_aliasing_2d,
    log_normalize_by_detector_response_2d,
    log_normalize_by_signal_2d,
)


class TestProcesses2D(unittest.TestCase):

    def test_detector_aliasing_folds_all_overlapping_harmonics(self):
        f = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
        S = np.ones((5, 5), dtype=np.float32)
        W = np.zeros_like(S)
        iv0 = int(np.argmin(np.abs(f)))
        W[:, iv0] = np.array([2.0, 3.0, 5.0, 7.0, 11.0], dtype=np.float32)
        sig = SignalND([f, f], S, W, mean_quanta=1.0)
        detector = type("DetectorStub", (), {"px_size": 1.0})()

        out = detector_aliasing_2d(detector, sig)

        iu0 = int(np.argmin(np.abs(f)))
        self.assertAlmostEqual(float(out.W[iu0, iv0]), 28.0)
        np.testing.assert_allclose(out.S, sig.S)

    def test_detector_aliasing_conserves_wiener_power_in_one_nyquist_cell(self):
        df = 0.125
        f = np.arange(-2.0, 2.0, df, dtype=np.float64)
        FU, FV = np.meshgrid(f, f, indexing="ij")
        S = np.ones(FU.shape, dtype=np.float64)
        W = 1.0 + 0.2 * FU + 0.1 * FV + 0.03 * FU * FV
        sig = SignalND([f, f], S, W, mean_quanta=1.0)
        detector = type("DetectorStub", (), {"px_size": 1.0})()

        out = detector_aliasing_2d(
            detector,
            sig,
            harmonics_u=2,
            harmonics_v=2,
        )

        one_nyquist_cell = (f >= -0.5) & (f < 0.5)
        presampled_power = float(np.sum(W) * df**2)
        aliased_power = float(
            np.sum(out.W[np.ix_(one_nyquist_cell, one_nyquist_cell)]) * df**2
        )
        self.assertAlmostEqual(aliased_power, presampled_power, places=12)

    def test_detector_aliasing_is_unchanged_after_all_overlapping_replicas(self):
        f = np.arange(-1.0, 1.0, 0.125, dtype=np.float64)
        FU, FV = np.meshgrid(f, f, indexing="ij")
        S = np.ones(FU.shape, dtype=np.float64)
        W = np.exp(-(FU**2 + FV**2))
        sig = SignalND([f, f], S, W, mean_quanta=1.0)
        detector = type("DetectorStub", (), {"px_size": 1.0})()

        one_replica = detector_aliasing_2d(
            detector,
            sig,
            harmonics_u=1,
            harmonics_v=1,
        )
        two_replicas = detector_aliasing_2d(
            detector,
            sig,
            harmonics_u=2,
            harmonics_v=2,
        )

        np.testing.assert_allclose(two_replicas.W, one_replica.W)

    def test_detector_response_log_transform_uses_air_kerma_response(self):
        f = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.ones((3, 3), dtype=np.float32) * 20.0
        W = np.ones_like(S) * 80.0
        sig = SignalND([f, f], S, W, mean_quanta=1.0)
        spectrum = type("SpectrumStub", (), {"dak": 2.0})()

        out, gain = log_normalize_by_detector_response_2d(sig, spectrum=spectrum, a=4.0, b=3.0)

        self.assertAlmostEqual(gain, 10.0)
        np.testing.assert_allclose(out.S, 2.0)
        np.testing.assert_allclose(out.W, 0.8)
        np.testing.assert_allclose(sig.S, 20.0)

    def test_signal_normalized_log_transform_uses_zero_frequency_signal(self):
        f = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.ones((3, 3), dtype=np.float32) * 5.0
        W = np.ones_like(S) * 20.0
        S[1, 1] = 10.0
        sig = SignalND([f, f], S, W, mean_quanta=1.0)

        out, gain = log_normalize_by_signal_2d(sig)

        self.assertAlmostEqual(gain, 10.0)
        self.assertAlmostEqual(float(out.S[1, 1]), 1.0)
        np.testing.assert_allclose(out.W, 0.2)
        np.testing.assert_allclose(sig.S[1, 1], 10.0)

    def test_electronic_noise_adds_constant_wiener_variance(self):
        f = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.ones((3, 3), dtype=np.float32)
        W = np.ones_like(S) * 3.0
        sig = SignalND([f, f], S, W, mean_quanta=1.0)

        out, noise_var = add_electronic_noise_2d(sig, noise_std=4.0)

        self.assertAlmostEqual(noise_var, 16.0)
        np.testing.assert_allclose(out.W, 19.0)
        np.testing.assert_allclose(sig.W, 3.0)

    def test_cbct_ambr_name_is_compatible_with_discrete_ram_lak_name(self):
        self.assertIs(cbct_ambr_ramp_filter_2d, cbct_discrete_ram_lak_filter_2d)

    def test_dbt_discrete_ram_lak_uses_radial_dbt_scaling(self):
        f = np.arange(-0.5, 0.5, 0.125, dtype=np.float64)
        S = np.ones((len(f), len(f)), dtype=np.float64)
        sig = SignalND([f, f], S, S.copy(), mean_quanta=1.0)
        detector = type("DetectorStub", (), {"px_size": 1.0})()
        theta_total_rad = 0.3

        cbct = cbct_discrete_ram_lak_filter_2d(sig, detector=detector)
        dbt = dbt_discrete_ram_lak_filter_2d(
            sig,
            detector=detector,
            theta_total_rad=theta_total_rad,
            theta_rad=0.0,
        )

        fr_ny = 1.0 / (2.0 * detector.px_size)
        expected = (2.0 * np.tan(theta_total_rad) / fr_ny) * cbct
        np.testing.assert_allclose(dbt, expected)


if __name__ == "__main__":
    unittest.main()
