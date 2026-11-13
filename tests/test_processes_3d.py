import unittest

import numpy as np

from casymir.casymir import SignalND, SignalStack
from casymir.processes_3d import (
    apply_voxel_sampling_3d,
    dbt_slice_thickness_filter,
    depth_aliasing_dbt_3d,
    map_dbt_stack_to_volume,
    map_cbct_stack_to_volume,
    projection_stack_from_signal,
    scan_aliasing_dbt_3d,
    voxel_aliasing_3d,
)


class TestProcesses3D(unittest.TestCase):

    def test_signalnd_deterministic_blur_updates_mean_quanta_like_1d_signal(self):
        f = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.ones((3, 3), dtype=np.float32)
        W = np.ones_like(S)
        sig = SignalND([f, f], S, W, mean_quanta=7.0)

        H = np.ones_like(S) * 2.0
        sig.deterministic_blur(H)

        self.assertAlmostEqual(sig.mean_quanta, 14.0)
        np.testing.assert_allclose(sig.S, 2.0)
        np.testing.assert_allclose(sig.W, 4.0)

    def test_projection_stack_from_signal_repeats_projection_for_each_angle(self):
        f = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.arange(9, dtype=np.float32).reshape(3, 3)
        W = S + 10.0
        sig = SignalND([f, f], S, W, mean_quanta=3.0)
        angles = np.array([-0.1, 0.0, 0.1])

        stack = projection_stack_from_signal(sig, angles)

        self.assertEqual(stack.Nv, 3)
        np.testing.assert_allclose(stack.angles, angles)
        np.testing.assert_allclose(stack.S[0], S)
        np.testing.assert_allclose(stack.S[2], S)
        np.testing.assert_allclose(stack.W[1], W)
        np.testing.assert_allclose(stack.mean_quanta, 3.0)

    def test_dbt_slice_thickness_filter_is_windowed(self):
        fy = np.linspace(-0.5, 0.5, 101, dtype=np.float32)
        Hst = dbt_slice_thickness_filter(
            fy,
            theta_rad=0.0,
            detector_pitch_mm=0.259,
            theta_total_rad=np.deg2rad(50.0),
            width=0.05,
        )

        fr_ny = 1.0 / (2.0 * 0.259)
        cutoff = 0.05 * fr_ny
        outside = np.abs(fy) > cutoff

        self.assertAlmostEqual(float(Hst[np.argmin(np.abs(fy))]), 1.0)
        np.testing.assert_allclose(Hst[outside], 0.0)

    def test_dbt_mapping_uses_reconstruction_axis_order(self):
        fx = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        fz_detector = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        fy_depth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        S = np.arange(fx.size * fz_detector.size, dtype=np.float32).reshape(fx.size, fz_detector.size) + 1.0
        W = S * 10.0

        sig = SignalND([fx, fz_detector], S, W, mean_quanta=5.0)
        stack = SignalStack(fx, fz_detector, Nv=1, angles_rad=np.array([0.0], dtype=float))
        stack.append(sig, angle_rad=0.0)

        vol = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="constant",
            apply_slice_thickness=False,
            progress_label=None,
        )

        self.assertEqual(vol.S.shape, (fx.size, fy_depth.size, fz_detector.size))
        np.testing.assert_array_equal(vol.fx, fx)
        np.testing.assert_array_equal(vol.fy, fy_depth)
        np.testing.assert_array_equal(vol.fz, fz_detector)

        iy0 = int(np.argmin(np.abs(fy_depth)))
        np.testing.assert_allclose(vol.S[:, iy0, :], S)
        np.testing.assert_allclose(vol.W[:, iy0, :], W)

        off_depth = np.ones(fy_depth.size, dtype=bool)
        off_depth[iy0] = False
        np.testing.assert_allclose(vol.S[:, off_depth, :], 0.0)
        np.testing.assert_allclose(vol.W[:, off_depth, :], 0.0)

    def test_dbt_mapping_angular_kernel_broadens_with_in_plane_frequency(self):
        fx_projection = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
        fx_reconstruction = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
        fz_detector = np.array([0.0], dtype=np.float32)
        fy_depth = np.linspace(-1.0, 1.0, 21, dtype=np.float32)

        sig = SignalND(
            [fx_projection, fz_detector],
            np.ones((3, 1), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
            5.0,
        )
        stack = SignalStack(fx_projection, fz_detector, Nv=2, angles_rad=np.array([-0.1, 0.1], dtype=float))
        stack.append(sig, angle_rad=-0.1)
        stack.append(sig, angle_rad=0.1)

        vol = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            fx=fx_reconstruction,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="angular",
            apply_slice_thickness=False,
            progress_label=None,
        )

        ix_edge = 0
        ix_center = 1
        active_edge_bins = np.count_nonzero(vol.S[ix_edge, :, 0] > 1e-4)
        active_center_bins = np.count_nonzero(vol.S[ix_center, :, 0] > 1e-4)

        self.assertGreater(active_edge_bins, active_center_bins)

    def test_dbt_mapping_splat_toggle_controls_view_deposition(self):
        fx_projection = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
        fx_reconstruction = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
        fz_detector = np.array([0.0], dtype=np.float32)
        fy_depth = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
        angles = np.array([-0.1, 0.1], dtype=float)

        sig = SignalND(
            [fx_projection, fz_detector],
            np.ones((3, 1), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
            5.0,
        )
        stack = SignalStack(fx_projection, fz_detector, Nv=len(angles), angles_rad=angles)
        for angle in angles:
            stack.append(sig, angle_rad=angle)

        nearest = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            fx=fx_reconstruction,
            splat=False,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="angular",
            apply_slice_thickness=False,
            progress_label=None,
        )
        splatted = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            fx=fx_reconstruction,
            splat=True,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="angular",
            apply_slice_thickness=False,
            progress_label=None,
        )

        edge_ix = 0
        nearest_bins = np.count_nonzero(nearest.S[edge_ix, :, 0] > 1e-4)
        splatted_bins = np.count_nonzero(splatted.S[edge_ix, :, 0] > 1e-4)

        self.assertLessEqual(nearest_bins, len(angles))
        self.assertGreater(splatted_bins, nearest_bins)

    def test_dbt_mapping_angular_integral_weights_signal_and_noise(self):
        fx = np.array([0.0], dtype=np.float32)
        fz_detector = np.array([0.0], dtype=np.float32)
        fy_depth = np.array([0.0], dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=float)
        theta_total_rad = 0.5

        sig = SignalND(
            [fx, fz_detector],
            np.array([[3.0]], dtype=np.float32),
            np.array([[5.0]], dtype=np.float32),
            7.0,
        )
        stack = SignalStack(fx, fz_detector, Nv=2, angles_rad=angles)
        stack.append(sig, angle_rad=0.0)
        stack.append(sig, angle_rad=0.0)

        raw = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="constant",
            theta_total_rad=theta_total_rad,
            detector_pitch_mm=1.0,
            apply_slice_thickness=False,
            progress_label=None,
        )
        weighted = map_dbt_stack_to_volume(
            stack,
            fy_depth,
            kernel="gaussian",
            kernel_width=0.0,
            kernel_width_mode="constant",
            theta_total_rad=theta_total_rad,
            detector_pitch_mm=1.0,
            apply_slice_thickness=False,
            view_weighting="angular_integral",
            progress_label=None,
        )

        view_weight = theta_total_rad / len(angles)
        np.testing.assert_allclose(weighted.S, raw.S * view_weight)
        np.testing.assert_allclose(weighted.W, raw.W * (view_weight ** 2))

    def test_noise_aliasing_3d_folds_shifted_wiener_spectrum(self):
        fx = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        fy = fx.copy()
        fz = fx.copy()
        S = np.ones((7, 7, 7), dtype=np.float32)
        W = np.zeros_like(S)

        ix0 = int(np.argmin(np.abs(fx)))
        iy0 = int(np.argmin(np.abs(fy)))
        iz0 = int(np.argmin(np.abs(fz)))
        ix_alias = int(np.argmin(np.abs(fx - 1.0)))
        W[ix_alias, iy0, iz0] = 7.0

        vol = SignalND([fx, fy, fz], S, W, mean_quanta=1.0)
        aliased = voxel_aliasing_3d(
            vol,
            spacing_x_mm=1.0,
            spacing_y_mm=1.0,
            spacing_z_mm=1.0,
            harmonics_x=1,
            harmonics_y=0,
            harmonics_z=0,
        )

        self.assertEqual(aliased.S.shape, vol.S.shape)
        np.testing.assert_allclose(aliased.S, vol.S)
        self.assertAlmostEqual(float(aliased.W[ix0, iy0, iz0]), 7.0)

    def test_voxel_aliasing_3d_folds_all_overlapping_harmonics(self):
        f = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
        S = np.ones((5, 5, 5), dtype=np.float32)
        W = np.zeros_like(S)
        i0 = int(np.argmin(np.abs(f)))
        W[:, i0, i0] = np.array([2.0, 3.0, 5.0, 7.0, 11.0], dtype=np.float32)
        vol = SignalND([f, f, f], S, W, mean_quanta=1.0)

        out = voxel_aliasing_3d(
            vol,
            spacing_x_mm=1.0,
            spacing_y_mm=1.0,
            spacing_z_mm=1.0,
            harmonics_x=None,
            harmonics_y=0,
            harmonics_z=0,
        )

        self.assertAlmostEqual(float(out.W[i0, i0, i0]), 28.0)

    def test_noise_aliasing_3d_can_fold_to_output_nyquist_axes(self):
        fx_pre = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        fy_pre = fx_pre.copy()
        fz_pre = fx_pre.copy()
        fx_out = np.array([-0.5, 0.0, 0.5], dtype=np.float32)

        S = np.ones((7, 7, 7), dtype=np.float32)
        W = np.zeros_like(S)

        ix_alias = int(np.argmin(np.abs(fx_pre - 1.0)))
        iy0 = int(np.argmin(np.abs(fy_pre - 0.0)))
        iz0 = int(np.argmin(np.abs(fz_pre - 0.0)))
        W[ix_alias, iy0, iz0] = 5.0

        vol = SignalND([fx_pre, fy_pre, fz_pre], S, W, mean_quanta=1.0)
        aliased, folded_fraction = voxel_aliasing_3d(
            vol,
            spacing_x_mm=1.0,
            spacing_y_mm=1.0,
            spacing_z_mm=1.0,
            harmonics_x=1,
            harmonics_y=0,
            harmonics_z=0,
            fx_out=fx_out,
            fy_out=fx_out,
            fz_out=fx_out,
            return_folded_fraction=True,
        )

        ix0_out = int(np.argmin(np.abs(aliased.fx - 0.0)))
        iy0_out = int(np.argmin(np.abs(aliased.fy - 0.0)))
        iz0_out = int(np.argmin(np.abs(aliased.fz - 0.0)))

        self.assertEqual(aliased.W.shape, (3, 3, 3))
        np.testing.assert_array_equal(aliased.fx, fx_out)
        self.assertAlmostEqual(float(aliased.W[ix0_out, iy0_out, iz0_out]), 5.0)
        self.assertGreater(folded_fraction, 0.99)

    def test_apply_voxel_sampling_3d_uses_isotropic_voxel_size(self):
        fx_pre = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        S = np.ones((7, 7, 7), dtype=np.float32)
        W = np.zeros_like(S)

        ix_alias = int(np.argmin(np.abs(fx_pre - 1.0)))
        i0 = int(np.argmin(np.abs(fx_pre - 0.0)))
        W[ix_alias, i0, i0] = 5.0

        vol = SignalND([fx_pre, fx_pre, fx_pre], S, W, mean_quanta=1.0)
        aliased = apply_voxel_sampling_3d(
            vol,
            spacing_mm=1.0,
            harmonics_x=1,
            harmonics_y=0,
            harmonics_z=0,
        )

        np.testing.assert_allclose(aliased.fx, np.array([-0.5, 0.0, 0.5], dtype=np.float32))
        self.assertEqual(aliased.W.shape, (3, 3, 3))
        self.assertAlmostEqual(float(aliased.W[1, 1, 1]), 5.0)

    def test_noise_aliasing_depth_folds_only_depth_axis(self):
        fx = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        fy = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        fz = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        S = np.zeros((fx.size, fy.size, fz.size), dtype=np.float32)
        W = np.zeros_like(S)

        ix0 = int(np.argmin(np.abs(fx)))
        iy0 = int(np.argmin(np.abs(fy)))
        iy_alias = int(np.argmin(np.abs(fy - 1.0)))
        iz0 = int(np.argmin(np.abs(fz)))

        S[ix0, iy0, iz0] = 2.0
        S[ix0, iy_alias, iz0] = 99.0
        W[ix0, iy_alias, iz0] = 7.0

        vol = SignalND([fx, fy, fz], S, W, mean_quanta=1.0)
        aliased, folded_fraction = depth_aliasing_dbt_3d(
            vol,
            spacing_y_mm=1.0,
            harmonics_y=1,
            return_folded_fraction=True,
        )

        self.assertEqual(aliased.S.shape, (fx.size, 3, fz.size))
        np.testing.assert_array_equal(aliased.fx, fx)
        np.testing.assert_allclose(aliased.fy, np.array([-0.5, 0.0, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(aliased.fz, fz)
        self.assertAlmostEqual(float(aliased.W[ix0, 1, iz0]), 7.0)
        self.assertAlmostEqual(float(aliased.S[ix0, 1, iz0]), 2.0)
        self.assertGreater(folded_fraction, 0.99)

    def test_scan_aliasing_dbt_uses_automatic_harmonic_count(self):
        fx = np.linspace(-2.0, 2.0, 5, dtype=np.float32)
        fy = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        fz = fy.copy()
        S = np.ones((5, 3, 3), dtype=np.float32)
        W = np.zeros_like(S)
        ix0 = int(np.argmin(np.abs(fx)))
        iy0 = int(np.argmin(np.abs(fy)))
        iz0 = int(np.argmin(np.abs(fz)))
        W[:, iy0, iz0] = np.array([2.0, 3.0, 5.0, 7.0, 11.0], dtype=np.float32)
        vol = SignalND([fx, fy, fz], S, W, mean_quanta=1.0)

        out = scan_aliasing_dbt_3d(vol, spacing_x_mm=1.0)

        ix0_out = int(np.argmin(np.abs(out.fx)))
        self.assertAlmostEqual(float(out.W[ix0_out, iy0, iz0]), 28.0)

    def test_cbct_limited_angle_backprojection_is_wedge_shaped(self):
        fx = np.linspace(-1.0, 1.0, 161, dtype=np.float32)
        fz_detector = fx.copy()
        fz = np.array([0.0], dtype=np.float32)

        S = np.ones((fx.size, fz_detector.size), dtype=np.float32)
        W = np.ones_like(S)

        span_rad = np.deg2rad(50.0)
        angles = np.linspace(-0.5 * span_rad, 0.5 * span_rad, 181)
        stack = SignalStack(fx, fz_detector, Nv=angles.size, angles_rad=angles, dtype=np.float32)
        sig = SignalND([fx, fz_detector], S, W, mean_quanta=1.0)

        for angle in angles:
            stack.append(sig, angle_rad=angle)

        vol = map_cbct_stack_to_volume(
            stack,
            fz,
            backprojection_extent_mm=80.0,
            normalize_by_views=True,
            progress_label="CBCT limited-angle test",
        )

        A = vol.W[:, :, 0]

        def value_at_angle(alpha_deg, radius=0.55):
            alpha = np.deg2rad(alpha_deg)
            fx_target = radius * np.sin(alpha)
            fy_target = radius * np.cos(alpha)
            ix = int(np.argmin(np.abs(vol.fx - fx_target)))
            iy = int(np.argmin(np.abs(vol.fy - fy_target)))
            return float(A[ix, iy])

        inside = value_at_angle(20.0)
        outside = value_at_angle(60.0)

        self.assertGreater(inside, 0.0)
        self.assertLess(outside, 0.01 * inside)

    def test_cbct_accepts_anisotropic_depth_axis(self):
        fx = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
        fy_depth = np.linspace(-0.5, 0.5, 21, dtype=np.float32)
        fz_detector = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
        fz = np.array([0.0], dtype=np.float32)

        S = np.ones((fx.size, fz_detector.size), dtype=np.float32)
        W = np.ones_like(S)

        angles = np.array([0.0], dtype=float)
        stack = SignalStack(fx, fz_detector, Nv=1, angles_rad=angles, dtype=np.float32)
        stack.append(SignalND([fx, fz_detector], S, W, mean_quanta=1.0), angle_rad=0.0)

        vol = map_cbct_stack_to_volume(
            stack,
            fz,
            fy=fy_depth,
            backprojection_extent_mm=40.0,
            progress_label="CBCT anisotropic-axis test",
        )

        self.assertEqual(vol.S.shape, (fx.size, fy_depth.size, fz.size))
        np.testing.assert_array_equal(vol.fx, fx)
        np.testing.assert_array_equal(vol.fy, fy_depth)
        np.testing.assert_array_equal(vol.fz, fz)

    def test_cbct_tward_view_normalization_uses_k_squared_for_noise(self):
        fx = np.linspace(-0.1, 0.1, 3, dtype=np.float32)
        fz = np.array([0.0], dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
        magnification = 1.5

        stack = SignalStack(fx, fx, Nv=angles.size, angles_rad=angles, dtype=np.float32)
        sig = SignalND(
            [fx, fx],
            np.ones((fx.size, fx.size), dtype=np.float32),
            np.ones((fx.size, fx.size), dtype=np.float32),
            mean_quanta=1.0,
        )
        for angle in angles:
            stack.append(sig, angle_rad=angle)

        unnormalized = map_cbct_stack_to_volume(
            stack,
            fz,
            magnification=magnification,
            backprojection_extent_mm=40.0,
            normalize_by_views=False,
            progress_label=None,
        )
        normalized = map_cbct_stack_to_volume(
            stack,
            fz,
            magnification=magnification,
            backprojection_extent_mm=40.0,
            normalize_by_views=True,
            progress_label=None,
        )

        m = float(len(angles))
        k = np.pi * magnification / m
        np.testing.assert_allclose(normalized.S, unnormalized.S * (np.pi / m))
        np.testing.assert_allclose(normalized.W, unnormalized.W * (k * k))

    def test_cbct_mapper_can_apply_stage14_voxel_aliasing(self):
        fx = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        fz_detector = fx.copy()
        fz = fx.copy()

        S = np.ones((fx.size, fz_detector.size), dtype=np.float32)
        W = np.ones_like(S)

        angles = np.array([0.0], dtype=float)
        stack = SignalStack(fx, fz_detector, Nv=1, angles_rad=angles, dtype=np.float32)
        stack.append(SignalND([fx, fz_detector], S, W, mean_quanta=1.0), angle_rad=0.0)

        f_out = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        presampled = map_cbct_stack_to_volume(
            stack,
            fz,
            fy=fx,
            backprojection_extent_mm=40.0,
            progress_label="CBCT aliasing test",
        )
        vol = voxel_aliasing_3d(
            presampled,
            spacing_x_mm=1.0,
            spacing_y_mm=1.0,
            spacing_z_mm=1.0,
            harmonics_x=1,
            harmonics_y=1,
            harmonics_z=1,
            fx_out=f_out,
            fy_out=f_out,
            fz_out=f_out,
            progress_label="CBCT Stage 14 aliasing test",
        )

        self.assertEqual(vol.S.shape, (3, 3, 3))
        np.testing.assert_array_equal(vol.fx, f_out)
        np.testing.assert_array_equal(vol.fy, f_out)
        np.testing.assert_array_equal(vol.fz, f_out)
        self.assertTrue(np.all(np.isfinite(vol.W)))

    def test_cbct_half_scan_backprojection_is_angle_complete(self):
        fx = np.linspace(-1.0, 1.0, 161, dtype=np.float32)
        fz_detector = fx.copy()
        fz = np.array([0.0], dtype=np.float32)

        S = np.ones((fx.size, fz_detector.size), dtype=np.float32)
        W = np.ones_like(S)

        angles = np.linspace(np.deg2rad(-90.0), np.deg2rad(90.0), 181)
        stack = SignalStack(fx, fz_detector, Nv=angles.size, angles_rad=angles, dtype=np.float32)
        sig = SignalND([fx, fz_detector], S, W, mean_quanta=1.0)

        for angle in angles:
            stack.append(sig, angle_rad=angle)

        vol = map_cbct_stack_to_volume(
            stack,
            fz,
            backprojection_extent_mm=80.0,
            normalize_by_views=True,
            progress_label="CBCT half-scan test",
        )

        A = vol.W[:, :, 0]

        def value_at_angle(alpha_deg, radius=0.55):
            alpha = np.deg2rad(alpha_deg)
            fx_target = radius * np.sin(alpha)
            fy_target = radius * np.cos(alpha)
            ix = int(np.argmin(np.abs(vol.fx - fx_target)))
            iy = int(np.argmin(np.abs(vol.fy - fy_target)))
            return float(A[ix, iy])

        values = np.array([value_at_angle(a) for a in (0.0, 30.0, 60.0, 85.0)])
        self.assertLess(values.max() / values.min(), 1.05)

if __name__ == "__main__":
    unittest.main()
