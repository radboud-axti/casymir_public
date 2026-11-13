"""
Reconstruction-domain process blocks.

The 3D model starts from a stack of 2D projection-domain SignalND objects.
It then maps those views into a reconstructed frequency volume and
applies voxel sampling.

Coordinate convention for reconstructed volumes
-----------------------------------------------
x: detector-u / transverse scan direction
y: reconstructed depth for DBT; second transverse axis in the CBCT rotation plane
z: detector-v / longitudinal direction
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from casymir.casymir import SignalND, SignalStack
from casymir.processes_2d import sample_grid_2d

# Stack and axis helpers


def projection_stack_from_signal(
    signal: SignalND,
    angles_rad: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> SignalStack:
    """
    Process that builds a projection stack by reusing one 2D projection signal for every view.

    This is useful for the CBCT geometry where the detector cascade is identical for
    all views. For DBT, build the stack view by view when focal-spot or
    obliquity terms depend on the projection angle.

    :param signal: CASYMIR SignalND object in the 2D projection domain.
    :param angles_rad: Projection angles in radians.
    :param dtype: Numeric dtype used to store the stack arrays.

    :return: SignalStack object containing one projection-domain signal per view.
    """
    if signal.ndim != 2:
        raise ValueError("projection_stack_from_signal expects a 2D SignalND.")

    angles = np.asarray(angles_rad, dtype=float)
    stack = SignalStack(
        signal.fx, signal.fy, Nv=len(angles), angles_rad=angles, dtype=dtype
    )
    for angle in angles:
        stack.append(signal, angle_rad=float(angle))
    return stack


def centered_fft_axis(n: int, spacing_mm: float) -> np.ndarray:
    """
    Return a centered FFT frequency axis for a sampled real-space grid.

    :param n: Number of real-space samples.
    :param spacing_mm: Real-space sampling interval in millimeters.

    :return: Centered frequency axis in inverse millimeters.
    """
    return np.fft.fftshift(np.fft.fftfreq(int(n), d=float(spacing_mm))).astype(
        np.float32
    )


def nyquist_axis(axis: np.ndarray, spacing_mm: float) -> np.ndarray:
    """
    Select samples from a centered axis inside the Nyquist band defined by spacing_mm.

    :param axis: Centered frequency axis.
    :param spacing_mm: Real-space sampling interval in millimeters.

    :return: Frequency samples inside the corresponding Nyquist band.
    """
    fny = 1.0 / (2.0 * float(spacing_mm))
    axis = np.asarray(axis)
    return axis[(axis >= -fny) & (axis <= fny)]


def _sampling_spacings(
    spacing_mm: float | None,
    spacing_x_mm: float | None,
    spacing_y_mm: float | None,
    spacing_z_mm: float | None,
) -> tuple[float, float, float]:
    """Resolve isotropic or axis-specific sampling intervals."""
    if spacing_mm is not None:
        spacing_x_mm = spacing_mm if spacing_x_mm is None else spacing_x_mm
        spacing_y_mm = spacing_mm if spacing_y_mm is None else spacing_y_mm
        spacing_z_mm = spacing_mm if spacing_z_mm is None else spacing_z_mm

    if spacing_x_mm is None or spacing_y_mm is None or spacing_z_mm is None:
        raise ValueError(
            "Provide spacing_mm for isotropic sampling, or all of "
            "spacing_x_mm, spacing_y_mm, and spacing_z_mm."
        )

    return float(spacing_x_mm), float(spacing_y_mm), float(spacing_z_mm)


def _harmonic_count(fmax: float, fs: float) -> int:
    """Return the number of sampling replicas needed to cover the support."""
    return int(np.ceil(abs(float(fmax)) / abs(float(fs))))


# Sampling kernels


def sample_grid_3d(
    values: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    FXq: np.ndarray,
    FYq: np.ndarray,
    FZq: np.ndarray,
) -> np.ndarray:
    """
    Sample a 3D array on centered frequency axes.

    :param values: Values tabulated on the input frequency grid.
    :param fx: Input x frequency axis.
    :param fy: Input y frequency axis.
    :param fz: Input z frequency axis.
    :param FXq: Query frequencies along x.
    :param FYq: Query frequencies along y.
    :param FZq: Query frequencies along z.

    :return: Trilinearly interpolated values on the query grid.
    """
    values = np.asarray(values)
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    fz = np.asarray(fz)

    dx = float(fx[1] - fx[0])
    dy = float(fy[1] - fy[0])
    dz = float(fz[1] - fz[0])

    X = (FXq - fx[0]) / dx
    Y = (FYq - fy[0]) / dy
    Z = (FZq - fz[0]) / dz

    i0 = np.floor(X).astype(np.int64)
    j0 = np.floor(Y).astype(np.int64)
    k0 = np.floor(Z).astype(np.int64)
    valid = (
        (i0 >= 0)
        & (i0 < len(fx))
        & (j0 >= 0)
        & (j0 < len(fy))
        & (k0 >= 0)
        & (k0 < len(fz))
    )

    out = np.zeros_like(FXq, dtype=values.dtype)
    if not np.any(valid):
        return out

    i0v, j0v, k0v = i0[valid], j0[valid], k0[valid]
    i1v = np.minimum(i0v + 1, len(fx) - 1)
    j1v = np.minimum(j0v + 1, len(fy) - 1)
    k1v = np.minimum(k0v + 1, len(fz) - 1)

    tx = np.where(i1v == i0v, 0.0, X[valid] - i0v)
    ty = np.where(j1v == j0v, 0.0, Y[valid] - j0v)
    tz = np.where(k1v == k0v, 0.0, Z[valid] - k0v)

    c000 = values[i0v, j0v, k0v]
    c001 = values[i0v, j0v, k1v]
    c010 = values[i0v, j1v, k0v]
    c011 = values[i0v, j1v, k1v]
    c100 = values[i1v, j0v, k0v]
    c101 = values[i1v, j0v, k1v]
    c110 = values[i1v, j1v, k0v]
    c111 = values[i1v, j1v, k1v]

    c00 = (1.0 - tx) * c000 + tx * c100
    c01 = (1.0 - tx) * c001 + tx * c101
    c10 = (1.0 - tx) * c010 + tx * c110
    c11 = (1.0 - tx) * c011 + tx * c111
    c0 = (1.0 - ty) * c00 + ty * c10
    c1 = (1.0 - ty) * c01 + ty * c11
    out[valid] = (1.0 - tz) * c0 + tz * c1
    return out


def _sample_axis_3d(
    values: np.ndarray, axis_in: np.ndarray, axis_out: np.ndarray, axis: int
) -> np.ndarray:
    """Linearly interpolate a 3D array along one axis."""
    values = np.moveaxis(values, axis, 0)
    flat = values.reshape(values.shape[0], -1)
    out = np.zeros((len(axis_out), flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(axis_out, axis_in, flat[:, col], left=0.0, right=0.0)
    out = out.reshape((len(axis_out),) + values.shape[1:])
    return np.moveaxis(out, 0, axis)


# Voxel sampling


def voxel_aliasing_3d(
    volume: SignalND,
    *,
    spacing_mm: float | None = None,
    spacing_x_mm: float | None = None,
    spacing_y_mm: float | None = None,
    spacing_z_mm: float | None = None,
    harmonics_x: int | None = None,
    harmonics_y: int | None = None,
    harmonics_z: int | None = None,
    fx_out: np.ndarray | None = None,
    fy_out: np.ndarray | None = None,
    fz_out: np.ndarray | None = None,
    return_folded_fraction: bool = False,
    progress_label: str | None = None,
) -> SignalND | tuple[SignalND, float]:
    """
    Process that folds reconstructed-volume noise replicas created by voxel sampling.

    This is the 3D version of the original 1D CASYMIR aliasing block:

        W_alias(fx, fy, fz) =
            sum_kx sum_ky sum_kz W(fx + kx fs_x, fy + ky fs_y, fz + kz fs_z)

    The harmonic ranges include every shifted copy that can overlap the
    tabulated presampling support unless the caller supplies explicit limits.

    Only the Wiener spectrum is affected.

    :param volume: Presampling 3D SignalND object.
    :param spacing_mm: Isotropic voxel spacing in millimeters.
    :param spacing_x_mm: Voxel spacing along reconstructed x in millimeters.
    :param spacing_y_mm: Voxel spacing along reconstructed y in millimeters.
    :param spacing_z_mm: Voxel spacing along reconstructed z in millimeters.
    :param harmonics_x: Number of sampling replicas to include along x. If None, the range is inferred.
    :param harmonics_y: Number of sampling replicas to include along y. If None, the range is inferred.
    :param harmonics_z: Number of sampling replicas to include along z. If None, the range is inferred.
    :param fx_out: Output frequency axis along x.
    :param fy_out: Output frequency axis along y.
    :param fz_out: Output frequency axis along z.
    :param return_folded_fraction: If True, also return the fraction of folded Wiener-spectrum energy.
    :param progress_label: Optional label for the progress bar.

    :return: SignalND object with voxel-sampled Wiener spectrum, or tuple with SignalND and folded fraction.
    """
    if volume.ndim != 3:
        raise ValueError("voxel_aliasing_3d expects a 3D SignalND.")

    sx, sy, sz = _sampling_spacings(
        spacing_mm, spacing_x_mm, spacing_y_mm, spacing_z_mm
    )

    fx = volume.fx.astype(np.float32)
    fy = volume.fy.astype(np.float32)
    fz = volume.fz.astype(np.float32)
    fxo = fx if fx_out is None else np.asarray(fx_out, dtype=np.float32)
    fyo = fy if fy_out is None else np.asarray(fy_out, dtype=np.float32)
    fzo = fz if fz_out is None else np.asarray(fz_out, dtype=np.float32)
    FX, FY, FZ = np.meshgrid(fxo, fyo, fzo, indexing="ij")

    fsx, fsy, fsz = 1.0 / sx, 1.0 / sy, 1.0 / sz
    if harmonics_x is None:
        harmonics_x = _harmonic_count(max(abs(fx[0]), abs(fx[-1])), fsx)
    if harmonics_y is None:
        harmonics_y = _harmonic_count(max(abs(fy[0]), abs(fy[-1])), fsy)
    if harmonics_z is None:
        harmonics_z = _harmonic_count(max(abs(fz[0]), abs(fz[-1])), fsz)

    shifts = [
        (kx, ky, kz)
        for kx in range(-harmonics_x, harmonics_x + 1)
        for ky in range(-harmonics_y, harmonics_y + 1)
        for kz in range(-harmonics_z, harmonics_z + 1)
    ]
    iterator = (
        tqdm(shifts, desc=progress_label, unit="shift") if progress_label else shifts
    )

    W = np.zeros((len(fxo), len(fyo), len(fzo)), dtype=np.float32)
    W_folded = np.zeros_like(W)
    for kx, ky, kz in iterator:
        sampled = sample_grid_3d(
            volume.W, fx, fy, fz, FX + kx * fsx, FY + ky * fsy, FZ + kz * fsz
        )
        W += sampled
        if (kx, ky, kz) != (0, 0, 0):
            W_folded += sampled

    S = sample_grid_3d(volume.S, fx, fy, fz, FX, FY, FZ)
    out = SignalND([fxo, fyo, fzo], S, W, volume.mean_quanta)

    if return_folded_fraction:
        frac = float(np.nansum(W_folded) / (np.nansum(W) + 1e-30))
        return out, frac
    return out


def apply_voxel_sampling_3d(
    volume: SignalND,
    *,
    spacing_mm: float | None = None,
    spacing_x_mm: float | None = None,
    spacing_y_mm: float | None = None,
    spacing_z_mm: float | None = None,
    harmonics_x: int | None = None,
    harmonics_y: int | None = None,
    harmonics_z: int | None = None,
    fx_out: np.ndarray | None = None,
    fy_out: np.ndarray | None = None,
    fz_out: np.ndarray | None = None,
    progress_label: str | None = None,
) -> SignalND:
    """
    Process that samples a presampling reconstruction volume on a voxel grid.

    :param volume: Presampling 3D SignalND object.
    :param spacing_mm: Isotropic voxel spacing in millimeters.
    :param spacing_x_mm: Voxel spacing along reconstructed x in millimeters.
    :param spacing_y_mm: Voxel spacing along reconstructed y in millimeters.
    :param spacing_z_mm: Voxel spacing along reconstructed z in millimeters.
    :param harmonics_x: Number of sampling replicas to include along x. If None, the range is inferred.
    :param harmonics_y: Number of sampling replicas to include along y. If None, the range is inferred.
    :param harmonics_z: Number of sampling replicas to include along z. If None, the range is inferred.
    :param fx_out: Output frequency axis along x.
    :param fy_out: Output frequency axis along y.
    :param fz_out: Output frequency axis along z.
    :param progress_label: Optional label for the progress bar.

    :return: Voxel-sampled 3D SignalND object.
    """
    sx, sy, sz = _sampling_spacings(
        spacing_mm, spacing_x_mm, spacing_y_mm, spacing_z_mm
    )
    fx_out = nyquist_axis(volume.fx, sx) if fx_out is None else fx_out
    fy_out = nyquist_axis(volume.fy, sy) if fy_out is None else fy_out
    fz_out = nyquist_axis(volume.fz, sz) if fz_out is None else fz_out
    return voxel_aliasing_3d(
        volume,
        spacing_x_mm=sx,
        spacing_y_mm=sy,
        spacing_z_mm=sz,
        harmonics_x=harmonics_x,
        harmonics_y=harmonics_y,
        harmonics_z=harmonics_z,
        fx_out=fx_out,
        fy_out=fy_out,
        fz_out=fz_out,
        progress_label=progress_label,
    )


def depth_aliasing_dbt_3d(
    volume: SignalND,
    *,
    spacing_y_mm: float,
    harmonics_y: int | None = None,
    fy_out: np.ndarray | None = None,
    return_folded_fraction: bool = False,
) -> SignalND | tuple[SignalND, float]:
    """
    Process that folds DBT noise aliases along reconstructed depth only.

    This is the same Dirac-comb sum as voxel_aliasing_3d, restricted to the
    reconstructed depth axis for DBT.

    :param volume: Presampling 3D SignalND object.
    :param spacing_y_mm: Voxel spacing along reconstructed depth in millimeters.
    :param harmonics_y: Number of sampling replicas to include along depth. If None, the range is inferred.
    :param fy_out: Output reconstructed depth frequency axis.
    :param return_folded_fraction: If True, also return the fraction of folded Wiener-spectrum energy.

    :return: SignalND object with depth-aliased Wiener spectrum, or tuple with SignalND and folded fraction.
    """
    if fy_out is None:
        fy_out = nyquist_axis(volume.fy, spacing_y_mm)
    fsy = 1.0 / float(spacing_y_mm)
    if harmonics_y is None:
        harmonics_y = _harmonic_count(max(abs(volume.fy[0]), abs(volume.fy[-1])), fsy)

    W = np.zeros((len(volume.fx), len(fy_out), len(volume.fz)), dtype=np.float32)
    W_folded = np.zeros_like(W) if return_folded_fraction else None
    for ky in range(-int(harmonics_y), int(harmonics_y) + 1):
        sampled = _sample_axis_3d(volume.W, volume.fy, fy_out + ky * fsy, axis=1)
        W += sampled
        if ky != 0 and W_folded is not None:
            W_folded += sampled

    S = _sample_axis_3d(volume.S, volume.fy, fy_out, axis=1)
    out = SignalND([volume.fx, fy_out, volume.fz], S, W, volume.mean_quanta)
    if return_folded_fraction:
        frac = float(np.nansum(W_folded) / (np.nansum(W) + 1e-30))
        return out, frac
    return out


def scan_aliasing_dbt_3d(
    volume: SignalND,
    *,
    spacing_x_mm: float,
    harmonics_x: int | None = None,
    fx_out: np.ndarray | None = None,
    return_folded_fraction: bool = False,
) -> SignalND | tuple[SignalND, float]:
    """
    Process that folds DBT noise aliases along the scan direction only.

    This is the same Dirac-comb sum as voxel_aliasing_3d, restricted to the
    scan axis for DBT.

    :param volume: Presampling 3D SignalND object.
    :param spacing_x_mm: Voxel spacing along reconstructed scan direction in millimeters.
    :param harmonics_x: Number of sampling replicas to include along scan direction. If None, the range is inferred.
    :param fx_out: Output scan-direction frequency axis.
    :param return_folded_fraction: If True, also return the fraction of folded Wiener-spectrum energy.

    :return: SignalND object with scan-aliased Wiener spectrum, or tuple with SignalND and folded fraction.
    """
    if fx_out is None:
        fx_out = nyquist_axis(volume.fx, spacing_x_mm)
    fsx = 1.0 / float(spacing_x_mm)
    if harmonics_x is None:
        harmonics_x = _harmonic_count(max(abs(volume.fx[0]), abs(volume.fx[-1])), fsx)

    W = np.zeros((len(fx_out), len(volume.fy), len(volume.fz)), dtype=np.float32)
    W_folded = np.zeros_like(W) if return_folded_fraction else None
    for kx in range(-int(harmonics_x), int(harmonics_x) + 1):
        sampled = _sample_axis_3d(volume.W, volume.fx, fx_out + kx * fsx, axis=0)
        W += sampled
        if kx != 0 and W_folded is not None:
            W_folded += sampled

    S = _sample_axis_3d(volume.S, volume.fx, fx_out, axis=0)
    out = SignalND([fx_out, volume.fy, volume.fz], S, W, volume.mean_quanta)
    if return_folded_fraction:
        frac = float(np.nansum(W_folded) / (np.nansum(W) + 1e-30))
        return out, frac
    return out


# DBT reconstruction


def dbt_slice_thickness_filter(
    f_depth: np.ndarray,
    *,
    theta_rad: float,
    detector_pitch_mm: float,
    theta_total_rad: float,
    width: float = 0.05,
) -> np.ndarray:
    """
    Process that creates a DBT slice thickness Hanning window on reconstructed depth frequency.

    :param f_depth: Reconstructed-depth frequency axis.
    :param theta_rad: Projection angle in radians.
    :param detector_pitch_mm: Detector pixel pitch in millimeters.
    :param theta_total_rad: Total DBT angular span in radians.
    :param width: Slice-thickness window width relative to radial Nyquist frequency.

    :return: Slice-thickness transfer function.
    """
    f_depth = np.asarray(f_depth, dtype=np.float32)
    fr_ny = abs(1.0 / (2.0 * float(detector_pitch_mm) * np.cos(theta_rad)))
    inside = (np.abs(f_depth) <= width * fr_ny) & (
        np.abs(f_depth) <= np.tan(theta_total_rad) * fr_ny
    )

    H = np.zeros_like(f_depth, dtype=np.float32)
    if np.any(inside):
        H[inside] = 0.5 * (1.0 + np.cos(np.pi * f_depth[inside] / (width * fr_ny)))
    return H


def _kernel_weights(
    center: float, axis: np.ndarray, width: float, kind: str, truncate: float
) -> np.ndarray:
    """
    Interpolation kernels used by the frequency-domain reconstruction mappers.
    """
    axis = np.asarray(axis, dtype=np.float32)
    if width <= 0:
        weights = np.zeros_like(axis)
        weights[int(np.argmin(np.abs(axis - center)))] = 1.0
        return weights

    x = axis - float(center)
    if kind == "gaussian":
        mask = np.abs(x) <= truncate * width
        weights = np.zeros_like(axis)
        weights[mask] = np.exp(-0.5 * (x[mask] / width) ** 2)
    elif kind == "triangular":
        weights = np.maximum(1.0 - np.abs(x) / width, 0.0)
    elif kind == "sinc":
        mask = np.abs(x) <= truncate * width
        weights = np.zeros_like(axis)
        weights[mask] = np.sinc(x[mask] / width)
    else:
        raise ValueError("kernel must be 'gaussian', 'triangular', or 'sinc'.")

    norm = float(np.sum(weights))
    if norm != 0.0:
        weights = weights / norm
    return weights.astype(np.float32)


def _dbt_depth_kernel_widths(
    fx: np.ndarray,
    f_depth: np.ndarray,
    angles_rad: np.ndarray,
    *,
    mode: str,
    kernel_width: float | None,
) -> np.ndarray:
    """
    Choose the depth frequency kernel width for each in-plane frequency.

    In DBT, a projection sample at frequency fx lands at fy = fx tan(theta).
    Neighboring views therefore separate more as |fx| grows.  The angular mode
    follows that sampling geometry while keeping a small frequency-bin floor.
    """
    fx = np.asarray(fx, dtype=np.float32)
    df_depth = float(np.mean(np.diff(f_depth))) if len(f_depth) > 1 else 1.0
    width_floor = 0.5 * df_depth if kernel_width is None else float(kernel_width)

    if mode == "constant":
        width = df_depth if kernel_width is None else float(kernel_width)
        return np.full_like(fx, width, dtype=np.float32)

    if mode == "angular":
        if len(angles_rad) > 1:
            dtheta = float(np.mean(np.diff(np.sort(angles_rad))))
        else:
            dtheta = 0.0
        angular_width = np.abs(fx) * np.sin(abs(dtheta))
        return np.maximum(width_floor, angular_width).astype(np.float32)

    raise ValueError("kernel_width_mode must be 'angular' or 'constant'.")


def _dbt_view_weights(
    angles_rad: np.ndarray, theta_total_rad: float, mode: str
) -> np.ndarray:
    """
    Weights for turning a discrete DBT view sum into an angular integral.

    Signal amplitudes are linear in the view weight. Independent Wiener spectra
    are quadratic in that same weight, so use w_i for S and w_i**2 for W.
    """
    angles = np.asarray(angles_rad, dtype=float)
    mode = str(mode).lower()

    if mode == "sum":
        return np.ones(len(angles), dtype=np.float32)

    if theta_total_rad <= 0.0:
        raise ValueError(
            "theta_total_rad must be positive for angular DBT view weighting."
        )

    if mode == "angular_integral":
        return np.full(
            len(angles), float(theta_total_rad) / len(angles), dtype=np.float32
        )

    if mode == "trapezoid":
        if len(angles) == 1:
            return np.array([float(theta_total_rad)], dtype=np.float32)

        order = np.argsort(angles)
        sorted_angles = angles[order]
        sorted_weights = np.empty(len(angles), dtype=float)
        sorted_weights[0] = 0.5 * (sorted_angles[1] - sorted_angles[0])
        sorted_weights[-1] = 0.5 * (sorted_angles[-1] - sorted_angles[-2])
        sorted_weights[1:-1] = 0.5 * (sorted_angles[2:] - sorted_angles[:-2])

        angular_span = sorted_angles[-1] - sorted_angles[0]
        if angular_span > 0.0:
            sorted_weights *= float(theta_total_rad) / angular_span

        weights = np.empty_like(sorted_weights)
        weights[order] = sorted_weights
        return weights.astype(np.float32)

    raise ValueError(
        "view_weighting must be 'sum', 'angular_integral', or 'trapezoid'."
    )


def map_dbt_stack_to_volume(
    stack: SignalStack,
    f_depth: np.ndarray,
    *,
    fx: np.ndarray | None = None,
    splat: bool = True,
    kernel: str = "gaussian",
    kernel_width: float | None = None,
    kernel_width_mode: str = "angular",
    slice_width: float = 0.05,
    theta_total_rad: float | None = None,
    detector_pitch_mm: float | None = None,
    apply_slice_thickness: bool = True,
    view_weighting: str = "sum",
    progress_label: str | None = "DBT reconstruction",
) -> SignalND:
    """
    Process that maps a DBT projection stack into a 3D reconstruction-frequency volume.

    Set ``splat=False`` for pure nearest-bin deposition of each view. The
    default ``splat=True`` spreads each view with the selected kernel, which
    gives a more continuous reconstruction-domain volume on practical grids.

    The default angular kernel width follows the DBT sampling geometry: samples
    from adjacent projection angles are farther apart in reconstructed-depth
    frequency as |fx| increases.  Use ``kernel_width_mode="constant"`` when a
    fixed-width mapper is useful for controlled numerical experiments.

    ``view_weighting`` controls how the discrete projection views represent the
    angular reconstruction integral. The default ``"sum"`` preserves raw
    superposition. ``"angular_integral"`` gives each view a uniform
    theta_total / N weight. ``"trapezoid"`` uses the actual angle spacing. In
    the weighted modes, signal uses w_i and noise power uses w_i**2.

    :param stack: SignalStack object containing DBT projection-domain views.
    :param f_depth: Reconstructed-depth frequency axis.
    :param fx: Output reconstructed x frequency axis. If None, stack.fx is used.
    :param splat: If True, spread each view with the selected kernel. If False, use nearest-bin deposition.
    :param kernel: Deposition kernel. Options are 'gaussian', 'triangular', and 'sinc'.
    :param kernel_width: Deposition kernel width. If None, the width is inferred from the selected mode.
    :param kernel_width_mode: Kernel width rule. Options are 'angular' and 'constant'.
    :param slice_width: DBT slice-thickness window width.
    :param theta_total_rad: Total DBT angular span in radians.
    :param detector_pitch_mm: Detector pixel pitch in millimeters.
    :param apply_slice_thickness: If True, apply the DBT slice-thickness filter.
    :param view_weighting: View weighting mode. Options are 'sum', 'angular_integral', and 'trapezoid'.
    :param progress_label: Optional label for the progress bar.

    :return: Reconstructed 3D SignalND object.
    """
    f_depth = np.asarray(f_depth, dtype=np.float32)
    fx_in = stack.fx.astype(np.float32)
    fx = fx_in if fx is None else np.asarray(fx, dtype=np.float32)
    fz = stack.fy.astype(np.float32)
    angles = np.asarray(stack.angles, dtype=float)

    if theta_total_rad is None:
        theta_total_rad = (
            float(np.max(angles) - np.min(angles)) if len(angles) > 1 else 1.0
        )
    if detector_pitch_mm is None:
        detector_pitch_mm = 1.0 / (2.0 * max(abs(fx_in[0]), abs(fx_in[-1])))

    view_weights = _dbt_view_weights(angles, theta_total_rad, view_weighting)
    kernel_widths = _dbt_depth_kernel_widths(
        fx,
        f_depth,
        angles,
        mode=kernel_width_mode,
        kernel_width=kernel_width,
    )

    S = np.zeros((len(fx), len(f_depth), len(fz)), dtype=np.float32)
    W = np.zeros_like(S)
    iterator = (
        tqdm(range(len(angles)), desc=progress_label, unit="view")
        if progress_label
        else range(len(angles))
    )

    for i in iterator:
        theta = float(angles[i])
        view_weight = float(view_weights[i])
        c, s = np.cos(theta), np.sin(theta)
        fu_query = fx / max(c, 1e-9)
        depth_center = fu_query * s

        view_S = np.asarray(
            [
                np.interp(fu_query, fx_in, stack.S[i, :, iz], left=0.0, right=0.0)
                for iz in range(len(fz))
            ]
        ).T
        view_W = np.asarray(
            [
                np.interp(fu_query, fx_in, stack.W[i, :, iz], left=0.0, right=0.0)
                for iz in range(len(fz))
            ]
        ).T

        if apply_slice_thickness:
            H_depth = dbt_slice_thickness_filter(
                f_depth,
                theta_rad=theta,
                detector_pitch_mm=detector_pitch_mm,
                theta_total_rad=theta_total_rad,
                width=slice_width,
            )
        else:
            H_depth = np.ones_like(f_depth)

        for ix, center in enumerate(depth_center):
            width = float(kernel_widths[ix]) if splat else 0.0
            weights = _kernel_weights(center, f_depth, width, kernel, truncate=3.0)
            S[ix, :, :] += (
                view_weight * weights[:, None] * view_S[ix, None, :] * H_depth[:, None]
            )
            W[ix, :, :] += (
                (view_weight**2)
                * weights[:, None]
                * view_W[ix, None, :]
                * (H_depth[:, None] ** 2)
            )

    return SignalND([fx, f_depth, fz], S, W, float(np.mean(stack.mean_quanta)))


# CBCT reconstruction


def map_cbct_stack_to_volume(
    stack: SignalStack,
    fz: np.ndarray,
    *,
    fx: np.ndarray | None = None,
    fy: np.ndarray | None = None,
    magnification: float = 1.0,
    backprojection_extent_mm: float = 100.0,
    normalize_by_views: bool = False,
    progress_label: str | None = "CBCT backprojection",
) -> SignalND:
    """
    Process that maps a circular-orbit CBCT projection stack into a 3D reconstruction-frequency volume.

    This is the explicit finite-view Stage-13 backprojection block. It keeps the
    ramp/apodization/interpolation filters outside this function so the user can
    see and stack those blocks in the projection-domain chain.

    :param stack: SignalStack object containing CBCT projection-domain views.
    :param fz: Output longitudinal frequency axis.
    :param fx: Output reconstructed x frequency axis. If None, stack.fx is used.
    :param fy: Output reconstructed y frequency axis. If None, fx is used.
    :param magnification: System magnification.
    :param backprojection_extent_mm: Finite backprojection extent in millimeters.
    :param normalize_by_views: If True, apply Tward's Stage-13 normalization
        k = pi * magnification / number_of_views to signal and k squared to
        independent-view noise power.
    :param progress_label: Optional label for the progress bar.

    :return: Reconstructed 3D SignalND object.
    """
    fu_axis = stack.fx.astype(np.float32)
    fv_axis = stack.fy.astype(np.float32)
    fx = fu_axis if fx is None else np.asarray(fx, dtype=np.float32)
    fy = fx.copy() if fy is None else np.asarray(fy, dtype=np.float32)
    fz = np.asarray(fz, dtype=np.float32)

    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
    S = np.zeros((len(fx), len(fy), len(fz)), dtype=np.float32)
    W = np.zeros_like(S)

    M = float(magnification)
    inv_M = 1.0 / M
    d = float(backprojection_extent_mm)
    angles = np.asarray(stack.angles, dtype=float)
    iterator = (
        tqdm(range(len(angles)), desc=progress_label, unit="view")
        if progress_label
        else range(len(angles))
    )

    for i in iterator:
        theta = float(angles[i])
        c, s = np.cos(theta), np.sin(theta)

        f_xi = FX * c + FY * s
        f_yi = -FX * s + FY * c
        T13 = d * np.sinc(d * f_xi)

        S12M = np.empty_like(S)
        W12M = np.empty_like(W)
        for iz in range(len(fz)):
            FUq = f_yi[:, :, iz] * inv_M
            FVq = FZ[:, :, iz] * inv_M
            S12M[:, :, iz] = sample_grid_2d(stack.S[i], fu_axis, fv_axis, FUq, FVq) / (
                M * M
            )
            W12M[:, :, iz] = sample_grid_2d(stack.W[i], fu_axis, fv_axis, FUq, FVq) / (
                M * M
            )

        S += S12M * T13
        W += W12M * (T13**2) / d

    if normalize_by_views and len(angles) > 0:
        # Tward and Siewerdsen, Med Phys 35(12), Appendix A:
        # k = pi*M/m. Independent views combine in quadrature, so the
        # accumulated NPS is multiplied by k^2, not M^2/m.
        k = np.pi * M / float(len(angles))
        S *= k
        W *= k * k
    else:
        S *= M

    return SignalND([fx, fy, fz], S, W, float(np.mean(stack.mean_quanta)))


