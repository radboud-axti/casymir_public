# Module: processes_3d

import numpy as np
from tqdm import tqdm

from casymir.casymir import SignalND, SignalStack
from casymir.processes_2d import _bilinear_sample2d


# ============================================================================
# Shared kernel utilities
# ============================================================================

def gaussian_weights(
    center: float,
    axis: np.ndarray,
    sigma: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Gaussian kernel (normalized).
    """
    axis = np.asarray(axis, dtype=np.float32)

    if sigma <= 0:
        j = int(np.argmin(np.abs(axis - center)))
        w = np.zeros_like(axis, dtype=np.float32)
        w[j] = 1.0
        return w

    lo = center - truncate * sigma
    hi = center + truncate * sigma
    m = (axis >= lo) & (axis <= hi)
    x = axis[m]

    w = np.exp(-0.5 * ((x - center) / sigma) ** 2, dtype=np.float64).astype(np.float32)
    s = float(w.sum())

    W = np.zeros_like(axis, dtype=np.float32)
    if s > 0:
        W[m] = w / s
    return W


def sinc_weights(
    center: float,
    axis: np.ndarray,
    width: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Truncated sinc kernel.
    """
    axis = np.asarray(axis, dtype=np.float32)
    x = axis - center
    w = np.sinc(x / width)
    m = np.abs(x) <= truncate * width

    W = np.zeros_like(axis, dtype=np.float32)
    W[m] = w[m]

    s = float(np.sum(np.abs(W)))
    if s > 0:
        W /= s
    return W


def triangular_weights(
    center: float,
    axis: np.ndarray,
    width: float,
) -> np.ndarray:
    """
    Triangular kernel (normalized).
    """
    axis = np.asarray(axis, dtype=np.float32)
    x = np.abs(axis - center)
    w = np.maximum(1.0 - x / width, 0.0)

    s = float(np.sum(w))
    if s > 0:
        w /= s
    return w.astype(np.float32)


_INTERP_KERNELS = {
    "gaussian": gaussian_weights,
    "sinc": sinc_weights,
    "triangular": triangular_weights,
}


def _get_interp_kernel(kernel: str):
    if kernel not in _INTERP_KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Available: {list(_INTERP_KERNELS.keys())}"
        )
    return _INTERP_KERNELS[kernel]


# ============================================================================
# DBT-specific functions
# ============================================================================

def slice_thickness_filter_dbt(
    fz: np.ndarray,
    *,
    theta_rad: float,
    px_size_mm: float,
    Theta_rad: float,
    B: float = 0.05,
) -> np.ndarray:
    """
    Slice-thickness window H_ST(fz) for DBT recon, defined as a Hanning window.
    """
    fz = np.asarray(fz, dtype=np.float32)
    c = np.cos(theta_rad)

    fr_ny = np.abs(1.0 / (2.0 * float(px_size_mm) * c))

    mask = (np.abs(fz) <= B * fr_ny) & (np.abs(fz) <= np.tan(Theta_rad) * fr_ny)
    Hst = np.ones_like(fz, dtype=np.float32)

    if np.any(mask):
        Hst[mask] = 0.5 * (1.0 + np.cos((np.pi * fz[mask]) / (B * fr_ny)))

    return Hst


def map_stack_to_volume_dbt(
    stack: SignalStack,
    fz: np.ndarray,
    *,
    kernel: str = "gaussian",
    B: float = 0.05,
    Theta_rad: float | None = None,
    px_size_mm: float | None = None,
    truncate: float = 3.0,
    sigma_mode: str = "angled",
    sigma_const: float | None = None,
    apply_slice_thickness: bool = True,
    spoke_density_normalize: bool = False,
    progress_desc: str = "DBT recon",
) -> SignalND:
    """
    DBT-specific 2D -> 3D frequency-domain mapping.

    This implements the Zhao/Hu-style DBT mapping:
    - per-view rotation in the reconstruction plane
    - deposition along fz using a finite-width interpolation kernel
    - optional DBT slice-thickness filter
    - optional DBT spoke-density normalization
    """
    interp_fn = _get_interp_kernel(kernel)

    fz = np.asarray(fz, dtype=np.float32)
    fx = stack.fx.astype(np.float32)
    fy = stack.fy.astype(np.float32)

    S_views = stack.S
    W_views = stack.W
    angles = np.asarray(stack.angles, dtype=float)

    Nv, Nx, Ny = S_views.shape
    Nz = fz.size

    if Theta_rad is None:
        if Nv > 1:
            Theta_rad = np.max(angles) + np.abs(np.min(angles))
        else:
            Theta_rad = 1.0

    if px_size_mm is None:
        px_size_mm = float(getattr(stack, "px_size", 0.085))

    dfz = float(fz[1] - fz[0]) if Nz > 1 else 1.0

    if Nv > 1:
        dtheta = float(np.mean(np.diff(np.sort(angles))))
    else:
        dtheta = 0.0

    S3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    W3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    if kernel == "gaussian" and sigma_mode == "angled":
        sigma_arr = np.maximum(0.5 * dfz, np.abs(fx) * np.sin(abs(dtheta)))
    elif kernel == "gaussian" and sigma_mode == "constant":
        if sigma_const is None:
            raise ValueError("sigma_const must be given when sigma_mode='constant'")
        sigma_arr = np.full_like(fx, float(sigma_const), dtype=np.float32)
    else:
        sigma_arr = None

    norm_const = (Nv / Theta_rad) if (Theta_rad > 0 and spoke_density_normalize) else 1.0

    for i in tqdm(range(Nv), desc=progress_desc, unit="view"):
        theta_i = float(angles[i])
        c = np.cos(theta_i)
        s = np.sin(theta_i)

        S2 = S_views[i]
        W2 = W_views[i]

        if apply_slice_thickness:
            Hst = slice_thickness_filter_dbt(
                fz,
                theta_rad=theta_i,
                px_size_mm=px_size_mm,
                Theta_rad=Theta_rad,
                B=B,
            )
        else:
            Hst = np.ones_like(fz, dtype=np.float32)

        fx_in = fx
        fx_out_for_in = fx_in * c

        S2x = np.empty_like(S2)
        W2x = np.empty_like(W2)

        for iy in range(Ny):
            S2x[:, iy] = np.interp(fx, fx_out_for_in, S2[:, iy], left=0.0, right=0.0)
            W2x[:, iy] = np.interp(fx, fx_out_for_in, W2[:, iy], left=0.0, right=0.0)

        fz_line = (fx / max(c, 1e-6)) * s

        for ix in range(Nx):
            center = float(fz_line[ix])

            if kernel == "gaussian":
                sigma = float(sigma_arr[ix])
                w = interp_fn(center, fz, sigma=sigma, truncate=truncate)
            elif kernel == "sinc":
                w = interp_fn(center, fz, width=dfz, truncate=truncate)
            elif kernel == "triangular":
                w = interp_fn(center, fz, width=2.0 * dfz)
            else:
                raise RuntimeError("Unsupported kernel")

            S3[ix, :, :] += S2x[ix, :, None] * w[None, :] * Hst[None, :]
            W3[ix, :, :] += W2x[ix, :, None] * w[None, :] * (Hst[None, :] ** 2)

    if norm_const != 1.0:
        S3 *= norm_const
        W3 *= norm_const ** 2

    mean_quanta_3d = float(np.mean(stack.mean_quanta))
    return SignalND(axes=[fx, fy, fz], S=S3, W=W3, mean_quanta=mean_quanta_3d)


# ============================================================================
# CBCT-specific functions
# ============================================================================

def map_stack_to_volume_cbct(
    stack: SignalStack,
    fz: np.ndarray,
    *,
    M: float = 1.0,
    d_extent_mm: float = 100.0,
    normalize_by_views: bool = False,
    apply_asymptotic_1_over_f: bool = False,
    f_epsilon: float = 1e-9,
    progress_desc: str = "CBCT backprojection",
) -> SignalND:
    """
    CBCT Stage 13 backprojection following Tward & Siewerdsen (2008).

    Coordinate convention
    ---------------------
    Detector:
        u-axis || x
        v-axis || z

    Rotated reconstruction coordinates for the i-th view:
        x_i =  x cos(theta_i) + y sin(theta_i)
        y_i = -x sin(theta_i) + y cos(theta_i)
        z   =  z

    Frequency coordinates:
        f_xi =  f_x cos(theta_i) + f_y sin(theta_i)
        f_yi = -f_x sin(theta_i) + f_y cos(theta_i)
        f_z  =  f_z

    Stage 13 equations (Tward):
        T13i(f_xi,f_yi,f_z) = d * sinc(d * f_xi)
        S13i(f_xi,f_yi,f_z) = S12M(f_yi,f_z) * (1/d) * T13i^2

    Notes
    -----
    - Magnification remap S12 -> S12M is included.
    - Signal is propagated linearly by T13i.
    - Wiener/NPS is propagated by (1/d) * T13i^2.
    - Superposition over views is a sum; optional normalization can be applied later.
    - `apply_asymptotic_1_over_f` is OFF by default. Use it only for a
      separate Appendix-A-style validation, not together with explicit view summation.
    """
    fz = np.asarray(fz, dtype=np.float32)
    fx = stack.fx.astype(np.float32)
    fv = stack.fy.astype(np.float32)   # stack second axis is detector-v, parallel to z

    S_views = stack.S
    W_views = stack.W
    angles = np.asarray(stack.angles, dtype=float)

    Nv, Nu, Nv_det = S_views.shape
    Nx = len(fx)
    Ny = len(fx)        # reconstruct y-grid same as x-grid
    Nz = len(fz)

    if Nv_det != len(fv):
        raise ValueError("SignalStack second axis does not match detector-v axis length.")

    # Reconstruction y-frequency axis.
    # For CBCT cylindrical symmetry in the x-y plane, use the same sampling as fx.
    fy = fx.copy()

    # 3D reconstruction grid
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")

    S3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    W3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # Detector-domain sampling axes after magnification remap:
    #   S12M(f_yi, f_z) = S12(f_u/M, f_v/M) / M^2
    fu_axis = fx                      # detector-u axis sampled like current sig2d.fx
    fv_axis = fv                      # detector-v axis sampled like current sig2d.fy

    inv_M = 1.0 / float(M)
    d = float(d_extent_mm)

    for i in tqdm(range(Nv), desc=progress_desc, unit="view"):
        theta_i = float(angles[i])
        c = np.cos(theta_i)
        s = np.sin(theta_i)

        # Rotated reconstruction coordinates in frequency space
        f_xi = FX * c + FY * s
        f_yi = -FX * s + FY * c

        # Stage 13 transfer for this view:
        # T13i(f_xi, f_yi, f_z) = d * sinc(d * f_xi)
        T13i = d * np.sinc(d * f_xi)

        # Magnification remap of Stage 12 output:
        # S12M(f_yi, f_z) = S12(f_u/M, f_v/M) / M^2
        #
        # Here:
        #   detector-u <-> f_yi
        #   detector-v <-> f_z
        #
        # because backprojection smears along x_i and preserves the in-plane
        # projection-view direction y_i plus longitudinal z.
        S12 = S_views[i]
        W12 = W_views[i]

        S12M = np.empty((Nx, Ny, Nz), dtype=np.float32)
        W12M = np.empty((Nx, Ny, Nz), dtype=np.float32)

        for iz in range(Nz):
            FUq = f_yi[:, :, iz] * inv_M
            FVq = FZ[:, :, iz] * inv_M

            S12M[:, :, iz] = _bilinear_sample2d(S12, fu_axis, fv_axis, FUq, FVq) / (M * M)
            W12M[:, :, iz] = _bilinear_sample2d(W12, fu_axis, fv_axis, FUq, FVq) / (M * M)

        # Signal superposition:
        # T13 = (M/m) sum_i T13i   [paper Eq. 9a]
        # Here we accumulate the per-view signal contribution and can optionally
        # normalize by views at the end.
        S3 += S12M * T13i

        # NPS / Wiener propagation:
        # S13i = S12M * (1/d) * T13i^2   [paper Eq. 8b]
        W3 += W12M * ((T13i ** 2) / d)

    if normalize_by_views and Nv > 0:
        # View-averaged version for easier comparison across m
        S3 *= (M / float(Nv))
        W3 *= (M * M) / float(Nv)
    else:
        # Keep raw superposition, but include M scaling on signal side
        S3 *= M

    if apply_asymptotic_1_over_f:
        # Optional Appendix-A asymptotic form:
        # T13(f,fz) ~ M / f
        # Use only for separate validation, not as part of the explicit Stage 13 sum.
        FR = np.sqrt(FX**2 + FY**2)
        H = M / np.maximum(FR, float(f_epsilon))
        S3 *= H
        W3 *= H**2

    mean_quanta_3d = float(np.mean(stack.mean_quanta))
    return SignalND(axes=[fx, fy, fz], S=S3, W=W3, mean_quanta=mean_quanta_3d)

# ============================================================================
# Compatibility wrapper
# ============================================================================

def map_stack_to_volume(
    stack: SignalStack,
    fz: np.ndarray,
    *,
    geometry: str = "dbt",
    **kwargs,
) -> SignalND:
    """
    Compatibility wrapper.

    Parameters
    ----------
    geometry:
        "dbt" or "cbct"
    """
    geometry = geometry.lower()

    if geometry == "dbt":
        return map_stack_to_volume_dbt(stack, fz, **kwargs)
    elif geometry == "cbct":
        return map_stack_to_volume_cbct(stack, fz, **kwargs)
    else:
        raise ValueError("geometry must be 'dbt' or 'cbct'")


# ============================================================================
# Backward-compatible alias
# ============================================================================

slice_thickness_filter = slice_thickness_filter_dbt