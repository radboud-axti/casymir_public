import numpy as np

from casymir.casymir import SignalND, SignalStack


def slice_thickness_filter(
        fz: np.ndarray,
        *,
        theta_rad: float,
        px_size_mm: float,
        Theta_rad: float,
        B: float = 0.05,
) -> np.ndarray:
    """
    Slice-thickness window H_ST(fz) for DBT recon, defined as a Hanning window function.

    :param fz: Frequency vector in the z direction.
    :param theta_rad: Projection angle in radians.
    :param px_size_mm: Pixel size in mm.
    :param Theta_rad: Total angular range of the acquisition, in radians.
    :param B: Window width parameter for the slice thickness filter.

    :return: Vector containing the H_ST values at fz
    :rtype: np.ndarray
    """
    fz = np.asarray(fz, dtype=np.float32)
    c = np.cos(theta_rad)

    # View-dependent detector Nyquist in radial direction
    fr_ny = np.abs(1.0 / (2.0 * float(px_size_mm) * c))
    # The following approximation can be used too
    # fr_ny = 1 / (2 * px)

    # Bounds
    mask = (np.abs(fz) <= B * fr_ny) & (np.abs(fz) <= np.tan(Theta_rad) * fr_ny)
    Hst = np.ones_like(fz, dtype=np.float32)

    if np.any(mask):
        Hst[mask] = 0.5 * (1 + np.cos((np.pi * fz[mask]) / (B * fr_ny)))

    return Hst


def gaussian_weights(
        center: float,
        axis: np.ndarray,
        sigma: float,
        truncate: float = 3.0,
) -> np.ndarray:
    """
    Gaussian kernel (normalized).


    :param center: Kernel center in the specified (fz) axis
    :param axis: Specified axis vector (fz)
    :param sigma: Kernel sigma.
    :param truncate: Kernel support cutoff parameter (multiple of sigma)

    :rtype: np.ndarray
    :return: Vector containing the weights along the specified axis.
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
    Sinc kernel.

    :param center: Kernel center in the specified (fz) axis
    :param axis: Specified axis vector (fz)
    :param width: Kernel width.
    :param truncate: Kernel support cutoff parameter (multiple of the kernel width)

    :rtype: np.ndarray
    :return: Vector containing the weights along the specified axis.
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
    Simple triangular kernel.

    :param center: Kernel center in the specified (fz) axis
    :param axis: Specified axis vector (fz)
    :param width: Kernel width.

    :rtype: np.ndarray
    :return: Vector containing the weights along the specified axis.
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


def map_stack_to_volume(
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
) -> SignalND:
    """
    Maps a stack of 2D frequency-domain projections acquired at angles theta_i (SignalStack) into a 3D frequency-domain
    volume (SignalND) using a finite-width kernel along fz and applying a slice thickness filter H_ST(fz).

    This implemetation follows the expressions presented in:

    - Zhao, B. and Zhao, W. (2008), Three-dimensional linear system analysis for breast tomosynthesis. Med. Phys., 35: 5219-5232. https://doi.org/10.1118/1.2996014

    - Hu, Y.-H. and Zhao, W. (2014), The effect of amorphous selenium detector thickness on dual-energy digital breast imaging. Med. Phys., 41: 111904. https://doi.org/10.1118/1.4897244

    In the referenced works, the 2D-3D mapping is defined in terms of a delta function δ(fx*sin(theta_i) - fz*cos(theta_i))

    In order to generate smooth curves for the model, finite-width Gaussian, Sinc, and Triangular interpolation kernels
    have been implemented.

    :param stack: SignalStack containing the frequency-domain Signal magnitude and Wiener spectra of 2D projections.
    :param fz: Frequency vector in the z direction.
    :param kernel: Interpolation kernel type.
    :param B: Window width parameter for the slice thickness filter.
    :param Theta_rad: Total angular range of the acquisition, in radians.
    :param px_size_mm: Pixel size in mm.
    :param truncate: Controls the cutoff of the interpolation kernel.
    :param sigma_mode: Gaussian filter's sigma definition: angular step-dependent ("angled") or fixed ("constant")
    :param sigma_const: Specifies the value of the Gaussian filter's sigma
    :param apply_slice_thickness: Toggles the slice thickness filter on or off.
    :param spoke_density_normalize: Toggles the spoke density normalization on or off.

    :rtype: SignalND
    :return: SignalND object containing the 3D Signal magnitude and Wiener spectrum.
    """
    if kernel not in _INTERP_KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. "
                         f"Available: {list(_INTERP_KERNELS.keys())}")

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

    # z-frequency sampling
    dfz = float(fz[1] - fz[0]) if Nz > 1 else 1.0

    if Nv > 1:
        dtheta = float(np.mean(np.diff(np.sort(angles))))
    else:
        dtheta = 0.0

    # allocate volumes
    S3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    W3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # choose kernel function
    interp_fn = _INTERP_KERNELS[kernel]

    # precompute sigma(ix) for Gaussian mode if needed
    if kernel == "gaussian" and sigma_mode == "angled":
        sigma_arr = np.maximum(0.5 * dfz, 1 * np.abs(fx) * np.sin(abs(dtheta)))
    elif kernel == "gaussian" and sigma_mode == "constant":
        if sigma_const is None:
            raise ValueError("sigma_const must be given when sigma_mode='constant'")
        sigma_arr = np.full_like(fx, float(sigma_const), dtype=np.float32)
    else:
        sigma_arr = None

    # Optional spoke-density normalization
    norm_const = (Nv / Theta_rad) if (Theta_rad > 0 and spoke_density_normalize) else 1.0

    for i in range(Nv):
        theta_i = float(angles[i])
        c = np.cos(theta_i)
        s = np.sin(theta_i)

        S2 = S_views[i]
        W2 = W_views[i]

        # Slice-thickness window H_ST(fz) for current view
        if apply_slice_thickness:
            Hst = slice_thickness_filter(fz, theta_rad=theta_i, px_size_mm=px_size_mm, Theta_rad=Theta_rad, B=B)
        else:
            Hst = np.ones_like(fz, dtype=np.float32)

        # X resample: rotate fr -> fx
        fx_in = fx
        fx_out_for_in = fx_in * c

        S2x = np.empty_like(S2)
        W2x = np.empty_like(W2)

        for iy in range(Ny):
            S2x[:, iy] = np.interp(fx, fx_out_for_in, S2[:, iy], left=0.0, right=0.0)
            W2x[:, iy] = np.interp(fx, fx_out_for_in, W2[:, iy], left=0.0, right=0.0)

        # Interp along fz with chosen kernel and H_ST
        fz_line = (fx / max(c, 1e-6)) * s

        for ix in range(Nx):
            center = float(fz_line[ix])
            # choose fz weights
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

    # Apply normalization to the whole volume
    if norm_const != 1.0:
        S3 *= norm_const
        W3 *= norm_const ** 2

    mean_quanta_3d = float(np.mean(stack.mean_quanta))

    return SignalND(axes=[fx, fy, fz], S=S3, W=W3, mean_quanta=mean_quanta_3d)
