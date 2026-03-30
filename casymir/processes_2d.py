# Module: processes_2d

from typing import Tuple
import numpy as np
from scipy import integrate

import casymir.casymir
import casymir.parallel
from casymir.casymir import Signal, SignalND, Detector


# ============================================================================
# Shared helpers
# ============================================================================

def centered_axis_from_nonneg(f_nonneg: np.ndarray) -> np.ndarray:
    f_nonneg = np.asarray(f_nonneg)
    df = float(np.mean(np.diff(f_nonneg)))
    fmax = float(f_nonneg[-1])
    N_centered = int(round(2 * fmax / df))
    return -fmax + df * np.arange(N_centered)


def _make_freq_grid_2d(f1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    FX, FY = np.meshgrid(f1d, f1d, indexing="xy")
    return FX, FY


def _lift_1d_to_2d_isotropic_centered(sig, FX: np.ndarray, FY: np.ndarray):
    R = np.sqrt(FX**2 + FY**2)
    f1d = sig.freq
    S1d, W1d = sig.signal, sig.wiener
    Rclip = np.clip(R, f1d[0], f1d[-1])
    S2d = np.interp(Rclip, f1d, S1d)
    W2d = np.interp(Rclip, f1d, W1d)
    return S2d, W2d


def _bilinear_sample2d(
    W: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    FXq: np.ndarray,
    FYq: np.ndarray,
) -> np.ndarray:
    Nx, Ny = len(fx), len(fy)
    dfx = fx[1] - fx[0]
    dfy = fy[1] - fy[0]
    fx0, fy0 = fx[0], fy[0]

    X = (FXq - fx0) / dfx
    Y = (FYq - fy0) / dfy

    i0 = np.floor(X).astype(np.int64)
    j0 = np.floor(Y).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1

    valid = (i0 >= 0) & (i1 < Nx) & (j0 >= 0) & (j1 < Ny)

    out = np.zeros_like(FXq, dtype=W.dtype)
    if not np.any(valid):
        return out

    tx = X[valid] - i0[valid]
    ty = Y[valid] - j0[valid]

    V00 = W[i0[valid], j0[valid]]
    V01 = W[i0[valid], j1[valid]]
    V10 = W[i1[valid], j0[valid]]
    V11 = W[i1[valid], j1[valid]]

    V0 = (1 - tx) * V00 + tx * V10
    V1 = (1 - tx) * V01 + tx * V11
    out[valid] = (1 - ty) * V0 + ty * V1
    return out


def _default_harmonic_limits(fmax: float, fs: float) -> int:
    return int(np.ceil(abs(fmax) / abs(fs)))


def _copy_signalnd(sig2d: SignalND) -> SignalND:
    out = SignalND(sig2d.axes, sig2d.S.copy(), sig2d.W.copy(), sig2d.mean_quanta)
    out.mtf = None if sig2d.mtf is None else sig2d.mtf.copy()
    out.nnps = None if sig2d.nnps is None else sig2d.nnps.copy()
    return out


def _resolve_detector_pitch_xy(detector) -> tuple[float, float]:
    px_x = getattr(detector, "px_size_x", getattr(detector, "px_size", None))
    px_y = getattr(detector, "px_size_y", getattr(detector, "px_size", None))
    if px_x is None or px_y is None:
        raise ValueError("Detector must define px_size or both px_size_x / px_size_y.")
    return float(px_x), float(px_y)


# ============================================================================
# Shared 2D detector / projection-domain stages
# ============================================================================

def integration_2d(detector, sig) -> Tuple[SignalND, float, np.ndarray]:
    """
    Quantum integration (2D).
    Convention: x=rows, y=cols.
    """
    fx_c = centered_axis_from_nonneg(sig.freq)
    fy_c = fx_c.copy()

    FX, FY = np.meshgrid(fx_c, fy_c, indexing="ij")

    S2d, W2d = _lift_1d_to_2d_isotropic_centered(sig, FX, FY)

    # Current implementation assumes square detector elements through detector.pxa
    a_pd = np.sqrt(detector.pxa)
    Tpx_2d = np.sinc(a_pd * FX) * np.sinc(a_pd * FY)

    sig2d = SignalND(axes=[fx_c, fy_c], S=S2d, W=W2d, mean_quanta=sig.mean_quanta)

    sig2d.deterministic_blur(Tpx_2d)
    sig2d.stochastic_gain(detector.pxa, 0.0)

    return sig2d, detector.pxa, Tpx_2d


def noise_aliasing_2d(
    detector,
    sig2d_pre: SignalND,
    Kx: int | None = None,
    Ky: int | None = None,
) -> SignalND:
    """
    2D Dirac-comb aliasing (centered coords, x=rows, y=cols).
    S is unchanged. W becomes the aliased sum of shifted pre-sampling W.
    """
    assert sig2d_pre.ndim == 2, "noise_aliasing_2d expects a 2D SignalND."

    fx, fy = sig2d_pre.fx, sig2d_pre.fy
    FX, FY = sig2d_pre.FX, sig2d_pre.FY
    Wpre = sig2d_pre.W

    px_x, px_y = _resolve_detector_pitch_xy(detector)
    fsx = 1.0 / px_x
    fsy = 1.0 / px_y

    fmax_x = max(abs(fx[0]), abs(fx[-1]))
    fmax_y = max(abs(fy[0]), abs(fy[-1]))
    if Kx is None:
        Kx = _default_harmonic_limits(fmax_x, fsx)
    if Ky is None:
        Ky = _default_harmonic_limits(fmax_y, fsy)

    W_alias = np.zeros_like(Wpre, dtype=float)

    for kx in range(-Kx, Kx + 1):
        for ky in range(-Ky, Ky + 1):
            FXq = FX + kx * fsx
            FYq = FY + ky * fsy
            W_shift = _bilinear_sample2d(Wpre, fx, fy, FXq, FYq)
            W_alias += W_shift

    out = SignalND(
        axes=[fx, fy],
        S=sig2d_pre.S.copy(),
        W=W_alias,
        mean_quanta=sig2d_pre.mean_quanta,
    )
    out.mtf = sig2d_pre.mtf
    out.nnps = None
    return out


def focal_spot_blur(
    sig2d: SignalND,
    a1: float,
    direction: str = "x",
    angle_rad: float | None = None,
    inplace: bool = False,
):
    """
    Apply 2D focal-spot motion blur as a deterministic blur stage.

    For your current validation setup, this is intended for DBT and typically
    should be disabled for CBCT.
    """
    if sig2d.ndim != 2:
        raise ValueError("focal_spot_blur expects a 2D SignalND.")

    if angle_rad is not None:
        ux = np.cos(angle_rad)
        uy = np.sin(angle_rad)
    else:
        if direction.lower() == "x":
            ux, uy = 1.0, 0.0
        elif direction.lower() == "y":
            ux, uy = 0.0, 1.0
        else:
            raise ValueError("direction must be 'x' or 'y' when angle_rad is not provided.")

    FX, FY = sig2d.FX, sig2d.FY
    f_parallel = ux * FX + uy * FY

    H_fsb = np.sinc(a1 * f_parallel)

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H_fsb)

    return out, H_fsb, a1


def beam_obliquity_blur_2d(
    sig2d: SignalND,
    *,
    spectrum,
    detector,
    theta_rad: float,
    use_complex_phase: bool = False,
    inplace: bool = False,
):
    """
    Apply beam-obliquity blur along x (rows).

    For your current validation setup, this is intended for DBT and typically
    should be disabled for CBCT.
    """
    if sig2d.ndim != 2:
        raise ValueError("beam_obliquity_blur_2d expects a 2D SignalND.")

    d_mm = detector.thickness / 1000
    T_line = _obliquity_T_fx(sig2d.fx, theta_rad, d_mm, spectrum, detector)
    H_line = T_line if use_complex_phase else np.abs(T_line)

    H_obl = H_line[:, None]

    out = sig2d if inplace else _copy_signalnd(sig2d)

    # Preserve current behavior: signal-only blur
    out.S = sig2d.S * H_obl
    return out, H_obl, T_line


def _linear_mu_mm_from_detector(detector, E_keV: np.ndarray) -> np.ndarray:
    """
    Returns linear attenuation μ(E) in mm^-1 for the detector's active layer.
    Detector.get_mu(E) populates mass attenuation (cm^2/g) in detector.mu.
    Linear μ = (μ_mass * density * packing_factor) [cm^-1] -> mm^-1.
    """
    detector.get_mu(E_keV)
    rho = float(detector.material["density"])
    pf = float(detector.material.get("pf", 1.0))
    mu_linear_cm = detector.mu * rho * pf
    mu_linear_mm = mu_linear_cm / 10.0
    return mu_linear_mm


def _obliquity_T_fx(
    fx_c: np.ndarray,
    theta_rad: float,
    d_mm: float,
    spectrum,
    detector,
) -> np.ndarray:
    """
    Compute T_theta(fx) using spectrum.energy / spectrum.fluence and detector μ(E).
    """
    E = np.asarray(spectrum.energy, dtype=float)
    Phi = np.asarray(spectrum.fluence, dtype=float)
    mu = _linear_mu_mm_from_detector(detector, E)

    MU = mu[:, None]
    FX = fx_c[None, :]

    tprime = d_mm / np.cos(theta_rad)
    alpha = 2.0 * np.pi * FX * tprime * np.tan(theta_rad)
    beta = 2.0 * np.pi * FX * np.sin(theta_rad) / MU

    Ew = (E * Phi)[:, None]

    numerator = (1.0 - np.exp(-MU * tprime - 1j * alpha)) / (1.0 + 1j * beta)
    num_int = integrate.trapezoid(Ew * numerator, E, axis=0)

    denom = (1.0 - np.exp(-MU * tprime))
    den_int = integrate.trapezoid(Ew * denom, E, axis=0)

    T_fx = num_int / den_int
    return T_fx


def log_transform_2d(
    sig2d: SignalND,
    *,
    spectrum,
    a: float,
    b: float,
):
    k_air = spectrum.dak
    gain_factor = a + b * k_air

    sig2d.S = sig2d.S / gain_factor
    sig2d.W = sig2d.W / (gain_factor ** 2)

    return sig2d, gain_factor


# ============================================================================
# DBT-specific helpers
# ============================================================================

def _dbt_fr_on_plane(sig2d: SignalND, theta_i_rad: float) -> np.ndarray:
    """
    DBT per-view radial frequency in the rotation plane:
        f_r = f_x / cos(theta_i)
    """
    FX = sig2d.FX
    c = np.cos(theta_i_rad)
    if np.isclose(c, 0.0):
        raise ValueError("cos(theta_i_rad) ~ 0; invalid DBT view.")
    return FX / c


def _wrap_to_band(x: np.ndarray, half_width: float) -> np.ndarray:
    """
    Periodically wrap real values to (-half_width, half_width] with period 2*half_width.
    """
    if half_width <= 0:
        raise ValueError("half_width must be > 0")
    return ((x + half_width) % (2.0 * half_width)) - half_width


def _ambr_ramp_kernel_1d(N: int) -> np.ndarray:
    n = np.fft.fftfreq(N, d=1.0 / N).astype(float)

    n_safe = n.copy()
    n_safe[0] = 1.0

    h = -1.0 / (np.pi * n_safe) ** 2
    even_mask = (np.mod(n, 2.0) == 0.0)
    h[even_mask] = 0.0

    h[0] = 0.25
    return h


def _ambr_ramp_freq_response_centered(N: int) -> np.ndarray:
    h = _ambr_ramp_kernel_1d(N)
    H = np.fft.fft(h)
    Hc = np.fft.fftshift(H)
    return Hc


# ============================================================================
# Reconstruction filters: DBT
# ============================================================================

def ramp_filter_dbt(
    sig2d: SignalND,
    *,
    detector,
    theta_total_rad: float,
    theta_i_rad: float,
) -> np.ndarray:
    """
    DBT ramp filter in radial frequency coordinates.
    """
    px_x, _ = _resolve_detector_pitch_xy(detector)
    fny = 1.0 / (2.0 * px_x)

    c = np.cos(theta_i_rad)
    fr_ny = np.abs(fny / c)

    FR = _dbt_fr_on_plane(sig2d, theta_i_rad)
    FRm = _wrap_to_band(FR, fr_ny)

    scale = 2.0 * np.tan(theta_total_rad) / fr_ny
    H = scale * np.abs(FRm)
    return H


def apodization_filter_dbt(
    sig2d: SignalND,
    *,
    detector,
    A: float = 1.5,
    theta_i_rad: float = 0.0,
    replicate: bool = True,
) -> np.ndarray:
    """
    DBT spectral apodization filter in radial frequency coordinates.
    """
    px_x, _ = _resolve_detector_pitch_xy(detector)
    fny = 1.0 / (2.0 * px_x)

    c = np.cos(theta_i_rad)
    fr_ny = np.abs(fny / c)

    FR = _dbt_fr_on_plane(sig2d, theta_i_rad)
    if replicate:
        FR = _wrap_to_band(FR, fr_ny)

    r = np.abs(FR) / (A * fr_ny)
    H = np.zeros_like(FR)
    inside = r <= 1.0
    H[inside] = 0.5 * (1.0 + np.cos(np.pi * r[inside]))
    return H


def interpolation_filter_dbt(
    sig2d: SignalND,
    *,
    a_x: float | None = None,
    a_y: float | None = None,
    m_x: float | None = None,
    m_y: float | None = None,
    theta_i_rad: float = 0.0,
    power: int = 2,
) -> np.ndarray:
    """
    DBT bilinear interpolation filter using (f_r, f_y).
    """
    FY = sig2d.FY
    FR = _dbt_fr_on_plane(sig2d, theta_i_rad)

    if (a_x is not None) and (a_y is not None):
        s_r = float(a_x)
        s_y = float(a_y)
    elif (m_x is not None) and (m_y is not None):
        s_r = np.cos(theta_i_rad) * float(m_x)
        s_y = float(m_y)
    else:
        raise ValueError("Provide either (a_x, a_y) or (m_x, m_y).")

    Hr = np.sinc(s_r * FR)
    Hy = np.sinc(s_y * FY)
    H = (Hr * Hr) * (Hy * Hy) if power == 2 else (Hr * Hy)
    return H


# ============================================================================
# Reconstruction filters: CBCT
# ============================================================================

def ramp_filter_cbct(sig2d: SignalND, eps: float = 1e-12) -> np.ndarray:
    """
    CBCT ramp filter in detector coordinates.
    Assumes sig2d.FX corresponds to detector-u frequency.
    """
    H = np.abs(sig2d.FX)
    if eps > 0.0:
        H = np.maximum(H, eps)
    return H


def apodization_filter_cbct(
    sig2d: SignalND,
    *,
    detector=None,
    a_u: float | None = None,
    B_u: float = 1.0,
    hwin: float = 0.0,
) -> np.ndarray:
    """
    CBCT apodization filter in detector-u frequency.

    First-pass implementation:
        H = hwin + (1-hwin) * cos(2*pi*f_u*a_u*B_u)
    inside the supported detector-u band, zero outside.

    Notes
    -----
    - No DBT radial-frequency mapping.
    - No theta_i dependence.
    - Assumes sig2d.FX is detector-u frequency.
    """
    if a_u is None:
        if detector is None:
            raise ValueError("Provide either detector or a_u.")
        a_u, _ = _resolve_detector_pitch_xy(detector)

    FU = sig2d.FX
    f_u_lim = 1.0 / (2.0 * float(a_u) * float(B_u))

    H = np.zeros_like(FU, dtype=np.float32)
    inside = np.abs(FU) <= f_u_lim
    H[inside] = hwin + (1.0 - hwin) * np.cos(2.0 * np.pi * FU[inside] * a_u * B_u)
    return H


def interpolation_filter_cbct(
    sig2d: SignalND,
    *,
    detector=None,
    a_u: float | None = None,
    a_v: float | None = None,
    power: int = 2,
) -> np.ndarray:
    """
    CBCT bilinear interpolation filter in detector coordinates (f_u, f_v).
    """
    if (a_u is None) or (a_v is None):
        if detector is None:
            raise ValueError("Provide either detector or both a_u / a_v.")
        a_u, a_v = _resolve_detector_pitch_xy(detector)

    FU, FV = sig2d.FX, sig2d.FY

    Hu = np.sinc(float(a_u) * FU)
    Hv = np.sinc(float(a_v) * FV)

    H = (Hu * Hu) * (Hv * Hv) if power == 2 else (Hu * Hv)
    return H


# ============================================================================
# Apply wrappers
# ============================================================================

def apply_ramp_filter_dbt_ambr(
    sig2d: SignalND,
    *,
    detector=None,
    inplace: bool = False,
):
    fx = sig2d.fx
    N_fx = fx.size
    N_fy = sig2d.fy.size

    Hx = _ambr_ramp_freq_response_centered(N_fx).real
    H2d = Hx[:, None] * np.ones((1, N_fy), dtype=Hx.dtype)

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H2d)

    return out, H2d


def apply_ramp_filter_dbt(
    sig2d: SignalND,
    *,
    detector,
    theta_total_rad: float,
    theta_i_rad: float,
    inplace: bool = False,
):
    H = ramp_filter_dbt(
        sig2d,
        detector=detector,
        theta_total_rad=theta_total_rad,
        theta_i_rad=theta_i_rad,
    )

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


def apply_ramp_filter_cbct(
    sig2d: SignalND,
    *,
    eps: float = 0.0,
    inplace: bool = False,
):
    H = ramp_filter_cbct(sig2d, eps=eps)

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


def apply_apodization_filter_dbt(
    sig2d: SignalND,
    *,
    detector,
    A: float = 1.5,
    theta_i_rad: float,
    replicate: bool = True,
    inplace: bool = False,
):
    H = apodization_filter_dbt(
        sig2d,
        detector=detector,
        A=A,
        theta_i_rad=theta_i_rad,
        replicate=replicate,
    )

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


def apply_apodization_filter_cbct(
    sig2d: SignalND,
    *,
    detector=None,
    a_u: float | None = None,
    B_u: float = 1.0,
    hwin: float = 0.5,
    inplace: bool = False,
):
    H = apodization_filter_cbct(
        sig2d,
        detector=detector,
        a_u=a_u,
        B_u=B_u,
        hwin=hwin,
    )

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


def apply_interpolation_filter_dbt(
    sig2d: SignalND,
    detector,
    *,
    a_x: float | None = None,
    a_y: float | None = None,
    m_x: float | None = None,
    m_y: float | None = None,
    theta_i_rad: float = 0.0,
    power: int = 2,
    inplace: bool = False,
):
    if a_x is None or a_y is None:
        a_x, a_y = _resolve_detector_pitch_xy(detector)

    H = interpolation_filter_dbt(
        sig2d,
        a_x=a_x,
        a_y=a_y,
        m_x=m_x,
        m_y=m_y,
        theta_i_rad=theta_i_rad,
        power=power,
    )

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


def apply_interpolation_filter_cbct(
    sig2d: SignalND,
    detector=None,
    *,
    a_u: float | None = None,
    a_v: float | None = None,
    power: int = 2,
    inplace: bool = False,
):
    H = interpolation_filter_cbct(
        sig2d,
        detector=detector,
        a_u=a_u,
        a_v=a_v,
        power=power,
    )

    out = sig2d if inplace else _copy_signalnd(sig2d)
    out.deterministic_blur(H)
    return out, H


# ============================================================================
# Backward-compatible aliases
# ============================================================================

# Keep old names alive for now so the rest of the codebase doesn't explode.
spectrum_apodization_filter = apodization_filter_dbt
apply_sa_filter_dbt = apply_apodization_filter_dbt
interpolation_filter_bilinear = interpolation_filter_dbt
apply_interpolation_filter_bilinear_dbt = apply_interpolation_filter_dbt


# ============================================================================
# Legacy 1D-style output helper
# ============================================================================

def model_output_2D(
    detector: casymir.casymir.Detector,
    signal: casymir.casymir.SignalND,
) -> casymir.casymir.SignalND:
    """
    Legacy 1D-style output helper retained as-is.
    """
    mtf = signal.signal / signal.signal[0]
    add_noise = detector.add_noise

    wiener2 = signal.wiener[0:int(signal.length / 2)] + ((add_noise ** 2) * (detector.pxa / detector.ff))
    signal2 = signal.signal[0:int(signal.length / 2)]

    f2 = signal.freq[0:int(signal.length / 2)]
    mtf = mtf[0:int(signal.length / 2)]

    nnps = wiener2 / (signal.signal[0] ** 2)

    output_signal = casymir.casymir.Signal(
        freq=f2,
        signal=signal2,
        wiener=wiener2,
        mean_quanta=signal.mean_quanta,
    )

    output_signal.mtf = mtf
    output_signal.nnps = nnps

    return output_signal