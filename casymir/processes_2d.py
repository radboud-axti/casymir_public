"""
Projection-domain process blocks.

This module is the bridge between the original 1D CASYMIR detector cascade and
3D reconstruction models.

Coordinate convention for 2D signals
------------------------------------
axis 0: detector-u / scan-x frequency
axis 1: detector-v / longitudinal-z frequency
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from casymir.casymir import Detector, Signal, SignalND, Spectrum

# Numerical helpers


def centered_axis_from_nonnegative(f_nonnegative: np.ndarray) -> np.ndarray:
    """
    Mirror a non-negative 1D frequency axis into a centered Fourier axis.

    The original 1D CASYMIR model stores frequencies from zero upward. The 2D
    and 3D models work on centered axes; zero frequency is near the middle
    of the array.

    :param f_nonnegative: Uniformly sampled non-negative frequency axis.

    :return: Centered frequency axis with the same sampling interval.
    """
    f_nonnegative = np.asarray(f_nonnegative, dtype=float)
    if f_nonnegative.ndim != 1 or f_nonnegative.size < 2:
        raise ValueError(
            "Expected a one-dimensional frequency axis with at least two samples."
        )

    df = float(np.mean(np.diff(f_nonnegative)))
    fmax = float(f_nonnegative[-1])
    n_centered = int(round(2.0 * fmax / df))
    return (-fmax + df * np.arange(n_centered)).astype(f_nonnegative.dtype, copy=False)


def detector_pitch_uv(detector: Detector) -> tuple[float, float]:
    """
    Return detector pitch along u and v.

    Most systems use square pixels and define detector.px_size. The helper
    also accepts px_size_x / px_size_y for anisotropic detector models.

    :param detector: CASYMIR Detector object.

    :return: Detector pitch along u and v in millimeters.
    """
    pitch_u = getattr(detector, "px_size_x", getattr(detector, "px_size", None))
    pitch_v = getattr(detector, "px_size_y", getattr(detector, "px_size", None))
    if pitch_u is None or pitch_v is None:
        raise ValueError(
            "Detector must define px_size, or both px_size_x and px_size_y."
        )
    return float(pitch_u), float(pitch_v)


def _copy_signal(signal: SignalND) -> SignalND:
    """Return an independent copy of a SignalND object."""
    return signal.copy()


def _harmonic_count(fmax: float, fs: float) -> int:
    """Return the number of sampling replicas needed to cover the support."""
    return int(np.ceil(abs(float(fmax)) / abs(float(fs))))


def sample_grid_2d(
    values: np.ndarray,
    fu: np.ndarray,
    fv: np.ndarray,
    FUq: np.ndarray,
    FVq: np.ndarray,
) -> np.ndarray:
    """
    Sample a 2D array on centered frequency axes.

    Queries outside the support return zero.

    :param values: Values tabulated on the input frequency grid.
    :param fu: Input detector-u frequency axis.
    :param fv: Input detector-v frequency axis.
    :param FUq: Query frequencies along detector-u.
    :param FVq: Query frequencies along detector-v.

    :return: Bilinearly interpolated values on the query grid.
    """
    fu = np.asarray(fu)
    fv = np.asarray(fv)
    values = np.asarray(values)

    du = float(fu[1] - fu[0])
    dv = float(fv[1] - fv[0])
    U = (FUq - fu[0]) / du
    V = (FVq - fv[0]) / dv

    i0 = np.floor(U).astype(np.int64)
    j0 = np.floor(V).astype(np.int64)
    valid = (i0 >= 0) & (i0 < len(fu)) & (j0 >= 0) & (j0 < len(fv))

    out = np.zeros_like(FUq, dtype=values.dtype)
    if not np.any(valid):
        return out

    i0v = i0[valid]
    j0v = j0[valid]
    i1v = np.minimum(i0v + 1, len(fu) - 1)
    j1v = np.minimum(j0v + 1, len(fv) - 1)

    tu = np.where(i1v == i0v, 0.0, U[valid] - i0v)
    tv = np.where(j1v == j0v, 0.0, V[valid] - j0v)

    c00 = values[i0v, j0v]
    c01 = values[i0v, j1v]
    c10 = values[i1v, j0v]
    c11 = values[i1v, j1v]

    c0 = (1.0 - tu) * c00 + tu * c10
    c1 = (1.0 - tu) * c01 + tu * c11
    out[valid] = (1.0 - tv) * c0 + tv * c1
    return out


def _lift_1d_isotropically(
    signal_1d: Signal, FU: np.ndarray, FV: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a radial 1D signal and Wiener spectrum onto a 2D grid."""
    radius = np.sqrt(FU**2 + FV**2)
    radius = np.clip(radius, signal_1d.freq[0], signal_1d.freq[-1])
    signal = np.interp(radius, signal_1d.freq, signal_1d.signal)
    wiener = np.interp(radius, signal_1d.freq, signal_1d.wiener)
    return signal, wiener


# Projection-domain detector blocks


def pixel_integration_2d(
    detector: Detector, signal_1d: Signal
) -> tuple[SignalND, float, np.ndarray]:
    """
    Process that lifts a 1D detector signal into the 2D detector plane and integrates the signal over the
    active pixel area.

    :param detector: CASYMIR Detector object.
    :param signal_1d: CASYMIR Signal object from the 1D detector cascade.

    :return: Tuple with updated SignalND object, active pixel area, and pixel aperture transfer function.
    """
    fu = centered_axis_from_nonnegative(signal_1d.freq)
    fv = fu.copy()
    FU, FV = np.meshgrid(fu, fv, indexing="ij")

    S, W = _lift_1d_isotropically(signal_1d, FU, FV)
    signal_2d = SignalND([fu, fv], S, W, signal_1d.mean_quanta)

    aperture_width = float(np.sqrt(detector.pxa))
    aperture = np.sinc(aperture_width * FU) * np.sinc(aperture_width * FV)
    signal_2d.deterministic_blur(aperture)
    signal_2d.stochastic_gain(float(detector.pxa), 0.0)
    return signal_2d, float(detector.pxa), aperture


def detector_aliasing_2d(
    detector: Detector,
    presampled_signal: SignalND,
    *,
    harmonics_u: int | None = None,
    harmonics_v: int | None = None,
) -> SignalND:
    """
    Process that folds detector sampling aliases into the 2D projection-domain Wiener spectrum.

    This is the 2D version of the 1D CASYMIR aliasing block:

    The harmonic ranges include every shifted copy that can overlap the
    presampling support. Using + k fs is equivalent to the 1D
    f - k fs convention because the integer harmonics are symmetric.

    The mean signal is not altered; only the Wiener spectrum is folded.

    :param detector: CASYMIR Detector object.
    :param presampled_signal: Presampling 2D SignalND object.
    :param harmonics_u: Number of sampling replicas to include along detector-u. If None, the range is inferred.
    :param harmonics_v: Number of sampling replicas to include along detector-v. If None, the range is inferred.

    :return: SignalND object with aliased 2D Wiener spectrum.
    """
    if presampled_signal.ndim != 2:
        raise ValueError("detector_aliasing_2d expects a 2D SignalND.")

    fu, fv = presampled_signal.fx, presampled_signal.fy
    FU, FV = presampled_signal.FX, presampled_signal.FY
    pitch_u, pitch_v = detector_pitch_uv(detector)
    fs_u = 1.0 / pitch_u
    fs_v = 1.0 / pitch_v

    if harmonics_u is None:
        harmonics_u = _harmonic_count(max(abs(fu[0]), abs(fu[-1])), fs_u)
    if harmonics_v is None:
        harmonics_v = _harmonic_count(max(abs(fv[0]), abs(fv[-1])), fs_v)

    W = np.zeros_like(presampled_signal.W, dtype=float)
    for ku in range(-harmonics_u, harmonics_u + 1):
        for kv in range(-harmonics_v, harmonics_v + 1):
            W += sample_grid_2d(
                presampled_signal.W,
                fu,
                fv,
                FU + ku * fs_u,
                FV + kv * fs_v,
            )

    out = SignalND(
        [fu, fv], presampled_signal.S.copy(), W, presampled_signal.mean_quanta
    )
    out.mtf = None if presampled_signal.mtf is None else presampled_signal.mtf.copy()
    return out


def add_electronic_noise_2d(
    signal: SignalND,
    noise_std: float,
    *,
    inplace: bool = False,
) -> tuple[SignalND, float]:
    """
    Process that adds electronic (white) noise to the 2D Wiener spectrum.

    :param signal: CASYMIR SignalND object.
    :param noise_std: Standard deviation of the electronic noise.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and electronic noise variance.
    """
    out = signal if inplace else _copy_signal(signal)
    noise_variance = float(noise_std) ** 2
    out.W = out.W + noise_variance
    return out, noise_variance


def apply_transfer_2d(
    signal: SignalND,
    transfer: np.ndarray,
    *,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies a deterministic 2D transfer function to signal and Wiener spectra.

    :param signal: CASYMIR SignalND object.
    :param transfer: Deterministic transfer function sampled on the signal frequency grid.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied transfer function.
    """
    out = signal if inplace else _copy_signal(signal)
    out.deterministic_blur(np.asarray(transfer))
    return out, np.asarray(transfer)


def focal_spot_blur_2d(
    signal: SignalND,
    length_mm: float,
    *,
    direction: str = "u",
    angle_rad: float | None = None,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray, float]:
    """
    Process that applies a rectangular focal-spot blur in the projection plane.

    :param signal: CASYMIR SignalND object.
    :param length_mm: Focal spot length in millimeters.
    :param direction: Detector direction of the rectangular blur when angle_rad is not provided.
    :param angle_rad: Optional blur direction angle in radians.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object, focal spot transfer function, and focal spot length.
    """
    if signal.ndim != 2:
        raise ValueError("focal_spot_blur_2d expects a 2D SignalND.")

    if angle_rad is None:
        if direction.lower() in ("u", "x"):
            u_hat, v_hat = 1.0, 0.0
        elif direction.lower() in ("v", "y"):
            u_hat, v_hat = 0.0, 1.0
        else:
            raise ValueError("direction must be 'u'/'x' or 'v'/'y'.")
    else:
        u_hat, v_hat = np.cos(angle_rad), np.sin(angle_rad)

    f_parallel = u_hat * signal.FX + v_hat * signal.FY
    transfer = np.sinc(float(length_mm) * f_parallel)
    out, _ = apply_transfer_2d(signal, transfer, inplace=inplace)
    return out, transfer, float(length_mm)


def beam_obliquity_blur_2d(
    signal: SignalND,
    *,
    spectrum: Spectrum,
    detector: Detector,
    theta_rad: float,
    use_complex_phase: bool = False,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray, np.ndarray]:
    """
    Process that applies a spectrum-weighted beam-obliquity transfer along detector-u.

    :param signal: CASYMIR SignalND object.
    :param spectrum: CASYMIR Spectrum object.
    :param detector: CASYMIR Detector object.
    :param theta_rad: Projection angle in radians.
    :param use_complex_phase: If True, retain the complex phase term in the signal transfer.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object, applied transfer function, and complex obliquity line response.
    """
    if signal.ndim != 2:
        raise ValueError("beam_obliquity_blur_2d expects a 2D SignalND.")

    thickness_mm = float(detector.thickness) / 1000.0
    line = _obliquity_transfer_u(signal.fx, theta_rad, thickness_mm, spectrum, detector)
    line_for_signal = line if use_complex_phase else np.abs(line)
    transfer = line_for_signal[:, None]

    out = signal if inplace else _copy_signal(signal)
    out.S = out.S * transfer
    return out, transfer, line


def _linear_mu_mm(detector: Detector, energy_keV: np.ndarray) -> np.ndarray:
    """Return the detector linear attenuation coefficient in inverse millimeters."""
    detector.get_mu(energy_keV)
    rho = float(detector.material["density"])
    packing = float(detector.material.get("pf", 1.0))
    return detector.mu * rho * packing / 10.0


def _obliquity_transfer_u(
    fu: np.ndarray,
    theta_rad: float,
    thickness_mm: float,
    spectrum: Spectrum,
    detector: Detector,
) -> np.ndarray:
    """Return the spectrum-weighted obliquity transfer along detector-u."""
    energy = np.asarray(spectrum.energy, dtype=float)
    fluence = np.asarray(spectrum.fluence, dtype=float)
    mu = _linear_mu_mm(detector, energy)

    MU = mu[:, None]
    FU = np.asarray(fu, dtype=float)[None, :]
    tprime = thickness_mm / np.cos(theta_rad)

    alpha = 2.0 * np.pi * FU * tprime * np.tan(theta_rad)
    beta = 2.0 * np.pi * FU * np.sin(theta_rad) / MU
    weights = (energy * fluence)[:, None]

    numerator = (1.0 - np.exp(-MU * tprime - 1j * alpha)) / (1.0 + 1j * beta)
    denominator = 1.0 - np.exp(-MU * tprime)

    num = integrate.trapezoid(weights * numerator, energy, axis=0)
    den = integrate.trapezoid(weights * denominator, energy, axis=0)
    return num / den


# Log normalization


def _apply_log_normalization(
    signal: SignalND,
    gain: float,
    *,
    inplace: bool = False,
) -> tuple[SignalND, float]:
    """Normalize signal and Wiener spectra by a scalar detector response."""
    gain = float(gain)
    if not np.isfinite(gain) or gain == 0.0:
        raise ValueError("Log normalization gain must be finite and nonzero.")

    out = signal if inplace else _copy_signal(signal)
    out.S = out.S / gain
    out.W = out.W / (gain**2)
    return out, gain


def log_normalize_by_detector_response_2d(
    signal: SignalND,
    *,
    spectrum: Spectrum,
    a: float,
    b: float,
    inplace: bool = False,
) -> tuple[SignalND, float]:
    """
    Process that applies mammography/DBT log normalization using detector response to air kerma.

    :param signal: CASYMIR SignalND object.
    :param spectrum: CASYMIR Spectrum object.
    :param a: Intercept of the detector response model.
    :param b: Air-kerma coefficient of the detector response model.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with normalized SignalND object and normalization gain.
    """
    gain = float(a) + float(b) * float(spectrum.dak)
    return _apply_log_normalization(signal, gain, inplace=inplace)


def log_normalize_by_signal_2d(
    signal: SignalND,
    *,
    inplace: bool = False,
) -> tuple[SignalND, float]:
    """
    Process that applies CT-style log normalization using the zero-frequency signal value.

    :param signal: CASYMIR SignalND object.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with normalized SignalND object and normalization gain.
    """
    if signal.ndim != 2:
        raise ValueError("log_normalize_by_signal_2d expects a 2D SignalND.")

    iu = int(np.argmin(np.abs(signal.fx)))
    iv = int(np.argmin(np.abs(signal.fy)))
    gain = float(np.abs(signal.S[iu, iv]))
    return _apply_log_normalization(signal, gain, inplace=inplace)


# Reconstruction filters


def _wrap_to_band(f: np.ndarray, half_width: float) -> np.ndarray:
    """Wrap frequencies into a centered periodic band."""
    return ((f + half_width) % (2.0 * half_width)) - half_width


def _dbt_radial_frequency(signal: SignalND, theta_rad: float) -> np.ndarray:
    """Return DBT radial frequency for one projection angle."""
    c = np.cos(theta_rad)
    if np.isclose(c, 0.0):
        raise ValueError("DBT view has cos(theta) close to zero.")
    return signal.FX / c


def _discrete_ram_lak_response(
    frequency: np.ndarray, spacing_mm: float, n_samples: int
) -> np.ndarray:
    """Evaluate the DFT of the sampled Ram-Lak kernel on a frequency grid."""
    n_samples = int(n_samples)
    spacing_mm = float(spacing_mm)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if spacing_mm <= 0.0:
        raise ValueError("spacing_mm must be positive.")

    indices = np.arange(n_samples)
    lag = np.minimum(indices, n_samples - indices).astype(float)
    kernel = np.zeros(n_samples, dtype=float)
    kernel[0] = 0.25
    odd = (lag.astype(np.int64) % 2) == 1
    kernel[odd] = -1.0 / (np.pi * lag[odd]) ** 2

    response = np.real(np.fft.fft(kernel)) / spacing_mm
    response = np.maximum(np.fft.fftshift(response), 0.0)
    frequency_samples = np.fft.fftshift(np.fft.fftfreq(n_samples, d=spacing_mm))
    return np.interp(frequency, frequency_samples, response, left=0.0, right=0.0)


def dbt_ramp_filter_2d(
    signal: SignalND,
    *,
    detector: Detector,
    theta_total_rad: float,
    theta_rad: float,
) -> np.ndarray:
    """
    Process that creates the ideal continuous DBT ramp filter in per-view radial frequency.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_total_rad: Total DBT angular span in radians.
    :param theta_rad: Projection angle in radians.

    :return: Ideal DBT ramp-filter transfer function.
    """
    pitch_u, _ = detector_pitch_uv(detector)
    fny = 1.0 / (2.0 * pitch_u)
    fr_ny = abs(fny / np.cos(theta_rad))
    fr = _wrap_to_band(_dbt_radial_frequency(signal, theta_rad), fr_ny)
    return (2.0 * np.tan(theta_total_rad) / fr_ny) * np.abs(fr)


def dbt_discrete_ram_lak_filter_2d(
    signal: SignalND,
    *,
    detector: Detector,
    theta_total_rad: float,
    theta_rad: float,
    n_samples_u: int | None = None,
) -> np.ndarray:
    """
    Process that creates the discrete Ram-Lak DBT filter in per-view radial frequency.

    Unlike "dbt_ramp_filter_2d", this function constructs the sampled
    spatial-domain Ram-Lak kernel and evaluates its DFT. The detector-u sampling
    interval is mapped into radial coordinates for the selected projection angle.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_total_rad: Total DBT angular span in radians.
    :param theta_rad: Projection angle in radians.
    :param n_samples_u: Number of detector-u samples used for the discrete kernel.
        If None, len(signal.fx) is used.

    :return: Discrete DBT Ram-Lak transfer function.
    """
    fr = _dbt_radial_frequency(signal, theta_rad)
    pitch_u, _ = detector_pitch_uv(detector)
    radial_pitch = pitch_u * abs(np.cos(theta_rad))
    fr_ny = 1.0 / (2.0 * radial_pitch)
    fr = _wrap_to_band(fr, fr_ny)
    n_samples_u = len(signal.fx) if n_samples_u is None else int(n_samples_u)
    response = _discrete_ram_lak_response(fr, radial_pitch, n_samples_u)
    return (2.0 * np.tan(theta_total_rad) / fr_ny) * response


def dbt_apodization_filter_2d(
    signal: SignalND,
    *,
    detector: Detector,
    theta_rad: float,
    width: float = 1.5,
) -> np.ndarray:
    """
    Process that creates the DBT spectral apodization filter in radial frequency.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_rad: Projection angle in radians.
    :param width: Apodization width relative to the radial Nyquist frequency.

    :return: DBT apodization transfer function.
    """
    pitch_u, _ = detector_pitch_uv(detector)
    fny = 1.0 / (2.0 * pitch_u)
    fr_ny = abs(fny / np.cos(theta_rad))
    fr = _wrap_to_band(_dbt_radial_frequency(signal, theta_rad), fr_ny)

    r = np.abs(fr) / (float(width) * fr_ny)
    transfer = np.zeros_like(fr, dtype=np.float32)
    inside = r <= 1.0
    transfer[inside] = 0.5 * (1.0 + np.cos(np.pi * r[inside]))
    return transfer


def dbt_interpolation_filter_2d(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    theta_rad: float = 0.0,
    spacing_u_mm: float | None = None,
    spacing_v_mm: float | None = None,
    power: int = 2,
) -> np.ndarray:
    """
    Process that creates the DBT interpolation filter in radial and detector-v frequency.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if spacing_u_mm or spacing_v_mm is not provided.
    :param theta_rad: Projection angle in radians.
    :param spacing_u_mm: Interpolation spacing along detector-u in millimeters.
    :param spacing_v_mm: Interpolation spacing along detector-v in millimeters.
    :param power: Interpolation filter power.

    :return: DBT interpolation transfer function.
    """
    if spacing_u_mm is None or spacing_v_mm is None:
        if detector is None:
            raise ValueError("Provide detector or both interpolation spacings.")
        spacing_u_mm, spacing_v_mm = detector_pitch_uv(detector)

    fr = _dbt_radial_frequency(signal, theta_rad)
    transfer = np.sinc(float(spacing_u_mm) * fr) * np.sinc(
        float(spacing_v_mm) * signal.FY
    )
    if power == 2:
        transfer = transfer * transfer
    return transfer


def cbct_ramp_filter_2d(signal: SignalND, *, epsilon: float = 0.0) -> np.ndarray:
    """
    Process that creates the ideal continuous CBCT ramp filter in detector-u frequency.

    :param signal: CASYMIR SignalND object.
    :param epsilon: Optional lower bound for the ramp transfer function.

    :return: Ideal CBCT ramp-filter transfer function.
    """
    transfer = np.abs(signal.FX)
    if epsilon > 0.0:
        transfer = np.maximum(transfer, float(epsilon))
    return transfer


def cbct_discrete_ram_lak_filter_2d(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    n_samples_u: int | None = None,
) -> np.ndarray:
    """
    Process that creates the discrete Ram-Lak CBCT filter in detector-u frequency.

    Unlike "cbct_ramp_filter_2d", this function constructs the sampled
    spatial-domain Ram-Lak kernel:

        h[0] = 1 / 4

        h[n] = -1 / (pi * n)^2   for odd n

        h[n] = 0                 for even n

    The kernel is Fourier transformed and converted from cycles per sample to
    physical frequency units of inverse millimeters.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm is not provided.
    :param pitch_u_mm: Detector-u pixel pitch in millimeters.
    :param n_samples_u: Number of detector-u samples used for the discrete kernel.
        If None, len(signal.fx) is used.

    :return: Discrete CBCT Ram-Lak transfer function.
    """
    if pitch_u_mm is None:
        if detector is None:
            raise ValueError("Provide detector or pitch_u_mm.")
        pitch_u_mm, _ = detector_pitch_uv(detector)

    n_samples_u = len(signal.fx) if n_samples_u is None else int(n_samples_u)
    line = _discrete_ram_lak_response(signal.fx, float(pitch_u_mm), n_samples_u)
    return np.broadcast_to(line[:, None], signal.S.shape).copy()


# Name used during validation development
cbct_ambr_ramp_filter_2d = cbct_discrete_ram_lak_filter_2d


def cbct_apodization_filter_2d(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    bandwidth: float = 1.0,
    hwin: float = 0.5,
) -> np.ndarray:
    """
    Process that creates the CBCT apodization filter in detector-u frequency.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm is not provided.
    :param pitch_u_mm: Detector-u pitch in millimeters.
    :param bandwidth: Relative filter bandwidth.
    :param hwin: Hanning-window offset.

    :return: CBCT apodization transfer function.
    """
    if pitch_u_mm is None:
        if detector is None:
            raise ValueError("Provide detector or pitch_u_mm.")
        pitch_u_mm, _ = detector_pitch_uv(detector)

    f_limit = 1.0 / (2.0 * float(pitch_u_mm) * float(bandwidth))
    transfer = np.zeros_like(signal.FX, dtype=np.float32)
    inside = np.abs(signal.FX) <= f_limit
    transfer[inside] = float(hwin) + (1.0 - float(hwin)) * np.cos(
        2.0 * np.pi * signal.FX[inside] * float(pitch_u_mm) * float(bandwidth)
    )
    return transfer


def cbct_interpolation_filter_2d(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    pitch_v_mm: float | None = None,
    power: int = 2,
) -> np.ndarray:
    """
    Process that creates the CBCT bilinear interpolation filter in detector coordinates.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm or pitch_v_mm is not provided.
    :param pitch_u_mm: Detector-u pitch in millimeters.
    :param pitch_v_mm: Detector-v pitch in millimeters.
    :param power: Interpolation filter power.

    :return: CBCT interpolation transfer function.
    """
    if pitch_u_mm is None or pitch_v_mm is None:
        if detector is None:
            raise ValueError("Provide detector or both detector pitches.")
        pitch_u_mm, pitch_v_mm = detector_pitch_uv(detector)

    transfer = np.sinc(float(pitch_u_mm) * signal.FX) * np.sinc(
        float(pitch_v_mm) * signal.FY
    )
    if power == 2:
        transfer = transfer * transfer
    return transfer


def apply_dbt_ramp_filter(
    signal: SignalND,
    *,
    detector: Detector,
    theta_total_rad: float,
    theta_rad: float,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the ideal continuous DBT ramp filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_total_rad: Total DBT angular span in radians.
    :param theta_rad: Projection angle in radians.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied ideal ramp transfer function.
    """
    transfer = dbt_ramp_filter_2d(
        signal, detector=detector, theta_total_rad=theta_total_rad, theta_rad=theta_rad
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_dbt_discrete_ram_lak_filter(
    signal: SignalND,
    *,
    detector: Detector,
    theta_total_rad: float,
    theta_rad: float,
    n_samples_u: int | None = None,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the discrete Ram-Lak DBT filter.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_total_rad: Total DBT angular span in radians.
    :param theta_rad: Projection angle in radians.
    :param n_samples_u: Number of detector-u samples used for the discrete kernel.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied Ram-Lak transfer function.
    """
    transfer = dbt_discrete_ram_lak_filter_2d(
        signal,
        detector=detector,
        theta_total_rad=theta_total_rad,
        theta_rad=theta_rad,
        n_samples_u=n_samples_u,
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_dbt_apodization_filter(
    signal: SignalND,
    *,
    detector: Detector,
    theta_rad: float,
    width: float = 1.5,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the DBT spectral apodization filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object.
    :param theta_rad: Projection angle in radians.
    :param width: Apodization width relative to the radial Nyquist frequency.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied apodization transfer function.
    """
    transfer = dbt_apodization_filter_2d(
        signal, detector=detector, theta_rad=theta_rad, width=width
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_dbt_interpolation_filter(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    theta_rad: float = 0.0,
    spacing_u_mm: float | None = None,
    spacing_v_mm: float | None = None,
    power: int = 2,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the DBT interpolation filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if spacing_u_mm or spacing_v_mm is not provided.
    :param theta_rad: Projection angle in radians.
    :param spacing_u_mm: Interpolation spacing along detector-u in millimeters.
    :param spacing_v_mm: Interpolation spacing along detector-v in millimeters.
    :param power: Interpolation filter power.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied interpolation transfer function.
    """
    transfer = dbt_interpolation_filter_2d(
        signal,
        detector=detector,
        theta_rad=theta_rad,
        spacing_u_mm=spacing_u_mm,
        spacing_v_mm=spacing_v_mm,
        power=power,
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_cbct_ramp_filter(
    signal: SignalND,
    *,
    epsilon: float = 0.0,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the ideal continuous CBCT ramp filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param epsilon: Optional lower bound for the ramp transfer function.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied ideal ramp transfer function.
    """
    transfer = cbct_ramp_filter_2d(signal, epsilon=epsilon)
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_cbct_discrete_ram_lak_filter(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    n_samples_u: int | None = None,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the discrete Ram-Lak CBCT filter.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm is not provided.
    :param pitch_u_mm: Detector-u pixel pitch in millimeters.
    :param n_samples_u: Number of detector-u samples used for the discrete kernel.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied Ram-Lak transfer function.
    """
    transfer = cbct_discrete_ram_lak_filter_2d(
        signal,
        detector=detector,
        pitch_u_mm=pitch_u_mm,
        n_samples_u=n_samples_u,
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


# Backward-compatible name used during validation development.
apply_cbct_ambr_ramp_filter = apply_cbct_discrete_ram_lak_filter


def apply_cbct_apodization_filter(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    bandwidth: float = 1.0,
    hwin: float = 0.5,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the CBCT apodization filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm is not provided.
    :param pitch_u_mm: Detector-u pitch in millimeters.
    :param bandwidth: Relative filter bandwidth.
    :param hwin: Hanning-window offset.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied apodization transfer function.
    """
    transfer = cbct_apodization_filter_2d(
        signal,
        detector=detector,
        pitch_u_mm=pitch_u_mm,
        bandwidth=bandwidth,
        hwin=hwin,
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)


def apply_cbct_interpolation_filter(
    signal: SignalND,
    *,
    detector: Detector | None = None,
    pitch_u_mm: float | None = None,
    pitch_v_mm: float | None = None,
    power: int = 2,
    inplace: bool = False,
) -> tuple[SignalND, np.ndarray]:
    """
    Process that applies the CBCT interpolation filter to a 2D projection-domain signal.

    :param signal: CASYMIR SignalND object.
    :param detector: CASYMIR Detector object. Required if pitch_u_mm or pitch_v_mm is not provided.
    :param pitch_u_mm: Detector-u pitch in millimeters.
    :param pitch_v_mm: Detector-v pitch in millimeters.
    :param power: Interpolation filter power.
    :param inplace: If True, modify the input SignalND object in place.

    :return: Tuple with updated SignalND object and applied interpolation transfer function.
    """
    transfer = cbct_interpolation_filter_2d(
        signal,
        detector=detector,
        pitch_u_mm=pitch_u_mm,
        pitch_v_mm=pitch_v_mm,
        power=power,
    )
    return apply_transfer_2d(signal, transfer, inplace=inplace)
