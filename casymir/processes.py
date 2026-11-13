"""One-dimensional detector-cascade process blocks.

The functions in this module operate on the original radial-frequency
``Signal`` representation. Projection-domain and reconstruction-domain
extensions are provided by :mod:`casymir.processes_2d` and
:mod:`casymir.processes_3d`.
"""

from __future__ import annotations

import copy

import casymir.casymir
import casymir.parallel
import numpy as np
from scipy import integrate
from casymir.casymir import Signal

# Input signals


def initial_signal(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    oversampling: int = 1,
    *,
    detector_position_uv_mm: tuple[float, float] | None = None,
    sdd_mm: float | None = None,
) -> tuple[Signal, float, np.ndarray]:
    """
    Initializes a CASYMIR Signal object with spatial frequency information and fluence reaching the detector.

    The spectrum fluence is assumed to correspond to the central ray. If a
    detector plane position is supplied, the fluence is corrected for
    point-source inverse-square falloff and oblique incidence onto a flat
    detector:

        q(u, v) = q0 * cos(theta)^3

        cos(theta) = SDD / sqrt(SDD^2 + u^2 + v^2)

    This geometry applies to both DBT and CBCT. Detector coordinates (u, v)
    are specified in millimeters relative to the central ray.

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object.
    :param oversampling: Factor defining the maximum frequency relative to the sampling frequency.
    :param detector_position_uv_mm: Optional detector-plane position ``(u, v)``
        in millimeters relative to the central ray.
    :param sdd_mm: Source-to-detector distance in millimeters. Required when
        detector_position_uv_mm is provided.

    :return: Tuple containing the initialized CASYMIR Signal object, mean gain factor, and spread function.
    """
    f_s = 1 / detector.px_size
    f_max = oversampling * f_s
    n = int(detector.elems)
    f = np.linspace(0, f_max, n, endpoint=False)

    q0 = spectrum.get_fluence()
    if (detector_position_uv_mm is None) != (sdd_mm is None):
        raise ValueError("Provide detector_position_uv_mm and sdd_mm together.")

    if detector_position_uv_mm is not None:
        u_mm, v_mm = detector_position_uv_mm
        sdd_mm = float(sdd_mm)
        if not np.all(np.isfinite([u_mm, v_mm])):
            raise ValueError("Detector position must contain finite values.")
        if not np.isfinite(sdd_mm) or sdd_mm <= 0.0:
            raise ValueError("sdd_mm must be finite and positive.")

        ray_distance_mm = np.sqrt(sdd_mm**2 + float(u_mm)**2 + float(v_mm)**2)
        q0 *= (sdd_mm / ray_distance_mm) ** 3

    sig = casymir.casymir.Signal(
        f,
        q0,
        q0 * np.ones(np.size(f), dtype=np.float64),
        q0 * np.ones(np.size(f), dtype=np.float64),
    )

    gain = q0
    spread = np.ones(np.size(f))

    return sig, gain, spread

# Absorbed by initial_signal, kept for validation script compatibility
def initial_signal_cbct_position(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    detector_position_uv: tuple[float, float],
    detector_shape_uv: tuple[int, int],
    sdd_mm: float,
    oversampling: int = 1,
) -> tuple[Signal, float, np.ndarray]:

    u_index, v_index = detector_position_uv
    n_u, _ = detector_shape_uv
    u_mm = (u_index + 0.5 - n_u / 2) * detector.px_size
    v_mm = (v_index + 0.5) * detector.px_size
    return initial_signal(
        detector,
        spectrum,
        oversampling,
        detector_position_uv_mm=(u_mm, v_mm),
        sdd_mm=sdd_mm,
    )


# Detector cascade


def quantum_selection(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    signal: casymir.casymir.Signal,
) -> tuple[Signal, float, np.ndarray]:
    """
    Quantum selection block. Updates the signal to only consider quanta absorbed by the detector layer, based on the
    Quantum Efficiency (QE) of the detector. This is modeled as a stochastic gain process with Bernoulli statistics.

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object.
    :param signal: CASYMIR Signal object to be updated.

    :return: Tuple with updated CASYMIR Signal object, mean gain, and spread function.
    """
    # Quantum Efficiency (QE) for the given energy spectrum
    QE = get_cached_QE(detector, spectrum.energy)

    # Normalized fluence spectrum
    spec_norm = spectrum.fluence / integrate.simpson(spectrum.fluence, spectrum.energy)

    # Mean quantum efficiency
    g1 = integrate.simpson(QE * spec_norm, spectrum.energy)
    signal_2 = copy.copy(signal)
    # Apply stochastic gain to the signal
    signal_2.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)))

    # Mean gain and spread functions associated to the process. In this case, the gain is equal to the mean Quantum
    # Efficiency, and there is no spread introduced by this selection process.
    gain = g1
    spread = np.ones(np.size(signal.freq))

    return signal_2, gain, spread


def absorption(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, float, np.ndarray]:
    """
    Process that models the conversion of x-rays into secondary quanta (electron-hole pairs or optical photons for
    direct and indirect conversion detectors, respectively). For both types of detectors, the conversion process is
    assumed to follow Poisson statistics.

    The mean gain factor gA is calculated according to:

    Vedantham et al. Solid-state fluoroscopic imager for high-resolution angiography: Parallel-cascaded
    linear systems analysis. Medical Physics 31, 1258 (2004); doi: 10.1118/1.1689014

    Zhao, Rowlands. Effects of characteristic x-rays on the noise power spectra and detective quantum
    efficiency of photoconductive x-ray detectors. Medical Physics 28, 2039 (2001); doi: 10.1118/1.1405845

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    energy = spectrum.energy
    fluence = spectrum.fluence
    QE = get_cached_QE(detector, energy)

    xi = detector.material["xi"]

    diffEnergy = float(energy[1] - energy[0])
    kV = spectrum.kV

    if detector.type == "direct":
        Ma, MaDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, _ = calculate_diff(E1, detector.material)
            Ma[k] = (1 - xi * w0) * (E1 / detector.material["w"])
            MaDenom[k] = QE[k] * (1 - xi * w0)

    else:
        Ma, MaDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        com_term_z = detector.com_term_z(energy)
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, _ = calculate_diff(E1, detector.material)
            com_term = com_term_z[:, k]
            Ma[k] = (
                integrate.simpson(com_term, dx=1)
                * (1 - xi * w0)
                * E1
                * detector.material["m0"]
            )
            MaDenom[k] = QE[k] * (1 - xi * w0)

    # Mean gain and spread functions associated to the process.
    gA = integrate.simpson(Ma * fluence, energy) / integrate.simpson(
        MaDenom * fluence, energy
    )
    spread = np.ones(np.size(sig.freq))

    mu1 = Ma
    mu2 = mu1**2 + mu1
    valid = mu1 > 0

    Ia = calculate_effective_swank(
        energy[valid], fluence[valid], QE[valid], mu1[valid], mu2[valid]
    )

    swank_eff = Ia
    excess = gA * ((1 / swank_eff) - 1) - 1
    var_A = gA * (excess + 1)

    sig2A = casymir.casymir.Signal(sig.freq, sig.mean_quanta, sig.signal, sig.wiener)
    sig2A.stochastic_gain(gA, np.sqrt(var_A))

    sig3 = copy.copy(sig)
    sig3.mean_quanta = sig2A.mean_quanta
    sig3.signal = sig2A.signal
    sig3.wiener = sig2A.wiener

    return sig3, gA, spread


def local_k_absorption(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, float, np.ndarray]:
    """
    Process that models the local absorption and conversion of characteristic x-rays into secondary quanta
    (electron-hole pairs or optical photons for direct and indirect conversion detectors, respectively).
    For both types of detectors, the conversion process is assumed to follow Poisson statistics.

    The mean gain factor gB is calculated according to:

    Vedantham et al. Solid-state fluoroscopic imager for high-resolution angiography: Parallel-cascaded
    linear systems analysis. Medical Physics 31, 1258 (2004); doi: 10.1118/1.1689014

    Zhao, Rowlands. Effects of characteristic x-rays on the noise power spectra and detective quantum
    efficiency of photoconductive x-ray detectors. Medical Physics 28, 2039 (2001); doi: 10.1118/1.1405845

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    energy = spectrum.energy
    fluence = spectrum.fluence
    QE = get_cached_QE(detector, energy)

    xi = detector.material["xi"]
    omega = detector.material["omega"]
    ek = detector.material["ek"]

    # Set probability to zero below K-edge, include xi * omega
    prob_k_absorption = np.where(energy < ek, 0.0, xi * omega)
    diffEnergy = float(energy[1] - energy[0])
    kV = spectrum.kV

    if spectrum.kV <= detector.material["ek"]:
        spread = np.ones(np.size(sig.freq))
        return copy.copy(sig), 0.0, spread

    if detector.type == "direct":
        Mb, MbDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, diff = calculate_diff(E1, detector.material)
            Mb[k] = prob_k_absorption[k] * (diff / detector.material["w"])
            MbDenom[k] = QE[k] * prob_k_absorption[k]

    else:
        Mb, MbDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        com_term_z = detector.com_term_z(energy)
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, diff = calculate_diff(E1, detector.material)
            com_term = com_term_z[:, k]
            Mb[k] = (
                integrate.simpson(com_term, dx=1)
                * prob_k_absorption[k]
                * detector.material["m0"]
                * diff
            )
            MbDenom[k] = QE[k] * prob_k_absorption[k]

    # Mean gain and spread functions associated to the process
    gB = integrate.simpson(Mb * fluence, energy) / integrate.simpson(
        MbDenom * fluence, energy
    )
    spread = np.ones(np.size(sig.freq))

    mu1 = Mb
    mu2 = mu1**2 + mu1
    valid = mu1 > 0

    Ib = calculate_effective_swank(
        energy[valid], fluence[valid], QE[valid], mu1[valid], mu2[valid]
    )

    swank_eff = Ib
    excess = gB * ((1 / swank_eff) - 1) - 1
    var_B = gB * (excess + 1)

    sig2B = casymir.casymir.Signal(sig.freq, sig.mean_quanta, sig.signal, sig.wiener)
    sig2B.stochastic_gain(gB, np.sqrt(var_B))

    sig3 = copy.copy(sig)
    sig3.mean_quanta = sig2B.mean_quanta
    sig3.signal = sig2B.signal
    sig3.wiener = sig2B.wiener

    return sig3, gB, spread


def remote_k_absorption(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, float, np.ndarray]:
    """
    Process that models the remote absorption and conversion of characteristic x-rays into secondary quanta
    (electron-hole pairs or optical photons for direct and indirect conversion detectors, respectively).
    For both types of detectors, the conversion process is assumed to follow Poisson statistics.

    The mean gain factor gC is calculated according to:

    Vedantham et al. Solid-state fluoroscopic imager for high-resolution angiography: Parallel-cascaded
    linear systems analysis. Medical Physics 31, 1258 (2004); doi: 10.1118/1.1689014

    Zhao, Rowlands. Effects of characteristic x-rays on the noise power spectra and detective quantum
    efficiency of photoconductive x-ray detectors. Medical Physics 28, 2039 (2001); doi: 10.1118/1.1405845

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    energy = spectrum.energy
    fluence = spectrum.fluence
    tk = detector.calculate_Tk(sig)
    QE = get_cached_QE(detector, energy)

    xi = detector.material["xi"]
    ek = detector.material["ek"]

    diffEnergy = float(energy[1] - energy[0])
    kV = spectrum.kV

    if spectrum.kV <= detector.material["ek"]:
        spread = np.ones(np.size(sig.freq))
        return copy.copy(sig), 0.0, spread

    if detector.type == "direct":
        Mc, McDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, _ = calculate_diff(E1, detector.material)
            Mc[k] = xi * w0 * detector.fk * ek / detector.material["w"]
            McDenom[k] = QE[k] * xi * w0 * detector.fk

    else:
        Mc, McDenom = [np.zeros(np.shape(energy)) for _ in range(2)]
        com_term_z = detector.com_term_z(energy)
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, _ = calculate_diff(E1, detector.material)
            com_term = com_term_z[:, k]
            Mc[k] = (
                integrate.simpson(com_term, dx=1)
                * xi
                * w0
                * detector.fk
                * ek
                * detector.material["m0"]
            )
            McDenom[k] = QE[k] * xi * w0 * detector.fk

    # Mean gain associated to this process
    gC = integrate.simpson(Mc * fluence, energy) / integrate.simpson(
        McDenom * fluence, energy
    )

    mu1 = Mc
    mu2 = mu1**2 + mu1
    valid = mu1 > 0

    Ic = calculate_effective_swank(
        energy[valid], fluence[valid], QE[valid], mu1[valid], mu2[valid]
    )

    swank_eff = Ic
    excess = gC * ((1 / swank_eff) - 1) - 1
    var_C = gC * (excess + 1)

    sig2C = copy.copy(sig)
    fk = detector.fk
    sig2C.stochastic_gain(fk, np.sqrt(fk * (1 - fk)))
    sig2C.stochastic_blur(tk)
    sig2C.stochastic_gain(gC, np.sqrt(var_C))

    return sig2C, gC * detector.fk, tk


def absorption_block(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> casymir.casymir.Signal:
    """
    Parallel block that models the conversion of x-rays into secondary quanta, taking into account
    three possible interactions:

    - Absorption of x-rays (no characteristic x-ray produced)
    - Generation and local absorption of characteristic x-rays
    - Generation and remote absorption of characteristic x-rays

    :param detector: CASYMIR Detector object (either direct or indirect conversion).
    :param spectrum: CASYMIR Spectrum object.
    :param sig: CASYMIR Signal object representing the input signal.

    :return: CASYMIR Signal object after applying the parallel processes.
    """
    # Create Paths (branches)
    # A: absorption (no k-fluorescence)
    # B: local absorption of characteristic x-ray
    # C: remote absorption of characteristic x-ray
    PathA = casymir.parallel.Path(
        processes=[casymir.processes.absorption], name="Path A"
    )
    PathB = casymir.parallel.Path(
        processes=[casymir.processes.local_k_absorption], name="Path B"
    )
    PathC = casymir.parallel.Path(
        processes=[casymir.processes.remote_k_absorption], name="Path C"
    )

    # Create Nodes
    Node1 = casymir.parallel.Node(
        node_type="Bernoulli"
    )  # First split: Absorption vs. Fluorescence
    Node2 = casymir.parallel.Node(
        node_type="Fork"
    )  # Second split: Local vs. Remote Fluorescence Absorption

    # Attach Paths to Nodes.
    Node1.add(
        PathA, probability=1 - detector.material["xi"] * detector.material["omega"]
    )  # Absorption
    Node2.add(
        PathB, probability=1
    )  # Local Fluorescence (always occurs if fluorescence occurs)
    Node2.add(
        PathC, probability=1
    )  # Remote Fluorescence (always occurs if fluorescence occurs)

    # Attach second Node to parent Node. The probability of the fork split is equal to xi*omega
    Node1.add(Node2, probability=detector.material["xi"] * detector.material["omega"])

    # Apply parallel process goes through all Nodes and Paths, returning the combined signal with
    # the cross-spectral terms taken into account.
    comb_sig = casymir.parallel.apply_parallel_process(Node1, sig, detector, spectrum)

    return comb_sig


def charge_trapping(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, int, np.ndarray]:
    """
    Models the blurring caused by charge trapping in the semiconductor layer. The implementation
    uses the expression presented in Zhao et al. Imaging performance of amorphous selenium based flat-panel
    detectors for digital mammography: Characterization of a small area prototype detector.
    Med Phys. 2003;30(2):254-263. doi:10.1118/1.1538233

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object. Used as an argument for consistency.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    t = detector.thickness * 1e-3  # Nominal detector layer thickness [mm]
    l = detector.trapping_depth * 1e-3  # Charge collection layer distance [mm]
    f = sig.freq
    # Calculate charge redistribution function (tb)
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = (t * np.sinh(2 * np.pi * f * (t - l))) / (
            (t - l) * np.sinh(2 * np.pi * f * t)
        )
    tb[0] = 1

    signal_2 = copy.copy(sig)

    signal_2.stochastic_blur(tb)

    return signal_2, 1, tb


def optical_blur(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, int, np.ndarray]:
    """
    Models the stochastic blurring process caused by the spreading of optical photons
    in the scintillator material of an indirect conversion detector.

    Pautasso et al. Technical note: Characterization, validation, and spectral optimization of a dedicated breast CT
    system for contrast-enhanced imaging. Med Phys. 2024;51(5):3322-3333. doi:10.1002/mp.17069

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object. Used as an argument for consistency.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    H = detector.material["spread_coeff"]
    f = sig.freq

    # Calculate the optical spread function (osf)
    osf = 1 / (1 + H * f + H * f**2 + H**2 * f**3)
    signal_2 = copy.copy(sig)

    # Apply stochastic blur using the optical spread function (osf)
    signal_2.stochastic_blur(osf)

    return signal_2, 1, osf


def optical_coupling(
    detector: casymir.casymir.Detector,
    spectrum: casymir.casymir.Spectrum,
    sig: casymir.casymir.Signal,
) -> tuple[Signal, float, np.ndarray]:
    """
    This function models the coupling of optical photons to the photodiode of a detector, where the probability
    of an incident quantum being coupled is determined by the average coupling efficiency "ce". This process is modeled
    as a quantum selection; that is, a stochastic gain stage with Bernoulli statistics.

    :param detector: CASYMIR Detector object.
    :param spectrum: CASYMIR Spectrum object. Used as an argument for consistency.
    :param sig: CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """
    # Extract the coupling efficiency from the detector
    gc = detector.material["ce"]

    # Optical coupling is represented by a stochastic gain stage with Bernoulli statistics.
    signal_2 = copy.copy(sig)
    signal_2.stochastic_gain(gc, (gc * (1 - gc)) ** (1 / 2))

    return signal_2, gc, np.ones(np.size(sig.freq))


# Sampling and output


def q_integration(
    detector: casymir.casymir.Detector, sig: casymir.casymir.Signal
) -> tuple[Signal, float, np.ndarray]:
    """
    Quantum integration block. Includes deterministic blurring by the pixel aperture function (Ta) and integration
    of the signal over the active pixel area.

    :param detector: A CASYMIR Detector object.
    :param sig: A CASYMIR Signal object.

    :return: Tuple with updated Signal object, mean gain, and spread function.
    """

    # Deterministic blurring by pixel aperture function Ta
    sampling_aperture = np.sqrt(detector.pxa)
    ta = np.abs(np.sinc(sampling_aperture * sig.freq))
    signal_2 = copy.copy(sig)
    signal_2.deterministic_blur(ta)
    # Integration as deterministic gain by the active pixel area
    signal_2.stochastic_gain(detector.pxa, 0)

    return signal_2, detector.pxa, ta


def noise_aliasing(
    detector: casymir.casymir.Detector, sig: casymir.casymir.Signal
) -> casymir.casymir.Signal:
    """
    Apply aliasing to the Wiener spectrum of a signal object.

    The aliasing process is modeled by summing shifted copies of the Wiener spectrum at multiples of the sampling
    frequency (f_s). This operation is equivalent to a convolution in the frequency domain with a Dirac comb
    with period f_s.

    :param detector: A CASYMIR Detector object containing pixel size information.
    :param sig: A CASYMIR Signal object representing the input signal.

    :return: A new CASYMIR Signal object with the aliased Wiener spectrum.
    """

    f_s = 1 / detector.px_size

    # Extend frequency vector and Wiener spectrum symmetrically
    f_max = sig.freq[-1]
    f_mirrored = np.concatenate((-np.flip(sig.freq[1:]), sig.freq))
    wiener_mirrored = np.concatenate((np.flip(sig.wiener[1:]), sig.wiener))

    # Number of valid harmonics given the frequency range.
    num_k_samples = int(np.ceil(f_max / f_s))
    # Vector containing valid k indices.
    k_values = np.arange(-num_k_samples, num_k_samples + 1)

    wiener_resampled = np.zeros(np.shape(wiener_mirrored))

    # Summation of W(f-k*f_s) over k.
    for k in k_values:
        shifted_indices = np.searchsorted(f_mirrored, f_mirrored - k * f_s)
        # Check for values outside the frequency range.
        valid_mask = (shifted_indices >= 0) & (shifted_indices < len(f_mirrored))
        wiener_resampled[valid_mask] += wiener_mirrored[shifted_indices[valid_mask]]

    # Extract only the positive frequency components
    valid_indices = f_mirrored >= 0
    f_resampled = f_mirrored[valid_indices]
    resampled_wiener = wiener_resampled[valid_indices]

    # Create a new Signal object with the resampled Wiener spectrum
    resampled_signal = casymir.casymir.Signal(
        freq=f_resampled,
        signal=sig.signal,
        wiener=resampled_wiener,
        mean_quanta=sig.mean_quanta,
    )

    return resampled_signal


def model_output(
    detector: casymir.casymir.Detector, signal: casymir.casymir.Signal
) -> casymir.casymir.Signal:
    """
    Add electronic noise to the Wiener spectrum and compute the model output (MTF and NNPS).

    :param detector: CASYMIR Detector object containing electronic noise properties.
    :param signal: CASYMIR Signal object.
    :return: A new CASYMIR Signal object with electronic noise applied, as well as MTF and NNPS attributes.
    """

    mtf = signal.signal / signal.signal[0]
    add_noise = detector.add_noise

    # Apply electronic noise to the Wiener spectrum (NPS) using the entire pixel area
    wiener2 = signal.wiener[0 : int(signal.length / 2)] + (
        (add_noise**2) * (detector.pxa / detector.ff)
    )
    signal2 = signal.signal[0 : int(signal.length / 2)]

    # Frequency vector up to Nyquist frequency
    f2 = signal.freq[0 : int(signal.length / 2)]
    mtf = mtf[0 : int(signal.length / 2)]

    # Normalized NPS (Noise Power Spectrum divided by large area signal)
    nnps = wiener2 / (signal.signal[0] ** 2)

    output_signal = casymir.casymir.Signal(
        freq=f2, signal=signal2, wiener=wiener2, mean_quanta=signal.mean_quanta
    )

    output_signal.mtf = mtf
    output_signal.nnps = nnps

    return output_signal


# Numerical helpers


def calculate_diff(E1: float, material: dict) -> tuple[float, float]:
    """
    Computes the absorbed energy from K-fluorescence x-rays.

    :param E1: Incident x-ray energy in keV.
    :param material: Detector material properties.

    :return: Fluorescence yield and energy deposited locally.
    """
    if E1 < material["ek"]:
        w0 = 0.0
        diff = 0.0
    else:
        w0 = material["omega"]
        diff = E1 - material["ek"]
    return w0, diff


def get_cached_QE(detector: casymir.casymir.Detector, energy: np.ndarray) -> np.ndarray:
    """
    Checks whether the Detector's Quantum Efficiency attribute is already present, to avoid unnecessary calls to the
    get_QE() method.

    :param detector: CASYMIR Detector object.
    :param energy: X-ray energy samples in keV.

    :return: Quantum efficiency evaluated at the supplied energies.
    """
    if (
        detector.QE is None
        or not isinstance(detector.QE, np.ndarray)
        or detector.QE.size == 0
    ):
        detector.get_QE(energy)
    return detector.QE


def calculate_effective_swank(
    energy: np.ndarray,
    fluence: np.ndarray,
    QE: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
) -> float:
    """
    Calculates the effective Swank factor by integrating the absorbed-spectrum-weighted gain moments.

    :param energy: X-ray energy samples in keV.
    :param fluence: Spectral photon fluence.
    :param QE: Quantum efficiency at each energy.
    :param mu1: First gain moment at each energy.
    :param mu2: Second gain moment at each energy.

    :return: Effective Swank factor.
    """
    absorbed = fluence * QE
    norm = integrate.simpson(absorbed, energy)

    absorbed_pdf = absorbed / norm

    num = integrate.simpson(mu1 * absorbed_pdf, energy) ** 2
    denom = integrate.simpson(mu2 * absorbed_pdf, energy)
    return num / denom
