"""
casymir.processes
~~~~~~~~~~~~~~~~~~

This module provides implementations of common blocks in the detector signal chain.

Functions:
- initial_signal: Creates an initial signal object based on the input spectrum.

- quantum_selection: Quantum selection block.

- parallel_gain_direct: Parallel gain block simulating fluorescent x-ray production and absoprtion in semiconductors.

- parallel_gain_indirect: Parallel gain block simulating fluorescent x-ray production and conversion in scintillators.

- charge_trapping: Charge redistribution for direct conversion detector.

- optical_blur: Optical photon spreading by scintillator materials.

- optical_coupling: Coupling of optical photons to photodiode in indirect conversion detectors.

- px_integration: Integration of the signal by detector pixel element, followed by sampling and pixel readout.
"""

import casymir.casymir
import numpy as np
from scipy import integrate
import copy


def initial_signal(detector: casymir.casymir.Detector, spectrum: casymir.casymir.Spectrum) -> casymir.casymir.Signal:
    """
    This function initializes a CASYMIR Signal object with spatial frequency information and the magnitude equal to
    the fluence reaching the detector from the given spectrum.

    :param detector: A CASYMIR Detector object containing the characteristics of the detector.
    :param spectrum: A CASYMIR Spectrum object representing the input spectrum.

    :return: A CASYMIR Signal object initialized with spatial frequency and magnitude information.
    :rtype: casymir.casymir.Signal
    """
    # Number of elements within the frequency range of 0 to Nyquist
    n = int(detector.elems / 2)

    # Array length
    R = np.arange(2 * n)

    # Spatial frequency cycles/mm
    f = R / (detector.px_size * (2 * n))

    # Photon fluence at detector surface [photons/mm2]
    q0 = spectrum.get_fluence()

    # Create a CASYMIR Signal object
    sig = casymir.casymir.Signal(f, q0 * np.ones(np.size(f), dtype=np.float64),
                                 q0 * np.ones(np.size(f), dtype=np.float64))

    return sig


def quantum_selection(detector: casymir.casymir.Detector, spectrum: casymir.casymir.Spectrum,
                      signal: casymir.casymir.Signal) -> casymir.casymir.Signal:
    """
    Quantum selection block. Updates the signal to only consider quanta absorbed by the detector layer, based on the
    Quantum Efficiency (QE) of the detector.

    :param detector: A CASYMIR Detector object containing the characteristics of the detector.
    :param spectrum: A CASYMIR Spectrum object representing the input spectrum.
    :param signal: A CASYMIR Signal object representing the signal to be updated.

    :return: An updated CASYMIR Signal object only considering quanta absorbed by the detector layer.
    :rtype: casymir.casymir.Signal
    """
    # Quantum Efficiency (QE) for the given energy spectrum
    QE = detector.get_QE(spectrum.energy)

    # Normalized fluence spectrum
    spec_norm = spectrum.fluence / integrate.simps(spectrum.fluence, spectrum.energy)

    # Mean quantum efficiency
    g1 = integrate.simps(QE * spec_norm, spectrum.energy)

    # Create a copy of the input signal
    signal_2 = copy.copy(signal)

    # Apply stochastic gain to the signal
    signal_2.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)), weight=1)

    return signal_2


def k_fluorescence(detector: casymir.casymir.Detector, spectrum: casymir.casymir.Spectrum,
                   sig: casymir.casymir.Signal, selection: list = None) -> casymir.casymir.Signal:
    """
    Simulate the K-fluorescence gain process for both direct and indirect conversion detectors.

    This function models K-fluorescence generation and absorption in both direct conversion (electron-hole pairs)
    and indirect conversion (optical photons in scintillators) detectors.

    :param detector: CASYMIR Detector object (either direct or indirect conversion).
    :param spectrum: CASYMIR Spectrum object.
    :param sig: CASYMIR Signal object representing the input signal.
    :param selection: Binary list [A, B, C] indicating if a given parallel path will be taken into account.

    :return: CASYMIR Signal object after applying the K-fluorescence gain process.
    """
    energy = spectrum.energy
    fluence = spectrum.fluence
    kV = spectrum.kV

    if selection is None:
        selection = [1, 1, 1]

    diffEnergy = float(energy[1] - energy[0])
    QE = detector.get_QE(energy)
    fk = detector.calculate_fk()

    # Process depends on detector type
    if detector.type == "direct":
        Ma, Mb, Mc, MaDenom, MbDenom, McDenom = [np.zeros(np.shape(energy)) for _ in range(6)]
        # Direct conversion: electron-hole pair generation
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, diff = calculate_diff(E1, detector.material)
            Ma[k] = (1 - detector.material["xi"] * w0) * (E1 / detector.material["w"])
            Mb[k] = detector.material["xi"] * w0 * (diff / detector.material["w"])
            Mc[k] = detector.material["xi"] * w0 * fk * (detector.material["ek"] / detector.material["w"])
            MaDenom[k] = QE[k] * (1 - detector.material["xi"] * w0)
            MbDenom[k] = QE[k] * detector.material["xi"] * w0
            McDenom[k] = QE[k] * detector.material["xi"] * w0 * fk

    elif detector.type == "indirect":
        Ma, Mb, Mc, Mabc, MaDenom, MbDenom, McDenom, com_e = [np.zeros(np.shape(energy)) for _ in range(8)]
        # Indirect conversion: optical photon generation and reabsorption
        com_term_z = detector.com_term_z(energy)
        for k, E1 in enumerate(np.arange(1.1, kV + diffEnergy, diffEnergy)):
            w0, diff = calculate_diff(E1, detector.material)

            com_term = com_term_z[:, k]

            Ma[k] = integrate.simps(com_term, dx=1) * (1 - detector.material["xi"] * w0) * E1 * detector.material["m0"]
            Mb[k] = integrate.simps(com_term, dx=1) * (detector.material["xi"] * w0) * detector.material["m0"] * diff
            Mc[k] = integrate.simps(com_term, dx=1) * (
                    detector.material["xi"] * w0) * detector.material["ek"] * fk * detector.material["m0"]
            com_e[k] = integrate.simps(com_term, dx=1)
            Mabc[k] = Ma[k] + Mb[k] + Mc[k]

            MaDenom[k] = QE[k] * (1 - detector.material["xi"] * w0)
            MbDenom[k] = QE[k] * detector.material["xi"] * w0
            McDenom[k] = QE[k] * detector.material["xi"] * w0 * fk

    # Calculate stochastic gains (gA, gB, gC)
    gA = integrate.simps(Ma * fluence, energy) / integrate.simps(MaDenom * fluence, energy)
    gB = integrate.simps(Mb * fluence, energy) / integrate.simps(MbDenom * fluence, energy)
    gC = integrate.simps(Mc * fluence, energy) / integrate.simps(McDenom * fluence, energy)

    # Apply stochastic gains to the signal
    wA = (1 - detector.material["xi"] * detector.material["omega"]) * selection[0]
    wB = detector.material["xi"] * detector.material["omega"] * selection[1]
    wC = detector.material["xi"] * detector.material["omega"] * selection[2]
    tk = detector.calculate_Tk()

    sig2A = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2B = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2C = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)

    # Path A: stochastic gain gA
    sig2A.stochastic_gain(gA, gA ** (1 / 2), wA)

    # Path B: stochastic gain gB
    sig2B.stochastic_gain(gB, gB ** (1 / 2), wB)

    # Path C: remote reabsorption
    sig2C.stochastic_blur(tk, wC * fk)
    sig2C.stochastic_gain(gC, gC ** (1 / 2), 1)

    sig3 = copy.copy(sig)
    sig3.signal = sig2A.signal + sig2B.signal + sig2C.signal
    sig3.wiener = sig2A.wiener + sig2B.wiener + sig2C.wiener + 2 * wB * tk * fk * gB * gC * sig.wiener

    return sig3


def calculate_diff(E1, material):
    """
    Auxiliary function to compute energy of K-fluorescence x-rays.
    """
    if E1 < material["ek"]:
        w0 = 0.0
        diff = 0.0
    else:
        w0 = material["omega"]
        diff = E1 - material["ek"]
    return w0, diff


def charge_trapping(detector: casymir.casymir.Detector, signal: casymir.casymir.Signal) -> casymir.casymir.Signal:
    """
    Apply blurring by charge collection in direct conversion detectors.

    This function simulates the blurring caused by charge trapping in the semiconductor layer. The implementation
    uses the expression presented in Zhao et al. Imaging performance of amorphous selenium based flat-panel
    detectors for digital mammography: Characterization of a small area prototype detector.
    Med Phys. 2003;30(2):254-263. doi:10.1118/1.1538233

    :param detector: CASYMIR Detector object containing the characteristics of the detector.
    :param signal: A CASYMIR Signal object representing the input signal.

    :return: A new CASYMIR Signal object after applying charge redistribution.
    :rtype: casymir.casymir.Signal
    """
    # Extract relevant parameters from the detector
    t = detector.thickness * 1e-3  # Nominal detector layer thickness [mm]
    l = detector.trapping_depth * 1e-3  # Charge collection layer distance [mm]
    # Extract frequency vector from the signal
    f = signal.freq
    # Calculate charge redistribution function (tb)
    tb = (t * np.sinh(2 * np.pi * f * (t - l))) / ((t - l) * np.sinh(2 * np.pi * f * t))
    tb[0] = 1

    signal_2 = copy.copy(signal)

    signal_2.stochastic_blur(tb, 1)

    return signal_2


def optical_blur(detector: casymir.casymir.Detector, signal: casymir.casymir.Signal) -> casymir.casymir.Signal:
    """
    This function models the stochastic blurring process caused by the spreading of optical photons
    in the scintillator material of an indirect conversion detector.

    :param detector: CASYMIR Detector object containing the characteristics of the detector.
    :param signal: A CASYMIR Signal object representing the input signal.

    :return: A new CASYMIR Signal object after simulating optical photon spreading.
    :rtype: casymir.casymir.Signal
    """
    # Extract relevant parameters from the detector
    H = detector.material["spread_coeff"]  # Spread coefficient

    # Extract frequency vector from the signal
    f = signal.freq

    # Calculate the optical spread function (osf)
    osf = 1 / (1 + H * f + H * f ** 2 + H ** 2 * f ** 3)

    # Create a copy of the input signal
    signal_2 = copy.copy(signal)

    # Apply stochastic blur using the optical spread function (osf)
    signal_2.stochastic_blur(osf, 1)

    return signal_2


def optical_coupling(detector: casymir.casymir.Detector, signal: casymir.casymir.Signal) -> casymir.casymir.Signal:
    """
    This function models the coupling of optical photons to the photodiode of a detector, where the probability
    of an incident quantum being coupled is determined by the average coupling efficiency "ce".

    :param detector: A CASYMIR Detector object containing the characteristics of the detector.
    :param signal: A CASYMIR Signal object representing the input signal.

    :return: A new CASYMIR Signal object after simulating optical coupling.
    :rtype: casymir.casymir.Signal
    """
    # Extract the coupling efficiency from the detector
    gc = detector.material["ce"]

    # Create a copy of the input signal
    signal_2 = copy.copy(signal)

    # Optical coupling is represented by a stochastic gain stage
    signal_2.stochastic_gain(gc, (gc * (1 - gc)) ** (1 / 2), 1)

    return signal_2


def px_integration(detector: casymir.casymir.Detector, signal: casymir.casymir.Signal) -> casymir.casymir.Signal:
    """
    Simulate the integration of the signal by detector pixel element, followed by sampling and pixel readout.

    This function represents the final block of the model, which includes deterministic blurring by the pixel aperture
    function (Ta), integration of the signal, resampling, and the addition of electronic noise.

    :param detector: A CASYMIR Detector object containing the characteristics of the detector.
    :param signal: A CASYMIR Signal object representing the input signal.

    :return: A new CASYMIR Signal object after simulating pixel integration, sampling and readout.
    :rtype: casymir.casymir.Signal
    """
    # Deterministic blurring by pixel aperture function Ta
    ta = np.abs(np.sinc(detector.px_size * detector.ff * signal.freq))

    # Create a copy of the input signal
    signal_2 = copy.copy(signal)

    # Apply deterministic blur using the pixel aperture function
    signal_2.deterministic_blur(ta, 1)

    # Integrated Signal and Wiener spectrum
    signal_2.signal = signal_2.signal * detector.pxa
    signal_2.wiener = signal_2.wiener * (detector.pxa ** 2)

    # MTF (Modulation Transfer Function)
    mtf = signal_2.signal / signal_2.signal[0]

    # Aliased NPS (Noise Power Spectrum) up to Nyquist frequency
    signal_2.resample()

    # Additive electronic noise
    add_noise = detector.add_noise
    nps = signal_2.wiener[0:int(signal_2.length / 2)] + ((add_noise ** 2) * detector.pxa)
    f2 = signal_2.freq[0:int(signal_2.length / 2)]
    mtf = mtf[0:int(signal_2.length / 2)]

    # Normalized NPS (Noise Power Spectrum divided by large area signal)
    nnps = nps / (signal_2.signal[0] ** 2)

    signal_2.mtf = mtf
    signal_2.nnps = nnps

    signal_2.signal = signal_2.signal[0:int(signal_2.length / 2)] + ((add_noise ** 2) * detector.pxa)
    signal_2.wiener = signal_2.wiener[0:int(signal_2.length / 2)] + ((add_noise ** 2) * detector.pxa)
    signal_2.freq = f2
    signal_2.length = int(signal_2.length / 2)

    return signal_2
