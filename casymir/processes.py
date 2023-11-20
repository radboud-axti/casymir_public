# Functions related to the different processes:

import casymir.casymir
import numpy as np
from scipy import integrate


def parallel_gains_direct(detector, energy, spectrum, fk, kV, selection=None):
    if selection is None:
        selection = [1, 1, 1]

    kbi = 1.1
    shape = np.shape(energy)
    Ma, Mb, Mc, Mabc, MaDenom, MbDenom, McDenom, com_e = [np.zeros(shape) for _ in range(8)]

    diffEnergy = float(energy[1] - energy[0])
    QE = detector.get_QE(energy)

    for k, E1 in enumerate(np.arange(kbi, kV + diffEnergy, diffEnergy)):
        w0, diff = calculate_diff(E1, detector.material)

        Ma[k] = (1 - detector.material["xi"] * w0) * (E1 / detector.material["w"])
        Mb[k] = (detector.material["xi"] * w0) * (diff / detector.material["w"])
        Mc[k] = (detector.material["xi"] * w0 * fk) * (detector.material["ek"] / detector.material["w"])
        Mabc[k] = Ma[k] + Mb[k] + Mc[k]

        MaDenom[k] = QE[k] * (1 - detector.material["xi"] * w0)
        MbDenom[k] = QE[k] * detector.material["xi"] * w0
        McDenom[k] = QE[k] * detector.material["xi"] * w0 * fk

    Mean_Ma = integrate.simps(Ma * spectrum, energy) / integrate.simps(MaDenom * spectrum, energy)
    Mean_Mb = integrate.simps(Mb * spectrum, energy) / integrate.simps(MbDenom * spectrum, energy)
    Mean_Mc = integrate.simps(Mc * spectrum, energy) / integrate.simps(McDenom * spectrum, energy)

    wA = (1 - (detector.material["xi"] * detector.material["omega"]))*selection[0]
    wB = (detector.material["xi"] * detector.material["omega"])*selection[1]
    wC = (detector.material["xi"] * detector.material["omega"])*selection[2]

    return Mean_Ma, Mean_Mb, Mean_Mc, wA, wB, wC


def parallel_gains_indirect(detector, energy, spectrum, fk, kV, selection=None):
    if selection is None:
        selection = [1, 1, 1]

    kbi = 1.1
    shape = np.shape(energy)
    Ma, Mb, Mc, Mabc, MaDenom, MbDenom, McDenom, com_e = [np.zeros(shape) for _ in range(8)]

    diffEnergy = float(energy[1] - energy[0])

    com_term_z = detector.com_term_z(energy)
    QE = detector.get_QE(energy)

    for k, E1 in enumerate(np.arange(kbi, kV + diffEnergy, diffEnergy)):
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

    Mean_Ma = integrate.simps(Ma * spectrum, energy) / integrate.simps(MaDenom * spectrum, energy)
    Mean_Mb = integrate.simps(Mb * spectrum, energy) / integrate.simps(MbDenom * spectrum, energy)
    Mean_Mc = integrate.simps(Mc * spectrum, energy) / integrate.simps(McDenom * spectrum, energy)

    wA = (1 - (detector.material["xi"] * detector.material["omega"]))*selection[0]
    wB = (detector.material["xi"] * detector.material["omega"])*selection[1]
    wC = (detector.material["xi"] * detector.material["omega"])*selection[2]

    return Mean_Ma, Mean_Mb, Mean_Mc, wA, wB, wC


def calculate_diff(E1, material):
    if E1 < material["ek"]:
        w0 = 0.0
        diff = 0.0
    else:
        w0 = material["omega"]
        diff = E1 - material["ek"]
    return w0, diff


def spread_direct(detector: casymir.casymir.Detector, signal: casymir.casymir.Signal) -> np.array:
    """
    Charge redistribution for direct conversion detector.
    :param detector: CASYMIR Detector object.
    :param signal: CASYMIR Signal object.
    :return: Array containing spread function due to charge redistribution
    """
    t = detector.thick
    l = detector.layer
    f = signal.freq
    tb = (t * np.sinh(2 * np.pi * f * (t - l) * 1e-3)) / ((t - l) * np.sinh(2 * np.pi * f * t * 1e-3))
    tb[0] = 1

    return tb


def spread_indirect(detector, signal):
    f = signal.freq
    H = detector.material["spread_coeff"]
    osf = 1 / (1 + H * f + H * f ** 2 + H ** 2 * f ** 3)

    return osf
