# tasks/utils/attenuation.py

import numpy as np
import xraydb as xrdb

try:
    import mucoeff
    _HAS_MUCOEFF = True
except ImportError:
    _HAS_MUCOEFF = False


def linear_mu_spectrum(
    energy_keV: np.ndarray,
    composition: list[tuple[str, float]],
    density_g_cm3: float,
    *,
    mu_source: str = "BOONE",
) -> np.ndarray:
    """
    Compute linear attenuation coefficient μ(E) [mm^-1]
    for an arbitrary material.

    Parameters
    ----------
    energy_keV : ndarray
        Energy vector in keV.
    composition : list of (element, fraction)
        Mass fractions, must sum to 1.
        Example: [('I', 0.5), ('H', 0.5)]
    density_g_cm3 : float
        Material density in g/cm^3.
    mu_source : {'BOONE', 'XRAYDB'}

    Returns
    -------
    mu_linear_mm : ndarray
        Linear attenuation coefficient μ(E) in mm^-1.
    """

    energy_keV = np.asarray(energy_keV, dtype=float)
    nE = energy_keV.size

    mu_mass = np.zeros(nE, dtype=float)

    if mu_source.upper() == "BOONE":
        if not _HAS_MUCOEFF:
            raise ImportError("mucoeff not available")

        for element, weight in composition:
            Z = xrdb.atomic_number(element)
            mu_i = mucoeff.mu(16, Z, energy_keV)  # cm^2 / g
            mu_mass += weight * mu_i

    elif mu_source.upper() == "XRAYDB":
        for element, weight in composition:
            mu_i = (
                xrdb.mu_elam(element, energy=energy_keV * 1e3, kind="total")
                - xrdb.mu_elam(element, energy=energy_keV * 1e3, kind="coh")
            )  # cm^2 / g
            mu_mass += weight * mu_i
    else:
        raise ValueError(f"Unknown mu_source '{mu_source}'")

    # Convert: (cm^2/g * g/cm^3) → cm^-1 → mm^-1
    mu_linear_mm = mu_mass * density_g_cm3 / 10.0

    return mu_linear_mm


def spectral_average_mu(
    energy_keV: np.ndarray,
    fluence: np.ndarray,
    composition: list[tuple[str, float]],
    density_g_cm3: float,
    *,
    mu_source: str = "BOONE",
) -> float:
    """
    Compute spectrally weighted linear attenuation coefficient <μ> [mm^-1].

    Parameters
    ----------
    energy_keV : ndarray
        Energy vector in keV.
    fluence : ndarray
        Photon fluence spectrum (any consistent units).
    composition : list of (element, fraction)
    density_g_cm3 : float

    Returns
    -------
    mu_eff : float
        Spectrally averaged μ in mm^-1.
    """

    mu_E = linear_mu_spectrum(
        energy_keV,
        composition,
        density_g_cm3,
        mu_source=mu_source,
    )

    fluence = np.asarray(fluence, dtype=float)

    num = np.trapz(mu_E * fluence, energy_keV)
    den = np.trapz(fluence, energy_keV)

    if den <= 0:
        raise ValueError("Fluence integral is zero")

    return num / den


def spectral_mu_from_spectrum(
    spectrum,
    composition: list[tuple[str, float]],
    density_g_cm3: float,
    *,
    mu_source: str = "BOONE",
) -> float:
    """
    Compute <μ> for a CASYMIR Spectrum object.
    """
    return spectral_average_mu(
        spectrum.energy,
        spectrum.fluence,
        composition,
        density_g_cm3,
        mu_source=mu_source,
    )

def mu_eff_polychromatic(
    energy_keV: np.ndarray,
    fluence: np.ndarray,
    mu_E: np.ndarray,
    thickness_mm: float,
) -> float:
    """
    Compute effective linear attenuation coefficient μ_eff [mm^-1]
    for a polychromatic x-ray spectrum in the log-transmission domain.

    Implements:
        μ_eff = -(1/L) * log( ∫ψ(E) exp(-μ(E)L) dE / ∫ψ(E) dE )

    Parameters
    ----------
    energy_keV : ndarray
        Energy vector in keV.
    fluence : ndarray
        Photon fluence spectrum ψ(E) (photons / keV).
    mu_E : ndarray
        Linear attenuation coefficient μ(E) in mm^-1.
    thickness_mm : float
        Material thickness L in mm.

    Returns
    -------
    mu_eff : float
        Effective linear attenuation coefficient in mm^-1.
    """

    energy_keV = np.asarray(energy_keV, dtype=float)
    fluence    = np.asarray(fluence,    dtype=float)
    mu_E       = np.asarray(mu_E,       dtype=float)

    if thickness_mm <= 0:
        raise ValueError("thickness_mm must be > 0")

    # Transmission-weighted fluence
    trans = np.exp(-mu_E * thickness_mm)

    num = np.trapz(fluence * trans, energy_keV)
    den = np.trapz(fluence, energy_keV)

    if num <= 0 or den <= 0:
        raise ValueError("Invalid spectrum or attenuation: transmission integral <= 0")

    mu_eff = -np.log(num / den) / thickness_mm
    return mu_eff