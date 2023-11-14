import numpy as np
from scipy import integrate

"""
Fractional energy deposition calculated for each photon - modified for amorphous Selenium detector.
The energy deposition due to secondary processes will be neglected,except for the reabsorption of 
the K-fluorescence x-rays emitted from the detector layer following photoelectric interaction.
"""


def fk(thickness: float, mat: dict, w_hZ=1):
    """
    @param thickness: thickness [um] of detector layer
    @param mat: dictionary containing material information.
    @param w_hZ: fractional weight of the high-Z element (only applies for phosphors)
    @return: fk - fraction of K-fluorescence reabsorbed
    """
    density = mat["density"]
    PF = mat["pf"]
    xi = mat["xi"]
    omega = mat["omega"]

    theta = 0  # Incident x-ray angle. Equal to the polar angle of the photon as it exits from the phantom.

    # Mass absorption coefficients obtained from http://physics.nist.gov/PhysRefData/FFast/html/form.html

    mu_mass_abs = mat["mu_mass_abs"]  # Photoelectric mass absorption coeff in cm²/g at k_energy for detector material
    mu_mass_tot = mat["mu_mass_tot"]  # Total mass absorption coeff in cm²/g at k_energy for detector material

    L = thickness * 1e-4 * density * PF  # Phosphor coating density (g/cm²)
    n = int(L * 700)  # Number of layers
    deltaL = L / n  # L per layer
    Ix = 1  # Relative frequency of k-alpha and k-beta
    m = 2 * np.pi  # Solid angle 4*pi steradians

    low = 0.0  # Lower limit
    high = 2.0 * m  # Upper limit
    inc = 0.001 * np.pi  # Step size

    max_count = int((high - low) / inc) + 1  # To determine the dimension

    P1 = np.zeros(max_count)
    P2 = np.zeros(max_count)
    P0 = np.zeros(max_count)
    solid_angle = np.zeros(max_count)
    Layers = np.zeros(n)
    eta = np.pi / (2 * m)  # delta n
    Pix = np.zeros(n)  # Absorption probability for a K-alpha or K-beta photon originating from the ith layer

    """
    The screen is asummed to be a uniform medium of infinite area and thickness L. 
    The medium is subdivided into n layers, each of thickness deltaL. 
    First, it is calculated  the absorption probability for a K-alpha or K-beta photon originating from the ith layer, Pix.
    The solid angle 4pi subtended at the centrre of the later is divided into 2m solid-angle elements. 
    The jth solid-angle element is obtained from integration of the solid angle between the (j-1)th and jth polar angles.
    Then, assuming that fluorescence photons are emitted isotropically and that the K photon is totally absorbed,
    if an interaction occurs, the absorption probability (Pijx) for a K x-ray emitted into the solid angle is given by:
    """

    for i in range(n):

        Layers[i] = i + 1
        j = 0

        for count in range(max_count):

            solid_angle[count] = count * inc

            if j < m:

                P1[count] = 2.0 * np.pi * (np.cos((j) * eta) - np.cos((j + 1.0) * eta)) / (4.0 * np.pi)
                P2[count] = (1 - np.exp(
                    -mu_mass_tot * (n - i - 1.0 + 0.5) * deltaL / (abs(np.cos((j + 1.0 - 0.5) * eta)))))
                P0[count] = P1[count] * P2[count]

            else:

                P1[count] = 2.0 * np.pi * (np.cos(j * eta) - np.cos((j + 1.0) * eta)) / (4.0 * np.pi)
                P2[count] = (1 - np.exp(
                    -mu_mass_tot * (i + 1.0 - 0.5) * deltaL / (abs(np.cos((j + 1.0 - 0.5) * eta)))))
                P0[count] = P1[count] * P2[count]

            j = j + inc

        """
        Pix is given by the summation over the 2m solid-angle elements.
        """

        Pix[i] = integrate.simps(P0, solid_angle)
        # Pix[i] = np.trapz(P0, solid_angle)

        # PF calculation
    PF1 = np.zeros(n)
    PF2 = np.zeros(n)
    PF3 = np.zeros(n)

    """
    Then, the probability of generating a Kx fluorescencce x-ray at the ith layer, PFix, 
    for an incident photon of energy E and angle theta, is given by:

    """

    PF0 = ((w_hZ * mu_mass_abs * xi * omega * Ix) / mu_mass_tot)

    """
    Where:
    - w_hZ is the fractional weight of the high-Z element in the phosphor,
    - mu_mass_abs is the probability that an intereaction will be a photoelectric event with the high-Z element of energy E,
    - xi is the K-shell contribution to the photoelectric effect,
    - omega is the K-fluorescent yield
    - Ix is the relative frequency of K-alpha or K-beta x-ray production
    """

    for k in range(n):
        PF1[k] = (np.exp(-(mu_mass_tot * k * deltaL)) / np.cos(theta))
        PF2[k] = (np.exp(-(mu_mass_tot * (k + 1) * deltaL)) / np.cos(theta))

    """
    The total probability of Kx-reabsorption for an incident x-ray of energy E and incident angle theta is given by:
    """

    PF3 = PF0 * (PF1 - PF2)

    """
    Summation of a PFix over the n layers yields the total probability of Kx fluorescence, PFx.
    And the probability of Kx reabsorption per Kx photon emitted, fk, is given by:
    """
    # Pfinal calculation
    fk = integrate.simps(Pix * PF3) / integrate.simps(PF3)
    # fk = np.trapz(Pix * PF3) / np.trapz(PF3)

    return fk
