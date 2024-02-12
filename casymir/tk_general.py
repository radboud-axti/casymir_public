"""
Monte Carlo simulation yielding cylindrical response to an isotropic point source in an infinite homogeneous
medium with no boundaries. This program is a minimal Monte Carlo program scoring photon distributions in a cylindrical
shells.

Source code available online: http://omlc.ogi.edu/classroom/ece532/class4/ssmc/index.html

Original implementation for scintillator detectors by J.J. Pautasso, adapted by G. Pacheco
"""
import numpy as np
import scipy.special as ss
from scipy import integrate


def tk(mat: dict, n: int, f: np.ndarray, seed: int = 10) -> np.ndarray:

    """
    Calculate spread function for remotely absorbed K-fluorescence photons.

    :param mat: dictionary containing material properties
    :param n: number of frequency vector samples to consider (frequency elements below the Nyquist frequency)
    :param f: frequency vector
    :param seed: seed used for random number generator subroutine
    :return: array containing frenquency and the spread function associated with K-fluorescence reabsorption.
    """

    density = mat["density"]
    PF = mat["pf"]

    theta = 0  # Incident x-ray angle. Equal to the polar angle of the photon as it exits from the phantom.

    # Mass absorption coefficients obtained from http://physics.nist.gov/PhysRefData/FFast/html/form.html

    mu_mass_abs = mat["mu_mass_abs"]  # Photoelectric mass absorption coeff in cm²/g at k_energy for detector material
    mu_mass_total = mat["mu_mass_tot"]  # Total mass absorption coeff in cm²/g at k_energy for detector material

    mu_mass_scatter = mu_mass_total - mu_mass_abs

    mua = mu_mass_abs * density * PF  # Convert from mass absorption coeff to linear absorption coeff expressed in 1/cm
    mus = mu_mass_scatter * density * PF  # Convert from mass scatter coeff to linear scatter coeff expressed in 1/cm

    low = 0.0
    high = np.max(f)
    incr = f[1] - f[0]
    dim = int((high/incr))+1
    ms = (dim, 2)

    tk_emission = np.zeros(ms)
    result = np.zeros(ms)

    threshold = 0.01  # Used in roulette
    chance = 0.1  # Used in roulette
    one_minus_coszero = 1.0e-12
    seed_value = np.random.RandomState(seed)  # Initializes the seed for the random number generator subroutine

    radial_size = 0.5  # Total range over which bins extend in cm
    nr = 1000  # Number of bins
    dr = radial_size / nr  # Bin width assuming a CsI needle thickness of 5 um
    c_cyl = np.zeros(nr)
    r_array = np.zeros(nr)

    n_photons = 1e4  # Number of photons used in the simulation
    albedo = mus / (mus + mua)  # Proportion of the incident light that is reflected by the surface
    i_photon = 0.0

    # Photon position and trajectory

    for i_photon in np.arange(n_photons):

        # Initialize photon position and trajectory.
        i_photon = i_photon + 1.0  # Initialization of photon count
        w = 1.0  # Photon weight
        x = 0  # Set photon position to origin
        y = 0
        z = 0

        # Randomly set photon trajectory to yield an isotropic source

        randomvalue = seed_value.uniform(0, 1)
        costheta = 2.0 * randomvalue - 1.0
        sintheta = np.sqrt(1.0 - costheta * costheta)
        psi = 2.0 * np.pi * randomvalue  # azimuthal angle
        ux = sintheta * np.cos(psi)  # photon trajectory as cosines
        uy = sintheta * np.sin(psi)
        uz = costheta

        """
        Hop_Drop_Spin_Check loop
        Propagate one photon until it dies is determined by Roulette
        """

        while (1):  # while photon_status == alive
            """
            Hop 
            Take a step to new position
            """
            randomvalue = seed_value.uniform(0, 1)
            s = -(np.log(randomvalue) / (mua + mus))  # Step size.

            x = x + s * ux  # Update positions.
            y = y + s * uy  # ux,uy,uz specify the current projection trajectory
            z = z + s * uz

            """
            Drop
            Drop weight into local bin
            """
            absorb = w * (1 - albedo)  # Photon weight absorbed at this step
            w = w - absorb  # Decrement weight by amount absorbed
            r = np.sqrt(x * x + y * y)  # Current cylindrical radial position
            ir = r / dr  # ir = index to spatial bin
            if ir > nr:
                ir = nr  # last bin is for overflow

            c_cyl[int(ir)] = c_cyl[int(ir)] + absorb  # Drop absorbed weight into bin

            """
            Spin
            - Scatter photon into new trajectory defined by 'theta' and 'psi'.
            - Theta is specified by cos(theta), which is determined
            - based on the Henyey-Greenstein scattering function.
            - Convert theta and psi into cosines ux, uy, uz.
            """

            # Sample the deflection angle (theta)
            # The deflection angle (theta) describes how the photon is deflected from its current trajectory.
            # A random number (rnd) selects a choice for the value cos(theta), called costheta.
            # The value sin(theta) called sintheta is also calculated.

            randomvalue = seed_value.uniform(0, 1)
            costheta = 2.0 * randomvalue - 1.0
            sintheta = np.sqrt(1.0 - costheta * costheta)

            # Sample the azimuthal angle (psi)
            # The deflection is assumed to be directed with equal probability into any azimuthal angle (psi).
            # A random number (rnd) selects a choice for psi which is used to generate values for cos(psi) and sin(psi),
            # called cospsi and sinpsi.

            psi = 2.0 * np.pi * randomvalue
            cospsi = np.cos(psi);

            if psi < np.pi:
                sinpsi = np.sqrt(1.0 - cospsi * cospsi)
            else:
                sinpsi = -(np.sqrt(1.0 - cospsi * cospsi))

            # Calculate the new trajectory.
            # The new trajectory (uxx, uyy, uzz) is calculated based on costheta, sintheta,
            # cospsi, and sinpsi and on the current trajectory (ux, uy, uz).

            if (1.0 - np.absolute(uz)) < one_minus_coszero:

                uxx = sintheta * cospsi
                uyy = sintheta * sinpsi

                if uz > 0.0:
                    sign = 1.0
                else:
                    sign = - 1.0

                uzz = costheta * sign

            else:
                temp = np.sqrt(1.0 - uz * uz)
                uxx = sintheta * ((ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta)
                uyy = sintheta * ((uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta)
                uzz = -sintheta * cospsi * temp + uz * costheta

            # Update current trajectory
            # The current trajectory is updated, (ux, uy, uz) = (uxx, uyy, uzz).
            ux = uxx
            uy = uyy
            uz = uzz

            """   
            Check Rouelette
            If photon weight below 'threshold', then terminate photon using Roulette technique.
            Photon has 'chance' probability of having its weight increased by factor of 1/'chance',
            and 1-'chance' probability of terminating.
            """
            if w < threshold:
                if randomvalue < chance:
                    w = w / chance  # photon_status = alive - photon not yet terminated
                else:
                    break  # photon_status = dead - photon is to be terminated

            # If photon dead, then launch new photon

    # Characteristic Spread Function (CSF)

    """
    Probability that a characteristic x-ray will be reabsorbed in the screens
    at a distance between R and R + dR from the cite of its production, given that it is reabsorbed.
    """

    for ir in range(nr):
        r = (ir + 0.5) * dr
        shellvolume = 2.0 * np.pi * r * dr  # Per cm length of cylinder
        c_cyl[ir] = c_cyl[ir] / n_photons / shellvolume / mua  # Provides data in relative fluence rate (1/cm^2)
        c_cyl[ir] = c_cyl[ir] / 100  # To convert from 1/cm^2 to 1/mm^2
        r_array[ir] = r * 10  # To convert from cm to mm

    CSF = np.zeros(nr)
    finalrarray = np.zeros(nr)

    finalrarray = np.array(r_array[0:n - 1])
    CSF = np.array(c_cyl[0:n - 1])

    k = 0
    csf_array = np.zeros(dim)
    x_axis = np.zeros(dim)

    """
    The characteristic spread function describes the imaging properties of reabsorbed K X-rays in terms of spatial 
    frquency. 
    Since the CSF is rotationally symmetric, the CSF can be written as 'csf_array' by using the J0 Bessel function.
    """

    for mu in np.linspace(low, high, num=dim, endpoint=True):
        csf_array[k] = 2 * np.pi * integrate.simps(CSF * (ss.j0(2 * np.pi * mu * finalrarray)) * finalrarray,
                                                   finalrarray)
        x_axis[k] = mu
        k = k + 1

    csf_array = np.array(csf_array / np.max(csf_array))

    tk_emission[:, 0] = x_axis
    tk_emission[:, 1] = csf_array

    result[:, 0] = x_axis
    result[:, 1] = tk_emission[:, 1]

    return result
