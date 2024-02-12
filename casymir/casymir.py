"""
casymir.casymir
~~~~~~~~~~~

This module defines the main classes for the CASYMIR package.

Classes:

- Detector: Contains the characteristics and parameters of the detector.

- Tube: Contains relevant parameters for the X-ray source.

- System: Represents the overall system configuration.

- Spectrum: Manages X-ray spectrum parameters and derived measurements.

- Signal: Stores frequency vector, magnitude, and wiener spectrum for the propagated signals.
"""

from scipy.optimize import curve_fit
import xraydb as xrdb
import spekpy as sp
import numpy as np
import yaml
from importlib import resources

from casymir import fk_general
from casymir import tk_general


class Detector:
    """
    Detector class representing an X-ray imaging detector.

    Attributes:
    ----------
    - px_size (float): Pixel pitch (in mm).

    - ff (float): Fill factor. For direct conversion aSe, it's effectively 1.

    - pxa (None): Pixel area in mm^2.

    - thick (int): Nominal detector layer thickness in micrometers.

    - layer (int): Charge collection layer thickness in micrometers. This is an optimizable parameter for direct
        conversion detectors.

    - elems (int): Number of detector elements.

    - add_noise (int): Additive electronic noise, expresserd in charge quanta per unit area.

    - QE (list): Detector Quantum Efficiency.

    - mu (list): Attenuation coefficients of the detection layer for a given energy spectrum.

    - components (list): Active layer elemental composition.

    - type (str): Detector type (direct or indirect conversion).

    - material (dict): Dictionary containing physical properties of the active layer material.

    - active_layer (str): Active layer material name.

    - extra_materials (list): Extra materials such as detector covers and air gaps.

    Methods:
    ----------
    - fill_from_dict: Fills Detector object attributes from a dictionary of properties.

    - calculate_fk: Calculates probability of K-fluorescence reabsorption.

    - calculate_Tk: Calculates blurring due to K-fluorescence reabsorption in the detector.

    - get_mu: Populates detector attenuation coefficients given an energy vector.

    - get_QE: Calculates detector quantum efficiency.

    - com_term_z: Calculates scintillator escape efficiency.
    """
    def __init__(self, det_type: str, yml_mat: str, props: dict = None) -> None:
        """
        Initialize a CASYMIR Detector object.

        :param det_type: Detector type ["indirect" | "direct"]
        :param yml_mat: Path to YAML file containing the physical characteristics of the active layer material.
        :param props: Dictionary containing properties to initialize Detector attributes. Default is None.
        """
        self.px_size = 0
        self.ff = 0
        self.pxa = None
        self.thick = 200
        self.layer = 7
        self.elems = 2816
        self.add_noise = 0
        self.QE = []

        self.mu = []
        self.components = []
        self.type = det_type
        self.material = {}
        self.active_layer = " "
        self.extra_materials = []

        self._load_yml(yml_file=yml_mat)

        if props is not None:
            self.fill_from_dict(props)

    def _load_yml(self, yml_file: str):
        with open(yml_file, 'r') as stream:
            try:
                prop = yaml.safe_load(stream)   
                mat_components = prop["Components"]
                for i in range(len(mat_components)):
                    mat = mat_components[i + 1].split(",")
                    self.components.append((mat[0], float(mat[1])))

                for key in prop.keys():
                    if key != "Components":
                        self.material[key] = prop[key]

            except yaml.YAMLError as err:
                print(err)

    def fill_from_dict(self, props: dict):
        for attr_name, value in props.items():
            setattr(self, attr_name, value)

            if attr_name == "extra_materials":
                extra_materials_field = props["extra_materials"]

                mats = [(extra_materials_field[2 * i].strip('()'), float(extra_materials_field[2 * i + 1].strip('()')))
                        for i in range(int(len(extra_materials_field) / 2))]

                self.extra_materials = mats

    def calculate_fk(self) -> float:
        """
        Calculate the probability of K-fluorescence reabsorption (fk).

        :return: Probability of K-fluorescence reabsorption.
        """
        if self.type == "direct":
            fk = fk_general.fk(self.thick, self.material)
            return fk
        else:
            fk = fk_general.fk(self.thick, self.material, w_hZ=self.components[0][1])
            return fk

    def calculate_Tk(self):
        """
        Calculates blurring due to K-fluorescence reabsorption in the detector (Tk) as a function of spatial
        frequency.

        :return: Blurring due to K-fluorescence reabsorption.
        """
        n = int(self.elems / 2)             # Number of elements within the freq range (0 to f_nyquist)
        R = np.arange(2 * n)                # Array length
        f = R / (self.px_size * (2 * n))    # Spatial frequency vector [1/mm]
        csf = tk_general.tk(self.material, n, f)
        tk = csf[:, 1] / np.max(csf[:, 1])  # Normalized result
        return tk

    def get_mu(self, energy: np.ndarray) -> None:
        """
        Populate detector attenuation coefficients given the energy vector.
        Attenuation coefficients are calculated for each energy value and stored in the mu attribute.

        :param energy: Energy vector
        """
        mu = np.empty((len(energy), len(self.components)))
        i = 0
        for component in self.components:
            mu[:, i] = xrdb.mu_elam(component[0], energy=energy * 1e3, kind='total')
            i += 1
        weights = [x[1] for x in self.components]
        mu_comp = np.dot(mu, weights)
        self.mu = mu_comp

    def get_QE(self, energy: np.ndarray) -> np.ndarray:
        """
        Calculates detector quantum efficiency. The calculated QE values are also stored in the QE attribute.

        :param energy: Energy vector
        :return: Detector quantum efficiency
        """
        self.get_mu(energy)
        QE = 1 - np.exp(-(self.mu * self.thick * 1e-4 * self.material["density"] * self.material["pf"]))
        self.QE = QE
        return QE

    def com_term_z(self, energy: np.ndarray) -> np.ndarray:
        """
        Calculates scintillator escape efficiency. The calculation is based on the attenuation coefficients,
        scintillator thickness, and material properties.

        :param energy: Energy vector
        :return: Scintillator escape efficiency
        """
        com_term = np.zeros([self.thick, len(energy)])
        for z in range(self.thick):
            com_term[z, :] = np.exp(-1*self.mu * (self.thick - z) * 1e-4 * self.material["density"]
                                    * self.material["pf"]) * (0.19497 + (0.59363 * np.exp(-z / 310.10304))) \
                             * (1 - np.exp(-1*self.mu * 1e-4 * self.material["density"] * self.material["pf"]))

        return com_term


class Tube:
    """
    Tube class represents x-ray source parameters including the anode angle, anode material, source to image distance (SID),
    and internal and external filters.

    Attributes:
    ----------
    - target_angle (float): Anode angle in degrees.

    - target (str): Anode material, default is Tungsten (W).

    - SID (float): Source to Image Distance (SID) in centimeters.

    - filter (list): List of internal filters.

    - external_filter (list): List of external filters.

    Methods:
    ----------

    - fill_from_dict: Fills Tube object attributes from a dictionary of properties.
    """
    def __init__(self, props: dict = None) -> None:
        """
        Initialize a CASYMIR Tube object.

        :param props: Dictionary containing properties to initialize Tube attributes. Default is None.
        """
        self.target_angle = 25
        self.target = 'W'
        self.SID = 65
        self.filter = []
        self.external_filter = []

        if props is not None:
            self.fill_from_dict(props)

    def fill_from_dict(self, props: dict):
        for attr_name, value in props.items():
            setattr(self, attr_name, value)

            if attr_name == "external_filter" or attr_name == "filter":
                extra_materials_field = props[attr_name]

                mats = [(extra_materials_field[2 * i].strip('()'), float(extra_materials_field[2 * i + 1].strip('()')))
                        for i in range(int(len(extra_materials_field) / 2))]

                setattr(self, attr_name, mats)


class System:
    """
    Contains all the information needed to run the system parallel cascaded model from a YAML file.

    Attributes:
    ----------
    - system_id (str): System name.

    - description (str): System description.

    - detector (dict): Dictionary containing all the detector parameters. Used to create Detector object.

    - source (dict): Dictionary containing all the x-ray source parameters. Used to create Tube object.

    - yml_dict (dict): Dictionary containing all System information loaded from the YAML file.

    Methods:
    ----------

    - _load_sys_yml: Loads System information from a YAML file.

    Notes:
    -----
    - The System object is initialized with system_id, description, detector, and source attributes populated from
      a YAML file specified by the yml_sys parameter.
    - The _load_sys_yml method loads information from the YAML file and returns it as a dictionary.
    """
    def __init__(self, yml_sys: str):
        self.system_id = " "    # System name
        self.description = " "  # System description
        self.detector = {}      # Dictionary containing all the detector parameters. Used to create Detector object.
        self.source = {}        # Dictionary containing all the x-ray source parameters. Used to create Tube object.
        self.yml_dict = self._load_sys_yml(yml_file=yml_sys)    # Dictionary containing all System information
        self.system_id = self.yml_dict["system_id"]
        self.description = self.yml_dict["description"]
        self.detector = self.yml_dict["detector"]
        self.source = self.yml_dict["source"]

        self.detector["pxa"] = (self.detector["px_size"] ** 2) * self.detector["ff"]

    def _load_sys_yml(self, yml_file: str):
        with open(yml_file, 'r') as stream:
            try:
                prop = yaml.safe_load(stream)
                return prop

            except yaml.YAMLError as err:
                print(err)


def _create_matls() -> None:
    """
    Create/overwrite custom SpekPy materials.

    """
    folder_path = resources.files("casymir.data.materials")

    for material_file in resources.contents("casymir.data.materials"):
        if material_file.endswith(".yaml"):
            with folder_path.joinpath(material_file).open('r') as stream:
                try:
                    mat = yaml.safe_load(stream)
                    materials = mat.get("material", [])

                    for material_data in materials:
                        name = material_data.get("name")
                        density = material_data.get("density")
                        formula = material_data.get("formula", "")
                        comment = material_data.get("comment", "")

                        composition = material_data.get("composition", {})

                        if isinstance(composition, list):
                            composition_dict = {int(key): value for key, value in composition}
                        else:
                            composition_dict = composition

                        if composition_dict:
                            composition_list = [(element, weight) for element, weight in composition_dict.items()]
                            sp.Spek.make_matl(matl_name=name, matl_density=density, wt_matl_comp=composition_list,
                                              matl_comment=comment)
                        else:
                            sp.Spek.make_matl(matl_name=name, matl_density=density, chemical_formula=formula,
                                              matl_comment=comment)

                except yaml.YAMLError as err:
                    print(err)


class Spectrum:
    """
    A wrapper class for the Spek class from the SpekPy package.

    This class contains x-ray spectrum parameters and methods to derive relevant measurements.

    Attributes:
    ----------
    - name (str): Name of the spectrum (e.g., HE, LE, etc).

    - kV (float): Potential (kV) of the X-ray tube.

    - mAs (float): Tube current-time product (mAs) of the X-ray tube.

    - det_filtration (list): Tuple list containing material and thickness of detector filtration.

    - filter (str): Tube filter.

    - ext_filtration (list): External filtration.

    - spec (SpekPy.Spek): SpekPy Spek object.

    - energy (np.ndarray): Energy vector (keV).

    - fluence (np.ndarray): Fluence corresponding to the energy vector, per cm^2 per keV.

    - tube (casymir.Tube): CASYMIR Tube object.

    Methods:
    ----------
    - add_filtration: Adds additional filtration to the spectrum.

    - get_hvl: Calculates the half value layer (HVL) at a given reference point.

    - get_k_air: Calculates the air kerma at a given reference point.

    - get_incident_spec: Retrieves the incident spectrum at the detector entrance.

    - get_fluence: Calculates the photon fluence at the detector entrance.

    Note:
    -----
    This class serves as a wrapper around the Spek class from the SpekPy package,
    extending its functionality with additional methods specific to CASYMIR simulations the CASYMIR package.

    SpekPy reference: https://doi.org/10.1002/mp.14945
    """

    def __init__(self, name, kV, mAs, detector, tube):
        """
        Initialize CASYMIR a Spectrum object.

        ----
        - Initializes various attributes of the Spectrum object based on the provided system parameters.

        - Creates a SpekPy spectrum object using the provided parameters (kV, mAs, Detector, and Tube).

        - Applies multiple filters to the spectrum, including tube filter, external filter, and optional detector cover
          materials.

        - Retrieves the energy vector and corresponding fluence at the detector entrance.
        """

        self.th = tube.target_angle
        self.name = name
        self.kV = kV
        self.mAs = mAs
        self.det_filtration = detector.extra_materials
        self.filter = tube.filter
        self.ext_filtration = tube.external_filter
        self.spec = []
        self.energy = []
        self.fluence = []
        self.tube = tube

        # Create Spek object
        s = sp.Spek(kvp=self.kV, mas=self.mAs, targ=self.tube.target, x=0, y=0, z=self.tube.SID,
                            th=self.th, dk=0.1)
        # Create filter materials
        _create_matls()
        # Apply filters
        s.multi_filter(filter_list=self.filter)
        s.multi_filter(filter_list=self.ext_filtration)
        s.multi_filter(filter_list=self.det_filtration)
        # Spectrum at detector entrance
        self.spec = s
        # Retrieve energy and fluence
        self.energy, self.fluence = self.spec.get_spectrum(edges=False, flu=True, diff=False, x=0, y=0, z=self.tube.SID)

    def add_filtration(self, materials: list[tuple[str, float]]) -> None:
        """
        Adds filtration to the spectrum.


        :param materials: List of tuples, each containing a material and its thickness (in mm).

        Example:

        spectrum.add_filtration([('Aluminum', 2), ('Copper', 1.5)])
        """
        self.spec.multi_filter(materials)

    def get_hvl(self, coordinates: list) -> float:
        """
        Calculates the first half value layer (HVL) at a given reference point using the SpekPy `get_hvl1` method.

        :param coordinates: list containing the x, y, and z coordinates of the reference point (in cm).
        :return: The first half value layer (HVL) at the specified reference point, in mm Al.

        Example:

        hvl = spectrum.get_hvl([6, 0, 65.5])
        """
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        hvl = self.spec.get_hvl1(x=x, y=y, z=z)
        print("\n1st HVL value at reference point: ", "{:.3f}".format(hvl), "mm")
        return hvl

    def get_k_air(self, coordinates: list) -> float:
        """
        Calculates the air kerma at a given reference point using the SpekPy `get_kerma` method.

        :param coordinates: list containing the x, y, and z coordinates of the reference point (in cm).
        :return: The air kerma (in μGy) in at the specified reference point.

        Example:

        k_air = spectrum.get_k_air([6, 0, 65.5])
        """
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        k_air = self.spec.get_kerma(norm=False, x=x, y=y, z=z)
        print("\nAir Kerma at reference point: ", "{:.3f}".format(k_air), "μGy")
        return k_air

    def get_incident_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the incident spectrum at the detector entrance using the SpekPy `get_spectrum` method.

        :return: A tuple containing the energy vector (keV) and the incident spectrum at the detector entrance.
        """

        energy, spectrum = self.spec.get_spectrum(edges=False, flu=True, diff=False, x=0, y=0, z=self.tube.SID)

        return energy, spectrum

    def get_fluence(self) -> float:
        """
        Calculates the photon fluence at the detector entrance using the SpekPy `get_flu` method.

        :return: The photon fluence at the detector entrance, measured in photons per square millimeter.
        """

        fluence = self.spec.get_flu(x=0, y=0, z=self.tube.SID) * 1e-2
        return fluence


class Signal:
    """
    Signal class stores frequency vector, signal, and its Wiener spectrum.


    Contains methods to apply the generalized gain and blurring processes to a signal. The general processes are the
    building blocks of the parallel cascaded model.

    Attributes:
    ----------
    - freq (np.ndarray): Frequency vector.

    - signal (np.ndarray): Signal magnitude.

    - wiener (np.ndarray): Wiener spectrum magnitude.

    - length (int): Number of elements in the frequency vector.

    - mtf (np.ndarray): Modulation Transfer Function (MTF).

    - nnps (np.ndarray): Normalized Noise Power Spectrum (NNPS).

    Methods:
    ----------
    - stochastic_gain: Adds stochastic gain to the signal.

    - stochastic_blur: Adds stochastic blur to the signal.

    - deterministic_blur: Adds deterministic blur to the signal.

    - resample: Resamples the signal.

    - fit: Fits MTF and NNPS curves.
    """
    def __init__(self, freq: np.ndarray, signal: np.ndarray, wiener: np.ndarray):
        """
        Initialize a CASYMIR Signal object.

        :param freq: Frequency vector
        :param signal: Signal magnitude
        :param wiener: Wiener spectrum magnitude.
        """
        self.signal = signal
        self.wiener = wiener
        self.freq = freq
        self.length = np.size(freq)
        self.mtf = np.empty(np.size(freq))
        self.nnps = np.empty(np.size(freq))

    def stochastic_gain(self, mean_gain: float, gain_std: float, weight: float) -> None:
        """
        Apply stochastic gain process to a Signal object.

        :param mean_gain: mean gain associated with the process.
        :param gain_std: standard deviation of the gain.
        :param weight: weight of the process.
        """
        signal_2 = weight * (mean_gain * self.signal)
        wiener_2 = weight * (((mean_gain ** 2) * self.wiener) + ((gain_std ** 2) * self.signal))
        self.signal = signal_2
        self.wiener = wiener_2

    def stochastic_blur(self, t: np.ndarray, weight: float) -> None:
        """
        Apply stochastic blur to a Signal object.

        :param t: Spread function in frequency domain
        :param weight: weight of the process
        """
        signal_2 = weight * (t * self.signal)
        wiener_2 = weight * ((t ** 2) * self.wiener + (1 - t ** 2) * self.signal)
        self.signal = signal_2
        self.wiener = wiener_2

    def deterministic_blur(self, a: np.array, weight: float) -> None:
        """
        Apply deterministic blur (filtering) to a Signal object.

        :param a: Spread function in frequency domain.
        :param weight: weight of the process.
        """
        self.signal = weight * (a * self.signal)
        self.wiener = weight * ((a ** 2) * self.wiener)

    def resample(self):
        """
        Aliasing implementation
        """
        reverse = np.flip(self.wiener)
        self.wiener = self.wiener + reverse

    def fit(self, type_mtf: str = "poly4", type_nps: str = "poly4") -> None:
        """
        Fit curves to the MTF and NNPS.

        Currently, only fourth degree polynomial fits are supported. Alternative fit functions can be defined
        within this method.

        :param type_mtf: type of fit for the MTF.
        :param type_nps: type of fit for the NNPS.
        """
        def poly4_mtf(x, a, b, c, d):
            return 1 + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)

        def poly4(x, m, a, b, c, d):
            return m + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)

        if type_mtf == "poly4":
            popt, pcov = curve_fit(poly4_mtf, self.freq, self.mtf)
            print('MTF fit: a=%6.5f, b=%6.5f, c=%6.5f, d=%6.5f' % tuple(popt))
        else:
            print("Undefined fit function for MTF")

        if type_nps == "poly4":
            popt2, pcov2 = curve_fit(poly4, self.freq, self.nnps)
            print('NNPS fit: a=%4.3E, b=%4.3E, c=%4.3E, d=%4.3E, e=%4.3E' % tuple(popt2))

        else:
            print("Undefined fit function for NNPS")

