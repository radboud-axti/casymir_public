"""
casymir.casymir
~~~~~~~~~~~

This module defines the main classes for the CASYMIR package.

Classes:

- Detector: Contains the characteristics and parameters of the detector.

- Tube: Contains relevant parameters for the X-ray source.

- System: Stores all information required to run the model. Designed to run the model from a YAML file.

- Spectrum: Manages X-ray spectrum parameters and derived measurements.

- Signal: Stores frequency vector, magnitude, and wiener spectrum for the propagated signals.
"""
from typing import Tuple, Any

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

    - thickness (int): Nominal detector trapping_depth thickness in micrometers.

    - trapping_depth (int): Depth of the charge collection layer in the semiconductor. Exclusive to direct-conversion
                            detectors

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
        self.thickness = 200
        self.trapping_depth = 7
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
            self.pxa = (self.px_size ** 2) * self.ff

        self.fk = self.calculate_fk()

    def _load_yml(self, yml_file: str):
        """
        Loads material properties from a YAML file and fills the Detector object's attributes.

        :param yml_file: Path to the YAML file containing material properties.
        """
        try:
            with open(yml_file, 'r') as stream:
                prop = yaml.safe_load(stream)

                # Parse the components and other material properties
                self._parse_components(prop.get("Components", {}))
                self._parse_material_properties(prop)

        except yaml.YAMLError as err:
            print(f"Error loading YAML file {yml_file}: {err}")

    def _parse_components(self, components_dict: dict):
        """
        Parse the components of the material from the 'Components' field in the YAML.

        :param components_dict: Dictionary of components from the YAML file.
        """
        for key, value in components_dict.items():
            element, fraction = value.split(",")
            self.components.append((element.strip(), float(fraction)))

    def _parse_material_properties(self, material_properties: dict):
        """
        Parse the material properties from the YAML file, excluding 'Components'.

        :param material_properties: Dictionary of material properties from the YAML file.
        """
        for key, value in material_properties.items():
            if key != "Components":
                self.material[key] = value

    def fill_from_dict(self, props: dict):
        for attr_name, value in props.items():
            setattr(self, attr_name, value)

    def calculate_fk(self) -> float:
        """
        Calculate the probability of K-fluorescence reabsorption (fk).

        :return: Probability of K-fluorescence reabsorption.
        """
        if self.type == "direct":
            fk = fk_general.fk(self.thickness, self.material)
            return fk
        else:
            fk = fk_general.fk(self.thickness, self.material, w_hZ=self.components[0][1])
            return fk

    def calculate_Tk(self, signal):
        """
        Calculates blurring due to K-fluorescence reabsorption in the detector (Tk) as a function of spatial
        frequency.

        :param signal: A CASYMIR signal object.
        :return: Blurring due to K-fluorescence reabsorption.
        """
        f = signal.freq
        n = int(signal.freq.size / 2)
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
            mu[:, i] = xrdb.mu_elam(component[0], energy=energy * 1e3, kind='total') \
                       - xrdb.mu_elam(component[0], energy=energy * 1e3, kind='coh')
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
        QE = 1 - np.exp(-(self.mu * self.thickness * 1e-4 * self.material["density"] * self.material["pf"]))
        self.QE = QE
        return QE

    def com_term_z(self, energy: np.ndarray) -> np.ndarray:
        """
        Calculates scintillator escape efficiency. The calculation is based on the attenuation coefficients,
        scintillator thickness, and material properties.

        :param energy: Energy vector
        :return: Scintillator escape efficiency
        """
        com_term = np.zeros([self.thickness, len(energy)])
        for z in range(self.thickness):
            com_term[z, :] = np.exp(-1 * self.mu * (self.thickness - z) * 1e-4 * self.material["density"]
                                    * self.material["pf"]) * (0.19497 + (0.59363 * np.exp(-z / 310.10304))) \
                             * (1 - np.exp(-1 * self.mu * 1e-4 * self.material["density"] * self.material["pf"]))

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
        self.system_id = " "  # System name
        self.description = " "  # System description
        self.detector = {}  # Dictionary containing all the detector parameters. Used to create Detector object.
        self.source = {}  # Dictionary containing all the x-ray source parameters. Used to create Tube object.
        self.yml_dict = self._load_sys_yml(yml_file=yml_sys)  # Dictionary containing all System information
        self.system_id = self.yml_dict["system_id"]
        self.description = self.yml_dict["description"]
        # self.detector = self.yml_dict["detector"]

        tube_data = self.yml_dict["source"]
        tube_data["filter"] = self._parse_tuple_list(tube_data.get("filter", []))
        tube_data["external_filter"] = self._parse_tuple_list(tube_data.get("external_filter", []))

        detector_data = self.yml_dict["detector"]
        detector_data["extra_materials"] = self._parse_tuple_list(detector_data.get('extra_materials', []))

        self.source = tube_data
        self.detector = detector_data

        self.detector["pxa"] = (self.detector["px_size"] ** 2) * self.detector["ff"]

    def _load_sys_yml(self, yml_file: str):
        with open(yml_file, 'r') as stream:
            try:
                prop = yaml.safe_load(stream)
                return prop

            except yaml.YAMLError as err:
                print(err)

    def _parse_tuple_list(self, list_to_parse):
        """
        Ensure that a list alternating between strings and floats is formatted correctly as a list of tuples.
        This can be used for filters, external filters, or extra materials.
        """
        return [(list_to_parse[2 * i].strip('()'), float(list_to_parse[2 * i + 1].strip('()')))
                for i in range(len(list_to_parse) // 2)]


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

    - dak (float): Air kerma at the detector cover entrance, in μGy.

    - dak_filt (float): Air kerma at the entrance of the detector sensitive volume, in μGy.

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
    extending its functionality with additional methods specific to the CASYMIR package.

    SpekPy reference: https://doi.org/10.1002/mp.14945
    """

    def __init__(self, name, kV, mAs, detector, tube, coordinates=None):
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
        self.dak = 0
        self.dak_filt = 0

        if coordinates is None:
            # The default coordinates represent a point in the detector entrance plane
            self.coordinates = (0, 0, self.tube.SID)
        else:
            self.coordinates = coordinates

        # Create Spek object
        s = sp.Spek(kvp=self.kV, mas=self.mAs, targ=self.tube.target, x=self.coordinates[0], y=self.coordinates[1],
                    z=self.coordinates[2], th=self.th, dk=0.1)
        # Create filter materials
        _create_matls()
        # Apply filters
        s.multi_filter(filter_list=self.filter)
        s.multi_filter(filter_list=self.ext_filtration)
        # Get DAK (Detector air kerma)
        self.dak = s.get_kerma()
        s.multi_filter(filter_list=self.det_filtration)
        # Get DAK taking the detector cover (if any) into account
        self.dak_filt = s.get_kerma()
        # Spectrum at detector entrance
        self.spec = s
        # Retrieve energy and fluence
        self.energy, self.fluence = self.spec.get_spectrum(edges=False, flu=True, diff=False, x=self.coordinates[0],
                                                           y=self.coordinates[1], z=self.coordinates[2])

    def add_filtration(self, materials: list[tuple[str, float]]) -> None:
        """
        Adds filtration to the spectrum.

        :param materials: List of tuples, each containing a material and its thickness (in mm).

        Example:

        spectrum.add_filtration([('Al', 2), ('Cu', 1.5)])
        """
        if not hasattr(self, "ext_filtration"):
            self.ext_filtration = []

        self.spec.multi_filter(materials)
        # Update filter list
        self.ext_filtration.extend(materials)

    def get_hvl(self, coordinates=None) -> float:
        """
        Calculates the first half-value layer (HVL) at a given reference point using the SpekPy `get_hvl1` method.

        :param coordinates: list or tuple containing the x, y, and z coordinates of the reference point (in cm).
                            If not provided, uses the coordinates set during initialization.
        :return: The first half-value layer (HVL) at the specified reference point, in mm Al.

        Example:
        hvl = spectrum.get_hvl([6, 0, 65.5])
        """
        if coordinates is None:
            x, y, z = self.coordinates
        else:
            x, y, z = coordinates

        hvl = self.spec.get_hvl1(x=x, y=y, z=z)
        return hvl

    def get_k_air(self, coordinates=None) -> float:
        """
        Calculates the air kerma at a given reference point using the SpekPy `get_kerma` method.

        :param coordinates: Optional list or tuple containing the x, y, and z coordinates (in cm).
                            If not provided, uses the coordinates set during initialization.
        :return: The air kerma (in μGy) at the specified reference point.

        Example:

        k_air = spectrum.get_k_air([6, 0, 65.5])
        """

        if coordinates is None:
            x, y, z = self.coordinates
        else:
            x, y, z = coordinates

        k_air = self.spec.get_kerma(norm=False, x=x, y=y, z=z)
        return k_air

    def get_incident_spec(self, coordinates=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the incident spectrum using the SpekPy `get_spectrum` method.

        :param coordinates: Optional list or tuple containing the x, y, and z coordinates (in cm).
                            If not provided, uses the coordinates set during initialization.

        :return: A tuple containing the energy vector (keV) and the incident spectrum at the specified reference point.
        """

        if coordinates is None:
            x, y, z = self.coordinates
        else:
            x, y, z = coordinates

        energy, spectrum = self.spec.get_spectrum(edges=False, flu=True, diff=False, x=x, y=y, z=z)
        return energy, spectrum

    def get_fluence(self, coordinates=None) -> float:
        """
        Calculates the photon fluence using the SpekPy `get_flu` method.

        :param coordinates: Optional list or tuple containing the x, y, and z coordinates (in cm).
                            If not provided, uses the coordinates set during initialization.

        :return: The photon fluence at the specified reference point, measured in photons per square millimeter per keV.
        """

        if coordinates is None:
            x, y, z = self.coordinates
        else:
            x, y, z = coordinates

        fluence = self.spec.get_flu(x=x, y=y, z=z) * 1e-2
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

    def __init__(self, freq: np.ndarray, mean_quanta: float, signal: np.ndarray, wiener: np.ndarray):
        """
        Initialize a CASYMIR Signal object.

        :param freq: Frequency vector
        :param signal: Signal magnitude
        :param wiener: Wiener spectrum magnitude.
        """
        self.mean_quanta = mean_quanta
        self.signal = signal
        self.wiener = wiener
        self.freq = freq
        self.length = np.size(freq)
        self.mtf = np.zeros(np.size(freq))
        self.nnps = np.zeros(np.size(freq))

    def stochastic_gain(self, mean_gain: float, gain_std: float) -> None:
        """
        Apply stochastic gain process to a Signal object.

        :param mean_gain: mean gain associated with the process.
        :param gain_std: standard deviation of the gain.
        """
        mean_quanta_2 = mean_gain * self.mean_quanta
        signal_2 = mean_gain * self.signal
        wiener_2 = ((mean_gain ** 2) * self.wiener) + ((gain_std ** 2) * self.mean_quanta)
        self.mean_quanta = mean_quanta_2
        self.signal = signal_2
        self.wiener = wiener_2

    def stochastic_blur(self, t: np.ndarray) -> None:
        """
        Apply stochastic blur to a Signal object.

        :param t: Spread function in frequency domain
        """
        signal_2 = t * self.signal
        wiener_2 = ((t ** 2) * self.wiener + (1 - t ** 2) * self.mean_quanta)
        self.signal = signal_2
        self.wiener = wiener_2

    def deterministic_blur(self, a: np.array) -> None:
        """
        Apply deterministic blur (filtering) to a Signal object.

        :param a: Spread function in frequency domain.
        """
        mean_quanta_2 = a[0] * self.mean_quanta
        signal_2 = a * self.signal
        wiener_2 = ((a ** 2) * self.wiener)
        self.mean_quanta = mean_quanta_2
        self.signal = signal_2
        self.wiener = wiener_2

    def fit(self, type_mtf: str = "lorentzian", type_nps: str = "gaussian") -> Tuple[Any, Any]:
        """
        Fit curves to the MTF and NPS.

        Supported fit functions:
        - MTF:
            "lorentzian": A double Lorentzian function:
                MTF(f) = A / (1 + (f/B)^2) + (1-A) / (1 + (f/C)^2)
            "poly4": A fourth-degree polynomial:
                MTF(f) = 1 + A*f + B*f^2 + C*f^3 + D*f^4
        - NPS:
            "gaussian": A double Gaussian function:
                NPS(f) = A * exp(-(f/B)^2) + C * exp(-(f/D)^2)
            "poly4": A fourth-degree polynomial:
                NPS(f) = M + A*f + B*f^2 + C*f^3 + D*f^4

        :param type_mtf: Type of fit for the MTF. Options: "lorentzian", "poly4".
        :param type_nps: Type of fit for the NPS. Options: "gaussian", "poly4".
        :return: Tuple containing the coefficients of the MTF and NPS fits. Returns None if an unsupported function
                 is passed
        """
        # Fit functions.
        fit_functions = {
            "gaussian": lambda x, a, b, c, d: a * np.exp(- (x / b) ** 2) + c * np.exp(- (x / d) ** 2),
            "poly4_mtf": lambda x, a, b, c, d: 1 + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4),
            "lorentzian": lambda x, a, b, c: a / (1 + (x / b) ** 2) + (1 - a) / (1 + (x / c) ** 2),
            "poly4": lambda x, m, a, b, c, d: m + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)
        }
        # Fit initalization. This is needed for the NPS Gaussian fit.
        p0 = {"gaussian": [np.mean(self.nnps), 10, np.mean(self.nnps), 10],
              "poly4_mtf": None, "lorentzian": None, "poly4": None}

        def generic_fit(fit_type: str, func_name: str, x_data, y_data):
            if func_name not in fit_functions:
                print(f"Undefined fit function: {fit_type}")
                return None
            try:
                popt, _ = curve_fit(fit_functions[func_name], x_data, y_data, p0=p0[func_name])
                return popt
            except Exception as e:
                print(f"Error fitting {fit_type}: {e}")
                return None

        # Fit MTF
        mtf_func_name = "lorentzian" if type_mtf == "lorentzian" else "poly4_mtf"
        popt_mtf = generic_fit("MTF", mtf_func_name, self.freq, self.mtf)

        if popt_mtf is not None:
            print(f"MTF {type_mtf} fit parameters: " +
                  ", ".join(f"c{i + 1}={p:.5f}" for i, p in enumerate(popt_mtf)))

        # Fit NPS
        nps_func_name = "gaussian" if type_nps == "gaussian" else "poly4"
        popt_nps = generic_fit("NPS", nps_func_name, self.freq, self.nnps)

        if popt_nps is not None:
            print(f"NPS {type_nps} fit parameters: " +
                  ", ".join(f"c{i + 1}={p:.5E}" for i, p in enumerate(popt_nps)))

        return popt_mtf, popt_nps
