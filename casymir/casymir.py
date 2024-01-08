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
    def __init__(self, det_type: str, yml_mat: str):
        self.px_size = 0    # Pixel size
        self.ff = 0         # Fill factor. For direct conversion aSe, its effectively 1.
        self.pxa = None     # Pixel area
        self.thick = 200    # Nominal detector layer thickness [um]
        self.layer = 7      # Charge collection layer thickness [um]. Optimizable parameter
        self.elems = 2816   # Detector elements
        self.add_noise = 0  # Additive noise. Optimizable parameter
        self.QE = []        # Detector Quantum Efficiency

        self.mu = []            # Attenuation coefficients for given spectrum.
        self.components = []    # Active layer elemental composition
        self.type = det_type    # Detector type (direct or indirect conversion)
        self.material = {}      # Dictionary containing physical properties of the active layer material
        self.active_layer = " "     # Active layer material name
        self.extra_materials = []   # Extra materials such as detector covers

        self._load_yml(yml_file=yml_mat)

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

    def calculate_fk(self):
        # Calculates probability of K-fluorescence reabsorption (fk)
        if self.type == "direct":
            fk = fk_general.fk(self.thick, self.material)
            return fk
        else:
            fk = fk_general.fk(self.thick, self.material, w_hZ=self.components[0][1])
            return fk

    def calculate_Tk(self):
        # Calculates blurring due to K-fluorescence reabsorption in detector
        n = int(self.elems / 2)             # Number of elements within the freq range (0 to f_nyquist)
        R = np.arange(2 * n)                # Array length
        f = R / (self.px_size * (2 * n))    # Spatial frequency vector [1/mm]
        csf = tk_general.tk(self.material, n, f)
        tk = csf[:, 1] / np.max(csf[:, 1])  # Normalized result
        return tk

    def get_mu(self, energy):
        # Populates attenuation coefficient given the energy vector
        mu = np.empty((len(energy), len(self.components)))
        i = 0
        for component in self.components:
            mu[:, i] = xrdb.mu_elam(component[0], energy=energy * 1e3, kind='total')
            i += 1
        weights = [x[1] for x in self.components]
        mu_comp = np.dot(mu, weights)
        self.mu = mu_comp

    def get_QE(self, energy):
        # Calculates detector quantum efficiency
        self.get_mu(energy)
        QE = 1 - np.exp(-(self.mu * self.thick * 1e-4 * self.material["density"] * self.material["pf"]))
        self.QE = QE
        return QE

    def com_term_z(self, energy):
        # Scintillator escape efficiency
        com_term = np.zeros([self.thick, len(energy)])
        for z in range(self.thick):
            com_term[z, :] = np.exp(-1*self.mu * (self.thick - z) * 1e-4 * self.material["density"]
                                    * self.material["pf"]) * (0.19497 + (0.59363 * np.exp(-z / 310.10304))) \
                             * (1 - np.exp(-1*self.mu * 1e-4 * self.material["density"] * self.material["pf"]))

        return com_term


class Tube:
    # Relevant x-ray source parameters
    def __init__(self):
        self.target_angle = 25  # Anode angle [degrees]
        self.target = 'W'       # Anode material
        self.SID = 65           # Source to Image Distance [cm]
        self.filter = []        # List of internal filters
        self.external_filter = []   # List of external filters

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
    Contains all the information needed to run the system parallel cascaded model. The Detector and Tube objects
    are created using the System attributes.
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
    Creates/overwrites custom SpekPy materials.

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
    # Contains x-ray spectrum parameters and methods to derive relevant measurements
    def __init__(self, name, kV, mAs, detector, tube):
        self.th = tube.target_angle  # Tube angle
        self.name = name  # Spectrum name (eg HE, LE, etc)
        self.kV = kV
        self.mAs = mAs
        self.det_filtration = detector.extra_materials  # Tuple list containing material and thickness
        self.filter = tube.filter
        self.ext_filtration = tube.external_filter
        self.spec = []  # SpekPy spectrum object
        self.energy = []    # Energy vector (keV)
        self.fluence = []   # Fluence corresponding to the energy vector, per cm2 per keV
        self.tube = tube

        s = sp.Spek(kvp=self.kV, mas=self.mAs, targ=self.tube.target, x=0, y=0, z=self.tube.SID,
                            th=self.th, dk=0.1)

        _create_matls()

        s.multi_filter(filter_list=self.filter)
        s.multi_filter(filter_list=self.ext_filtration)

        # Optional consider detector cover materials:
        s.multi_filter(filter_list=self.det_filtration)

        self.spec = s
        energy, fluence = s.get_spectrum(edges=False, flu=True, diff=False, x=0, y=0, z=self.tube.SID)
        self.energy = energy
        self.fluence = fluence

    def add_filtration(self, materials):
        self.spec.multi_filter(materials)

    def get_hvl(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        hvl = self.spec.get_hvl1(x=x, y=y, z=z)
        print("\n1st HVL value at reference point: ", "{:.3f}".format(hvl), "mm")
        return hvl

    def get_k_air(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        k_air = self.spec.get_kerma(norm=False, x=x, y=y, z=z)
        print("\nAir Kerma at reference point: ", "{:.3f}".format(k_air), "uGy")
        return k_air

    def get_incident_spec(self):
        energy, spectrum = self.spec.get_spectrum(edges=False, flu=True, diff=False, x=0, y=0, z=self.tube.SID)
        return energy, spectrum

    def get_fluence(self):
        # Photons per square millimeter
        fluence = self.spec.get_flu(x=0, y=0, z=self.tube.SID) * 1e-2
        return fluence


class Signal:
    # Signal class stores frequency vector, signal and its wiener spectrum. Contains methods to add deterministic and
    # stochastic blurring and gain stages.
    def __init__(self, freq, signal, wiener):
        self.signal = signal
        self.wiener = wiener
        self.freq = freq
        self.length = np.size(freq)
        self.mtf = np.empty(np.size(freq))
        self.nnps = np.empty(np.size(freq))

    def stochastic_gain(self, mean_gain, gain_std, weight):
        signal_2 = weight * (mean_gain * self.signal)
        wiener_2 = weight * (((mean_gain ** 2) * self.wiener) + ((gain_std ** 2) * self.signal))
        self.signal = signal_2
        self.wiener = wiener_2

    def stochastic_blur(self, t, weight):
        signal_2 = weight * (t * self.signal)
        wiener_2 = weight * ((t ** 2) * self.wiener + (1 - t ** 2) * self.signal)
        self.signal = signal_2
        self.wiener = wiener_2

    def deterministic_blur(self, a, weight):
        self.signal = weight * (a * self.signal)
        self.wiener = weight * ((a ** 2) * self.wiener)

    def resample(self):
        reverse = np.flip(self.wiener)
        self.wiener = self.wiener + reverse

    def fit(self, type_mtf="poly4", type_nps="poly4"):
        def lorentzian(x, a, b, c):
            return 1 / (1 + a * x + b * (x ** 2) + (c ** 2) * (x ** 3))

        def poly4(x, a, b, c, d):
            return 1 + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)

        def poly4_2(x, m, a, b, c, d):
            return m + a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)

        # freq2 = self.freq[0:int(np.size(self.freq) / 2)]

        popt, pcov = curve_fit(poly4, self.freq, self.mtf)

        popt2, pcov2 = curve_fit(poly4_2, self.freq, self.nnps)

        print('MTF fit: a=%6.5f, b=%6.5f, c=%6.5f, d=%6.5f' % tuple(popt))
        print('NNPS fit: a=%4.3E, b=%4.3E, c=%4.3E, d=%4.3E, e=%4.3E' % tuple(popt2))



