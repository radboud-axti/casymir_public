from scipy import integrate
from scipy.optimize import curve_fit
import xraydb as xrdb
import spekpy as sp
import numpy as np
import yaml

from casymir import fk_general
from casymir import tk_general


class Detector:
    # Detector class. Contains all dimensions and material. Material is a dictionary containing the properties.
    def __init__(self, det_type: str, yml_mat: str):
        self.px_size = 0.085  # Pixel size
        self.ff = 1  # Fill factor. For direct conversion aSe, its effectively 1.
        self.pxa = (self.px_size ** 2) * self.ff  # Pixel area
        self.thick = 200  # Nominal detector layer thickness [um]
        self.layer = 7  # Charge collection layer thickness [um]. Optimizable parameter
        self.elems = 2816  # Detector elements
        self.add_noise = 0  # Additive noise. Optimizable parameter

        self.mu = []  # Attenuation coefficients for given spectrum.
        self.components = []
        self.type = det_type
        self.material = {}
        self.active_layer = " "
        self.extra_materials = []

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
        # As Tk depends on the frequency, the frequency vector is also returned
        n = int(self.elems / 2)  # N° of elements within the freq range of 0 to Nyq (detector size is 2816)
        R = np.arange(2 * n)  # length of the array
        f = R / (self.px_size * (2 * n))  # Spatial frequency cycles/mm
        csf = tk_general.tk(self.material, n, f)
        tk = csf[:, 1] / np.max(csf[:, 1])  # Normalized result
        freq = csf[:, 0]  # Frequency vector
        return freq, tk

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
        QE = 1 - np.exp(-(self.mu * self.thick * 1e-4 * self.material["density"]))
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
    # Relevant tube parameters: angle, material and source-image distance
    def __init__(self):
        self.target_angle = 25
        self.target = 'W'
        self.SID = 65
        self.filter = []
        self.external_filter = []

    def fill_from_dict(self, props: dict):
        for attr_name, value in props.items():
            setattr(self, attr_name, value)

            if attr_name == "external_filter" or attr_name == "filter":
                extra_materials_field = props[attr_name]

                mats = [(extra_materials_field[2 * i].strip('()'), float(extra_materials_field[2 * i + 1].strip('()')))
                        for i in range(int(len(extra_materials_field) / 2))]

                setattr(self, attr_name, mats)


class System:
    def __init__(self, yml_sys: str):
        self.system_id = " "
        self.description = " "
        self.detector = {}
        self.source = {}
        self.yml_dict = self._load_sys_yml(yml_file=yml_sys)
        self.system_id = self.yml_dict["system_id"]
        self.description = self.yml_dict["description"]
        self.detector = self.yml_dict["detector"]
        self.source = self.yml_dict["source"]

    def _load_sys_yml(self, yml_file: str):
        with open(yml_file, 'r') as stream:
            try:
                prop = yaml.safe_load(stream)
                return prop

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
        self.spec = []
        self.tube = tube

        s = sp.Spek(kvp=self.kV, mas=self.mAs, targ=self.tube.target, x=0, y=0, z=self.tube.SID,
                            th=self.th, dk=0.1)

        # Check and create SpekPy materials before attempting to add filtration
        self._check_matls()

        s.multi_filter(filter_list=self.filter)
        s.multi_filter(filter_list=self.ext_filtration)

        # Optional consider detector cover materials:
        s.multi_filter(filter_list=self.det_filtration)

        self.spec = s

    def add_filtration(self, materials):
        self.spec.multi_filter(materials)

    def _check_matls(self):
        all_materials = self.filter + self.ext_filtration + self.det_filtration

        unique_mats = set()
        for material, thick in all_materials:
            unique_mats.add(material)
        # TODO: update/place somewhere else for easier access
        mat_dictionary = {'Carbon Fiber': [1.9, 'C', 'Defined by Chemical Formula'],
                          'Silica': [2.196, 'Si02', 'Defined by Chemical Formula']}

        print("Checking SpekPy materials... \n")
        for elem in unique_mats:
            try:
                sp.Spek.show_matls(elem)
            except:
                name = elem
                density = mat_dictionary[name][0]
                formula = mat_dictionary[name][1]
                comment = mat_dictionary[name][2]
                sp.Spek.make_matl(matl_name=name, matl_density=density, chemical_formula=formula, matl_comment=comment)

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
    # stochastic blurring and gain stages. Other specific signal processing methods can be added later.
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

        freq2 = self.freq[0:int(np.size(self.freq) / 2)]

        popt, pcov = curve_fit(poly4, freq2, self.mtf)

        popt2, pcov2 = curve_fit(poly4_2, freq2, self.nnps)

        print('MTF fit: a=%6.5f, b=%6.5f, c=%6.5f, d=%6.5f' % tuple(popt))
        print('NNPS fit: a=%4.3E, b=%4.3E, c=%4.3E, d=%4.3E, e=%4.3E' % tuple(popt2))

    # Pending: add more robust signal processing routines.


