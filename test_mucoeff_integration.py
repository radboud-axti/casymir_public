import casymir.casymir
import matplotlib.pyplot as plt
import numpy as np
import xraydb as xrdb
import mucoeff

from matplotlib import pyplot as plt, rcParams
import matplotlib.font_manager as font_manager

import csv
from importlib import resources
import casymir.casymir
import casymir.processes


font_path = 'C:/Users/Z639176/AppData/Local/Microsoft/Windows/Fonts/helvetica-light-587ebe5a59211.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
# Plotting options
rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],

    # Font sizes
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,

    # Line styles
    "lines.linewidth": 2,
    "lines.markersize": 8,

    # Axes styling
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 2,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,

    # Legends
    "legend.frameon": False,

    # Grid
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

def run_1d_model(system, spec_name, kV, mAs, detector_type, mu_source = "BOONE"):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    # Define the package where detector materials are stored
    detectors_package = 'casymir.data.detectors'
    material_filename = f'{material}.yaml'

    # Check if the material file exists
    available_materials = [file_name for file_name in resources.contents(detectors_package) if file_name.endswith('.yaml')]

    if material_filename not in available_materials:
        available_materials_list = ', '.join([mat[:-5] for mat in available_materials])  # Remove '.yaml' extension
        raise FileNotFoundError(
            f"Material file '{material_filename}' not found in '{detectors_package}'.\n"
            f"Available detector materials are: {available_materials_list}\n"
            "Please ensure the material exists and the name is correct."
        )

    # If the file exists, proceed to load it
    with resources.path(detectors_package, material_filename) as yaml_file_path:
        material_path = str(yaml_file_path)

    det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
    det.mu_source = mu_source
    tube = casymir.casymir.Tube(sys.source)
    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)

    sig, _, _ = casymir.processes.initial_signal(det, spec)
    sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
    sig = casymir.processes.absorption_block(det, spec, sig)
    if detector_type == "direct":
        sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)
    else:
        sig, _, _ = casymir.processes.optical_blur(det, spec, sig)
        sig, _, _ = casymir.processes.optical_coupling(det, spec, sig)

    sig, _, _ = casymir.processes.q_integration(det, sig)
    sig = casymir.processes.noise_aliasing(det, sig)
    sig = casymir.processes.model_output(det, sig)

    print("Fit Results:\n")
    sig.fit()
    print("\n")

    results = np.array([sig.freq, sig.mtf, sig.nnps])
    results = np.transpose(results)

    return results

material_path = "casymir/data/detectors/Se.yaml"
sys = casymir.casymir.System("example_dbt_v2.yaml")
det = casymir.casymir.Detector(sys.detector, material_path, sys.detector)
# E = np.linspace(10, 50, 100)
E = np.linspace(start=1E3, stop=5E5, num=1000)

mu_xrdb_pe = xrdb.mu_elam("I", E, kind="photo")
mu_xrdb_coh = xrdb.mu_elam("I", E, kind="coh")
mu_xrdb_incoh = xrdb.mu_elam("I", E, kind="incoh")

mu_xrdb_tot = xrdb.mu_elam("I", E, kind="total")

plt.figure(figsize=(6, 7))

plt.plot(E, mu_xrdb_tot / 4.94 , label="$\\mu/\\rho$ (total)")
plt.plot(E, mu_xrdb_pe / 4.94 , label="$\\tau$ (photoelectric)", linestyle="-")
plt.plot(E, mu_xrdb_coh / 4.94 , label="$\\sigma_{coh}$ (coherent)", linestyle="--")
plt.plot(E, mu_xrdb_incoh / 4.94 , label="$\\sigma$ (Compton)", linestyle="--")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy (eV)")
plt.ylabel(r"$\mu/\rho$  (cm$^2$/g)")
# plt.xlim(E[0], 50)  # typically 1 keV … 50 MeV
plt.ylim(1e-5, 1e4)
plt.grid(True, which="major", alpha=1)

title = f"I"
plt.title(f"{title}  (Z=53)")

# Slightly smarter legend placement
legend = plt.legend(loc="best", fontsize=12, framealpha=0.5)
plt.tight_layout()
plt.show()

# XrayDB path
det.mu_source = "XRAYDB"
det.get_mu(E)
det.get_QE(E)
mu_xr = det.mu.copy()
QE_xr = det.QE.copy()


# MUCOEFF path
det.mu_source = "BOONE"
det.get_mu(E)
det.get_QE(E)
mu_boone = det.mu.copy()
QE_boone = det.QE.copy()

plt.figure()
plt.plot(E, mu_boone, label="BOONE", color="#E15A97", linewidth=3)
plt.plot(E, mu_xr, label="XRAYDB", color="#496A81", linewidth=3)
plt.xlabel("Energy [keV]")
plt.ylabel("Mass attenuation [cm$^2$/g]")
plt.title("Attenuation data for aSe")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(E, QE_boone, label="BOONE", color="#E15A97", linewidth=3)
plt.plot(E, QE_xr, label="XRAYDB", color="#496A81", linewidth=3)
plt.xlabel("Energy [keV]")
plt.ylabel("Quantum efficiency")
plt.title("Quantum efficiency for aSe")
plt.ylim([0, 1.05])
plt.legend()
plt.tight_layout()
plt.show()



results1 = run_1d_model("example_dbt_v2.yaml", "HE, XRDB", 49, 2.04, detector_type="direct", mu_source="XRAYDB")
results2 = run_1d_model("example_dbt_v2.yaml", "HE, XRDB", 49, 2.04, detector_type="direct", mu_source="BOONE")


plt.figure()
plt.plot(results2[:, 0], results2[:, 2], label="BOONE", color="#E15A97", linewidth=3)
plt.plot(results1[:, 0], results1[:, 2], label="XRAYDB", color="#496A81", linewidth=3)
plt.xlabel("Frequency [1/mm]")
plt.ylabel("NNPS [mm$^2$]")
plt.ylim([0.5E-5, 1.8E-5])
# plt.title("Attenuation data for aSe")
plt.legend()
plt.tight_layout()
plt.show()

print("end")