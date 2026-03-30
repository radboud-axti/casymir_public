import numpy as np
import casymir.casymir
import casymir.processes
import casymir.processes_2d
import casymir.fitting as fitting

from matplotlib import pyplot as plt
from importlib import resources

# --- System setup ---
system = "example_dbt_v2.yaml"
sys = casymir.casymir.System(system)

material = sys.detector["active_layer"]
detector_type = sys.detector["type"]
name = sys.system_id

detectors_package = 'casymir.data.detectors'
material_filename = f'{material}.yaml'

with resources.path(detectors_package, material_filename) as yaml_file_path:
    material_path = str(yaml_file_path)

det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
tube = casymir.casymir.Tube(sys.source)

# single projection angle
theta = np.deg2rad(25.0)

# spectrum
spec = casymir.casymir.Spectrum(
    name="fit_test",
    kV=28,
    mAs=4,
    detector=det,
    tube=tube,
)

# --- 1D ---
sig, _, _ = casymir.processes.initial_signal(det, spec)
sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
sig        = casymir.processes.absorption_block(det, spec, sig)
sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)

# --- 2D ---
sig2, _, _ = casymir.processes_2d.integration_2d(det, sig)
sig2        = casymir.processes_2d.noise_aliasing_2d(det, sig2)
sig2, _, _  = casymir.processes_2d.focal_spot_blur(sig2, 0.065, "x")
sig2, _, _  = casymir.processes_2d.beam_obliquity_blur_2d(
    sig2, spectrum=spec, detector=det, theta_rad=theta
)
sig2, _ = casymir.processes_2d.log_transform_2d(
    sig2, spectrum=spec, a=49.99, b=13.79
)

fx, fy = sig2.fx, sig2.fy
ix0 = len(fx) // 2
iy0 = len(fy) // 2

S0 = np.abs(sig2.S[ix0, iy0])

sig2.mtf = np.abs(sig2.S) / (S0 + 1e-12)
sig2.nnps = sig2.W / ((S0 + 1e-12) ** 2)

res_u = fitting.fit_nd(sig2, direction="u")
res_v = fitting.fit_nd(sig2, direction="v")


def plot_fit(res, title_prefix=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --- MTF ---
    ax[0].plot(res.mtf.x, res.mtf.y, "k.", label="MTF data")
    if res.mtf.y_fit is not None:
        ax[0].plot(res.mtf.x, res.mtf.y_fit, "r-", label="Lorentzian fit")
    ax[0].set_xlabel("Spatial frequency (mm$^{-1}$)")
    ax[0].set_ylabel("MTF")
    ax[0].set_title(f"{title_prefix} MTF ({res.mtf.direction})")
    ax[0].legend()
    ax[0].grid(True)

    # --- NNPS ---
    ax[1].plot(res.nnps.x, res.nnps.y, "k.", label="NNPS data")
    if res.nnps.y_fit is not None:
        ax[1].plot(res.nnps.x, res.nnps.y_fit, "r-", label="Gaussian fit")
    ax[1].set_xlabel("Spatial frequency (mm$^{-1}$)")
    ax[1].set_ylabel("NNPS")
    ax[1].set_title(f"{title_prefix} NNPS ({res.nnps.direction})")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


plot_fit(res_u, title_prefix=" ")
plot_fit(res_v, title_prefix=" ")

print("end")