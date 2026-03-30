import argparse
import copy
import gc
import os
import numpy as np
import pickle
import csv
from importlib import resources
import casymir.casymir
import casymir.processes
import casymir.processes_2d
import casymir.processes_3d
from matplotlib import pyplot as plt
from tqdm import tqdm

from matplotlib import pyplot as plt, rcParams
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors

np.seterr(divide='ignore', invalid='ignore')

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


def nearest_idx(axis, val):
    return int(np.argmin(np.abs(axis - val)))


def show_2d(A, x_axis, y_axis, xlabel, ylabel, title, aspect="equal", cmap="gray", mode="in-plane"):
    dims = np.shape(A)
    idx_x = dims[0] // 2
    idx_y = dims[1] // 2
    plt.figure(figsize=(6,5))
    if mode == "in-plane":
        extent = [x_axis.min() / 2, x_axis.max() / 2, y_axis.min() / 2, y_axis.max() / 2]
        plt.imshow(A[idx_x - idx_x // 2: idx_x + idx_x // 2,
                     idx_y - idx_y // 2: idx_y + idx_y // 2],
                   extent=extent, origin="lower", cmap=cmap, aspect=aspect)
    else:
        extent = [x_axis.min() / 2, x_axis.max() / 2, y_axis.min(), y_axis.max()]
        plt.imshow(A[idx_x - idx_x // 2: idx_x + idx_x // 2, :],
                   extent=extent, origin="lower", cmap=cmap, aspect=aspect)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    # plt.colorbar()
    plt.tight_layout()
    plt.show()


# MODEL STARTS HERE
spectrum = "Example DBT Spectrum"
kV = 28
mAs = 4
system = "example_dbt_v2.yaml"

sys = casymir.casymir.System(system)
material = sys.detector["active_layer"]
name = sys.system_id
detector_type = sys.detector["type"]

# Define the package where detector materials are stored
detectors_package = 'casymir.data.detectors'
material_filename = f'{material}.yaml'

with resources.path(detectors_package, material_filename) as yaml_file_path:
    material_path = str(yaml_file_path)

det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
t = det.thickness
tube = casymir.casymir.Tube(sys.source)
d = tube.SID

# angle vector
# angles
N = 25
angles = np.linspace(np.deg2rad(-25), np.deg2rad(25), N)
Theta = np.deg2rad(50)

stack = None
q0_bar = None

for i, theta_i in enumerate(tqdm(angles, desc="DBT projections", unit="view")):
    # print(f"Projection for {theta_i:.4f} rad")

    # angle-dependent geometry
    det.thickness = t / np.cos(theta_i)
    tube.SID      = d / np.cos(theta_i)

    # spectrum for this view
    spec = casymir.casymir.Spectrum(name="test_spec", kV=kV, mAs=mAs, detector=det, tube=tube)

    # 1D model
    sig, _, _ = casymir.processes.initial_signal(det, spec)
    if q0_bar is None:
        q0_bar = sig.mean_quanta
    sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
    sig        = casymir.processes.absorption_block(det, spec, sig)
    sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)
    # 2D model
    sig2, _, _ = casymir.processes_2d.integration_2d(det, sig)
    sig2        = casymir.processes_2d.noise_aliasing_2d(det, sig2)
    sig2, _, _  = casymir.processes_2d.focal_spot_blur(sig2, 0.065, "x")
    sig2, _, _  = casymir.processes_2d.beam_obliquity_blur_2d(sig2, spectrum=spec, detector=det, theta_rad=theta_i)
    sig2, _     = casymir.processes_2d.log_transform_2d(sig2, spectrum=spec, a=49.99, b=13.79)
    # Recon filters
    # sig2, _ = casymir.processes_2d.apply_ramp_filter_dbt_ambr(sig2, detector=det)
    sig2, _ = casymir.processes_2d.apply_ramp_filter_dbt(sig2, detector=det, theta_total_rad=Theta, theta_i_rad=theta_i)
    sig2, _ = casymir.processes_2d.apply_sa_filter_dbt(sig2, detector=det, A=1.5, theta_i_rad=theta_i)
    sig2, _ = casymir.processes_2d.apply_interpolation_filter_bilinear_dbt(sig2, detector=det)

    if stack is None:
        stack = casymir.casymir.SignalStack(sig2.axes[0], sig2.axes[1], Nv=N, angles_rad=angles, dtype=np.float32)

    stack.append(sig2, angle_rad=theta_i)
    del sig2, sig
    gc.collect()

# with open("projections.pkl", "wb") as file:
#     pickle.dump(stack, file)

Nz = 64
dz = 1.0
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
px = getattr(stack, "px_size", None)

vol3d = casymir.processes_3d.map_stack_to_volume(stack, fz, kernel="gaussian", B=0.05, Theta_rad=np.deg2rad(50.0),
                                         px_size_mm=px, spoke_density_normalize=False)

S3 = vol3d.S
W3 = vol3d.W
fx = vol3d.axes[0]
fy = vol3d.axes[1]
fz = vol3d.axes[2]
# --- 3D MTF & NNPS (normalized, like standard CASYMIR) ---

# zero-frequency index (center of each axis)
ix0 = len(fx)//2
iy0 = len(fy)//2
iz0 = len(fz)//2

S0 = np.abs(S3[ix0, iy0, iz0])

MTF3 = np.abs(S3) / (S0 + 1e-12)
NNPS3 = W3 / ((S0 + 1e-12)**2)

N_views = S3.shape[0]
Theta_rad = Theta

# build frequency grids
FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")

# radial frequency in the scan plane (x–z)
fr = np.sqrt(FX**2 + FZ**2)

eps = 1e-20

DQE3 = (Theta_rad * fr / (N_views * (q0_bar + eps))) * (np.abs(S3)**2 / (W3 + eps))

# In-plane
S_xy = np.abs(S3[:, :, iz0])
W_xy = W3[:, :, iz0]
DQE_xy = DQE3[:, :, iz0]

show_2d(S_xy, fy, fx, r"$f_y$ (mm$^{-1}$)", r"$f_x$ (mm$^{-1}$)",
        "In-plane MTF magnitude ($f_z = 0$)")
show_2d(W_xy, fy, fx, r"$f_y$ (mm$^{-1}$)", r"$f_x$ (mm$^{-1}$)",
        "In-plane Wiener spectrum ($f_z = 0$)")
show_2d(DQE_xy, fy, fx,
        r"$f_y$ (mm$^{-1}$)", r"$f_x$ (mm$^{-1}$)",
        "In-plane DQE ($f_z = 0$)")

# In-depth
S_xz = (np.abs(S3[:, iy0, :])/np.max(np.abs(S3[:, iy0, :]))).T
W_xz = W3[:, iy0, :].T
DQE_xz = DQE3[:, iy0, :].T

show_2d(S_xz, fx, fz, r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
        "In-depth MTF magnitude ($f_y = 0$)", aspect=2.0, mode="in-depth")
show_2d(W_xz, fx, fz, r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
        "In-depth Wiener spectrum ($f_y = 0$)", aspect=2.0, mode="in-depth")
show_2d(DQE_xz, fx, fz,
        r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
        "In-depth DQE ($f_y = 0$)", aspect=2.0, mode="in-depth")

print("end")