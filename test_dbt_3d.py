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


def nearest_idx(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))

def show_2d(
    A: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str = "",
    log: bool = False,
    aspect: str = "equal",
    cmap: str = "viridis"
):
    # extent expects [xmin, xmax, ymin, ymax]
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    Z = np.log10(A + 1e-30) if log else A

    plt.figure(figsize=(6,5))
    plt.imshow(
        Z,
        extent=extent,
        origin="lower",   # so increasing axes go up/right
        aspect=aspect,
        cmap=cmap,
    )
    cbar = plt.colorbar()
    cbar.set_label("log10" if log else "linear")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
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
Theta = np.deg2rad(50.0)

stack = None

for i, theta_i in enumerate(tqdm(angles, desc="DBT projections", unit="view")):
    # print(f"Projection for {theta_i:.4f} rad")

    # angle-dependent geometry
    det.thickness = t / np.cos(theta_i)
    tube.SID      = d / np.cos(theta_i)

    # spectrum for this view
    spec = casymir.casymir.Spectrum(name="test_spec", kV=kV, mAs=mAs, detector=det, tube=tube)

    # 1D model
    sig, _, _ = casymir.processes.initial_signal(det, spec)
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
    sig2, _ = casymir.processes_2d.apply_ramp_filter_dbt(sig2, detector=det, theta_total_rad=Theta, theta_i_rad=theta_i)
    sig2, _ = casymir.processes_2d.apply_sa_filter_dbt(sig2, detector=det, A=1.5, theta_i_rad=theta_i)
    sig2, _ = casymir.processes_2d.apply_interpolation_filter_bilinear_dbt(sig2, detector=det)

    # lazily create the stack once we know fx, fy
    if stack is None:
        stack = casymir.casymir.SignalStack(sig2.axes[0], sig2.axes[1], Nv=N, angles_rad=angles, dtype=np.float32)

    # append *unfiltered* 2D view (so we can tune recon filters later)
    stack.append(sig2, angle_rad=theta_i)
    del sig2, sig
    gc.collect()

with open("projections.pkl", "wb") as file:
    pickle.dump(stack, file)

Nz, dz = 128, 1.0   # 1 mm slice spacing
fz_axis = casymir.processes_2d.make_fz_axis(Nz, dz)

S3, W3 = casymir.processes_2d.map_stack_to_3d_minimal(stack, detector=det, theta_total_rad=Theta, fz_axis=fz_axis, B=0.05)

# pick fz≈0
iz0 = nearest_idx(fz_axis, 0.0)

# SIGNAL MTF-like magnitude, normalized to DC
S_xy = np.abs(S3[:, :, iz0])                 # shape (Nx, Ny)
# normalize w.r.t DC at (fx=0, fy=0)
ix0 = nearest_idx(stack.fx, 0.0)
iy0 = nearest_idx(stack.fy, 0.0)
S_xy_n = S_xy

show_2d(
    A=S_xy_n.T,                      # transpose so x->fx (cols), y->fy (rows)
    x_axis=stack.fx,
    y_axis=stack.fy,
    xlabel=r"$f_x$ (mm$^{-1}$)",
    ylabel=r"$f_y$ (mm$^{-1}$)",
    title="In-plane MTF magnitude (fz=0)",
    log=False, aspect="equal", cmap="gray"
)

# NPS: usually shown in log scale
W_xy = W3[:, :, iz0]
show_2d(
    A=W_xy.T,
    x_axis=stack.fx,
    y_axis=stack.fy,
    xlabel=r"$f_x$ (mm$^{-1}$)",
    ylabel=r"$f_y$ (mm$^{-1}$)",
    title="In-plane Wiener spectrum (fz=0)",
    log=False, aspect="equal", cmap="gray"
)


# pick fy≈0
iy0 = nearest_idx(stack.fy, 0.0)

# SIGNAL magnitude in (fx, fz)
S_xz = np.abs(S3[:, iy0, :])        # shape (Nx, Nz)
# normalize to DC at (fx=0, fz=0)
ix0 = nearest_idx(stack.fx, 0.0)
iz0 = nearest_idx(fz_axis, 0.0)
S_xz_n = S_xz

# transpose so columns → fx, rows → fz
show_2d(
    A=S_xz_n.T,
    x_axis=stack.fx,
    y_axis=fz_axis,
    xlabel=r"$f_x$ (mm$^{-1}$)",
    ylabel=r"$f_z$ (mm$^{-1}$)",
    title="In-depth MTF magnitude (fy=0)",
    log=False, aspect="auto", cmap="gray"
)

# NPS in log scale
W_xz = W3[:, iy0, :]
show_2d(
    A=W_xz.T,
    x_axis=stack.fx,
    y_axis=fz_axis,
    xlabel=r"$f_x$ (mm$^{-1}$)",
    ylabel=r"$f_z$ (mm$^{-1}$)",
    title="In-depth Wiener spectrum (fy=0)",
    log=False, aspect="auto", cmap="gray"
)


print("end")
