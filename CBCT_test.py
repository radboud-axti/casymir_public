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

from matplotlib import pyplot as plt
from importlib import resources

# PLOTTING UTILS
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

def crop_to_freq_window(A, x_axis, y_axis, fmax_x, fmax_y=None):
    if fmax_y is None:
        fmax_y = fmax_x

    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)

    ix = np.where((x_axis >= -fmax_x) & (x_axis <= fmax_x))[0]
    iy = np.where((y_axis >= -fmax_y) & (y_axis <= fmax_y))[0]

    A_crop = A[np.ix_(iy, ix)] if A.shape == (len(y_axis), len(x_axis)) else A[np.ix_(ix, iy)]

    x_crop = x_axis[ix]
    y_crop = y_axis[iy]

    return A_crop, x_crop, y_crop

def show_2d_window(
    A,
    x_axis,
    y_axis,
    xlabel,
    ylabel,
    title,
    fmax_x,
    fmax_y=None,
    cmap="gray",
    aspect="equal",
    transpose=False,
):
    """
    Plot A over a physical frequency window.

    Parameters
    ----------
    A : 2D ndarray
        Data array. By default assumed shape (len(y_axis), len(x_axis)) if transpose=False.
    x_axis, y_axis : 1D ndarray
        Physical axes.
    fmax_x, fmax_y : float
        Plot limits in physical units.
    transpose : bool
        Set True if A is stored as (x, y) and should be shown with x horizontal, y vertical.
    """
    if fmax_y is None:
        fmax_y = fmax_x

    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)

    ix = np.where((x_axis >= -fmax_x) & (x_axis <= fmax_x))[0]
    iy = np.where((y_axis >= -fmax_y) & (y_axis <= fmax_y))[0]

    if transpose:
        A_plot = A[np.ix_(ix, iy)].T
    else:
        A_plot = A[np.ix_(iy, ix)]

    x_crop = x_axis[ix]
    y_crop = y_axis[iy]

    extent = [x_crop[0], x_crop[-1], y_crop[0], y_crop[-1]]

    plt.figure(figsize=(6, 6))
    plt.imshow(A_plot, extent=extent, origin="lower", cmap=cmap, aspect=aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(-fmax_x, fmax_x)
    plt.ylim(-fmax_y, fmax_y)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def inspect_sig2d(sig2d, title="", eps=1e-12):
    fx = sig2d.axes[0]
    fy = sig2d.axes[1]

    S = np.abs(sig2d.S)
    W = sig2d.W

    ix0 = len(fx) // 2
    iy0 = len(fy) // 2
    S0 = max(S[ix0, iy0], eps)

    MTF2 = S / S0
    NNPS2 = W / (S0 ** 2)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axs[0, 0].imshow(
        S.T, origin="lower",
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        aspect="auto"
    )
    axs[0, 0].set_title(f"{title} | |S|")
    axs[0, 0].set_xlabel(r"$f_x$")
    axs[0, 0].set_ylabel(r"$f_y$")
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(
        W.T, origin="lower",
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        aspect="auto"
    )
    axs[0, 1].set_title(f"{title} | W")
    axs[0, 1].set_xlabel(r"$f_x$")
    axs[0, 1].set_ylabel(r"$f_y$")
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].imshow(
        MTF2.T, origin="lower",
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        aspect="auto"
    )
    axs[1, 0].set_title(f"{title} | MTF2")
    axs[1, 0].set_xlabel(r"$f_x$")
    axs[1, 0].set_ylabel(r"$f_y$")
    plt.colorbar(im2, ax=axs[1, 0])

    im3 = axs[1, 1].imshow(
        NNPS2.T, origin="lower",
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        aspect="auto"
    )
    axs[1, 1].set_title(f"{title} | NNPS2")
    axs[1, 1].set_xlabel(r"$f_x$")
    axs[1, 1].set_ylabel(r"$f_y$")
    plt.colorbar(im3, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


def extend_freq_axis(f_axis: np.ndarray, factor: float = 2.0) -> np.ndarray:
    f_axis = np.asarray(f_axis, dtype=np.float32)
    df = float(np.mean(np.diff(f_axis)))
    fmax = max(abs(f_axis[0]), abs(f_axis[-1]))
    fmax_ext = factor * fmax
    N_ext = int(np.floor((2.0 * fmax_ext) / df)) + 1
    return np.linspace(-fmax_ext, fmax_ext, N_ext, dtype=np.float32)

# MODEL STARTS HERE
spectrum = "Example DBT Spectrum"
kV = 28
mAs = 4
system = "example_cbct.yaml"

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

# N = 50
# angles = np.linspace(np.deg2rad(-180), np.deg2rad(180), N)
# Theta = np.deg2rad(360)
#
# theta_i = angles[0]
#
# spec = casymir.casymir.Spectrum(name="test_spec", kV=kV, mAs=mAs, detector=det, tube=tube)
#
# sig, _, _ = casymir.processes.initial_signal(det, spec)
# sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
# sig = casymir.processes.absorption_block(det, spec, sig)
# sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)
#
# sig2, _, _ = casymir.processes_2d.integration_2d(det, sig)
# inspect_sig2d(sig2, "after integration_2d")
#
# sig2 = casymir.processes_2d.noise_aliasing_2d(det, sig2)
# inspect_sig2d(sig2, "after noise_aliasing_2d")
#
# sig2, _ = casymir.processes_2d.log_transform_2d(sig2, spectrum=spec, a=49.99, b=13.79)
# inspect_sig2d(sig2, "after log_transform_2d")
#
# sig2, _ = casymir.processes_2d.apply_ramp_filter_cbct(sig2)
# inspect_sig2d(sig2, "after ramp_filter_cbct")
#
# sig2, _ = casymir.processes_2d.apply_apodization_filter_cbct(sig2, detector=det)
# inspect_sig2d(sig2, "after apodization_filter_cbct")
#
# sig2, _ = casymir.processes_2d.apply_interpolation_filter_cbct(sig2, detector=det)
# inspect_sig2d(sig2, "after interpolation_filter_cbct")

# angle vector
# angles
N = 320
angles = np.linspace(np.deg2rad(-25), np.deg2rad(25), N)
Theta = np.deg2rad(50)

stack = None
q0_bar = None

for i, theta_i in enumerate(tqdm(angles, desc="DBT projections", unit="view")):
    # print(f"Projection for {theta_i:.4f} rad")

    # angle-dependent geometry
    det.thickness = t
    tube.SID      = d

    # spectrum for this view
    spec = casymir.casymir.Spectrum(name="test_spec", kV=kV, mAs=mAs, detector=det, tube=tube)

    # 1D model
    sig, _, _ = casymir.processes.initial_signal(det, spec)
    if q0_bar is None:
        q0_bar = sig.mean_quanta
    sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
    sig = casymir.processes.absorption_block(det, spec, sig)
    sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)
    # 2D model
    sig2, _, _ = casymir.processes_2d.integration_2d(det, sig)
    sig2 = casymir.processes_2d.noise_aliasing_2d(det, sig2)
    # sig2, _, _ = casymir.processes_2d.focal_spot_blur(sig2, 0.065, "x")
    # sig2, _, _ = casymir.processes_2d.beam_obliquity_blur_2d(sig2, spectrum=spec, detector=det, theta_rad=theta_i)
    sig2, _ = casymir.processes_2d.log_transform_2d(sig2, spectrum=spec, a=49.99, b=13.79)
    # Recon filters
    sig2, _ = casymir.processes_2d.apply_ramp_filter_cbct(sig2)
    sig2, _ = casymir.processes_2d.apply_apodization_filter_cbct(sig2, detector=det)
    sig2, _ = casymir.processes_2d.apply_interpolation_filter_cbct(sig2, detector=det)

    if stack is None:
        stack = casymir.casymir.SignalStack(sig2.axes[0], sig2.axes[1], Nv=N, angles_rad=angles, dtype=np.float32)

    stack.append(sig2, angle_rad=theta_i)
    del sig2, sig
    gc.collect()

Nz = 32
dz = 0.259
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
fz_ny = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
fz = extend_freq_axis(fz_ny, factor=2.0)

px = getattr(stack, "px_size", None)

vol3d = casymir.processes_3d.map_stack_to_volume_cbct(
    stack,
    fz,
    M=1.0,
    d_extent_mm=50,
    normalize_by_views=False,
    apply_asymptotic_1_over_f=False,
)

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

f_ny = 1 / (2 * dz)

show_2d_window(
    S_xy, fx, fy,
    r"$f_x$ (mm$^{-1}$)", r"$f_y$ (mm$^{-1}$)",
    "In-plane MTF magnitude ($f_z = 0$)",
    fmax_x=f_ny, fmax_y=f_ny,
    aspect="equal",
    transpose=False,
)

show_2d_window(
    W_xy, fx, fy,
    r"$f_x$ (mm$^{-1}$)", r"$f_y$ (mm$^{-1}$)",
    "In-plane Wiener spectrum ($f_z = 0$)",
    fmax_x=f_ny, fmax_y=f_ny,
    aspect="equal",
    transpose=False,
)

# In-depth
S_xz = (np.abs(S3[:, iy0, :])/np.max(np.abs(S3[:, iy0, :]))).T
W_xz = W3[:, iy0, :].T
DQE_xz = DQE3[:, iy0, :].T

show_2d_window(
    S_xz, fx, fz,
    r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
    "In-depth MTF magnitude ($f_y = 0$)",
    fmax_x=f_ny, fmax_y=f_ny,
    aspect="equal",
    transpose=False,
)

show_2d_window(
    W_xz, fx, fz,
    r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
    "In-depth Wiener spectrum ($f_y = 0$)",
    fmax_x=f_ny, fmax_y=f_ny,
    aspect="equal",
    transpose=False,
)

print("end")