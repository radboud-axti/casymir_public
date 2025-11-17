import argparse
import copy
import os
import numpy as np
import csv
from importlib import resources
import casymir.casymir
import casymir.processes
import casymir.processes_2d
from matplotlib import pyplot as plt

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

# MODEL STARTS HERE
spectrum = "Example DBT Spectrum"   # Name of the spectrum
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

# angle in radians - loop starts here
theta_i = 0.0
# Set effective thickness
det.thickness = t/np.cos(theta_i)
# Create spectrum using updated, angle-dependet coordinates
spec = casymir.casymir.Spectrum(name="test_spec", kV=kV, mAs=mAs, detector=det, tube=tube)

sig, _, _ = casymir.processes.initial_signal(det, spec)
sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
sig = casymir.processes.absorption_block(det, spec, sig)
sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)

# 1D pixel integration
# sig, _, _ = casymir.processes.q_integration(det, sig)

# 2D pixel integration
sig2, _, MTF2 = casymir.processes_2d.integration_2d(det, sig)

ix0 = len(sig2.fx) // 2   # index where fx=0 (row center)
iy0 = len(sig2.fy) // 2   # index where fy=0 (col center)
FX = sig2.fx
FY = sig2.fy

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Pre-sampling Wiener Spectrum $W(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.S/np.max(sig2.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Pre-sampling MTF $S(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# NPS aliasing
sig2 = casymir.processes_2d.noise_aliasing_2d(det, sig2)

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{a}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.S/np.max(sig2.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{a}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# Focal spot blur
a1 = 0.065
sig2, H_fsb, _ = casymir.processes_2d.focal_spot_blur(sig2, a1, "x")

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{FSB}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.S/np.max(sig2.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{FSB}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# Oblique incidence - 25 degrees
sig2, H_obl, T_line = casymir.processes_2d.beam_obliquity_blur_2d(sig2, spectrum=spec, detector=det, theta_rad=0.43)

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{G}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2.S/np.max(sig2.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{G}(f_x, f_y), 25$ deg")
plt.colorbar()
plt.tight_layout()
plt.show()

# Log transform (a + b*K_air)
a = 49.99
b = 13.79
sig2, g_fac = casymir.processes_2d.log_transform_2d(sig2, spectrum=spec, a=a, b=b)

Theta = np.deg2rad(50.0)     # total angular range (example)
theta_i = np.deg2rad(0.0)    # current view angle

# Ramp filter
sig2d_h, Hra = casymir.processes_2d.apply_ramp_filter_dbt(sig2, detector=det, theta_total_rad=Theta, theta_i_rad=theta_i)

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_h.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{ra}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_h.S/np.max(sig2d_h.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{ra}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# SA filter
sig2d_sa, H_sa = casymir.processes_2d.apply_sa_filter_dbt(sig2d_h, detector=det, A=1.5, theta_i_rad=theta_i)

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_sa.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{sa}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_sa.S/np.max(sig2d_sa.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{sa}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# Bilinear filter
sig2d_in, H_in = casymir.processes_2d.apply_interpolation_filter_bilinear_dbt(sig2d_sa, detector=det)

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_in.W),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener Spectrum $W_{in}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(np.abs(sig2d_in.S/np.max(sig2d_in.S)),extent=[FY.min(), FY.max(), FX.max(), FX.min()], cmap='gray',aspect='auto')
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("MTF $S_{in}(f_x, f_y)$")
plt.colorbar()
plt.tight_layout()
plt.show()

# sanity slices at fy=0
fx_pos = sig2d_h.fx[ix0:]

plt.plot(fx_pos, Hra[ix0:, iy0], label=r"$H_{RA}, \theta_{TOT} = 50^\circ$")
plt.plot(fx_pos, H_sa[ix0:, iy0], label=r"$H_{SA}, A = 1.5$")
plt.plot(fx_pos, H_in[ix0:, iy0], label=r"$H_{IN}$")
plt.xlabel(r"$f_x$")
plt.title("Filters at $f_y = 0$")
plt.legend()
plt.tight_layout()
plt.show()

# --- 2D images with correct axes labeling (optional) ---
plt.imshow(sig2.W, extent=[sig2.fy.min(), sig2.fy.max(), sig2.fx.max(), sig2.fx.min()])
plt.xlabel(r"$f_y$ (mm$^{-1}$)")
plt.ylabel(r"$f_x$ (mm$^{-1}$)")
plt.title("Wiener (centered)")
plt.show()

# === SLICES ===
a1 = 0.065
sig2, H_fsb, _ = casymir.processes_2d.focal_spot_blur(sig2, a1, "x")

plt.imshow(H_fsb)
plt.show()
# 1) Along f_y = 0  (what you asked): vary fx, fix fy=0 ⇒ column iy0
fx_pos = sig2.fx[ix0:]
S_fy0   = sig2.S[ix0:, iy0]
W_fy0   = sig2.W[ix0:, iy0]
plt.plot(fx_pos, S_fy0 / S_fy0[0]); plt.xlim(0, fx_pos[-1]); plt.xlabel(r"$f_x$"); plt.ylabel("MTF"); plt.title("MTF | $f_y=0$")
plt.show()

# 2) Along f_x = 0 (sanity): vary fy, fix fx=0 ⇒ row ix0
fy_pos = sig2.fy[iy0:]
S_fx0   = sig2.S[ix0, iy0:]
W_fx0   = sig2.W[ix0, iy0:]
plt.plot(fy_pos, S_fx0 / S_fx0[0]); plt.xlim(0, fy_pos[-1]); plt.xlabel(r"$f_y$"); plt.ylabel("MTF"); plt.title("MTF | $f_x=0$")
plt.show()