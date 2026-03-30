import numpy as np
import casymir.casymir
import casymir.processes
import casymir.processes_2d
import casymir.fitting as fitting
from tqdm import tqdm
import gc
import csv

from matplotlib import pyplot as plt
from importlib import resources

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

    fx, fy = sig2.fx, sig2.fy
    ix0 = len(fx) // 2
    iy0 = len(fy) // 2

    S0 = np.abs(sig2.S[ix0, iy0])

    sig2.mtf = np.abs(sig2.S) / (S0 + 1e-12)
    sig2.nnps = sig2.W / ((S0 + 1e-12) ** 2)


    if stack is None:
        stack = casymir.casymir.SignalStack(sig2.axes[0], sig2.axes[1], Nv=N, angles_rad=angles, dtype=np.float32)

    stack.append(sig2, angle_rad=theta_i)
    del sig2, sig
    gc.collect()

res_stack_u = fitting.fit_stack(stack, direction="u")
res_stack_v = fitting.fit_stack(stack, direction="v")

angles_deg = np.rad2deg(stack.angles)

rows = []

for i, (theta, res) in enumerate(zip(angles_deg, res_stack_u)):
    row = {
        "view": i,
        "angle_deg": theta,
    }

    # --- MTF ---
    if res.mtf.params is not None:
        a, b, c = res.mtf.params
        row.update({
            "mtf_a": a,
            "mtf_b": b,
            "mtf_c": c,
        })
    else:
        row.update({
            "mtf_a": None,
            "mtf_b": None,
            "mtf_c": None,
        })

    # --- NNPS ---
    if res.nnps.params is not None:
        a, b, c, d = res.nnps.params
        row.update({
            "nnps_a": a,
            "nnps_b": b,
            "nnps_c": c,
            "nnps_d": d,
        })
    else:
        row.update({
            "nnps_a": None,
            "nnps_b": None,
            "nnps_c": None,
            "nnps_d": None,
        })

    rows.append(row)

out_csv = "dbt_fit_parameters_u.csv"

fieldnames = [
    "view",
    "angle_deg",
    "mtf_a", "mtf_b", "mtf_c",
    "nnps_a", "nnps_b", "nnps_c", "nnps_d",
]

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved fit parameters to {out_csv}")


rows = []

for i, (theta, res) in enumerate(zip(angles_deg, res_stack_v)):
    row = {
        "view": i,
        "angle_deg": theta,
    }

    # --- MTF ---
    if res.mtf.params is not None:
        a, b, c = res.mtf.params
        row.update({
            "mtf_a": a,
            "mtf_b": b,
            "mtf_c": c,
        })
    else:
        row.update({
            "mtf_a": None,
            "mtf_b": None,
            "mtf_c": None,
        })

    # --- NNPS ---
    if res.nnps.params is not None:
        a, b, c, d = res.nnps.params
        row.update({
            "nnps_a": a,
            "nnps_b": b,
            "nnps_c": c,
            "nnps_d": d,
        })
    else:
        row.update({
            "nnps_a": None,
            "nnps_b": None,
            "nnps_c": None,
            "nnps_d": None,
        })

    rows.append(row)

out_csv = "dbt_fit_parameters_v.csv"

fieldnames = [
    "view",
    "angle_deg",
    "mtf_a", "mtf_b", "mtf_c",
    "nnps_a", "nnps_b", "nnps_c", "nnps_d",
]

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved fit parameters to {out_csv}")

print("end")