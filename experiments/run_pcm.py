import gc
import numpy as np
from importlib import resources
from tqdm import tqdm

import casymir.casymir
import casymir.processes
import casymir.processes_2d
import casymir.processes_3d

np.seterr(divide='ignore', invalid='ignore')

def run_pcm_dbt(
    *,
    kV: float,
    mAs: float,
    system_yaml: str,
    angles_rad: np.ndarray,
    fz: np.ndarray,
    recon_params: dict | None = None,
):
    """
    Run CASYMIR DBT PCM and return 3D system transfer and noise.

    Returns quantities suitable for task-based observer analysis.
    """

    # -----------------------------
    # Load system
    # -----------------------------
    sys = casymir.casymir.System(system_yaml)

    material = sys.detector["active_layer"]
    detector_type = sys.detector["type"]

    detectors_package = "casymir.data.detectors"
    material_filename = f"{material}.yaml"

    with resources.path(detectors_package, material_filename) as p:
        material_path = str(p)

    det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)

    t0 = det.thickness
    SID0 = tube.SID

    # -----------------------------
    # Reconstruction parameters
    # -----------------------------
    if recon_params is None:
        recon_params = {}

    Theta_rad = recon_params.get(
        "Theta_rad",
        np.max(angles_rad) + abs(np.min(angles_rad))
    )

    sa_A = recon_params.get("sa_A", 1.5)
    interp_kernel = recon_params.get("kernel", "gaussian")
    slice_B = recon_params.get("slice_B", 0.05)

    Nv = len(angles_rad)

    # -----------------------------
    # Allocate outputs
    # -----------------------------
    stack = None
    q0_per_view = np.zeros(Nv, dtype=float)
    dak_per_view = np.zeros(Nv, dtype=float)

    # -----------------------------
    # Loop over projections
    # -----------------------------
    for i, theta_i in enumerate(tqdm(angles_rad, desc="DBT projections", unit="view")):

        # angle-dependent geometry
        det.thickness = t0 / np.cos(theta_i)
        tube.SID = SID0 / np.cos(theta_i)

        # spectrum
        spec = casymir.casymir.Spectrum(
            name="dbt_spec",
            kV=kV,
            mAs=mAs,
            detector=det,
            tube=tube,
        )

        dak_per_view[i] = spec.dak

        # 1D cascaded model
        sig, _, _ = casymir.processes.initial_signal(det, spec)
        q0_per_view[i] = sig.mean_quanta

        sig, _, _ = casymir.processes.quantum_selection(det, spec, sig)
        sig = casymir.processes.absorption_block(det, spec, sig)
        sig, _, _ = casymir.processes.charge_trapping(det, spec, sig)

        # 2D
        sig2, _, _ = casymir.processes_2d.integration_2d(det, sig)
        sig2 = casymir.processes_2d.noise_aliasing_2d(det, sig2)
        sig2, _, _ = casymir.processes_2d.focal_spot_blur(sig2, 0.065, "x")
        sig2, _, _ = casymir.processes_2d.beam_obliquity_blur_2d(
            sig2, spectrum=spec, detector=det, theta_rad=theta_i
        )
        sig2, _ = casymir.processes_2d.log_transform_2d(
            sig2, spectrum=spec, a=49.99, b=13.79
        )

        sig2, _ = casymir.processes_2d.apply_ramp_filter_dbt(
            sig2, detector=det, theta_total_rad=Theta_rad, theta_i_rad=theta_i
        )
        sig2, _ = casymir.processes_2d.apply_sa_filter_dbt(
            sig2, detector=det, A=sa_A, theta_i_rad=theta_i
        )
        sig2, _ = casymir.processes_2d.apply_interpolation_filter_bilinear_dbt(
            sig2, detector=det
        )

        if stack is None:
            stack = casymir.casymir.SignalStack(
                sig2.axes[0],
                sig2.axes[1],
                Nv=Nv,
                angles_rad=angles_rad,
                dtype=np.float32,
            )

        stack.append(sig2, angle_rad=theta_i)

        del sig, sig2
        gc.collect()

    # -----------------------------
    # 2D → 3D mapping
    # -----------------------------
    px = getattr(stack, "px_size", None)

    vol3d = casymir.processes_3d.map_stack_to_volume(
        stack,
        fz,
        kernel=interp_kernel,
        B=slice_B,
        Theta_rad=Theta_rad,
        px_size_mm=px,
        spoke_density_normalize=False,
    )

    S3 = vol3d.S
    W3 = vol3d.W
    fx, fy, fz = vol3d.axes

    # -----------------------------
    # 3D MTF
    # -----------------------------
    ix0 = len(fx) // 2
    iy0 = len(fy) // 2
    iz0 = len(fz) // 2

    S0 = abs(S3[ix0, iy0, iz0]) + 1e-12
    MTF3 = abs(S3) / S0

    return {
        "MTF3": MTF3,
        "W3": W3,
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "q0_per_view": q0_per_view,
        "q0_mean": float(np.mean(q0_per_view)),
        "dak_per_view": dak_per_view,
        "dak_mean": float(np.mean(dak_per_view)),
        "angles_rad": angles_rad,
        "Theta_rad": Theta_rad,
    }
