import matplotlib.pyplot as plt
import numpy as np
from experiments.run_pcm import run_pcm_dbt

def clip_to_nyquist(MTF3, W3, fx, fy, fz):
    """
    Clip 3D frequency-domain quantities to the base Nyquist cube.
    Assumes inputs span approximately [-2 f_NY, +2 f_NY].
    """

    fny_x = np.max(np.abs(fx)) / 2
    fny_y = np.max(np.abs(fy)) / 2
    fny_z = np.max(np.abs(fz)) / 2

    ix = np.where(np.abs(fx) <= fny_x)[0]
    iy = np.where(np.abs(fy) <= fny_y)[0]
    iz = np.where(np.abs(fz) <= fny_z)[0]

    MTFc = MTF3[np.ix_(ix, iy, iz)]
    Wc   = W3[np.ix_(ix, iy, iz)]

    fx_c = fx[ix]
    fy_c = fy[iy]
    fz_c = fz[iz]

    return MTFc, Wc, fx_c, fy_c, fz_c


def dprime2_ideal_quantum(MTF3, W3, FX, FY, FZ, sigma_mm, C):
    """
    Quantum-limited ideal observer detectability for a Gaussian task.
    """

    # Gaussian task spectrum (Hu-style)
    W_task = C * np.exp(
        -np.pi * sigma_mm**2 * (FX**2 + FY**2 + FZ**2)
    )

    dfx = abs(FX[1,0,0] - FX[0,0,0])
    dfy = abs(FY[0,1,0] - FY[0,0,0])
    dfz = abs(FZ[0,0,1] - FZ[0,0,0])

    eps = 1e-30
    integrand = (MTF3**2 * W_task**2) / (W3 + eps)

    return np.sum(integrand) * dfx * dfy * dfz


# -----------------------------
# Geometry
# -----------------------------
N_views = 25
angles = np.linspace(np.deg2rad(-25), np.deg2rad(25), N_views)

Nz = 32
dz = 1.0  # mm
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)

# -----------------------------
# Acquisition
# -----------------------------
kV  = 28
mAs = 4
system_path = "C:\\Users\\Z639176\\Documents\\Projects\\casymir_v2\\casymir_public\\example_dbt_v2.yaml"
# -----------------------------
# Run PCM
# -----------------------------
out = run_pcm_dbt(
    kV=kV,
    mAs=mAs,
    system_yaml=system_path,
    angles_rad=angles,
    fz=fz,
)

MTF3 = out["MTF3"]
W3   = out["W3"]
fx   = out["fx"]
fy   = out["fy"]
fz   = out["fz"]

print("Mean q0:", out["q0_mean"])
print("Mean DAK [µGy]:", out["dak_mean"])

# -----------------------------
# Clip to Nyquist
# -----------------------------
MTF3c, W3c, fx_c, fy_c, fz_c = clip_to_nyquist(
    MTF3, W3, fx, fy, fz
)

FX, FY, FZ = np.meshgrid(fx_c, fy_c, fz_c, indexing="ij")

# -----------------------------
# Iodine detectability test
# -----------------------------
sigma_mm = 1.0  # lesion size parameter (adjust later)

d2 = dprime2_ideal_quantum(
    MTF3c,
    W3c,
    FX, FY, FZ,
    sigma_mm=sigma_mm,
    C = 0.05
)

print(f"d'^2 (iodine, quantum-limited) = {d2:.4e}")
print(f"d'  (iodine, quantum-limited) = {np.sqrt(d2):.3f}")

print("end")