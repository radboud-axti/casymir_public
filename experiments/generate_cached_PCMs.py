# build_pcm_cache.py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from run_pcm import run_pcm_dbt   # your function

# -------------------------------------------------
# User-defined grids
# -------------------------------------------------
KV_LE = np.arange(23, 36, 1)   # 23, 26, 29, 32, 35
KV_HE = np.arange(38, 51, 1)   # 38, 41, 44, 47, 50

mAs_LE = 4.0
mAs_HE = 2.0   # 50/50 (ish) dose split (for now)

LE_SYSTEM_YAML = Path(r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_LE.yaml")
HE_SYSTEM_YAML = Path(r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_HE.yaml")

# -------------------------------------------------
# Geometry / reconstruction (fixed for all runs)
# -------------------------------------------------
N_views = 25
angles_rad = np.linspace(
    np.deg2rad(-25), np.deg2rad(25), N_views
)

Nz = 64
dz = 1.0  # mm
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)

recon_params = dict(
    sa_A=1.5,
    kernel="gaussian",
    slice_B=0.05,
)

# -------------------------------------------------
# Output directory
# -------------------------------------------------
out_dir = Path("pcm_cache")
out_dir.mkdir(exist_ok=True)

# -------------------------------------------------
# Helper: clip to Nyquist cube
# -------------------------------------------------
def clip_to_nyquist(MTF3, W3, fx, fy, fz):
    fny_x = np.max(np.abs(fx)) / 2
    fny_y = np.max(np.abs(fy)) / 2
    fny_z = np.max(np.abs(fz)) / 2

    mx = np.abs(fx) <= fny_x
    my = np.abs(fy) <= fny_y
    mz = np.abs(fz) <= fny_z

    return (
        MTF3[np.ix_(mx, my, mz)],
        W3[np.ix_(mx, my, mz)],
        fx[mx],
        fy[my],
        fz[mz],
    )

# -------------------------------------------------
# LE cache
# -------------------------------------------------
print("=== Building LE PCM cache ===")

MTF_LE = []
W_LE   = []
meta_LE = []

for kV in tqdm(KV_LE, desc="LE kV"):
    out = run_pcm_dbt(
        kV=float(kV),
        mAs=mAs_LE,
        system_yaml=LE_SYSTEM_YAML,
        angles_rad=angles_rad,
        fz=fz,
        recon_params=recon_params,
    )

    MTFc, Wc, fx_c, fy_c, fz_c = clip_to_nyquist(
        out["MTF3"], out["W3"], out["fx"], out["fy"], out["fz"]
    )

    MTF_LE.append(MTFc)
    W_LE.append(Wc)

    meta_LE.append(dict(
        kV=kV,
        q0_mean=out["q0_mean"],
        dak_mean=out["dak_mean"],
    ))

np.savez_compressed(
    out_dir / "pcm_LE_fine.npz",
    kV=np.array(KV_LE),
    MTF=np.array(MTF_LE),
    W=np.array(W_LE),
    fx=fx_c,
    fy=fy_c,
    fz=fz_c,
    meta=np.array(meta_LE, dtype=object),
    angles_rad=angles_rad,
    mAs=mAs_LE,
    system_yaml=LE_SYSTEM_YAML,
)

# -------------------------------------------------
# HE cache
# -------------------------------------------------
print("=== Building HE PCM cache ===")

MTF_HE = []
W_HE   = []
meta_HE = []

for kV in tqdm(KV_HE, desc="HE kV"):
    out = run_pcm_dbt(
        kV=float(kV),
        mAs=mAs_HE,
        system_yaml=HE_SYSTEM_YAML,
        angles_rad=angles_rad,
        fz=fz,
        recon_params=recon_params,
    )

    MTFc, Wc, fx_c, fy_c, fz_c = clip_to_nyquist(
        out["MTF3"], out["W3"], out["fx"], out["fy"], out["fz"]
    )

    MTF_HE.append(MTFc)
    W_HE.append(Wc)

    meta_HE.append(dict(
        kV=kV,
        q0_mean=out["q0_mean"],
        dak_mean=out["dak_mean"],
    ))

np.savez_compressed(
    out_dir / "pcm_HE_fine.npz",
    kV=np.array(KV_HE),
    MTF=np.array(MTF_HE),
    W=np.array(W_HE),
    fx=fx_c,
    fy=fy_c,
    fz=fz_c,
    meta=np.array(meta_HE, dtype=object),
    angles_rad=angles_rad,
    mAs=mAs_HE,
    system_yaml=HE_SYSTEM_YAML,
)

print("✅ PCM cache generation complete")
