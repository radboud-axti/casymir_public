"""
optimize_cyst_mass_3D.py

Compute 3D ideal-observer d' for cyst-vs-mass discrimination using:
- cached CASYMIR PCM outputs (already clipped to Nyquist)
- DE subtraction: I_DE = I_HE - w * I_LE
- 3D Gaussian task
- quantum + anatomical (power-law) noise

Outputs:
- dprime[LE, HE, w]
- CDE[LE, HE, w]
- kappa[LE, HE, w]
and metadata grids, saved to NPZ.

Assumptions (as per our running plan):
- MTF_LE = MTF_HE (use HE MTF3 for the system transfer in the observer)
- cached PCM W3 is the quantum NPS in log-recon domain
- cyst/mass attenuation expressed in (PMMA, Al) basis:
    mu_c(E) = 0.812 * mu_PMMA(E) + 0.0323 * mu_Al(E)
    mu_m(E) = 0.859 * mu_PMMA(E) + 0.0296 * mu_Al(E)
- breast composition for anatomical noise: 50/50 adipose + glandular (Hammerstein)
- anatomical noise: S_B = kappa / f^beta with beta=3
- kappa = 0.0079 * (mu_DE_fg - mu_DE_ad)^2  (Hu-style scaling)

You will likely only need to adjust:
- CACHE_DIR_LE / CACHE_DIR_HE and filename patterns in load_pcm_npz()
- LE_kVs / HE_kVs / w_vals
- YAML paths for spectra generation (LE/HE)
"""

from __future__ import annotations
import os
import glob
import numpy as np

# --- use your uploaded helper modules ---
import utils.attenuation as at
import tasks.task_models as tm
import tasks.task_observers as tobs
import tasks.backgrounds as bg

# If CASYMIR is in your environment:
import casymir.casymir
from importlib import resources


LE_kVs = np.arange(23, 36, 3)
HE_kVs = np.arange(38, 51, 3)

# DE weight grid
w_vals = np.linspace(0.1, 1.0, 10)

LE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_LE.yaml"
HE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_HE.yaml"
mAs_LE = 4.0
mAs_HE = 2.0

CACHE_DIR_LE = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\experiments\pcm_cache\pcm_LE.npz"
CACHE_DIR_HE = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\experiments\pcm_cache\pcm_HE.npz"

# Lesion/task settings
t_lesion_mm = 10.0
sigma_mm    = 10.0

# Anatomical noise settings
beta = 3.0
kappa_scale = 0.0079

# Background thickness for w_sub / mu_eff tissue estimation
T_breast_mm = 40.0

L_ref_lesion_mm = 10.0

# Output file
OUT_NPZ = "opt_cyst_mass_dprime_3D.npz"


# Hammerstein adipose/glandular (mass fractions)
glandular_comp = [
    ("H",  0.102),
    ("C",  0.184),
    ("N",  0.032),
    ("O",  0.677),
    ("P",  0.00125),
    ("S",  0.00125),
    ("K",  0.00125),
    ("Ca", 0.00125),
]
rho_gland = 1.04

adipose_comp = [
    ("H",  0.112),
    ("C",  0.619),
    ("N",  0.017),
    ("O",  0.251),
    ("P",  0.00025),
    ("S",  0.00025),
    ("K",  0.00025),
    ("Ca", 0.00025),
]
rho_adipose = 0.93

# PMMA (approx by mass fractions) + density
# (common approx; you can swap to your lab standard if different)
pmma_comp = [("H", 0.0805), ("C", 0.5998), ("O", 0.3197)]
rho_pmma = 1.19

# Aluminum
al_comp = [("Al", 1.0)]
rho_al = 2.70

def mu_cyst_from_basis(mu_pmma_E: np.ndarray, mu_al_E: np.ndarray) -> np.ndarray:
    return 0.812 * mu_pmma_E + 0.0323 * mu_al_E

def mu_mass_from_basis(mu_pmma_E: np.ndarray, mu_al_E: np.ndarray) -> np.ndarray:
    return 0.859 * mu_pmma_E + 0.0296 * mu_al_E

def build_detector_and_tube(system_yaml: str):
    sys = casymir.casymir.System(system_yaml)
    material = sys.detector["active_layer"]
    detector_type = sys.detector["type"]

    detectors_package = "casymir.data.detectors"
    material_filename = f"{material}.yaml"
    with resources.path(detectors_package, material_filename) as p:
        material_path = str(p)

    det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)
    return det, tube


def make_spectrum(det, tube, kV: float, mAs: float, name: str):
    return casymir.casymir.Spectrum(
        name=name, kV=float(kV), mAs=float(mAs), detector=det, tube=tube
    )


def mu_eff_from_comp(spec, comp, rho, L_mm: float) -> float:
    mu_E = at.linear_mu_spectrum(spec.energy, comp, rho)
    return at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_E, L_mm)


def load_pcm_bundle(path):
    data = np.load(path, allow_pickle=True)

    return {
        "fx": data["fx"],
        "fy": data["fy"],
        "fz": data["fz"],
        "kV": data["kV"],
        "W":  data["W"],
        "MTF": data["MTF"]
    }



def meshgrids_from_axes(fx, fy, fz):
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
    return FX, FY, FZ



def main():
    det_LE, tube_LE = build_detector_and_tube(LE_system_yaml)
    det_HE, tube_HE = build_detector_and_tube(HE_system_yaml)

    LE_kVs = np.arange(23, 36, 3)
    HE_kVs = np.arange(38, 51, 3)

    muLE_ad = np.zeros(len(LE_kVs))
    muLE_gl = np.zeros(len(LE_kVs))
    muHE_ad = np.zeros(len(HE_kVs))
    muHE_gl = np.zeros(len(HE_kVs))

    muLE_c = np.zeros(len(LE_kVs))
    muLE_m = np.zeros(len(LE_kVs))
    muHE_c = np.zeros(len(HE_kVs))
    muHE_m = np.zeros(len(HE_kVs))

    for i, kV in enumerate(LE_kVs):
        spec = make_spectrum(det_LE, tube_LE, kV, mAs_LE, "LE")
        muLE_ad[i] = mu_eff_from_comp(spec, adipose_comp, rho_adipose, T_breast_mm)
        muLE_gl[i] = mu_eff_from_comp(spec, glandular_comp, rho_gland, T_breast_mm)

        mu_pmma_E = at.linear_mu_spectrum(spec.energy, pmma_comp, rho_pmma)
        mu_al_E   = at.linear_mu_spectrum(spec.energy, al_comp, rho_al)

        mu_c_E = mu_cyst_from_basis(mu_pmma_E, mu_al_E)
        mu_m_E = mu_mass_from_basis(mu_pmma_E, mu_al_E)

        muLE_c[i] = at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_c_E, L_ref_lesion_mm)
        muLE_m[i] = at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_m_E, L_ref_lesion_mm)

    for j, kV in enumerate(HE_kVs):
        spec = make_spectrum(det_HE, tube_HE, kV, mAs_HE, "HE")
        muHE_ad[j] = mu_eff_from_comp(spec, adipose_comp, rho_adipose, T_breast_mm)
        muHE_gl[j] = mu_eff_from_comp(spec, glandular_comp, rho_gland, T_breast_mm)

        mu_pmma_E = at.linear_mu_spectrum(spec.energy, pmma_comp, rho_pmma)
        mu_al_E   = at.linear_mu_spectrum(spec.energy, al_comp, rho_al)

        mu_c_E = mu_cyst_from_basis(mu_pmma_E, mu_al_E)
        mu_m_E = mu_mass_from_basis(mu_pmma_E, mu_al_E)

        muHE_c[j] = at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_c_E, L_ref_lesion_mm)
        muHE_m[j] = at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_m_E, L_ref_lesion_mm)

    dprime = np.zeros((len(LE_kVs), len(HE_kVs), len(w_vals)), dtype=float)
    CDE    = np.zeros_like(dprime)
    kappa  = np.zeros_like(dprime)

    pcm_cache_LE = {}
    pcm_cache_HE = {}

    LE = np.load("pcm_cache/pcm_LE.npz", allow_pickle=True)
    HE = np.load("pcm_cache/pcm_HE.npz", allow_pickle=True)

    fx = LE["fx"]
    fy = LE["fy"]
    fz = LE["fz"]

    FX, FY, FZ = meshgrids_from_axes(fx, fy, fz)

    LE_kVs = LE["kV"]
    HE_kVs = HE["kV"]

    N_LE = len(LE_kVs)
    N_HE = len(HE_kVs)
    N_w = len(w_vals)

    O3D = tm.gaussian_task_spectrum(
        FX, FY, FZ,
        sigma_mm=sigma_mm
    )

    for iLE in range(N_LE):

        W_LE = LE["W"][iLE]

        for jHE in range(N_HE):


            W_HE = HE["W"][jHE]
            MTF = HE["MTF"][jHE]/np.max(HE["MTF"][jHE])

            for iw, w in enumerate(w_vals):
                W_Q = W_HE + (w ** 2) * W_LE

                C_HE = muHE_m[jHE] - muHE_c[jHE]
                C_LE = muLE_m[iLE] - muLE_c[iLE]

                C_de = (C_HE - w * C_LE) * t_lesion_mm
                CDE[iLE, jHE, iw] = C_de

                muDE_ad = muHE_ad[jHE] - w * muLE_ad[iLE]
                muDE_gl = muHE_gl[jHE] - w * muLE_gl[iLE]

                kap = kappa_scale * (muDE_gl - muDE_ad) ** 2
                kappa[iLE, jHE, iw] = kap

                W_bg = bg.powerlaw_background(
                    FX, FY, FZ,
                    kappa=kap,
                    beta=beta
                )

                W_task = C_de * O3D

                dp2 = tobs.ideal_observer_dprime2(
                    MTF3=MTF,
                    W3=W_Q,
                    FX=FX, FY=FY, FZ=FZ,
                    W_task=W_task,
                    W_bg=W_bg
                )

                dprime[iLE, jHE, iw] = np.sqrt(max(dp2, 0.0))
                # print("LE mu_m, mu_c:", muLE_m[iLE], muLE_c[iLE])
                # print("HE mu_m, mu_c:", muHE_m[jHE], muHE_c[jHE])
                # if iLE == 0 and jHE == 0 and (iw in [0, 10, 20, 30, 40]):
                #     print("w=", w,
                #           "C_HE=", C_HE,
                #           "C_LE=", C_LE,
                #           "C_de=", C_de)

            best_iw = int(np.argmax(dprime[iLE, jHE, :]))

            print(
                f"LE={LE_kVs[iLE]:>2.0f}, HE={HE_kVs[jHE]:>2.0f} | "
                f"best d'={dprime[iLE, jHE, best_iw]:.3e} at w={w_vals[best_iw]:.3f} | "
                f"C_DE={CDE[iLE, jHE, best_iw]:.3e} | "
                f"kappa={kappa[iLE, jHE, best_iw]:.3e}"
            )

    print(f"d' max = {np.max(dprime):.6e}")
    idx = np.unravel_index(np.argmax(dprime), dprime.shape)
    print(f"BEST @ (iLE,jHE,iw)={idx} -> LE={LE_kVs[idx[0]]}, HE={HE_kVs[idx[1]]}, w={w_vals[idx[2]]:.3f}")

    np.savez(
        OUT_NPZ,
        dprime=dprime,
        CDE=CDE,
        kappa=kappa,
        LE_kVs=LE_kVs,
        HE_kVs=HE_kVs,
        w_vals=w_vals,
        fx=fx,
        fy=fy,
        fz=fz,
    )

    print(f"\nSaved: {OUT_NPZ}")


if __name__ == "__main__":
    main()
