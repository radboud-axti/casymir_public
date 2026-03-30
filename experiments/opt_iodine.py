"""
optimize_iodine_3D.py

3D ideal-observer optimization for iodine detection using:
- cached CASYMIR PCM outputs (Nyquist-clipped)
- DE subtraction: I_DE = I_HE - w * I_LE
- 3D Gaussian task
- quantum + anatomical (power-law) noise

Outputs:
- dprime[LE, HE, w]
- CDE[LE, HE, w]
- kappa[LE, HE, w]
- w_hu[LE, HE]   (Hu Eq. 9 tissue-cancel reference)

This script is structurally IDENTICAL to optimize_cyst_mass_3D.py.
Only the task contrast definition differs.
"""

from __future__ import annotations
import numpy as np

import casymir.casymir
from importlib import resources

import utils.attenuation as at
import tasks.task_models as tm
import tasks.task_observers as tobs
import tasks.backgrounds as bg


# -----------------------------
# USER SETTINGS
# -----------------------------
LE_kVs = np.arange(23, 36, 1)
HE_kVs = np.arange(38, 51, 1)

# Free DE weight (sanity check vs Hu)
# w_vals = np.linspace(0.1, 0.4, 2)
w_vals = np.array([0.125])

LE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_LE.yaml"
HE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_HE.yaml"
mAs_LE = 4.0
mAs_HE = 2.0

PCM_LE_NPZ = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\experiments\pcm_cache\pcm_LE_fine.npz"
PCM_HE_NPZ = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\experiments\pcm_cache\pcm_HE_fine.npz"

# Iodine / task
lesion_fwhm_mm = 5
lesion_thickness_mm = 5
iodine_mg_per_ml = 5.0
rho_iodine = 4.93

# Anatomy (same as cyst–mass)
beta = 3.0
kappa_scale = 0.0079
T_breast_mm = 40.0
background_for_iodine = "mix"

OUT_NPZ = "opt_iodine_dprime_3D_coarse_for_d.npz"


# -----------------------------
# MATERIAL DEFINITIONS
# -----------------------------
glandular_comp = [
    ("H", 0.102), ("C", 0.184), ("N", 0.032), ("O", 0.677),
    ("P", 0.00125), ("S", 0.00125), ("K", 0.00125), ("Ca", 0.00125),
]
rho_gland = 1.04

adipose_comp = [
    ("H", 0.112), ("C", 0.619), ("N", 0.017), ("O", 0.251),
    ("P", 0.00025), ("S", 0.00025), ("K", 0.00025), ("Ca", 0.00025),
]
rho_adipose = 0.93

iodine_comp = [("I", 1.0)]


# -----------------------------
# HELPERS
# -----------------------------
def build_detector_and_tube(system_yaml):
    sys = casymir.casymir.System(system_yaml)
    material = sys.detector["active_layer"]
    with resources.path("casymir.data.detectors", f"{material}.yaml") as p:
        material_path = str(p)
    det = casymir.casymir.Detector(sys.detector["type"], material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)
    return det, tube


def make_spectrum(det, tube, kV, mAs, name):
    return casymir.casymir.Spectrum(name, float(kV), float(mAs), det, tube)


def mu_eff(spec, comp, rho, L):
    muE = at.linear_mu_spectrum(spec.energy, comp, rho)
    return at.mu_eff_polychromatic(spec.energy, spec.fluence, muE, L)


def mu_I_eff_through_bg(E, psi, mu_bg_E, mu_I_E, L):
    w = psi * np.exp(-mu_bg_E * L)
    return np.trapz(w * mu_I_E, E) / (np.trapz(w, E) + 1e-30)


def iodine_equiv_thickness_mm(t_mm, conc_mg_ml, rho_I):
    rho = conc_mg_ml * 1e-3
    mass_thick = rho * (t_mm * 0.1)
    return (mass_thick / rho_I) * 10.0


def meshgrids(fx, fy, fz):
    return np.meshgrid(fx, fy, fz, indexing="ij")


# -----------------------------
# MAIN
# -----------------------------
def main():
    LE_pcm = np.load(PCM_LE_NPZ, allow_pickle=True)
    HE_pcm = np.load(PCM_HE_NPZ, allow_pickle=True)

    fx, fy, fz = LE_pcm["fx"], LE_pcm["fy"], LE_pcm["fz"]
    FX, FY, FZ = meshgrids(fx, fy, fz)

    det_LE, tube_LE = build_detector_and_tube(LE_system_yaml)
    det_HE, tube_HE = build_detector_and_tube(HE_system_yaml)

    N_LE, N_HE, N_w = len(LE_kVs), len(HE_kVs), len(w_vals)

    muLE_ad = np.zeros(N_LE)
    muLE_gl = np.zeros(N_LE)
    muHE_ad = np.zeros(N_HE)
    muHE_gl = np.zeros(N_HE)

    muI_LE = np.zeros(N_LE)
    muI_HE = np.zeros(N_HE)

    tI_mm = iodine_equiv_thickness_mm(
        lesion_thickness_mm, iodine_mg_per_ml, rho_iodine
    )

    for i, kV in enumerate(LE_kVs):
        spec = make_spectrum(det_LE, tube_LE, kV, mAs_LE, "LE")
        muLE_ad[i] = mu_eff(spec, adipose_comp, rho_adipose, T_breast_mm)
        muLE_gl[i] = mu_eff(spec, glandular_comp, rho_gland, T_breast_mm)

        mu_bg_E = 0.5 * (
            at.linear_mu_spectrum(spec.energy, adipose_comp, rho_adipose)
            + at.linear_mu_spectrum(spec.energy, glandular_comp, rho_gland)
        )
        muI_LE[i] = mu_I_eff_through_bg(
            spec.energy, spec.fluence,
            mu_bg_E,
            at.linear_mu_spectrum(spec.energy, iodine_comp, rho_iodine),
            T_breast_mm
        )

    for j, kV in enumerate(HE_kVs):
        spec = make_spectrum(det_HE, tube_HE, kV, mAs_HE, "HE")
        muHE_ad[j] = mu_eff(spec, adipose_comp, rho_adipose, T_breast_mm)
        muHE_gl[j] = mu_eff(spec, glandular_comp, rho_gland, T_breast_mm)

        mu_bg_E = 0.5 * (
            at.linear_mu_spectrum(spec.energy, adipose_comp, rho_adipose)
            + at.linear_mu_spectrum(spec.energy, glandular_comp, rho_gland)
        )
        muI_HE[j] = mu_I_eff_through_bg(
            spec.energy, spec.fluence,
            mu_bg_E,
            at.linear_mu_spectrum(spec.energy, iodine_comp, rho_iodine),
            T_breast_mm
        )

    w_hu = np.zeros((N_LE, N_HE))
    for i in range(N_LE):
        for j in range(N_HE):
            w_hu[i, j] = (muHE_gl[j] - muHE_ad[j]) / (
                muLE_gl[i] - muLE_ad[i] + 1e-30
            )

    dprime = np.zeros((N_LE, N_HE, N_w))
    CDE = np.zeros_like(dprime)
    kappa = np.zeros_like(dprime)

    sigma_mm = lesion_fwhm_mm / 2.355
    O3D = tm.gaussian_task_spectrum(FX, FY, FZ, sigma_mm=sigma_mm)

    for iLE in range(N_LE):
        W_LE, H_LE = LE_pcm["W"][iLE], LE_pcm["MTF"][iLE]

        for jHE in range(N_HE):
            W_HE, H_HE = HE_pcm["W"][jHE], HE_pcm["MTF"][jHE]

            for iw, w in enumerate(w_vals):
                w_opt = w_hu[iLE, jHE]
                W_Q = W_HE + w_opt**2 * W_LE
                H_sub = H_HE

                C_de = (muI_HE[jHE] - w_opt * muI_LE[iLE]) * tI_mm
                CDE[iLE, jHE, iw] = C_de

                muDE_ad = muHE_ad[jHE] - w * muLE_ad[iLE]
                muDE_gl = muHE_gl[jHE] - w * muLE_gl[iLE]
                kap = kappa_scale * (muDE_gl - muDE_ad) ** 2
                kappa[iLE, jHE, iw] = kap

                W_bg = bg.powerlaw_background(FX, FY, FZ, 0.0079, beta)

                W_task = C_de * O3D

                dp2 = tobs.ideal_observer_dprime2_inplane(
                    H_sub, W_Q, FX, FY, FZ, W_task, W_bg
                )

                dprime[iLE, jHE, iw] = np.sqrt(max(dp2, 0.0))

            best = np.argmax(dprime[iLE, jHE])
            print(
                f"LE={LE_kVs[iLE]}, HE={HE_kVs[jHE]} | "
                f"CDE={CDE[iLE, jHE, best]:.5f}, "
                f"w*={w_vals[best]:.3f}, w_Hu={w_hu[iLE,jHE]:.3f}, "
                f"d'={dprime[iLE,jHE,best]:.3e}, "
                f"kappa={kappa[iLE, jHE,best]:.3e}"
            )

    np.savez(
        OUT_NPZ,
        dprime=dprime,
        CDE=CDE,
        kappa=kappa,
        w_hu=w_hu,
        LE_kVs=LE_kVs,
        HE_kVs=HE_kVs,
        w_vals=w_vals,
        fx=fx, fy=fy, fz=fz,
    )
    print(f"\nSaved: {OUT_NPZ}")


if __name__ == "__main__":
    main()
