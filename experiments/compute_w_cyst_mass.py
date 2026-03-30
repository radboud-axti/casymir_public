import numpy as np
from importlib import resources
from matplotlib import pyplot as plt

import casymir.casymir
import utils.attenuation as at

# ============================================================
# Helpers
# ============================================================

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


def mu_eff_for_material(spec, comp, rho_g_cm3, L_mm):
    mu_E = at.linear_mu_spectrum(spec.energy, comp, rho_g_cm3)
    return at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_E, L_mm)


def mu_I_eff_through_background(
    E_keV,
    psi,
    mu_bg_E,
    mu_I_E,
    L_bg_mm,
):
    """
    Linearized iodine attenuation coefficient for a log image,
    weighted by background transmission.
    Hu Eq. (11) -> Eq. (12)
    """
    w = psi * np.exp(-mu_bg_E * L_bg_mm)
    num = np.trapz(w * mu_I_E, E_keV)
    den = np.trapz(w, E_keV)
    return num / (den + 1e-30)


def iodine_equivalent_thickness_mm(
    lesion_thickness_mm,
    iodine_mg_per_ml,
    rho_iodine_g_cm3=4.93,
):
    """
    Convert iodine concentration + lesion thickness
    to equivalent pure-iodine thickness (mm).
    """
    rho_I_g_cm3 = iodine_mg_per_ml * 1e-3  # mg/mL -> g/cm^3
    mass_thickness = rho_I_g_cm3 * (lesion_thickness_mm * 0.1)  # g/cm^2
    t_cm = mass_thickness / rho_iodine_g_cm3
    return t_cm * 10.0  # cm -> mm


# ============================================================
# User inputs
# ============================================================

LE_kVp_list = np.arange(22, 37, 1)
HE_kVp_list = np.arange(38, 53, 1)

mAs = 1.0  # cancels in μ_eff

LE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_LE.yaml"
HE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_HE.yaml"

L_breast_mm = 40.0  # 4 cm breast
lesion_thickness_mm = 5.0
iodine_mg_per_ml = 5.0


# ============================================================
# Hammerstein tissue compositions
# ============================================================

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
rho_iodine = 4.93


# ============================================================
# Iodine equivalent thickness
# ============================================================

tI_equiv_mm = iodine_equivalent_thickness_mm(
    lesion_thickness_mm,
    iodine_mg_per_ml,
    rho_iodine,
)

print(f"Equivalent iodine thickness tI = {tI_equiv_mm:.6f} mm")


# ============================================================
# Build systems
# ============================================================

det_LE, tube_LE = build_detector_and_tube(LE_system_yaml)
det_HE, tube_HE = build_detector_and_tube(HE_system_yaml)


# ============================================================
# Allocate outputs
# ============================================================

w_sub = np.zeros((len(LE_kVp_list), len(HE_kVp_list)))
C_sub = np.zeros_like(w_sub)


# ============================================================
# Main sweep
# ============================================================

for i, kVp_LE in enumerate(LE_kVp_list):
    spec_LE = casymir.casymir.Spectrum(
        name="LE", kV=float(kVp_LE), mAs=mAs,
        detector=det_LE, tube=tube_LE
    )

    muM_ad_E = at.linear_mu_spectrum(spec_LE.energy, adipose_comp, rho_adipose)
    muM_gl_E = at.linear_mu_spectrum(spec_LE.energy, glandular_comp, rho_gland)

    muM_ad_eff = at.mu_eff_polychromatic(spec_LE.energy, spec_LE.fluence, muM_ad_E, L_breast_mm)
    muM_gl_eff = at.mu_eff_polychromatic(spec_LE.energy, spec_LE.fluence, muM_gl_E, L_breast_mm)

    dmu_M = muM_gl_eff - muM_ad_eff

    muI_LE_E = at.linear_mu_spectrum(spec_LE.energy, iodine_comp, rho_iodine)

    for j, kVp_HE in enumerate(HE_kVp_list):
        spec_HE = casymir.casymir.Spectrum(
            name="HE", kV=float(kVp_HE), mAs=mAs,
            detector=det_HE, tube=tube_HE
        )

        muC_ad_E = at.linear_mu_spectrum(spec_HE.energy, adipose_comp, rho_adipose)
        muC_gl_E = at.linear_mu_spectrum(spec_HE.energy, glandular_comp, rho_gland)

        muC_ad_eff = at.mu_eff_polychromatic(spec_HE.energy, spec_HE.fluence, muC_ad_E, L_breast_mm)
        muC_gl_eff = at.mu_eff_polychromatic(spec_HE.energy, spec_HE.fluence, muC_gl_E, L_breast_mm)

        dmu_C = muC_gl_eff - muC_ad_eff

        # Hu Eq. (9)
        w = dmu_C / (dmu_M + 1e-30)
        w_sub[i, j] = w

        # Iodine μ(E)
        muI_HE_E = at.linear_mu_spectrum(spec_HE.energy, iodine_comp, rho_iodine)

        # Hu Eq. (12): iodine-only DE contrast
        muI_M_eff = mu_I_eff_through_background(
            spec_LE.energy, spec_LE.fluence,
            muM_ad_E, muI_LE_E, L_breast_mm
        )

        muI_C_eff = mu_I_eff_through_background(
            spec_HE.energy, spec_HE.fluence,
            muC_ad_E, muI_HE_E, L_breast_mm
        )

        C_sub[i, j] = (muI_C_eff - w * muI_M_eff) * tI_equiv_mm


# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(7, 5))
plt.imshow(
    w_sub,
    origin="lower",
    aspect="auto",
    extent=[HE_kVp_list.min(), HE_kVp_list.max(),
            LE_kVp_list.min(), LE_kVp_list.max()],
)
plt.colorbar(label=r"$w_{sub}$")
plt.xlabel("HE tube potential (kVp) [C]")
plt.ylabel("LE tube potential (kVp) [M]")
plt.title(r"$w_{sub}$ (Hu Eq. 9)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
plt.imshow(
    C_sub,
    origin="lower",
    aspect="auto",
    extent=[HE_kVp_list.min(), HE_kVp_list.max(),
            LE_kVp_list.min(), LE_kVp_list.max()],
)
plt.colorbar(label=r"$C_{sub}$ (log-domain)")
plt.xlabel("HE tube potential (kVp) [C]")
plt.ylabel("LE tube potential (kVp) [M]")
plt.title(r"$C_{sub}$ (Hu Eq. 12, iodine-only)")
plt.tight_layout()
plt.show()


print("w_sub min / max:", w_sub.min(), w_sub.max())
print("C_sub min / max:", C_sub.min(), C_sub.max())
