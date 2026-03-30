import gc
import numpy as np
from importlib import resources
from tqdm import tqdm
from matplotlib import pyplot as plt
import casymir.casymir
import casymir.processes
import casymir.processes_2d
import casymir.processes_3d

import utils.attenuation as at

def mu_eff_from_material(
    spectrum,
    composition: list[tuple[str, float]],
    density_g_cm3: float,
    thickness_mm: float,
    *,
    mu_source: str = "BOONE",
) -> float:
    """
    Compute μ_eff for a material using a CASYMIR Spectrum object.
    """

    mu_E = at.linear_mu_spectrum(
        spectrum.energy,
        composition,
        density_g_cm3,
        mu_source=mu_source,
    )

    return at.mu_eff_polychromatic(
        spectrum.energy,
        spectrum.fluence,
        mu_E,
        thickness_mm,
    )

def mu_mixture_from_iodine_conc(
    E_keV,
    mu_tissue_E,
    mu_iodine_E,
    iodine_mg_per_mL,
    rho_bg_g_cm3=1.0,
):
    """
    Build linear attenuation μ(E) for a tissue + iodine mixture.
    iodine_mg_per_mL is the iodine mass concentration (mg/mL).
    """
    # mg/mL == g/L; and 1 mL == 1 cm^3, so:
    rho_I_g_cm3 = iodine_mg_per_mL / 1000.0  # g/cm^3 iodine mass concentration

    # iodine mass fraction (approx, dilute)
    wI = rho_I_g_cm3 / rho_bg_g_cm3

    # linear mixture in mass fraction
    return (1.0 - wI) * mu_tissue_E + wI * mu_iodine_E


# -----------------------------
# Acquisition
# -----------------------------
kV  = 49
mAs = 4
system_path = "C:\\Users\\Z639176\\Documents\\Projects\\casymir_v2\\casymir_public\\optimize_dbt_HE.yaml"

# -----------------------------
# Load system
# -----------------------------
sys = casymir.casymir.System(system_path)

material = sys.detector["active_layer"]
detector_type = sys.detector["type"]

detectors_package = "casymir.data.detectors"
material_filename = f"{material}.yaml"

with resources.path(detectors_package, material_filename) as p:
    material_path = str(p)

det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
tube = casymir.casymir.Tube(sys.source)

spec = casymir.casymir.Spectrum(
    name="dbt_spec",
    kV=kV,
    mAs=mAs,
    detector=det,
    tube=tube,
)

E = spec.energy
# --- material definitions ---
iodine_comp = [('I', 1.0)]
iodine_density = 4.93  # g/cm^3 (solid iodine, placeholder)

tissue_comp = [
    ('H', 0.11),
    ('C', 0.12),
    ('N', 0.03),
    ('O', 0.74),
]
tissue_density = 1.0  # g/cm^3

# --- compute linear μ(E) ---
mu_I = at.linear_mu_spectrum(
    E,
    iodine_comp,
    iodine_density,
)

mu_tissue = at.linear_mu_spectrum(
    E,
    tissue_comp,
    tissue_density,
)

# --- plot ---
plt.figure(figsize=(6, 4))
plt.semilogy(E, mu_I, label="Iodine", linewidth=2)
plt.semilogy(E, mu_tissue, label="Soft tissue", linewidth=2)

plt.axvline(33.2, color="k", linestyle="--", alpha=0.5, label="I K-edge")

plt.xlabel("Energy (keV)")
plt.ylabel(r"Linear attenuation $\mu(E)$ (mm$^{-1}$)")
plt.title("Spectral linear attenuation coefficients")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


iodine_mg_per_mL = 5.0  # Hu
mu_bg_E = at.linear_mu_spectrum(E, tissue_comp, tissue_density)
mu_I_E  = at.linear_mu_spectrum(E, [('I',1.0)], iodine_density)

mu_lesion_E = mu_mixture_from_iodine_conc(
    E, mu_bg_E, mu_I_E,
    iodine_mg_per_mL=iodine_mg_per_mL,
    rho_bg_g_cm3=1.0,
)


L_bg = 40.0  # mm
t_lesion_path = 5.0  # mm (Hu's Gaussian lesion size scale; you can refine later)

mu_bg_eff = at.mu_eff_polychromatic(E, spec.fluence, mu_bg_E, L_bg)

# exponent should be μ_bg*L_bg + (μ_lesion - μ_bg)*t_path
delta_mu_E = mu_lesion_E - mu_bg_E
mu_tot_E = mu_bg_E + delta_mu_E * (t_lesion_path / L_bg)

mu_tot_eff = at.mu_eff_polychromatic(E, spec.fluence, mu_tot_E, L_bg)

C = mu_tot_eff - mu_bg_eff
print("C (log-domain iodine contrast):", C)

print("end")