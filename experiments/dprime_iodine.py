import numpy as np
from matplotlib import pyplot as plt
from importlib import resources

import casymir.casymir
import utils.attenuation as at

# ------------------------------------------------------------
# Import YOUR PCM runner (adjust import path as needed)
# ------------------------------------------------------------
# from your_module_name import run_pcm_dbt
from run_pcm import run_pcm_dbt  # <- change if needed


# ============================================================
# Helpers: build det+tube for spectral computations
# ============================================================

def build_detector_and_tube(system_yaml: str):
    sys = casymir.casymir.System(system_yaml)
    material = sys.detector["active_layer"]
    detector_type = sys.detector["type"]

    with resources.path("casymir.data.detectors", f"{material}.yaml") as p:
        material_path = str(p)

    det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)
    return det, tube


# ============================================================
# Hu Eq. (10) μ_eff and Hu Eq. (9) weight
# ============================================================

def mu_eff_for_material(spec, comp, rho_g_cm3, L_mm):
    mu_E = at.linear_mu_spectrum(spec.energy, comp, rho_g_cm3)
    return at.mu_eff_polychromatic(spec.energy, spec.fluence, mu_E, L_mm)


# ============================================================
# Hu Eq. (12): iodine μ through background (linearized)
# ============================================================

def mu_I_eff_through_background(E_keV, psi, mu_bg_E, mu_I_E, L_bg_mm):
    w = psi * np.exp(-mu_bg_E * L_bg_mm)
    num = np.trapz(w * mu_I_E, E_keV)
    den = np.trapz(w, E_keV)
    return num / (den + 1e-30)  # mm^-1


def iodine_equivalent_thickness_mm(lesion_thickness_mm, iodine_mg_per_ml, rho_iodine_g_cm3=4.93):
    # mg/mL -> g/cm^3
    rho_I_g_cm3 = iodine_mg_per_ml * 1e-3
    # mass thickness (g/cm^2): rho * thickness(cm)
    mass_thickness_g_cm2 = rho_I_g_cm3 * (lesion_thickness_mm * 0.1)
    # equivalent pure iodine thickness (cm)
    t_cm = mass_thickness_g_cm2 / rho_iodine_g_cm3
    return t_cm * 10.0  # mm


# ============================================================
# Task function (2D Gaussian in frequency domain)
# ============================================================

def gaussian_task_ft_2d(FX, FY, fwhm_mm):
    sigma = fwhm_mm / 2.355
    return np.exp(-2.0 * (np.pi**2) * (sigma**2) * (FX**2 + FY**2))


# ============================================================
# Frequency clipping (your model outputs span ±2 fNy)
# ============================================================

def nyquist_from_axis(f_axis):
    # axis expected to cover [-2 fNy, 2 fNy); so fNy is half of max abs
    return np.max(np.abs(f_axis)) / 2.0


def clip_to_nyquist_2d(fx, fy, A2d):
    fny_x = nyquist_from_axis(fx)
    fny_y = nyquist_from_axis(fy)

    mx = np.abs(fx) <= (fny_x + 1e-12)
    my = np.abs(fy) <= (fny_y + 1e-12)

    return fx[mx], fy[my], A2d[np.ix_(mx, my)]


# ============================================================
# Extract in-plane slices from 3D outputs (fz=0)
# ============================================================

def extract_inplane_from_pcm(pcm_out):
    fx = pcm_out["fx"]
    fy = pcm_out["fy"]
    fz = pcm_out["fz"]
    iz0 = len(fz) // 2

    # In-plane slices
    Hxy = pcm_out["MTF3"][:, :, iz0]   # transfer (dimensionless)
    Wxy = pcm_out["W3"][:, :, iz0]     # Wiener spectrum in recon domain
    return fx, fy, Hxy, Wxy


# ============================================================
# Quantum-limited scaling of W with mAs
# ============================================================

def scale_wiener_for_mAs(W_ref, mAs_ref, mAs_new):
    # Quantum-limited: NPS ~ 1/mAs
    return W_ref * (mAs_ref / max(mAs_new, 1e-12))


# ============================================================
# Compute d' for one (LE, HE) pair and optimize over fh
# ============================================================

def compute_pair_dprime(
    kVp_LE,
    kVp_HE,
    *,
    det_LE, tube_LE,
    det_HE, tube_HE,
    LE_system_yaml,
    HE_system_yaml,
    angles_rad,
    fz_axis,
    recon_params,
    mAs_ref_pcm=1.0,
    total_dose_uGy=1500.0,
    fh_grid=np.linspace(0.1, 0.9, 17),
    lesion_fwhm_mm=5.0,
    iodine_mg_per_ml=5.0,
    L_breast_mm=40.0,
    background_for_iodine="mix",  # "adipose" or "mix"
):
    # --- build spectra for μ computations (fast)
    spec_LE = casymir.casymir.Spectrum("LE", kV=float(kVp_LE), mAs=1.0, detector=det_LE, tube=tube_LE)
    spec_HE = casymir.casymir.Spectrum("HE", kV=float(kVp_HE), mAs=1.0, detector=det_HE, tube=tube_HE)

    # --- Hu Eq. (9) weight
    muM_ad_eff = mu_eff_for_material(spec_LE, adipose_comp,   rho_adipose, L_breast_mm)
    muM_gl_eff = mu_eff_for_material(spec_LE, glandular_comp, rho_gland,   L_breast_mm)
    muC_ad_eff = mu_eff_for_material(spec_HE, adipose_comp,   rho_adipose, L_breast_mm)
    muC_gl_eff = mu_eff_for_material(spec_HE, glandular_comp, rho_gland,   L_breast_mm)

    dmu_M = muM_gl_eff - muM_ad_eff
    dmu_C = muC_gl_eff - muC_ad_eff
    w_sub = dmu_C / (dmu_M + 1e-30)

    # --- iodine equivalent thickness
    tI_mm = iodine_equivalent_thickness_mm(lesion_fwhm_mm, iodine_mg_per_ml, rho_iodine)

    # --- iodine μ_eff through background
    muM_ad_E = at.linear_mu_spectrum(spec_LE.energy, adipose_comp, rho_adipose)
    muM_gl_E = at.linear_mu_spectrum(spec_LE.energy, glandular_comp, rho_gland)
    muC_ad_E = at.linear_mu_spectrum(spec_HE.energy, adipose_comp, rho_adipose)
    muC_gl_E = at.linear_mu_spectrum(spec_HE.energy, glandular_comp, rho_gland)

    if background_for_iodine == "mix":
        muM_bg_E = 0.5 * muM_ad_E + 0.5 * muM_gl_E
        muC_bg_E = 0.5 * muC_ad_E + 0.5 * muC_gl_E
    else:
        muM_bg_E = muM_ad_E
        muC_bg_E = muC_ad_E

    muI_LE_E = at.linear_mu_spectrum(spec_LE.energy, iodine_comp, rho_iodine)
    muI_HE_E = at.linear_mu_spectrum(spec_HE.energy, iodine_comp, rho_iodine)

    muI_M_eff = mu_I_eff_through_background(spec_LE.energy, spec_LE.fluence, muM_bg_E, muI_LE_E, L_breast_mm)
    muI_C_eff = mu_I_eff_through_background(spec_HE.energy, spec_HE.fluence, muC_bg_E, muI_HE_E, L_breast_mm)

    # Hu Eq. (12) iodine-only DE contrast (sign irrelevant)
    C_sub = (muI_C_eff - w_sub * muI_M_eff) * tI_mm

    # --- run PCM for LE and HE (slow)
    pcm_LE = run_pcm_dbt(
        kV=float(kVp_LE),
        mAs=mAs_ref_pcm,
        system_yaml=LE_system_yaml,
        angles_rad=angles_rad,
        fz=fz_axis,
        recon_params=recon_params,
    )

    pcm_HE = run_pcm_dbt(
        kV=float(kVp_HE),
        mAs=mAs_ref_pcm,
        system_yaml=HE_system_yaml,
        angles_rad=angles_rad,
        fz=fz_axis,
        recon_params=recon_params,
    )

    # --- extract in-plane
    fx, fy, H_LE_xy, W_LE_xy = extract_inplane_from_pcm(pcm_LE)
    fx2, fy2, H_HE_xy, W_HE_xy = extract_inplane_from_pcm(pcm_HE)

    if (not np.allclose(fx, fx2)) or (not np.allclose(fy, fy2)):
        raise ValueError("Frequency grids do not match between LE and HE PCM outputs.")

    # --- clip to Nyquist
    fx_c, fy_c, H_LE = clip_to_nyquist_2d(fx, fy, H_LE_xy)
    _,   _,   H_HE = clip_to_nyquist_2d(fx, fy, H_HE_xy)
    _,   _,   W_LE = clip_to_nyquist_2d(fx, fy, W_LE_xy)
    _,   _,   W_HE = clip_to_nyquist_2d(fx, fy, W_HE_xy)

    # --- task spectrum
    FX, FY = np.meshgrid(fx_c, fy_c, indexing="ij")
    T = gaussian_task_ft_2d(FX, FY, lesion_fwhm_mm)

    # --- subtraction transfer
    H_sub = H_HE - w_sub * H_LE

    # --- integration measure
    dfx = float(np.mean(np.diff(fx_c)))
    dfy = float(np.mean(np.diff(fy_c)))
    dA = dfx * dfy

    # --- convert dose (uGy) to mAs using PCM's mean DAK per view (uGy)
    # If dac_mean scales with mAs (it does), we can use dak_mean at mAs_ref_pcm:
    dak_LE_per_mAs = pcm_LE["dak_mean"] / mAs_ref_pcm
    dak_HE_per_mAs = pcm_HE["dak_mean"] / mAs_ref_pcm

    best_dprime = -np.inf
    best_fh = None

    for fh in fh_grid:
        dose_HE = fh * total_dose_uGy
        dose_LE = (1.0 - fh) * total_dose_uGy

        mAs_HE = dose_HE / max(dak_HE_per_mAs, 1e-12)
        mAs_LE = dose_LE / max(dak_LE_per_mAs, 1e-12)

        # scale Wiener spectra to those mAs values (quantum-limited)
        W_HE_s = scale_wiener_for_mAs(W_HE, mAs_ref_pcm, mAs_HE)
        W_LE_s = scale_wiener_for_mAs(W_LE, mAs_ref_pcm, mAs_LE)

        # subtraction noise (independent)
        W_sub = W_HE_s + (w_sub**2) * W_LE_s

        # detectability integral
        # H(f) includes system blur; task amplitude is C_sub
        Numer = np.abs(C_sub * T * H_sub)**2
        d2 = np.sum(Numer / (W_sub + 1e-30)) * dA
        dprime = np.sqrt(max(d2, 0.0))

        if dprime > best_dprime:
            best_dprime = dprime
            best_fh = fh

    return best_dprime, best_fh, w_sub, C_sub


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # Energy sweep (coarser steps)
    # -----------------------------
    step = 3  # <- set to 2 or 3
    LE_vals = np.arange(23, 36, step)   # 23..35
    HE_vals = np.arange(38, 53, step)   # 38..52

    # -----------------------------
    # Dose / lesion (Hu-like)
    # -----------------------------
    total_dose_uGy = 1500.0
    fh_grid = np.linspace(0.1, 0.9, 17)

    lesion_fwhm_mm = 5.0
    iodine_mg_per_ml = 5.0
    L_breast_mm = 40.0

    # -----------------------------
    # Geometry & recon params
    # -----------------------------
    Nv = 25
    angles_rad = np.linspace(np.deg2rad(-25), np.deg2rad(25), Nv)
    Nz = 64
    dz = 1.0
    fz_axis = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)

    recon_params = {
        "Theta_rad": np.deg2rad(50.0),
        "sa_A": 1.5,
        "kernel": "gaussian",
        "slice_B": 0.05,
    }

    # -----------------------------
    # Systems (separate LE + HE)
    # -----------------------------
    LE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_LE.yaml"
    HE_system_yaml = r"C:\Users\Z639176\Documents\Projects\casymir_v2\casymir_public\optimize_dbt_HE.yaml"

    det_LE, tube_LE = build_detector_and_tube(LE_system_yaml)
    det_HE, tube_HE = build_detector_and_tube(HE_system_yaml)

    # -----------------------------
    # Materials (Hammerstein + iodine)
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
    rho_iodine = 4.93

    # -----------------------------
    # Allocate surfaces
    # -----------------------------
    Dp = np.zeros((len(LE_vals), len(HE_vals)), dtype=float)
    fh_opt = np.zeros_like(Dp)
    w_surf = np.zeros_like(Dp)
    C_surf = np.zeros_like(Dp)

    # -----------------------------
    # Sweep
    # -----------------------------
    for i, kLE in enumerate(LE_vals):
        for j, kHE in enumerate(HE_vals):
            dprime, fh, w, C = compute_pair_dprime(
                kLE, kHE,
                det_LE=det_LE, tube_LE=tube_LE,
                det_HE=det_HE, tube_HE=tube_HE,
                LE_system_yaml=LE_system_yaml,
                HE_system_yaml=HE_system_yaml,
                angles_rad=angles_rad,
                fz_axis=fz_axis,
                recon_params=recon_params,
                mAs_ref_pcm=1.0,
                total_dose_uGy=total_dose_uGy,
                fh_grid=fh_grid,
                lesion_fwhm_mm=lesion_fwhm_mm,
                iodine_mg_per_ml=iodine_mg_per_ml,
                L_breast_mm=L_breast_mm,
                background_for_iodine="mix",  # try "adipose" too
            )

            Dp[i, j] = dprime
            fh_opt[i, j] = fh
            w_surf[i, j] = w
            C_surf[i, j] = C

            print(f"LE={kLE:>2}, HE={kHE:>2} | d'={dprime:.3f} | fh*={fh:.2f} | w={w:.3f} | C={C:.5f}")

    # -----------------------------
    # Plot helper
    # -----------------------------
    def show_surface(A, title, cbar_label):
        plt.figure(figsize=(7, 5))
        plt.imshow(
            A,
            origin="lower",
            aspect="auto",
            extent=[HE_vals.min(), HE_vals.max(), LE_vals.min(), LE_vals.max()],
        )
        plt.colorbar(label=cbar_label)
        plt.xlabel("HE tube potential (kVp)")
        plt.ylabel("LE tube potential (kVp)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plots
    # -----------------------------
    show_surface(Dp, r"$d'$ (ideal observer, DE subtraction)", r"$d'$")
    show_surface(fh_opt, r"Optimal dose fraction $f_h$", r"$f_h$")
    show_surface(w_surf, r"$w_{sub}$ (Hu Eq. 9)", r"$w_{sub}$")
    show_surface(C_surf, r"$C_{sub}$ (Hu Eq. 12)", r"$C_{sub}$")

    # -----------------------------
    # Best overall
    # -----------------------------
    idx = np.unravel_index(np.argmax(Dp), Dp.shape)
    print("\n=== BEST OVERALL ===")
    print(f"LE kVp: {LE_vals[idx[0]]}")
    print(f"HE kVp: {HE_vals[idx[1]]}")
    print(f"d'    : {Dp[idx]:.4f}")
    print(f"fh*   : {fh_opt[idx]:.3f}")
    print(f"w_sub : {w_surf[idx]:.4f}")
    print(f"C_sub : {C_surf[idx]:.6f}")
