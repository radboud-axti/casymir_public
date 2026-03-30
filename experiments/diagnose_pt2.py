import numpy as np
import matplotlib.pyplot as plt

import tasks.task_models as tm
import tasks.backgrounds as bg

# -----------------------------
# USER SETTINGS
# -----------------------------
PCM_LE_NPZ = "pcm_cache/pcm_LE.npz"
PCM_HE_NPZ = "pcm_cache/pcm_HE.npz"

# pick a kV pair by index into the cached arrays
iLE = 0          # e.g., LE=23
jHE = 4          # e.g., HE=50

# sweep w around where you see peaks
w_vals = np.array([0.10, 0.15, 0.18, 0.20, 0.24, 0.25, 0.30])

# task params (match your iodine script)
lesion_fwhm_mm = 10.0
lesion_thickness_mm = 10.0

# anatomy params (match your cyst/mass + iodine scripts)
beta = 3.0
kappa_scale = 0.0079

# low-frequency cutoff for reporting
f_probe = [0.00, 0.02, 0.05, 0.10, 0.20]

# if you want: enforce a “safe” f floor when building power-law noise for diagnostics
# (this does NOT change your main pipeline unless you copy the idea there)
f0_bg = 1e-3  # mm^-1

# -----------------------------
# HELPERS
# -----------------------------
def meshgrids(fx, fy, fz):
    return np.meshgrid(fx, fy, fz, indexing="ij")

def radial_mean(volume, r, nbins=200):
    """Radial mean of a 3D volume vs |f|."""
    rmax = np.max(r)
    edges = np.linspace(0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    vol_flat = volume.ravel()
    r_flat = r.ravel()

    which = np.digitize(r_flat, edges) - 1
    out = np.zeros(nbins, dtype=float)
    cnt = np.zeros(nbins, dtype=int)

    for k in range(nbins):
        m = which == k
        if np.any(m):
            out[k] = np.mean(vol_flat[m])
            cnt[k] = np.sum(m)
        else:
            out[k] = np.nan
    return centers, out, cnt

def summarize(name, A):
    A = np.asarray(A)
    finite = np.isfinite(A)
    pos = finite & (A > 0)
    print(f"{name}")
    if not np.any(finite):
        print("  (no finite values)")
        return
    Af = A[finite]
    print(f"  min   = {np.min(Af):.3e}")
    print(f"  p1    = {np.percentile(Af, 1):.3e}")
    print(f"  p50   = {np.percentile(Af, 50):.3e}")
    print(f"  p99   = {np.percentile(Af, 99):.3e}")
    print(f"  max   = {np.max(Af):.3e}")
    if np.any(pos):
        Ap = A[pos]
        print(f"  (positive) p50 = {np.percentile(Ap, 50):.3e}")

def contribution_report(integrand, dV, fmag, cuts=(0.5, 0.9, 0.99)):
    """How many bins contribute X% of d'^2."""
    weights = (integrand * dV).ravel()
    weights = weights[np.isfinite(weights)]
    weights = weights[weights > 0]
    if weights.size == 0:
        print("No positive contributions.")
        return

    total = np.sum(weights)
    w_sorted = np.sort(weights)[::-1]
    cumsum = np.cumsum(w_sorted) / total

    for c in cuts:
        n = int(np.searchsorted(cumsum, c) + 1)
        frac_bins = 100.0 * n / fmag.size
        print(f"{int(c*100)}% of d'^2 from {n} bins ({frac_bins:.5f}% of spectrum)")

def argmax_location(A, FX, FY, FZ):
    idx = np.unravel_index(np.nanargmax(A), A.shape)
    fx = FX[idx]
    fy = FY[idx]
    fz = FZ[idx]
    f = np.sqrt(fx*fx + fy*fy + fz*fz)
    return idx, fx, fy, fz, f

# -----------------------------
# MAIN
# -----------------------------
def main():
    LE = np.load(PCM_LE_NPZ, allow_pickle=True)
    HE = np.load(PCM_HE_NPZ, allow_pickle=True)

    fx, fy, fz = LE["fx"], LE["fy"], LE["fz"]
    FX, FY, FZ = meshgrids(fx, fy, fz)
    fmag = np.sqrt(FX**2 + FY**2 + FZ**2)

    LE_kVs = LE["kV"]
    HE_kVs = HE["kV"]

    print("\n==============================")
    print(f"Case: iLE={iLE}, jHE={jHE}")
    print(f"LE kV = {LE_kVs[iLE]}, HE kV = {HE_kVs[jHE]}")
    print("==============================")

    W_LE = LE["W"][iLE]
    W_HE = HE["W"][jHE]
    MTF  = HE["MTF"][jHE]
    MTF  = MTF / (np.max(MTF) + 1e-30)   # match your pipeline

    summarize("W_LE", W_LE)
    summarize("W_HE", W_HE)

    # Task spectrum: your gaussian_task_spectrum uses exp(-(pi*sigma)^2 f^2)
    sigma_mm = lesion_fwhm_mm / 2.355
    O3D = tm.gaussian_task_spectrum(FX, FY, FZ, sigma_mm=sigma_mm)

    # volume element
    dfx = abs(FX[1,0,0] - FX[0,0,0])
    dfy = abs(FY[0,1,0] - FY[0,0,0])
    dfz = abs(FZ[0,0,1] - FZ[0,0,0])
    dV = dfx * dfy * dfz

    # Quick check: where are W_Q zeros?
    probe_w = 0.25
    WQ_probe = W_HE + (probe_w**2) * W_LE
    frac_zero = 100.0 * np.mean(WQ_probe == 0)
    print(f"\nExact zeros in W_Q (probe w={probe_w}): {frac_zero:.3f}%")

    # --- Sweep w and plot radial means for each
    fig, axes = plt.subplots(len(w_vals), 2, figsize=(12, 3.0*len(w_vals)), constrained_layout=True)

    for row, w in enumerate(w_vals):
        print("\n------------------------------")
        print(f"w = {w:.3f}")
        print("------------------------------")

        # Quantum-only DE noise
        W_Q = W_HE + (w**2) * W_LE

        # Anatomy powerlaw (diagnostic-safe floor so you can SEE if f=0 is the issue)
        f_safe = np.maximum(fmag, f0_bg)
        # If you want to compare to your exact bg.powerlaw_background, keep this AND the original.
        # Here, we compute a "manual" W_bg using the same kappa/beta form.
        # But we ALSO call your bg.powerlaw_background to catch bugs inside it.
        # (If bg.powerlaw_background already protects f=0, these should match.)
        # kappa is unknown here unless you plug in mu terms; for instability hunting we can set kappa=1.
        # Better: just inspect the SHAPE of W_bg for a given kappa.
        kap = 1.0
        W_bg_manual = kap / (f_safe**beta)

        W_bg_mod = bg.powerlaw_background(FX, FY, FZ, kappa=kap, beta=beta)

        # Total noise in your IO: W_tot = W_Q + (MTF^2)*W_bg
        W_tot_manual = W_Q + (MTF**2) * W_bg_manual
        W_tot_mod    = W_Q + (MTF**2) * W_bg_mod

        summarize("W_Q", W_Q)
        summarize("W_bg (manual, f0 protected)", W_bg_manual)
        summarize("W_bg (module)", W_bg_mod)

        # Use a generic contrast amplitude (since we’re diagnosing stability, not physics here)
        C_de = 1.0
        W_task = C_de * O3D

        # Integrand (manual version, no cutoff/floor)
        eps = 1e-30
        integrand_manual = (np.abs(MTF * W_task)**2) / (W_tot_mod + eps)

        summarize("W_tot (using module W_bg)", W_tot_mod)
        summarize("integrand", integrand_manual)

        idx, fxm, fym, fzm, fm = argmax_location(integrand_manual, FX, FY, FZ)
        print("MAX integrand at:")
        print(f"  fx = {fxm:.4f} mm^-1")
        print(f"  fy = {fym:.4f} mm^-1")
        print(f"  fz = {fzm:.4f} mm^-1")
        print(f"  |f| = {fm:.4e} mm^-1")

        # Contribution concentration
        contribution_report(integrand_manual, dV, fmag)

        # Report how much is below some low-f cutoff
        for fcut in [0.01, 0.02, 0.05, 0.1]:
            m = fmag < fcut
            num = np.sum(integrand_manual[m]) * dV
            den = np.sum(integrand_manual) * dV + 1e-30
            print(f"Fraction of d'^2 from |f|<{fcut:.2f}: {num/den:.3f}")

        # Radial means
        r, rq, _ = radial_mean(W_Q, fmag)
        _, rbg, _ = radial_mean((MTF**2) * W_bg_mod, fmag)
        _, rtot, _ = radial_mean(W_tot_mod, fmag)
        _, rint, _ = radial_mean(integrand_manual, fmag)

        axL = axes[row, 0]
        axR = axes[row, 1]

        axL.semilogy(r, rq, label="Quantum")
        axL.semilogy(r, rbg, label="MTF^2 * Anatomy")
        axL.semilogy(r, rtot, label="Total (rough)")
        axL.set_title(f"Noise spectra (radial mean), w={w:.3f}")
        axL.set_xlabel("|f| (mm$^{-1}$)")
        axL.set_ylabel("Radial mean")
        axL.grid(True, alpha=0.3)
        axL.legend()

        axR.semilogy(r, rint)
        axR.set_title(f"IO integrand (radial mean), w={w:.3f}")
        axR.set_xlabel("|f| (mm$^{-1}$)")
        axR.set_ylabel("Radial mean integrand")
        axR.grid(True, alpha=0.3)

        # Visualize the cutoff boundary you’re using in the observer, if any
        axR.axvline(0.05, linestyle="--", linewidth=1)

    plt.show()

    print("\nDone. If your integrand is basically a delta-spike at ultra-low |f|,")
    print("your d' will look 'funky' and your optimum will be hypersensitive to")
    print("how you handle f~0, noise floors, and exact-zero bins in W_Q.")

if __name__ == "__main__":
    main()
