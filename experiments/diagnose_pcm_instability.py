import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER SETTINGS
# -----------------------------
PCM_LE_PATH = "pcm_cache/pcm_LE.npz"
PCM_HE_PATH = "pcm_cache/pcm_HE.npz"

# Pick a known problematic case
iLE = 0          # index into LE_kVs
jHE = 2          # index into HE_kVs
w   = 0.22       # DE weight near spike

beta  = 3.0
kappa = 1e-6     # pick representative anatomy scaling

fmin_report = 0.05  # mm^-1 threshold for "near-DC"

# -----------------------------
# LOAD PCM
# -----------------------------
LE = np.load(PCM_LE_PATH, allow_pickle=True)
HE = np.load(PCM_HE_PATH, allow_pickle=True)

fx, fy, fz = LE["fx"], LE["fy"], LE["fz"]
FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")

W_LE = LE["W"][iLE]
W_HE = HE["W"][jHE]
MTF  = HE["MTF"][jHE] / np.max(HE["MTF"][jHE])

# -----------------------------
# FREQUENCY MAGNITUDE
# -----------------------------
f = np.sqrt(FX**2 + FY**2 + FZ**2)

# -----------------------------
# QUANTUM NOISE
# -----------------------------
W_Q = W_HE + (w**2) * W_LE

# -----------------------------
# ANATOMICAL NOISE (raw)
# -----------------------------
W_bg = kappa / (f**beta + 1e-30)

# -----------------------------
# TOTAL NOISE
# -----------------------------
W_tot = W_Q + (MTF**2) * W_bg

# -----------------------------
# TASK (normalized Gaussian)
# -----------------------------
sigma_mm = 10.0
O3D = np.exp(-2 * np.pi**2 * sigma_mm**2 * (FX**2 + FY**2 + FZ**2))

# -----------------------------
# IO INTEGRAND
# -----------------------------
integrand = (MTF * O3D)**2 / (W_tot + 1e-30)

# -----------------------------
# BASIC STATS
# -----------------------------
def stats(name, A):
    print(f"\n{name}")
    print(f"  min   = {np.min(A):.3e}")
    print(f"  p1    = {np.percentile(A,1):.3e}")
    print(f"  p50   = {np.percentile(A,50):.3e}")
    print(f"  p99   = {np.percentile(A,99):.3e}")
    print(f"  max   = {np.max(A):.3e}")

stats("W_Q (quantum)", W_Q)
stats("W_bg (anatomy)", W_bg)
stats("W_tot", W_tot)
stats("integrand", integrand)

# -----------------------------
# WHERE IS THE MAX CONTRIBUTION?
# -----------------------------
imax = np.unravel_index(np.argmax(integrand), integrand.shape)
fx0, fy0, fz0 = FX[imax], FY[imax], FZ[imax]

print("\nMAX integrand at:")
print(f"  fx = {fx0:.4f} mm^-1")
print(f"  fy = {fy0:.4f} mm^-1")
print(f"  fz = {fz0:.4f} mm^-1")
print(f"  |f| = {np.sqrt(fx0**2 + fy0**2 + fz0**2):.4e} mm^-1")

# -----------------------------
# HOW MANY BINS DOMINATE?
# -----------------------------
flat = integrand.ravel()
order = np.argsort(flat)[::-1]
cum = np.cumsum(flat[order])
cum /= cum[-1]

for frac in [0.5, 0.9, 0.99]:
    n = np.searchsorted(cum, frac) + 1
    print(f"{frac*100:.0f}% of d'^2 from {n} bins "
          f"({100*n/flat.size:.5f}% of spectrum)")

# -----------------------------
# DC / NEAR-DC CONTRIBUTION
# -----------------------------
dc_mask = f < fmin_report
dc_frac = np.sum(integrand[dc_mask]) / np.sum(integrand)

print(f"\nFraction of d'^2 from |f|<{fmin_report} mm^-1: {dc_frac:.3f}")

# -----------------------------
# RADIAL AVERAGES
# -----------------------------
bins = np.linspace(0, np.max(f), 50)
fb = 0.5 * (bins[1:] + bins[:-1])

def radial_avg(A):
    out = np.zeros(len(fb))
    for i in range(len(fb)):
        m = (f >= bins[i]) & (f < bins[i+1])
        out[i] = np.mean(A[m]) if np.any(m) else np.nan
    return out

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.semilogy(fb, radial_avg(W_Q), label="Quantum")
plt.semilogy(fb, radial_avg(W_bg), label="Anatomy")
plt.semilogy(fb, radial_avg(W_tot), label="Total")
plt.xlabel("|f| (mm⁻¹)")
plt.ylabel("Radial mean NPS")
plt.legend()
plt.title("Noise spectra")

plt.subplot(1,2,2)
plt.semilogy(fb, radial_avg(integrand))
plt.xlabel("|f| (mm⁻¹)")
plt.ylabel("Radial mean integrand")
plt.title("IO integrand")
plt.tight_layout()
plt.show()
