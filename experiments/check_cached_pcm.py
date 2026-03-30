import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load cached PCMs
# -----------------------------
LE = np.load("pcm_cache/pcm_LE.npz", allow_pickle=True)
HE = np.load("pcm_cache/pcm_HE.npz", allow_pickle=True)

fx = LE["fx"]
fy = LE["fy"]
fz = LE["fz"]

iy0 = np.argmin(np.abs(fy))
iz0 = np.argmin(np.abs(fz))

# -----------------------------
# Select extreme indices
# -----------------------------
idx_LE_low  = 0
idx_LE_high = -1

idx_HE_low  = 0
idx_HE_high = -1

# -----------------------------
# Extract NPS profiles
# -----------------------------
W_LE_low  = LE["W"][idx_LE_low,  :, iy0, iz0]
W_LE_high = LE["W"][idx_LE_high, :, iy0, iz0]

W_HE_low  = HE["W"][idx_HE_low,  :, iy0, iz0]
W_HE_high = HE["W"][idx_HE_high, :, iy0, iz0]

kV_LE = LE["kV"]
kV_HE = HE["kV"]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

plt.plot(fx, W_LE_low,  label=f"LE {kV_LE[idx_LE_low]} kVp",  lw=2)
plt.plot(fx, W_LE_high, label=f"LE {kV_LE[idx_LE_high]} kVp", lw=2)

plt.plot(fx, W_HE_low,  label=f"HE {kV_HE[idx_HE_low]} kVp",  lw=2, ls="--")
plt.plot(fx, W_HE_high, label=f"HE {kV_HE[idx_HE_high]} kVp", lw=2, ls="--")

plt.xlabel(r"Spatial frequency $f_x$ (mm$^{-1}$)")
plt.ylabel(r"NPS $W(f_x, 0, 0)$")
plt.title("PCM sanity check: in-plane NPS profiles")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Extract MTF profiles
# -----------------------------
MTF_LE_low  = LE["MTF"][idx_LE_low,  :, iy0, iz0]
MTF_LE_high = LE["MTF"][idx_LE_high, :, iy0, iz0]

MTF_HE_low  = HE["MTF"][idx_HE_low,  :, iy0, iz0]
MTF_HE_high = HE["MTF"][idx_HE_high, :, iy0, iz0]

kV_LE = LE["kV"]
kV_HE = HE["kV"]

ix0 = np.argmin(np.abs(fx)) + 1

MTF_LE_low = np.abs(MTF_LE_low)
MTF_LE_low /= np.max(MTF_LE_low)

MTF_LE_high = np.abs(MTF_LE_high)
MTF_LE_high /= np.max(MTF_LE_high)

MTF_HE_low = np.abs(MTF_HE_low)
MTF_HE_low /= np.max(MTF_HE_low)

MTF_HE_high = np.abs(MTF_HE_high)
MTF_HE_high /= np.max(MTF_HE_high)
# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

plt.plot(fx, MTF_LE_low,  label=f"LE {kV_LE[idx_LE_low]} kVp",  lw=2)
plt.plot(fx, MTF_LE_high, label=f"LE {kV_LE[idx_LE_high]} kVp", lw=2)

plt.plot(fx, MTF_HE_low,  label=f"HE {kV_HE[idx_HE_low]} kVp",  lw=2, ls="--")
plt.plot(fx, MTF_HE_high, label=f"HE {kV_HE[idx_HE_high]} kVp", lw=2, ls="--")

plt.xlabel(r"Spatial frequency $f_x$ (mm$^{-1}$)")
plt.ylabel("MTF")
plt.title("PCM sanity check: in-plane MTF profiles")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()