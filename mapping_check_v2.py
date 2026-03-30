import pickle, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- helpers ----------------
def nearest_idx(axis, val):
    return int(np.argmin(np.abs(axis - val)))

def gaussian_weights(center, axis, sigma, truncate=3.0):
    """
    1D normalized Gaussian weights on 'axis' centered at 'center' with std 'sigma'.
    Truncated to |x-center| <= truncate*sigma for speed.
    """
    if sigma <= 0:
        j = nearest_idx(axis, center)
        w = np.zeros_like(axis, dtype=np.float32)
        w[j] = 1.0
        return w
    lo = center - truncate * sigma
    hi = center + truncate * sigma
    m  = (axis >= lo) & (axis <= hi)
    x  = axis[m]
    # use float64 for the exp then cast
    w  = np.exp(-0.5 * ((x - center) / sigma)**2, dtype=np.float64).astype(np.float32)
    s  = float(w.sum())
    if s > 0:
        w /= s
    W = np.zeros_like(axis, dtype=np.float32)
    W[m] = w
    return W

def sinc_weights(center, axis, width=1.0, truncate=3.0):
    """
    1D sinc-based interpolation weights centered at 'center' (in same units as 'axis').
    width controls the main-lobe width; truncate cuts off far side lobes.
    """
    x = axis - center
    w = np.sinc(x / width)  # normalized sinc = sin(pi*x)/(pi*x)
    m = np.abs(x) <= truncate * width
    W = np.zeros_like(axis, dtype=np.float32)
    W[m] = w[m]
    s = np.sum(np.abs(W))
    if s > 0:
        W /= s
    return W

def triangular_weights(center, axis, width):
    x = np.abs(axis - center)
    w = np.maximum(1 - x / width, 0)
    w /= np.sum(w)
    return w

def show_2d(A, x_axis, y_axis, xlabel, ylabel, title, aspect="equal", cmap="gray", mode="in-plane"):
    dims = np.shape(A)
    idx_x = dims[0] // 2
    idx_y = dims[1] // 2
    plt.figure(figsize=(6,5))
    if mode == "in-plane":
        extent = [x_axis.min() / 2, x_axis.max() / 2, y_axis.min() / 2, y_axis.max() / 2]
        plt.imshow(A[idx_x - idx_x // 2: idx_x + idx_x // 2,
                     idx_y - idx_y // 2: idx_y + idx_y // 2],
                   extent=extent, origin="lower", cmap=cmap, aspect=aspect)
    else:
        extent = [x_axis.min() / 2, x_axis.max() / 2, y_axis.min(), y_axis.max()]
        plt.imshow(A[idx_x - idx_x // 2: idx_x + idx_x // 2, :],
                   extent=extent, origin="lower", cmap=cmap, aspect=aspect)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    # plt.colorbar()
    plt.tight_layout()
    plt.show()

# ---------------- load stack ----------------
with open("projections.pkl", "rb") as f:
    stack = pickle.load(f)

S_views = stack.S         # (Nv, Nx, Ny) float32
W_views = stack.W         # (Nv, Nx, Ny)
angles  = np.asarray(stack.angles, dtype=float)  # (Nv,)
fx = stack.fx.astype(np.float32)                 # (Nx,)
fy = stack.fy.astype(np.float32)                 # (Ny,)
Nv, Nx, Ny = S_views.shape
print(f"Nv={Nv}, Nx={Nx}, Ny={Ny}")

# ---------------- z-axis & constants ----------------
Nz  = 64
dz  = 1.0                                          # mm
fz  = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
dfz = float(fz[1] - fz[0])

# detector Nyquist and ST window bound
px = getattr(stack, "px_size", None)
if px is None:
    px = 0.085                                     # mm (85 µm) if not carried in stack
fr_ny = 1.0 / (2.0 * float(px))

# total angular span (radians) and constant part of normalization
Theta = float(np.deg2rad(50.0)) if Nv > 1 else 1.0
norm_const = (Nv / Theta) if Theta > 0 else 1.0     # the N/Θ part

# average angular step for splat width
if Nv > 1:
    dtheta = float(np.mean(np.diff(np.sort(angles))))
else:
    dtheta = 0.0
sin_dtheta = np.sin(abs(dtheta))

# ---------------- allocate 3D volumes ----------------
S3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)
W3 = np.zeros((Nx, Ny, Nz), dtype=np.float32)

# ---------------- loop over views with Gaussian splat ----------------
B = 0.05
for i in tqdm(range(Nv), desc="Map views -> 3D", unit="view"):
    theta_i = float(angles[i])

    S2 = S_views[i]  # (Nx,Ny), already filtered in (f_r,f_y)
    W2 = W_views[i]

    theta = float(angles[i])
    c, s = np.cos(theta), np.sin(theta)

    # 1) slice-thickness window H_ST(fz)
    fr_ny = np.abs(1.0 / (2 * px * np.cos(theta)))
    # fr_ny = 1.0 / (2.0 * float(px))
    # lim = min(B * fr_ny, np.tan(Theta) * fr_ny)
    m = np.where((np.abs(fz) <= B * fr_ny) & (np.abs(fz) <= np.tan(Theta) * fr_ny))
    # m = np.where(np.abs(fz) <= lim)
    # L = min(B * fr_ny, abs(np.tan(theta)) * fr_ny)
    Hst = np.ones_like(fz, np.float32)
    Hst[m] = 0.5 * (1 + np.cos((np.pi * fz[m])/(B * fr_ny)))
    # if L > 0:
    #     m = np.abs(fz) <= L
    #     x = fz[m] / L
    #     Hst[m] = 0.5 * (1 + np.cos(np.pi * x))

    # Hst = np.ones_like(fz, dtype=np.float32)
    # 2) x-resample (rotate fr→fx)
    # input grids
    fx_in = fx  # ≈ fr
    fx_out_for_in = fx_in * c  # where each input bin lands on the fx axis
    # build an interpolator from fx_out_for_in -> input row values
    # (vectorized: do per-row linear resample of S2,W2 from x_in to x_out grid 'fx')
    S2x = np.empty_like(S2)
    W2x = np.empty_like(W2)
    for iy in range(Ny):
        S2x[:, iy] = np.interp(fx, fx_out_for_in, S2[:, iy], left=0, right=0)
        W2x[:, iy] = np.interp(fx, fx_out_for_in, W2[:, iy], left=0, right=0)

    # 3) z-splat (finite width) with H_ST
    # center line in z: fz_line = fr*sinθ ≈ fx_in*sinθ; after x-resample, use fx grid mapped from fr: fz_line = (fx/c)*s
    fz_line = (fx / max(c, 1e-6)) * s
    # Gaussian sigma (small) – or use your previous rule based on angular step
    sigma = np.maximum(0.5 * dfz, 1 * np.abs(fx) * np.sin(abs(dtheta)))
    # sigma = 0.05 # example
    for ix in range(Nx):
        # --- choose 1 of the 3 interpolation kernels below ---
        w = gaussian_weights(float(fz_line[ix]), fz, sigma=float(sigma[ix]))                     # ← Gaussian
        # w = gaussian_weights(float(fz_line[ix]), fz, sigma=float(sigma))
        # w = sinc_weights(float(fz_line[ix]), fz, width=dfz, truncate=5.0)                        # ← Sinc
        # w = triangular_weights(float(fz_line[ix]), fz, width= 2)
        S3[ix, :, :] += S2x[ix, :, None] * w[None, :] * Hst[None, :]
        W3[ix, :, :] += W2x[ix, :, None] * w[None, :] * (Hst[None, :]**2)

# ---------------- visualize ----------------
iy0 = nearest_idx(fy, 0.0)
iz0 = nearest_idx(fz, 0.0)

# in-plane (fz≈0)
S_xy = np.abs(S3[:, :, iz0])
W_xy = W3[:, :, iz0]
show_2d(S_xy, fy, fx, r"$f_y$ (mm$^{-1}$)", r"$f_x$ (mm$^{-1}$)",
        "In-plane MTF magnitude (fz≈0)")
show_2d(W_xy, fy, fx, r"$f_y$ (mm$^{-1}$)", r"$f_x$ (mm$^{-1}$)",
        "In-plane Wiener spectrum (fz≈0)")

# in-depth (fy≈0)
S_xz = (np.abs(S3[:, iy0, :])/np.max(np.abs(S3[:, iy0, :]))).T
W_xz = W3[:, iy0, :].T
show_2d(S_xz, fx, fz, r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
        "In-depth MTF magnitude (fy≈0)", aspect=2.0, mode="in-depth")
show_2d(W_xz, fx, fz, r"$f_x$ (mm$^{-1}$)", r"$f_z$ (mm$^{-1}$)",
        "In-depth Wiener spectrum (fy≈0)", aspect=2.0, mode="in-depth")

print("end")