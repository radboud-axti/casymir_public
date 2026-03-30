from casymir import casymir, processes_2d, processes_3d

import pickle, numpy as np
import matplotlib.pyplot as plt

def nearest_idx(axis, val):
    return int(np.argmin(np.abs(axis - val)))


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


with open("projections.pkl", "rb") as f:
    stack = pickle.load(f)

Nz = 64
dz = 1.0
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
px = getattr(stack, "px_size", None)

vol3d = processes_3d.map_stack_to_volume(stack, fz, kernel="gaussian", B=0.05, Theta_rad=np.deg2rad(50.0),
                                         px_size_mm=px, spoke_density_normalize=False)

fx = vol3d.axes[0]
fy = vol3d.axes[1]
iy0 = nearest_idx(fy, 0.0)
iz0 = nearest_idx(fz, 0.0)

S3 = vol3d.S
W3 = vol3d.W

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
