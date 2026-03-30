import numpy as np
import matplotlib.pyplot as plt

from experiments.run_pcm import run_pcm_dbt

def show_2d(
    A,
    x_axis,
    y_axis,
    xlabel,
    ylabel,
    title,
    *,
    aspect="equal",
    cmap="gray",
    mode="in-plane",
):
    """
    Visualize central slices of a 3D frequency-domain quantity,
    reproducing the original CASYMIR plotting behavior.

    Parameters
    ----------
    A : ndarray
        2D or 3D array (fx, fy) or (fx, fy, fz) slice already extracted.
    x_axis, y_axis : ndarray
        Frequency axes corresponding to A.
    mode : {'in-plane', 'in-depth'}
        Controls slicing and axis extents.
    """

    dims = A.shape
    idx_x = dims[0] // 2
    idx_y = dims[1] // 2

    plt.figure(figsize=(6, 5))

    if mode == "in-plane":
        # Central crop in x and y
        x0 = idx_x - idx_x // 2
        x1 = idx_x + idx_x // 2
        y0 = idx_y - idx_y // 2
        y1 = idx_y + idx_y // 2

        extent = [
            x_axis.min() / 2,
            x_axis.max() / 2,
            y_axis.min() / 2,
            y_axis.max() / 2,
        ]

        plt.imshow(
            A[x0:x1, y0:y1],
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect=aspect,
        )

    else:  # in-depth
        x0 = idx_x - idx_x // 2
        x1 = idx_x + idx_x // 2

        extent = [
            x_axis.min() / 2,
            x_axis.max() / 2,
            y_axis.min(),
            y_axis.max(),
        ]

        plt.imshow(
            A[x0:x1, :],
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect=aspect,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- geometry ---
N = 25
angles = np.linspace(np.deg2rad(-25), np.deg2rad(25), N)

Nz = 64
dz = 1.0
fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)

system_path = "C:\\Users\\Z639176\\Documents\\Projects\\casymir_v2\\casymir_public\\example_dbt_v2.yaml"

# --- run PCM ---
out = run_pcm_dbt(
    kV=28,
    mAs=4,
    system_yaml=system_path,
    angles_rad=angles,
    fz=fz,
)

print("q0 per view (photons / mm^2):")
print(out["q0_per_view"])
print("mean q0:", out["q0_mean"])

print("\ndak per view (µGy at detector entrance):")
print(out["dak_per_view"])
print("mean dak:", out["dak_mean"])

MTF3 = out["MTF3"]
W3   = out["W3"]
fx   = out["fx"]
fy   = out["fy"]
fz   = out["fz"]

iz0 = len(fz) // 2
iy0 = len(fy) // 2

# In-plane
show_2d(
    np.abs(MTF3[:, :, iz0]),
    fy, fx,
    r"$f_y$ (mm$^{-1}$)",
    r"$f_x$ (mm$^{-1}$)",
    "In-plane MTF magnitude ($f_z = 0$)",
)

show_2d(
    W3[:, :, iz0],
    fy, fx,
    r"$f_y$ (mm$^{-1}$)",
    r"$f_x$ (mm$^{-1}$)",
    "In-plane Wiener spectrum ($f_z = 0$)",
    cmap="inferno",
)

# In-depth
show_2d(
    np.abs(MTF3[:, iy0, :]).T,
    fx, fz,
    r"$f_x$ (mm$^{-1}$)",
    r"$f_z$ (mm$^{-1}$)",
    "In-depth MTF magnitude ($f_y = 0$)",
    aspect=2.0,
    mode="in-depth",
)

show_2d(
    W3[:, iy0, :].T,
    fx, fz,
    r"$f_x$ (mm$^{-1}$)",
    r"$f_z$ (mm$^{-1}$)",
    "In-depth Wiener spectrum ($f_y = 0$)",
    aspect=2.0,
    mode="in-depth",
    cmap="inferno",
)

print("end")