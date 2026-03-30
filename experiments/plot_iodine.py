import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------
# Load results
# -----------------------------
data = np.load("opt_iodine_dprime_3D_coarse_for_d.npz", allow_pickle=True)
# or: opt_cyst_mass_dprime_3D.npz

dprime = data["dprime"]   # (N_LE, N_HE, N_w)
CDE    = data["CDE"]
kappa  = data["kappa"]

LE_kVs = data["LE_kVs"]
HE_kVs = data["HE_kVs"]
w_vals = data["w_vals"]

N_LE, N_HE, N_w = dprime.shape

LE_grid, HE_grid = np.meshgrid(LE_kVs, HE_kVs, indexing="ij")


# -----------------------------
# Optimize over w
# -----------------------------
iw_opt = np.argmax(dprime, axis=2)

dprime_opt = np.zeros((N_LE, N_HE))
CDE_opt    = np.zeros((N_LE, N_HE))
kappa_opt  = np.zeros((N_LE, N_HE))
w_opt      = np.zeros((N_LE, N_HE))

for iLE in range(N_LE):
    for jHE in range(N_HE):
        iw = iw_opt[iLE, jHE]
        dprime_opt[iLE, jHE] = dprime[iLE, jHE, iw]
        CDE_opt[iLE, jHE]    = CDE[iLE, jHE, iw]
        kappa_opt[iLE, jHE]  = kappa[iLE, jHE, iw]
        w_opt[iLE, jHE]      = w_vals[iw]


# -----------------------------
# Generic 3D surface plotter
# -----------------------------
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def plot_surface(
    Z,
    title,
    zlabel,
    cmap="cividis",
    elev=28,
    azim=135,
    vmin=None,
    vmax=None,
    show_optimum=False
):
    # Global font scaling
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize if limits provided
    norm = None
    if vmin is not None and vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    surf = ax.plot_surface(
        HE_grid,
        LE_grid,
        Z,
        cmap=cmap,
        norm=norm,
        edgecolor="black",   # Remove mesh lines
        antialiased=True
    )

    # Axis labels
    ax.set_xlabel(r"$kV_{HE}$", labelpad=10)
    ax.set_ylabel(r"$kV_{LE}$", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=8)


    # Z limits
    if vmin is not None and vmax is not None:
        ax.set_zlim(vmin, vmax)
        ax.set_zticks(np.linspace(vmin, vmax, 4))

    # Cleaner panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # View angle
    ax.view_init(elev=elev, azim=azim)

    # Optional optimum marker
    if show_optimum:
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        ax.scatter(
            HE_grid[max_idx],
            LE_grid[max_idx],
            Z[max_idx],
            color="red",
            s=60,
            depthshade=False
        )

    # Colorbar
    # cbar = fig.colorbar(surf, shrink=0.65, pad=0.08)
    # cbar.ax.tick_params(labelsize=12)
    ax.set_title(title, pad=-120)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 1) Optimal detectability
# -----------------------------
plot_surface(
    dprime_opt / np.max(dprime_opt),
    title="In-plane detectability",
    zlabel=r"$d'_{IP}$",
    vmin=0.55,
    vmax=1.0,
    cmap="magma"
)


# -----------------------------
# 2) Optimal DE contrast
# -----------------------------
plot_surface(
    CDE_opt,
    title=r"Optimal DE contrast $C_{DE}(w^*)$",
    zlabel=r"$C_{DE}$"
)


# -----------------------------
# 3) Optimal DE weight
# -----------------------------
plot_surface(
    w_opt,
    title=r"Optimal DE weight $w^*(LE, HE)$",
    zlabel=r"$w^*$",
    cmap="plasma"
)


# -----------------------------
# 4) Anatomical noise scaling
# -----------------------------
plot_surface(
    kappa_opt,
    title=r"Anatomical noise scaling $\kappa(w^*)$",
    zlabel=r"$\kappa$",
    cmap="inferno"
)
