import numpy as np
import matplotlib.pyplot as plt

data = np.load("opt_iodine_dprime_3D.npz", allow_pickle=True)

dprime = data["dprime"]    # shape: (N_LE, N_HE, N_w)
CDE    = data["CDE"]
kappa  = data["kappa"]

LE_kVs = data["LE_kVs"]
HE_kVs = data["HE_kVs"]
w_vals = data["w_vals"]

N_LE, N_HE, N_w = dprime.shape

def plot_C_vs_w(iLE_list, jHE_list):
    plt.figure(figsize=(7, 5))

    for iLE in iLE_list:
        for jHE in jHE_list:
            C = CDE[iLE, jHE, :]
            plt.plot(
                w_vals, C,
                label=f"LE={LE_kVs[iLE]}, HE={HE_kVs[jHE]}"
            )

    plt.xlabel("w")
    plt.ylabel(r"$C_{\mathrm{DE}}$")
    plt.title(r"DE contrast $C_{\mathrm{DE}}(w)$")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_relative_dprime_vs_w(iLE_list, jHE_list):
    plt.figure(figsize=(7, 5))

    for iLE in iLE_list:
        for jHE in jHE_list:
            dp = dprime[iLE, jHE, :]
            dp_rel = dp

            plt.plot(
                w_vals, dp_rel,
                label=f"LE={LE_kVs[iLE]}, HE={HE_kVs[jHE]}"
            )

    plt.xlabel("w")
    plt.ylabel(r"$d'(w)$")
    plt.title("Detectability vs DE weight")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_optimal_w_surface():
    w_opt = np.zeros((N_LE, N_HE))

    for iLE in range(N_LE):
        for jHE in range(N_HE):
            iw = np.argmax(dprime[iLE, jHE, :])
            w_opt[iLE, jHE] = w_vals[iw]

    plt.figure(figsize=(7, 5))
    plt.imshow(
        w_opt,
        origin="lower",
        aspect="auto",
        extent=[
            HE_kVs.min(), HE_kVs.max(),
            LE_kVs.min(), LE_kVs.max()
        ]
    )
    plt.colorbar(label=r"$w^*$")
    plt.xlabel("HE tube potential (kVp)")
    plt.ylabel("LE tube potential (kVp)")
    plt.title("Optimal DE weight $w^*$")
    plt.tight_layout()
    plt.show()

    return w_opt

def plot_kappa_vs_w(iLE, jHE):
    plt.figure(figsize=(6, 4))
    plt.plot(w_vals, kappa[iLE, jHE, :])
    plt.xlabel("w")
    plt.ylabel(r"$\kappa(w)$")
    plt.title(f"Anatomical noise scaling | LE={LE_kVs[iLE]}, HE={HE_kVs[jHE]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_optimal_dprime_surface():
    dprime_opt = np.zeros((N_LE, N_HE))
    w_at_opt   = np.zeros((N_LE, N_HE))

    for iLE in range(N_LE):
        for jHE in range(N_HE):
            iw = np.argmax(dprime[iLE, jHE, :])
            dprime_opt[iLE, jHE] = dprime[iLE, jHE, iw]
            w_at_opt[iLE, jHE]   = w_vals[iw]

    dprime_opt = dprime_opt / np.max(dprime_opt)

    plt.figure(figsize=(7, 5))
    plt.imshow(
        dprime_opt,
        origin="lower",
        aspect="auto",
        extent=[
            HE_kVs.min(), HE_kVs.max(),
            LE_kVs.min(), LE_kVs.max()
        ]
    )
    plt.colorbar(label=r"$d'_{\mathrm{max}}$")
    plt.xlabel("HE tube potential (kVp)")
    plt.ylabel("LE tube potential (kVp)")
    # plt.title(r"Optimal detectability $d'_{\mathrm{max}}$ (over $w$)")
    plt.tight_layout()
    plt.show()

    return dprime_opt, w_at_opt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_optimal_dprime_surface_3d():
    dprime_opt = np.zeros((N_LE, N_HE))

    for iLE in range(N_LE):
        for jHE in range(N_HE):
            dprime_opt[iLE, jHE] = np.max(dprime[iLE, jHE, :])
    dprime_opt = dprime_opt / np.max(dprime_opt)
    HE_grid, LE_grid = np.meshgrid(HE_kVs, LE_kVs)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        LE_grid, HE_grid, dprime_opt,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.4,
        antialiased=True
    )

    ax.set_xlabel("Tube potential – LE (kVp)")
    ax.set_ylabel("Tube potential – HE (kVp)")
    ax.set_zlabel(r"$d'_{\max}$")

    ax.set_title(r"Optimal detectability surface $d'_{\max}$(LE, HE)")

    fig.colorbar(surf, shrink=0.6, aspect=18, label=r"$d'_{\max}$")

    # --- viewing angle (very similar to SPIE figures) ---
    ax.view_init(elev=25, azim=-135)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    return dprime_opt


iLE_list = [0, 2, 4]
jHE_list = [0, 2, 4]

plot_C_vs_w(iLE_list, jHE_list)
plot_relative_dprime_vs_w(iLE_list, jHE_list)
w_opt = plot_optimal_w_surface()

plot_kappa_vs_w(iLE=0, jHE=4)
dprime_opt, w_at_opt = plot_optimal_dprime_surface()
plot_optimal_dprime_surface_3d()

print("end")
