import pickle, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Nz  = 128
dz  = 1.0
fz  = np.fft.fftshift(np.fft.fftfreq(Nz, d=dz)).astype(np.float32)
dfz = float(fz[1] - fz[0])

px = 0.085
fr_ny = 1.0 / (2.0 * float(px))

N = 25
angles = np.linspace(np.deg2rad(-25), np.deg2rad(25), N)
Theta = np.deg2rad(50.0)

B = 0.1

for i in tqdm(range(N), desc="Calculating Hz filter", unit="view"):
    theta_i = float(angles[i])

    theta = float(angles[i])
    c, s = np.cos(theta), np.sin(theta)

    # 1) slice-thickness window H_ST(fz)
    fr_ny = np.abs(1.0 / (2 * px * np.cos(theta)))
    # lim = min(B * fr_ny, np.tan(Theta) * fr_ny)
    m = np.where((np.abs(fz) <= B * fr_ny) & (np.abs(fz) <= np.tan(Theta) * fr_ny))
    # m = np.where(np.abs(fz) <= lim)
    # L = min(B * fr_ny, abs(np.tan(theta)) * fr_ny)
    Hst = np.zeros_like(fz, np.float32)
    Hst[m] = 0.5 * (1 + np.cos((np.pi * fz[m])/(B * fr_ny)))

    plt.plot(fz, Hst),
    plt.title(f"Hz filter for angle {np.rad2deg(theta)}")
    plt.xlabel("fz [1/mm]")
    plt.ylabel("Hz filter magnitude")
    plt.show()
