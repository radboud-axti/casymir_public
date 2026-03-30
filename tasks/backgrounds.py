import numpy as np

def powerlaw_background(FX, FY, FZ, kappa, beta, a=1.0):
    f = np.sqrt(FX**2 + FY**2 + FZ**2)
    f[f == 0] = np.min(f[f > 0])
    return kappa / ((a * f)**beta)
