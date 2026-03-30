import numpy as np

def gaussian_task_spectrum(FX, FY, FZ, sigma_mm):
    return np.exp(
        -(np.pi * sigma_mm)**2 * (FX**2 + FY**2 + FZ**2)
    )

def contrast_weighted_task(FX, FY, FZ, sigma_mm, C):
    return C * gaussian_task_spectrum(FX, FY, FZ, sigma_mm)