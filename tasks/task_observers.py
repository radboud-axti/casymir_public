import numpy as np

def ideal_observer_dprime2(
    MTF3, W3, FX, FY, FZ,
    W_task,
    W_bg=None,
    *,
    fmin=0.05,          # mm^-1 : low-frequency cutoff
    noise_floor_frac=1e-6
):
    """
    Ideal observer detectability index d'^2 with
    low-frequency regularization and noise floor.
    """

    # -----------------------------
    # Total noise
    # -----------------------------
    if W_bg is None:
        W_tot = W3.copy()
    else:
        W_tot = W3 + (MTF3**2) * W_bg

    # -----------------------------
    # Frequency grid
    # -----------------------------
    f = np.sqrt(FX**2 + FY**2 + FZ**2)

    # -----------------------------
    # Noise floor (data-driven)
    # -----------------------------
    positive = W_tot > 0
    if np.any(positive):
        W_floor = noise_floor_frac * np.median(W_tot[positive])
    else:
        raise RuntimeError("All noise values are zero.")

    W_tot_safe = np.maximum(W_tot, W_floor)

    # -----------------------------
    # Low-frequency cutoff mask
    # -----------------------------
    valid = f >= fmin

    # -----------------------------
    # Integration weights
    # -----------------------------
    dfx = abs(FX[1,0,0] - FX[0,0,0])
    dfy = abs(FY[0,1,0] - FY[0,0,0])
    dfz = abs(FZ[0,0,1] - FZ[0,0,0])

    # -----------------------------
    # Ideal observer integrand
    # -----------------------------
    integrand = np.zeros_like(W_tot_safe)
    integrand[valid] = (
        np.abs(MTF3[valid] * W_task[valid])**2
        / W_tot_safe[valid]
    )

    return np.sum(integrand) * dfx * dfy * dfz


def ideal_observer_dprime2_inplane(
    MTF3, W3, FX, FY, FZ,
    W_task,
    W_bg=None,
    *,
    noise_floor_frac=1e-6
):
    """
    In-plane ideal observer detectability.
    Collapses z first, then integrates in (fx, fy).
    """

    # -----------------------------
    # Total 3D noise
    # -----------------------------
    if W_bg is None:
        W_tot = W3.copy()
    else:
        W_tot = W3 + (MTF3**2) * W_bg

    # -----------------------------
    # Frequency steps
    # -----------------------------
    dfx = abs(FX[1,0,0] - FX[0,0,0])
    dfy = abs(FY[0,1,0] - FY[0,0,0])
    dfz = abs(FZ[0,0,1] - FZ[0,0,0])

    # -----------------------------
    # Collapse along z
    # -----------------------------
    # Integrated task (numerator part)
    T_int = np.sum(MTF3 * W_task, axis=2) * dfz

    # Integrated noise
    S_int = np.sum(W_tot, axis=2) * dfz

    # -----------------------------
    # Noise floor (2D now)
    # -----------------------------
    positive = S_int > 0
    if np.any(positive):
        floor = noise_floor_frac * np.median(S_int[positive])
    else:
        raise RuntimeError("All in-plane noise values are zero.")

    S_safe = np.maximum(S_int, floor)

    # -----------------------------
    # 2D ideal observer integral
    # -----------------------------
    integrand = np.abs(T_int)**2 / S_safe

    return np.sum(integrand) * dfx * dfy
