"""
casymir.fitting
~~~~~~~~~~~~~~~

1D fitting utilities for 2D CASYMIR SignalND objects.

- Extracts profiles along the 2D frequency domaion axes (u,v)
- Fits a Lorentzian or Gaussian to the MTF profiles and a double Gaussian to the NPS profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import curve_fit


Direction = Literal["u", "v"]


@dataclass
class FitResult1D:
    kind: Literal["MTF", "NNPS"]
    model: str
    direction: Direction
    params: Optional[np.ndarray]
    cov: Optional[np.ndarray]
    x: np.ndarray
    y: np.ndarray
    y_fit: Optional[np.ndarray]
    extra: Dict[str, Any]


@dataclass
class FitResultND:
    mtf: FitResult1D
    nnps: FitResult1D


def lorentzian_mtf(x, a, b, c):
    """Double Lorentzian MTF model."""
    return a / (1 + (x / b) ** 2) + (1 - a) / (1 + (x / c) ** 2)


def gaussian_mtf(x, a, b, c):
    """Double Gaussian MTF model."""
    return a * np.exp(-(x / b) ** 2) + (1 - a) * np.exp(-(x / c) ** 2)


def gaussian_nnps(x, a, b, c, d):
    """Double Gaussian NNPS model."""
    return a * np.exp(-(x / b) ** 2) + c * np.exp(-(x / d) ** 2)


def mask_freq(
    x: np.ndarray,
    y: np.ndarray,
    f_ny: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict arrays to 0 <= f <= f_NY for fitting purposes
    """
    mask = (x >= 0) & (x <= f_ny)
    return x[mask], y[mask]


def _safe_curve_fit(
    func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    p0: Optional[list] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        popt, pcov = curve_fit(func, x, y, p0=p0)
        return popt, pcov
    except Exception:
        return None, None


def axis_cut_1d(
    A: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    direction: Direction,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central axis cut through a 2D spectrum.

    Convention:
        u -> vary fx, fix fy = 0
        v -> vary fy, fix fx = 0

    Assumes zero frequency at index N//2.
    """
    A = np.asarray(A)
    cx = len(fx) // 2
    cy = len(fy) // 2

    if direction == "u":
        return fx, A[:, cy]
    else:
        return fy, A[cx, :]


def normalize_mtf(y: np.ndarray) -> np.ndarray:
    """
    Normalize MTF to unity at DC.
    """
    y = np.asarray(y)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y

    dc = y[finite][0]
    if dc != 0:
        return y / dc

    mx = np.nanmax(y)
    return y / mx if mx != 0 else y


def fit_nd(
    signal_nd,
    direction: Direction = "u",
    type_mtf: str = "lorentzian",
) -> FitResultND:
    if signal_nd.mtf is None or signal_nd.nnps is None:
        raise ValueError("SignalND.mtf and SignalND.nnps must be defined.")

    fx = signal_nd.fx
    fy = signal_nd.fy

    # Extract central axis cuts
    x_mtf, y_mtf = axis_cut_1d(signal_nd.mtf, fx, fy, direction)
    x_nps, y_nps = axis_cut_1d(signal_nd.nnps, fx, fy, direction)

    # Nyquist frequency definition here assumes the model is defined up to 2*f_NY!
    f_ny = 0.5 * np.max(x_mtf)

    # Restrict to 0, f_NY
    x_mtf, y_mtf = mask_freq(x_mtf, y_mtf, f_ny)
    x_nps, y_nps = mask_freq(x_nps, y_nps, f_ny)

    # MTF normalization
    y_mtf_n = normalize_mtf(y_mtf)

    # Inital coefficient guesses
    p0_mtf = [
        0.5,
        max(np.median(x_mtf), 1e-6),
        max(2 * np.median(x_mtf), 1e-6),
    ]

    p0_nps = [
        float(np.nanmax(y_nps)),
        max(np.median(x_nps), 1e-6),
        float(np.nanmax(y_nps) * 0.5),
        max(2 * np.median(x_nps), 1e-6),
    ]

    if type_mtf == "lorentzian":
        func_mtf = lorentzian_mtf
    elif type_mtf == "gaussian":
        func_mtf = gaussian_mtf
    else:
        raise ValueError("Unsupported MTF fit type: {}".format(type_mtf))

    popt_mtf, pcov_mtf = _safe_curve_fit(func_mtf, x_mtf, y_mtf_n, p0_mtf)
    popt_nps, pcov_nps = _safe_curve_fit(gaussian_nnps, x_nps, y_nps, p0_nps)
    yfit_mtf = func_mtf(x_mtf, *popt_mtf) if popt_mtf is not None else None
    yfit_nps = gaussian_nnps(x_nps, *popt_nps) if popt_nps is not None else None

    mtf_res = FitResult1D(
        kind="MTF",
        model=type_mtf,
        direction=direction,
        params=popt_mtf,
        cov=pcov_mtf,
        x=x_mtf,
        y=y_mtf_n,
        y_fit=yfit_mtf,
        extra={},
    )

    nnps_res = FitResult1D(
        kind="NNPS",
        model="gaussian",
        direction=direction,
        params=popt_nps,
        cov=pcov_nps,
        x=x_nps,
        y=y_nps,
        y_fit=yfit_nps,
        extra={},
    )

    return FitResultND(mtf=mtf_res, nnps=nnps_res)


def fit_stack(
    stack,
    direction: Direction = "u",
    type_mtf: str = "lorentzian",
) -> list[FitResultND]:
    """
    Fit all views in a SignalStack.
    """
    return [fit_nd(sig, direction=direction, type_mtf=type_mtf) for _, sig in stack.iter_views()]
