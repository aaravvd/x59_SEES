from __future__ import annotations

import numpy as np


def rusanov_nonlinear_step(p: np.ndarray, dtau: float, dr: float, k_nl: float) -> tuple[np.ndarray, float]:
    """Finite-volume step for dp/dr = k p dp/dtau using flux F=-0.5 k p^2."""

    pressure = np.asarray(p, dtype=float)
    flux = -0.5 * float(k_nl) * pressure * pressure
    a = float(abs(k_nl) * np.nanmax(np.abs(pressure)))
    f_half = 0.5 * (flux[:-1] + flux[1:]) - 0.5 * a * (pressure[1:] - pressure[:-1])
    out = pressure.copy()
    out[1:-1] -= float(dr) / float(dtau) * (f_half[1:] - f_half[:-1])
    out[0] = pressure[0]
    out[-1] = pressure[-1]
    return out, a


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_sign = a * b > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _muscl_spatial_operator(
    pressure: np.ndarray,
    dtau: float,
    k_nl: float,
) -> tuple[np.ndarray, float]:
    """Return a TVD MUSCL-Rusanov approximation to ``-dF/dtau``."""

    p = np.asarray(pressure, dtype=float)
    slope = np.zeros_like(p)
    slope[1:-1] = _minmod(p[1:-1] - p[:-2], p[2:] - p[1:-1])
    left = p[:-1] + 0.5 * slope[:-1]
    right = p[1:] - 0.5 * slope[1:]
    flux_left = -0.5 * float(k_nl) * left * left
    flux_right = -0.5 * float(k_nl) * right * right
    interface_speed = np.maximum(np.abs(float(k_nl) * left), np.abs(float(k_nl) * right))
    flux = 0.5 * (flux_left + flux_right) - 0.5 * interface_speed * (right - left)
    derivative = np.zeros_like(p)
    derivative[1:-1] = -(flux[1:] - flux[:-1]) / float(dtau)
    return derivative, float(np.max(interface_speed)) if interface_speed.size else 0.0


def muscl_rusanov_nonlinear_step(
    p: np.ndarray,
    dtau: float,
    dr: float,
    k_nl: float,
) -> tuple[np.ndarray, float]:
    """Second-order TVD MUSCL spatial update with SSP-RK2 range stepping.

    The minmod limiter preserves monotonicity near shocks while greatly
    reducing the grid-dependent numerical diffusion of the legacy first-order
    piecewise-constant Rusanov update.
    """

    pressure = np.asarray(p, dtype=float)
    operator_0, speed_0 = _muscl_spatial_operator(pressure, dtau, k_nl)
    stage_1 = pressure + float(dr) * operator_0
    stage_1[0] = pressure[0]
    stage_1[-1] = pressure[-1]
    operator_1, speed_1 = _muscl_spatial_operator(stage_1, dtau, k_nl)
    out = 0.5 * pressure + 0.5 * (stage_1 + float(dr) * operator_1)
    out[0] = pressure[0]
    out[-1] = pressure[-1]
    return out, max(speed_0, speed_1)


def diffusion_step(p: np.ndarray, dtau: float, dr: float, coefficient_s2_m: float) -> np.ndarray:
    """Explicit diffusion step for dp/dr = nu d2p/dtau2."""

    pressure = np.asarray(p, dtype=float)
    out = pressure.copy()
    lam = float(coefficient_s2_m) * float(dr) / (float(dtau) * float(dtau))
    out[1:-1] += lam * (pressure[2:] - 2.0 * pressure[1:-1] + pressure[:-2])
    out[0] = pressure[0]
    out[-1] = pressure[-1]
    return out


def stable_step_limit(dtau: float, p: np.ndarray, k_nl: float, diffusion_coeff_s2_m: float, cfl: float) -> float:
    a = float(abs(k_nl) * max(np.nanmax(np.abs(p)), 1.0e-12))
    adv = np.inf if a <= 0.0 else float(cfl) * float(dtau) / a
    diff = np.inf if diffusion_coeff_s2_m <= 0.0 else 0.45 * float(dtau) ** 2 / float(diffusion_coeff_s2_m)
    return float(min(adv, diff))

