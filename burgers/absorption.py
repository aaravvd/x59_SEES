from __future__ import annotations

import numpy as np

from .standard_atmosphere import GAMMA_AIR, R_AIR


PRANDTL_AIR = 0.71
SUTHERLAND_REF_T = 273.15
SUTHERLAND_REF_MU = 1.716e-5
SUTHERLAND_C = 110.4


def dynamic_viscosity_sutherland(temperature_K: float) -> float:
    """Dynamic viscosity of air in Pa s using Sutherland's law."""

    t = float(temperature_K)
    return float(SUTHERLAND_REF_MU * (t / SUTHERLAND_REF_T) ** 1.5 * (SUTHERLAND_REF_T + SUTHERLAND_C) / (t + SUTHERLAND_C))


def cp_air(gamma: float = GAMMA_AIR) -> float:
    return float(gamma * R_AIR / (gamma - 1.0))


def thermal_conductivity_from_prandtl(temperature_K: float, gamma: float = GAMMA_AIR, prandtl: float = PRANDTL_AIR) -> float:
    mu = dynamic_viscosity_sutherland(temperature_K)
    return float(mu * cp_air(gamma) / float(prandtl))


def sound_diffusivity_m2_s(
    temperature_K: float,
    density_kg_m3: float,
    gamma: float = GAMMA_AIR,
    bulk_viscosity_Pa_s: float = 0.0,
) -> float:
    """Return thermoviscous sound diffusivity delta in m^2/s.

    delta = [4/3 mu + mu_bulk + kappa (gamma - 1) / cp] / rho.
    Molecular relaxation is intentionally not hidden in bulk viscosity here.
    """

    mu = dynamic_viscosity_sutherland(temperature_K)
    kappa = thermal_conductivity_from_prandtl(temperature_K, gamma)
    cp = cp_air(gamma)
    numerator = (4.0 / 3.0) * mu + float(bulk_viscosity_Pa_s) + kappa * (gamma - 1.0) / cp
    return float(numerator / max(float(density_kg_m3), 1.0e-12))


def thermoviscous_coefficient_s2_m(
    temperature_K: float,
    density_kg_m3: float,
    sound_speed_m_s: float,
    gamma: float = GAMMA_AIR,
    bulk_viscosity_Pa_s: float = 0.0,
) -> float:
    """Coefficient multiplying d2p/dtau2 in dp/dr equation, units s^2/m."""

    delta = sound_diffusivity_m2_s(temperature_K, density_kg_m3, gamma, bulk_viscosity_Pa_s)
    return float(delta / (2.0 * float(sound_speed_m_s) ** 3))


def frequency_absorption_np_per_m(
    frequency_hz: np.ndarray,
    temperature_K: float,
    density_kg_m3: float,
    sound_speed_m_s: float,
    gamma: float = GAMMA_AIR,
    bulk_viscosity_Pa_s: float = 0.0,
) -> np.ndarray:
    """Small-signal thermoviscous alpha ~= nu_tv omega^2, Np/m."""

    nu = thermoviscous_coefficient_s2_m(temperature_K, density_kg_m3, sound_speed_m_s, gamma, bulk_viscosity_Pa_s)
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    return nu * omega * omega


