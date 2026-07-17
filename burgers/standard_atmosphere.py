from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


G0 = 9.80665
R_AIR = 287.05287
GAMMA_AIR = 1.4
T0 = 288.15
P0 = 101325.0
LAPSE_TROPOSPHERE = -0.0065
H_TROP_PAUSE = 11000.0
T_TROP_PAUSE = 216.65


@dataclass(frozen=True)
class AtmosphereState:
    altitude_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    sound_speed_m_s: float
    gamma: float = GAMMA_AIR
    beta: float = (GAMMA_AIR + 1.0) / 2.0

    @property
    def impedance(self) -> float:
        return self.density_kg_m3 * self.sound_speed_m_s


def speed_of_sound(temperature_K: float, gamma: float = GAMMA_AIR) -> float:
    return float(np.sqrt(float(gamma) * R_AIR * float(temperature_K)))


def standard_state(altitude_m: float) -> AtmosphereState:
    h = float(max(0.0, altitude_m))
    if h <= H_TROP_PAUSE:
        temperature = T0 + LAPSE_TROPOSPHERE * h
        pressure = P0 * (temperature / T0) ** (-G0 / (LAPSE_TROPOSPHERE * R_AIR))
    else:
        t11 = T0 + LAPSE_TROPOSPHERE * H_TROP_PAUSE
        p11 = P0 * (t11 / T0) ** (-G0 / (LAPSE_TROPOSPHERE * R_AIR))
        temperature = T_TROP_PAUSE
        pressure = p11 * np.exp(-G0 * (h - H_TROP_PAUSE) / (R_AIR * temperature))
    density = pressure / (R_AIR * temperature)
    gamma = GAMMA_AIR
    return AtmosphereState(
        altitude_m=h,
        temperature_K=float(temperature),
        pressure_Pa=float(pressure),
        density_kg_m3=float(density),
        sound_speed_m_s=speed_of_sound(temperature, gamma),
        gamma=gamma,
        beta=(gamma + 1.0) / 2.0,
    )


def make_slant_profile(
    source_altitude_m: float,
    ground_altitude_m: float,
    slant_range_m: float,
    range_steps: int,
) -> pd.DataFrame:
    """Return atmosphere sampled along a straight Mach-angle slant path."""

    r = np.linspace(0.0, float(slant_range_m), int(range_steps) + 1)
    frac = r / max(float(slant_range_m), 1.0)
    altitude = float(source_altitude_m) - frac * (float(source_altitude_m) - float(ground_altitude_m))
    rows = []
    for ri, h in zip(r, altitude):
        state = standard_state(float(h))
        rows.append(
            {
                "range_m": float(ri),
                "altitude_m": state.altitude_m,
                "temperature_K": state.temperature_K,
                "pressure_Pa": state.pressure_Pa,
                "density_kg_m3": state.density_kg_m3,
                "sound_speed_m_s": state.sound_speed_m_s,
                "gamma": state.gamma,
                "beta": state.beta,
                "impedance_Pa_s_per_m": state.impedance,
            }
        )
    return pd.DataFrame(rows)


