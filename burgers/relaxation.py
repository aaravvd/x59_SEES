from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


REFERENCE_PRESSURE_PA = 101325.0
REFERENCE_TEMPERATURE_K = 293.15
TRIPLE_POINT_TEMPERATURE_K = 273.16
DB_PER_NEPER_AMPLITUDE = 20.0 / np.log(10.0)
RELAXATION_REFERENCE = (
    "ISO 9613-1:1993 Eq. 3-5 / Bass, Sutherland, Zuckerwar, Blackstock, "
    "Hester 1995 atmospheric absorption formula; applied here as an "
    "operator-split frequency-domain layer filter."
)


@dataclass(frozen=True)
class MolecularRelaxationStatus:
    enabled: bool = False
    implemented: bool = False
    model_type: str = "frequency_domain_layer_filter"
    formula_source: str = RELAXATION_REFERENCE
    relative_humidity: float = 0.3
    include_oxygen_relaxation: bool = True
    include_nitrogen_relaxation: bool = True
    include_classical_absorption: bool = True
    pad_factor: int = 2
    filter_max_alpha_dr: float | None = 50.0
    frequency_min_Hz: float = 0.0
    frequency_max_Hz: float = 0.0
    relaxation_frequency_O2_Hz: float = np.nan
    relaxation_frequency_N2_Hz: float = np.nan
    water_vapor_mole_fraction: float = np.nan
    max_alpha_np_per_m: float = 0.0
    max_alpha_dr: float = 0.0
    min_filter_gain: float = 1.0
    fallback_mode_used: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: str = (
        "Frequency-domain atmospheric absorption operator. This is not the "
        "time-domain O2/N2 relaxation-variable system used in validated sBOOM/PCBoom-class tools."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "implemented": self.implemented,
            "formula_source": self.formula_source,
            "relaxation_model_type": self.model_type,
            "O2_enabled": self.include_oxygen_relaxation,
            "N2_enabled": self.include_nitrogen_relaxation,
            "classical_absorption_enabled": self.include_classical_absorption,
            "humidity_value": self.relative_humidity,
            "pad_factor": self.pad_factor,
            "filter_max_alpha_dr": self.filter_max_alpha_dr,
            "frequency_range_Hz": [self.frequency_min_Hz, self.frequency_max_Hz],
            "coefficient_table": {
                "water_vapor_mole_fraction": self.water_vapor_mole_fraction,
                "relaxation_frequency_O2_Hz": self.relaxation_frequency_O2_Hz,
                "relaxation_frequency_N2_Hz": self.relaxation_frequency_N2_Hz,
                "max_alpha_np_per_m": self.max_alpha_np_per_m,
                "max_alpha_dr": self.max_alpha_dr,
                "min_filter_gain": self.min_filter_gain,
            },
            "unit_checks": {
                "alpha_dB_per_m_to_alpha_np_per_m": "alpha_np_per_m = alpha_dB_per_m / (20/log(10))",
                "filter_gain": "exp(-alpha_np_per_m * dr_m), dimensionless",
                "relaxation_frequencies": "Hz",
                "water_vapor_mole_fraction": "dimensionless partial-pressure ratio",
            },
            "warnings": self.warnings,
            "fallback_mode_used": self.fallback_mode_used,
            "notes": self.notes,
        }


def saturation_vapor_pressure_pa(T_K: float | np.ndarray) -> np.ndarray:
    """ISO 9613-1 saturation vapor pressure ratio converted to Pa."""

    T = np.asarray(T_K, dtype=float)
    exponent = -6.8346 * (TRIPLE_POINT_TEMPERATURE_K / T) ** 1.261 + 4.6151
    return REFERENCE_PRESSURE_PA * np.power(10.0, exponent)


def water_vapor_mole_fraction(T_K: float, relative_humidity: float, pressure_Pa: float) -> float:
    """Return water-vapor mole fraction from relative humidity and pressure.

    The ISO/Bass absorption formula uses the molar concentration of water vapor
    as a dimensionless partial-pressure ratio. Values are clipped to defensible
    numerical bounds rather than allowed to produce negative relaxation rates.
    """

    rh = float(np.clip(relative_humidity, 0.0, 1.0))
    pressure = max(float(pressure_Pa), 1.0)
    h = rh * float(saturation_vapor_pressure_pa(float(T_K))) / pressure
    return float(np.clip(h, 0.0, 0.2))


def relaxation_frequencies_air(
    T_K: float,
    pressure_Pa: float,
    relative_humidity: float,
) -> dict[str, float]:
    """Compute ISO/Bass oxygen and nitrogen relaxation frequencies in Hz."""

    T = float(T_K)
    pressure_ratio = max(float(pressure_Pa), 1.0) / REFERENCE_PRESSURE_PA
    h = water_vapor_mole_fraction(T, relative_humidity, pressure_Pa)
    fr_o = pressure_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pressure_ratio * (T / REFERENCE_TEMPERATURE_K) ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / REFERENCE_TEMPERATURE_K) ** (-1.0 / 3.0) - 1.0))
    )
    return {
        "frO_Hz": float(max(fr_o, 1.0e-12)),
        "frN_Hz": float(max(fr_n, 1.0e-12)),
        "water_vapor_mole_fraction": h,
    }


def atmospheric_absorption_coefficient(
    f_Hz: np.ndarray,
    T_K: float,
    pressure_Pa: float,
    relative_humidity: float,
    include_classical: bool = True,
    include_molecular: bool = True,
    include_oxygen: bool = True,
    include_nitrogen: bool = True,
) -> dict[str, np.ndarray | float]:
    """Return atmospheric absorption coefficient for pressure amplitude.

    The returned `alpha_np_per_m` is for pressure amplitude attenuation, so the
    layer filter is `exp(-alpha_np_per_m * dr_m)`.
    """

    f = np.asarray(f_Hz, dtype=float)
    f_abs = np.abs(f)
    T = float(T_K)
    pressure = max(float(pressure_Pa), 1.0)
    pressure_ratio = pressure / REFERENCE_PRESSURE_PA
    temperature_ratio = T / REFERENCE_TEMPERATURE_K
    freqs = relaxation_frequencies_air(T, pressure, relative_humidity)
    fr_o = freqs["frO_Hz"]
    fr_n = freqs["frN_Hz"]

    classical = np.zeros_like(f_abs)
    if include_classical:
        classical = 1.84e-11 * (REFERENCE_PRESSURE_PA / pressure) * np.sqrt(temperature_ratio) * np.ones_like(f_abs)

    oxygen = np.zeros_like(f_abs)
    nitrogen = np.zeros_like(f_abs)
    if include_molecular:
        common = temperature_ratio ** -2.5
        if include_oxygen:
            oxygen = common * 0.01275 * np.exp(-2239.1 / T) / (fr_o + f_abs * f_abs / fr_o)
        if include_nitrogen:
            nitrogen = common * 0.1068 * np.exp(-3352.0 / T) / (fr_n + f_abs * f_abs / fr_n)

    alpha_db_per_m = 8.686 * f_abs * f_abs * (classical + oxygen + nitrogen)
    alpha_np_per_m = np.maximum(alpha_db_per_m / DB_PER_NEPER_AMPLITUDE, 0.0)
    alpha_np_per_m = np.where(f_abs <= 0.0, 0.0, alpha_np_per_m)
    return {
        "alpha_np_per_m": alpha_np_per_m,
        "alpha_dB_per_m": alpha_db_per_m,
        "classical_term": classical,
        "oxygen_term": oxygen,
        "nitrogen_term": nitrogen,
        **freqs,
    }


def _config_value(config: dict[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def apply_molecular_relaxation_filter(
    p_tau: np.ndarray,
    dt: float,
    layer: dict[str, float] | None = None,
    dr: float = 0.0,
    config: dict[str, Any] | Any | None = None,
    enabled: bool | None = None,
) -> tuple[np.ndarray, MolecularRelaxationStatus]:
    """Apply an operator-split ISO/Bass frequency-domain absorption filter."""

    cfg = {} if config is None else config
    is_enabled = bool(_config_value(cfg, "molecular_relaxation_enabled", False) if enabled is None else enabled)
    relative_humidity = float(_config_value(cfg, "relative_humidity", 0.3))
    include_o2 = bool(_config_value(cfg, "include_oxygen_relaxation", True))
    include_n2 = bool(_config_value(cfg, "include_nitrogen_relaxation", True))
    include_classical = bool(_config_value(cfg, "include_classical_absorption", True))
    pad_factor = int(max(_config_value(cfg, "pad_factor", 2), 1))
    cap_value = _config_value(cfg, "filter_max_alpha_dr", 50.0)
    filter_cap = None if cap_value in {None, "none", "None"} else float(cap_value)

    p = np.asarray(p_tau, dtype=float)
    if not is_enabled:
        return p.copy(), MolecularRelaxationStatus(
            enabled=False,
            implemented=True,
            relative_humidity=relative_humidity,
            include_oxygen_relaxation=include_o2,
            include_nitrogen_relaxation=include_n2,
            include_classical_absorption=include_classical,
            pad_factor=pad_factor,
            filter_max_alpha_dr=filter_cap,
            notes="Molecular relaxation operator is implemented but disabled for this run.",
        )

    warnings: list[str] = []
    if layer is None:
        layer = {}
        warnings.append("No layer state supplied; standard sea-level fallback was used.")
    T_K = float(layer.get("temperature_K", REFERENCE_TEMPERATURE_K))
    pressure_Pa = float(layer.get("pressure_Pa", REFERENCE_PRESSURE_PA))
    if T_K < 253.15 or T_K > 323.15:
        warnings.append("Temperature is outside the ISO 9613-1 common tabulated range; formula is extrapolated.")
    if pressure_Pa < 20000.0:
        warnings.append("Pressure is in high-altitude range; ISO formula allows lower pressure use but validation is limited.")
    if p.size < 4 or float(dt) <= 0.0 or float(dr) <= 0.0:
        warnings.append("Degenerate waveform, time step, or layer thickness; waveform returned unchanged.")
        return p.copy(), MolecularRelaxationStatus(
            enabled=True,
            implemented=True,
            relative_humidity=relative_humidity,
            include_oxygen_relaxation=include_o2,
            include_nitrogen_relaxation=include_n2,
            include_classical_absorption=include_classical,
            pad_factor=pad_factor,
            filter_max_alpha_dr=filter_cap,
            warnings=warnings,
        )

    n = int(p.size)
    n_fft = int(2 ** np.ceil(np.log2(max(n * pad_factor, n))))
    padded = np.zeros(n_fft, dtype=float)
    start = (n_fft - n) // 2
    padded[start : start + n] = p
    freqs = np.fft.rfftfreq(n_fft, d=float(dt))
    coeffs = atmospheric_absorption_coefficient(
        freqs,
        T_K=T_K,
        pressure_Pa=pressure_Pa,
        relative_humidity=relative_humidity,
        include_classical=include_classical,
        include_molecular=include_o2 or include_n2,
        include_oxygen=include_o2,
        include_nitrogen=include_n2,
    )
    alpha = np.asarray(coeffs["alpha_np_per_m"], dtype=float)
    alpha_dr = alpha * float(dr)
    if filter_cap is not None:
        alpha_dr = np.minimum(alpha_dr, filter_cap)
    gain = np.exp(-alpha_dr)
    filtered = np.fft.irfft(np.fft.rfft(padded) * gain, n=n_fft)
    out = filtered[start : start + n]
    out[np.abs(out) < 1.0e-300] = 0.0
    status = MolecularRelaxationStatus(
        enabled=True,
        implemented=True,
        relative_humidity=relative_humidity,
        include_oxygen_relaxation=include_o2,
        include_nitrogen_relaxation=include_n2,
        include_classical_absorption=include_classical,
        pad_factor=pad_factor,
        filter_max_alpha_dr=filter_cap,
        frequency_min_Hz=float(freqs[1]) if freqs.size > 1 else 0.0,
        frequency_max_Hz=float(freqs[-1]) if freqs.size else 0.0,
        relaxation_frequency_O2_Hz=float(coeffs["frO_Hz"]),
        relaxation_frequency_N2_Hz=float(coeffs["frN_Hz"]),
        water_vapor_mole_fraction=float(coeffs["water_vapor_mole_fraction"]),
        max_alpha_np_per_m=float(np.nanmax(alpha)) if alpha.size else 0.0,
        max_alpha_dr=float(np.nanmax(alpha_dr)) if alpha_dr.size else 0.0,
        min_filter_gain=float(np.nanmin(gain)) if gain.size else 1.0,
        warnings=warnings,
    )
    return out, status


def molecular_relaxation_status(
    config: dict[str, Any] | Any,
    coefficients: dict[str, float] | None = None,
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable status record for a run or configuration."""

    coefficients = coefficients or {}
    source_info = source_info or {}
    status = MolecularRelaxationStatus(
        enabled=bool(_config_value(config, "molecular_relaxation_enabled", False)),
        implemented=True,
        relative_humidity=float(_config_value(config, "relative_humidity", 0.3)),
        include_oxygen_relaxation=bool(_config_value(config, "include_oxygen_relaxation", True)),
        include_nitrogen_relaxation=bool(_config_value(config, "include_nitrogen_relaxation", True)),
        include_classical_absorption=bool(_config_value(config, "include_classical_absorption", True)),
        pad_factor=int(_config_value(config, "pad_factor", 2)),
        filter_max_alpha_dr=float(_config_value(config, "filter_max_alpha_dr", 50.0)),
        relaxation_frequency_O2_Hz=float(coefficients.get("frO_Hz", np.nan)),
        relaxation_frequency_N2_Hz=float(coefficients.get("frN_Hz", np.nan)),
        water_vapor_mole_fraction=float(coefficients.get("water_vapor_mole_fraction", np.nan)),
    )
    data = status.to_dict()
    data["source_info"] = source_info
    return data

