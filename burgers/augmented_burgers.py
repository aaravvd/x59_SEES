from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .absorption import thermoviscous_coefficient_s2_m
from .numerics import (
    diffusion_step,
    muscl_rusanov_nonlinear_step,
    rusanov_nonlinear_step,
    stable_step_limit,
)
from .relaxation import MolecularRelaxationStatus, apply_molecular_relaxation_filter
from .standard_atmosphere import make_slant_profile


@dataclass
class SolverSettings:
    source_altitude_m: float = 16000.0
    ground_altitude_m: float = 0.0
    mach: float = 1.4
    spreading_exponent: float = 1.0
    reference_range_m: float = 1000.0
    range_steps: int = 900
    max_step_m: float = 35.0
    snapshot_count: int = 8
    nonlinear_enabled: bool = True
    thermoviscous_enabled: bool = True
    molecular_relaxation_enabled: bool = False
    geometric_spreading_enabled: bool = True
    stratification_enabled: bool = True
    artificial_viscosity_factor: float = 0.01
    smoothing_enabled: bool = False
    bulk_viscosity_Pa_s: float = 0.0
    cfl: float = 0.45
    relative_humidity: float = 0.3
    include_oxygen_relaxation: bool = True
    include_nitrogen_relaxation: bool = True
    include_classical_absorption: bool = True
    relaxation_operator: str = "frequency_domain_layer_filter"
    relaxation_reference: str = "ISO 9613-1:1993 Eq. 3-5 / Bass-Sutherland-Zuckerwar atmospheric absorption"
    pad_factor: int = 2
    filter_max_alpha_dr: float = 50.0
    humidity_sweep_enabled: bool = False
    nonlinear_scheme: str = "rusanov_first_order"


@dataclass
class SolverResult:
    tau_s: np.ndarray
    source_pressure_Pa: np.ndarray
    ground_pressure_Pa: np.ndarray
    snapshots: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def mach_angle_rad(mach: float) -> float:
    if mach <= 1.0:
        raise ValueError("Mach angle requires M > 1")
    return float(np.arcsin(1.0 / float(mach)))


def slant_range_m(source_altitude_m: float, ground_altitude_m: float, mach: float) -> float:
    return float((float(source_altitude_m) - float(ground_altitude_m)) / np.sin(mach_angle_rad(mach)))


def settings_from_config(config: dict, overrides: dict[str, Any] | None = None) -> SolverSettings:
    values = dict(config.get("propagation", {}))
    if overrides:
        values.update(overrides)
    allowed = SolverSettings.__dataclass_fields__.keys()
    return SolverSettings(**{k: values[k] for k in values if k in allowed})


class AugmentedBurgersSolver:
    """Dimensional 1D operator-split Burgers-style propagation solver."""

    def __init__(self, settings: SolverSettings):
        self.settings = settings
        self.slant_range_m = slant_range_m(settings.source_altitude_m, settings.ground_altitude_m, settings.mach)
        self.profile = make_slant_profile(
            source_altitude_m=settings.source_altitude_m,
            ground_altitude_m=settings.ground_altitude_m,
            slant_range_m=self.slant_range_m,
            range_steps=max(settings.range_steps, 2),
        )

    def _state_at_range(self, r_m: float) -> dict[str, float]:
        cols = [
            "altitude_m",
            "temperature_K",
            "pressure_Pa",
            "density_kg_m3",
            "sound_speed_m_s",
            "gamma",
            "beta",
            "impedance_Pa_s_per_m",
        ]
        values = {"range_m": float(r_m)}
        for col in cols:
            values[col] = float(np.interp(r_m, self.profile["range_m"], self.profile[col]))
        return values

    def _coefficients(self, state: dict[str, float]) -> dict[str, float]:
        rho = state["density_kg_m3"]
        c = state["sound_speed_m_s"]
        beta = state["beta"]
        k_nl = beta / (rho * c**3)
        nu_tv = thermoviscous_coefficient_s2_m(
            temperature_K=state["temperature_K"],
            density_kg_m3=rho,
            sound_speed_m_s=c,
            gamma=state["gamma"],
            bulk_viscosity_Pa_s=self.settings.bulk_viscosity_Pa_s,
        )
        return {"k_nonlinear_1_per_Pa_m": float(k_nl), "nu_thermoviscous_s2_per_m": float(nu_tv)}

    def propagate(self, tau_s: np.ndarray, pressure_pa: np.ndarray) -> SolverResult:
        start = perf_counter()
        tau = np.asarray(tau_s, dtype=float)
        p0 = np.asarray(pressure_pa, dtype=float)
        order = np.argsort(tau)
        tau = tau[order]
        p = p0[order].copy()
        dtau = float(np.median(np.diff(tau)))
        base_dr = min(float(self.settings.max_step_m), self.slant_range_m / max(int(self.settings.range_steps), 1))
        target_snapshot_ranges = np.linspace(0.0, self.slant_range_m, max(int(self.settings.snapshot_count), 2))
        next_snapshot = 0
        snapshots: list[pd.DataFrame] = []
        r = 0.0
        steps = 0
        max_cfl_observed = 0.0
        min_stable_dr = np.inf
        max_k_nl = 0.0
        max_nu_tv = 0.0
        molecular_status = MolecularRelaxationStatus(enabled=self.settings.molecular_relaxation_enabled)
        relaxation_filters_applied = 0
        relaxation_max_alpha = 0.0
        relaxation_max_alpha_dr = 0.0
        relaxation_min_gain = 1.0
        relaxation_last_fr_o = np.nan
        relaxation_last_fr_n = np.nan
        relaxation_last_h = np.nan
        relaxation_warning_count = 0

        def maybe_snapshot(range_m: float, pressure: np.ndarray) -> None:
            nonlocal next_snapshot
            while next_snapshot < len(target_snapshot_ranges) and range_m >= target_snapshot_ranges[next_snapshot] - 1.0e-9:
                snapshots.append(
                    pd.DataFrame(
                        {
                            "range_m": float(range_m),
                            "tau_s": tau,
                            "p_Pa": pressure,
                        }
                    )
                )
                next_snapshot += 1

        maybe_snapshot(0.0, p.copy())
        while r < self.slant_range_m - 1.0e-9:
            state = self._state_at_range(r)
            coeffs = self._coefficients(state)
            k_nl = coeffs["k_nonlinear_1_per_Pa_m"] if self.settings.nonlinear_enabled else 0.0
            nu_tv = coeffs["nu_thermoviscous_s2_per_m"] if self.settings.thermoviscous_enabled else 0.0
            stable = stable_step_limit(dtau, p, k_nl, nu_tv, self.settings.cfl)
            dr = min(base_dr, stable, self.slant_range_m - r)
            if not np.isfinite(dr) or dr <= 0.0:
                raise FloatingPointError(f"invalid propagation step at r={r}: {dr}")
            old_state = state
            old_r = r
            old_p = p.copy()
            r += dr

            if self.settings.nonlinear_enabled:
                if self.settings.nonlinear_scheme == "rusanov_first_order":
                    p, char_speed = rusanov_nonlinear_step(p, dtau, dr, k_nl)
                elif self.settings.nonlinear_scheme == "muscl_ssprk2":
                    p, char_speed = muscl_rusanov_nonlinear_step(p, dtau, dr, k_nl)
                else:
                    raise ValueError(
                        f"unsupported nonlinear scheme: {self.settings.nonlinear_scheme}"
                    )
                if char_speed > 0:
                    max_cfl_observed = max(max_cfl_observed, float(char_speed * dr / dtau))
            if self.settings.thermoviscous_enabled:
                p = diffusion_step(p, dtau, dr, nu_tv)
            if self.settings.artificial_viscosity_factor > 0.0:
                max_char = abs(k_nl) * max(np.nanmax(np.abs(old_p)), 1.0e-12)
                nu_art = 0.5 * float(self.settings.artificial_viscosity_factor) * max_char * dtau
                p = diffusion_step(p, dtau, dr, nu_art)
            if self.settings.molecular_relaxation_enabled:
                p, molecular_status = apply_molecular_relaxation_filter(
                    p,
                    dtau,
                    layer=state,
                    dr=dr,
                    config=self.settings,
                    enabled=True,
                )
                relaxation_filters_applied += 1
                relaxation_max_alpha = max(relaxation_max_alpha, molecular_status.max_alpha_np_per_m)
                relaxation_max_alpha_dr = max(relaxation_max_alpha_dr, molecular_status.max_alpha_dr)
                relaxation_min_gain = min(relaxation_min_gain, molecular_status.min_filter_gain)
                relaxation_last_fr_o = molecular_status.relaxation_frequency_O2_Hz
                relaxation_last_fr_n = molecular_status.relaxation_frequency_N2_Hz
                relaxation_last_h = molecular_status.water_vapor_mole_fraction
                relaxation_warning_count += len(molecular_status.warnings)
            if self.settings.geometric_spreading_enabled:
                old_eff = self.settings.reference_range_m + old_r
                new_eff = self.settings.reference_range_m + r
                p *= (old_eff / new_eff) ** float(self.settings.spreading_exponent)
            if self.settings.stratification_enabled:
                new_state = self._state_at_range(r)
                z_old = old_state["impedance_Pa_s_per_m"]
                z_new = new_state["impedance_Pa_s_per_m"]
                if z_old > 0 and z_new > 0:
                    p *= np.sqrt(z_new / z_old)
            if not np.all(np.isfinite(p)):
                raise FloatingPointError(f"non-finite waveform at r={r}")

            min_stable_dr = min(min_stable_dr, stable)
            max_k_nl = max(max_k_nl, abs(k_nl))
            max_nu_tv = max(max_nu_tv, abs(nu_tv))
            steps += 1
            maybe_snapshot(r, p.copy())

        if next_snapshot < len(target_snapshot_ranges):
            maybe_snapshot(self.slant_range_m, p.copy())

        diagnostics = {
            "runtime_s": float(perf_counter() - start),
            "steps": int(steps),
            "dtau_s": dtau,
            "slant_range_m": self.slant_range_m,
            "mach_angle_deg": float(np.rad2deg(mach_angle_rad(self.settings.mach))),
            "max_cfl_observed": max_cfl_observed,
            "min_stable_step_m": float(min_stable_dr) if np.isfinite(min_stable_dr) else np.nan,
            "base_step_m": base_dr,
            "max_k_nonlinear_1_per_Pa_m": max_k_nl,
            "max_nu_thermoviscous_s2_per_m": max_nu_tv,
            "molecular_relaxation_implemented": molecular_status.implemented,
            "molecular_relaxation_notes": molecular_status.notes,
            "molecular_relaxation_enabled": bool(self.settings.molecular_relaxation_enabled),
            "molecular_relaxation_operator": self.settings.relaxation_operator,
            "relative_humidity": float(self.settings.relative_humidity),
            "include_oxygen_relaxation": bool(self.settings.include_oxygen_relaxation),
            "include_nitrogen_relaxation": bool(self.settings.include_nitrogen_relaxation),
            "include_classical_absorption": bool(self.settings.include_classical_absorption),
            "relaxation_filters_applied": int(relaxation_filters_applied),
            "relaxation_max_alpha_np_per_m": float(relaxation_max_alpha),
            "relaxation_max_alpha_dr": float(relaxation_max_alpha_dr),
            "relaxation_min_filter_gain": float(relaxation_min_gain),
            "relaxation_last_frO_Hz": float(relaxation_last_fr_o),
            "relaxation_last_frN_Hz": float(relaxation_last_fr_n),
            "relaxation_last_water_vapor_mole_fraction": float(relaxation_last_h),
            "relaxation_warning_count": int(relaxation_warning_count),
            "nonlinear_enabled": bool(self.settings.nonlinear_enabled),
            "nonlinear_scheme": self.settings.nonlinear_scheme,
            "thermoviscous_enabled": bool(self.settings.thermoviscous_enabled),
            "geometric_spreading_enabled": bool(self.settings.geometric_spreading_enabled),
            "stratification_enabled": bool(self.settings.stratification_enabled),
            "artificial_viscosity_factor": float(self.settings.artificial_viscosity_factor),
            "smoothing_enabled": bool(self.settings.smoothing_enabled),
        }
        return SolverResult(
            tau_s=tau,
            source_pressure_Pa=p0[order],
            ground_pressure_Pa=p,
            snapshots=pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame(),
            diagnostics=diagnostics,
        )


VARIANT_OVERRIDES = {
    "geometric": {
        "nonlinear_enabled": False,
        "thermoviscous_enabled": False,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "nonlinear": {
        "nonlinear_enabled": True,
        "thermoviscous_enabled": False,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "thermoviscous": {
        "nonlinear_enabled": False,
        "thermoviscous_enabled": True,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "nonlinear_thermoviscous": {
        "nonlinear_enabled": True,
        "thermoviscous_enabled": True,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "full": {
        "nonlinear_enabled": True,
        "thermoviscous_enabled": True,
        "molecular_relaxation_enabled": True,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "full_no_molecular_relaxation": {
        "nonlinear_enabled": True,
        "thermoviscous_enabled": True,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "full_with_molecular_relaxation": {
        "nonlinear_enabled": True,
        "thermoviscous_enabled": True,
        "molecular_relaxation_enabled": True,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "geometric_only": {
        "nonlinear_enabled": False,
        "thermoviscous_enabled": False,
        "molecular_relaxation_enabled": False,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
    "molecular_relaxation_only_plus_spreading": {
        "nonlinear_enabled": False,
        "thermoviscous_enabled": False,
        "molecular_relaxation_enabled": True,
        "geometric_spreading_enabled": True,
        "stratification_enabled": True,
    },
}

