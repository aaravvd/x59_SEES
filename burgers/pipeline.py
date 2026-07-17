"""Audited Beta propagation and NASA-2025 Mark VII evaluation.

This module deliberately leaves the legacy ``propagate_beta`` path unchanged.
It supplies the refined methodology used for the 0.65 s RANS benchmark and the
lower-information 0.8 s AoA sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    SourceMetadata,
    _broadband_spl,
    _pressure_metrics,
)

from .augmented_burgers import AugmentedBurgersSolver, SolverSettings
from .pldb import stevens_mark_vii_pldb, stevens_mark_vii_pldb_nasa_2025


@dataclass(frozen=True)
class AuditedPropagationConfig:
    """Explicit settings for a screening-level Beta propagation run."""

    aircraft_altitude_m: float = 16_000.0
    ground_altitude_m: float = 0.0
    mach: float = 1.40
    relative_humidity: float = 0.30
    spreading_exponent: float = 0.50
    range_steps: int = 900
    max_step_m: float = 35.0
    snapshot_count: int = 10
    artificial_viscosity_factor: float = 0.01
    nonlinear_enabled: bool = True
    nonlinear_scheme: str = "muscl_ssprk2"
    thermoviscous_enabled: bool = True
    molecular_relaxation_enabled: bool = True
    geometric_spreading_enabled: bool = True
    stratification_enabled: bool = True
    include_classical_absorption: bool = False
    ground_pressure_factor: float = 1.90
    pldb_taper_duration_s: float = 0.010
    pldb_minimum_padded_duration_s: float = 2.0
    pldb_event_correction_db: float = 0.0


def _metric_payload(metric: Any, *, pressure_scale: float = 1.0) -> dict[str, Any]:
    return {
        "pldb": float(metric.pldb),
        "pressure_scale": float(pressure_scale),
        "method": metric.method,
        "total_loudness_sones": float(metric.total_loudness_sones),
        "max_sones": float(metric.max_sones),
        "summation_factor": float(metric.summation_factor),
        "sample_rate_hz": metric.sample_rate_hz,
        "nyquist_hz": metric.nyquist_hz,
        "fft_length": metric.fft_length,
        "fft_duration_s": metric.fft_duration_s,
        "event_correction_db": float(metric.event_correction_db),
    }


def propagate_audited_beta(
    tau_s: np.ndarray,
    source_pressure_pa: np.ndarray,
    metadata: SourceMetadata,
    config: AuditedPropagationConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Propagate one source and return summary, waveform, snapshots, and PL bands."""

    cfg = config or AuditedPropagationConfig()
    if cfg.mach <= 1.0:
        raise ValueError("Beta propagation requires a supersonic Mach number")
    source_altitude = cfg.aircraft_altitude_m - metadata.source_radius_m
    if source_altitude <= cfg.ground_altitude_m:
        raise ValueError("extraction radius is below the assumed ground altitude")
    if cfg.ground_pressure_factor <= 0.0:
        raise ValueError("ground pressure factor must be positive")

    settings = SolverSettings(
        source_altitude_m=source_altitude,
        ground_altitude_m=cfg.ground_altitude_m,
        mach=cfg.mach,
        spreading_exponent=cfg.spreading_exponent,
        reference_range_m=metadata.source_radius_m,
        range_steps=cfg.range_steps,
        max_step_m=cfg.max_step_m,
        snapshot_count=cfg.snapshot_count,
        nonlinear_enabled=cfg.nonlinear_enabled,
        nonlinear_scheme=cfg.nonlinear_scheme,
        thermoviscous_enabled=cfg.thermoviscous_enabled,
        molecular_relaxation_enabled=cfg.molecular_relaxation_enabled,
        geometric_spreading_enabled=cfg.geometric_spreading_enabled,
        stratification_enabled=cfg.stratification_enabled,
        artificial_viscosity_factor=cfg.artificial_viscosity_factor,
        smoothing_enabled=False,
        relative_humidity=cfg.relative_humidity,
        # Classical thermoviscous absorption is already represented by the
        # explicit diffusion operator.  Leaving this False prevents the
        # ISO/Bass frequency-domain operator from counting it a second time.
        include_classical_absorption=cfg.include_classical_absorption,
    )
    propagated = AugmentedBurgersSolver(settings).propagate(tau_s, source_pressure_pa)
    ground = propagated.ground_pressure_Pa

    metric_kwargs = {
        "taper_duration_s": cfg.pldb_taper_duration_s,
        "minimum_padded_duration_s": cfg.pldb_minimum_padded_duration_s,
        "event_correction_db": cfg.pldb_event_correction_db,
    }
    incident = stevens_mark_vii_pldb_nasa_2025(propagated.tau_s, ground, **metric_kwargs)
    reflected = stevens_mark_vii_pldb_nasa_2025(
        propagated.tau_s,
        ground,
        pressure_scale=cfg.ground_pressure_factor,
        **metric_kwargs,
    )
    rigid = stevens_mark_vii_pldb_nasa_2025(
        propagated.tau_s,
        ground,
        pressure_scale=2.0,
        **metric_kwargs,
    )
    legacy_incident = stevens_mark_vii_pldb(
        propagated.tau_s,
        ground,
        pad_front_multiplier=6,
        pad_rear_multiplier=6,
        len_window_points=800,
    )

    source_p2p, source_rms = _pressure_metrics(source_pressure_pa)
    ground_p2p, ground_rms = _pressure_metrics(ground)
    band_table = pd.DataFrame(
        {
            "band_center_hz": incident.band_centers_hz,
            "band_spl_incident_db": incident.band_spl_db,
            "equivalent_loudness_incident_db": incident.equivalent_loudness_db,
            "sones_incident": incident.sones,
            "band_energy_incident_pa2": incident.band_energy_pa2,
            "band_supported_fraction": incident.band_supported_fraction,
            "band_spl_ground_scaled_db": reflected.band_spl_db,
            "sones_ground_scaled": reflected.sones,
        }
    )
    waveform = pd.DataFrame(
        {
            "tau_s": propagated.tau_s,
            "source_pressure_pa": propagated.source_pressure_Pa,
            "ground_incident_pressure_pa": ground,
            "ground_scaled_pressure_pa": cfg.ground_pressure_factor * ground,
            "ground_ideal_rigid_x2_pressure_pa": 2.0 * ground,
        }
    )
    summary: dict[str, Any] = {
        "source": asdict(metadata),
        "source_altitude_m": float(source_altitude),
        "aircraft_altitude_m": float(cfg.aircraft_altitude_m),
        "ground_altitude_m": float(cfg.ground_altitude_m),
        "mach": float(cfg.mach),
        "source_peak_to_peak_pa": source_p2p,
        "source_rms_pa": source_rms,
        "source_native_nyquist_hz": 0.5 * metadata.native_equivalent_sample_rate_hz,
        "ground_peak_to_peak_pa": ground_p2p,
        "ground_rms_pa": ground_rms,
        "ground_broadband_spl_db": _broadband_spl(ground),
        "nasa_2025_incident": _metric_payload(incident),
        "nasa_2025_ground_scaled": _metric_payload(
            reflected,
            pressure_scale=cfg.ground_pressure_factor,
        ),
        "nasa_2025_ideal_rigid_x2": _metric_payload(rigid, pressure_scale=2.0),
        "legacy_bolander_2019_incident": _metric_payload(legacy_incident),
        "solver_settings": asdict(settings),
        "solver_diagnostics": propagated.diagnostics,
        "methodology_status": {
            "classification": "screening-level augmented Burgers propagation",
            "source_coordinate": "Mach-equivalent x/L",
            "time_mapping": "tau = L * (x_eq - min(x_eq)) / U_inf",
            "path": "straight Mach-angle slant from extraction radius to ground",
            "spreading": "cylindrical line-source ray-tube approximation",
            "ground_comparison_convention": f"pressure multiplied by {cfg.ground_pressure_factor:.2f}",
            "classical_absorption_double_count_prevented": not cfg.include_classical_absorption,
            "post_smoothing": False,
            "not_implemented": [
                "refracted ray tracing through winds and temperature gradients",
                "ray-tube Jacobian from a three-dimensional ray family",
                "finite-impedance ground reflection",
                "turbulence and caustics",
                "validation against sBOOM, PCBoom, or measured ground waveforms",
            ],
        },
    }
    return summary, waveform, propagated.snapshots, band_table
