"""Core source-construction and propagation routines for Beta."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


from .augmented_burgers import AugmentedBurgersSolver, SolverSettings
from .pldb import stevens_mark_vii_pldb


L_REF_M = 27.432
MACH = 1.4
P_INF_PA = 10520.0
U_INF_M_S = 413.097
AIRCRAFT_ALTITUDE_M = 16_000.0
RELATIVE_HUMIDITY = 0.30
TIME_GRID_POINTS = 4096
PAD_TIME_S = 0.035
ENDPOINT_BASELINE_FRACTION = 0.05


@dataclass(frozen=True)
class SourceMetadata:
    trace_file: str
    aoa_deg: float
    cfd_time_s: float
    r_over_l: float
    source_radius_m: float
    reference_length_m: float
    freestream_velocity_m_s: float
    sample_count: int
    valid_fraction: float
    x_equivalent_min: float
    x_equivalent_max: float
    native_duration_s: float
    raw_peak_to_peak_pa: float
    corrected_peak_to_peak_pa: float
    leading_baseline_pa: float
    trailing_baseline_pa: float
    baseline_method: str
    native_equivalent_sample_rate_hz: float
    computational_sample_rate_hz: float
    ambient_pressure_pa: float | None
    ambient_density_kg_m3: float | None
    ambient_temperature_k: float | None
    local_mach: float | None
    local_velocity_m_s: float | None


@dataclass(frozen=True)
class BetaResult:
    source: SourceMetadata
    source_altitude_m: float
    mach_angle_deg: float
    slant_path_m: float
    spreading_model: str
    source_peak_to_peak_pa: float
    source_rms_pa: float
    ground_peak_to_peak_pa: float
    ground_rms_pa: float
    ground_broadband_spl_db: float
    ground_pldb: float
    ground_reflected_x2_pldb: float
    pldb_method: str
    solver_steps: int
    solver_runtime_s: float
    screening_status: str


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"none of the required columns are present: {names}")


def _single_value(frame: pd.DataFrame, names: tuple[str, ...], override: float | None) -> float:
    if override is not None:
        return float(override)
    column = _first_column(frame, names)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} must contain exactly one finite value; found {values}")
    return float(values[0])


def _optional_median(frame: pd.DataFrame, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                return float(np.median(values))
    return None


def load_and_construct_source(
    trace_file: Path | str,
    *,
    aoa_override: float | None = None,
    time_override: float | None = None,
    r_over_l_override: float | None = None,
    grid_points: int = TIME_GRID_POINTS,
    pad_time_s: float = PAD_TIME_S,
    baseline_method: str = "linear_endpoints",
    freestream_velocity_m_s: float = U_INF_M_S,
    reference_length_m: float = L_REF_M,
    ambient_pressure_pa: float = P_INF_PA,
) -> tuple[np.ndarray, np.ndarray, SourceMetadata, pd.DataFrame]:
    """Load a Mach-equivalent CFD line and construct a padded pressure-time source.

    Beta preserves the CFD waveform without smoothing.  It removes only a
    linear endpoint bias defined by robust medians over the first and last 5%
    of the extraction window, then maps equivalent axial distance to retarded
    time using tau = L * (x_eq - min(x_eq)) / U_inf.
    """

    trace_file = Path(trace_file).resolve()
    frame = pd.read_csv(trace_file)
    if len(frame) < 100:
        raise ValueError(f"trace has too few samples: {len(frame)}")

    valid_column = next(
        (name for name in ("valid_sample", "vtk_valid_point_mask") if name in frame.columns),
        None,
    )
    if valid_column is None:
        raise ValueError("trace must contain valid_sample or vtk_valid_point_mask")
    valid = pd.to_numeric(frame[valid_column], errors="coerce").fillna(0.0) >= 0.5
    valid_fraction = float(valid.mean())
    if valid_fraction != 1.0:
        raise ValueError(f"Beta requires a fully valid extraction; valid fraction={valid_fraction:.6f}")
    frame = frame[valid].copy()

    x_column = _first_column(
        frame,
        ("x_equivalent_over_L", "x_over_L_equivalent", "x_over_L_mach_shifted"),
    )
    x_equivalent = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(x_equivalent)):
        raise ValueError("equivalent-x coordinates contain non-finite values")

    if "delta_p_Pa" in frame.columns:
        raw_pressure = pd.to_numeric(frame["delta_p_Pa"], errors="coerce").to_numpy(dtype=float)
    elif "delta_p" in frame.columns:
        raw_pressure = pd.to_numeric(frame["delta_p"], errors="coerce").to_numpy(dtype=float)
    else:
        q_column = _first_column(frame, ("delta_p_over_p_inf",))
        p_column = next((name for name in ("p_inf_Pa", "p_inf") if name in frame.columns), None)
        ambient = (
            pd.to_numeric(frame[p_column], errors="coerce").to_numpy(dtype=float)
            if p_column
            else np.full(len(frame), float(ambient_pressure_pa))
        )
        raw_pressure = pd.to_numeric(frame[q_column], errors="coerce").to_numpy(dtype=float) * ambient
    if not np.all(np.isfinite(raw_pressure)):
        raise ValueError("pressure trace contains non-finite values")

    order = np.argsort(x_equivalent)
    x_equivalent = x_equivalent[order]
    raw_pressure = raw_pressure[order]
    if np.any(np.diff(x_equivalent) <= 0.0):
        raise ValueError("equivalent-x coordinates must be strictly increasing")

    aoa_deg = _single_value(
        frame,
        ("trial_target_absolute_angle_deg", "aoa_deg"),
        aoa_override,
    )
    cfd_time_s = _single_value(frame, ("time_s", "time"), time_override)
    r_over_l = _single_value(frame, ("R_over_L", "r_over_L"), r_over_l_override)
    if r_over_l <= 0.0:
        raise ValueError("R/L must be positive")

    if int(grid_points) < 128:
        raise ValueError("source grid must contain at least 128 points")
    if float(pad_time_s) < 0.0:
        raise ValueError("source padding duration cannot be negative")
    if float(freestream_velocity_m_s) <= 0.0 or float(reference_length_m) <= 0.0:
        raise ValueError("freestream velocity and reference length must be positive")
    if float(ambient_pressure_pa) <= 0.0:
        raise ValueError("ambient pressure must be positive")

    endpoint_count = max(3, int(round(len(raw_pressure) * ENDPOINT_BASELINE_FRACTION)))
    leading = float(np.median(raw_pressure[:endpoint_count]))
    trailing = float(np.median(raw_pressure[-endpoint_count:]))
    if baseline_method == "linear_endpoints":
        baseline = np.linspace(leading, trailing, len(raw_pressure))
        baseline_description = "linear bias between first/last 5% pressure medians; no waveform smoothing"
    elif baseline_method == "constant_endpoints":
        baseline = np.full(len(raw_pressure), 0.5 * (leading + trailing))
        baseline_description = "constant bias from mean of first/last 5% pressure medians; no waveform smoothing"
    elif baseline_method == "none":
        baseline = np.zeros(len(raw_pressure))
        baseline_description = "no baseline removal; endpoint taper/padding only"
    else:
        raise ValueError(f"unsupported baseline method: {baseline_method}")
    corrected = raw_pressure - baseline

    native_tau = float(reference_length_m) * (
        x_equivalent - float(x_equivalent.min())
    ) / float(freestream_velocity_m_s)
    tau = np.linspace(
        -float(pad_time_s),
        float(native_tau.max()) + float(pad_time_s),
        int(grid_points),
    )
    pressure = np.interp(tau, native_tau, corrected, left=0.0, right=0.0)
    metadata = SourceMetadata(
        trace_file=str(trace_file),
        aoa_deg=aoa_deg,
        cfd_time_s=cfd_time_s,
        r_over_l=r_over_l,
        source_radius_m=r_over_l * float(reference_length_m),
        reference_length_m=float(reference_length_m),
        freestream_velocity_m_s=float(freestream_velocity_m_s),
        sample_count=int(len(frame)),
        valid_fraction=valid_fraction,
        x_equivalent_min=float(x_equivalent.min()),
        x_equivalent_max=float(x_equivalent.max()),
        native_duration_s=float(native_tau.max()),
        raw_peak_to_peak_pa=float(np.ptp(raw_pressure)),
        corrected_peak_to_peak_pa=float(np.ptp(corrected)),
        leading_baseline_pa=leading,
        trailing_baseline_pa=trailing,
        baseline_method=baseline_description,
        native_equivalent_sample_rate_hz=float(1.0 / np.median(np.diff(native_tau))),
        computational_sample_rate_hz=float(1.0 / np.median(np.diff(tau))),
        ambient_pressure_pa=(
            _optional_median(frame, ("p_inf_Pa", "p_inf"))
            or float(ambient_pressure_pa)
        ),
        ambient_density_kg_m3=_optional_median(frame, ("rho_kg_m3", "rho")),
        ambient_temperature_k=_optional_median(frame, ("T_K", "T")),
        local_mach=_optional_median(frame, ("Ma", "Mach")),
        local_velocity_m_s=_optional_median(frame, ("U_mag_m_s", "U_mag")),
    )
    source_frame = pd.DataFrame(
        {
            "tau_s": tau,
            "source_pressure_pa": pressure,
            "r_over_l": r_over_l,
            "aoa_deg": aoa_deg,
            "cfd_time_s": cfd_time_s,
        }
    )
    return tau, pressure, metadata, source_frame


def _pressure_metrics(pressure: np.ndarray) -> tuple[float, float]:
    pressure = np.asarray(pressure, dtype=float)
    centered = pressure - float(np.mean(pressure))
    return float(np.ptp(pressure)), float(np.sqrt(np.mean(centered * centered)))


def _broadband_spl(pressure: np.ndarray) -> float:
    _, rms = _pressure_metrics(pressure)
    return float(20.0 * np.log10(max(rms, 1.0e-300) / 20.0e-6))


def propagate_beta(
    tau: np.ndarray,
    source_pressure: np.ndarray,
    metadata: SourceMetadata,
) -> tuple[BetaResult, pd.DataFrame, pd.DataFrame]:
    """Propagate along the Mach ray with a cylindrical ray-tube approximation."""

    source_altitude = AIRCRAFT_ALTITUDE_M - metadata.source_radius_m
    if source_altitude <= 0.0:
        raise ValueError("extraction radius is below the assumed ground altitude")
    settings = SolverSettings(
        source_altitude_m=source_altitude,
        ground_altitude_m=0.0,
        mach=MACH,
        spreading_exponent=0.5,
        reference_range_m=metadata.source_radius_m,
        range_steps=900,
        max_step_m=35.0,
        snapshot_count=8,
        nonlinear_enabled=True,
        thermoviscous_enabled=True,
        molecular_relaxation_enabled=True,
        geometric_spreading_enabled=True,
        stratification_enabled=True,
        artificial_viscosity_factor=0.01,
        smoothing_enabled=False,
        relative_humidity=RELATIVE_HUMIDITY,
    )
    propagated = AugmentedBurgersSolver(settings).propagate(tau, source_pressure)
    pldb = stevens_mark_vii_pldb(
        propagated.tau_s,
        propagated.ground_pressure_Pa,
        pad_front_multiplier=6,
        pad_rear_multiplier=6,
        len_window_points=800,
    )
    reflected = stevens_mark_vii_pldb(
        propagated.tau_s,
        2.0 * propagated.ground_pressure_Pa,
        pad_front_multiplier=6,
        pad_rear_multiplier=6,
        len_window_points=800,
    )
    source_p2p, source_rms = _pressure_metrics(source_pressure)
    ground_p2p, ground_rms = _pressure_metrics(propagated.ground_pressure_Pa)
    result = BetaResult(
        source=metadata,
        source_altitude_m=source_altitude,
        mach_angle_deg=float(propagated.diagnostics["mach_angle_deg"]),
        slant_path_m=float(propagated.diagnostics["slant_range_m"]),
        spreading_model=(
            "cylindrical line-source ray tube: A proportional to total radius, "
            "therefore p proportional to radius^(-1/2)"
        ),
        source_peak_to_peak_pa=source_p2p,
        source_rms_pa=source_rms,
        ground_peak_to_peak_pa=ground_p2p,
        ground_rms_pa=ground_rms,
        ground_broadband_spl_db=_broadband_spl(propagated.ground_pressure_Pa),
        ground_pldb=float(pldb.pldb),
        ground_reflected_x2_pldb=float(reflected.pldb),
        pldb_method=pldb.method,
        solver_steps=int(propagated.diagnostics["steps"]),
        solver_runtime_s=float(propagated.diagnostics["runtime_s"]),
        screening_status=(
            "Beta screening result; physical ray-tube Jacobian, ground impedance, "
            "winds, turbulence, and sBOOM/PCBoom validation are not yet implemented"
        ),
    )
    waveform = pd.DataFrame(
        {
            "tau_s": propagated.tau_s,
            "source_pressure_pa": propagated.source_pressure_Pa,
            "ground_incident_pressure_pa": propagated.ground_pressure_Pa,
            "ground_ideal_rigid_x2_pressure_pa": 2.0 * propagated.ground_pressure_Pa,
        }
    )
    return result, waveform, propagated.snapshots


def result_to_dict(result: BetaResult) -> dict[str, object]:
    data = asdict(result)
    data["method"] = {
        "source_coordinate": "Mach-equivalent x/L",
        "time_mapping": "tau = L * (x_eq - min(x_eq)) / U_inf",
        "path": "Mach-angle slant from extraction plane to ground",
        "atmosphere": "US standard atmosphere, 30% relative humidity",
        "physics": [
            "nonlinear steepening",
            "thermoviscous absorption",
            "molecular relaxation",
            "stratification",
            "cylindrical spreading approximation",
        ],
        "ground_reflection_in_primary": False,
        "post_smoothing": False,
    }
    return data


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
