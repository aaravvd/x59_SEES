"""Command-line interface for Beta propagation and PLdB evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .core import (
    AIRCRAFT_ALTITUDE_M,
    L_REF_M,
    MACH,
    PAD_TIME_S,
    P_INF_PA,
    TIME_GRID_POINTS,
    U_INF_M_S,
    load_and_construct_source,
)
from .pipeline import AuditedPropagationConfig, propagate_audited_beta
from .pldb import stevens_mark_vii_pldb, stevens_mark_vii_pldb_nasa_2025


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _add_common_metric_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pressure-scale", type=float, default=1.0)
    parser.add_argument("--event-correction-db", type=float, default=0.0)
    parser.add_argument("--taper-duration-s", type=float, default=0.010)
    parser.add_argument("--minimum-padded-duration-s", type=float, default=2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beta-boom",
        description="Propagate a Mach-equivalent CFD pressure signature and calculate Mark VII PLdB.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    propagate = subparsers.add_parser(
        "propagate",
        help="Construct a source waveform, propagate it to the ground, and calculate PLdB.",
    )
    propagate.add_argument("--trace", type=Path, required=True, help="Input CFD trace CSV.")
    propagate.add_argument("--output-dir", type=Path, required=True)
    propagate.add_argument("--aoa-deg", type=float, default=None)
    propagate.add_argument("--cfd-time-s", type=float, default=None)
    propagate.add_argument("--r-over-l", type=float, default=None)
    propagate.add_argument("--reference-length-m", type=float, default=L_REF_M)
    propagate.add_argument("--freestream-velocity-m-s", type=float, default=U_INF_M_S)
    propagate.add_argument("--ambient-pressure-pa", type=float, default=P_INF_PA)
    propagate.add_argument("--grid-points", type=int, default=TIME_GRID_POINTS)
    propagate.add_argument("--pad-time-s", type=float, default=PAD_TIME_S)
    propagate.add_argument(
        "--baseline-method",
        choices=("linear_endpoints", "constant_endpoints", "none"),
        default="linear_endpoints",
    )
    propagate.add_argument("--aircraft-altitude-m", type=float, default=AIRCRAFT_ALTITUDE_M)
    propagate.add_argument("--ground-altitude-m", type=float, default=0.0)
    propagate.add_argument("--mach", type=float, default=MACH)
    propagate.add_argument("--relative-humidity", type=float, default=0.30)
    propagate.add_argument("--spreading-exponent", type=float, default=0.50)
    propagate.add_argument("--range-steps", type=int, default=900)
    propagate.add_argument("--max-step-m", type=float, default=35.0)
    propagate.add_argument("--snapshot-count", type=int, default=10)
    propagate.add_argument("--artificial-viscosity-factor", type=float, default=0.01)
    propagate.add_argument(
        "--nonlinear-scheme",
        choices=("muscl_ssprk2", "rusanov_first_order"),
        default="muscl_ssprk2",
    )
    propagate.add_argument(
        "--molecular-relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    propagate.add_argument(
        "--thermoviscous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    propagate.add_argument(
        "--nonlinear",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    propagate.add_argument(
        "--geometric-spreading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    propagate.add_argument(
        "--stratification",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    propagate.add_argument("--ground-pressure-factor", type=float, default=1.90)
    propagate.add_argument("--event-correction-db", type=float, default=0.0)
    propagate.add_argument("--taper-duration-s", type=float, default=0.010)
    propagate.add_argument("--minimum-padded-duration-s", type=float, default=2.0)

    metric = subparsers.add_parser(
        "pldb",
        help="Calculate NASA-2025 and legacy Mark VII results from a waveform CSV.",
    )
    metric.add_argument("--waveform", type=Path, required=True)
    metric.add_argument("--time-column", default="tau_s")
    metric.add_argument("--pressure-column", default="ground_incident_pressure_pa")
    metric.add_argument("--output", type=Path, required=True)
    _add_common_metric_options(metric)
    return parser


def _run_propagate(args: argparse.Namespace) -> int:
    tau, pressure, metadata, source = load_and_construct_source(
        args.trace,
        aoa_override=args.aoa_deg,
        time_override=args.cfd_time_s,
        r_over_l_override=args.r_over_l,
        grid_points=args.grid_points,
        pad_time_s=args.pad_time_s,
        baseline_method=args.baseline_method,
        freestream_velocity_m_s=args.freestream_velocity_m_s,
        reference_length_m=args.reference_length_m,
        ambient_pressure_pa=args.ambient_pressure_pa,
    )
    config = AuditedPropagationConfig(
        aircraft_altitude_m=args.aircraft_altitude_m,
        ground_altitude_m=args.ground_altitude_m,
        mach=args.mach,
        relative_humidity=args.relative_humidity,
        spreading_exponent=args.spreading_exponent,
        range_steps=args.range_steps,
        max_step_m=args.max_step_m,
        snapshot_count=args.snapshot_count,
        artificial_viscosity_factor=args.artificial_viscosity_factor,
        nonlinear_enabled=args.nonlinear,
        nonlinear_scheme=args.nonlinear_scheme,
        thermoviscous_enabled=args.thermoviscous,
        molecular_relaxation_enabled=args.molecular_relaxation,
        geometric_spreading_enabled=args.geometric_spreading,
        stratification_enabled=args.stratification,
        ground_pressure_factor=args.ground_pressure_factor,
        pldb_taper_duration_s=args.taper_duration_s,
        pldb_minimum_padded_duration_s=args.minimum_padded_duration_s,
        pldb_event_correction_db=args.event_correction_db,
    )
    summary, waveform, snapshots, bands = propagate_audited_beta(
        tau,
        pressure,
        metadata,
        config,
    )
    summary["package"] = {"name": "beta-boom-toolkit", "version": __version__}
    summary["run_configuration"] = asdict(config)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source.to_csv(output / "source_waveform.csv", index=False)
    waveform.to_csv(output / "ground_waveform.csv", index=False)
    snapshots.to_csv(output / "range_snapshots.csv", index=False)
    bands.to_csv(output / "pldb_bands.csv", index=False)
    _write_json(output / "result.json", summary)

    print(f"incident PLdB: {summary['nasa_2025_incident']['pldb']:.3f}")
    print(
        f"ground x{args.ground_pressure_factor:g} PLdB: "
        f"{summary['nasa_2025_ground_scaled']['pldb']:.3f}"
    )
    print(f"outputs: {output}")
    return 0


def _metric_payload(metric: Any) -> dict[str, Any]:
    return {
        "pldb": float(metric.pldb),
        "total_loudness_sones": float(metric.total_loudness_sones),
        "max_sones": float(metric.max_sones),
        "summation_factor": float(metric.summation_factor),
        "method": metric.method,
        "sample_rate_hz": metric.sample_rate_hz,
        "nyquist_hz": metric.nyquist_hz,
        "fft_length": metric.fft_length,
        "fft_duration_s": metric.fft_duration_s,
        "event_correction_db": metric.event_correction_db,
    }


def _run_pldb(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.waveform)
    missing = {args.time_column, args.pressure_column} - set(frame.columns)
    if missing:
        raise ValueError(f"waveform CSV is missing columns: {sorted(missing)}")
    time = pd.to_numeric(frame[args.time_column], errors="raise").to_numpy(dtype=float)
    pressure = pd.to_numeric(frame[args.pressure_column], errors="raise").to_numpy(dtype=float)
    nasa = stevens_mark_vii_pldb_nasa_2025(
        time,
        pressure,
        pressure_scale=args.pressure_scale,
        event_correction_db=args.event_correction_db,
        taper_duration_s=args.taper_duration_s,
        minimum_padded_duration_s=args.minimum_padded_duration_s,
    )
    legacy = stevens_mark_vii_pldb(
        time,
        pressure,
        pressure_scale=args.pressure_scale,
    )
    payload = {
        "input": {
            "waveform": str(args.waveform.resolve()),
            "time_column": args.time_column,
            "pressure_column": args.pressure_column,
            "pressure_scale": args.pressure_scale,
        },
        "nasa_2025": _metric_payload(nasa),
        "legacy_bolander_2019": _metric_payload(legacy),
    }
    _write_json(args.output.resolve(), payload)
    band_path = args.output.resolve().with_name(args.output.stem + "_bands.csv")
    pd.DataFrame(
        {
            "band_center_hz": nasa.band_centers_hz,
            "band_spl_db": nasa.band_spl_db,
            "equivalent_loudness_db": nasa.equivalent_loudness_db,
            "sones": nasa.sones,
            "band_energy_pa2": nasa.band_energy_pa2,
            "supported_fraction": nasa.band_supported_fraction,
        }
    ).to_csv(band_path, index=False)
    print(f"NASA-2025 PLdB: {nasa.pldb:.3f}")
    print(f"legacy PLdB: {legacy.pldb:.3f}")
    print(f"outputs: {args.output.resolve()}, {band_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "propagate":
        return _run_propagate(args)
    if args.command == "pldb":
        return _run_pldb(args)
    raise AssertionError(f"unhandled command: {args.command}")
