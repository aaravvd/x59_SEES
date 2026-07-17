#!/usr/bin/env python3
"""Select an interpolation model for angle of attack versus PLdB.

The default response is the documented primary metric, NASA-2025 incident
Mark VII PLdB at the R/L=0.25 extraction station.  The script tests global
polynomials against a cubic smoothing spline using two held-out checks:

* leave-one-out cross-validation (single missing AoA cases), and
* rolling contiguous-triplet cross-validation (small missing AoA stretches).

The latter is the primary selection safeguard: it prevents a high-order
polynomial from being selected merely because it interpolates individual
points while behaving poorly across a short gap.  Outputs are strictly
in-domain; this is an interpolation surrogate, not an extrapolation model or
a flight-validated PLdB prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline, make_smoothing_spline


DEFAULT_INPUT = "/Users/aaravdixit/Downloads/complete_aoa_beta_sweep.csv"
DEFAULT_OUTPUT = "/Users/aaravdixit/Downloads/aoa_pldb_regression"

# These are dimensionless because the spline implementation standardizes both
# AoA and the response before fitting.  The range spans lightly smoothed to
# deliberately smooth cubic splines without depending on PLdB's raw scale.
DEFAULT_SPLINE_LAMBDAS = (
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
)

NAVY = "#15324B"
BLUE = "#2B6F9E"
ORANGE = "#D6652F"
INK = "#25313B"
MUTED = "#657582"
GRID = "#DCE3E8"


@dataclass(frozen=True)
class ModelSpec:
    """A candidate model configuration.

    ``degree`` applies only to polynomial regression.  ``smoothing_lambda``
    applies only to cubic smoothing splines fitted after standardization.
    """

    family: str
    degree: int | None = None
    smoothing_lambda: float | None = None

    @property
    def label(self) -> str:
        if self.family == "polynomial":
            return f"polynomial_degree_{self.degree}"
        return f"cubic_smoothing_spline_lambda_{self.smoothing_lambda:.0e}"

    @property
    def display_name(self) -> str:
        if self.family == "polynomial":
            return f"polynomial (degree {self.degree})"
        return f"cubic smoothing spline (lambda={self.smoothing_lambda:.0e})"


@dataclass
class PolynomialModel:
    """A normalized-basis polynomial model with AoA in degrees."""

    polynomial: Polynomial

    def predict(self, aoa_deg: np.ndarray) -> np.ndarray:
        return np.asarray(self.polynomial(aoa_deg), dtype=float)


@dataclass
class SmoothingSplineModel:
    """A cubic smoothing spline standardized in AoA degrees and response units."""

    spline: BSpline
    x_min: float
    x_span: float
    y_mean: float
    y_scale: float

    def predict(self, aoa_deg: np.ndarray) -> np.ndarray:
        x = np.asarray(aoa_deg, dtype=float)
        x_standardized = (x - self.x_min) / self.x_span
        return np.asarray(self.spline(x_standardized) * self.y_scale + self.y_mean, dtype=float)


@dataclass
class EvaluatedModel:
    """Full-fit model plus validation evidence for one candidate."""

    spec: ModelSpec
    fitted_model: PolynomialModel | SmoothingSplineModel
    train_metrics: dict[str, float]
    loo_metrics: dict[str, float]
    triplet_metrics: dict[str, float]
    loo_predictions: np.ndarray
    triplet_mean_predictions: np.ndarray
    triplet_prediction_counts: np.ndarray

    @property
    def selection_score(self) -> float:
        """Balanced local-interpolation score in response units (PLdB by default)."""

        return 0.5 * (self.loo_metrics["rmse"] + self.triplet_metrics["rmse"])


def parse_lambdas(text: str) -> tuple[float, ...]:
    """Parse comma-separated positive, dimensionless spline lambda values."""

    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--spline-lambdas must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("--spline-lambdas must contain one or more positive finite values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Audited sweep CSV to model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for model-selection artifacts (created if needed).",
    )
    parser.add_argument("--predictor-column", default="aoa_deg", help="Predictor column; default units are degrees.")
    parser.add_argument(
        "--response-column",
        default="incident_pldb",
        help="Response column; defaults to the primary NASA-2025 incident PLdB metric.",
    )
    parser.add_argument(
        "--max-polynomial-degree",
        type=int,
        default=16,
        help="Largest polynomial degree to test (default: 14).",
    )
    parser.add_argument(
        "--spline-lambdas",
        type=parse_lambdas,
        default=DEFAULT_SPLINE_LAMBDAS,
        help="Comma-separated standardized smoothing values to test.",
    )
    parser.add_argument(
        "--min-relative-cv-improvement",
        type=float,
        default=0.05,
        help="Minimum balanced-CV improvement required to replace the polynomial (default: 0.05).",
    )
    parser.add_argument(
        "--strict-mesh-only",
        action="store_true",
        help="Fit only rows whose check_mesh_reported_mesh_ok value is true; use as a sensitivity run.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a content hash for source-data provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_true(value: Any) -> bool:
    """Interpret the boolean spellings used by CSV exports."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def provenance_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return stable, JSON-safe category counts when an optional column exists."""

    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False)
    return {
        "missing" if pd.isna(value) else str(value): int(count)
        for value, count in counts.items()
    }


def format_counts(counts: dict[str, int]) -> str:
    """Format category counts compactly for the Markdown handoff."""

    return "; ".join(f"`{label}`: {count}" for label, count in counts.items()) or "not supplied"


def load_and_validate_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the requested CSV and fail closed on unusable model rows."""

    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {source}")

    frame = pd.read_csv(source)
    input_row_count = len(frame)
    required = [args.predictor_column, args.response_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")

    if args.strict_mesh_only:
        mesh_flag = "check_mesh_reported_mesh_ok"
        if mesh_flag not in frame.columns:
            raise ValueError(f"--strict-mesh-only requires a {mesh_flag!r} column")
        frame = frame.loc[frame[mesh_flag].map(is_true)].copy()

    frame["_predictor"] = pd.to_numeric(frame[args.predictor_column], errors="coerce")
    frame["_response"] = pd.to_numeric(frame[args.response_column], errors="coerce")
    valid = np.isfinite(frame["_predictor"].to_numpy(dtype=float)) & np.isfinite(
        frame["_response"].to_numpy(dtype=float)
    )
    if not bool(valid.all()):
        invalid_rows = frame.index[~valid].tolist()
        raise ValueError(
            "The predictor and response must be finite for every selected row; "
            f"invalid source row index/indices: {invalid_rows}"
        )

    frame = frame.sort_values("_predictor").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after the requested filter.")
    if frame["_predictor"].duplicated().any():
        duplicate_values = frame.loc[frame["_predictor"].duplicated(keep=False), "_predictor"].tolist()
        raise ValueError(f"The predictor must be unique; duplicate values: {duplicate_values}")
    if frame["_response"].nunique() < 2:
        raise ValueError("The response is constant, so regression model selection is not meaningful.")
    if args.max_polynomial_degree < 1:
        raise ValueError("--max-polynomial-degree must be at least 1")
    # Triplet CV removes three rows; Polynomial.fit needs degree + 1 remaining rows.
    if args.max_polynomial_degree > len(frame) - 4:
        raise ValueError(
            "--max-polynomial-degree is too high for triplet cross-validation: "
            f"must be <= {len(frame) - 4} for {len(frame)} selected rows"
        )
    if len(frame) < 8:
        raise ValueError("At least eight unique rows are required for the two validation checks.")

    mesh_counts = provenance_counts(frame, "mesh_quality")
    run_family_counts = provenance_counts(frame, "run_family")
    end_time_counts = provenance_counts(frame, "end_time_s")
    analysis_time_counts = provenance_counts(frame, "analysis_time_s")

    x = frame["_predictor"].to_numpy(dtype=float)
    metadata: dict[str, Any] = {
        "input_path": str(source),
        "input_sha256": sha256_file(source),
        "input_row_count": int(input_row_count),
        "rows_used": int(len(frame)),
        "strict_mesh_only": bool(args.strict_mesh_only),
        "predictor_range": [float(x.min()), float(x.max())],
        "mesh_quality_counts": mesh_counts,
        "run_family_counts": run_family_counts,
        "solve_end_time_s_counts": end_time_counts,
        "analysis_time_s_counts": analysis_time_counts,
        "mixed_run_family_or_time_provenance": bool(
            len(run_family_counts) > 1 or len(end_time_counts) > 1 or len(analysis_time_counts) > 1
        ),
    }
    return frame, metadata


def fit_model(spec: ModelSpec, x: np.ndarray, y: np.ndarray) -> PolynomialModel | SmoothingSplineModel:
    """Fit one candidate using AoA degrees and a response such as PLdB."""

    if spec.family == "polynomial":
        assert spec.degree is not None
        # Polynomial.fit uses a normalized basis internally, preventing raw-
        # degree powers from becoming ill-conditioned during the comparison.
        model = Polynomial.fit(x, y, deg=spec.degree, domain=[float(x.min()), float(x.max())])
        return PolynomialModel(model)

    if spec.family == "smoothing_spline":
        assert spec.smoothing_lambda is not None
        x_min = float(x.min())
        x_span = float(x.max() - x_min)
        y_mean = float(y.mean())
        y_scale = float(y.std(ddof=0))
        if x_span <= 0.0 or y_scale <= 0.0:
            raise ValueError("Smoothing-spline standardization requires nonconstant predictor and response values")
        spline = make_smoothing_spline(
            (x - x_min) / x_span,
            (y - y_mean) / y_scale,
            lam=spec.smoothing_lambda,
        )
        return SmoothingSplineModel(spline, x_min, x_span, y_mean, y_scale)

    raise ValueError(f"Unsupported model family: {spec.family}")


def predict(model: PolynomialModel | SmoothingSplineModel, x: np.ndarray) -> np.ndarray:
    """Return finite predictions or fail with the candidate name exposed upstream."""

    values = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("A candidate produced non-finite predictions")
    return values


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics in the native response unit (PLdB by default)."""

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = actual - prediction
    sse = float(np.sum(error**2))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
    }


def leave_one_out_predictions(spec: ModelSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict every row from a model refit without that row."""

    output = np.empty_like(y, dtype=float)
    for index in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[index] = False
        output[index] = predict(fit_model(spec, x[keep], y[keep]), x[index : index + 1])[0]
    return output


def contiguous_triplet_predictions(
    spec: ModelSpec, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run rolling three-row holdouts that remain inside the fitted AoA range.

    Each middle point starts a test triplet with its immediate neighbors.  This
    intentionally tests interpolation across a small unresolved AoA interval,
    unlike a random fold which can mix adjacent solved cases into training.
    """

    prediction_sum = np.zeros_like(y, dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    held_out_actual: list[float] = []
    held_out_prediction: list[float] = []

    for center in range(1, len(x) - 1):
        test_indices = np.arange(center - 1, center + 2)
        keep = np.ones(len(x), dtype=bool)
        keep[test_indices] = False
        candidate = fit_model(spec, x[keep], y[keep])
        values = predict(candidate, x[test_indices])
        prediction_sum[test_indices] += values
        prediction_count[test_indices] += 1
        held_out_actual.extend(y[test_indices])
        held_out_prediction.extend(values)

    average_prediction = np.full_like(y, np.nan, dtype=float)
    supported = prediction_count > 0
    average_prediction[supported] = prediction_sum[supported] / prediction_count[supported]
    return average_prediction, prediction_count, metrics(
        np.asarray(held_out_actual), np.asarray(held_out_prediction)
    )


def evaluate_spec(spec: ModelSpec, x: np.ndarray, y: np.ndarray) -> EvaluatedModel:
    """Fit a candidate once on all data and evaluate its two held-out checks."""

    fitted = fit_model(spec, x, y)
    training_prediction = predict(fitted, x)
    loo_prediction = leave_one_out_predictions(spec, x, y)
    triplet_prediction, triplet_counts, triplet_metrics = contiguous_triplet_predictions(spec, x, y)
    return EvaluatedModel(
        spec=spec,
        fitted_model=fitted,
        train_metrics=metrics(y, training_prediction),
        loo_metrics=metrics(y, loo_prediction),
        triplet_metrics=triplet_metrics,
        loo_predictions=loo_prediction,
        triplet_mean_predictions=triplet_prediction,
        triplet_prediction_counts=triplet_counts,
    )


def candidate_specs(max_degree: int, lambdas: Iterable[float]) -> list[ModelSpec]:
    """Create the global-polynomial and local-spline model candidates."""

    polynomials = [ModelSpec("polynomial", degree=degree) for degree in range(1, max_degree + 1)]
    splines = [ModelSpec("smoothing_spline", smoothing_lambda=float(value)) for value in lambdas]
    return polynomials + splines


def model_sort_key(result: EvaluatedModel) -> tuple[float, float, float, int]:
    """Prefer balanced CV, then triplet CV, LOO CV, then lower polynomial degree."""

    degree = result.spec.degree if result.spec.degree is not None else 0
    return (
        result.selection_score,
        result.triplet_metrics["rmse"],
        result.loo_metrics["rmse"],
        degree,
    )


def choose_models(
    results: list[EvaluatedModel], minimum_relative_improvement: float
) -> tuple[EvaluatedModel, EvaluatedModel, EvaluatedModel, float]:
    """Choose the stable polynomial baseline and decide whether spline wins materially."""

    polynomial_results = [result for result in results if result.spec.family == "polynomial"]
    spline_results = [result for result in results if result.spec.family == "smoothing_spline"]
    best_polynomial = min(polynomial_results, key=model_sort_key)
    best_spline = min(spline_results, key=model_sort_key)
    improvement = (best_polynomial.selection_score - best_spline.selection_score) / best_polynomial.selection_score
    selected = best_spline if improvement >= minimum_relative_improvement else best_polynomial
    if selected.spec.family == "polynomial":
        print(selected.fitted_model.polynomial)
    return best_polynomial, best_spline, selected, float(improvement)


def comparison_frame(
    results: list[EvaluatedModel],
    best_polynomial: EvaluatedModel,
    best_spline: EvaluatedModel,
    selected: EvaluatedModel,
) -> pd.DataFrame:
    """Build a machine-readable model-comparison table."""

    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=model_sort_key):
        rows.append(
            {
                "model": result.spec.label,
                "model_family": result.spec.family,
                "polynomial_degree": result.spec.degree,
                "smoothing_lambda_standardized": result.spec.smoothing_lambda,
                "selection_score_rmse": result.selection_score,
                "train_rmse": result.train_metrics["rmse"],
                "train_mae": result.train_metrics["mae"],
                "train_r2": result.train_metrics["r2"],
                "loocv_rmse": result.loo_metrics["rmse"],
                "loocv_mae": result.loo_metrics["mae"],
                "loocv_r2": result.loo_metrics["r2"],
                "triplet_cv_rmse": result.triplet_metrics["rmse"],
                "triplet_cv_mae": result.triplet_metrics["mae"],
                "triplet_cv_r2": result.triplet_metrics["r2"],
                "best_polynomial": int(result.spec == best_polynomial.spec),
                "best_smoothing_spline": int(result.spec == best_spline.spec),
                "selected": int(result.spec == selected.spec),
            }
        )
    return pd.DataFrame(rows)


def response_axis_label(column: str) -> str:
    """Return a readable axis label without assuming another PLdB convention."""

    return "PLdB" if "pldb" in column.lower() else column


def write_figure(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    response_column: str,
    best_polynomial: EvaluatedModel,
    best_spline: EvaluatedModel,
    selected: EvaluatedModel,
) -> list[Path]:
    """Write a fit and held-out-residual diagnostic in PNG and SVG formats."""

    grid = np.linspace(float(x.min()), float(x.max()), 600)
    polynomial_curve = predict(best_polynomial.fitted_model, grid)
    spline_curve = predict(best_spline.fitted_model, grid)
    selected_curve = predict(selected.fitted_model, grid)
    label = response_axis_label(response_column)

    fig, (fit_ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    fit_ax.scatter(x, y, s=32, color=INK, edgecolor="white", linewidth=0.55, zorder=3, label="Solved CFD cases")
    fit_ax.plot(
        grid,
        polynomial_curve,
        color=ORANGE,
        linewidth=1.8,
        linestyle="--",
        label=(
            f"Best polynomial: degree {best_polynomial.spec.degree} "
            f"(CV score {best_polynomial.selection_score:.3f})"
        ),
    )
    fit_ax.plot(
        grid,
        spline_curve,
        color=BLUE,
        linewidth=1.8,
        linestyle=":",
        label=(
            "Best spline: "
            f"lambda {best_spline.spec.smoothing_lambda:.0e} "
            f"(CV score {best_spline.selection_score:.3f})"
        ),
    )

    
    if selected.spec != best_spline.spec:
        fit_ax.plot(grid, selected_curve, color=NAVY, linewidth=2.7, label=f"Selected: {selected.spec.display_name}")
    else:
        fit_ax.plot(grid, selected_curve, color=NAVY, linewidth=2.7, alpha=0.78, label="Selected interpolation model")
    fig.suptitle(
        "AoA-to-response regression: held-out validation selects the interpolation model",
        x=0.126,
        y=0.992,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fit_ax.set_ylabel(label)
    fit_ax.grid(color=GRID, linewidth=0.8)
    fit_ax.legend(frameon=False, fontsize=8.6, loc="best")
    fit_ax.text(
        0.0,
        1.012,
        "Curves are shown only inside the solved AoA range; no extrapolation is produced.",
        transform=fit_ax.transAxes,
        fontsize=8.3,
        color=MUTED,
    )

    residual = y - selected.loo_predictions
    residual_ax.axhline(0.0, color=MUTED, linewidth=1.0)
    residual_ax.axhline(selected.loo_metrics["rmse"], color=BLUE, linewidth=1.0, linestyle=":")
    residual_ax.axhline(-selected.loo_metrics["rmse"], color=BLUE, linewidth=1.0, linestyle=":")
    residual_ax.scatter(x, residual, s=28, color=NAVY, edgecolor="white", linewidth=0.5, zorder=3)
    residual_ax.set_xlabel("Angle of attack [deg]")
    residual_ax.set_ylabel(f"LOO residual [{label}]")
    residual_ax.grid(color=GRID, linewidth=0.8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"aoa_pldb_regression.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_summary_markdown(
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    best_polynomial: EvaluatedModel,
    best_spline: EvaluatedModel,
    selected: EvaluatedModel,
    relative_improvement: float,
) -> None:
    """Write a concise human-readable interpretation and model-use boundary."""

    selected_is_spline = selected.spec.family == "smoothing_spline"
    conclusion = (
        "The cubic smoothing spline is selected because it clears the requested material-improvement threshold."
        if selected_is_spline
        else "The polynomial is retained because the spline did not clear the requested material-improvement threshold."
    )
    lines = [
        "# AoA-to-PLdB Regression Selection",
        "",
        f"Source: `{metadata['input_path']}`  ",
        f"Rows used: **{metadata['rows_used']}**  ",
        f"AoA range: **{metadata['predictor_range'][0]:.2f} to {metadata['predictor_range'][1]:.2f} deg**  ",
        f"Response: **{args.response_column}**  ",
    ]
    if metadata["run_family_counts"]:
        lines.append(f"Run families: {format_counts(metadata['run_family_counts'])}  ")
    if metadata["solve_end_time_s_counts"]:
        lines.append(f"Solver end times [s]: {format_counts(metadata['solve_end_time_s_counts'])}  ")
    if metadata["analysis_time_s_counts"]:
        lines.append(f"Explicit analysis times [s]: {format_counts(metadata['analysis_time_s_counts'])}")
    lines.extend([
        "",
        "## Selection result",
        "",
        f"- Best polynomial: **degree {best_polynomial.spec.degree}**; balanced CV score **{best_polynomial.selection_score:.3f} PLdB** "
        f"(LOO RMSE {best_polynomial.loo_metrics['rmse']:.3f}; triplet RMSE {best_polynomial.triplet_metrics['rmse']:.3f}).",
        f"- Best non-polynomial candidate: **cubic smoothing spline**, standardized lambda **{best_spline.spec.smoothing_lambda:.0e}**; "
        f"balanced CV score **{best_spline.selection_score:.3f} PLdB** "
        f"(LOO RMSE {best_spline.loo_metrics['rmse']:.3f}; triplet RMSE {best_spline.triplet_metrics['rmse']:.3f}).",
        f"- Relative improvement of the spline over the polynomial: **{relative_improvement:.1%}** "
        f"(selection threshold {args.min_relative_cv_improvement:.1%}).",
        f"- Selected model: **{selected.spec.display_name}**. {conclusion}",
        "",
        "## Validation method",
        "",
        "The selection score is the mean of leave-one-out RMSE and rolling contiguous-triplet RMSE. "
        "Triplet validation simulates three unresolved adjacent AoA stations and penalizes high-order polynomials "
        "that look good only when a single neighboring case remains in training.",
        "",
        "## Use boundary",
        "",
        "- Interpolate only inside the observed AoA range shown above; this script does not license extrapolation.",
        "- The default `incident_pldb` is the documented NASA-2025 incident convention. Do not mix it with the x1.9 ground, ideal rigid x2, or legacy PLdB columns.",
        "- The source sweep remains screening-level CFD/propagation evidence. A fitted minimum is a surrogate feature, not a validated physical optimum or flight-certified PLdB claim.",
        "- `--strict-mesh-only` is available for a mesh-status sensitivity run; it is intentionally not the default filter.",
        "",
    ])
    if metadata["mixed_run_family_or_time_provenance"]:
        lines.insert(
            -1,
            "- The full input combines documented run families and solve/sample times. The model describes that merged "
            "screening sweep; it does not erase those CFD-provenance differences.",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    best_polynomial: EvaluatedModel,
    best_spline: EvaluatedModel,
    selected: EvaluatedModel,
    relative_improvement: float,
    output_dir: Path,
) -> None:
    """Write provenance, selection criteria, metrics, and output paths as JSON."""

    def model_payload(result: EvaluatedModel) -> dict[str, Any]:
        return {
            "label": result.spec.label,
            "family": result.spec.family,
            "polynomial_degree": result.spec.degree,
            "smoothing_lambda_standardized": result.spec.smoothing_lambda,
            "selection_score_rmse": result.selection_score,
            "training": result.train_metrics,
            "leave_one_out": result.loo_metrics,
            "contiguous_triplet": result.triplet_metrics,
        }

    payload = {
        "source": metadata,
        "modeling_scope": {
            "predictor_column": args.predictor_column,
            "predictor_unit": "deg" if args.predictor_column == "aoa_deg" else "source column unit",
            "response_column": args.response_column,
            "response_unit": "PLdB" if "pldb" in args.response_column.lower() else "source column unit",
            "interpolation_only": True,
        },
        "selection_method": {
            "score": "mean(leave-one-out RMSE, rolling contiguous-triplet RMSE)",
            "spline_replacement_threshold": args.min_relative_cv_improvement,
            "best_polynomial": model_payload(best_polynomial),
            "best_smoothing_spline": model_payload(best_spline),
            "relative_spline_improvement": relative_improvement,
            "selected_model": model_payload(selected),
        },
        "outputs": {
            "model_comparison": str(output_dir / "model_comparison.csv"),
            "observed_predictions": str(output_dir / "predictions.csv"),
            "fitted_curve": str(output_dir / "fitted_curve.csv"),
            "figure_png": str(output_dir / "aoa_pldb_regression.png"),
            "figure_svg": str(output_dir / "aoa_pldb_regression.svg"),
            "summary_markdown": str(output_dir / "fit_summary.md"),
        },
        "caveats": [
            "The fitted curve is restricted to interpolation across the observed AoA range.",
            "The default response is the NASA-2025 incident PLdB convention and must not be mixed with other PLdB conventions.",
            "This screening-level CFD/propagation sweep is not external validation of an absolute physical optimum.",
        ]
        + (
            [
                "The input combines documented CFD run families and solve/sample times; the regression is a merged-screening-sweep surrogate."
            ]
            if metadata["mixed_run_family_or_time_provenance"]
            else []
        ),
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.min_relative_cv_improvement) or args.min_relative_cv_improvement < 0.0:
        raise ValueError("--min-relative-cv-improvement must be a finite value >= 0")

    frame, metadata = load_and_validate_data(args)
    x = frame["_predictor"].to_numpy(dtype=float)
    y = frame["_response"].to_numpy(dtype=float)

    results = [evaluate_spec(spec, x, y) for spec in candidate_specs(args.max_polynomial_degree, args.spline_lambdas)]
    best_polynomial, best_spline, selected, improvement = choose_models(
        results, args.min_relative_cv_improvement
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame(results, best_polynomial, best_spline, selected).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    observed = frame.drop(columns=["_predictor", "_response"]).copy()
    observed["best_polynomial_fit"] = predict(best_polynomial.fitted_model, x)
    observed["best_polynomial_loocv_prediction"] = best_polynomial.loo_predictions
    observed["best_polynomial_loocv_residual"] = y - best_polynomial.loo_predictions
    observed["best_spline_fit"] = predict(best_spline.fitted_model, x)
    observed["best_spline_loocv_prediction"] = best_spline.loo_predictions
    observed["best_spline_loocv_residual"] = y - best_spline.loo_predictions
    observed["selected_fit"] = predict(selected.fitted_model, x)
    observed["selected_loocv_prediction"] = selected.loo_predictions
    observed["selected_loocv_residual"] = y - selected.loo_predictions
    observed["selected_triplet_cv_prediction_mean"] = selected.triplet_mean_predictions
    observed["selected_triplet_cv_prediction_count"] = selected.triplet_prediction_counts
    observed.to_csv(output_dir / "predictions.csv", index=False)

    grid = np.linspace(float(x.min()), float(x.max()), 600)
    pd.DataFrame(
        {
            args.predictor_column: grid,
            "best_polynomial_prediction": predict(best_polynomial.fitted_model, grid),
            "best_smoothing_spline_prediction": predict(best_spline.fitted_model, grid),
            "selected_prediction": predict(selected.fitted_model, grid),
        }
    ).to_csv(output_dir / "fitted_curve.csv", index=False)

    write_figure(output_dir, x, y, args.response_column, best_polynomial, best_spline, selected)
    write_summary_markdown(
        output_dir / "fit_summary.md",
        args,
        metadata,
        best_polynomial,
        best_spline,
        selected,
        improvement,
    )
    write_json_summary(
        output_dir / "fit_summary.json",
        args,
        metadata,
        best_polynomial,
        best_spline,
        selected,
        improvement,
        output_dir,
    )

    print(f"Best polynomial: degree {best_polynomial.spec.degree} (CV score {best_polynomial.selection_score:.3f})")
    print(
        "Best smoothing spline: "
        f"lambda {best_spline.spec.smoothing_lambda:.0e} (CV score {best_spline.selection_score:.3f})"
    )
    print(f"Selected model: {selected.spec.display_name} (spline improvement {improvement:.1%})")
    print(f"Wrote regression artifacts to {output_dir}")

    if selected.spec.family == "smoothing_spline":
        spline_obj = selected.fitted_model.spline
        print("\n--- Spline Model Parameters ---")
        print(f"Standardized Knots (t): {spline_obj.t}")
        print(f"B-Spline Coefficients (c): {spline_obj.c}")
        print(f"Spline Degree (k): {spline_obj.k}")
        print(f"X-min (Shift): {selected.fitted_model.x_min}")
        print(f"X-span (Scale): {selected.fitted_model.x_span}")


if __name__ == "__main__":
    main()
