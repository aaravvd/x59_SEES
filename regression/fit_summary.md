# AoA-to-PLdB Regression Selection

Source: `/Users/aaravdixit/Downloads/complete_aoa_beta_sweep.csv`  
Rows used: **101**  
AoA range: **0.00 to 5.00 deg**  
Response: **incident_pldb**  
Run families: `baseline_t5`: 87; `relaxed_skew_recovery_t1_sample_t0p8`: 8; `relaxed_skew_recovery_t0p8`: 4; `endpoint_recovery_t1_sample_t0p8`: 2  
Solver end times [s]: `5.0`: 87; `1.0`: 10; `0.8`: 4  
Explicit analysis times [s]: `missing`: 87; `0.8`: 14

## Selection result

- Best polynomial: **degree 9**; balanced CV score **0.823 PLdB** (LOO RMSE 0.746; triplet RMSE 0.899).
- Best non-polynomial candidate: **cubic smoothing spline**, standardized lambda **1e-06**; balanced CV score **0.471 PLdB** (LOO RMSE 0.409; triplet RMSE 0.533).
- Relative improvement of the spline over the polynomial: **42.8%** (selection threshold 5.0%).
- Selected model: **cubic smoothing spline (lambda=1e-06)**. The cubic smoothing spline is selected because it clears the requested material-improvement threshold.

## Validation method

The selection score is the mean of leave-one-out RMSE and rolling contiguous-triplet RMSE. Triplet validation simulates three unresolved adjacent AoA stations and penalizes high-order polynomials that look good only when a single neighboring case remains in training.

## Use boundary

- Interpolate only inside the observed AoA range shown above; this script does not license extrapolation.
- The default `incident_pldb` is the documented NASA-2025 incident convention. Do not mix it with the x1.9 ground, ideal rigid x2, or legacy PLdB columns.
- The source sweep remains screening-level CFD/propagation evidence. A fitted minimum is a surrogate feature, not a validated physical optimum or flight-certified PLdB claim.
- `--strict-mesh-only` is available for a mesh-status sensitivity run; it is intentionally not the default filter.
- The full input combines documented run families and solve/sample times. The model describes that merged screening sweep; it does not erase those CFD-provenance differences.
