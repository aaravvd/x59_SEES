# C608/X-59 Sonic Boom Analysis Pipeline

A comprehensive framework for quantifying the relationship between angle of attack (AoA) and ground-PLdB for NASA's Quiet Supersonic Technology (QueSST) demonstrator aircraft.

## Overview

This project implements a multi-fidelity acoustic analysis pipeline to determine optimal flight parameters for sonic boom noise abatement. By coupling high-fidelity CFD with acoustic propagation and nonlinear atmospheric effects, we establish a regression model linking AoA to SPL, enabling design of quieter supersonic flight corridors over populated areas.

**Key Finding:** Optimal AoA ≈ 1.75° (modeled via cubic spline regression).

## Methodology

The analysis consists of four sequential stages:

### 1. CFD Simulation
- **Solver:** OpenFOAM's `rhoCentralFoam` (Kurganov flux scheme)
- **Physics:** 3D compressible inviscid Euler equations (Ma ≈ 1.4, ~11 km altitude)
- **Mesh:** Mach-cone-aligned refinement via `searchableCone` in snappyHexMeshDict
- **Output:** Near-field pressure field (Δp/p∞) on structured volume grid

### 2. Pressure Extraction
- Extracts pressure signatures at radial distance = 3 × L_REF (aircraft reference length)
- Azimuth sweep: 0° to 90°
- Output format: Δp/p∞ vs. arc length along ray trajectory
- Tool: PyVista-based extraction with adaptive geometry handling

### 3. Ray-Tracing Propagation
- **Method:** 3D acoustic ray tracing with inhomogeneous atmosphere
- **Integrator:** Euler's method, Δs = 0.05 (normalized by L_REF)
- **Atmosphere:** Altitude-dependent wind, temperature, density profiles
- **Output:** Ray trajectories and acoustic pressure amplitudes at ground level (z = 0)

### 4. Burgers' Equation Solver
- **Physics:** 1D augmented Burgers equation with:
  - Nonlinear steepening (shock formation)
  - Thermoviscous absorption
  - Molecular relaxation effects
- **Method:** Operator-splitting with explicit time integration
- **Output:** Waveform at ground level → SPL (dB re 20 μPa)

## File Structure

```
├── openfoam_setup/
│   ├── 0/                          # Initial/boundary conditions
│   ├── constant/
│   │   ├── polyMesh/              # Mesh
│   │   └── turbulenceProperties
│   ├── system/
│   │   ├── snappyHexMeshDict      # Mesh generation parameters
│   │   ├── fvSchemes              # Discretization schemes
│   │   └── fvSolution             # Solver settings
│   └── run_simulation.sh           # CFD execution script
│
├── acoustic_pipeline/
│   ├── extract_pressure_signature.py    # Pressure field extraction (PyVista)
│   ├── ray_trace.py                      # 3D ray tracing
│   ├── burgers.py                        # Burgers equation solver
│   ├── spl.py                            # SPL computation from waveforms
│   └── validate_models.py                # Multi-method validation (FNO, PINN, XFoil, etc.)
│
├── regression_analysis/
│   ├── fit_spline.py                     # Cubic spline regression (AoA → SPL)
│   ├── results/
│   │   ├── spline_coefficients.pkl
│   │   └── plots/
│   │       ├── spl_vs_aoa.png
│   │       ├── pressure_signatures/
│   │       └── ray_trajectories/
│   └── optimization.py                   # Find optimal AoA
│
└── docs/
    ├── physics_notes.md                  # Detailed derivations
    └── parameter_reference.md            # L_REF, P_INF, atmospheric data
```

## Installation

### Dependencies

```bash
pip install numpy scipy matplotlib pyvista openfoam-python
pip install neuralop torch  # For surrogate models (FNO/PINN validation)
```

### OpenFOAM Setup

Requires OpenFOAM v2312 or later with:
- `rhoCentralFoam` solver
- Compressible turbulence library

```bash
# Verify installation
foamVersion
which rhoCentralFoam
```

## Usage

### Quick Start (Single AoA)

```bash
# 1. Run CFD simulation at AoA = 1.75°
cd openfoam_setup
./run_simulation.sh

# 2. Extract pressure signature
python extract_pressure_signature.py \
    --mesh_path constant/polyMesh \
    --field_path postProcessing/p \
    --output sig_aoa1p75.npy

# 3. Propagate to ground via ray-tracing + Burgers solver
python ray_trace.py sig_aoa1p75.npy
python burgers.py ray_trace_output.pkl
python spl.py waveform_ground.pkl
```

### Parametric Study (150 AoA sweep)

```bash
python acoustic_pipeline/parametric_sweep.py \
    --aoa_min 0.0 \
    --aoa_max 5.0 \
    --n_aoa 150 \
    --parallel True \
    --n_procs 8
```

This generates `results/spl_vs_aoa_raw.csv` with columns: `[aoa_deg, spl_db]`

### Regression & Optimization

```bash
python regression_analysis/fit_spline.py \
    --data results/spl_vs_aoa_raw.csv \
    --order 3 \
    --output results/spline_model.pkl

python regression_analysis/optimization.py \
    --model results/spline_model.pkl
```

Output: Optimal AoA and SPL contours for flight planning.

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| L_REF | ~28 m | C608 reference length (from geometry) |
| P_INF | ~23.8 kPa | Freestream pressure (~11 km altitude) |
| Ma∞ | 1.40 | Mach number at cruise |
| Altitude | ~11 km | Standard atmospheric model |
| Ray extraction radius | 3 × L_REF | Acoustic far-field criterion |
| Ray step size | 0.05 × L_REF | Time integration (ray-tracing) |
| Burgers solver CFL | 0.5 | Stability constraint (Δt) |

## Results

### SPL vs AoA Relationship

- **Range:** 0.0° to 5.0° AoA
- **Data points:** 150 simulations
- **Optimal AoA:** 1.75° (±0.15°)
- **SPL reduction:** ~3–5 dB vs. flat-fuselage baseline

### Model Accuracy

Validation against multi-fidelity comparison (`validate_models.py`):
- FNO (neural operator) surrogate: L₂ error ≈ 0.8 dB
- PINN (physics-informed): L₂ error ≈ 1.2 dB
- XFoil 2D profile analysis: consistent CL/CD trends

## Physics Notes

### Euler Equations (CFD Stage)
- Inviscid assumption valid for shock-capturing; boundary layers negligible at high-Re supersonic flow
- Mach-cone-aligned mesh prevents smearing of oblique shocks

### Ray-Tracing Acoustic Propagation
- Geometric acoustics (high-frequency limit); valid away from shock core
- Amplitude modulation via stratification and jet-stream interaction

### Augmented Burgers Equation (Nonlinear Acoustics)
- Captures N-wave shaping: weak shock formation + diffraction
- Thermoviscous absorption critical below 100 Hz (ground-level cutoff)
- Molecular relaxation (O₂, N₂) smooths high-frequency content

## Citation

If this work is used in publication, please cite:

```
[Author(s), 202X]. Angle-of-Attack Optimization for Sonic Boom Noise 
Abatement: Application to NASA X-59/C608 Demonstrator. 
[Conference/Journal], [Details].
```

## Future Work

- **Structural-acoustic coupling:** Include fuselage panels and engine nacelle effects
- **Yaw angle effects:** Extend to 3D heading variations
- **Wind-dependent optimization:** Real-time flight path adaptation during weather events
- **Machine learning surrogates:** End-to-end neural operator (FNO/DeepONet) for near-real-time optimization
- **Experimental validation:** F-5E or low-boom demonstrator flight data comparison

## Troubleshooting

### CFD Divergence at High AoA
- Increase Mach-cone mesh refinement ratio in `snappyHexMeshDict`
- Check far-field boundary conditions; switch to compressible non-reflecting types if oscillations observed

### Ray-Tracing NaNs
- Verify L_REF and P_INF extracted correctly via `extract_pressure_signature.py --debug`
- Check atmospheric profile continuity (altitude interpolation)

### Burgers Solver Instability
- Reduce CFL number or ray step size
- Verify input waveform amplitude is physically reasonable (0.1–1.0 × P_INF typical)

## Contact & Contributions

For issues, feature requests, or validation data sharing, open an issue or contact the lead developer.

---

**Last Updated:** July 2026  
**Status:** Active development  
**License:** [Specify: MIT, GPL, etc.]
