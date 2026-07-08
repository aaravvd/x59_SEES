def compute_airfoil_forces(x, y, pressure, reference_pressure=None):
    """
    Compute lift and drag forces on an airfoil from pressure distribution.

    Parameters:
    -----------
    x : array-like
        x-coordinates of airfoil points (101 points)
    y : array-like
        y-coordinates of airfoil points (101 points)
    pressure : array-like
        Pressure values at each point (101 values)
    reference_pressure : float, optional
        Reference pressure (freestream static pressure). If None, uses
        the average pressure at the trailing edge as reference.

    Returns:
    --------
    dict : Dictionary containing:
        - 'lift': Total lift force (positive upward)
        - 'drag': Total drag force (positive downstream)
        - 'chord_length': Chord length of airfoil
        - 'reference_pressure': Reference pressure used

    Note:
    -----
    Reference pressure is the freestream static pressure far from the airfoil.
    If you don't know it, the function will estimate it from trailing edge pressure.
    """



    # Convert to numpy arrays
    x = x.numpy()
    y = y.numpy()
    pressure = pressure.numpy()
    # Validate input
    if len(x) != len(y) or len(x) != len(pressure):
        raise ValueError("x, y, and pressure arrays must have the same length")

    if len(x) != 101:
        print(f"Warning: Expected 101 points, got {len(x)} points")

    # Determine reference pressure if not provided
    if reference_pressure is None:
        # Estimate reference pressure from trailing edge
        # Find trailing edge (typically rightmost point)
        te_idx = np.argmax(x)

        # For better estimate, average pressures near trailing edge
        # Take points within 5% of chord from trailing edge
        chord = np.max(x) - np.min(x)
        te_region = x > (np.max(x) - 0.05 * chord)

        if np.sum(te_region) > 0:
            reference_pressure = np.mean(pressure[te_region])
        else:
            reference_pressure = pressure[te_idx]

    # Calculate differential pressure from reference
    dp = pressure #- reference_pressure

    # Initialize force components
    lift = 0.0
    drag = 0.0

    # Numerical integration using trapezoidal rule
    # For each panel between consecutive points
    for i in range(len(x) - 1):
        # Panel endpoints
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]

        # Panel length
        ds = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Average pressure on panel
        p_avg = 0.5 * (dp[i] + dp[i + 1])

        # Outward normal vector (pointing away from airfoil)
        # For a closed airfoil, normal points outward
        nx = -(y2 - y1) / ds
        ny = (x2 - x1) / ds
        # Force components on this panel
        # Pressure force = pressure * area * normal_vector
        fx = p_avg * ds * nx  # Force in x-direction
        fy = p_avg * ds * ny  # Force in y-direction

        # Accumulate forces
        drag += fx  # Drag is force in x-direction
        lift += fy  # Lift is force in y-direction

    # Calculate chord length (distance from leading to trailing edge)
    # Assume airfoil goes from x_min to x_max
    chord_length = np.max(x) - np.min(x)

    # Prepare results
    results = {
        'lift': lift,
        'drag': drag,
        'chord_length': chord_length,
        'reference_pressure': reference_pressure,
        'lift_per_unit_chord': lift / chord_length if chord_length > 0 else 0,
        'drag_per_unit_chord': drag / chord_length if chord_length > 0 else 0
    }

    return results
