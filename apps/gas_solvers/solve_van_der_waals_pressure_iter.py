import numpy as np

def van_der_waals(P, V, a, b):
    """
    Calculates the pressure for a substance using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    V (float): Volume (m^3).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.

    Returns:
    float: Pressure (Pa).
    """
    return (P + a / V**2) * (V - b) - 8.314 * 298.15 / V

def solve_van_der_waals(V, a, b, tolerance=1e-6):
    """
    Solves the Van der Waals equation of state for a substance.

    Parameters:
    V (float): Volume (m^3).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    float: Pressure (Pa).
    """
    # Initial estimate for pressure
    P = 1e5
    # Iterate until the desired tolerance is reached
    while True:
        f = van_der_waals(P, V, a, b)
        df = (van_der_waals(P + tolerance, V, a, b) - f) / tolerance
        P_new = P - f / df
        if abs(P_new - P) < tolerance:
            break
        P = P_new
    return P
