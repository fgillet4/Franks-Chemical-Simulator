import numpy as np

def van_der_waals(P, V, a, b):
    """
    Calculates the pressure for a substance using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    V (float): Molar volume (m^3/mol).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.

    Returns:
    float: Pressure (Pa).
    """
    return (P + a / V**2) * (V - b) - 8.314 * 298.15 / V

def solve_van_der_waals_molar_volume(P, a, b, tolerance=1e-6):
    """
    Solves for the molar volume using the Van der Waals equation of state.

    Parameters:
    P (float): Pressure (Pa).
    a (float): Constant that depends on the substance.
    b (float): Constant that depends on the substance.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    float: Molar volume (m^3/mol).
    """
    # Initial estimate for molar volume
    V = 0.1
    # Iterate until the desired tolerance is reached
    while True:
        f = van_der_waals(P, V, a, b)
        df = (van_der_waals(P, V + tolerance, a, b) - f) / tolerance
        V_new = V - f / df
        if abs(V_new - V) < tolerance:
            break
        V = V_new
    return V
