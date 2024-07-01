import numpy as np

def virial(P, T, z, B0, B1, B2, B3):
    # Cubic equation coefficients
    A = B0 + B1 / T + B2 / T**2 + B3 / T**3
    B = B1 + 2 * B2 / T + 3 * B3 / T**2
    C = B2 + 3 * B3 / T
    D = B3
    # Cubic equation
    coeffs = [1, -1, A - B * z + C * z**2 - D * z**3, -A * z + (B - C * z + D * z**2) * z - D * z**3]
    roots = np.roots(coeffs)
    # Find the real root
    for root in roots:
        if np.isreal(root):
            v = np.real(root)
            break
    return v

def solve_virial(P, T, z, B0, B1, B2, B3, tolerance=1e-6):
    # Initial estimate for v
    v = 0.001
    # Iterate until the desired tolerance is reached
    while True:
        f = virial(P, T, z, B0, B1, B2, B3)
        df = (virial(P, T, z, B0, B1, B2, B3, v + tolerance) - f) / tolerance
        v_new = v - f / df
        if abs(v_new - v) < tolerance:
            break
        v = v_new
    return v
