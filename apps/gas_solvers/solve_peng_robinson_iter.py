import numpy as np

def peng_robinson(P, T, z, w, Tr, Pr):
    # Constants
    R = 8.314 # J/mol*K
    a = (0.45724 * R**2 * Tr**2) / Pr
    b = 0.0778 * R * Tr / Pr

    # Cubic equation coefficients
    A = a * P / (R * T)**2
    B = b * P / (R * T)

    # Cubic equation
    coeffs = [1, -1, -A + B - B**2, -A * B + B**2]
    roots = np.roots(coeffs)

    # Find the real root
    for root in roots:
        if np.isreal(root):
            v = np.real(root)
            break

    # Ideal gas contribution
    alpha = (1 + (0.37464 + 1.54226 * w - 0.26992 * w**2) * (1 - (T / Tr)**0.5))**2
    ideal = z - 1 - np.log(z - B)

    # Residual contribution
    residual = -(2 * np.sqrt(a * B) / (R * T)) * np.log((v + (1 + np.sqrt(2)) * B) / (v + (1 - np.sqrt(2)) * B))

    return ideal + residual * alpha

def solve_peng_robinson(P, T, z, w, Tr, Pr, tolerance=1e-6):
    # Initial estimate for v
    v = 0.001

    # Iterate until the desired tolerance is reached
    while True:
        f = peng_robinson(P, T, z, w, Tr, Pr)
        df = (peng_robinson(P, T, z, w, Tr, Pr, v + tolerance) - f) / tolerance
        v_new = v - f / df

        if abs(v_new - v) < tolerance:
            break

        v = v_new

    return v
