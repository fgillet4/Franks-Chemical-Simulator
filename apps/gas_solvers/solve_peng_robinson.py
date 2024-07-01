import numpy as np
from scipy.optimize import root

def PR_EoS(V, T, P, Tc, omega, k, a, b):
    """
    Calculates the molar volume for a mixture of components using the Peng-Robinson (PR) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): PR equation of state parameter of components.
    a (array): PR equation of state parameter of components.
    b (array): PR equation of state parameter of components.

    Returns:
    float: Residual function for the PR equation of state.
    """
    def f_PR(V, T, Tc, omega, k, a, b):
        """
        Peng-Robinson equation of state for a single component.
        """
        Tr = T / Tc
        return (a / V**2) * (1 - k * (1 - np.sqrt(Tr))) - P / (R * T) + (b * P) / (V * R * T)

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = f_PR(V, T, Tc[i], omega[i], k[i], a[i], b[i])
    return np.sum(Z)

def PR_EoS_solver(T, P, Tc, omega, k, a, b):
    """
    Solves the Peng-Robinson (PR) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): PR equation of state parameter of components.
    a (array): PR equation of state parameter of components.
    b (array): PR equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the PR equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) / P
    V = root(PR_EoS, V0, args=(T, P, Tc, omega, k, a, b)).x[0]
    return V
