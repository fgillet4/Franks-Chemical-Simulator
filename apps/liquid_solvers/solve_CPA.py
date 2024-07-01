import numpy as np
from scipy.optimize import root
#Cubic-Plus-Association (CPA) Equation of State for a mixture of components
def CPA_EoS(V, T, P, Zc, Tc, omega, k, m, a, b):
    """
    Calculates the molar volume for a mixture of components using the Cubic-Plus-Association (CPA) Equation of State.

    Parameters:
    V (float): Molar volume (m^3/mol).
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Zc (array): Critical compressibility factor of components.
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): Association constant of components.
    m (array): Association parameter of components.
    a (array): CPA cubic equation of state parameter of components.
    b (array): CPA cubic equation of state parameter of components.

    Returns:
    float: Residual function for the CPA equation of state.
    """
    def f_cubic(V, T, a, b):
        """
        Cubic equation of state for a single component.
        """
        return (a / (V - b)) - (8 * T / (3 * V**2))

    def f_association(V, T, Zc, Tc, omega, k, m):
        """
        Association term for a single component.
        """
        Tr = T / Tc
        return np.exp(-m * (1 - np.sqrt(Tr))**2) * (Zc - 1 - k * V / (T * Zc))

    N = len(Tc)
    Z = np.zeros(N)
    for i in range(N):
        Z[i] = root(f_cubic, Zc[i], args=(T, a[i], b[i])).x[0] + f_association(V, T, Zc[i], Tc[i], omega[i], k[i], m[i])
    return np.sum(Z) - P * V / (R * T)

def CPA_EoS_solver(T, P, Zc, Tc, omega, k, m, a, b):
    """
    Solves the Cubic-Plus-Association (CPA) Equation of State for a mixture of components.

    Parameters:
    T (float): Temperature (K).
    P (float): Pressure (Pa).
    Zc (array): Critical compressibility factor of components.
    Tc (array): Critical temperature (K) of components.
    omega (array): Acentric factor of components.
    k (array): Association constant of components.
    m (array): Association parameter of components.
    a (array): CPA cubic equation of state parameter of components.
    b (array): CPA cubic equation of state parameter of components.

    Returns:
    float: Molar volume (m^3/mol) that satisfies the CPA equation of state for the given conditions.
    """
    R = 8.314
    V0 = (R * T) /P
    V = root(CPA_EoS, V0, args=(T, P, Zc, Tc, omega, k, m, a, b)).x[0]
    return V
"""
This code defines two functions `CPA_EoS` and `CPA_EoS_solver`. 
The `CPA_EoS` function calculates the residual function for the CPA equation of state,
The `CPA_EoS_solver` function solves the CPA equation of state for a mixture of components. 
The input parameters for the `CPA_EoS_solver` function include:
temperature, pressure, critical compressibility factor, critical temperature, acentric factor, 
association constant, association parameter, and CPA cubic equation of state parameters 
for each component. The function returns the molar volume that satisfies the CPA equation of state 
#for the given conditions.
To use the `CPA_EoS_solver` function, you would provide the necessary input parameters and call the function. The function will return the molar volume that satisfies the CPA equation of state for the given conditions.
"""
