import numpy as np

def UNIFAC(T, x, z, Tc, Vc, omega, q1, q2):
    """
    Calculates the activity coefficients for a binary mixture using the UNIFAC model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    Tc (array): Critical temperature (K) of components.
    Vc (array): Critical volume (m^3/mol) of components.
    omega (array): Acentric factor of components.
    q1 (array): UNIFAC binary interaction parameters for component 1.
    q2 (array): UNIFAC binary interaction parameters for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    k = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + k * (1 - np.sqrt(T / Tc)))**2
    a = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a[i, j] = alpha[i] * alpha[j] * np.exp((q1[i] * q2[j]) / (1 + k[i] * k[j]))
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = np.exp(np.sum(x * np.log(z * a[i, :])))
    return activity
