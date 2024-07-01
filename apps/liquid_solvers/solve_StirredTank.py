import numpy as np

def StirredTank(T, x, z, Tc, Vc, omega, k1, k2):
    """
    Calculates the activity coefficients for a binary mixture using the Stirred Tank Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    Tc (array): Critical temperature (K) of components.
    Vc (array): Critical volume (m^3/mol) of components.
    omega (array): Acentric factor of components.
    k1 (array): Stirred Tank binary interaction parameters for component 1.
    k2 (array): Stirred Tank binary interaction parameters for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = np.exp(np.sum(x * np.log(z * (1 + k1[i] * x + k2[i] * x**2))))
    return activity
