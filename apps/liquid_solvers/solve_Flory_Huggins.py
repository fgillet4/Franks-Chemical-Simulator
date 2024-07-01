import numpy as np

def FloryHuggins(T, x, chi):
    """
    Calculates the activity coefficients for a binary mixture using the Flory-Huggins Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    chi (array): Flory-Huggins interaction parameter.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    activity[0] = np.exp(-chi * x[1])
    activity[1] = np.exp(-chi * x[0])
    return activity
