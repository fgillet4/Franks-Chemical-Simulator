import numpy as np

def HenrysLawModel(T, x, z, H1, H2):
    """
    Calculates the activity coefficients for a binary mixture using the Henry's Law Model.

    Parameters:
    T (float): Temperature (K).
    x (array): Mole fractions of components in the mixture.
    z (array): Overall molality of components in the mixture.
    H1 (array): Henry's Law constant for component 1.
    H2 (array): Henry's Law constant for component 2.

    Returns:
    array: Activity coefficients of components in the mixture.
    """
    activity = np.zeros(2)
    for i in range(2):
        activity[i] = H1[i] * x[i] / (H2[i] * (1 - x[i]))
    return activity

