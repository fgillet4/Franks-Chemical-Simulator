import numpy as np

def NRTL(x, T, gamma, alpha, tau):
    """
    Calculates the activity coefficients for a binary mixture using the NRTL equation.

    Parameters:
    x (list or numpy array): Composition of the mixture (mol fraction).
    T (float): Temperature (K).
    gamma (numpy array): Matrix of binary interaction parameters.
    alpha (numpy array): Matrix of temperature-dependent parameters.
    tau (numpy array): Matrix of tau values.

    Returns:
    numpy array: Activity coefficients.
    """
    n = len(x)
    ln_gamma = np.zeros(n)
    for i in range(n):
        for j in range(n):
            ln_gamma[i] += x[j] * (gamma[i, j] + alpha[i, j] * (1 - np.exp(-tau[i, j] * (1 / T - 1 / 298.15))))
    return np.exp(ln_gamma)

def solve_NRTL(x, T, gamma, alpha, tau, tolerance=1e-6):
    """
    Solves the NRTL equation for a binary mixture.

    Parameters:
    x (list or numpy array): Composition of the mixture (mol fraction).
    T (float): Temperature (K).
    gamma (numpy array): Matrix of binary interaction parameters.
    alpha (numpy array): Matrix of temperature-dependent parameters.
    tau (numpy array): Matrix of tau values.
    tolerance (float, optional): Tolerance for the iteration (default is 1e-6).

    Returns:
    numpy array: Activity coefficients.
    """
    # Initial estimate for activity coefficients
    gamma_new = np.ones(len(x))
    # Iterate until the desired tolerance is reached
    while True:
        gamma = gamma_new
        gamma_new = NRTL(x, T, gamma, alpha, tau)
        if np.linalg.norm(gamma_new - gamma) < tolerance:
            break
    return gamma_new
