import tkinter as tk
from tkinter import ttk, StringVar
from math import log
from thermo import ChemicalConstantsPackage
from scipy.constants import psi
from math import log, exp
import numpy as np
from scipy.constants import psi
from thermo import *
from chemicals import *

# ... (Include all the functions you provided above: Tc_Kesler_Lee_SG_Tb, Pc_Kesler_Lee_SG_Tb, etc.)
def Tc_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates critical temperature of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        T_c = 341.7 + 811.1SG + [0.4244 + 0.1174SG]T_b
        + \frac{[0.4669 - 3.26238SG]10^5}{T_b}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    Tc : float
        Estimated critical temperature [K]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R.

    >>> Tc_Kesler_Lee_SG_Tb(0.7365, 365.555)
    545.0124354151242

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    Tc = 341.7 + 811.1*SG + (0.4244 + 0.1174*SG)*Tb + ((0.4669 - 3.26238*SG)*1E5)/Tb
    Tc = 5/9.*Tc # R to K
    return Tc

def Pc_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates critical pressure of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        \ln(P_c) = 8.3634 - \frac{0.0566}{SG} - \left[0.24244 + \frac{2.2898}
        {SG} + \frac{0.11857}{SG^2}\right]10^{-3}T_b
        + \left[1.4685 + \frac{3.648}{SG} + \frac{0.47227}{SG^2}\right]
        10^{-7}T_b^2-\left[0.42019 + \frac{1.6977}{SG^2}\right]10^{-10}T_b^3

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    Pc : float
        Estimated critical pressure [Pa]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine and psi.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> Pc_Kesler_Lee_SG_Tb(0.7365, 365.555)
    3238323.346840464

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    Pc = exp(8.3634 - 0.0566/SG - (0.24244 + 2.2898/SG + 0.11857/SG**2)*1E-3*Tb
    + (1.4685 + 3.648/SG + 0.47227/SG**2)*1E-7*Tb**2
    -(0.42019 + 1.6977/SG**2)*1E-10*Tb**3)
    Pc = Pc*psi
    return Pc

def MW_Kesler_Lee_SG_Tb(SG, Tb):
    r'''Estimates molecular weight of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_.

    .. math::
        MW = -12272.6 + 9486.4SG + [4.6523 - 3.3287SG]T_b + [1-0.77084SG
        - 0.02058SG^2]\left[1.3437 - \frac{720.79}{T_b}\right]\frac{10^7}{T_b}
        + [1-0.80882SG + 0.02226SG^2][1.8828 - \frac{181.98}{T_b}]
        \frac{10^{12}}{T_b^3}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]

    Returns
    -------
    MW : float
        Estimated molecular weight [g/mol]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> MW_Kesler_Lee_SG_Tb(0.7365, 365.555)
    98.70887589833501

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    Tb = 9/5.*Tb # K to R
    MW = (-12272.6 + 9486.4*SG + (4.6523 - 3.3287*SG)*Tb + (1.-0.77084*SG - 0.02058*SG**2)*
    (1.3437 - 720.79/Tb)*1E7/Tb + (1.-0.80882*SG + 0.02226*SG**2)*
    (1.8828 - 181.98/Tb)*1E12/Tb**3)
    return MW

def omega_Kesler_Lee_SG_Tb_Tc_Pc(SG, Tb, Tc=None, Pc=None):
    r'''Estimates accentric factor of a hydrocarbon compound or petroleum
    fraction using only its specific gravity and boiling point, from
    [1]_ as presented in [2]_. If Tc and Pc are provided, the Kesler-Lee
    routines for estimating them are not used.

    For Tbr > 0.8:
    .. math::
        \omega = -7.904 + 0.1352K - 0.007465K^2 + 8.359T_{br}
        + ([1.408-0.01063K]/T_{br})

    Otherwise:
    .. math::
        \omega = \frac{-\ln\frac{P_c}{14.7} - 5.92714 + \frac{6.09648}{T_{br}}
        + 1.28862\ln T_{br} - 0.169347T_{br}^6}{15.2518 - \frac{15.6875}{T_{br}}
         - 13.4721\ln T_{br} + 0.43577T_{br}^6}

        K = \frac{T_b^{1/3}}{SG}

        T_{br} = \frac{T_b}{T_c}

    Parameters
    ----------
    SG : float
        Specific gravity of the fluid at 60 degrees Farenheight [-]
    Tb : float
        Boiling point the fluid [K]
    Tc : float, optional
        Estimated critical temperature [K]
    Pc : float, optional
        Estimated critical pressure [Pa]

    Returns
    -------
    omega : float
        Acentric factor [-]

    Notes
    -----
    Model shows predictions for Tc, Pc, MW, and omega.
    Original units in degrees Rankine and psi.

    Examples
    --------
    Example 2.2 from [2]_, but with K instead of R and Pa instead of psi.

    >>> omega_Kesler_Lee_SG_Tb_Tc_Pc(0.7365, 365.555, 545.012, 3238323.)
    0.306392118159797

    References
    ----------
    .. [1] Kesler, M. G., and B. I. Lee. "Improve Prediction of Enthalpy of
       Fractions." Hydrocarbon Processing (March 1976): 153-158.
    .. [2] Ahmed, Tarek H. Equations of State and PVT Analysis: Applications
       for Improved Reservoir Modeling. Gulf Pub., 2007.
    '''
    if Tc is None:
        Tc = Tc_Kesler_Lee_SG_Tb(SG, Tb)
    if Pc is None:
        Pc = Pc_Kesler_Lee_SG_Tb(SG, Tb)
    Tb = 9/5.*Tb # K to R
    Tc = 9/5.*Tc # K to R
    K = Tb**(1/3.)/SG
    Tbr = Tb/Tc
    if Tbr > 0.8:
        omega = -7.904 + 0.1352*K - 0.007465*K**2 + 8.359*Tbr + ((1.408-0.01063*K)/Tbr)
    else:
        omega = ((-log(Pc/101325.) - 5.92714 + 6.09648/Tbr + 1.28862*log(Tbr)
        - 0.169347*Tbr**6) / (15.2518 - 15.6875/Tbr - 13.4721*log(Tbr) +0.43577*Tbr**6))
    return omega

def calculate_properties():
    # Retrieve values from the entry widgets
    SG = float(sg_var.get())
    Tb = float(tb_var.get())
    
    # Calculate properties
    Tc = Tc_Kesler_Lee_SG_Tb(SG, Tb)
    Pc = Pc_Kesler_Lee_SG_Tb(SG, Tb)
    MW = MW_Kesler_Lee_SG_Tb(SG, Tb)
    omega = omega_Kesler_Lee_SG_Tb_Tc_Pc(SG, Tb, Tc, Pc)
    
    # Update the results in the output labels
    tc_result_var.set(f"Tc: {Tc:.2f} K")
    pc_result_var.set(f"Pc: {Pc:.2f} Pa")
    mw_result_var.set(f"MW: {MW:.2f} g/mol")
    omega_result_var.set(f"ω: {omega:.4f}")

# Create the main application window
app = tk.Tk()
app.title("Petroleum Pseudocomponents Properties Calculator")

# Create and place the input labels and fields
sg_label = ttk.Label(app, text="Specific Gravity:")
sg_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
sg_var = StringVar()
sg_entry = ttk.Entry(app, textvariable=sg_var)
sg_entry.grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)

tb_label = ttk.Label(app, text="Boiling Point (K):")
tb_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
tb_var = StringVar()
tb_entry = ttk.Entry(app, textvariable=tb_var)
tb_entry.grid(column=1, row=1, sticky=tk.EW, padx=5, pady=5)

# Create and place the button to perform the calculations
calculate_button = ttk.Button(app, text="Calculate", command=calculate_properties)
calculate_button.grid(column=0, row=2, columnspan=2, pady=10)

# Create and place labels to display the results
tc_result_var = StringVar()
tc_result_label = ttk.Label(app, textvariable=tc_result_var)
tc_result_label.grid(column=0, row=3, columnspan=2, sticky=tk.W, padx=5, pady=5)

pc_result_var = StringVar()
pc_result_label = ttk.Label(app, textvariable=pc_result_var)
pc_result_label.grid(column=0, row=4, columnspan=2, sticky=tk.W, padx=5, pady=5)

mw_result_var = StringVar()
mw_result_label = ttk.Label(app, textvariable=mw_result_var)
mw_result_label.grid(column=0, row=5, columnspan=2, sticky=tk.W, padx=5, pady=5)

omega_result_var = StringVar()
omega_result_label = ttk.Label(app, textvariable=omega_result_var)
omega_result_label.grid(column=0, row=6, columnspan=2, sticky=tk.W, padx=5, pady=5)

app.mainloop()
