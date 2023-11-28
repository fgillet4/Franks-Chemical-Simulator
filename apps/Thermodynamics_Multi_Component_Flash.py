import tkinter as tk
from tkinter import ttk
from thermo import *
from scipy.constants import atm
import numpy as np

class ThermoApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Thermo-Scipy GUI")
        self.geometry("600x450")

        # Input Fields
        self.T_label = ttk.Label(self, text="Temperature (K):")
        self.T_entry = ttk.Entry(self)
        
        self.P_label = ttk.Label(self, text="Pressure (Pa):")
        self.P_entry = ttk.Entry(self)
        
        self.zs_label = ttk.Label(self, text="Molar Fractions (comma separated):")
        self.zs_entry = ttk.Entry(self, width=50)
        
        # Calculate Button
        self.calc_btn = ttk.Button(self, text="Calculate", command=self.calculate)
        
        # Output Labels
        self.output_label = ttk.Label(self, text="Results:")
        self.phases_label = ttk.Label(self)
        self.densities_label = ttk.Label(self)
        self.fugacity_label = ttk.Label(self)
        
        # Layout
        self.T_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.T_entry.grid(row=0, column=1, padx=10, pady=5)

        self.P_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.P_entry.grid(row=1, column=1, padx=10, pady=5)

        self.zs_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.zs_entry.grid(row=2, column=1, padx=10, pady=5)
        
        self.calc_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.output_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.phases_label.grid(row=5, column=0, columnspan=2, padx=10)
        self.densities_label.grid(row=6, column=0, columnspan=2, padx=10)
        self.fugacity_label.grid(row=7, column=0, columnspan=2, padx=10)
        
    def calculate(self):
        # Get inputs
        T = float(self.T_entry.get())
        P = float(self.P_entry.get())
        zs = [float(z) for z in self.zs_entry.get().split(",")]

        # The calculation based on the provided code
        pure_constants = ChemicalConstantsPackage.constants_from_IDs(
            ['methane', 'ethane', 'propane', 'n-butane', 'n-pentane', 'n-hexane'])

        pseudos = ChemicalConstantsPackage(Tcs=[606.28,825.67], Pcs=[25.42*atm, 14.39*atm],
                                           omegas=[0.4019, 0.7987], MWs=[140.0, 325.0])
        constants = pure_constants + pseudos

        properties = PropertyCorrelationsPackage(constants=constants)
        
        kijs = [[0.0, 0.002, 0.017, 0.015, 0.02, 0.039, 0.05, 0.09],
                [0.002, 0.0, 0.0, 0.025, 0.01, 0.056, 0.04, 0.055],
                [0.017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01],
                [0.015, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.039, 0.056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.09, 0.055, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]

        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

        gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
        liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
        liq2 = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

        flashN = FlashVLN(constants, properties, liquids=[liq, liq2], gas=gas)
        res = flashN.flash(T=T, P=P, zs=zs)

        # Set the output to the labels
        self.phases_label.config(text=f"There are {res.phase_count} phases present")
        self.densities_label.config(text=f"Mass densities of each liquid are {res.liquid0.rho_mass()} and {res.liquid0.rho_mass()} kg/m^3")
        max_fugacity_err = np.max(np.abs(1-np.array(res.liquid0.fugacities())/res.liquid1.fugacities()))
        self.fugacity_label.config(text=f"The maximum relative difference in fugacity is {max_fugacity_err:.8f}.")

if __name__ == "__main__":
    app = ThermoApp()
    app.mainloop()
