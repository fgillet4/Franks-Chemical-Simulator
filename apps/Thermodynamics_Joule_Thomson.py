import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS
import CoolProp
class JTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Joule-Thomson Effect Calculator")

        # Molecule input
        ttk.Label(self, text="Molecule:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecule_entry = ttk.Entry(self, width=30)
        self.molecule_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecule_entry.insert(0, "nitrogen")

        # Mole fraction input (though mostly this will be 1 for pure species)
        ttk.Label(self, text="Mole fraction:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fraction_entry = ttk.Entry(self, width=30)
        self.molar_fraction_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fraction_entry.insert(0, "1")

        # Initial conditions input
        ttk.Label(self, text="Initial Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.T1_entry = ttk.Entry(self, width=30)
        self.T1_entry.grid(row=2, column=1, padx=10, pady=5)
        self.T1_entry.insert(0, "300")

        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.P1_entry = ttk.Entry(self, width=30)
        self.P1_entry.grid(row=3, column=1, padx=10, pady=5)
        self.P1_entry.insert(0, "200")

        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.P2_entry = ttk.Entry(self, width=30)
        self.P2_entry.grid(row=4, column=1, padx=10, pady=5)
        self.P2_entry.insert(0, "1")

        # Calculate button
        self.btn_calculate = ttk.Button(self, text="Calculate Outlet Temperatures", command=self.calculate)
        self.btn_calculate.grid(row=5, column=0, columnspan=2, pady=20)

        # Results display
        self.result_text = tk.Text(self, height=5, width=40)
        self.result_text.grid(row=6, column=0, columnspan=2, padx=10, pady=5)
        self.result_text.insert(tk.END, "Results will be displayed here...")

    def calculate(self):
        fluid_input = self.molecule_entry.get().replace(" ", "").split(",")
        zs_input = self.molar_fraction_entry.get().replace(" ", "").split(",")

        if len(fluid_input) != len(zs_input):
            messagebox.showerror("Error", "Number of molecules and mole fractions do not match.")
            return

        try:
            zs = [float(z) for z in zs_input]
            constants, correlations = ChemicalConstantsPackage.from_IDs(fluid_input)  # Use fluid_input here

            T1 = float(self.T1_entry.get())
            P1 = float(self.P1_entry.get()) * bar
            P2 = float(self.P2_entry.get()) * bar

            # Calculate using high precision
            backend = 'HEOS'
            # Determine if it's a mixture or pure fluid
            if len(fluid_input) > 1:  # Mixture
                fluid_string = '&'.join([f"{f}[%] = {100*z}" for f, z in zip(fluid_input, zs)])
                H1 = CoolProp.PropsSI('H', 'T', T1, 'P', P1, fluid_string)
                T2_precise = CoolProp.PropsSI('T', 'H', H1, 'P', P2, fluid_string)
            else:  # Pure fluid
                gas = CoolPropGas(backend, fluid_input[0], T=T1, P=P1, zs=zs)
                liquid = CoolPropLiquid(backend, fluid_input[0], T=T1, P=P1, zs=zs)
                flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
                state_1 = flasher.flash(T=T1, P=P1)
                state_2 = flasher.flash(H=state_1.H(), P=P2)
                T2_precise = state_2.T
            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            T2_precise = state_2.T

            # Calculate using Peng-Robinson
            eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            T2_PR = state_2.T

            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Outlet Temp (High Precision): {T2_precise:.2f} K\n")
            self.result_text.insert(tk.END, f"Outlet Temp (Peng-Robinson): {T2_PR:.2f} K\n")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")


app = JTApp()
app.mainloop()
