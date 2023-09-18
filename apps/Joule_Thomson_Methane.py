import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashPureVLS

class JTCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Joule-Thomson Coefficient Calculator for Methane")
        self.geometry("500x200")

        # Label and Entry for Temperature
        ttk.Label(self, text="Enter Temperature (K):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Pressure
        ttk.Label(self, text="Enter Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure_entry = ttk.Entry(self)
        self.pressure_entry.grid(row=1, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate JT Coefficient", command=self.calculate_JT)
        self.calculate_btn.grid(row=2, column=0, columnspan=2, pady=20)

        # Label to Display the Result
        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

    def calculate_JT(self):
        try:
            T = float(self.temperature_entry.get())
            P = float(self.pressure_entry.get()) * bar

            fluid = 'methane'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])

            zs = [1]
            eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
            flasher = FlashPureVLS(constants, correlations, liquids=[liquid], gas=gas, solids=[])

            res = flasher.flash(T=T, P=P, zs=zs)
            self.result_label["text"] = f"The JT coefficient at the specified conditions is {res.Joule_Thomson():.4g} K/Pa"

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = JTCalculator()
    app.mainloop()
