import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CoolPropLiquid, CEOSGas, CoolPropGas, FlashPureVLS

class TemperatureCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Temperature Change Upon Ethylene Expansion")
        self.geometry("600x300")

        # Label and Entry for Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (K):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Label and Entry for Initial Pressure
        ttk.Label(self, text="Enter Initial Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure1_entry = ttk.Entry(self)
        self.pressure1_entry.grid(row=1, column=1, padx=10, pady=5)

        # Label and Entry for Pressure after first valve
        ttk.Label(self, text="Enter Pressure After First Valve (bar):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pressure2_entry = ttk.Entry(self)
        self.pressure2_entry.grid(row=2, column=1, padx=10, pady=5)

        # Label and Entry for Pressure after second valve
        ttk.Label(self, text="Enter Pressure After Second Valve (bar):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.pressure3_entry = ttk.Entry(self)
        self.pressure3_entry.grid(row=3, column=1, padx=10, pady=5)

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Temperatures", command=self.calculate_temperatures)
        self.calculate_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Labels to Display the Results
        self.result_label2 = ttk.Label(self, text="")
        self.result_label2.grid(row=5, column=0, columnspan=2, pady=5)

        self.result_label3 = ttk.Label(self, text="")
        self.result_label3.grid(row=6, column=0, columnspan=2, pady=5)

    def calculate_temperatures(self):
        try:
            T1 = float(self.temperature_entry.get())
            P1 = float(self.pressure1_entry.get()) * bar
            P2 = float(self.pressure2_entry.get()) * bar
            P3 = float(self.pressure3_entry.get()) * bar

            fluid = 'ethylene'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
            backend = 'HEOS'
            gas = CoolPropGas(backend, fluid, T=T1, P=P1, zs=[1])
            liquid = CoolPropLiquid(backend, fluid, T=T1, P=P1, zs=[1])

            flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

            state_1 = flasher.flash(T=T1, P=P1)
            state_2 = flasher.flash(H=state_1.H(), P=P2)
            state_3 = flasher.flash(H=state_1.H(), P=P3)

            self.result_label2["text"] = f"The temperature after the first valve is {state_2.T: .2f} K"
            self.result_label3["text"] = f"The temperature after the second valve is {state_3.T: .2f} K"

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = TemperatureCalculator()
    app.mainloop()
