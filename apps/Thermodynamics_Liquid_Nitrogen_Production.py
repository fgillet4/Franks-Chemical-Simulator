import tkinter as tk
from tkinter import ttk, messagebox
from thermo import *
from thermo.interaction_parameters import SPDB

class LiquidNitrogenCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Liquid Nitrogen Production Via Volume Expansion of the Compressed Gas")
        self.geometry("800x400")

        # Initial Temperature
        ttk.Label(self, text="Enter Initial Temperature (°C):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=0, column=1, padx=10, pady=5)

        # Initial Pressure
        ttk.Label(self, text="Enter Initial Pressure (bar):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.pressure1_entry = ttk.Entry(self)
        self.pressure1_entry.grid(row=1, column=1, padx=10, pady=5)

        # Pressure after Valve
        ttk.Label(self, text="Enter Pressure After Valve (bar):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pressure2_entry = ttk.Entry(self)
        self.pressure2_entry.grid(row=2, column=1, padx=10, pady=5)

        # EOS selection
        ttk.Label(self, text="Select EOS:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.eos_combobox = ttk.Combobox(self, values=[
            "High-Precision", "PR-Pina-Martinez", "SRK-Pina-Martinez", "PR", "SRK", 
            "VDW", "PRSV", "PRSV2", "TWUPR", "TWUSRK", 
            "PRTranslatedConsistent", "SRKTranslatedConsistent"])
        self.eos_combobox.grid(row=3, column=1, padx=10, pady=5)
        self.eos_combobox.set("High-Precision")

        # Button to Calculate
        self.calculate_btn = ttk.Button(self, text="Calculate Liquid Fraction", command=self.calculate_fraction)
        self.calculate_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Result Display
        self.result_label = ttk.Label(self, text="", font=("Arial", 14))
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

    def calculate_fraction(self):
        T1 = float(self.temperature_entry.get()) + 273.15
        P1 = float(self.pressure1_entry.get()) * 1e5
        P2 = float(self.pressure2_entry.get()) * 1e5
        eos_selection = self.eos_combobox.get()

        # Perform calculations using the Thermo library
        try:
            fluid = 'nitrogen'
            constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
            zs = [1]

            flasher = None
            if eos_selection == "High-Precision":
                gas = CoolPropGas('HEOS', fluid, T=T1, P=P1, zs=zs)
                liquid = CoolPropLiquid('HEOS', fluid, T=T1, P=P1, zs=zs)
                flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

            # ... (You can add code for other EOSs in a similar way)

            if flasher:
                state_1 = flasher.flash(T=T1, P=P1, zs=zs)
                state_2 = flasher.flash(P=P2, H=state_1.H(), zs=zs)
                self.result_label["text"] = f"The {eos_selection} EOS predicted liquid molar fraction is {state_2.LF:.8f}."
            else:
                self.result_label["text"] = f"The EOS {eos_selection} is not supported yet."

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = LiquidNitrogenCalculator()
    app.mainloop()
