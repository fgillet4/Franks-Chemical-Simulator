import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import bar
from thermo import ChemicalConstantsPackage, Mixture, SRKMIX, IdealGas, CEOSLiquid, CEOSGas, FlashPureVLS

class CompressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Isentropic Compression of Fluid")

        # Molecule input
        ttk.Label(self, text="Molecules (CAS, Formula, Common Name):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecule_entry = ttk.Entry(self, width=30)
        self.molecule_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecule_entry.insert(0, "O2,N2")

        # Mole fraction input
        ttk.Label(self, text="Mole fractions:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.mole_fraction_entry = ttk.Entry(self, width=30)
        self.mole_fraction_entry.grid(row=1, column=1, padx=10, pady=5)
        self.mole_fraction_entry.insert(0, "0.21,0.79")

        # Temperature input
        ttk.Label(self, text="Temperature (°C):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temp_entry = ttk.Entry(self, width=30)
        self.temp_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temp_entry.insert(0, "15")

        # Initial pressure input
        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.p1_entry = ttk.Entry(self, width=30)
        self.p1_entry.grid(row=3, column=1, padx=10, pady=5)
        self.p1_entry.insert(0, "1")

        # Final pressure input
        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.p2_entry = ttk.Entry(self, width=30)
        self.p2_entry.grid(row=4, column=1, padx=10, pady=5)
        self.p2_entry.insert(0, "8")

        # Calculate button
        self.calc_button = ttk.Button(self, text="Calculate", command=self.calculate_values_gui)
        self.calc_button.grid(row=5, columnspan=2, pady=10)

        # Result Text Area
        self.result_text = tk.Text(self, width=50, height=10)
        self.result_text.grid(row=6, columnspan=2, padx=10, pady=5)


    def calculate_values_gui(self):
        molecule_input = self.molecule_entry.get().split(',')
        molecules = [m.strip() for m in molecule_input]
        mole_fractions = list(map(float, self.mole_fraction_entry.get().split(',')))
        
        T1 = float(self.temp_entry.get()) + 273.15
        P1 = float(self.p1_entry.get()) * bar
        P2 = float(self.p2_entry.get()) * bar

        try:
            actual_power_ideal, T2_ideal, actual_power_srk, T2_srk = calculate_values(T1, P1, P2, molecules, mole_fractions)
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ideal Gas Power: {actual_power_ideal:.4f} J/mol\n")
            self.result_text.insert(tk.END, f"Ideal Gas Outlet Temp: {T2_ideal:.2f} K\n")
            self.result_text.insert(tk.END, f"SRK Power: {actual_power_srk:.4f} J/mol\n")
            self.result_text.insert(tk.END, f"SRK Outlet Temp: {T2_srk:.2f} K\n")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")


def calculate_values(T1, P1, P2, molecules, mole_fractions):
    fluid = Mixture(molecules, zs=mole_fractions)

    # For Ideal-Gas EOS
    Cp = fluid.Cp(T=T1)  # Molar heat capacity at constant pressure
    gamma = fluid.isentropic_exponent(T=T1)  # Heat capacity ratio
    T2_ideal = T1 * (P2 / P1)**((gamma - 1) / gamma)
    actual_power_ideal = Cp * (T1 - T2_ideal)

    # SRK
    eos_kwargs = dict(Tcs=fluid.Tcs, Pcs=fluid.Pcs, omegas=fluid.omegas)
    liquid = CEOSLiquid(SRKMIX, T=T1, P=P1, zs=mole_fractions, eos_kwargs=eos_kwargs)
    gas_srk = CEOSGas(SRKMIX, T=T1, P=P1, zs=mole_fractions, eos_kwargs=eos_kwargs)
    
    flasher_srk = FlashPureVLS(fluid, gas=gas_srk, liquids=[liquid])
    state_1_srk = flasher_srk.flash(T=T1, P=P1)
    state_2_srk = flasher_srk.flash(S=state_1_srk.S, P=P2)
    actual_power_srk = state_2_srk.H - state_1_srk.H


    return actual_power_ideal, T2_ideal, actual_power_srk, state_2_srk.T
if __name__ == "__main__":
    app = CompressionApp()
    app.mainloop()




