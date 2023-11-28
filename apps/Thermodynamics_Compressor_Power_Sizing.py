import tkinter as tk
from tkinter import ttk, messagebox
from scipy.constants import hour
from thermo import ChemicalConstantsPackage, PRMIX, IGMIX, FlashVL, CEOSLiquid, CEOSGas
from thermo.interaction_parameters import IPDB
from scipy.integrate import quad

class CompressionPowerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Compression Power Calculator")

        ttk.Label(self, text="Molecules (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecules_entry = ttk.Entry(self, width=50)
        self.molecules_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecules_entry.insert(0, "CO2, O2")

        ttk.Label(self, text="Molar fractions (comma separated):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fractions_entry = ttk.Entry(self, width=50)
        self.molar_fractions_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fractions_entry.insert(0, "0.5, 0.5")

        ttk.Label(self, text="Initial Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self)
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temperature_entry.insert(0, "290")

        ttk.Label(self, text="Initial Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.initial_pressure_entry = ttk.Entry(self)
        self.initial_pressure_entry.grid(row=3, column=1, padx=10, pady=5)
        self.initial_pressure_entry.insert(0, "1")

        ttk.Label(self, text="Final Pressure (bar):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.final_pressure_entry = ttk.Entry(self)
        self.final_pressure_entry.grid(row=4, column=1, padx=10, pady=5)
        self.final_pressure_entry.insert(0, "5")

        ttk.Label(self, text="Flow (mol/hour):").grid(row=5, column=0, sticky=tk.W, padx=10, pady=5)
        self.flow_entry = ttk.Entry(self)
        self.flow_entry.grid(row=5, column=1, padx=10, pady=5)
        self.flow_entry.insert(0, "2000")

        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=6, column=0, columnspan=2, pady=20)

        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=7, column=0, columnspan=2, pady=5)

    def calculate(self):
        # Extract values from the GUI
        molecules = [m.strip() for m in self.molecules_entry.get().split(',')]
        zs = [float(fraction.strip()) for fraction in self.molar_fractions_entry.get().split(',')]

        if len(molecules) != len(zs):
            messagebox.showerror("Error", "Number of molecules and fractions don't match!")
            return

        T1 = float(self.temperature_entry.get())
        P1 = float(self.initial_pressure_entry.get()) * 1e5  # Convert to Pascals
        P2 = float(self.final_pressure_entry.get()) * 1e5    # Convert to Pascals
        flow = float(self.flow_entry.get()) / hour            # Convert to mol/s

        # Your provided calculations
        constants, correlations = ChemicalConstantsPackage.from_IDs(molecules)
        kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')

        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas,
                 kijs=kijs)
        liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

        state_1 = flasher.flash(T=T1, P=P1, zs=zs)
        state_2 = flasher.flash(S=state_1.S(), P=P2, zs=zs)
        shaft_duty_pr = (state_2.H() - state_1.H())*flow

        liquid = CEOSLiquid(IGMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(IGMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        flasher_ideal = FlashVL(constants, correlations, liquid=liquid, gas=gas)

        state_1 = flasher_ideal.flash(T=T1, P=P1, zs=zs)
        state_2 = flasher_ideal.flash(S=state_1.S(), P=P2, zs=zs)
        shaft_duty_ideal = (state_2.H() - state_1.H())*flow

        self.result_label.config(text=f"Shaft power with Peng-Robinson: {shaft_duty_pr:.4f} W\nShaft power with ideal-gas: {shaft_duty_ideal:.4f} W")

app = CompressionPowerApp()
app.mainloop()
