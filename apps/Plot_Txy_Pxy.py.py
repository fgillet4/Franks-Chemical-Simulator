import tkinter as tk
from tkinter import ttk, messagebox
from thermo import *
from thermo.unifac import DOUFSG, DOUFIP2016
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DiagramApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Txy, Pxy, and xy Diagrams")

        # Molecules input
        ttk.Label(self, text="Molecules (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.molecules_entry = ttk.Entry(self, width=50)
        self.molecules_entry.grid(row=0, column=1, padx=10, pady=5)
        self.molecules_entry.insert(0, "ethanol, water")

        # Mole Fractions input
        ttk.Label(self, text="Molar fractions (comma separated):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.molar_fractions_entry = ttk.Entry(self, width=50)
        self.molar_fractions_entry.grid(row=1, column=1, padx=10, pady=5)
        self.molar_fractions_entry.insert(0, "0.5, 0.5")

        # Temperature or Pressure input
        ttk.Label(self, text="Temperature (K):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.temperature_entry = ttk.Entry(self, width=30)
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=5)
        self.temperature_entry.insert(0, "373")

        ttk.Label(self, text="Pressure (bar):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.pressure_entry = ttk.Entry(self, width=30)
        self.pressure_entry.grid(row=3, column=1, padx=10, pady=5)
        self.pressure_entry.insert(0, "1")

        # Plot buttons
        self.btn_Txy = ttk.Button(self, text="Plot Txy", command=self.plot_Txy)
        self.btn_Txy.grid(row=4, column=0, pady=20)
        
        self.btn_Pxy = ttk.Button(self, text="Plot Pxy", command=self.plot_Pxy)
        self.btn_Pxy.grid(row=4, column=1, pady=20)

        # Canvas for plots
        self.canvas = FigureCanvasTkAgg(plt.Figure(figsize=(6, 6)), master=self)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)
    
    def get_flasher(self):
        # Extract values from the GUI
        molecules = [m.strip() for m in self.molecules_entry.get().split(',')]
        zs = [float(fraction.strip()) for fraction in self.molar_fractions_entry.get().split(',')]

        if len(molecules) != len(zs):
            messagebox.showerror("Error", "Number of molecules and fractions don't match!")
            return None

        # Load constants and properties
        constants, properties = ChemicalConstantsPackage.from_IDs(molecules)

        # Configure the activity model
        GE = UNIFAC.from_subgroups(chemgroups=constants.UNIFAC_Dortmund_groups, version=1, T=300, xs=zs,
                                interaction_data=DOUFIP2016, subgroups=DOUFSG)

        # Configure the liquid model with activity coefficients
        liquid = GibbsExcessLiquid(
            VaporPressures=properties.VaporPressures,
            HeatCapacityGases=properties.HeatCapacityGases,
            VolumeLiquids=properties.VolumeLiquids,
            GibbsExcessModel=GE,
            equilibrium_basis='Psat', caloric_basis='Psat',
            T=300, P=1e5, zs=zs)

        # Use Peng-Robinson for the vapor phase
        eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
        gas = CEOSGas(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)

        # Create a flasher instance
        flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        return flasher

    def plot_Txy(self):
        flasher = self.get_flasher()
        if not flasher:
            return
        P = float(self.pressure_entry.get()) * 1e5  # Convert to Pascals
        fig, ax = plt.subplots()
        _ = flasher.plot_Txy(P=P, pts=100, ax=ax)
        self.canvas.figure = fig
        self.canvas.draw()

    def plot_Pxy(self):
        flasher = self.get_flasher()
        if not flasher:
            return
        T = float(self.temperature_entry.get())
        fig, ax = plt.subplots()
        _ = flasher.plot_Pxy(T=T, pts=100, ax=ax)
        self.canvas.figure = fig
        self.canvas.draw()

app = DiagramApp()
app.mainloop()
