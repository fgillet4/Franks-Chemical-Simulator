import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import chemicals.combustion

class CombustionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Combustion Calculator")
        self.geometry("600x400")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        
        # Tabs for each Functionality
        self.create_stoichiometry_tab()
        self.create_heat_tab()
        self.create_ratio_tab()
        self.create_ignition_tab()
    
    def create_stoichiometry_tab(self):
        # Stoichiometry Tab
        self.stoichiometry_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stoichiometry_tab, text="Stoichiometry")
        ttk.Label(self.stoichiometry_tab, text="Enter Compound (e.g. {'C': 1, 'H':4}): ").grid(row=0, column=0, sticky="w")
        self.compound_entry = ttk.Entry(self.stoichiometry_tab)
        self.compound_entry.grid(row=0, column=1, sticky="ew")
        self.stoichiometry_btn = ttk.Button(self.stoichiometry_tab, text="Calculate", command=self.calculate_stoichiometry)
        self.stoichiometry_btn.grid(row=1, column=0, columnspan=2)
        self.stoichiometry_result = ttk.Label(self.stoichiometry_tab, text="")
        self.stoichiometry_result.grid(row=2, column=0, columnspan=2, sticky="w")

    def calculate_stoichiometry(self):
        try:
            compound = eval(self.compound_entry.get())
            result = chemicals.combustion.combustion_stoichiometry(compound)
            self.stoichiometry_result["text"] = f"Result: {result}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_heat_tab(self):
        # Heat of Combustion Tab
        self.heat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.heat_tab, text="Heat of Combustion")
        ttk.Label(self.heat_tab, text="Enter Compound (e.g. {'C': 1, 'H':4, 'O': 1}): ").grid(row=0, column=0, sticky="w")
        self.heat_compound_entry = ttk.Entry(self.heat_tab)
        self.heat_compound_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(self.heat_tab, text="Heat of Formation (Hf): ").grid(row=1, column=0, sticky="w")
        self.hf_entry = ttk.Entry(self.heat_tab)
        self.hf_entry.grid(row=1, column=1, sticky="ew")
        self.heat_btn = ttk.Button(self.heat_tab, text="Calculate", command=self.calculate_heat)
        self.heat_btn.grid(row=2, column=0, columnspan=2)
        self.heat_result = ttk.Label(self.heat_tab, text="")
        self.heat_result.grid(row=3, column=0, columnspan=2, sticky="w")

    def calculate_heat(self):
        try:
            compound = eval(self.heat_compound_entry.get())
            Hf = float(self.hf_entry.get())
            combustion_data = chemicals.combustion.combustion_data(compound, Hf=Hf)
            self.heat_result["text"] = f"HHV: {combustion_data.HHV}, LHV: {chemicals.combustion.LHV_from_HHV(combustion_data.HHV, 2)}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def create_ratio_tab(self):
        # Fuel-to-Air Ratio Tab
        self.ratio_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ratio_tab, text="Fuel-to-Air Ratio")
        ttk.Label(self.ratio_tab, text="Enter n_fuel: ").grid(row=0, column=0, sticky="w")
        self.n_fuel_entry = ttk.Entry(self.ratio_tab)
        self.n_fuel_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(self.ratio_tab, text="Enter n_air: ").grid(row=1, column=0, sticky="w")
        self.n_air_entry = ttk.Entry(self.ratio_tab)
        self.n_air_entry.grid(row=1, column=1, sticky="ew")
        self.ratio_btn = ttk.Button(self.ratio_tab, text="Calculate", command=self.calculate_ratio)
        self.ratio_btn.grid(row=2, column=0, columnspan=2)
        self.ratio_result = ttk.Label(self.ratio_tab, text="")
        self.ratio_result.grid(row=3, column=0, columnspan=2, sticky="w")

    def calculate_ratio(self):
        try:
            n_fuel = float(self.n_fuel_entry.get())
            n_air = float(self.n_air_entry.get())
            Vm_air = 0.024936627188566596  # These are example constants; you might need to adjust or add input fields.
            Vm_fuel = 0.024880983160354486
            MW_air = 28.850334
            MW_fuel = 17.86651
            ratio = chemicals.combustion.air_fuel_ratio_solver(ratio=5.0, Vm_air=Vm_air, Vm_fuel=Vm_fuel, MW_air=MW_air, MW_fuel=MW_fuel, n_air=n_air, n_fuel=n_fuel, basis='mole')
            self.ratio_result["text"] = f"Results: {ratio}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_ignition_tab(self):
        # Ignition & Octane Ratings Tab
        self.ignition_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ignition_tab, text="Ignition & Octane Ratings")
        ttk.Label(self.ignition_tab, text="Enter CASRN: ").grid(row=0, column=0, sticky="w")
        self.casrn_entry = ttk.Entry(self.ignition_tab)
        self.casrn_entry.grid(row=0, column=1, sticky="ew")
        self.ignition_btn = ttk.Button(self.ignition_tab, text="Get Details", command=self.get_ignition_details)
        self.ignition_btn.grid(row=1, column=0, columnspan=2)
        self.ignition_result = ttk.Label(self.ignition_tab, text="")
        self.ignition_result.grid(row=2, column=0, columnspan=2, sticky="w")

    def get_ignition_details(self):
        try:
            casrn = self.casrn_entry.get()
            RON_value = chemicals.combustion.RON(CASRN=casrn)
            MON_value = chemicals.combustion.MON(CASRN=casrn)
            ignition_delay_value = chemicals.combustion.ignition_delay(CASRN=casrn)
            self.ignition_result["text"] = f"RON: {RON_value}, MON: {MON_value}, Ignition Delay: {ignition_delay_value}"
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = CombustionApp()
    app.mainloop()
