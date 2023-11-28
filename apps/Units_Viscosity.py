import tkinter as tk
from tkinter import ttk

class ViscosityConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Viscosity Converter")
        self.geometry("600x300")

        # Labels and Entries
        ttk.Label(self, text="Density:").grid(row=0, column=0, padx=10, pady=5)
        self.density_value = ttk.Entry(self)
        self.density_value.grid(row=0, column=1, padx=10, pady=5)
        self.density_units = ttk.Combobox(self, values=["kg/m³", "lb/ft³"], state="readonly")
        self.density_units.set("kg/m³")
        self.density_units.grid(row=0, column=2, padx=10, pady=5)
        self.density_value.bind('<Return>', self.on_entry)
        self.density_value.bind('<FocusOut>', self.on_entry)

        ttk.Label(self, text="Dynamic Viscosity:").grid(row=1, column=0, padx=10, pady=5)
        self.dynamic_value = ttk.Entry(self)
        self.dynamic_value.grid(row=1, column=1, padx=10, pady=5)
        self.dynamic_units = ttk.Combobox(self, values=["Pa·s", "P", "cP"], state="readonly")
        self.dynamic_units.set("Pa·s")
        self.dynamic_units.grid(row=1, column=2, padx=10, pady=5)
        self.dynamic_value.bind('<Return>', self.on_entry)
        self.dynamic_value.bind('<FocusOut>', self.on_entry)

        ttk.Label(self, text="Kinematic Viscosity:").grid(row=2, column=0, padx=10, pady=5)
        self.kinematic_value = ttk.Entry(self)
        self.kinematic_value.grid(row=2, column=1, padx=10, pady=5)
        self.kinematic_units = ttk.Combobox(self, values=["m²/s", "St", "cSt"], state="readonly")
        self.kinematic_units.set("m²/s")
        self.kinematic_units.grid(row=2, column=2, padx=10, pady=5)
        self.kinematic_value.bind('<Return>', self.on_entry)
        self.kinematic_value.bind('<FocusOut>', self.on_entry)

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        try:
            if event.widget == self.density_value or event.widget == self.dynamic_value:
                self.update_kinematic()

            elif event.widget == self.kinematic_value:
                self.update_dynamic()

        except ValueError:
            pass

    def update_kinematic(self):
        density = float(self.density_value.get()) * self.convert_density(self.density_units.get(), "kg/m³")
        dynamic_viscosity = float(self.dynamic_value.get()) * self.convert_dynamic_viscosity(self.dynamic_units.get(), "Pa·s")
        
        kinematic_viscosity = dynamic_viscosity / density

        self.kinematic_value.delete(0, tk.END)
        self.kinematic_value.insert(0, kinematic_viscosity / self.convert_kinematic_viscosity("m²/s", self.kinematic_units.get()))

    def update_dynamic(self):
        density = float(self.density_value.get()) * self.convert_density(self.density_units.get(), "kg/m³")
        kinematic_viscosity = float(self.kinematic_value.get()) * self.convert_kinematic_viscosity(self.kinematic_units.get(), "m²/s")
        
        dynamic_viscosity = kinematic_viscosity * density

        self.dynamic_value.delete(0, tk.END)
        self.dynamic_value.insert(0, dynamic_viscosity / self.convert_dynamic_viscosity("Pa·s", self.dynamic_units.get()))

    def convert_density(self, from_unit, to_unit):
        conversions = {
            "kg/m³": 1,
            "lb/ft³": 16.0185  # kg/m³ to lb/ft³ conversion factor
        }
        return conversions[to_unit] / conversions[from_unit]

    def convert_dynamic_viscosity(self, from_unit, to_unit):
        conversions = {
            "Pa·s": 1,
            "P": 0.1,
            "cP": 0.001
        }
        return conversions[to_unit] / conversions[from_unit]

    def convert_kinematic_viscosity(self, from_unit, to_unit):
        conversions = {
            "m²/s": 1,
            "St": 0.0001,
            "cSt": 0.000001
        }
        return conversions[to_unit] / conversions[from_unit]

if __name__ == "__main__":
    app = ViscosityConverter()
    app.mainloop()
