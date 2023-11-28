import tkinter as tk
from tkinter import ttk

class DensityConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Density Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 kg/m^3
        self.units = {
            "kg/m^3": 1,
            "g/cm^3": 1e-3,
            "g/ml": 1e-3,
            "lb/ft^3": 0.062427961,
            "lb/in^3": 0.0361273,
            "slugs/ft^3": 0.00194032,
            "tonnes/m^3": 1e-3
        }

        self.entries = {}
        row = 0
        for unit in self.units:
            ttk.Label(self, text=f"{unit}:").grid(row=row, column=0, padx=10, pady=5)
            entry = ttk.Entry(self)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entry.bind('<Return>', self.on_entry)
            entry.bind('<FocusOut>', self.on_entry)
            self.entries[unit] = entry
            row += 1

    def on_entry(self, event=None):
        """Callback when an entry is changed."""
        for unit, entry in self.entries.items():
            if entry == event.widget:
                try:
                    value_in_kg_per_m3 = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_kg_per_m3)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_kg_per_m3):
        """Update all entries based on the value in kg/m^3, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_kg_per_m3 * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = DensityConverter()
    app.mainloop()
