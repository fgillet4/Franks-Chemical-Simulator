import tkinter as tk
from tkinter import ttk

class EnergyConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy Converter")
        self.geometry("600x500")

        # Units and their respective conversion values relative to 1 Joule
        self.units = {
            "Joule (J)": 1,
            "Kilojoule (kJ)": 1e-3,
            "Megajoule (MJ)": 1e-6,
            "Gigajoule (GJ)": 1e-9,
            "Calorie (cal)": 0.239006,
            "Kilocalorie (kcal)": 0.000239006,
            "British Thermal Unit (BTU)": 0.000947817,
            "Therm (US)": 9.4804e-9,
            "Kilowatt-hour (kWh)": 2.7778e-7,
            "Electronvolt (eV)": 6.242e+18,
            "Erg": 1e7,
            "Foot-pound (ft·lbf)": 0.737562
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
                    value_in_j = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_j)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_j):
        """Update all entries based on the value in Joules, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_j * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = EnergyConverter()
    app.mainloop()
