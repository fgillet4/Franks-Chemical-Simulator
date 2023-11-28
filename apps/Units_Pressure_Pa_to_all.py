import tkinter as tk
from tkinter import ttk

class PressureConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pressure Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 Pascal
        self.units = {
            "Pa": 1,
            "kPa": 1e-3,
            "MPa": 1e-6,
            "bar": 1e-5,
            "bar(g)": 1e-5,
            "bar(a)": 1.01325e-5,
            "torr": 7.50062,
            "atm": 9.86923e-6,
            "psi": 0.000145038,
            "m water": 0.101972,
            "ft water": 0.0298907,
            "in water": 0.0360912,
            "mmHg": 0.00750062
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
                    if unit == "bar(g)":
                        value_in_pa = (float(entry.get()) + 1.01325) / self.units[unit]
                    else:
                        value_in_pa = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_pa)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_pa):
        """Update all entries based on the value in Pascal, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                if unit == "bar(g)":
                    entry_value = (value_in_pa * self.units[unit]) - 1.01325
                else:
                    entry_value = value_in_pa * self.units[unit]

                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = PressureConverter()
    app.mainloop()
