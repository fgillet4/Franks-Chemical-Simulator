import tkinter as tk
from tkinter import ttk

class LengthConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Length Converter")
        self.geometry("600x400")

        # Units and their respective conversion values relative to 1 meter
        self.units = {
            "meters": 1,
            "km": 0.001,
            "cm": 100,
            "mm": 1000,
            "inches": 39.3701,
            "feet": 3.28084,
            "miles": 0.000621371
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
                    value_in_m = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_m)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_m):
        """Update all entries based on the value in meters, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_m * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = LengthConverter()
    app.mainloop()
