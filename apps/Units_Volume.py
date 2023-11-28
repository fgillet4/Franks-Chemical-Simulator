import tkinter as tk
from tkinter import ttk

class VolumeConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Volume Converter")
        self.geometry("900x700")

        # Units and their respective conversion values relative to 1 cubic meter
        self.units = {
            "cubic meter (m^3)": 1,
            "litre": 1000,
            "millilitre (ml)": 1e6,
            "gallon (US)": 264.172,
            "gallon (UK)": 219.969,
            "quart (US)": 1056.69,
            "quart (UK)": 879.877,
            "20-foot container": 33.2,
            "40-foot container": 67.7,
            "53-foot truck trailer": 130,  # Approximate volume
            "Fuel Truck (small, 20,000l)": 20,
            "Fuel Truck (large, 36,000l)": 36,
            "Tanker Ship (small)": 1e3,   # Placeholder value
            "Tanker Ship (VLCC)": 3e5,    # Placeholder value
            "Air Freight ULD LD3": 4.5,   # Typical air cargo container
            "Rail Car (average)": 120,    # Typical volume, can vary widely
            "Standard Pallet": 1.5,       # Approximation, varies based on height/stack
            "Average Car": 3,             # Rough average, can vary widely
            "Semi-trailer": 85            # Approximation
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
                    value_in_m3 = float(entry.get()) / self.units[unit]
                    self.update_all_except(unit, value_in_m3)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, value_in_m3):
        """Update all entries based on the value in cubic meters, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = value_in_m3 * self.units[unit]
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = VolumeConverter()
    app.mainloop()
