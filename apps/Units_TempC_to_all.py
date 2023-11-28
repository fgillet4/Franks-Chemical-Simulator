import tkinter as tk
from tkinter import ttk

class TemperatureConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Temperature Converter")
        self.geometry("600x250")

        # Units and their respective conversion functions relative to Celsius
        self.units = {
            "Celsius": lambda x: x,
            "Kelvin": lambda x: x + 273.15,
            "Fahrenheit": lambda x: (x * 9/5) + 32,
            "Rankine": lambda x: (x + 273.15) * 9/5
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
                    if unit == "Celsius":
                        celsius_value = float(entry.get())
                    elif unit == "Kelvin":
                        celsius_value = float(entry.get()) - 273.15
                    elif unit == "Fahrenheit":
                        celsius_value = (float(entry.get()) - 32) * 5/9
                    else:  # Rankine
                        celsius_value = (float(entry.get()) - 491.67) * 5/9

                    self.update_all_except(unit, celsius_value)
                except ValueError:
                    pass
                break

    def update_all_except(self, unit_to_skip, celsius_value):
        """Update all entries based on the value in Celsius, except the specified one."""
        for unit, entry in self.entries.items():
            if unit != unit_to_skip:
                entry_value = self.units[unit](celsius_value)
                formatted_value = "{:.2f}".format(entry_value)  # Format to 2 decimal places
                entry.delete(0, tk.END)
                entry.insert(0, formatted_value)

if __name__ == "__main__":
    app = TemperatureConverter()
    app.mainloop()
