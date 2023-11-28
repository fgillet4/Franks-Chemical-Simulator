import tkinter as tk
from tkinter import ttk, messagebox
from ht import *

def fetch_material_properties():
    try:
        material_name = material_entry.get()
        temperature = float(temp_entry.get())
        
        material = nearest_material(material_name)

        if not material:
            result_var.set("No matching material found.")
            return

        k = k_material(material, T=temperature)
        rho = rho_material(material)
        Cp = Cp_material(material, T=temperature)

        result_var.set(f"Material: {material}\n"
                       f"Thermal Conductivity: {k} W/mK\n"
                       f"Density: {rho} kg/m^3\n"
                       f"Heat Capacity: {Cp} J/kgK")

    except Exception as e:
        result_var.set("Error!")
        messagebox.showerror("Error", str(e))

app = tk.Tk()
app.title("Material Property Fetcher")

# Material Input field
ttk.Label(app, text="Enter Material:").grid(row=0, column=0, padx=10, pady=5)
material_entry = ttk.Entry(app)
material_entry.grid(row=0, column=1, padx=10, pady=5)

# Temperature Input field
ttk.Label(app, text="Enter Temperature (K):").grid(row=1, column=0, padx=10, pady=5)
temp_entry = ttk.Entry(app)
temp_entry.insert(0, "298.15")  # Default temperature
temp_entry.grid(row=1, column=1, padx=10, pady=5)

# Fetch button
ttk.Button(app, text="Fetch Properties", command=fetch_material_properties).grid(row=2, columnspan=2, pady=10)

# Output label
result_var = tk.StringVar()
result_label = ttk.Label(app, textvariable=result_var, font=("Arial", 10), anchor="w", justify=tk.LEFT)
result_label.grid(row=3, columnspan=2, padx=10, pady=5)

app.mainloop()
